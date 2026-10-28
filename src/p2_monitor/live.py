"""Live monitoring: rates from the capture dumpcap is writing, as data arrive.

    p2-monitor live <subrun>/raw_daq_data --setup lab_saclay_2026 --window 60

Every `interval` seconds the newest capture is polled with p2_decode's
StreamDecoder (only the bytes appended since the last poll are read). Each
poll's hits become a channel-occupancy record per (fec, vmm, ch); the records
of the last `window` seconds of packet time are summed, and their rates
written atomically to a JSON file (default <raw_dir>/../live.json) that a GUI
can poll. When dumpcap rotates, the old file is read to its end first.

It is a sampling monitor, built not to add load when the DAQ is under stress:

  * at most max_read_mb is read per poll, so its throughput is bounded;
  * when more than max_behind_mb is unread it skips to the newest data
    (sampling["fraction_read"] says how much was read); rates stay right over
    what was read, because live time is taken from those packets only;
  * records are merged into at most ~60 time buckets, so a long window
    cannot grow its memory;
  * above rss_limit_mb of its own memory it drops the window and halves its
    read size, and it can raise its own oom_score_adj so the kernel takes it
    before anything of the DAQ.

None of the defaults assume a data rate: they bound work and memory per poll.
`data_age_s` says how long ago (wall clock) the newest packet arrived, which
shows a stalled DAQ at a glance.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import p2_mapping
from p2_decode import StreamDecoder

from .counts import chip_occupancy
from .rates import channel_rates, summarize
from .watch import captures

MB = 1024 * 1024
MAX_BUCKETS = 60


def own_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def set_oom_score_adj(value):
    """Make this process the kernel's first choice under memory pressure."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(str(int(value)))
        return True
    except OSError:
        return False


class LiveMonitor:
    """Follows the newest capture in raw_dir and keeps windowed rates."""

    def __init__(self, raw_dir, setup=None, when=None, mapping=True, window_s=60.0,
                 data_format="SRS", out=None, plot_dir=None, plot_every_s=0,
                 max_read_mb=16, max_behind_mb=64, rss_limit_mb=None):
        self.raw_dir = Path(raw_dir)
        self.setup, self.when, self.use_mapping = setup, when, mapping
        self.window_s = float(window_s)
        self.bucket_s = None if self.window_s == float("inf") else self.window_s / MAX_BUCKETS
        self.data_format = data_format
        self.out = Path(out) if out else self.raw_dir.parent / "live.json"
        self.plot_dir, self.plot_every_s = plot_dir, plot_every_s
        self.max_read = int(max_read_mb * MB)
        self.max_behind = int(max_behind_mb * MB)
        self.rss_limit_mb = rss_limit_mb
        self._last_plot = 0.0
        self.stream = None                 # StreamDecoder of the followed file
        self.mapping = None
        self.records = []                  # [t_first_ns, t_last_ns, occupancy, n_hits, bytes]
        self.files_done = []
        self.total_hits = 0
        self.read_bytes = 0
        self.skipped_bytes = 0
        self.skipped_packets = 0
        self.memory_events = 0

    # -- following ------------------------------------------------------------

    def _newest(self):
        c = captures(self.raw_dir) if self.raw_dir.is_dir() else []
        return c[-1][2] if c else None

    def _feed(self, cols, info):
        self.read_bytes += info.get("udp_payload_bytes", 0)
        if cols is None or info["t_first_ns"] is None:
            return
        n = len(cols["vmm"])
        self.total_hits += n
        occ = chip_occupancy(cols)
        t0, t1 = info["t_first_ns"], info["t_last_ns"]
        last = self.records[-1] if self.records else None
        if (last is not None and self.bucket_s is not None
                and t0 // (self.bucket_s * 1e9) == last[0] // (self.bucket_s * 1e9)):
            last[1] = max(last[1], t1)
            last[0] = min(last[0], t0)
            last[2] += occ
            last[3] += n
            last[4] += info["udp_payload_bytes"]
        else:
            self.records.append([t0, t1, occ, n, info["udp_payload_bytes"]])

    def _catch_up(self):
        """Skip the backlog beyond max_behind, then read what is left."""
        f = self.stream.follower
        if f.backlog() > self.max_behind:
            before = f.skipped_packets
            self.skipped_bytes += self.stream.skip_to_tail(self.max_read)
            self.skipped_packets += f.skipped_packets - before
        while True:
            cols, info = self.stream.poll()
            self._feed(cols, info)
            if info["packets"] == 0:
                return

    def _new_stream(self, path):
        sd = StreamDecoder(path, data_format=self.data_format)
        sd.follower.max_read = self.max_read
        return sd

    def _mapping_for(self, t_ns):
        if not self.use_mapping:
            return None
        when = self.when or datetime.fromtimestamp(t_ns / 1e9)
        return p2_mapping.load_mapping(self.setup, when)

    def step(self):
        """One poll. Returns the live summary dict (also written to `out`)."""
        newest = self._newest()
        if newest is not None and (self.stream is None or Path(self.stream.path) != newest):
            if self.stream is not None:
                self._catch_up()                   # rotation: finish the old file
                self.files_done.append(Path(self.stream.path).name)
            self.stream = self._new_stream(newest)
        if self.stream is not None:
            self._catch_up()
        if self.mapping is None and self.use_mapping and self.records:
            self.mapping = self._mapping_for(self.records[0][0])
        self._prune()
        self._check_memory()
        summary = self.summary()
        _write_json(self.out, summary)
        self._maybe_plot(summary)
        return summary

    def _prune(self):
        """Drop records ending more than window_s before the newest packet.

        By packet time, not by arrival order: a replayed or re-stamped file can
        carry times earlier than the file before it.
        """
        if not self.records:
            return
        cut = max(r[1] for r in self.records) - self.window_s * 1e9
        self.records = [r for r in self.records if r[1] >= cut]

    def _check_memory(self):
        if not self.rss_limit_mb:
            return
        rss = own_rss_mb()
        if rss is not None and rss > self.rss_limit_mb:
            self.memory_events += 1
            self.records = self.records[-1:]
            self.max_read = max(MB, self.max_read // 2)
            if self.stream is not None:
                self.stream.follower.max_read = self.max_read

    # -- output ---------------------------------------------------------------

    def window(self):
        """(occupancy counts, live seconds, hits, payload bytes) over the window."""
        if not self.records:
            return None, 0.0, 0, 0
        occ = np.sum([r[2] for r in self.records], axis=0)
        live = (max(r[1] for r in self.records) - min(r[0] for r in self.records)) / 1e9
        return ({"chip_ch": occ}, live, sum(r[3] for r in self.records),
                sum(r[4] for r in self.records))

    def summary(self):
        counts, live, hits, nbytes = self.window()
        seen = self.read_bytes + self.skipped_bytes
        s = {
            "written": datetime.now().isoformat(timespec="seconds"),
            "raw_dir": str(self.raw_dir),
            "capture": Path(self.stream.path).name if self.stream else None,
            "captures_finished": list(self.files_done),
            "setup": self.mapping.setup.name if self.mapping else None,
            "window_s": self.window_s,
            "live_time_s": round(live, 6),
            "hits_in_window": int(hits),
            "hits_total": int(self.total_hits),
            "data_age_s": None,
            "sampling": {"read_mb": round(self.read_bytes / 1e6, 3),
                         "skipped_mb": round(self.skipped_bytes / 1e6, 3),
                         "skipped_packets": int(self.skipped_packets),
                         "fraction_read": round(self.read_bytes / seen, 6) if seen else None,
                         "max_read_mb": round(self.max_read / MB, 3),
                         "memory_events": self.memory_events,
                         "rss_mb": own_rss_mb()},
        }
        if self.records:
            newest = max(r[1] for r in self.records)
            s["data_age_s"] = round(time.time() - newest / 1e9, 3)
            s["newest_packet"] = datetime.fromtimestamp(
                newest / 1e9).isoformat(timespec="milliseconds")
        if counts is not None and live > 0:
            s["volume"] = {"hits_per_s": round(hits / live, 6),
                           "mb_per_min_udp": round(nbytes / 1e6 / live * 60, 6)}
            s["rates"] = summarize(channel_rates(counts, live, self.mapping))
        return s

    def _maybe_plot(self, summary):
        if not (self.plot_dir and self.plot_every_s) or not self.records:
            return
        now = time.time()
        if now - self._last_plot < self.plot_every_s:
            return
        self._last_plot = now
        from .plots import plot_counts
        counts, live, hits, _ = self.window()
        if live > 0:
            frac = summary["sampling"]["fraction_read"]
            title = (f"live: last {self.window_s:.0f} s of {summary['capture']}\n"
                     f"{live:.1f} s of data, {hits:,} hits, newest packet "
                     f"{summary.get('data_age_s')} s ago"
                     + (f", {100 * frac:.0f} % of data read" if frac and frac < 1 else ""))
            plot_counts(counts, live, self.mapping, title, self.plot_dir)

    def run(self, interval_s=5.0, duration_s=None, log=print, oom_score_adj=None):
        """Poll every interval_s until interrupted (or for duration_s)."""
        if oom_score_adj is not None:
            set_oom_score_adj(oom_score_adj)
        t_end = time.time() + duration_s if duration_s is not None else None
        log(f"[live] {self.raw_dir} -> {self.out}  window {self.window_s} s, every "
            f"{interval_s} s, read <= {self.max_read / MB:g} MB/poll, skip beyond "
            f"{self.max_behind / MB:g} MB")
        while True:
            t0 = time.time()
            s = self.step()
            log(_status_line(s))
            if t_end is not None and time.time() >= t_end:
                return s
            time.sleep(max(0.0, interval_s - (time.time() - t0)))


def _status_line(s):
    if "rates" not in s:
        if s["hits_total"]:
            return (f"[live] {s['written']} {s['capture']}  {s['hits_total']:,} hits, "
                    "no rate yet (window spans no time)")
        return f"[live] {s['written']} waiting for data in {s['raw_dir']}"
    det = s["rates"].get("per_detector") or {}
    parts = [f"{k} {v['rate_hz']:.4g} Hz" for k, v in det.items()] or \
            [f"FEC {k} {v['rate_hz']:.4g} Hz" for k, v in s["rates"]["per_fec"].items()]
    frac = s["sampling"]["fraction_read"]
    samp = f"  read {100 * frac:.0f}%" if frac is not None and frac < 1 else ""
    return (f"[live] {s['written']} {s['capture']}  {s['volume']['hits_per_s']:.4g} hits/s"
            f"  {s['volume']['mb_per_min_udp']:.3g} MB/min{samp}  | " + "  ".join(parts)
            + f"  | age {s['data_age_s']} s")


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_np_default)
    os.replace(tmp, path)


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)
