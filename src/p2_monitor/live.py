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

With follow_newest, raw_dir is a runs directory (runs/<run>/<subrun>/raw_daq_data)
and the monitor moves by itself to the newest sub-run's captures, finishing
the previous file first: one long-lived process for a whole session.
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
from .watch import captures, raw_dirs

MB = 1024 * 1024
MAX_BUCKETS = 60
RECENT_S = 5.0          # "now": the electronics view's short window, for plugging cables


def _owners(mapping):
    """{(fec or -1, vmm): 'P2_IN c3 bot'} from a mapping's table (chip -> where)."""
    if mapping is None:
        return {}
    t = mapping.table
    out = {}
    for (fec, vmm), g in t.groupby(["fec", "vmm"]):
        r = g.iloc[0]
        out[(int(fec), int(vmm))] = f"{r['detector']} c{int(r['connector_N'])} {r['half']}"
    return out


def _hz(counts, live, frac):
    """Rates per channel, 3 significant digits, scaled to the full stream."""
    if not live or live <= 0:
        return None
    return [float(f"{v:.3g}") for v in np.asarray(counts, np.float64) / (live * frac)]


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

    def __init__(self, raw_dir=None, setup=None, when=None, mapping=True, window_s=60.0,
                 data_format="SRS", out=None, plot_dir=None, plot_every_s=0,
                 max_read_mb=16, max_behind_mb=64, rss_limit_mb=None,
                 follow_newest=False, raw_inner="raw_daq_data", udp=None):
        """raw_dir: capture directory (or runs directory with follow_newest).
        udp: "HOST:PORT" to read the FEC's stream from the network instead of
        a file; then each poll reads at most max_read_mb from the socket, the
        kernel drops (and counts) the rest, and rates are scaled by the
        fraction received.
        """
        if (raw_dir is None) == (udp is None):
            raise ValueError("give either raw_dir or udp")
        self.udp_addr = udp
        self.udp = None                    # UdpStreamDecoder, opened on the first step
        self.root = Path(raw_dir) if raw_dir is not None else None
        self.follow_newest, self.raw_inner = follow_newest, raw_inner
        self.raw_dir = self.root
        self.setup, self.when, self.use_mapping = setup, when, mapping
        self.window_s = float(window_s)
        self.bucket_s = None if self.window_s == float("inf") else self.window_s / MAX_BUCKETS
        self.data_format = data_format
        if out:
            self.out = Path(out)
        elif udp is not None:
            self.out = Path("live.json")
        else:
            self.out = (self.root / "live.json" if follow_newest
                        else self.raw_dir.parent / "live.json")
        self.plot_dir, self.plot_every_s = plot_dir, plot_every_s
        self.max_read = int(max_read_mb * MB)
        self.max_behind = int(max_behind_mb * MB)
        self.rss_limit_mb = rss_limit_mb
        self._last_plot = 0.0
        self.stream = None                 # StreamDecoder of the followed file
        self.mapping = None
        # [t_first_ns, t_last_ns, occupancy, n_hits, bytes, packets, dropped]
        self.records = []
        self._dropped_pending = 0          # drops reported before any record exists
        self.files_done = []
        self.total_hits = 0
        self.read_bytes = 0
        self.skipped_bytes = 0
        self.skipped_packets = 0
        self.memory_events = 0
        self.mapping_error = None
        self._mapping_tried = False
        self._owners, self._owners_for = {}, None    # chip -> where, for this mapping

    # -- following ------------------------------------------------------------

    def _newest_raw_dir(self):
        """The sub-run raw directory holding the most recently written capture.

        Looks only inside the most recently modified run, so the cost does not
        grow with the number of runs on disk.
        """
        runs = [d for d in self.root.iterdir() if d.is_dir()] if self.root.is_dir() else []
        if not runs:
            return None
        newest_run = max(runs, key=lambda d: d.stat().st_mtime)
        best, best_t = None, -1.0
        for raw in raw_dirs(newest_run, self.raw_inner):
            c = captures(raw)
            if c:
                t = c[-1][2].stat().st_mtime
                if t > best_t:
                    best, best_t = raw, t
        return best

    def _newest(self):
        if self.follow_newest:
            raw = self._newest_raw_dir()
            if raw is None:
                return None
            self.raw_dir = raw
        c = captures(self.raw_dir) if self.raw_dir.is_dir() else []
        return c[-1][2] if c else None

    def _feed(self, cols, info):
        self.read_bytes += info.get("udp_payload_bytes", 0)
        dropped = info.get("dropped", 0)
        if cols is None or info["t_first_ns"] is None:
            # no hit to place, but kernel drops still belong to the window
            if self.records:
                self.records[-1][6] += dropped
            else:
                self._dropped_pending += dropped
            return
        n = len(cols["vmm"])
        self.total_hits += n
        occ = chip_occupancy(cols)
        t0, t1 = info["t_first_ns"], info["t_last_ns"]
        packets = info.get("vm3_packets", info["packets"])
        dropped += self._dropped_pending
        self._dropped_pending = 0
        last = self.records[-1] if self.records else None
        if (last is not None and self.bucket_s is not None
                and t0 // (self.bucket_s * 1e9) == last[0] // (self.bucket_s * 1e9)):
            last[1] = max(last[1], t1)
            last[0] = min(last[0], t0)
            last[2] += occ
            last[3] += n
            last[4] += info["udp_payload_bytes"]
            last[5] += packets
            last[6] += dropped
        else:
            self.records.append([t0, t1, occ, n, info["udp_payload_bytes"], packets, dropped])

    def _read_udp(self):
        """Drain the socket up to max_read bytes (and ~1 s); the kernel drops the rest."""
        if self.udp is None:
            from p2_decode import UdpStreamDecoder
            self.udp = UdpStreamDecoder(self.udp_addr, data_format=self.data_format)
        budget, t_end = self.max_read, time.monotonic() + 1.0
        while budget > 0 and time.monotonic() < t_end:
            cols, info = self.udp.poll(max_bytes=budget,
                                       max_seconds=max(0.01, t_end - time.monotonic()))
            self._feed(cols, info)
            if info["packets"] == 0:
                return
            budget -= max(info["udp_payload_bytes"], 1)

    def _catch_up(self):
        """Skip the backlog beyond max_behind, then read what is left."""
        f = self.stream.follower
        if f.backlog() > self.max_behind:
            before = f.skipped_packets
            self.skipped_bytes += self.stream.skip_to_tail(self.max_read)
            self.skipped_packets += f.skipped_packets - before
            # skipped packets lie inside the window's time span: count them like
            # kernel drops, so rates are scaled by the fraction actually read
            self._dropped_pending += f.skipped_packets - before
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
        """The mapping at the data's time; on failure None, with the reason kept.

        The live monitor never stops for a mapping problem: it then shows the
        electronics view and says why the detector view is missing.
        """
        if not self.use_mapping:
            return None
        when = self.when or datetime.fromtimestamp(t_ns / 1e9)
        try:
            m = p2_mapping.load_mapping(self.setup, when)
            self.mapping_error = None
            return m
        except (LookupError, ValueError, OSError) as e:
            self.mapping_error = f"{type(e).__name__}: {e}"
            return None

    def step(self):
        """One poll. Returns the live summary dict (also written to `out`)."""
        if self.udp_addr is not None:
            self._read_udp()
            return self._finish_step()
        newest = self._newest()
        if newest is not None and (self.stream is None or Path(self.stream.path) != newest):
            if self.stream is not None:
                self._catch_up()                   # rotation: finish the old file
                self.files_done.append(Path(self.stream.path).name)
                if Path(self.stream.path).parent != newest.parent:
                    self.mapping = None            # new sub-run: cabling by its time
                    self._mapping_tried = False
            self.stream = self._new_stream(newest)
        if self.stream is not None:
            self._catch_up()
        return self._finish_step()

    def _finish_step(self):
        if (self.mapping is None and self.use_mapping and self.records
                and not self._mapping_tried):
            self.mapping = self._mapping_for(self.records[0][0])
            self._mapping_tried = True             # once per sub-run, not every poll
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

    def window_fraction(self):
        """Packets decoded / packets that arrived, over the window (1.0 if none lost).

        Lost = dropped by the kernel (UDP) or skipped (file). Rates and volume
        are divided by live time x this fraction, i.e. scaled to the full stream.
        """
        got = sum(r[5] for r in self.records)
        lost = sum(r[6] for r in self.records)
        return got / (got + lost) if got + lost else 1.0

    def summary(self):
        counts, live, hits, nbytes = self.window()
        frac = self.window_fraction()
        seen = self.read_bytes + self.skipped_bytes
        s = {
            "written": datetime.now().isoformat(timespec="seconds"),
            "source": (f"udp://{self.udp_addr}" if self.udp_addr is not None
                       else "file"),
            "raw_dir": str(self.raw_dir) if self.raw_dir is not None else None,
            "run": (self.raw_dir.parent.parent.name if self.follow_newest and self.stream
                    else None),
            "subrun": (self.raw_dir.parent.name if self.follow_newest and self.stream
                       else None),
            "capture": Path(self.stream.path).name if self.stream else None,
            "captures_finished": list(self.files_done),
            "setup": self.mapping.setup.name if self.mapping else None,
            "mapping_error": self.mapping_error,
            "window_s": self.window_s,
            "live_time_s": round(live, 6),
            "hits_in_window": int(hits),
            "hits_total": int(self.total_hits),
            "data_age_s": None,
            "sampling": {"read_mb": round(self.read_bytes / 1e6, 3),
                         "skipped_mb": round(self.skipped_bytes / 1e6, 3),
                         "skipped_packets": int(self.skipped_packets),
                         "fraction_read": round(self.read_bytes / seen, 6) if seen else None,
                         # packets decoded / arrived in the window; rates are scaled by it
                         "window_fraction": round(frac, 6),
                         "dropped_packets": (int(self.udp.receiver.dropped_total)
                                             if self.udp is not None else None),
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
            eff = live * frac                    # live time of the data actually read
            s["volume"] = {"hits_per_s": round(hits / eff, 6),
                           "mb_per_min_udp": round(nbytes / 1e6 / eff * 60, 6)}
            s["rates"] = summarize(channel_rates(counts, eff, self.mapping))
            s["electronics"] = self.electronics(frac)
        return s

    def electronics(self, frac=1.0):
        """Every chip that fired, channel by channel, mapped or not.

        For fast testing (which channel does this cable land on?): per chip
        "fec:vmm", the rate of each of its 64 channels over the whole window
        and over the last RECENT_S seconds of data, and where the setup puts
        the chip (None = not in the setup). Like a live packet view, decoded.
        """
        if not self.records:
            return None
        newest = max(r[1] for r in self.records)
        occ_w = np.sum([r[2] for r in self.records], axis=0)
        live_w = (newest - min(r[0] for r in self.records)) / 1e9
        rec = [r for r in self.records if r[1] >= newest - RECENT_S * 1e9]
        occ_r = np.sum([r[2] for r in rec], axis=0)
        live_r = (newest - min(r[0] for r in rec)) / 1e9
        if live_r < 1.0:                     # one short poll: too little time for a rate
            occ_r, live_r = occ_w, live_w
        if self._owners_for is not self.mapping:
            self._owners, self._owners_for = _owners(self.mapping), self.mapping
        chips = {}
        for c in np.flatnonzero(occ_w.sum(axis=1)):
            fec, vmm = divmod(int(c), 32)
            where = self._owners.get((fec, vmm)) or self._owners.get((-1, vmm))
            chips[f"{fec}:{vmm}"] = {"where": where, "hits": int(occ_w[c].sum()),
                                     "window_hz": _hz(occ_w[c], live_w, frac),
                                     "recent_hz": _hz(occ_r[c], live_r, frac)}
        return {"window_s": round(live_w, 3), "recent_s": round(live_r, 3), "chips": chips}

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
            frac = self.window_fraction()
            what = summary["capture"] or summary["source"]
            title = (f"live: last {self.window_s:.0f} s of {what}\n"
                     f"{live:.1f} s of data, {hits:,} hits, newest packet "
                     f"{summary.get('data_age_s')} s ago"
                     + (f", {100 * frac:.0f} % of packets decoded, rates scaled"
                        if frac < 1 else ""))
            plot_counts(counts, live * frac, self.mapping, title, self.plot_dir)

    def run(self, interval_s=5.0, duration_s=None, log=None, oom_score_adj=None):
        """Poll every interval_s until interrupted (or for duration_s)."""
        if log is None:
            def log(msg):                       # flushed: readable through a pipe too
                print(msg, flush=True)
        if oom_score_adj is not None:
            set_oom_score_adj(oom_score_adj)
        t_end = time.time() + duration_s if duration_s is not None else None
        src = f"udp://{self.udp_addr}" if self.udp_addr is not None else self.raw_dir
        log(f"[live] {src} -> {self.out}  window {self.window_s} s, every "
            f"{interval_s} s, read <= {self.max_read / MB:g} MB/poll"
            + ("" if self.udp_addr is not None else
               f", skip beyond {self.max_behind / MB:g} MB"))
        while True:
            t0 = time.time()
            s = self.step()
            log(_status_line(s))
            if t_end is not None and time.time() >= t_end:
                return s
            time.sleep(max(0.0, interval_s - (time.time() - t0)))


def _status_line(s):
    what = s["capture"] or s["source"]
    if "rates" not in s:
        if s["hits_total"]:
            return (f"[live] {s['written']} {what}  {s['hits_total']:,} hits, "
                    "no rate yet (window spans no time)")
        return f"[live] {s['written']} waiting for data from {s['raw_dir'] or s['source']}"
    det = s["rates"].get("per_detector") or {}
    parts = [f"{k} {v['rate_hz']:.4g} Hz" for k, v in det.items()] or \
            [f"FEC {k} {v['rate_hz']:.4g} Hz" for k, v in s["rates"]["per_fec"].items()]
    frac = s["sampling"]["window_fraction"]
    samp = f"  decoded {100 * frac:.0f}%" if frac < 1 else ""
    return (f"[live] {s['written']} {what}  {s['volume']['hits_per_s']:.4g} hits/s"
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
