"""Live monitoring: rates from the capture dumpcap is writing, as data arrive.

    p2-monitor live <subrun>/raw_daq_data --setup lab_saclay_2026 --window 60

Every `interval` seconds the newest capture is polled with p2_decode's
StreamDecoder (only the bytes appended since the last poll are read). Each
poll's hits become a small counts record; the records of the last `window`
seconds of packet time are summed, and their rates written atomically to a
JSON file (default <raw_dir>/../live.json) that a GUI can poll. When dumpcap
rotates to a new file, the old one is read to its end first, then followed
no further.

Times are the packets' capture timestamps, so the window and the rates are
in data time; `data_age_s` says how long ago (wall clock) the newest packet
arrived, which shows a stalled DAQ at a glance.
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import p2_mapping
from p2_decode import StreamDecoder

from .counts import build_counts, merge_counts
from .rates import channel_rates, summarize
from .watch import captures


class LiveMonitor:
    """Follows the newest capture in raw_dir and keeps windowed rates."""

    def __init__(self, raw_dir, setup=None, when=None, mapping=True, window_s=60.0,
                 data_format="SRS", out=None, plot_dir=None, plot_every_s=0):
        self.raw_dir = Path(raw_dir)
        self.setup, self.when, self.use_mapping = setup, when, mapping
        self.window_s = window_s
        self.data_format = data_format
        self.out = Path(out) if out else self.raw_dir.parent / "live.json"
        self.plot_dir, self.plot_every_s = plot_dir, plot_every_s
        self._last_plot = 0.0
        self.stream = None                 # StreamDecoder of the followed file
        self.mapping = None
        self.records = deque()             # (t_first_ns, t_last_ns, counts, n_hits, bytes)
        self.files_done = []
        self.total_hits = 0

    # -- following ------------------------------------------------------------

    def _newest(self):
        c = captures(self.raw_dir) if self.raw_dir.is_dir() else []
        return c[-1][2] if c else None

    def _feed(self, cols, info):
        if cols is None or info["t_first_ns"] is None:
            return
        n = len(cols["vmm"])
        self.total_hits += n
        self.records.append((info["t_first_ns"], info["t_last_ns"], build_counts(cols),
                             n, info["udp_payload_bytes"]))

    def _drain(self):
        """Read the followed file to its current end."""
        while True:
            cols, info = self.stream.poll()
            self._feed(cols, info)
            if info["packets"] == 0:
                return

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
                self._drain()                      # rotation: finish the old file
                self.files_done.append(Path(self.stream.path).name)
            self.stream = StreamDecoder(newest, data_format=self.data_format)
        if self.stream is not None:
            self._drain()
        if self.mapping is None and self.use_mapping and self.records:
            self.mapping = self._mapping_for(self.records[0][0])
        self._prune()
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
        if any(r[1] < cut for r in self.records):
            self.records = deque(r for r in self.records if r[1] >= cut)

    # -- output ---------------------------------------------------------------

    def window(self):
        """(merged counts, live seconds, hits, payload bytes) over the window."""
        if not self.records:
            return None, 0.0, 0, 0
        counts = merge_counts(r[2] for r in self.records)
        live = (max(r[1] for r in self.records) - min(r[0] for r in self.records)) / 1e9
        return (counts, live, sum(r[3] for r in self.records),
                sum(r[4] for r in self.records))

    def summary(self):
        counts, live, hits, nbytes = self.window()
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
            title = (f"live: last {self.window_s:.0f} s of {summary['capture']}\n"
                     f"{live:.1f} s of data, {hits:,} hits, "
                     f"newest packet {summary.get('data_age_s')} s ago")
            plot_counts(counts, live, self.mapping, title, self.plot_dir)

    def run(self, interval_s=5.0, duration_s=None, log=print):
        """Poll every interval_s until interrupted (or for duration_s)."""
        t_end = time.time() + duration_s if duration_s is not None else None
        log(f"[live] {self.raw_dir} -> {self.out}  window {self.window_s} s, "
            f"every {interval_s} s")
        while True:
            t0 = time.time()
            s = self.step()
            log(_status_line(s))
            if t_end and time.time() >= t_end:
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
            [f"VMM {k} {v['rate_hz']:.4g} Hz" for k, v in s["rates"]["per_vmm"].items()]
    return (f"[live] {s['written']} {s['capture']}  {s['volume']['hits_per_s']:.4g} hits/s"
            f"  {s['volume']['mb_per_min_udp']:.3g} MB/min  | " + "  ".join(parts)
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
