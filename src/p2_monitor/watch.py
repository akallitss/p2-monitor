"""Watch a run directory and reduce every capture once dumpcap has closed it.

    p2-monitor watch <runs_dir | run_dir | subrun_dir> --setup lab_saclay_2026

Same layout and rules as the DAQ's vmm_processor_watcher.py:

    <subrun>/raw_daq_data/<capture>.pcapng  ->  <subrun>/hits_store/<capture>/

A capture is final when its size is non-zero and unchanged since the previous
poll, AND a newer capture of the same interface exists, or the sub-run wrote
its .capture_done marker, or the file is overdue (no write for twice the
rotation interval). Newest captures first: when behind, the freshest one is
the one worth seeing. A capture whose store already has scalars.json is done
(idempotent by output), so the watcher can be stopped and restarted freely.

Each reduce runs in a child process at low CPU priority with a time limit, so
a capture that hangs or eats the machine cannot take the watcher with it.
After each new store, <subrun>/hits_store/latest.json is rewritten with that
store's rates and volume: a fixed path a GUI can poll.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CAPTURE_DONE_MARKER = ".capture_done"

# Capture-file naming. Two producers, one parser (as DAQ common_functions.py):
#   dumpcap ring buffer:  <iface>_<seq:05d>_<YYYYMMDDHHMMSS>.pcapng
#   loop_daq / tcpdump:   <iface>_<YYYYMMDD-HHMMSS>_<seq>.pcapng
_DUMPCAP_RE = re.compile(r"^(?P<iface>.+)_(?P<seq>\d{5})_(?P<ts>\d{14})\.pcapn?g$")
_LOOPDAQ_RE = re.compile(r"^(?P<iface>.+)_(?P<ts>\d{8}-\d{6})_(?P<seq>\d+)\.pcapn?g$")


def parse_capture_name(name):
    """(iface, seq, start datetime) of a capture file name, or None."""
    base = os.path.basename(name)
    m = _DUMPCAP_RE.match(base)
    if m:
        return m["iface"], int(m["seq"]), datetime.strptime(m["ts"], "%Y%m%d%H%M%S")
    m = _LOOPDAQ_RE.match(base)
    if m:
        return m["iface"], int(m["seq"]), datetime.strptime(m["ts"], "%Y%m%d-%H%M%S")
    return None


def captures(raw_dir):
    """[(iface, seq, path)] of the capture files in raw_dir, oldest first."""
    out = []
    for f in Path(raw_dir).iterdir():
        p = parse_capture_name(f.name)
        if p is not None:
            out.append((p[0], p[1], f))
    return sorted(out, key=lambda e: (e[0], e[1]))


def finalized_captures(raw_dir, capture_duration_s, last_sizes, now=None):
    """Captures safe to reduce, newest first (see the module docstring).

    `last_sizes` persists between calls: a file is only final once its size
    has been seen unchanged, which guards the race where dumpcap has opened the
    next file but is still flushing this one.
    """
    entries = captures(raw_dir)
    if not entries:
        return []
    max_seq = {}
    for iface, seq, _f in entries:
        max_seq[iface] = max(max_seq.get(iface, -1), seq)
    capture_done = (Path(raw_dir) / CAPTURE_DONE_MARKER).exists()
    now = time.time() if now is None else now

    final = []
    for iface, seq, f in reversed(entries):
        try:
            st = f.stat()
        except OSError:
            continue
        if st.st_size == 0:
            continue
        rotated = seq < max_seq[iface]
        overdue = (now - st.st_mtime) > 2 * capture_duration_s
        if not (rotated or capture_done or overdue):
            continue
        prev = last_sizes.get(str(f))
        last_sizes[str(f)] = st.st_size
        if prev is not None and prev == st.st_size:
            final.append(f)
    return final


def raw_dirs(root, raw_inner="raw_daq_data"):
    """Every capture directory under root (root itself if it holds captures)."""
    root = Path(root)
    if root.is_dir() and (root.name == raw_inner or captures(root)):
        return [root]
    return sorted(p for p in root.glob(f"**/{raw_inner}") if p.is_dir())


def _newest_first(dirs):
    def key(d):
        try:
            return d.stat().st_mtime
        except OSError:
            return 0
    return sorted(dirs, key=key, reverse=True)


def _reduce_cmd(pcap, store_dir, opts):
    cmd = [sys.executable, "-m", "p2_monitor.cli", "reduce", str(pcap),
           "--store-dir", str(store_dir), "--format", opts.get("data_format", "SRS")]
    if opts.get("setup"):
        cmd += ["--setup", opts["setup"]]
    if opts.get("when"):
        cmd += ["--when", str(opts["when"])]
    if opts.get("no_mapping"):
        cmd.append("--no-mapping")
    if opts.get("drop_columns"):
        cmd.append("--drop-columns")
    if opts.get("calibration"):
        cmd += ["--calibration", str(opts["calibration"])]
    return cmd


def _child_env(threads):
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(threads)
    return env


def _capture_order(name):
    p = parse_capture_name(name or "")
    return (p[0], p[1]) if p else ("", -1)


def write_latest(store_root, store_dir):
    """<store_root>/latest.json: the newest capture's rates and volume, atomically.

    The watcher reduces newest first and then backfills, so a store finished
    later can be an older capture; it does not replace a newer one here.
    """
    with open(Path(store_dir) / "scalars.json") as f:
        s = json.load(f)
    cur = Path(store_root) / "latest.json"
    if cur.exists():
        try:
            with open(cur) as f:
                if _capture_order(json.load(f).get("capture")) > _capture_order(s.get("capture")):
                    return
        except (OSError, ValueError):
            pass
    latest = {k: s.get(k) for k in ("capture", "capture_start", "setup", "n_hits",
                                     "rates", "volume", "versions")}
    latest["store"] = str(store_dir)
    latest["written"] = datetime.now().isoformat(timespec="seconds")
    tmp = Path(store_root) / ".latest.json.tmp"
    with open(tmp, "w") as f:
        json.dump(latest, f, indent=2)
    os.replace(tmp, Path(store_root) / "latest.json")


class Watcher:
    """Reduce finalized captures under `root`, one per scan step."""

    def __init__(self, root, setup=None, when=None, raw_inner="raw_daq_data",
                 store_inner="hits_store", capture_duration_s=44.4,
                 data_format="SRS", no_mapping=False, drop_columns=False,
                 calibration=None, max_attempts=3, timeout_s=3600, nice=19,
                 threads=4, log=print):
        self.root = Path(root)
        self.raw_inner, self.store_inner = raw_inner, store_inner
        self.capture_duration_s = capture_duration_s
        self.opts = {"setup": setup, "when": when, "data_format": data_format,
                     "no_mapping": no_mapping, "drop_columns": drop_columns,
                     "calibration": calibration}
        self.max_attempts, self.timeout_s = max_attempts, timeout_s
        self.nice, self.threads = nice, threads
        self.log = log
        self.last_sizes = {}
        self.failures = {}

    def store_dir_for(self, pcap):
        return Path(pcap).parent.parent / self.store_inner / Path(pcap).stem

    def pending(self):
        """Finalized captures without a complete store, newest first."""
        out = []
        for raw in _newest_first(raw_dirs(self.root, self.raw_inner)):
            for pcap in finalized_captures(raw, self.capture_duration_s, self.last_sizes):
                if (self.store_dir_for(pcap) / "scalars.json").exists():
                    continue
                if self.failures.get(str(pcap), 0) >= self.max_attempts:
                    continue
                out.append(pcap)
        return out

    def reduce_one(self, pcap):
        """Reduce one capture in a child process. True on success."""
        store = self.store_dir_for(pcap)
        store.parent.mkdir(parents=True, exist_ok=True)
        cmd = _reduce_cmd(pcap, store, self.opts)
        t0 = time.time()
        nice = self.nice
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=self.timeout_s, env=_child_env(self.threads),
                               preexec_fn=(lambda: os.nice(nice)) if nice else None)
            ok, why = r.returncode == 0, (r.stderr.strip().splitlines() or ["?"])[-1]
        except subprocess.TimeoutExpired:
            ok, why = False, f"timeout after {self.timeout_s} s"
        dt = time.time() - t0
        if ok:
            self.failures.pop(str(pcap), None)
            write_latest(store.parent, store)
            self.log(f"[watch] {pcap.name} -> {store}  ({dt:.1f} s)")
        else:
            n = self.failures.get(str(pcap), 0) + 1
            self.failures[str(pcap)] = n
            self.log(f"[watch] FAILED {pcap.name} ({why}), attempt {n}/{self.max_attempts}")
        return ok

    def step(self):
        """Reduce the newest pending capture. True if one was attempted."""
        todo = self.pending()
        if not todo:
            return False
        self.reduce_one(todo[0])
        return True

    def run(self, poll_s=10, once=False):
        """Loop forever (or, with once, until nothing is pending twice in a row)."""
        self.log(f"[watch] {self.root}  setup={self.opts['setup'] or '(resolved)'}  "
                 f"rotation {self.capture_duration_s} s, poll {poll_s} s")
        idle = 0
        while True:
            if self.step():
                idle = 0
                continue
            idle += 1
            if once and idle >= 2:     # sizes need two looks before a file is final
                return
            time.sleep(0 if once else poll_s)
