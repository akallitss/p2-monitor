"""`p2-monitor` command: reduce captures, print rates, plot, trend."""

import argparse
import json
import os
import sys

from .reduce import process_capture
from .store import find_stores, load_scalars, trend_table


def _print_rates(scal):
    r, v = scal.get("rates") or {}, scal.get("volume") or {}
    print(f"{scal.get('capture')}  setup={scal.get('setup')}  "
          f"start={scal.get('capture_start')}")
    print(f"  live {v.get('live_time_s')} s  hits {scal.get('n_hits'):,}  "
          f"{v.get('hits_per_s')} hits/s  {v.get('mb_per_min_file')} MB/min (file)")
    if v.get("frame_loss"):
        print(f"  frame loss: {v['frame_loss']}")
    for level in ("per_detector", "per_connector", "per_hybrid", "per_vmm"):
        rows = r.get(level)
        if not rows:
            continue
        print(f"  {level}:")
        for k, d in rows.items():
            areal = (f"  {d['rate_hz_per_cm2']:.4g} Hz/cm2"
                     if "rate_hz_per_cm2" in d else "")
            print(f"    {k:14s} {d['rate_hz']:>12.4g} Hz  live "
                  f"{d['live_channels']:>3}/{d['channels']:<3}{areal}")
    if r.get("hottest_channel"):
        h = r["hottest_channel"]
        print(f"  hottest channel: VMM {h['vmm']} ch {h['ch']} at {h['rate_hz']:.4g} Hz")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p2-monitor",
                                 description="P2 VMM readout monitoring.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("reduce", help="capture -> store (counts, scalars, rates)")
    p.add_argument("pcap_file")
    p.add_argument("--store-dir", required=True)
    p.add_argument("--setup", help="p2-mapping setup (else $P2_SETUP, else the "
                                   "machine default)")
    p.add_argument("--when", help="time that picks the cabling (default: the "
                                  "capture's first packet)")
    p.add_argument("--no-mapping", action="store_true",
                   help="electronics view only; no setup needed")
    p.add_argument("--format", default="SRS", choices=["SRS", "TRG"])
    p.add_argument("--legacy-word-count", action="store_true")
    p.add_argument("--drop-columns", action="store_true",
                   help="keep only counts.npz + scalars.json")
    p.add_argument("--calibration", help="vmm-sdat calibration JSON")
    p.add_argument("--max-words", type=int, default=None,
                   help="decode chunk size in words; bounds memory (default "
                        "p2_decode.MAX_WORDS, ~250 MB peak)")
    # accepted so p2-monitor-reduce is a drop-in for vmm_reduce.py as the DAQ
    # watcher's worker; there is no efficiency stage yet
    p.add_argument("--no-efficiency", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--eff-window", type=float, help=argparse.SUPPRESS)
    p.add_argument("--mapping-optional", action="store_true",
                   help="if no mapping can be had, reduce in electronics coordinates "
                        "and record why (scalars mapping_error) instead of failing")

    p = sub.add_parser("watch", help="reduce every capture once dumpcap closed it")
    p.add_argument("root", help="runs dir, run dir, sub-run dir or raw_daq_data dir")
    p.add_argument("--setup")
    p.add_argument("--when", help="cabling time for every capture (default: each "
                                  "capture's first packet); for replaying old data")
    p.add_argument("--no-mapping", action="store_true")
    p.add_argument("--format", default="SRS", choices=["SRS", "TRG"])
    p.add_argument("--capture-duration", type=float, default=44.4, metavar="S",
                   help="dumpcap rotation interval (default 44.4 s)")
    p.add_argument("--poll", type=float, default=10, metavar="S",
                   help="seconds between scans (default 10)")
    p.add_argument("--raw-inner", default="raw_daq_data")
    p.add_argument("--store-inner", default="hits_store")
    p.add_argument("--drop-columns", action="store_true")
    p.add_argument("--calibration")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--max-words", type=int, help="decode chunk size (bounds memory)")
    p.add_argument("--timeout", type=float, default=3600, metavar="S",
                   help="limit per capture (default 3600 s)")
    p.add_argument("--once", action="store_true",
                   help="reduce what is final now, then exit")

    p = sub.add_parser("live", help="rates as data arrive: from the capture being "
                                    "written, or from the network (--udp)")
    p.add_argument("raw_dir", nargs="?",
                   help="the directory dumpcap writes into (or, with --follow-newest, "
                        "the runs directory); omit with --udp")
    p.add_argument("--udp", metavar="HOST:PORT",
                   help="read the FEC's stream from this address (e.g. the data NIC "
                        "192.168.1.16:6006) instead of a file; dumpcap is not affected")
    p.add_argument("--follow-newest", action="store_true",
                   help="raw_dir is a runs dir: follow the newest sub-run by itself")
    p.add_argument("--setup")
    p.add_argument("--when", help="time that picks the cabling (default: first packet)")
    p.add_argument("--no-mapping", action="store_true")
    p.add_argument("--format", default="SRS", choices=["SRS", "TRG"])
    p.add_argument("--window", type=float, default=60, metavar="S",
                   help="rates over the last S seconds of data (default 60)")
    p.add_argument("--interval", type=float, default=5, metavar="S",
                   help="seconds between polls (default 5)")
    p.add_argument("--out", help="JSON written every poll (default <raw_dir>/../live.json)")
    p.add_argument("--plot-dir", help="also redraw the rate PNGs here")
    p.add_argument("--plot-every", type=float, default=60, metavar="S",
                   help="seconds between redraws (default 60)")
    p.add_argument("--duration", type=float, metavar="S", help="stop after S seconds")
    p.add_argument("--max-read-mb", type=float, default=16,
                   help="bytes read per poll at most (default 16 MB)")
    p.add_argument("--max-behind-mb", type=float, default=64,
                   help="skip to the newest data when more is unread (default 64 MB)")
    p.add_argument("--rss-limit-mb", type=float,
                   help="shed the window and halve reads above this own memory")
    p.add_argument("--oom-score-adj", type=int, default=500,
                   help="kernel OOM priority of this process (default 500, as the DAQ "
                        "gives its restartable jobs)")

    p = sub.add_parser("rates", help="print the rates of one store")
    p.add_argument("store")
    p.add_argument("--json", action="store_true", help="print the raw JSON")

    p = sub.add_parser("plot", help="rate PNGs for the store(s) under a directory")
    p.add_argument("root")
    p.add_argument("--out", required=True, help="output directory")

    p = sub.add_parser("trend", help="one row per store under a directory, as CSV")
    p.add_argument("root")
    p.add_argument("--csv", help="write here instead of printing")
    a = ap.parse_args(argv)

    if a.cmd == "reduce":
        cal = None
        if a.calibration:
            from p2_decode import load_calibration
            cal = load_calibration(a.calibration)
        scal = process_capture(a.pcap_file, a.store_dir, setup=a.setup, when=a.when,
                               mapping=not a.no_mapping, data_format=a.format,
                               calibration=cal, keep_store=not a.drop_columns,
                               legacy_word_count=a.legacy_word_count,
                               mapping_optional=a.mapping_optional,
                               **({"max_words": a.max_words} if a.max_words else {}))
        print(f"{os.path.basename(a.pcap_file)}: {scal['n_hits']:,} hits  "
              f"decode={scal['decode_s']}s  reduce={scal['reduce_s']}s  "
              f"total={scal['total_s']}s -> {a.store_dir}")
        return 0
    if a.cmd == "rates":
        scal = load_scalars(a.store)
        if a.json:
            print(json.dumps({k: scal.get(k) for k in ("rates", "volume")}, indent=2))
        else:
            _print_rates(scal)
        return 0
    if a.cmd == "plot":
        from .plots import plot_store
        for path in plot_store(a.root, a.out):
            print(path)
        return 0
    if a.cmd == "watch":
        from .watch import Watcher
        Watcher(a.root, setup=a.setup, when=a.when, raw_inner=a.raw_inner,
                store_inner=a.store_inner,
                capture_duration_s=a.capture_duration, data_format=a.format,
                no_mapping=a.no_mapping, drop_columns=a.drop_columns,
                calibration=a.calibration, max_attempts=a.max_attempts,
                timeout_s=a.timeout, max_words=a.max_words).run(poll_s=a.poll, once=a.once)
        return 0
    if a.cmd == "live":
        from .live import LiveMonitor
        if (a.raw_dir is None) == (a.udp is None):
            raise SystemExit("live: give either a raw_dir or --udp HOST:PORT")
        lm = LiveMonitor(a.raw_dir, udp=a.udp, setup=a.setup, when=a.when,
                         mapping=not a.no_mapping,
                         window_s=a.window, data_format=a.format, out=a.out,
                         plot_dir=a.plot_dir, plot_every_s=a.plot_every if a.plot_dir else 0,
                         max_read_mb=a.max_read_mb, max_behind_mb=a.max_behind_mb,
                         rss_limit_mb=a.rss_limit_mb, follow_newest=a.follow_newest)
        try:
            lm.run(interval_s=a.interval, duration_s=a.duration,
                   oom_score_adj=a.oom_score_adj)
        except KeyboardInterrupt:
            pass
        return 0
    if a.cmd == "trend":
        if not find_stores(a.root):
            raise SystemExit(f"no store under {a.root}")
        df = trend_table(a.root)
        if a.csv:
            df.to_csv(a.csv, index=False)
            print(f"{len(df)} stores -> {a.csv}")
        else:
            print(df.to_string(index=False))
        return 0


def reduce_main(argv=None):
    """`p2-monitor-reduce`: the reduce command with vmm_reduce.py's calling
    convention (capture first, then options), for the DAQ watcher's worker."""
    return main(["reduce"] + list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
