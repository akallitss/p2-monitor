"""`p2-monitor` command: reduce captures, print rates, plot, trend."""

import argparse
import json
import os

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
        scal = process_capture(a.pcap_file, a.store_dir, setup=a.setup, when=a.when,
                               mapping=not a.no_mapping, data_format=a.format,
                               keep_store=not a.drop_columns,
                               legacy_word_count=a.legacy_word_count)
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


if __name__ == "__main__":
    raise SystemExit(main())
