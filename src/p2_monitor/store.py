"""Reading stores back: one store, or every store under a run directory."""

import glob
import json
import os

import numpy as np
import pandas as pd

from .counts import merge_counts


def is_store(path):
    """A complete store: renamed into place, so never a .partial directory."""
    path = os.fspath(path)
    return (os.path.isdir(path) and not path.rstrip("/").endswith(".partial")
            and os.path.isfile(os.path.join(path, "scalars.json")))


def load_counts(store, chip=True):
    """Chip-level counts (counts_chip.npz); chip=False, or a store from before
    p2-monitor 0.3, gives the per-VMM counts.npz."""
    name = "counts_chip.npz" if chip else "counts.npz"
    if chip and not os.path.isfile(os.path.join(store, name)):
        name = "counts.npz"
    with np.load(os.path.join(store, name)) as z:
        return {k: z[k] for k in z.files}


def load_scalars(store):
    with open(os.path.join(store, "scalars.json")) as f:
        return json.load(f)


def find_stores(root):
    """Every complete store under `root` (the root itself if it is one)."""
    root = os.fspath(root)
    if is_store(root):
        return [root]
    found = glob.glob(os.path.join(root, "**", "scalars.json"), recursive=True)
    return sorted(os.path.dirname(p) for p in found if is_store(os.path.dirname(p)))


def store_mapping(store):
    """The Mapping a store was reduced with (from setup_resolved.json), or None.

    Rebuilt from the setup name, the recorded time and table options, so plots
    and later analysis use exactly the cabling of the reduce, not a fresh guess.
    """
    path = os.path.join(store, "setup_resolved.json")
    if not os.path.isfile(path):
        return None
    import p2_mapping
    with open(path) as f:
        r = json.load(f)
    opt = r.get("table_options") or {}
    orders = {}
    for k, o in (opt.get("orders") or {}).items():
        det, conn, half = k.split(":")
        orders[(det, int(conn.lstrip("c")), half)] = o
    m = p2_mapping.load_mapping(r["setup"], r["when"], order=opt.get("order"),
                                orders=orders, bottom_half=opt.get("bottom_half"))
    if m.setup.sha256 != r.get("setup_sha256"):
        import warnings
        warnings.warn(f"{store}: setup {r['setup']!r} changed since this store was "
                      "reduced; using the current file")
    return m


def merged_counts(stores, chip=True):
    return merge_counts(load_counts(s, chip) for s in stores)


def trend_table(root):
    """One row per store under `root`, ordered by capture start.

    Columns: store, capture, capture_start, live_time_s, n_hits, hits_per_s,
    mb_per_min_file, then rate_hz:<detector> and rate_hz:chip<fec:vmm> for
    each one present.
    """
    root = os.fspath(root)
    rows = []
    for s in find_stores(root):
        r = load_scalars(s)
        v, rt = r.get("volume") or {}, r.get("rates") or {}
        row = {"store": os.path.relpath(s, root) if s != root else ".",
               "capture": r.get("capture"), "capture_start": r.get("capture_start"),
               "live_time_s": v.get("live_time_s"), "n_hits": r.get("n_hits"),
               "hits_per_s": v.get("hits_per_s"),
               "mb_per_min_file": v.get("mb_per_min_file")}
        for det, d in (rt.get("per_detector") or {}).items():
            row[f"rate_hz:{det}"] = d["rate_hz"]
        for chip, d in (rt.get("per_chip") or {}).items():
            row[f"rate_hz:chip{chip}"] = d["rate_hz"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) and df["capture_start"].notna().any():
        df["capture_start"] = pd.to_datetime(df["capture_start"])
        df = df.sort_values("capture_start", kind="stable").reset_index(drop=True)
    return df
