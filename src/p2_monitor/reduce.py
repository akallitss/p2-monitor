"""One capture to one store: hit columns, counts.npz, scalars.json.

The store is built under <store_dir>.partial and renamed into place only once
everything succeeded, so a directory appearing at <store_dir> is complete by
construction. That atomic rename is the handoff to whatever plots or serves
the results: it never has to guess whether a store is still being written.

Store layout (a superset of the vmm_reduce.py store, so its readers keep
working):
    <column>.npy, meta.json   p2-decode column store
    counts.npz                per-VMM histograms (counts.py)
    scalars.json              trend record + "rates" + "volume" + versions
    setup_resolved.json       the mapping used (setup, cabling entry, hashes)
"""

import datetime as dt
import json
import os
import shutil
import time

import numpy as np
import p2_decode
import p2_mapping

from . import __version__
from .counts import build_counts
from .rates import channel_rates, summarize, volume
from .scalars import add_detector_scalars, build_scalars


def capture_start(meta):
    """Local wall-clock time of the capture's first packet, or None."""
    t = (meta.get("capture") or {}).get("t_first_ns")
    return dt.datetime.fromtimestamp(t / 1e9) if t is not None else None


def process_capture(pcap, store_dir, setup=None, when=None, mapping=True,
                    data_format="SRS", calibration=None, keep_store=True,
                    legacy_word_count=False):
    """pcapng -> store. Returns the scalars dict.

    setup:    p2-mapping setup name or path; None resolves it ($P2_SETUP, ...)
    when:     time that picks the cabling; default the capture's first packet
    mapping:  False reduces in electronics coordinates only (no setup needed)
    """
    t0 = time.time()
    store_dir = str(store_dir).rstrip("/")
    partial = store_dir + ".partial"
    if os.path.isdir(partial):
        shutil.rmtree(partial, ignore_errors=True)

    meta = p2_decode.decode_to(pcap, partial, data_format=data_format,
                               calibration=calibration,
                               legacy_word_count=legacy_word_count)
    t_decode = time.time() - t0

    m = None
    if mapping:
        m = p2_mapping.load_mapping(setup, when or capture_start(meta))
        m.write_resolved(os.path.join(partial, "setup_resolved.json"))

    cols, _ = p2_decode.load_hits(partial, derive_cols=False)
    counts = build_counts(cols)
    np.savez_compressed(os.path.join(partial, "counts.npz"), **counts)

    # VMMs expected in the data: the cabled ones and the trigger VMM. Unlike
    # vmm_reduce.py (which took vmm_stations' 2-19), trigger hits are not
    # counted as unknown.
    known = None
    if m is not None:
        known = set(int(v) for v in m.table["vmm"].unique())
        if m.setup.trigger:
            known.add(int(m.setup.trigger["vmm"]))
    scal = build_scalars(cols, counts, meta, known)
    scal["source"] = os.path.abspath(str(pcap))
    scal["capture"] = os.path.basename(str(pcap))
    scal["mtime"] = os.path.getmtime(pcap)
    scal["decode_s"] = round(t_decode, 2)
    if m is not None:
        add_detector_scalars(scal, cols, m)
    del cols

    vol = volume(meta, file_bytes=os.path.getsize(pcap))
    scal["volume"] = vol
    live = vol.get("live_time_s")
    scal["rates"] = {"live_time_s": live,
                     **summarize(channel_rates(counts, live, m))}
    t_start = capture_start(meta)
    scal["capture_start"] = t_start.isoformat() if t_start else None
    scal["setup"] = m.setup.name if m else None
    scal["versions"] = {"p2_decode": p2_decode.__version__,
                        "p2_mapping": p2_mapping.__version__ if m else None,
                        "p2_monitor": __version__}

    scal["reduce_s"] = round(time.time() - t0 - t_decode, 2)
    with open(os.path.join(partial, "scalars.json"), "w") as f:
        json.dump(scal, f, indent=2, default=str)

    if os.path.isdir(store_dir):
        shutil.rmtree(store_dir, ignore_errors=True)
    os.rename(partial, store_dir)          # <- the atomic handoff

    if not keep_store:
        # counts + scalars are the archive; the columns are regenerable
        for f in os.listdir(store_dir):
            if f.endswith(".npy"):
                os.unlink(os.path.join(store_dir, f))

    scal["total_s"] = round(time.time() - t0, 2)
    return scal
