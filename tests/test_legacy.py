"""Same store content as the reduce it replaces.

Runs only when both are available:
    P2_MONITOR_LEGACY   directory holding vmm_reduce.py, vmm_decode.py and
                        vmm_stations.py (P2_basket_online_analysis)
    P2_MONITOR_CAPTURE  one real .pcapng
The legacy reduce reads one word less per packet, so p2-monitor runs with
legacy_word_count=True here. Efficiency is not ported yet; both run without it.
"""

import importlib
import json
import os
import sys

import numpy as np
import pytest

import p2_monitor as pmon

LEGACY = os.environ.get("P2_MONITOR_LEGACY")
CAPTURE = os.environ.get("P2_MONITOR_CAPTURE")
pytestmark = pytest.mark.skipif(not (LEGACY and CAPTURE),
                                reason="set P2_MONITOR_LEGACY and P2_MONITOR_CAPTURE")

# keys whose values legitimately differ: paths, timings
VOLATILE = {"source", "decode_s", "reduce_s", "total_s"}


def test_same_counts_and_scalars(tmp_path):
    sys.path.insert(0, LEGACY)
    old = importlib.import_module("vmm_reduce")
    old_scal = old.process_file(CAPTURE, str(tmp_path / "old"), do_efficiency=False)
    new_scal = pmon.process_capture(CAPTURE, tmp_path / "new", setup="lab_saclay_2026",
                                    when="2026-10-24", legacy_word_count=True)

    a, b = pmon.load_counts(tmp_path / "old"), pmon.load_counts(tmp_path / "new")
    assert set(a) == set(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k

    old_disk = json.load(open(tmp_path / "old" / "scalars.json"))
    for k, v in old_disk.items():
        if k not in VOLATILE:
            assert new_scal[k] == v, k
