"""QA plots drawn from histograms: the histograms equal the hits, the plots exist."""

import json
import os

import numpy as np

import p2_decode
import p2_monitor as pmon
from p2_monitor.counts import NV
from p2_monitor.qa import render

from test_monitor import LAB, WHEN, capture  # noqa: F401  (fixture)


def _reduce(capture, d):
    pmon.process_capture(capture, d, setup=LAB, when=WHEN, keep_store=False)


def test_histograms_equal_hits(capture, tmp_path):
    _reduce(capture, tmp_path / "s")
    c = pmon.load_counts(tmp_path / "s")
    hits, _ = p2_decode.decode(capture)
    for (fec, vmm), g in hits.groupby(["fec", "vmm"]):
        chip = int(fec) * NV + int(vmm)
        assert np.array_equal(c["chip_bcid"][chip],
                              np.bincount(g["bcid"], minlength=c["chip_bcid"].shape[1]))
        assert np.array_equal(c["chip_tdc"][chip],
                              np.bincount(g["tdc"], minlength=c["chip_tdc"].shape[1]))
        assert c["chip_adc_ot1"][chip].sum() == int(g["over_threshold"].sum())
        assert c["chip_adc_vs_ch"][chip].sum() == len(g) == c["chip_ts"][chip].sum()
    assert int(c["rate_s"].sum()) == len(hits)


def test_plots_and_events(capture, tmp_path):
    _reduce(capture, tmp_path / "s")
    names = {os.path.basename(p) for p in render(tmp_path / "s", tmp_path / "qa",
                                                  events_json=True)}
    for n in ("adc", "adc_ot", "adc_vs_ch", "ot", "chno", "hits_per_vmm", "bcid", "tdc",
              "offset", "timestamp_ns", "rate_vs_time"):
        assert f"s_{n}.png" in names, n
    ev = json.load(open(tmp_path / "qa" / "events.json"))
    hits, _ = p2_decode.decode(capture)
    assert ev["n_hits"] == len(hits)
    assert ev["hits_per_vmm"] == {str(v): int(n) for v, n in hits["vmm"].value_counts().items()}


def test_merges_stores(capture, tmp_path):
    for d in ("a", "b"):
        _reduce(capture, tmp_path / d)
    render([tmp_path / "a", tmp_path / "b"], tmp_path / "qa", events_json=True)
    hits, _ = p2_decode.decode(capture)
    assert json.load(open(tmp_path / "qa" / "events.json"))["n_hits"] == 2 * len(hits)
