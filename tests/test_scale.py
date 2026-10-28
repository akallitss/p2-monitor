"""What the full experiment needs: many FECs, bounded memory, exact chunking."""

import json
import os

import numpy as np
import pandas as pd
import p2_decode
import pytest
from p2_decode.testing import hit_word, marker_word, srs_payload, write_pcapng

import p2_monitor as pmon
from p2_monitor.counts import CountsAccumulator, build_chip_counts, build_counts
from p2_monitor.live import LiveMonitor
from p2_monitor.scalars import percentile_from_hist

TWO_FECS = """
name: two_fecs
design: p2_basket
detectors:
  - {name: A, z_mm: 0.0}
  - {name: B, z_mm: 100.0}
cabling:
  - valid_from: 2026-01-01T00:00:00
    hybrids:
      - {fec: 1, detector: A, connector: 1, hybrid: 1, bottom_vmm: 2, top_vmm: 3}
      - {fec: 2, detector: B, connector: 1, hybrid: 2, bottom_vmm: 2, top_vmm: 3}
"""


def _rng_capture(path, n_packets=60, fecs=(1, 2, 3), seed=0, tsresol=3):
    rng = np.random.default_rng(seed)
    payloads = []
    for i in range(n_packets):
        words = [marker_word(v, int(rng.integers(1 << 30))) for v in (2, 3)]
        for _ in range(int(rng.integers(1, 30))):
            words.append(hit_word(int(rng.choice([2, 3, 7])), int(rng.integers(64)),
                                  int(rng.integers(1024)), int(rng.integers(4096)),
                                  int(rng.integers(256)), ot=int(rng.integers(2))))
        payloads.append(srs_payload(i, int(fecs[i % len(fecs)]), words))
    return write_pcapng(path, payloads, tsresol=tsresol)


def test_accumulator_equals_whole_capture(tmp_path):
    cap = _rng_capture(tmp_path / "c.pcapng")
    acc = CountsAccumulator()
    p2_decode.decode_to(cap, tmp_path / "s", max_words=37, on_chunk=acc.add)
    legacy, chip = acc.finish(tmp_path / "s")
    hits, _ = p2_decode.decode(cap)
    cols = {c: hits[c].to_numpy() for c in hits}
    ref, ref_chip = build_counts(cols), build_chip_counts(cols)
    for k in ref:
        assert legacy[k].dtype == ref[k].dtype and np.array_equal(legacy[k], ref[k]), k
    for k in ref_chip:
        assert np.array_equal(chip[k], ref_chip[k]), k


def test_percentile_from_hist_is_exact():
    rng = np.random.default_rng(3)
    for n in (1, 2, 7, 1000):
        x = rng.integers(0, 1024, size=n)
        h = np.bincount(x, minlength=1024)
        for q in (0.0, 0.1, 0.5, 0.9, 1.0):
            assert percentile_from_hist(h, q) == pytest.approx(np.percentile(x, 100 * q), abs=0)


def test_same_vmm_on_two_fecs_stays_separate(tmp_path):
    setup = tmp_path / "two_fecs.yaml"
    setup.write_text(TWO_FECS)
    words = lambda: [hit_word(2, 0, 10, 1, 1)] * 3
    cap = write_pcapng(tmp_path / "c.pcapng",
                       [srs_payload(0, 1, words()), srs_payload(0, 2, words()[:1])] * 5,
                       tsresol=3)
    scal = pmon.process_capture(cap, tmp_path / "st", setup=str(setup), when="2026-06-01")
    r = scal["rates"]
    assert r["per_chip"]["1:2"]["hits"] == 15 and r["per_chip"]["2:2"]["hits"] == 5
    assert r["per_detector"]["A"]["hits"] == 15 and r["per_detector"]["B"]["hits"] == 5
    assert scal["fecs"] == [1, 2]
    assert scal["per_station"] == {"A": {"n_hits": 15, "adc_p50": 10.0, "adc_p90": 10.0},
                                   "B": {"n_hits": 5, "adc_p50": 10.0, "adc_p90": 10.0}}
    # the old per-VMM counts.npz merges the two chips: that is why chip counts exist
    assert int(pmon.load_counts(tmp_path / "st", chip=False)["hits_per_vmm"][2]) == 20


def test_live_skips_backlog_and_reports_it(tmp_path):
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    _rng_capture(raw / "enp4s0f1_00001_20261024120000.pcapng", n_packets=400)
    size = os.path.getsize(raw / "enp4s0f1_00001_20261024120000.pcapng")
    lm = LiveMonitor(raw, mapping=False, window_s=3600,
                     max_read_mb=0.002, max_behind_mb=size / 4 / 2**20)
    s = lm.step()
    assert s["sampling"]["skipped_mb"] > 0 and 0 < s["sampling"]["fraction_read"] < 0.5
    assert "rates" in s and s["hits_total"] > 0


def test_live_window_buckets_are_bounded(tmp_path):
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    src = _rng_capture(tmp_path / "a.pcapng", n_packets=600)
    data = open(src, "rb").read()
    lm = LiveMonitor(raw, mapping=False, window_s=3600)
    dst = raw / "enp4s0f1_00001_20261024120000.pcapng"
    for i in range(1, 301):
        with open(dst, "wb") as f:
            f.write(data[:len(data) * i // 300])
        lm.step()
    assert len(lm.records) <= 61


def test_live_memory_cap_sheds_load(tmp_path):
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    _rng_capture(raw / "enp4s0f1_00001_20261024120000.pcapng", n_packets=50)
    lm = LiveMonitor(raw, mapping=False, rss_limit_mb=1)          # always exceeded
    s = lm.step()
    assert s["sampling"]["memory_events"] == 1 and len(lm.records) <= 1
    assert lm.max_read < 16 * 2**20
