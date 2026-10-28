import json
import os
import time

import numpy as np
import p2_decode
import pytest
from p2_decode.testing import hit_word, marker_word, srs_payload, write_pcapng

import p2_monitor as pmon
from p2_monitor.live import LiveMonitor
from p2_monitor.watch import Watcher, finalized_captures, parse_capture_name

LAB, WHEN = "lab_saclay_2026", "2026-10-24T12:00"


def _capture(path, n_packets, vmm=2):
    """n packets 1 ms apart, 3 hits on `vmm` and one on VMM 8 each."""
    words = [marker_word(vmm, 7), hit_word(vmm, 0, 1, 1, 1), hit_word(vmm, 1, 1, 1, 1),
             hit_word(vmm, 2, 1, 1, 1), hit_word(8, 3, 1, 1, 1)]
    return write_pcapng(path, [srs_payload(0, 1, words)] * n_packets, tsresol=3)


def _name(seq):
    return f"enp4s0f1_{seq:05d}_20261024120{seq:03d}.pcapng"


def test_parse_capture_name():
    assert parse_capture_name("enp4s0f1_00007_20261024120000.pcapng")[:2] == ("enp4s0f1", 7)
    assert parse_capture_name("enp4s0f1_20251115-205857_1.pcapng")[:2] == ("enp4s0f1", 1)
    assert parse_capture_name("notes.txt") is None


def test_finalized_rules(tmp_path):
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    a, b = _capture(raw / _name(1), 3), _capture(raw / _name(2), 3)
    sizes = {}
    assert finalized_captures(raw, 44.4, sizes) == []          # sizes not seen yet
    assert finalized_captures(raw, 44.4, sizes) == [a]         # rotated; b is newest
    (raw / ".capture_done").touch()
    # the newest file passes the gate now, but its size needs a second look
    assert finalized_captures(raw, 44.4, sizes) == [a]
    assert finalized_captures(raw, 44.4, sizes) == [b, a]
    os.unlink(raw / ".capture_done")
    assert finalized_captures(raw, 44.4, {}, now=time.time() + 100) == []
    sizes = {}
    finalized_captures(raw, 44.4, sizes, now=time.time() + 100)
    assert finalized_captures(raw, 44.4, sizes, now=time.time() + 100) == [b, a]


def test_watch_once_reduces_and_is_idempotent(tmp_path):
    raw = tmp_path / "run_5" / "sub_00" / "raw_daq_data"
    raw.mkdir(parents=True)
    _capture(raw / _name(1), 10)
    _capture(raw / _name(2), 20)
    (raw / ".capture_done").touch()
    log = []
    w = Watcher(tmp_path, setup=LAB, when=WHEN, log=log.append)
    w.run(once=True)
    store_root = raw.parent / "hits_store"
    assert sorted(os.listdir(store_root)) == ["enp4s0f1_00001_20261024120001",
                                              "enp4s0f1_00002_20261024120002",
                                              "latest.json"]
    latest = json.load(open(store_root / "latest.json"))
    # newest capture (seq 2, 20 packets), although seq 1 was reduced after it
    assert latest["capture"] == _name(2) and latest["n_hits"] == 20 * 4
    assert latest["rates"]["per_detector"]["P2_IN"]["hits"] == 60
    assert not any("FAILED" in m for m in log)
    n_logged = len(log)
    Watcher(tmp_path, setup=LAB, when=WHEN, log=log.append).run(once=True)
    assert len(log) == n_logged + 1                             # only the start line


def test_watch_counts_failures(tmp_path):
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    (raw / _name(1)).write_bytes(b"not a pcapng at all" * 10)
    (raw / ".capture_done").touch()
    log = []
    w = Watcher(raw, setup=LAB, when=WHEN, max_attempts=2, log=log.append)
    w.run(once=True)
    assert sum("FAILED" in m for m in log) == 2


def _grow(src, dst, pieces):
    raw = open(src, "rb").read()
    cuts = np.linspace(0, len(raw), pieces + 1).astype(int)[1:]
    prev = 0
    for c in cuts:
        with open(dst, "ab") as f:
            f.write(raw[prev:c])
        prev = c
        yield


def test_live_follows_growing_file_and_rotation(tmp_path):
    src1, src2 = _capture(tmp_path / "a.pcapng", 30), _capture(tmp_path / "b.pcapng", 20, vmm=3)
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    lm = LiveMonitor(raw, setup=LAB, when=WHEN, window_s=3600)
    assert "rates" not in lm.step()                             # nothing yet
    for _ in _grow(src1, raw / _name(1), 7):
        s = lm.step()
    assert s["hits_total"] == 30 * 4
    # rotation: the new file appears while the old one still has a tail unread
    raw1 = open(src1, "rb").read()
    _capture(tmp_path / "a_more.pcapng", 35)
    with open(raw / _name(1), "wb") as f:
        f.write(open(tmp_path / "a_more.pcapng", "rb").read())
    for _ in _grow(src2, raw / _name(2), 3):
        s = lm.step()
    assert s["captures_finished"] == [_name(1)]
    assert s["capture"] == _name(2)
    assert s["hits_total"] == 35 * 4 + 20 * 4
    r = s["rates"]
    assert r["per_vmm"]["2"]["hits"] == 35 * 3 and r["per_vmm"]["3"]["hits"] == 20 * 3
    assert r["per_detector"]["P2_MID"]["hits"] == 55
    on_disk = json.load(open(raw.parent / "live.json"))
    assert on_disk["hits_total"] == s["hits_total"]


def test_live_window_drops_old_data(tmp_path):
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    lm = LiveMonitor(raw, setup=LAB, when=WHEN, window_s=0.010)   # 10 ms of data
    for _ in _grow(_capture(tmp_path / "a.pcapng", 60), raw / _name(1), 30):
        s = lm.step()
    assert s["hits_total"] == 60 * 4
    assert s["hits_in_window"] < 60 * 4 and s["live_time_s"] <= 0.012


def test_live_matches_reduce(tmp_path):
    src = _capture(tmp_path / "a.pcapng", 40)
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    lm = LiveMonitor(raw, setup=LAB, when=WHEN, window_s=3600)
    for _ in _grow(src, raw / _name(1), 9):
        s = lm.step()
    scal = pmon.process_capture(src, tmp_path / "store", setup=LAB, when=WHEN)
    assert s["rates"]["per_detector"] == scal["rates"]["per_detector"]
    assert s["live_time_s"] == pytest.approx(scal["volume"]["live_time_s"])


def test_live_plots(tmp_path):
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    _capture(raw / _name(1), 20)
    lm = LiveMonitor(raw, setup=LAB, when=WHEN, plot_dir=tmp_path / "png", plot_every_s=1)
    lm.step()
    assert sorted(os.listdir(tmp_path / "png")) == \
        ["channel_rates.png", "level_rates.png", "rate_map.png"]


def test_live_window_by_packet_time_not_arrival(tmp_path):
    # a second file whose packets are stamped BEFORE the first file's (a replay)
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    _capture(raw / _name(1), 30)
    lm = LiveMonitor(raw, setup=LAB, when=WHEN, window_s=3600)
    lm.step()
    _capture(raw / _name(2), 10)                      # same times as file 1's start
    s = lm.step()
    assert s["hits_total"] == 40 * 4 and "rates" in s
    assert s["live_time_s"] == pytest.approx(0.029)


def test_live_duration_zero_returns(tmp_path):
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    _capture(raw / _name(1), 5)
    s = LiveMonitor(raw, setup=LAB, when=WHEN).run(interval_s=0, duration_s=0,
                                                    log=lambda m: None)
    assert s["hits_total"] == 20
