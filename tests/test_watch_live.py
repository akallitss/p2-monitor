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
    assert r["per_chip"]["1:2"]["hits"] == 35 * 3 and r["per_chip"]["1:3"]["hits"] == 20 * 3
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


def test_live_follows_newest_subrun(tmp_path):
    runs = tmp_path / "runs"
    a = runs / "run_1" / "sub_00" / "raw_daq_data"
    a.mkdir(parents=True)
    _capture(a / _name(1), 10)
    lm = LiveMonitor(runs, setup=LAB, when=WHEN, window_s=3600, follow_newest=True)
    s = lm.step()
    assert (s["run"], s["subrun"], s["hits_total"]) == ("run_1", "sub_00", 40)
    time.sleep(0.02)
    b = runs / "run_2" / "sub_00" / "raw_daq_data"
    b.mkdir(parents=True)
    _capture(b / _name(1), 5, vmm=3)
    s = lm.step()
    assert (s["run"], s["captures_finished"], s["hits_total"]) == ("run_2", [_name(1)], 60)
    assert lm.out == runs / "live.json"


def test_live_survives_mapping_failure(tmp_path):
    raw = tmp_path / "raw_daq_data"
    raw.mkdir()
    _capture(raw / _name(1), 5)                       # 1970 timestamps, no `when`
    s = LiveMonitor(raw, setup=LAB).step()
    assert s["mapping_error"].startswith("LookupError") and "rates" in s
    assert s["rates"]["per_chip"]["1:2"]["hits"] == 15


# -- live from the network --------------------------------------------------------

def _free_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _send(port, payloads):
    import socket
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for p in payloads:
        tx.sendto(p, ("127.0.0.1", port))
    tx.close()


def test_electronics_view(tmp_path):
    """Every chip that fired, per channel, mapped or not; 'recent' = last seconds."""
    lm = LiveMonitor(udp="127.0.0.1:1", setup=LAB, when=WHEN, window_s=3600)
    lm.mapping = lm._mapping_for(0)
    old = np.zeros((512, 64), np.uint64)
    new = np.zeros((512, 64), np.uint64)
    old[1 * 32 + 1, 7] = 120          # FEC 1 VMM 1 (H0: not in the setup), early
    new[1 * 32 + 1, 9] = 40           # ... a cable plugged later lands on ch 9
    old[1 * 32 + 2, 0] = 60           # FEC 1 VMM 2: P2_IN in the setup
    lm.records = [[0.0, 10e9, old, 180, 0, 10, 0], [56e9, 60e9, new, 40, 0, 4, 0]]
    e = lm.summary()["electronics"]
    v1, v2 = e["chips"]["1:1"], e["chips"]["1:2"]
    assert v1["where"] is None and v2["where"].startswith("P2_IN c4 ")
    assert e["window_s"] == 60.0 and e["recent_s"] == 4.0
    assert v1["window_hz"][7] == 2.0 and v1["window_hz"][9] == pytest.approx(0.667)
    assert v1["recent_hz"][7] == 0.0 and v1["recent_hz"][9] == 10.0
    assert v1["hits"] == 160 and len(v1["window_hz"]) == 64


def test_live_udp_matches_file(tmp_path):
    words = [marker_word(2, 7), hit_word(2, 0, 1, 1, 1), hit_word(2, 1, 1, 1, 1),
             hit_word(2, 2, 1, 1, 1), hit_word(8, 3, 1, 1, 1)]
    payloads = [srs_payload(0, 1, words)] * 40
    port = _free_port()
    lm = LiveMonitor(udp=f"127.0.0.1:{port}", setup=LAB, when=WHEN, window_s=3600,
                     out=tmp_path / "live.json")
    lm.step()                                       # opens the socket
    _send(port, payloads[:25])
    time.sleep(0.05)
    lm.step()
    _send(port, payloads[25:])
    time.sleep(0.05)
    s = lm.step()
    assert s["source"] == f"udp://127.0.0.1:{port}" and s["capture"] is None
    assert s["hits_total"] == 40 * 4 and s["sampling"]["window_fraction"] == 1.0
    assert s["rates"]["per_detector"]["P2_IN"]["hits"] == 120
    assert s["rates"]["per_detector"]["P2_MID"]["hits"] == 40
    assert json.load(open(tmp_path / "live.json"))["hits_total"] == 160
    lm.udp.close()


def test_live_rates_scaled_for_lost_packets(tmp_path):
    lm = LiveMonitor(udp="127.0.0.1:1", mapping=False, window_s=3600)
    occ = np.zeros((512, 64), np.uint64)
    occ[34, 0] = 100                                 # FEC 1 VMM 2 ch 0
    lm.records = [[0.0, 10e9, occ, 100, 1000, 25, 75]]   # 25 decoded, 75 lost
    s = lm.summary()
    assert s["sampling"]["window_fraction"] == 0.25
    # 100 hits in 25 % of 10 s of data -> 40 Hz for the full stream
    assert s["rates"]["per_chip"]["1:2"]["rate_hz"] == pytest.approx(40.0)
    assert s["volume"]["hits_per_s"] == pytest.approx(40.0)


def test_live_needs_exactly_one_source():
    with pytest.raises(ValueError, match="either raw_dir or udp"):
        LiveMonitor()
    with pytest.raises(ValueError, match="either raw_dir or udp"):
        LiveMonitor("x", udp="127.0.0.1:6006")
