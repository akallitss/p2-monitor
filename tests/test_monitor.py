import json
import os

import numpy as np
import pytest
from p2_decode.testing import hit_word, marker_word, srs_payload, write_pcapng

import p2_monitor as pmon
from p2_monitor.plots import plot_store, si

LAB = "lab_saclay_2026"
WHEN = "2026-10-24T12:00"             # lab cabling c4-c6: VMM 2/3 = P2_IN c4


@pytest.fixture
def capture(tmp_path):
    # 11 packets 1 ms apart (tsresol 6 = us, packet i stamped i*1000 us would
    # need large ticks; use tsresol 3 = ms): live time = 10 ms.
    # Per packet: VMM 2 ch 0 twice, VMM 3 ch 5 once, VMM 0 ch 44 once (the
    # SPS trigger VMM; not cabled in the lab), one marker.
    words = [marker_word(2, 100), hit_word(2, 0, 100, 1, 1), hit_word(2, 0, 120, 2, 2),
             hit_word(3, 5, 300, 3, 3), hit_word(0, 44, 50, 4, 4)]
    return write_pcapng(tmp_path / "cap.pcapng",
                        [srs_payload(0, 1, words)] * 11, tsresol=3)


def test_process_capture_store(capture, tmp_path):
    store = tmp_path / "stores" / "cap"
    scal = pmon.process_capture(capture, store, setup=LAB, when=WHEN)
    assert not os.path.exists(str(store) + ".partial")
    files = set(os.listdir(store))
    assert {"counts.npz", "scalars.json", "meta.json", "setup_resolved.json",
            "vmm.npy"} <= files
    assert scal["n_hits"] == 44 and scal["setup"] == LAB

    v = scal["volume"]
    assert v["live_time_s"] == pytest.approx(0.010)
    assert v["hits_per_s"] == pytest.approx(4400)
    assert v["frame_loss"].startswith("not measurable")
    assert v["per_fec"]["1"]["hits"] == 44

    r = scal["rates"]
    assert r["per_chip"]["1:2"]["rate_hz"] == pytest.approx(2200)
    assert r["per_chip"]["1:2"]["live_channels"] == 1
    assert r["per_detector"]["P2_IN"]["rate_hz"] == pytest.approx(3300)
    assert r["per_detector"]["P2_IN"]["channels"] == 384
    assert r["per_detector"]["P2_MID"]["rate_hz"] == 0
    assert r["per_connector"]["P2_IN:c4"]["live_channels"] == 2
    assert r["per_half"]["P2_IN:c4:bot"]["hits"] == 22
    assert r["per_hybrid"]["1"]["hits"] == 33
    assert r["uncabled"] == {"channels": 1, "hits": 11, "chips": ["1:0"]}
    assert r["hottest_channel"] == {"fec": 1, "vmm": 2, "ch": 0, "rate_hz": 2200.0}
    assert r["per_detector"]["P2_IN"]["rate_hz_per_cm2"] > 0

    # the trend keys the existing dashboard reads are still there
    assert scal["unknown_vmm_hits"] == {"0": 11}
    assert scal["per_station"]["P2_IN"]["n_hits"] == 33
    assert scal["live_channels_per_vmm"] == {"0": 1, "2": 1, "3": 1}
    assert json.load(open(store / "scalars.json"))["rates"] == r


def test_channel_rates_keep_dead_channels(capture, tmp_path):
    pmon.process_capture(capture, tmp_path / "s", setup=LAB, when=WHEN)
    m = pmon.store_mapping(tmp_path / "s")
    df = pmon.channel_rates(pmon.load_counts(tmp_path / "s"), 0.010, m)
    assert len(df) == 18 * 64 + 1                     # every cabled channel + VMM 0 ch 44
    assert (df["hits"] == 0).sum() == 18 * 64 - 2
    row = df[(df.vmm == 3) & (df.ch == 5)].iloc[0]
    assert row["rate_hz"] == pytest.approx(1100)
    assert row["rate_hz_per_cm2"] == pytest.approx(1100 / (row["pad_area_mm2"] / 100))


def test_store_mapping_uses_recorded_time(capture, tmp_path):
    pmon.process_capture(capture, tmp_path / "s", setup=LAB, when="2026-10-27")
    m = pmon.store_mapping(tmp_path / "s")
    assert m.cabling.note.startswith("re-cabled to connectors c3-c5")


def test_no_mapping_mode(capture, tmp_path, monkeypatch):
    monkeypatch.delenv("P2_SETUP", raising=False)
    monkeypatch.setenv("P2_SETUP_FILE", str(tmp_path / "none"))
    with pytest.raises(LookupError, match="no setup chosen"):
        pmon.process_capture(capture, tmp_path / "a")
    scal = pmon.process_capture(capture, tmp_path / "b", mapping=False)
    assert scal["setup"] is None and "per_detector" not in scal["rates"]
    assert scal["rates"]["per_chip"]["1:0"]["hits"] == 11
    assert not os.path.exists(tmp_path / "b" / "setup_resolved.json")


def test_merge_and_trend(capture, tmp_path):
    for i in range(3):
        pmon.process_capture(capture, tmp_path / "run" / f"cap{i}", setup=LAB, when=WHEN)
    stores = pmon.find_stores(tmp_path / "run")
    assert len(stores) == 3
    merged = pmon.merged_counts(stores)                    # per chip: FEC 1, VMM 2
    assert int(merged["chip_ch"][1 * 32 + 2].sum()) == 66
    assert merged["chip_ch"].dtype == np.uint64
    assert int(pmon.merged_counts(stores, chip=False)["hits_per_vmm"][2]) == 66
    t = pmon.trend_table(tmp_path / "run")
    assert t["rate_hz:P2_IN"].tolist() == pytest.approx([3300] * 3)


def test_plots(capture, tmp_path):
    pmon.process_capture(capture, tmp_path / "s", setup=LAB, when=WHEN)
    paths = plot_store(tmp_path / "s", tmp_path / "png")
    assert [os.path.basename(p) for p in paths] == \
        ["rate_map.png", "channel_rates.png", "level_rates.png"]
    assert all(os.path.getsize(p) > 10_000 for p in paths)


def test_si():
    assert si(2720, "Hz") == "2.72 kHz" and si(0.5, "Hz") == "500 mHz"
    assert si(43.78, "Hz/cm²") == "43.8 Hz/cm²"


def test_plots_when_no_detector_fired(tmp_path):
    # only the uncabled VMM 0 fires: every detector panel is empty, not an error
    cap = write_pcapng(tmp_path / "c.pcapng",
                       [srs_payload(0, 1, [hit_word(0, 44, 50, 4, 4)])] * 3, tsresol=3)
    pmon.process_capture(cap, tmp_path / "s", setup=LAB, when=WHEN)
    assert len(plot_store(tmp_path / "s", tmp_path / "png")) == 3


def test_drop_columns_writes_little(capture, tmp_path):
    scal = pmon.process_capture(capture, tmp_path / "s", setup=LAB, when=WHEN, keep_store=False)
    assert not any(f.endswith(".npy") for f in os.listdir(tmp_path / "s"))
    full = pmon.process_capture(capture, tmp_path / "f", setup=LAB, when=WHEN)
    assert scal["rates"] == full["rates"]
    a = pmon.load_counts(tmp_path / "s", chip=False)
    b = pmon.load_counts(tmp_path / "f", chip=False)
    assert all(np.array_equal(a[k], b[k]) for k in a)
