"""Hit rates in two views of the same counts, and data volume.

    electronics  per (fec, vmm, ch), per chip (fec, vmm), per hybrid, per FEC
    physical     per pad, per connector half, per connector, per detector

A chip is (fec, vmm): with several FECs the same VMM id repeats, so chip keys
are "fec:vmm" (e.g. "12:2"). Rates are hits divided by the capture's live
time (first to last packet, from the pcapng timestamps), not by wall-clock
time of the processing. Channels that are cabled but never fired are kept at
rate 0: dead channels matter as much as hot ones. Areal rates use the pad
areas of the detector design.
"""

import numpy as np
import pandas as pd

from p2_mapping import load_pads

from .counts import NV

MAP_COLUMNS = ["hybrid", "detector", "connector_N", "half", "sector", "strip",
               "channel_id", "pad_cx", "pad_cy", "mapped"]


def _occupancy(counts):
    """(fec, vmm, ch, hits) arrays of the channels that fired.

    Takes chip counts (chip_ch, per FEC) or, for old stores, the per-VMM
    counts (ch), whose FEC is unknown (-1).
    """
    if "chip_ch" in counts:
        h = np.asarray(counts["chip_ch"], dtype=np.int64)
        chip, ch = np.nonzero(h)
        return chip // NV, chip % NV, ch, h[chip, ch]
    h = np.asarray(counts["ch"], dtype=np.int64)
    vmm, ch = np.nonzero(h)
    return np.full(len(vmm), -1), vmm, ch, h[vmm, ch]


def channel_rates(counts, live_time_s, mapping=None):
    """One row per channel that fired or is cabled: fec, vmm, ch, hits, rate_hz.

    With a mapping, adds its columns and pad_area_mm2 / rate_hz_per_cm2.
    Channels that fired but are not cabled get detector NaN. A cabled channel
    that never fired takes its row's FEC, or the data's FEC when the setup
    names none and the data has exactly one.
    """
    fec, vmm, ch, hits = _occupancy(counts)
    df = pd.DataFrame({"fec": fec.astype(np.int64), "vmm": vmm.astype(np.int64),
                       "ch": ch.astype(np.int64), "hits": hits.astype(np.int64)})
    if mapping is not None:
        t = mapping.table
        rows = mapping.rows(df["vmm"], df["ch"], df["fec"] if (df["fec"] >= 0).any() else None)
        fired = df.join(t[MAP_COLUMNS].reindex(rows).reset_index(drop=True))
        silent = t.drop(index=np.unique(rows[rows >= 0]))
        data_fecs = sorted(set(df["fec"]) - {-1})
        silent_fec = np.where(silent["fec"] >= 0, silent["fec"],
                              data_fecs[0] if len(data_fecs) == 1 else -1)
        silent = pd.DataFrame({"fec": silent_fec.astype(np.int64),
                               "vmm": silent["vmm"].to_numpy(np.int64),
                               "ch": silent["ch"].to_numpy(np.int64), "hits": 0,
                               **{c: silent[c].to_numpy() for c in MAP_COLUMNS}})
        df = pd.concat([fired, silent], ignore_index=True)
        df["mapped"] = df["mapped"].astype("boolean").fillna(False).astype(bool)
        area = load_pads(mapping.setup.design)["pad_area"]
        df["pad_area_mm2"] = df["channel_id"].map(area)
    df = df.sort_values(["fec", "vmm", "ch"]).reset_index(drop=True)
    df["rate_hz"] = df["hits"] / live_time_s if live_time_s else np.nan
    if "pad_area_mm2" in df:
        df["rate_hz_per_cm2"] = df["rate_hz"] / (df["pad_area_mm2"] / 100.0)
    return df


def _level(df, keys, areal):
    g = df.groupby(keys, dropna=True)
    out = pd.DataFrame({
        "hits": g["hits"].sum(),
        "rate_hz": g["rate_hz"].sum(),
        "channels": g["ch"].size(),
        "live_channels": g["hits"].apply(lambda s: int((s > 0).sum())),
        "max_channel_rate_hz": g["rate_hz"].max(),
    })
    if areal:
        a = g["pad_area_mm2"].sum()
        out["rate_hz_per_cm2"] = out["rate_hz"] / (a / 100.0)
    return out


def _records(frame, name):
    """{key: {column: value}} with JSON-friendly keys and plain numbers."""
    out = {}
    for key, row in frame.iterrows():
        k = ":".join(name(x) for x in key) if isinstance(key, tuple) else name(key)
        out[k] = {c: (int(v) if c in ("hits", "channels", "live_channels")
                      else round(float(v), 6)) for c, v in row.items()}
    return out


def _number(v):
    return str(int(v))


def _place(v):
    """Key part of the physical view: detector and half as is, connector as cN."""
    return v if isinstance(v, str) else f"c{int(v)}"


def summarize(rates):
    """Rates per level, as the dict that goes into scalars.json["rates"].

    Keys: per_fec "12", per_chip "12:2" (fec:vmm), per_hybrid "1",
    per_detector "P2_IN", per_connector "P2_IN:c4", per_half "P2_IN:c4:bot".
    """
    out = {"per_fec": _records(_level(rates, "fec", False), _number),
           "per_chip": _records(_level(rates, ["fec", "vmm"], False), _number)}
    if "detector" in rates:
        mapped = rates[rates["mapped"]]
        out["per_hybrid"] = _records(_level(mapped, "hybrid", False), _number)
        out["per_detector"] = _records(_level(mapped, "detector", True), _place)
        out["per_connector"] = _records(
            _level(mapped, ["detector", "connector_N"], True), _place)
        out["per_half"] = _records(
            _level(mapped, ["detector", "connector_N", "half"], True), _place)
        unc = rates[~rates["mapped"]]
        out["uncabled"] = {"channels": int(len(unc)), "hits": int(unc["hits"].sum()),
                           "chips": sorted(f"{int(f)}:{int(v)}" for f, v in
                                           unc[["fec", "vmm"]].drop_duplicates()
                                           .itertuples(index=False))}
    if len(rates) and rates["hits"].sum():
        hot = rates.loc[rates["rate_hz"].idxmax()]
        out["hottest_channel"] = {"fec": int(hot["fec"]), "vmm": int(hot["vmm"]),
                                  "ch": int(hot["ch"]),
                                  "rate_hz": round(float(hot["rate_hz"]), 6)}
    return out


def volume(meta, file_bytes=None):
    """Data volume of one capture from the decoder's meta["capture"].

    mb_per_min_file uses the capture file size, which is what rate_guard
    limits; mb_per_min_udp counts only the FEC payloads.
    """
    c = meta.get("capture") or {}
    t = c.get("live_time_s")
    if not t:
        return {"live_time_s": t}

    def per_s(x):
        return round(x / t, 6)

    out = {
        "live_time_s": round(t, 6),
        "udp_payload_bytes": c.get("udp_payload_bytes"),
        "mb_per_min_udp": round(c.get("udp_payload_bytes", 0) / 1e6 / t * 60, 6),
        "hits_per_s": per_s(meta.get("n_hits", 0)),
        "packets_per_s": per_s(meta.get("n_vm3_packets", 0)),
        "per_fec": {f: {**d, "packets_per_s": per_s(d["packets"]),
                        "hits_per_s": per_s(d.get("hits", 0)),
                        "mb_per_min_udp": round(d["udp_payload_bytes"] / 1e6 / t * 60, 6)}
                    for f, d in (c.get("per_fec") or {}).items()},
        # the ALINX firmware's frame counter is constant, so lost frames are
        # not countable; say so instead of reporting zero
        "frame_loss": (None if c.get("frame_counter_changes")
                       else "not measurable: SRS frame counter does not change"),
    }
    if file_bytes is not None:
        out["file_bytes"] = int(file_bytes)
        out["mb_per_min_file"] = round(file_bytes / 1e6 / t * 60, 6)
    return out
