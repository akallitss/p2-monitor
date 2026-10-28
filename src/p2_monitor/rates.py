"""Hit rates in two views of the same counts, and data volume.

    electronics  per (vmm, ch), per VMM, per hybrid, per FEC
    physical     per pad, per connector half, per connector, per detector

Rates are hits divided by the capture's live time (first to last packet, from
the pcapng timestamps), not by wall-clock time of the processing. Channels
that are cabled but never fired are kept with rate 0: dead channels matter as
much as hot ones. Areal rates use the pad areas of the detector design.
"""

import numpy as np
import pandas as pd

from p2_mapping import load_pads

MAP_COLUMNS = ["hybrid", "detector", "connector_N", "half", "sector", "strip",
               "channel_id", "pad_cx", "pad_cy", "mapped"]


def channel_rates(counts, live_time_s, mapping=None):
    """One row per (vmm, ch) that fired or is cabled: hits and rate_hz.

    With a mapping, adds its columns and pad_area_mm2 / rate_hz_per_cm2;
    channels that fired but are not cabled get detector NaN.
    """
    h = np.asarray(counts["ch"], dtype=np.int64)          # (32, 64)
    vmm, ch = np.nonzero(h)
    df = pd.DataFrame({"vmm": vmm, "ch": ch, "hits": h[vmm, ch]})
    if mapping is not None:
        t = mapping.table[["vmm", "ch"] + MAP_COLUMNS]
        df = t.merge(df, on=["vmm", "ch"], how="outer")
        df["hits"] = df["hits"].fillna(0).astype(np.int64)
        area = load_pads(mapping.setup.design)["pad_area"]
        df["pad_area_mm2"] = df["channel_id"].map(area)
    df = df.sort_values(["vmm", "ch"]).reset_index(drop=True)
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

    Keys: per_vmm "2", per_hybrid "1", per_detector "P2_IN",
    per_connector "P2_IN:c4", per_half "P2_IN:c4:bot".
    """
    out = {"per_vmm": _records(_level(rates, "vmm", False), _number)}
    if "detector" in rates:
        mapped = rates[rates["mapped"].fillna(False).astype(bool)]
        out["per_hybrid"] = _records(_level(mapped, "hybrid", False), _number)
        out["per_detector"] = _records(_level(mapped, "detector", True), _place)
        out["per_connector"] = _records(
            _level(mapped, ["detector", "connector_N"], True), _place)
        out["per_half"] = _records(
            _level(mapped, ["detector", "connector_N", "half"], True), _place)
        unc = rates[rates["detector"].isna()]
        out["uncabled"] = {"channels": int(len(unc)), "hits": int(unc["hits"].sum()),
                           "vmms": sorted(int(v) for v in unc["vmm"].unique())}
    hot = rates.loc[rates["rate_hz"].idxmax()] if len(rates) and rates["hits"].sum() else None
    if hot is not None:
        out["hottest_channel"] = {"vmm": int(hot["vmm"]), "ch": int(hot["ch"]),
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
