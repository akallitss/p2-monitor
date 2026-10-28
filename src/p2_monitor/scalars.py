"""The per-capture trend record (scalars.json).

build_scalars is moved unchanged from vmm_qa/vmm_reduce.py, so the existing
trend dashboard keeps reading the same keys. add_detector_scalars replaces
add_station_scalars, taking the detector of each VMM from p2-mapping instead
of vmm_stations.STATION_VMMS.
"""

import numpy as np

from .counts import NV


def _pct_from_counts(hist, q):
    """Percentile of a 1D histogram over bin index, without the raw data."""
    tot = hist.sum()
    if tot == 0:
        return None
    return float(np.searchsorted(np.cumsum(hist), q * tot))


def build_scalars(cols, counts, meta, station_map=None) -> dict:
    """The per-file trend record.

    Deliberately picks the quantities that have already caught real problems on
    this setup: occupancy (dead/hot hybrids), live channel count (P2_MID's ~39
    dead channels), ADC percentiles (P2_IN's compressed spectrum, i.e. the gain
    story), invalid offsets and unknown VMM ids (firmware corruption), and the
    SRS marker cadence (the 1.6384 ms aliasing).
    """
    vmm = np.asarray(cols["vmm"])
    hits_per_vmm = counts["hits_per_vmm"]
    ch_counts = counts["ch"]

    live = (ch_counts > 0).sum(axis=1)          # channels that ever fired
    adc_p50 = [_pct_from_counts(counts["adc"][v], 0.50) for v in range(NV)]
    adc_p90 = [_pct_from_counts(counts["adc"][v], 0.90) for v in range(NV)]

    # frame counter spans the capture; it is a proxy for time, not seconds
    t0, t1 = (int(counts["time_range"][0]), int(counts["time_range"][1]))

    scal = {
        "n_packets": int(meta.get("n_packets", 0)),
        "n_vm3_packets": int(meta.get("n_vm3_packets", 0)),
        "n_hits": int(meta.get("n_hits", len(vmm))),
        "n_invalid_offset": int(meta.get("n_invalid_offset", 0)),
        "offset_constant": bool(meta.get("offset_constant", False)),
        "frame_first": t0,
        "frame_last": t1,
        "hits_per_vmm": {str(v): int(hits_per_vmm[v]) for v in range(NV)
                         if hits_per_vmm[v]},
        "live_channels_per_vmm": {str(v): int(live[v]) for v in range(NV)
                                  if hits_per_vmm[v]},
        "adc_p50_per_vmm": {str(v): adc_p50[v] for v in range(NV)
                            if hits_per_vmm[v]},
        "adc_p90_per_vmm": {str(v): adc_p90[v] for v in range(NV)
                            if hits_per_vmm[v]},
        "ot_fraction": (float(counts["ot"][:, 1].sum()) /
                        max(int(counts["ot"].sum()), 1)),
    }

    # VMM ids outside the cabling are firmware corruption, not physics
    if station_map:
        known = set(station_map)
        bad = {str(v): int(hits_per_vmm[v]) for v in range(NV)
               if hits_per_vmm[v] and v not in known}
        scal["unknown_vmm_hits"] = bad
    return scal


def add_detector_scalars(scal, cols, mapping):
    """per_station: hits and ADC percentiles per detector (key kept for the dashboard)."""
    vmm = np.asarray(cols["vmm"])
    adc = np.asarray(cols["adc"])
    t = mapping.table
    per = {}
    for det in mapping.setup.detectors:
        sel = np.isin(vmm, t.loc[t["detector"] == det, "vmm"].unique())
        n = int(sel.sum())
        if not n:
            continue
        a = adc[sel]
        per[det] = {"n_hits": n,
                    "adc_p50": float(np.percentile(a, 50)),
                    "adc_p90": float(np.percentile(a, 90))}
    if per:
        scal["per_station"] = per
    return scal


def percentile_from_hist(hist, q):
    """np.percentile(values, 100*q) (linear method) from an integer-value histogram.

    Exact, not binned: with integer data the k-th sorted value is the first
    bin whose cumulative count exceeds k.
    """
    hist = np.asarray(hist, dtype=np.int64)
    n = int(hist.sum())
    if n == 0:
        return None
    cum = np.cumsum(hist)
    h = (n - 1) * q
    lo, hi = int(np.floor(h)), int(np.ceil(h))
    x_lo = int(np.searchsorted(cum, lo, side="right"))
    x_hi = int(np.searchsorted(cum, hi, side="right"))
    return float(x_lo + (h - lo) * (x_hi - x_lo))


def detector_chips(mapping):
    """{detector: sorted chip ids}; a row without a FEC covers that VMM on every FEC."""
    from .counts import NFEC, NV
    out = {}
    t = mapping.table[["detector", "fec", "vmm"]].drop_duplicates()
    for det, fec, vmm in t.itertuples(index=False):
        chips = [f * NV + int(vmm) for f in range(NFEC)] if fec < 0 else [int(fec) * NV + int(vmm)]
        out.setdefault(det, set()).update(chips)
    return {d: sorted(c) for d, c in out.items()}


def add_detector_scalars_from_counts(scal, chip_counts, mapping):
    """per_station from the chip ADC histograms: same numbers as
    add_detector_scalars, without holding the hits."""
    per = {}
    for det, chips in detector_chips(mapping).items():
        if det not in mapping.setup.detectors:
            continue
        h = np.asarray(chip_counts["chip_adc"])[chips].sum(axis=0)
        n = int(h.sum())
        if not n:
            continue
        per[det] = {"n_hits": n, "adc_p50": percentile_from_hist(h, 0.50),
                    "adc_p90": percentile_from_hist(h, 0.90)}
    if per:
        scal["per_station"] = {d: per[d] for d in mapping.setup.detectors if d in per}
    return scal
