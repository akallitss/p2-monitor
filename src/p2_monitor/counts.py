"""Per-VMM histograms of one capture, additive across captures (counts.npz).

Moved unchanged from DAQ_Control_VMM_Beam/vmm_qa/vmm_reduce.py, so stores
written by p2-monitor stay readable by render_from_store.py and the dashboard.

Rendering plots per capture costs more than the capture lasts, and for quiet
VMMs a per-file 1024-bin ADC spectrum is under one entry per bin. Because
counts are additive, merging N files is a sum of arrays: that is how a quiet
VMM gets a real spectrum (382 hits/file becomes ~8000 over 20 files).
"""

import numpy as np

NV = 32                     # VMM id space (5 bits)
ADC_BINS = 1024
CH_BINS = 64
BCID_BINS = 4096
TDC_BINS = 256
OFFSET_BINS = 32            # signed offset -16..+15, stored shifted by +16
ADC_CH_ADC_BINS = 128       # adc >> 3 for the 2D adc-vs-channel map
TIME_BINS = 200             # frame-counter profile, for the rate-vs-time plot


def _per_vmm(vmm, vals, nbins):
    """Per-VMM histogram via one bincount on vmm*nbins + value."""
    idx = vmm.astype(np.int64) * nbins + vals.astype(np.int64)
    return np.bincount(idx, minlength=NV * nbins).reshape(NV, nbins).astype(np.uint32)


def build_counts(cols) -> dict:
    """Per-VMM histograms of every quantity the QA plots use.

    `cols` is a dict of arrays (from p2_decode.load_hits or decode). Only the
    raw columns are touched, so this works straight off a memory-mapped store.
    """
    vmm = np.asarray(cols["vmm"])
    ch = np.asarray(cols["ch"])
    adc = np.asarray(cols["adc"])
    tdc = np.asarray(cols["tdc"])
    bcid = np.asarray(cols["bcid"])
    offset = np.asarray(cols["offset"])
    ot = np.asarray(cols["over_threshold"]).astype(np.uint8)
    tcol = np.asarray(cols["time"])

    out = {
        "adc": _per_vmm(vmm, adc, ADC_BINS),
        "ch": _per_vmm(vmm, ch, CH_BINS),
        "bcid": _per_vmm(vmm, bcid, BCID_BINS),
        "tdc": _per_vmm(vmm, tdc, TDC_BINS),
        "offset": _per_vmm(vmm, offset.astype(np.int16) + 16, OFFSET_BINS),
        "ot": _per_vmm(vmm, ot, 2),
        "hits_per_vmm": np.bincount(vmm, minlength=NV).astype(np.uint64),
    }

    # ADC split by the over-threshold flag (the adc_ot plot)
    for name, sel in (("adc_ot0", ot == 0), ("adc_ot1", ot == 1)):
        out[name] = (_per_vmm(vmm[sel], adc[sel], ADC_BINS) if sel.any()
                     else np.zeros((NV, ADC_BINS), np.uint32))

    # 2D ADC vs channel, ADC coarsened 8x to keep the array small
    idx2 = ((vmm.astype(np.int64) * CH_BINS + ch) * ADC_CH_ADC_BINS
            + (adc.astype(np.int64) >> 3))
    out["adc_vs_ch"] = np.bincount(
        idx2, minlength=NV * CH_BINS * ADC_CH_ADC_BINS
    ).reshape(NV, CH_BINS, ADC_CH_ADC_BINS).astype(np.uint32)

    # rate vs frame counter. Edges are stored so files with different spans
    # can still be summed by the dashboard after re-binning.
    if len(tcol):
        t0, t1 = int(tcol.min()), int(tcol.max())
        span = max(t1 - t0, 1)
        tb = np.clip(((tcol.astype(np.int64) - t0) * TIME_BINS) // span,
                     0, TIME_BINS - 1)
        out["time_profile"] = _per_vmm(vmm, tb.astype(np.int64), TIME_BINS)
        out["time_range"] = np.array([t0, t1], dtype=np.int64)
    else:
        out["time_profile"] = np.zeros((NV, TIME_BINS), np.uint32)
        out["time_range"] = np.array([0, 0], dtype=np.int64)
    return out


def merge_counts(list_of_counts) -> dict:
    """Sum counts across files. The whole point of storing counts."""
    it = iter(list_of_counts)
    total = {k: np.array(v) for k, v in next(it).items()}
    for c in it:
        for k, v in c.items():
            if k == "time_range":
                total[k] = np.array([min(total[k][0], v[0]),
                                     max(total[k][1], v[1])], dtype=np.int64)
            elif k in total and total[k].shape == np.shape(v):
                total[k] = total[k] + v
    return total
