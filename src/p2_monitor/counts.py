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
    """Sum counts across files. The whole point of storing counts.

    Chip histograms are summed as uint64 (per capture some are uint32); the
    per-second rates are merged by second; time ranges take min and max.
    """
    def wide(v):
        v = np.array(v)
        return v.astype(np.uint64) if v.dtype == np.uint32 else v

    it = iter(list_of_counts)
    first = next(it)
    total = {k: (wide(v) if k.startswith("chip_") else np.array(v)) for k, v in first.items()}
    by_sec = {}

    def take_seconds(c):
        if "rate_s_sec" in c:
            for s, row in zip(np.asarray(c["rate_s_sec"]), np.asarray(c["rate_s"])):
                s = int(s)
                by_sec[s] = by_sec[s] + row.astype(np.uint64) if s in by_sec \
                    else row.astype(np.uint64)
    take_seconds(first)
    for c in it:
        take_seconds(c)
        for k, v in c.items():
            if k in ("rate_s_sec", "rate_s"):
                continue
            if k == "time_range":
                total[k] = np.array([min(total[k][0], v[0]),
                                     max(total[k][1], v[1])], dtype=np.int64)
            elif k == "abs_time_range":
                total[k] = np.array([np.nanmin([total[k][0], v[0]]),
                                     np.nanmax([total[k][1], v[1]])])
            elif k in total and total[k].shape == np.shape(v):
                total[k] = total[k] + (wide(v) if k.startswith("chip_") else v)
    if "rate_s_sec" in total:
        secs = np.array(sorted(by_sec), dtype=np.int64)
        total["rate_s_sec"] = secs
        total["rate_s"] = (np.stack([by_sec[s] for s in secs]) if len(secs)
                           else np.zeros((0, NCHIP), np.uint64))
    return total


# ---------------------------------------------------------------------------
# per chip, and built chunk by chunk
# ---------------------------------------------------------------------------
#
# A VMM id is 5 bits, so with several FECs (the full experiment: 180 hybrids,
# 360 VMMs) the same id repeats. Chip-level histograms are indexed by
# chip = fec * 32 + vmm (FEC id is 4 bits: 512 chips). They are uint64, so
# summing captures over a long run cannot overflow. They live in
# counts_chip.npz; counts.npz keeps the per-VMM layout above for the existing
# QA readers (summed over FECs, exact for single-FEC data).

NFEC = 16
NCHIP = NFEC * NV
CHUNK_ROWS = 5_000_000      # rows read per slice when re-reading a stored column


def chip_index(fec, vmm):
    return np.asarray(fec, dtype=np.int64) * NV + np.asarray(vmm, dtype=np.int64)


def chip_occupancy(cols) -> np.ndarray:
    """(NCHIP, 64) uint64 hits per (chip, channel): all a rate needs."""
    idx = chip_index(cols["fec"], cols["vmm"]) * CH_BINS + np.asarray(cols["ch"], np.int64)
    return np.bincount(idx, minlength=NCHIP * CH_BINS).reshape(NCHIP, CH_BINS).astype(np.uint64)


# Fine hit time within its frame, timestamp_ns = (offset*4096 + bcid)*22.5 ns
# + TDC part: a fixed range, so the histogram is additive. offset -1 (the
# earliest valid) to 15 covers [-92160, 1474560) ns.
TS_BINS = 1024
TS_LO, TS_HI = -4096 * 22.5, 16 * 4096 * 22.5


def _chip_hist(chip, vals, nbins, dtype=np.uint32):
    return (np.bincount(chip * nbins + vals, minlength=NCHIP * nbins)
            .reshape(NCHIP, nbins).astype(dtype))


def _timestamp_ns(cols):
    """timestamp_ns of each hit, the p2_decode.derive formula without calibration."""
    from p2_decode.schema import CLOCK_PERIOD_NS, TAC_SLOPE_NS, TDC_RANGE
    coarse = (np.asarray(cols["offset"], np.float64) * 4096
              + np.asarray(cols["bcid"], np.float64)) * CLOCK_PERIOD_NS
    return coarse + CLOCK_PERIOD_NS - np.asarray(cols["tdc"], np.float64) * TAC_SLOPE_NS / TDC_RANGE


def build_chip_counts(cols) -> dict:
    """Chip-level histograms of one chunk: everything the QA plots need.

    chip_ch, chip_adc, chip_ot are uint64; the larger ones (ADC vs channel,
    BCID, ...) are uint32 per capture and become uint64 when merged.
    """
    chip = chip_index(cols["fec"], cols["vmm"])
    adc = np.asarray(cols["adc"], np.int64)
    ot = np.asarray(cols["over_threshold"]).astype(np.int64)
    ch = np.asarray(cols["ch"], np.int64)
    ts = _timestamp_ns(cols)
    ts_bin = np.clip(((ts - TS_LO) * TS_BINS / (TS_HI - TS_LO)).astype(np.int64), 0, TS_BINS - 1)
    on = ot == 1
    return {
        "chip_ch": chip_occupancy(cols),
        "chip_adc": _chip_hist(chip, adc, ADC_BINS, np.uint64),
        "chip_ot": _chip_hist(chip, ot, 2, np.uint64),
        "chip_adc_ot1": _chip_hist(chip[on], adc[on], ADC_BINS),
        "chip_adc_vs_ch": (np.bincount((chip * CH_BINS + ch) * ADC_CH_ADC_BINS + (adc >> 3),
                                       minlength=NCHIP * CH_BINS * ADC_CH_ADC_BINS)
                           .reshape(NCHIP, CH_BINS, ADC_CH_ADC_BINS).astype(np.uint32)),
        "chip_bcid": _chip_hist(chip, np.asarray(cols["bcid"], np.int64), BCID_BINS),
        "chip_tdc": _chip_hist(chip, np.asarray(cols["tdc"], np.int64), TDC_BINS),
        "chip_offset": _chip_hist(chip, np.asarray(cols["offset"], np.int64) + 16, OFFSET_BINS),
        "chip_ts": _chip_hist(chip, ts_bin, TS_BINS),
    }


def per_second(cols):
    """{second: (NCHIP,) hits} by the wall-clock second their packet arrived.

    Uses p2_decode's pkt_time_ns (capture/receive time, p2-decode >= 0.8.1):
    the FEC's SRS clock is not a usable wall clock (on the lab data it jumps).
    Without packet times (e.g. a column store read back) it falls back to the
    SRS marker + fine time. Sparse (only the seconds that occur) and additive
    across chunks. Also returns the (min, max) of the hits' SRS absolute time,
    which events.json reports as t_first_ns / t_last_ns as before.
    """
    abs_ns = np.asarray(cols["srs_timestamp"], np.float64) * 25.0 + _timestamp_ns(cols)
    t = np.asarray(cols["pkt_time_ns"], np.float64) if "pkt_time_ns" in cols else abs_ns
    ok = np.isfinite(t)
    sec = np.floor(t[ok] / 1e9).astype(np.int64)
    chip = chip_index(cols["fec"], cols["vmm"])[ok]
    key, n = np.unique(sec * NCHIP + chip, return_counts=True)
    out = {}
    for s in np.unique(key // NCHIP):
        sel = (key // NCHIP) == s
        row = np.zeros(NCHIP, np.uint32)
        row[key[sel] % NCHIP] = n[sel]
        out[int(s)] = row
    return out, (float(abs_ns.min()), float(abs_ns.max())) if len(abs_ns) else None


def seconds_arrays(by_second):
    """{second: row} -> (rate_s_sec (n,), rate_s (n, NCHIP)) sorted by second."""
    secs = np.array(sorted(by_second), dtype=np.int64)
    rows = (np.stack([by_second[s] for s in secs]) if len(secs)
            else np.zeros((0, NCHIP), np.uint32))
    return secs, rows


def _add(total, part):
    for k, v in part.items():
        total[k] = v if k not in total else total[k] + v


class CountsAccumulator:
    """Histograms of a capture, filled one decode chunk at a time.

    Memory is the histograms plus one chunk, whatever the capture size: pass
    `add` as p2_decode.decode_to's on_chunk. Everything is additive except the
    frame-counter profile, whose bins span the capture's frame range; that one
    is filled afterwards from the stored columns (`finish`), also in slices.
    The result equals build_counts / build_chip_counts on the whole capture.
    """

    def __init__(self):
        self.legacy = {}
        self.chip = {}
        self.t_min = None
        self.t_max = None
        self.by_second = {}
        self.abs_range = None

    def add(self, cols):
        part = build_counts(cols)
        part.pop("time_profile")
        tr = part.pop("time_range")
        if len(cols["vmm"]):
            self.t_min = int(tr[0]) if self.t_min is None else min(self.t_min, int(tr[0]))
            self.t_max = int(tr[1]) if self.t_max is None else max(self.t_max, int(tr[1]))
        _add(self.legacy, part)
        _add(self.chip, build_chip_counts(cols))
        rows, rng = per_second(cols)
        for s, r in rows.items():
            self.by_second[s] = self.by_second[s] + r if s in self.by_second else r
        if rng is not None:
            self.abs_range = rng if self.abs_range is None else \
                (min(self.abs_range[0], rng[0]), max(self.abs_range[1], rng[1]))

    def finish(self, store_dir):
        """(legacy counts, chip counts); reads time.npy and vmm.npy in slices."""
        import os
        if not self.legacy:                  # no hits: shapes from an empty chunk
            empty = {"fec": np.zeros(0, np.uint8), "vmm": np.zeros(0, np.uint8),
                     "ch": np.zeros(0, np.uint8), "adc": np.zeros(0, np.uint16),
                     "tdc": np.zeros(0, np.uint8), "bcid": np.zeros(0, np.uint16),
                     "offset": np.zeros(0, np.int8), "over_threshold": np.zeros(0, bool),
                     "time": np.zeros(0, np.uint32), "srs_timestamp": np.zeros(0, np.uint64)}
            chip = build_chip_counts(empty)
            chip["rate_s_sec"], chip["rate_s"] = seconds_arrays({})
            chip["abs_time_range"] = np.array([np.nan, np.nan])
            return build_counts(empty), chip
        legacy = dict(self.legacy)
        t0, t1 = self.t_min, self.t_max
        span = max(t1 - t0, 1)
        prof = np.zeros((NV, TIME_BINS), np.uint32)
        if t0 == t1:
            # one frame-counter value (the ALINX firmware never changes it):
            # every hit falls in bin 0, exactly as the general loop would put it
            prof[:, 0] = legacy["hits_per_vmm"].astype(np.uint32)
            tcol = ()
        else:
            tcol = np.load(os.path.join(store_dir, "time.npy"), mmap_mode="r")
            vcol = np.load(os.path.join(store_dir, "vmm.npy"), mmap_mode="r")
        for lo in range(0, len(tcol), CHUNK_ROWS):
            t = np.asarray(tcol[lo:lo + CHUNK_ROWS], dtype=np.int64)
            v = np.asarray(vcol[lo:lo + CHUNK_ROWS])
            tb = np.clip(((t - t0) * TIME_BINS) // span, 0, TIME_BINS - 1)
            prof += _per_vmm(v, tb, TIME_BINS)
        legacy["time_profile"] = prof
        legacy["time_range"] = np.array([t0, t1], dtype=np.int64)
        # dtypes as build_counts gives them for one whole capture
        for k, v in legacy.items():
            if k not in ("hits_per_vmm", "time_range"):
                legacy[k] = v.astype(np.uint32)
        chip = dict(self.chip)
        chip["rate_s_sec"], chip["rate_s"] = seconds_arrays(self.by_second)
        chip["abs_time_range"] = np.array(self.abs_range if self.abs_range else [np.nan] * 2)
        return legacy, chip
