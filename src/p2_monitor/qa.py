"""Per-capture QA plots drawn from the stored histograms, never from the hits.

    p2-monitor qa <store> --out-dir DIR [--events-json]

Draws the plots the old per-hit QA script (vmm_pcapng_qa.py) drew, with the
same file names (<capture>_adc.png, ..._bcid.png, ...) and the same
events.json, but from counts_chip.npz. So it costs about the same for a
10 MB or a 1 GB capture, every capture can have its plots, and the hit
columns need not be kept for them. Several stores give one merged set (a
whole sub-run).

Everything is per chip (FEC:VMM), so it stays right with many FECs: up to 40
chips as small multiples, beyond that as one heatmap row per chip.
"""

import datetime
import json
import os

import numpy as np

from .counts import (ADC_BINS, ADC_CH_ADC_BINS, BCID_BINS, NV, OFFSET_BINS,
                     TDC_BINS, TS_BINS, TS_HI, TS_LO)
from .plots import BAR, GRID, INK, INK_2, _mpl
from .store import load_counts, load_scalars
from .watch import parse_capture_name

MAX_PANELS = 40          # small multiples up to here, heatmap rows beyond
NCOL = 5


def _label(chip, single_fec):
    f, v = divmod(int(chip), NV)
    return f"VMM {v}" if single_fec else f"FEC {f} VMM {v}"


def _rebin(h, factor):
    n = h.shape[-1] // factor * factor
    return h[..., :n].reshape(*h.shape[:-1], n // factor, factor).sum(-1)


def _upper(h, q=0.999):
    """Bin below which q of the summed histogram lies (x-axis limit)."""
    tot = np.asarray(h, np.float64).sum(axis=0)
    if tot.sum() == 0:
        return len(tot)
    return int(np.searchsorted(np.cumsum(tot), q * tot.sum())) + 1


class _Figs:
    def __init__(self, out_dir, base, subtitle):
        self.plt, self.cmap, self.LogNorm = _mpl()
        self.out_dir, self.base, self.subtitle = out_dir, base, subtitle
        self.paths = []

    def save(self, fig, name, title):
        fig.suptitle(f"{title}\n{self.subtitle}", x=0.01, ha="left", fontsize=10,
                     fontweight="bold", color=INK)
        p = os.path.join(self.out_dir, f"{self.base}_{name}.png")
        fig.savefig(p, dpi=110)
        self.plt.close(fig)
        self.paths.append(p)

    def grid(self, n):
        ncol = min(n, NCOL)
        nrow = -(-n // ncol)
        fig, axes = self.plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.1 * nrow + 0.9),
                                      squeeze=False, constrained_layout=True)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        shown = list(axes.flat[:n])
        for ax in shown:
            ax.grid(color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)
        return fig, shown              # a list: the callers iterate it more than once

    def per_chip_1d(self, name, title, hists, chips, labels, x, xlabel, log=False):
        """hists: (n_chips, n_bins) over bin edges x (len n_bins + 1)."""
        if len(chips) <= MAX_PANELS:
            fig, axes = self.grid(len(chips))
            for ax, h, lab in zip(axes, hists, labels):
                ax.stairs(h, x, color=BAR, linewidth=1.0)
                ax.set_title(f"{lab}  ({int(h.sum()):,})", fontsize=8, loc="left", color=INK)
                if log and h.any():
                    ax.set_yscale("log")
                ax.set_xlim(x[0], x[-1])
            for ax in list(axes)[-NCOL:]:
                ax.set_xlabel(xlabel, fontsize=7)
        else:
            fig, ax = self.plt.subplots(figsize=(10, 0.06 * len(chips) + 2),
                                        constrained_layout=True)
            data = np.where(hists > 0, hists, np.nan)
            im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=self.cmap,
                           norm=self.LogNorm(vmin=1, vmax=max(2, np.nanmax(data)
                                                             if np.isfinite(data).any() else 2)),
                           extent=(x[0], x[-1], len(chips) - 0.5, -0.5))
            step = max(1, len(chips) // 40)
            ax.set_yticks(range(0, len(chips), step), labels[::step], fontsize=6)
            ax.set_xlabel(xlabel)
            fig.colorbar(im, ax=ax, label="hits (log)")
        self.save(fig, name, title)


def _load(stores):
    from .counts import merge_counts
    chip = merge_counts([load_counts(s) for s in stores])
    legacy = merge_counts([load_counts(s, chip=False) for s in stores])
    scal = [load_scalars(s) for s in stores]
    meta = []
    for s in stores:
        try:
            with open(os.path.join(s, "meta.json")) as f:
                meta.append(json.load(f))
        except OSError:
            meta.append({})
    return chip, legacy, scal, meta


def render(stores, out_dir, events_json=False, base=None):
    """Draw the QA set for one store (or several merged) into out_dir.

    Returns the list of PNG paths (and writes events.json if asked).
    """
    stores = [stores] if isinstance(stores, (str, os.PathLike)) else list(stores)
    stores = [os.fspath(s) for s in stores]
    os.makedirs(out_dir, exist_ok=True)
    chip, legacy, scal, meta = _load(stores)
    base = base or os.path.basename(stores[0].rstrip("/"))
    if "chip_bcid" not in chip:
        raise ValueError(f"{stores[0]}: store made before p2-monitor 0.6; re-reduce it "
                         "to draw its QA plots from histograms")

    occ = np.asarray(chip["chip_ch"]).astype(np.int64)
    chips = np.flatnonzero(occ.sum(axis=1))
    fecs = sorted({int(c) // NV for c in chips})
    single = len(fecs) <= 1
    labels = [_label(c, single) for c in chips]
    n_hits = int(occ.sum())
    live = sum((s.get("volume") or {}).get("live_time_s") or 0 for s in scal)
    what = base if len(stores) == 1 else f"{len(stores)} captures"
    sub = (f"{what} | {n_hits:,} hits over {live:.1f} s live"
           + (f" | FEC {', '.join(map(str, fecs))}" if fecs else ""))
    F = _Figs(out_dir, base, sub)

    if len(chips):
        adc = np.asarray(chip["chip_adc"])[chips].astype(np.float64)
        top = min(ADC_BINS, max(64, _upper(adc) + 16))
        x = np.arange(top + 1)
        F.per_chip_1d("adc", "ADC per chip", adc[:, :top], chips, labels, x, "ADC", log=True)

        ot1 = np.asarray(chip["chip_adc_ot1"])[chips].astype(np.float64)[:, :top]
        ot0 = adc[:, :top] - ot1
        if len(chips) <= MAX_PANELS:
            fig, axes = F.grid(len(chips))
            for ax, a0, a1, lab in zip(axes, ot0, ot1, labels):
                ax.stairs(a0, x, color=INK_2, linewidth=0.9, label="OT 0")
                ax.stairs(a1, x, color=BAR, linewidth=0.9, label="OT 1")
                ax.set_title(lab, fontsize=8, loc="left", color=INK)
                if (a0 + a1).any():
                    ax.set_yscale("log")
            list(axes)[0].legend(fontsize=7, frameon=False)
            F.save(fig, "adc_ot", "ADC per chip, split by the over-threshold flag")

        avc = np.asarray(chip["chip_adc_vs_ch"])[chips].astype(np.float64)
        if len(chips) <= MAX_PANELS:
            fig, axes = F.grid(len(chips))
            vmax = max(2.0, avc.max())
            for ax, h, lab in zip(axes, avc, labels):
                ax.imshow(np.where(h.T > 0, h.T, np.nan), origin="lower", aspect="auto",
                          cmap=F.cmap, norm=F.LogNorm(vmin=1, vmax=vmax), interpolation="nearest",
                          extent=(0, 64, 0, ADC_CH_ADC_BINS * 8))
                ax.set_ylim(0, top)
                ax.set_title(lab, fontsize=8, loc="left", color=INK)
                ax.grid(False)
            F.save(fig, "adc_vs_ch", "ADC vs channel per chip (log colour)")

        otf = np.asarray(chip["chip_ot"])[chips].astype(np.float64)
        frac = np.divide(otf[:, 1], otf.sum(1), out=np.zeros(len(chips)), where=otf.sum(1) > 0)
        fig, ax = F.plt.subplots(figsize=(10, 0.22 * len(chips) + 1.6), constrained_layout=True)
        y = np.arange(len(chips))
        ax.scatter(frac, y, color=BAR, s=30, zorder=3)
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_ylim(len(chips) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of hits with the over-threshold flag")
        ax.grid(color=GRID, linewidth=0.5)
        F.save(fig, "ot", "Over-threshold fraction per chip")

        F.per_chip_1d("chno", "Channel occupancy per chip", occ[chips].astype(np.float64),
                      chips, labels, np.arange(65), "channel")

        tot = occ[chips].sum(axis=1)
        fig, ax = F.plt.subplots(figsize=(10, 0.22 * len(chips) + 1.6), constrained_layout=True)
        ax.scatter(np.maximum(tot, 0.5), y, color=BAR, s=30, zorder=3)
        ax.set_xscale("log")
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_ylim(len(chips) - 0.5, -0.5)
        ax.set_xlabel("hits (log scale)")
        ax.grid(color=GRID, linewidth=0.5)
        for yi, v in zip(y, tot):
            ax.text(v * 1.15, yi, f"{int(v):,}", va="center", fontsize=7, color=INK)
        F.save(fig, "hits_per_vmm", "Hits per chip")

        bc = _rebin(np.asarray(chip["chip_bcid"])[chips].astype(np.float64), 32)
        F.per_chip_1d("bcid", "BCID per chip (32-BCID bins)", bc, chips, labels,
                      np.arange(bc.shape[1] + 1) * 32, "BCID")
        F.per_chip_1d("tdc", "TDC per chip", np.asarray(chip["chip_tdc"])[chips].astype(np.float64),
                      chips, labels, np.arange(TDC_BINS + 1), "TDC")
        F.per_chip_1d("offset", "Frame offset per chip (firmware sanity)",
                      np.asarray(chip["chip_offset"])[chips].astype(np.float64), chips, labels,
                      np.arange(OFFSET_BINS + 1) - 16.5, "offset")
        ts = np.asarray(chip["chip_ts"])[chips].astype(np.float64)
        F.per_chip_1d("timestamp_ns", "Hit time within its frame (timestamp_ns)", ts, chips,
                      labels, np.linspace(TS_LO, TS_HI, TS_BINS + 1) / 1e3, "µs")

        secs = np.asarray(chip.get("rate_s_sec", np.zeros(0)))
        if len(secs):
            rs = np.asarray(chip["rate_s"])[:, chips].astype(np.float64).T   # (chips, seconds)
            t = secs - secs[0]
            if len(chips) <= MAX_PANELS:
                fig, axes = F.grid(len(chips))
                for ax, r, lab in zip(axes, rs, labels):
                    ax.plot(t, r, color=BAR, linewidth=0.9, drawstyle="steps-mid")
                    ax.set_title(lab, fontsize=8, loc="left", color=INK)
                    ax.set_ylim(bottom=0)
                for ax in list(axes)[-NCOL:]:
                    ax.set_xlabel("s since first packet", fontsize=7)
            else:
                fig, ax = F.plt.subplots(figsize=(10, 0.06 * len(chips) + 2),
                                         constrained_layout=True)
                ax.imshow(rs, aspect="auto", cmap=F.cmap, interpolation="nearest",
                          extent=(t[0], t[-1] + 1, len(chips) - 0.5, -0.5))
                ax.set_xlabel("s since first packet")
            F.save(fig, "rate_vs_time", "Hits per second per chip, through the capture")

    if events_json:
        _write_events(stores, out_dir, chip, legacy, scal, meta, n_hits)
    return F.paths


def _write_events(stores, out_dir, chip, legacy, scal, meta, n_hits):
    """events.json with the keys the per-hit QA script wrote (get_run_events.py sums them)."""
    src = scal[0].get("source") or stores[0]
    parsed = parse_capture_name(src)
    off = np.asarray(chip["chip_offset"]).astype(np.int64)
    rng = np.asarray(chip.get("abs_time_range", [np.nan, np.nan]), np.float64)
    hpv = np.asarray(legacy["hits_per_vmm"]).astype(np.int64)
    occ = np.asarray(chip["chip_ch"]).astype(np.int64).sum(axis=1)
    events = {
        "pcap_file": os.path.abspath(src),
        "iface": parsed[0] if parsed else None,
        "seq": parsed[1] if parsed else None,
        "n_packets": int(sum(s.get("n_packets", 0) for s in scal)),
        "n_vm3_packets": int(sum(s.get("n_vm3_packets", 0) for s in scal)),
        "n_hits": n_hits,
        "n_hits_valid": int(off[:, 15:].sum()),          # offset >= -1
        "hits_per_vmm": {str(v): int(hpv[v]) for v in range(NV) if hpv[v]},
        "hits_per_chip": {f"{c // NV}:{c % NV}": int(occ[c]) for c in np.flatnonzero(occ)},
        "t_first_ns": None if np.isnan(rng[0]) else float(rng[0]),
        "t_last_ns": None if np.isnan(rng[1]) else float(rng[1]),
        "fec_ips": sorted({ip for m in meta for ip in (m.get("fec_ips") or [])}),
        "data_format": (meta[0] or {}).get("data_format"),
        "processed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "drawn_from": "histograms (p2-monitor qa)",
    }
    with open(os.path.join(out_dir, "events.json"), "w") as f:
        json.dump(events, f, indent=2)
