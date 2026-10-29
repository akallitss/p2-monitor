"""Detector view: simple analysis plots common to every campaign.

    p2-monitor analyze <run or sub-run dir> --out-dir DIR [--store-inner hits_store]

Everything comes from the stored histograms (counts_chip.npz, scalars.json) and
the mapping each store was reduced with (setup_resolved.json): no hit-level
data, no event building, so it is the same for cosmics, sources, pulser or beam
and costs the same for any amount of data.

Per sub-run (all its captures merged), into <out>/<sub-run>/:
    rate_map.png       hit rate per pad area on the detector surface
    channel_rates.png  hit rate per channel, per chip
    level_rates.png    rate per detector / connector / hybrid / chip
    vmm_adc.png        per VMM: ADC per channel (map) and the mean ADC per channel
    amplitude.png      ADC spectrum per detector, per connector half
    time.png           hit rate vs time per detector, through the sub-run
    summary.json       per detector and per chip: hits, rate, Hz/cm2, live,
                       dead and noisy channels, mean ADC
For a run, the same over all its sub-runs into <out>/_run/, plus
    trend.png          rate and mean ADC per detector, sub-run by sub-run
"""

import json
import os

import numpy as np

from .counts import ADC_CH_ADC_BINS, NV, merge_counts
from .plots import BAR, GRID, INK, INK_2, SEQ, _mpl, plot_counts, si
from .rates import channel_rates
from .store import find_stores, load_counts, load_scalars, store_mapping

NOISY_FACTOR = 10        # a channel above 10x its chip's median rate is "noisy"
TIME_BINS_MAX = 300      # time plot: at most this many bins
PALETTE = ["#2a78d6", "#d6672a", "#2aa876", "#a52ad6", "#d6b52a", "#2ac4d6",
           "#d62a6a", "#6b7a8f"]


def _live(scal):
    return sum((s.get("volume") or {}).get("live_time_s") or 0 for s in scal)


def _chip_owner(r):
    """{chip: (detector, connector, half)} from the channel table (mapped rows)."""
    out = {}
    if "detector" not in r:
        return out
    m = r[r["mapped"]] if "mapped" in r else r
    for (fec, vmm), g in m.groupby(["fec", "vmm"]):
        det = g["detector"].dropna()
        if len(det):
            row = g.iloc[0]
            out[(int(fec) if fec >= 0 else 0) * NV + int(vmm)] = (
                str(row["detector"]), int(row["connector_N"]), str(row["half"]))
    return out


def _chip_name(chip, owner, single_fec):
    f, v = divmod(int(chip), NV)
    base = f"VMM {v}" if single_fec else f"FEC {f} VMM {v}"
    o = owner.get(int(chip))
    return f"{base}  {o[0]} c{o[1]} {o[2]}" if o else f"{base}  (not cabled)"


def summary(counts, live, r, owner):
    """Numbers per detector and per chip, for summary.json and the tab."""
    occ = np.asarray(counts["chip_ch"]).astype(np.int64)
    avc = np.asarray(counts["chip_adc_vs_ch"]).astype(np.float64)
    centers = np.arange(ADC_CH_ADC_BINS) * 8 + 3.5
    chips = np.flatnonzero(occ.sum(axis=1))
    out = {"live_time_s": live, "hits": int(occ.sum()), "chips": {}, "detectors": {}}
    for c in chips:
        hits = occ[c]
        n = avc[c].sum(axis=1)
        mean_adc = np.divide((avc[c] * centers).sum(axis=1), n, out=np.full(64, np.nan),
                             where=n > 0)
        med = np.median(hits[hits > 0]) if (hits > 0).any() else 0
        owner_c = owner.get(int(c))
        cabled = owner_c is not None
        out["chips"][f"{c // NV}:{c % NV}"] = {
            "detector": owner_c[0] if cabled else None,
            "connector": owner_c[1] if cabled else None, "half": owner_c[2] if cabled else None,
            "hits": int(hits.sum()), "rate_hz": hits.sum() / live if live else None,
            "live_channels": int((hits > 0).sum()),
            "dead_channels": [int(i) for i in np.flatnonzero(hits == 0)],
            "noisy_channels": [int(i) for i in np.flatnonzero(hits > NOISY_FACTOR * med)] if med else [],
            "mean_adc": float(np.nansum(mean_adc * n) / n.sum()) if n.sum() else None,
        }
    if "detector" in r:
        for det, g in r[r["mapped"]].groupby("detector"):
            hz = g["rate_hz"].sum()
            area = g["pad_area_mm2"].sum() / 100.0 if "pad_area_mm2" in g else np.nan
            chips_d = [v for v in out["chips"].values() if v["detector"] == det]
            nh = sum(v["hits"] for v in chips_d)
            out["detectors"][det] = {
                "hits": int(g["hits"].sum()), "rate_hz": float(hz),
                "rate_hz_per_cm2": float(hz / area) if area else None,
                "channels": int(len(g)), "live_channels": int((g["hits"] > 0).sum()),
                "dead_channels": int((g["hits"] == 0).sum()),
                "noisy_channels": int(sum(len(v["noisy_channels"]) for v in chips_d)),
                "mean_adc": (float(sum((v["mean_adc"] or 0) * v["hits"] for v in chips_d) / nh)
                             if nh else None),
            }
    return out


def _save(plt, fig, path, title, sub):
    fig.suptitle(f"{title}\n{sub}", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _vmm_adc(plt, cmap, LogNorm, counts, owner, single_fec, sub, path):
    """Per VMM: channel x ADC map (log colour) with the mean ADC per channel."""
    occ = np.asarray(counts["chip_ch"]).astype(np.int64)
    avc = np.asarray(counts["chip_adc_vs_ch"]).astype(np.float64)
    chips = np.flatnonzero(occ.sum(axis=1))
    if not len(chips):
        return None
    ncol = min(len(chips), 4)
    nrow = -(-len(chips) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow + 0.9),
                             squeeze=False, constrained_layout=True)
    vmax = max(2.0, avc[chips].max())
    centers = np.arange(ADC_CH_ADC_BINS) * 8 + 3.5
    top = 1023
    for ax, c in zip(axes.flat, chips):
        h = avc[c]
        ax.imshow(np.where(h.T > 0, h.T, np.nan), origin="lower", aspect="auto", cmap=cmap,
                  norm=LogNorm(vmin=1, vmax=vmax), interpolation="nearest",
                  extent=(-0.5, 63.5, 0, ADC_CH_ADC_BINS * 8))
        n = h.sum(axis=1)
        mean = np.divide((h * centers).sum(axis=1), n, out=np.full(64, np.nan), where=n > 0)
        ax.plot(np.arange(64), mean, color=INK, linewidth=1.0)
        ax.set_ylim(0, top)
        ax.set_title(_chip_name(c, owner, single_fec), fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(chips):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("channel", fontsize=7)
    return _save(plt, fig, path, "ADC per channel, per VMM (colour: hits, log; line: mean ADC)", sub)


def _amplitude(plt, counts, owner, sub, path):
    """ADC spectrum per detector, one curve per connector half."""
    adc = np.asarray(counts["chip_adc"]).astype(np.float64)
    dets = sorted({o[0] for o in owner.values()})
    if not dets:
        return None
    fig, axes = plt.subplots(1, len(dets), figsize=(4.4 * len(dets), 3.4), squeeze=False,
                             constrained_layout=True)
    x = np.arange(adc.shape[1] + 1)
    for ax, det in zip(axes[0], dets):
        mine = sorted((o[1], o[2], c) for c, o in owner.items() if o[0] == det)
        for i, (conn, half, c) in enumerate(mine):
            if adc[c].any():
                ax.stairs(adc[c], x, color=PALETTE[i % len(PALETTE)], linewidth=0.9,
                          label=f"c{conn} {half}")
        tot = sum(adc[c] for _, _, c in mine)
        ax.set_title(f"{det}  ({int(np.sum(tot)):,} hits)", loc="left")
        ax.set_yscale("log")
        ax.set_xlabel("ADC")
        ax.grid(color=GRID, linewidth=0.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False, ncol=2)
        else:
            ax.text(0.5, 0.5, "no hits", transform=ax.transAxes, ha="center", color=INK_2)
    return _save(plt, fig, path, "Amplitude: ADC spectrum per detector and connector half", sub)


def _time(plt, counts, owner, sub, path):
    """Hit rate vs time per detector (packet arrival time, all captures)."""
    secs = np.asarray(counts.get("rate_s_sec", np.zeros(0)))
    if not len(secs):
        return None
    rs = np.asarray(counts["rate_s"]).astype(np.float64)
    dets = sorted({o[0] for o in owner.values()}) or ["all chips"]
    t = secs - secs[0]
    span = int(t[-1]) + 1
    width = max(1, -(-span // TIME_BINS_MAX))
    edges = np.arange(0, span + width, width)
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    for i, det in enumerate(dets):
        cols = [c for c, o in owner.items() if o[0] == det] if owner else list(range(rs.shape[1]))
        per = rs[:, cols].sum(axis=1)
        h, _ = np.histogram(t, bins=edges, weights=per)
        ax.stairs(h / width, edges / 60.0, color=PALETTE[i % len(PALETTE)], linewidth=1.0, label=det)
    ax.set_xlabel("minutes since the first packet")
    ax.set_ylabel(f"hits/s ({width} s bins)")
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False)
    return _save(plt, fig, path, "Hit rate vs time, per detector", sub)


def analyze_stores(stores, out_dir, title=""):
    """The detector view of these stores, merged. Returns (paths, summary dict)."""
    stores = [os.fspath(s) for s in stores]
    if not stores:
        raise FileNotFoundError("no store to analyse")
    os.makedirs(out_dir, exist_ok=True)
    scal = [load_scalars(s) for s in stores]
    counts = merge_counts(load_counts(s) for s in stores)    # one store at a time
    if "chip_adc_vs_ch" not in counts:
        raise ValueError(f"{stores[0]}: store made before p2-monitor 0.6; re-reduce it")
    live = _live(scal)
    try:
        mapping = store_mapping(stores[0])
    except Exception:
        mapping = None
    plt, cmap, LogNorm = _mpl()
    hits = sum(int(s.get("n_hits", 0)) for s in scal)
    sub = (f"{title}{' · ' if title else ''}{len(stores)} capture(s), {live / 60:.1f} min live, "
           f"{hits:,} hits" + (f", setup {mapping.setup.name}" if mapping else ", no mapping"))
    paths = list(plot_counts(counts, live, mapping, sub, out_dir))
    r = channel_rates(counts, live, mapping)
    owner = _chip_owner(r)
    fecs = {int(k) // NV for k in np.flatnonzero(np.asarray(counts["chip_ch"]).sum(axis=1))}
    single = len(fecs) <= 1
    for p in (_vmm_adc(plt, cmap, LogNorm, counts, owner, single, sub,
                       os.path.join(out_dir, "vmm_adc.png")),
              _amplitude(plt, counts, owner, sub, os.path.join(out_dir, "amplitude.png")),
              _time(plt, counts, owner, sub, os.path.join(out_dir, "time.png"))):
        if p:
            paths.append(p)
    summ = summary(counts, live, r, owner)
    summ.update(title=title, captures=len(stores),
                setup=mapping.setup.name if mapping else None,
                first_capture=scal[0].get("capture_start"),
                last_capture=scal[-1].get("capture_start"))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    return paths, summ


def _trend(subruns, out_dir, run_name):
    """Rate and mean ADC per detector, sub-run by sub-run."""
    dets = sorted({d for _, s in subruns for d in s["detectors"]})
    if not dets or len(subruns) < 2:
        return None
    plt, _, _ = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(max(7, 0.55 * len(subruns) + 3), 6.2),
                             sharex=True, constrained_layout=True)
    x = np.arange(len(subruns))
    for i, det in enumerate(dets):
        col = PALETTE[i % len(PALETTE)]
        rate = [s["detectors"].get(det, {}).get("rate_hz") for _, s in subruns]
        adc = [s["detectors"].get(det, {}).get("mean_adc") for _, s in subruns]
        axes[0].plot(x, [np.nan if v is None else v for v in rate], "o-", color=col, label=det)
        axes[1].plot(x, [np.nan if v is None else v for v in adc], "o-", color=col, label=det)
    axes[0].set_ylabel("hit rate (Hz)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("mean ADC")
    for ax in axes:
        ax.grid(color=GRID, linewidth=0.5)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xticks(x, [n for n, _ in subruns], rotation=40, ha="right", fontsize=7)
    path = os.path.join(out_dir, "trend.png")
    return _save(plt, fig, path, "Per sub-run: hit rate and mean ADC per detector", run_name)


def analyze_run(run_dir, out_root, store_inner="hits_store", subrun=None):
    """Detector view of a run: each sub-run, then the whole run (+ trend).

    subrun: only that sub-run (and no run-level output). Returns {name: paths}.
    """
    run_dir = os.fspath(run_dir).rstrip("/")
    run = os.path.basename(run_dir)
    subs = sorted(d for d in os.listdir(run_dir)
                  if os.path.isdir(os.path.join(run_dir, d, store_inner))
                  and (subrun is None or d == subrun))
    done, every, summaries = {}, [], []
    for s in subs:
        stores = find_stores(os.path.join(run_dir, s, store_inner))
        if not stores:
            continue
        paths, summ = analyze_stores(stores, os.path.join(out_root, s), title=f"{run} / {s}")
        done[s] = paths
        every += stores
        summaries.append((s, summ))
    if subrun is None and every:
        paths, summ = analyze_stores(every, os.path.join(out_root, "_run"), title=f"{run}, all sub-runs")
        t = _trend(summaries, os.path.join(out_root, "_run"), run)
        done["_run"] = paths + ([t] if t else [])
    return done
