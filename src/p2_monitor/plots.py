"""Rate plots of one store (or several merged): PNGs for the GUI and the logbook.

    rate_map.png       hit rate per pad area on each detector, one shared scale
    channel_rates.png  rate per (VMM, channel); dead channels marked
    level_rates.png    rate per VMM and per detector, with the data volume

Magnitude is one hue, light to dark, on a log scale (rates span from dead
channels to MHz noisy ones). Uncabled or silent pads are neutral grey, never
a colour of the scale. Text stays in ink colours.
"""

import os

import numpy as np

from p2_mapping import load_pads

from .rates import channel_rates
from .store import find_stores, load_scalars, merged_counts, store_mapping

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
EMPTY = "#dcdbd6"                     # no hits / not cabled: neutral, off-scale
BAR = "#2a78d6"                       # one measure, one hue
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": GRID,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    })
    return plt, LinearSegmentedColormap.from_list("p2seq", SEQ), LogNorm


def _norm(LogNorm, values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if not len(v):
        return LogNorm(1e-3, 1.0)
    lo, hi = v.min(), v.max()
    return LogNorm(lo, hi if hi > lo else lo * 10)


def si(v, unit):
    """2720 Hz -> '2.72 kHz'."""
    for f, p in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m")):
        if abs(v) >= f:
            return f"{v / f:.3g} {p}{unit}"
    return f"{v:.3g} {unit}"


def _subtitle(scal_list, live):
    """Two lines: what was measured, then how much."""
    first = scal_list[0]
    n = len(scal_list)
    what = first.get("capture") if n == 1 else f"{n} captures"
    vol = first.get("volume") or {}
    mb = sum((s.get("volume") or {}).get("file_bytes", 0) for s in scal_list) / 1e6
    rate = f", {mb / live * 60:.0f} MB/min" if live else ""
    return (f"{what}, setup {first.get('setup')}\n"
            f"live {live:.1f} s, {sum(s.get('n_hits', 0) for s in scal_list):,} hits{rate}"
            + ("; frame loss not measurable (frame counter constant)"
               if vol.get("frame_loss") else ""))


def plot_store(root, out_dir, mapping=None):
    """Render the three PNGs for the store(s) under `root`. Returns their paths.

    `mapping` defaults to the one recorded with the first store
    (setup_resolved.json), so a store is drawn with the cabling it was reduced with.
    """
    stores = find_stores(root)
    if not stores:
        raise FileNotFoundError(f"no store under {root}")
    scal = [load_scalars(s) for s in stores]
    counts = merged_counts(stores)
    live = sum((s.get("volume") or {}).get("live_time_s") or 0 for s in scal)
    if mapping is None:
        mapping = store_mapping(stores[0])
    return plot_counts(counts, live, mapping, _subtitle(scal, live), out_dir)


def plot_counts(counts, live, mapping, subtitle, out_dir):
    """The three PNGs from counts and their live time (stores or the live window)."""
    plt, cmap, LogNorm = _mpl()
    r = channel_rates(counts, live, mapping)
    sub = subtitle
    os.makedirs(out_dir, exist_ok=True)
    out = []

    if mapping is not None:
        out.append(_rate_map(plt, cmap, LogNorm, r, mapping, sub,
                             os.path.join(out_dir, "rate_map.png")))
    out.append(_channel_rates(plt, cmap, LogNorm, r, sub,
                              os.path.join(out_dir, "channel_rates.png")))
    out.append(_level_rates(plt, r, sub, os.path.join(out_dir, "level_rates.png")))
    return out


def _rate_map(plt, cmap, LogNorm, r, mapping, sub, path):
    """One pad map per detector, on a grid with one shared colour scale."""
    dets = list(mapping.setup.detectors)
    pads = load_pads(mapping.setup.design)
    m = r[r["mapped"]]
    norm = _norm(LogNorm, m["rate_hz_per_cm2"])
    ncol = min(len(dets), 4)
    nrow = -(-len(dets) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.3 * nrow + 0.6),
                             squeeze=False, constrained_layout=True)
    sc = None
    for ax, det in zip(axes.flat, dets):
        ax.scatter(pads["pad_cx"], pads["pad_cy"], s=7, marker="s", c=EMPTY,
                   linewidths=0)
        d = m[m["detector"] == det]
        on = d[d["hits"] > 0]
        sc = ax.scatter(on["pad_cx"], on["pad_cy"], s=7, marker="s",
                        c=on["rate_hz_per_cm2"], cmap=cmap, norm=norm, linewidths=0)
        dead = int((d["hits"] == 0).sum())
        z = mapping.setup.detectors[det].get("z_mm")
        ax.set_title(f"{det}" + (f"  (z = {z:.0f} mm)" if z is not None else ""), loc="left")
        ax.text(0.0, -0.02, f"{len(d) - dead} of {len(d)} cabled pads fired",
                transform=ax.transAxes, va="top", color=INK_2, fontsize=8)
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    cb.set_label("Hz/cm² (log scale)")
    cb.outline.set_visible(False)
    fig.suptitle("Hit rate per pad area (grey: no hits or not cabled)\n" + sub,
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _chip_label(fec, vmm):
    return f"VMM {vmm}" if fec < 0 else f"FEC {fec} VMM {vmm}"


def _channel_rates(plt, cmap, LogNorm, r, sub, path):
    """Rate per (chip, channel), one row per chip; dead cabled channels marked."""
    chips = sorted({(int(f), int(v)) for f, v in r[["fec", "vmm"]].itertuples(index=False)})
    grid = np.full((len(chips), 64), np.nan)
    row = {c: i for i, c in enumerate(chips)}
    for f, v, c, x in zip(r["fec"], r["vmm"], r["ch"], r["rate_hz"]):
        grid[row[(int(f), int(v))], int(c)] = x
    norm = _norm(LogNorm, grid)
    h = min(0.28, max(0.06, 18.0 / max(len(chips), 1)))      # rows shrink when there are many
    fig, ax = plt.subplots(figsize=(10, h * len(chips) + 1.8), constrained_layout=True)
    from matplotlib.colors import ListedColormap
    ax.imshow(np.where(np.isnan(grid), 1, np.nan), cmap=ListedColormap([EMPTY]),
              aspect="auto", interpolation="nearest")
    ax.imshow(np.where(grid > 0, grid, np.nan), cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest")
    yy, xx = np.nonzero(grid == 0)
    if len(chips) <= 40:
        ax.scatter(xx, yy, marker="x", s=10, c=INK_2, linewidths=0.8, label="cabled, no hits")
    elif len(xx):
        ax.imshow(np.where(grid == 0, 1, np.nan), cmap=ListedColormap([INK_2]),
                  aspect="auto", interpolation="nearest")
    step = 1 if len(chips) <= 40 else max(1, len(chips) // 40)
    ax.set_yticks(range(0, len(chips), step),
                  [_chip_label(*chips[i]) for i in range(0, len(chips), step)],
                  fontsize=8 if len(chips) <= 40 else 6)
    ax.set_xticks(range(0, 64, 8))
    ax.set_xlabel("channel")
    if "detector" in r and len(chips) <= 40:
        lab = r.dropna(subset=["detector"]).groupby(["fec", "vmm"])["detector"].first()
        for c, i in row.items():
            if c in lab.index:
                ax.text(64.3, i, lab[c], va="center", fontsize=7, color=INK_2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.05)
    cb.set_label("rate per channel (Hz, log scale)")
    cb.outline.set_visible(False)
    if len(xx):
        if len(chips) <= 40:
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.06), frameon=False, fontsize=8)
        else:
            ax.text(0, -0.06, "dark grey: cabled, no hits", transform=ax.transAxes,
                    fontsize=8, color=INK_2, va="top")
    ax.set_title("Hit rate per channel (light grey: chip not cabled to a detector)\n" + sub,
                 loc="left")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _dots(ax, labels, values, unit):
    """A value per row as a dot on a log axis (bars on a log axis mislead:
    their length depends on where the axis starts). Zero is written out."""
    y = np.arange(len(labels))
    values = np.asarray(values, dtype=float)
    pos = values > 0
    ax.scatter(values[pos], y[pos], s=40, color=BAR, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    if pos.any():
        ax.set_xscale("log")
        lo, hi = values[pos].min(), values[pos].max()
        ax.set_xlim(lo / 3, hi * 6)
    else:
        # nothing to place on a log axis: keep the rows and their "no hits"
        ax.set_xticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for yi, v in zip(y, values):
        if v > 0:
            ax.text(v * 1.25, yi, si(v, unit), va="center", fontsize=7, color=INK)
        else:
            ax.text(0.0, yi, "  no hits", va="center", fontsize=7, color=INK_2,
                    transform=ax.get_yaxis_transform())


def _level_rates(plt, r, sub, path):
    """Rate per chip (or per hybrid / FEC when chips are many) and per detector area."""
    chips = r.groupby(["fec", "vmm"])["rate_hz"].sum()
    has_det = "detector" in r and r["detector"].notna().any()
    if len(chips) <= 40:
        labels = [_chip_label(f, v) for f, v in chips.index]
        values, what = chips.to_numpy(), "chip"
    elif has_det:
        hyb = r[r["mapped"]].groupby("hybrid")["rate_hz"].sum()
        labels, values, what = [f"H{int(h)}" for h in hyb.index], hyb.to_numpy(), "hybrid"
    else:
        fec = r.groupby("fec")["rate_hz"].sum()
        labels, values, what = [f"FEC {int(f)}" for f in fec.index], fec.to_numpy(), "FEC"
    fig, axes = plt.subplots(1, 2 if has_det else 1,
                             figsize=(10, min(0.22, max(0.1, 9.0 / len(labels))) * len(labels) + 1.8),
                             squeeze=False, constrained_layout=True,
                             gridspec_kw={"width_ratios": [3, 2] if has_det else [1]})
    _dots(axes[0][0], labels, values, "Hz")
    axes[0][0].set_title(f"Rate per {what} (Hz, log scale)", loc="left")
    if has_det:
        m = r[r["mapped"]]
        g = m.groupby("detector")
        areal = (g["rate_hz"].sum() / (g["pad_area_mm2"].sum() / 100.0))
        _dots(axes[0][1], list(areal.index), areal.to_numpy(), "Hz/cm²")
        axes[0][1].set_title("Rate per detector area (Hz/cm², log scale)", loc="left")
    fig.suptitle("Hit rate by level\n" + sub, x=0.01, ha="left", fontsize=10,
                 fontweight="bold", color=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
