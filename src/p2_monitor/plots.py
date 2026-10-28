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
    plt, cmap, LogNorm = _mpl()
    stores = find_stores(root)
    if not stores:
        raise FileNotFoundError(f"no store under {root}")
    scal = [load_scalars(s) for s in stores]
    counts = merged_counts(stores)
    live = sum((s.get("volume") or {}).get("live_time_s") or 0 for s in scal)
    if mapping is None:
        mapping = store_mapping(stores[0])
    r = channel_rates(counts, live, mapping)
    sub = _subtitle(scal, live)
    os.makedirs(out_dir, exist_ok=True)
    out = []

    if mapping is not None:
        out.append(_rate_map(plt, cmap, LogNorm, r, mapping, sub,
                             os.path.join(out_dir, "rate_map.png")))
    out.append(_channel_rates(plt, cmap, LogNorm, r, sub,
                              os.path.join(out_dir, "channel_rates.png")))
    out.append(_level_rates(plt, r, scal, sub, os.path.join(out_dir, "level_rates.png")))
    return out


def _rate_map(plt, cmap, LogNorm, r, mapping, sub, path):
    dets = list(mapping.setup.detectors)
    pads = load_pads(mapping.setup.design)
    m = r[r["mapped"].fillna(False).astype(bool)]
    norm = _norm(LogNorm, m["rate_hz_per_cm2"])
    fig, axes = plt.subplots(1, len(dets), figsize=(4.2 * len(dets), 4.6),
                             squeeze=False, constrained_layout=True)
    for ax, det in zip(axes[0], dets):
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
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, pad=0.01)
    cb.set_label("Hz/cm² (log scale)")
    cb.outline.set_visible(False)
    fig.suptitle("Hit rate per pad area (grey: no hits or not cabled)\n" + sub,
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _channel_rates(plt, cmap, LogNorm, r, sub, path):
    vmms = sorted(int(v) for v in r["vmm"].unique())
    grid = np.full((len(vmms), 64), np.nan)
    row = {v: i for i, v in enumerate(vmms)}
    for v, c, x in zip(r["vmm"], r["ch"], r["rate_hz"]):
        grid[row[int(v)], int(c)] = x
    norm = _norm(LogNorm, grid)
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(vmms) + 1.6), constrained_layout=True)
    from matplotlib.colors import ListedColormap
    ax.imshow(np.where(np.isnan(grid), 1, np.nan), cmap=ListedColormap([EMPTY]),
              aspect="auto", interpolation="nearest")
    zero = np.where(grid == 0, 1, np.nan)
    ax.imshow(np.where(grid > 0, grid, np.nan), cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest")
    yy, xx = np.nonzero(zero == 1)
    ax.scatter(xx, yy, marker="x", s=10, c=INK_2, linewidths=0.8, label="cabled, no hits")
    ax.set_yticks(range(len(vmms)), [f"VMM {v}" for v in vmms])
    ax.set_xticks(range(0, 64, 8))
    ax.set_xlabel("channel")
    det = r.set_index("vmm")["detector"].dropna() if "detector" in r else None
    if det is not None:
        lab = det.groupby(level=0).first()
        for v, i in row.items():
            if v in lab.index:
                ax.text(64.3, i, lab[v], va="center", fontsize=7, color=INK_2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.05)
    cb.set_label("rate per channel (Hz, log scale)")
    cb.outline.set_visible(False)
    if len(xx):
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.06), frameon=False, fontsize=8)
    ax.set_title("Hit rate per channel (grey rows: VMM not cabled to a detector)\n" + sub,
                 loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
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


def _level_rates(plt, r, scal, sub, path):
    per_vmm = r.groupby("vmm")["rate_hz"].sum()
    has_det = "detector" in r and r["detector"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_det else 1, figsize=(10, 0.22 * len(per_vmm) + 1.8),
                             squeeze=False, constrained_layout=True,
                             gridspec_kw={"width_ratios": [3, 2] if has_det else [1]})
    _dots(axes[0][0], [f"VMM {v}" for v in per_vmm.index], per_vmm.to_numpy(), "Hz")
    axes[0][0].set_title("Rate per VMM (Hz, log scale)", loc="left")
    if has_det:
        m = r[r["mapped"].fillna(False).astype(bool)]
        g = m.groupby("detector")
        areal = (g["rate_hz"].sum() / (g["pad_area_mm2"].sum() / 100.0))
        _dots(axes[0][1], list(areal.index), areal.to_numpy(), "Hz/cm²")
        axes[0][1].set_title("Rate per detector area (Hz/cm², log scale)", loc="left")
    fig.suptitle("Hit rate by level\n" + sub, x=0.01, ha="left", fontsize=10,
                 fontweight="bold", color=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
