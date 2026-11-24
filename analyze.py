#!/usr/bin/env python3
"""
Simple graphs from results/runs.csv (made by benchmark.py).

    python analyze.py                  # reads ./results, writes ./results/plots

Three graphs:
  1_time_vs_images.png   one small panel per text size, own y-scale each
  2_time_vs_text.png     one small panel per image count, own y-scale each
  3_where_time_goes.png  which pipeline stage takes the time

Every panel has its own y-axis, zoomed to its own data, so differences are
visible even when text sizes or image counts differ a lot in absolute time.
The Poppins font is loaded from the 'fonts' folder next to this script.
"""
import argparse
import glob
import os

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402
from matplotlib.ticker import MaxNLocator            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

BG, INK, MUTED, GRID = "#FFFFFF", "#222222", "#777777", "#E8E8E8"
BLUE, ORANGE = "#3B6FE0", "#F28C28"

STAGES = {"t_capacity_s": "HILL capacity", "t_embed_s": "STC embedding",
          "t_split_encrypt_s": "Split + AES-GCM", "t_keygen_wrap_s": "AES key + RSA wrap",
          "t_header_s": "Header sizes"}


def setup_style():
    for f in glob.glob(os.path.join(HERE, "fonts", "*.ttf")):
        font_manager.fontManager.addfont(f)
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ["Poppins", "Inter", "Segoe UI", "Helvetica Neue",
                               "Arial", "DejaVu Sans"] if f in names), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": family, "text.color": INK,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
        "axes.axisbelow": True,
        "axes.titlesize": 13, "axes.titleweight": "medium", "axes.titlepad": 10,
        "axes.labelcolor": INK, "axes.labelsize": 11,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def fmt_bytes(n):
    n = float(n)
    if n < 1000:
        return f"{n:g} B"
    return f"{n / 1000:g} KB" if n < 1e6 else f"{n / 1e6:g} MB"


def fmt_time(s):
    return f"{s * 1000:.1f} ms" if s < 0.1 else f"{s:.2f} s"


def zoom_y(ax, lo, hi):
    """Fit y-range to the data so small differences are visible."""
    span = hi - lo
    pad = span * 0.2 if span > 0 else max(hi * 0.1, 0.01)
    ax.set_ylim(max(0, lo - pad), hi + pad)


def panels(n, ncols):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.5 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def finish(fig, title, subtitle, xlabel, ylabel, path):
    fig.suptitle(title, x=0.01, ha="left", fontsize=17, fontweight="bold", y=1.10)
    fig.text(0.01, 1.045, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)
    fig.supxlabel(xlabel, fontsize=11)
    fig.supylabel(ylabel, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def draw_line(ax, x, mean, std, color):
    std = np.nan_to_num(std)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, lw=2.4, marker="o", ms=7,
            markeredgecolor="white", markeredgewidth=1.5)
    zoom_y(ax, (mean - std).min(), (mean + std).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()

    setup_style()
    out = os.path.join(args.dir, "plots")
    os.makedirs(out, exist_ok=True)

    runs = pd.read_csv(os.path.join(args.dir, "runs.csv"))
    ok = runs[runs.status == "ok"]
    if ok.empty:
        raise SystemExit("No successful runs in runs.csv")
    agg = (ok.groupby(["text_bytes", "n_images"])["t_total_s"]
             .agg(["mean", "std"]).reset_index())
    sizes = sorted(agg.text_bytes.unique())
    counts = sorted(agg.n_images.unique())

    # 1) time vs number of images: one panel per text size
    fig, axes = panels(len(sizes), min(3, len(sizes)))
    for ax, s in zip(axes, sizes):
        g = agg[agg.text_bytes == s].sort_values("n_images")
        draw_line(ax, g.n_images.values, g["mean"].values, g["std"].values, BLUE)
        ax.set_title(fmt_bytes(s))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    finish(fig, "Runtime vs number of images",
           "Each panel is one text size and has its own y-scale. Shaded band = ±1 std.",
           "number of images", "total time (s)",
           os.path.join(out, "1_time_vs_images.png"))

    # 2) time vs text size: one panel per image count (categorical x, even spacing)
    fig, axes = panels(len(counts), min(4, len(counts)))
    for ax, n in zip(axes, counts):
        g = agg[agg.n_images == n].sort_values("text_bytes")
        pos = [sizes.index(v) for v in g.text_bytes]
        draw_line(ax, np.array(pos), g["mean"].values, g["std"].values, ORANGE)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([fmt_bytes(v) for v in sizes], fontsize=8.5)
        ax.set_xlim(-0.4, len(sizes) - 0.6)
        ax.set_title(f"{n} image{'s' if n > 1 else ''}")
    finish(fig, "Runtime vs text size",
           "Each panel is one image count and has its own y-scale. Shaded band = ±1 std.",
           "text size", "total time (s)",
           os.path.join(out, "2_time_vs_text.png"))

    # 3) which stage takes the time
    means = ok[list(STAGES)].mean().sort_values()
    total = means.sum()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.barh([STAGES[k] for k in means.index], means.values,
                   color=BLUE, height=0.6)
    bars[-1].set_color(ORANGE)
    for b, v in zip(bars, means.values):
        ax.text(v + total * 0.01, b.get_y() + b.get_height() / 2,
                f"{fmt_time(v)}  ({v / total:.0%})", va="center", fontsize=10)
    ax.set_xlim(0, means.max() * 1.35)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("mean time per run (s)")
    ax.set_title("Where the time goes", loc="left", fontsize=17, fontweight="bold", pad=24)
    ax.text(0, 1.03, "Mean over all logged runs", transform=ax.transAxes,
            fontsize=10.5, color=MUTED)
    fig.tight_layout()
    path = os.path.join(out, "3_where_time_goes.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()