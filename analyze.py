#!/usr/bin/env python3
"""
Analyse results/runs.csv and results/images.csv produced by benchmark.py.

    python analyze.py                 # reads ./results, writes ./results/plots
    python analyze.py --dir results --show

Outputs
-------
    results/summary_by_config.csv   mean/std/min/max per (text_bytes, n_images)
    results/stage_breakdown.csv     mean stage times per (text_bytes, n_images)
    results/plots/*.png
Timing plots use only runs where status == ok AND every chunk fit its image
(otherwise embedding was skipped and the timing is not comparable).
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

STAGES = ["t_capacity_s", "t_keygen_wrap_s", "t_header_s",
          "t_split_encrypt_s", "t_embed_s"]
STAGE_LABELS = {"t_capacity_s": "HILL capacity", "t_keygen_wrap_s": "AES key + RSA wrap",
                "t_header_s": "header sizes", "t_split_encrypt_s": "split + AES-GCM",
                "t_embed_s": "STC embed"}


def load(d):
    runs = pd.read_csv(os.path.join(d, "runs.csv"))
    imgs = pd.read_csv(os.path.join(d, "images.csv"))
    runs["all_fit"] = runs["all_fit"].astype(str).str.lower().eq("true")
    return runs, imgs


def save(fig, d, name, show):
    path = os.path.join(d, "plots", name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if not show:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print("  wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = args.dir
    os.makedirs(os.path.join(d, "plots"), exist_ok=True)
    runs, imgs = load(d)

    ok = runs[(runs.status == "ok") & runs.all_fit].copy()
    print(f"{len(runs)} runs total, {len(ok)} usable for timing "
          f"({len(runs) - len(ok)} errored or did not fit).")
    if ok.empty:
        print("Nothing to plot: no run had every chunk fit. Try smaller text or more images.")
        return

    # ---------- tables ----------
    agg = (ok.groupby(["text_bytes", "n_images"])["t_total_s"]
             .agg(["count", "mean", "std", "min", "max"]).reset_index())
    agg.to_csv(os.path.join(d, "summary_by_config.csv"), index=False)
    stage = ok.groupby(["text_bytes", "n_images"])[STAGES + ["t_total_s"]].mean().reset_index()
    stage.to_csv(os.path.join(d, "stage_breakdown.csv"), index=False)
    print(agg.to_string(index=False))

    sizes = sorted(ok.text_bytes.unique())
    counts = sorted(ok.n_images.unique())

    # 1) total time vs number of images, one line per text size
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in sizes:
        g = agg[agg.text_bytes == s]
        ax.errorbar(g.n_images, g["mean"], yerr=g["std"].fillna(0), marker="o",
                    capsize=3, label=f"{s:,} B")
    ax.set_xlabel("number of images"); ax.set_ylabel("total time (s)")
    ax.set_title("Total runtime vs number of images")
    ax.legend(title="text size"); ax.grid(alpha=.3)
    save(fig, d, "01_time_vs_images.png", args.show)

    # 2) total time vs text size, one line per image count
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in counts:
        g = agg[agg.n_images == n]
        ax.errorbar(g.text_bytes, g["mean"], yerr=g["std"].fillna(0), marker="o",
                    capsize=3, label=f"{n} img")
    ax.set_xscale("log"); ax.set_xlabel("text size (bytes, log)")
    ax.set_ylabel("total time (s)"); ax.set_title("Total runtime vs text size")
    ax.legend(ncol=2); ax.grid(alpha=.3, which="both")
    save(fig, d, "02_time_vs_text_size.png", args.show)

    # 3) stage breakdown vs number of images (averaged over text sizes)
    by_n = ok.groupby("n_images")[STAGES].mean().reindex(counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(by_n))
    for st in STAGES:
        ax.bar(by_n.index.astype(str), by_n[st], bottom=bottom, label=STAGE_LABELS[st])
        bottom += by_n[st].values
    ax.set_xlabel("number of images"); ax.set_ylabel("mean time (s)")
    ax.set_title("Where the time goes (avg over text sizes)")
    ax.legend(); save(fig, d, "03_stage_breakdown_by_images.png", args.show)

    # 4) stage breakdown vs text size (averaged over image counts)
    by_s = ok.groupby("text_bytes")[STAGES].mean().reindex(sizes)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(by_s))
    for st in STAGES:
        ax.bar([f"{s:,}" for s in by_s.index], by_s[st], bottom=bottom,
               label=STAGE_LABELS[st])
        bottom += by_s[st].values
    ax.set_xlabel("text size (bytes)"); ax.set_ylabel("mean time (s)")
    ax.set_title("Where the time goes (avg over image counts)")
    ax.legend(); save(fig, d, "04_stage_breakdown_by_size.png", args.show)

    # 5) heatmap of mean total time
    piv = agg.pivot(index="text_bytes", columns="n_images", values="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{s:,}" for s in piv.index])
    ax.set_xlabel("number of images"); ax.set_ylabel("text size (bytes)")
    ax.set_title("Mean total time (s)"); fig.colorbar(im)
    save(fig, d, "05_heatmap_time.png", args.show)

    # 6) feasibility heatmap: fraction of runs where all chunks fit
    feas = (runs.groupby(["text_bytes", "n_images"])["all_fit"].mean()
                .unstack("n_images"))
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(feas.values, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(feas.columns))); ax.set_xticklabels(feas.columns)
    ax.set_yticks(range(len(feas.index))); ax.set_yticklabels([f"{s:,}" for s in feas.index])
    ax.set_xlabel("number of images"); ax.set_ylabel("text size (bytes)")
    ax.set_title("Fraction of runs where payload fits"); fig.colorbar(im)
    save(fig, d, "06_feasibility.png", args.show)

    # 7) throughput (text bytes processed per second)
    ok["throughput_Bps"] = ok.text_bytes / ok.t_total_s
    tp = ok.groupby(["text_bytes", "n_images"])["throughput_Bps"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in sizes:
        g = tp[tp.text_bytes == s]
        ax.plot(g.n_images, g.throughput_Bps, marker="o", label=f"{s:,} B")
    ax.set_yscale("log"); ax.set_xlabel("number of images")
    ax.set_ylabel("throughput (bytes/s, log)"); ax.set_title("Throughput")
    ax.legend(title="text size"); ax.grid(alpha=.3, which="both")
    save(fig, d, "07_throughput.png", args.show)

    # 8) capacity utilisation vs text size
    fig, ax = plt.subplots(figsize=(8, 5))
    u = runs[runs.status == "ok"].groupby(["text_bytes", "n_images"])["utilization"].mean().reset_index()
    for n in sorted(u.n_images.unique()):
        g = u[u.n_images == n]
        ax.plot(g.text_bytes, g.utilization * 100, marker="o", label=f"{n} img")
    ax.axhline(100, color="red", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("text size (bytes, log)")
    ax.set_ylabel("payload / total capacity (%)"); ax.set_title("Capacity utilisation")
    ax.legend(ncol=2); ax.grid(alpha=.3, which="both")
    save(fig, d, "08_utilization.png", args.show)

    # 9) per-image embed time vs payload size and vs pixels
    im_ok = imgs[imgs.embedded.astype(str).str.lower().eq("true")]
    if not im_ok.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(im_ok.encrypted_len_bits, im_ok.t_embed_s, s=12, alpha=.5)
        axes[0].set_xlabel("encrypted payload (bits)"); axes[0].set_ylabel("embed time (s)")
        axes[0].set_title("STC embed time vs payload")
        for name, g in im_ok.groupby("image"):
            axes[1].scatter(g.pixels, g.t_embed_s, s=12, alpha=.5, label=name)
        axes[1].set_xlabel("image pixels"); axes[1].set_title("STC embed time vs image size")
        axes[1].legend(fontsize=7)
        for a in axes:
            a.grid(alpha=.3)
        save(fig, d, "09_embed_per_image.png", args.show)

    # 10) overhead of header + nonce + tag as a fraction of payload
    fig, ax = plt.subplots(figsize=(8, 5))
    o = runs[runs.status == "ok"].groupby(["text_bytes", "n_images"])[["overhead_bits", "text_bits"]].mean().reset_index()
    o["overhead_pct"] = o.overhead_bits / o.text_bits * 100
    for n in sorted(o.n_images.unique()):
        g = o[o.n_images == n]
        ax.plot(g.text_bytes, g.overhead_pct, marker="o", label=f"{n} img")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("text size (bytes)"); ax.set_ylabel("overhead (% of text bits)")
    ax.set_title("Crypto + header overhead (nonce 96b + tag 128b + header per image)")
    ax.legend(ncol=2); ax.grid(alpha=.3, which="both")
    save(fig, d, "10_overhead.png", args.show)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()