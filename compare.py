#!/usr/bin/env python3
"""
Compare original cover images with their stego versions (HILL + STC pipeline).

Metrics per image pair
    PSNR            Peak Signal-to-Noise Ratio in dB (higher = less distortion;
                    "inf" if the images are identical)
    SSIM            Structural Similarity Index, 1.0 = identical
    changed_pct     percentage of pixels whose value differs

Pairing rule (matches main.py -> embed_chunks):
    images/<name>.png   <->   Stego/stego_<name>.png

main.py saves stego images as 8-bit grayscale, so the originals are converted
to grayscale ("L") the same way before comparing.

Usage
-----
    python compare.py
    python compare.py --orig-dir images --stego-dir Stego --out results/compare.csv
    python compare.py --stego-dir results/stego_runs/<run_id>     # from benchmark.py --keep-stego
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as _sk_ssim
except ImportError:  # fall back to a numpy implementation
    _sk_ssim = None

FIELDS = ["image", "stego", "width", "height", "pixels",
          "psnr_db", "ssim", "changed_pixels", "changed_pct", "max_abs_diff"]


# ------------------------------------------------------------------ metrics
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(255.0 ** 2 / mse)


def _box_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Mean over every w x w window (valid positions only), via integral image."""
    c = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    s = c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]
    return s / (w * w)


def _ssim_numpy(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """Same defaults as scikit-image (7x7 uniform window, K1=.01, K2=.03)."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    np_ = win * win
    cov_norm = np_ / (np_ - 1)
    ux, uy = _box_mean(x, win), _box_mean(y, win)
    vx = cov_norm * (_box_mean(x * x, win) - ux * ux)
    vy = cov_norm * (_box_mean(y * y, win) - uy * uy)
    vxy = cov_norm * (_box_mean(x * y, win) - ux * uy)
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return float(s.mean())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if min(a.shape) < 7:
        return float("nan")
    if _sk_ssim is not None:
        return float(_sk_ssim(a, b, data_range=255))
    return _ssim_numpy(a, b)


def changed_pixels(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    n = int(np.count_nonzero(a != b))
    return n, 100.0 * n / a.size


# --------------------------------------------------------------------- I/O
def load_gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"), dtype=np.uint8)


def find_pairs(orig_dir, stego_dir, pattern):
    pairs, unmatched = [], []
    for o in sorted(glob.glob(os.path.join(orig_dir, pattern))):
        base = os.path.splitext(os.path.basename(o))[0]
        s = os.path.join(stego_dir, f"stego_{base}.png")
        (pairs if os.path.exists(s) else unmatched).append((o, s) if os.path.exists(s) else o)
    return pairs, unmatched


def fmt_psnr(v):
    return "inf" if math.isinf(v) else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-dir", default="images", help="folder with original images")
    ap.add_argument("--stego-dir", default="Stego", help="folder with stego_<name>.png files")
    ap.add_argument("--glob", default="*.png", help="pattern for originals")
    ap.add_argument("--out", default="results/compare.csv", help="CSV output path ('' to skip)")
    args = ap.parse_args()

    pairs, unmatched = find_pairs(args.orig_dir, args.stego_dir, args.glob)
    if not pairs:
        sys.exit(f"No original/stego pairs found ({args.orig_dir}/{args.glob} "
                 f"vs {args.stego_dir}/stego_<name>.png)")

    rows = []
    for o, s in pairs:
        a, b = load_gray(o), load_gray(s)
        if a.shape != b.shape:
            print(f"WARNING: size mismatch for {os.path.basename(o)}: "
                  f"{a.shape[::-1]} vs {b.shape[::-1]} -> skipped")
            continue
        n_changed, pct = changed_pixels(a, b)
        rows.append({
            "image": os.path.basename(o), "stego": os.path.basename(s),
            "width": a.shape[1], "height": a.shape[0], "pixels": a.size,
            "psnr_db": psnr(a, b), "ssim": ssim(a, b),
            "changed_pixels": n_changed, "changed_pct": pct,
            "max_abs_diff": int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()),
        })

    # ---- table
    hdr = f"{'image':<24}{'PSNR (dB)':>11}{'SSIM':>10}{'changed %':>12}{'changed px':>13}{'max |d|':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['image']:<24}{fmt_psnr(r['psnr_db']):>11}{r['ssim']:>10.6f}"
              f"{r['changed_pct']:>12.4f}{r['changed_pixels']:>13,}{r['max_abs_diff']:>9}")

    if rows:
        finite = [r["psnr_db"] for r in rows if not math.isinf(r["psnr_db"])]
        print("-" * len(hdr))
        print(f"{'MEAN':<24}{(fmt_psnr(float(np.mean(finite))) if finite else 'inf'):>11}"
              f"{np.mean([r['ssim'] for r in rows]):>10.6f}"
              f"{np.mean([r['changed_pct'] for r in rows]):>12.4f}")
        if len(finite) < len(rows):
            print("(mean PSNR excludes identical pairs, where PSNR = inf)")

    if unmatched:
        print("\nNo stego image found for (not embedded / skipped):")
        for o in unmatched:
            print(f"  {os.path.basename(o)}")

    # ---- csv
    if args.out and rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()