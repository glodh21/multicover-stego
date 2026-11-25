"""Steganalysis module for HILL + STC grayscale stego images.

Implemented attacks only:
  Blind attacks: Chi-Square, RS, Weighted Steganographic, and LSB statistical tests.
  Known-cover attacks: Block-Structure, Change-Map FFT, and HILL Selection-Channel analysis.

Example:
    python attack.py --stego Stego --cover images --out steganalysis_res

Known-cover images must match stego names: stego_<base>.png -> <base>.<ext>.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from PIL import Image
from scipy import stats
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d

from image_percent import HILL

# Heuristic flag thresholds (indicators only; not proof of successful detection).
CHI_P_FLAG = 0.95
RS_P_FLAG = 0.05
WS_P_FLAG = 0.05
LSB_P_FLAG = 0.01
AUC_FLAG = 0.60
FFT_P_FLAG = 0.01
BLOCK_P_FLAG = 0.01

DEFAULT_PERMUTATIONS = 200
DEFAULT_BLOCK_SIZES = (8, 16, 32, 64)


def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def chi_square_attack(x):
    """Westfeld-Pfitzmann chi-square test on pairs (2k, 2k+1)."""
    h = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    h0, h1 = h[0::2], h[1::2]
    expected = (h0 + h1) / 2.0
    valid = expected >= 5
    chi = float(np.sum((h0[valid] - expected[valid]) ** 2 / expected[valid]))
    df = int(valid.sum()) - 1
    p = float(stats.chi2.sf(chi, df)) if df > 0 else float("nan")
    return {"chi2": chi, "chi2_df": df, "chi2_p_embed": p}


def _flip(v, mask):
    out = v.copy()
    for col, mode in enumerate(mask):
        if mode == 1:
            out[:, col] = v[:, col] ^ 1
        elif mode == -1:
            out[:, col] = ((v[:, col] + 1) ^ 1) - 1
    return out


def _rs_counts(groups, mask):
    f0 = np.abs(np.diff(groups, axis=1)).sum(axis=1)
    flipped = _flip(groups, mask)
    f1 = np.abs(np.diff(flipped, axis=1)).sum(axis=1)
    return (f1 > f0).sum() / len(groups), (f1 < f0).sum() / len(groups)


def rs_analysis(x):
    """RS analysis using groups of four horizontal pixels."""
    h, w = x.shape
    groups = x[:, : w // 4 * 4].reshape(-1, 4).astype(np.int32)
    inverted = groups ^ 1
    mask, neg_mask = [0, 1, 1, 0], [0, -1, -1, 0]
    rm, sm = _rs_counts(groups, mask)
    rn, sn = _rs_counts(groups, neg_mask)
    rm1, sm1 = _rs_counts(inverted, mask)
    rn1, sn1 = _rs_counts(inverted, neg_mask)
    d0, d_0 = rm - sm, rn - sn
    d1, d_1 = rm1 - sm1, rn1 - sn1
    a = 2 * (d1 + d0)
    b = d_0 - d_1 - d1 - 3 * d0
    c = d0 - d_0
    if abs(a) < 1e-12:
        root = -c / b if abs(b) > 1e-12 else float("nan")
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            root = float("nan")
        else:
            r1 = (-b + np.sqrt(disc)) / (2 * a)
            r2 = (-b - np.sqrt(disc)) / (2 * a)
            root = r1 if abs(r1) < abs(r2) else r2
    p = float(root / (root - 0.5)) if np.isfinite(root) and root != 0.5 else float("nan")
    return {"rs_Rm": rm, "rs_Sm": sm, "rs_R_minus_m": rn, "rs_S_minus_m": sn,
            "rs_p": p, "rs_est_bits": p * x.size if np.isfinite(p) else float("nan")}


def ws_analysis(x):
    """Weighted-stego estimator using a local predictor and variance weights."""
    xf = x.astype(np.float64)
    kernel = np.array([[-1, 2, -1], [2, 0, 2], [-1, 2, -1]], dtype=np.float64) / 4.0
    prediction = convolve2d(xf, kernel, mode="same", boundary="symm")
    variance = uniform_filter(xf ** 2, 3) - uniform_filter(xf, 3) ** 2
    weights = 1.0 / (5.0 + variance)
    flipped = (x ^ 1).astype(np.float64)
    inner = (slice(1, -1), slice(1, -1))
    weights, original, pred, flip = weights[inner], xf[inner], prediction[inner], flipped[inner]
    p = float(2.0 * np.sum(weights * (original - pred) * (original - flip)) / np.sum(weights))
    return {"ws_p": p, "ws_est_bits": p * x.size}


def lsb_tests(x):
    """Test LSB balance, adjacent-pixel correlation, and byte-level entropy."""
    lsb = x & 1
    n = lsb.size
    fraction = float(lsb.mean())
    z = (fraction - 0.5) * 2 * np.sqrt(n)
    p_mono = float(2 * stats.norm.sf(abs(z)))
    left = lsb[:, :-1].ravel().astype(float)
    right = lsb[:, 1:].ravel().astype(float)
    lag1 = float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else float("nan")
    packed = np.packbits(lsb.ravel())
    counts = np.bincount(packed, minlength=256) / len(packed)
    entropy = float(-np.sum(counts[counts > 0] * np.log2(counts[counts > 0])))
    return {"lsb_ones_fraction": fraction, "lsb_monobit_p": p_mono,
            "lsb_lag1_correlation": lag1, "lsb_byte_entropy": entropy}


def _upper_tail_p(observed, null_values):
    null_values = np.asarray(null_values, dtype=np.float64)
    return float((1 + np.count_nonzero(null_values >= observed)) / (len(null_values) + 1))


def _robust_peak_score(spectrum, exclude=3):
    a = np.asarray(spectrum, dtype=np.float64).copy()
    if a.ndim == 2:
        cy, cx = a.shape[0] // 2, a.shape[1] // 2
        a[max(0, cy - exclude):cy + exclude + 1,
          max(0, cx - exclude):cx + exclude + 1] = np.nan
    else:
        center = a.size // 2
        a[max(0, center - exclude):center + exclude + 1] = np.nan
    values = a[np.isfinite(a)]
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        scale = float(np.std(values)) + 1e-12
    return float((np.nanmax(a) - median) / scale)


def _fft_scores(change_map):
    cm = np.asarray(change_map, dtype=np.float64)
    h, w = cm.shape
    window2 = np.outer(np.hanning(h), np.hanning(w))
    spectrum2 = np.abs(np.fft.fftshift(np.fft.fft2((cm - cm.mean()) * window2)))
    score2 = _robust_peak_score(spectrum2)
    flat = cm.ravel()
    spectrum1 = np.abs(np.fft.fftshift(np.fft.fft((flat - flat.mean()) * np.hanning(flat.size))))
    score1 = _robust_peak_score(spectrum1)
    return score2, score1


def change_map_fft_analysis(changed, rng, n_perm=DEFAULT_PERMUTATIONS):
    """Compare FFT peaks in the change map with exact-count random maps."""
    changed = np.asarray(changed, dtype=bool)
    n_changed = int(changed.sum())
    if n_changed == 0:
        return {"fft2d_peak_score": float("nan"), "fft2d_p": float("nan"),
                "fft1d_peak_score": float("nan"), "fft1d_p": float("nan")}
    observed2, observed1 = _fft_scores(changed)
    null2 = np.empty(n_perm, dtype=np.float64)
    null1 = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        indices = rng.choice(changed.size, size=n_changed, replace=False)
        random_map = np.zeros(changed.size, dtype=np.uint8)
        random_map[indices] = 1
        null2[i], null1[i] = _fft_scores(random_map.reshape(changed.shape))
    mean2, mean1 = float(null2.mean()), float(null1.mean())
    return {
        "fft2d_peak_score": float(observed2), "fft2d_null_mean": mean2,
        "fft2d_p": _upper_tail_p(observed2, null2),
        "fft2d_peak_excess": float(observed2 / mean2) if mean2 > 0 else float("nan"),
        "fft1d_peak_score": float(observed1), "fft1d_null_mean": mean1,
        "fft1d_p": _upper_tail_p(observed1, null1),
        "fft1d_peak_excess": float(observed1 / mean1) if mean1 > 0 else float("nan"),
        "fft_permutations": int(n_perm),
    }


def block_structure_analysis(changed, rng, block_sizes=DEFAULT_BLOCK_SIZES,
                             n_perm=DEFAULT_PERMUTATIONS):
    """Test changed-pixel concentration across block sizes against random maps."""
    changed = np.asarray(changed, dtype=bool)
    h, w = changed.shape
    n_changed = int(changed.sum())
    results = {}
    if n_changed == 0:
        return results
    for block_size in block_sizes:
        gy = np.arange(h) // block_size
        gx = np.arange(w) // block_size
        ncols = int(gx.max()) + 1
        block_ids = (gy[:, None] * ncols + gx[None, :]).ravel()
        observed = np.bincount(block_ids[changed.ravel()], minlength=int(block_ids.max()) + 1)
        observed_max = float(observed.max())
        observed_var = float(observed.var())
        null_max = np.empty(n_perm, dtype=np.float64)
        null_var = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            indices = rng.choice(changed.size, size=n_changed, replace=False)
            counts = np.bincount(block_ids[indices], minlength=observed.size)
            null_max[i] = counts.max()
            null_var[i] = counts.var()
        prefix = f"block_{block_size}"
        results[f"{prefix}_max_occupancy"] = observed_max
        results[f"{prefix}_occupancy_variance"] = observed_var
        results[f"{prefix}_max_p"] = _upper_tail_p(observed_max, null_max)
        results[f"{prefix}_variance_p"] = _upper_tail_p(observed_var, null_var)
    return results


def _auc_low_cost(cost, changed):
    """AUC for predicting changed pixels from low HILL costs (0.5 is chance)."""
    flat = changed.ravel()
    positives = int(flat.sum())
    negatives = flat.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(-cost.ravel())
    return float((ranks[flat].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def known_cover_attacks(cover, stego, rng, n_perm=DEFAULT_PERMUTATIONS,
                        block_sizes=DEFAULT_BLOCK_SIZES):
    """Run only block-structure, FFT change-map, and HILL selection-channel tests."""
    if cover.shape != stego.shape:
        raise ValueError("Cover and stego images must have identical dimensions.")
    changed = stego != cover
    results = {
        "n_changed_pixels": int(changed.sum()),
        "change_rate": float(changed.mean()),
    }
    if changed.any():
        results.update(block_structure_analysis(changed, rng, block_sizes, n_perm))
        results.update(change_map_fft_analysis(changed, rng, n_perm))
    results["hill_auc_cover"] = _auc_low_cost(HILL(cover), changed)
    results["hill_auc_stego"] = _auc_low_cost(HILL(stego), changed)
    return results


def find_cover(stego_path, cover_dir):
    if not cover_dir:
        return None
    stem = os.path.splitext(os.path.basename(stego_path))[0]
    base = stem[6:] if stem.startswith("stego_") else stem
    for ext in (".png", ".bmp", ".pgm", ".tif", ".tiff", ".jpg", ".jpeg"):
        candidate = os.path.join(cover_dir, base + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def analyze(stego_path, cover_dir, n_perm=DEFAULT_PERMUTATIONS,
            block_sizes=DEFAULT_BLOCK_SIZES, seed=0):
    rng = np.random.default_rng(seed)
    stego = load_gray(stego_path)
    row = {"image": os.path.basename(stego_path), "height": stego.shape[0], "width": stego.shape[1]}

    # Blind attacks: stego image only.
    row.update(chi_square_attack(stego))
    row.update(rs_analysis(stego))
    row.update(ws_analysis(stego))
    row.update(lsb_tests(stego))

    # Known-cover attacks: original cover and stego image.
    cover_path = find_cover(stego_path, cover_dir)
    row["cover_found"] = cover_path is not None
    if cover_path:
        cover = load_gray(cover_path)
        if cover.shape == stego.shape:
            row.update(known_cover_attacks(cover, stego, rng, n_perm, block_sizes))
        else:
            row["cover_found"] = False
            row["cover_note"] = "Cover dimensions do not match the stego image."
    return row


def detector_flags(row):
    """Apply heuristic flags only to the seven selected attack families."""
    flags = {
        "flag_chi_square": bool(row.get("chi2_p_embed", 0) > CHI_P_FLAG),
        "flag_rs": bool(np.isfinite(row.get("rs_p", np.nan)) and row["rs_p"] > RS_P_FLAG),
        "flag_weighted_stego": bool(np.isfinite(row.get("ws_p", np.nan)) and row["ws_p"] > WS_P_FLAG),
        "flag_lsb": bool(np.isfinite(row.get("lsb_monobit_p", np.nan)) and row["lsb_monobit_p"] < LSB_P_FLAG),
    }
    if row.get("cover_found"):
        flags["flag_hill_selection_channel"] = bool(row.get("hill_auc_cover", 0) > AUC_FLAG)
        fft_ps = [row[k] for k in ("fft2d_p", "fft1d_p") if k in row and np.isfinite(row[k])]
        flags["flag_fft_change_map"] = bool(fft_ps and min(fft_ps) < FFT_P_FLAG)
        block_ps = [v for k, v in row.items() if k.endswith("_max_p") or k.endswith("_variance_p")]
        # Bonferroni correction across all block-size tests.
        flags["flag_block_structure"] = bool(block_ps and min(block_ps) < (BLOCK_P_FLAG / len(block_ps)))
    return flags


def fmt(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.5g}"
    return str(value)


def write_report(rows, summary, path):
    lines = [
        "STEGANALYSIS REPORT (HILL + STC, grayscale PNG)",
        "=" * 64,
        f"Images analysed: {len(rows)}",
        "",
        "A. BLIND ATTACKS",
        "image | chi2_p_embed | rs_p | ws_p | lsb_monobit_p | lsb_lag1_correlation | lsb_byte_entropy",
    ]
    blind_columns = ["image", "chi2_p_embed", "rs_p", "ws_p", "lsb_monobit_p",
                     "lsb_lag1_correlation", "lsb_byte_entropy"]
    for row in rows:
        lines.append(" | ".join(fmt(row.get(k, "N/A")) for k in blind_columns))

    lines.extend(["", "B. KNOWN-COVER ATTACKS",
                  "image | change_rate | hill_auc_cover | hill_auc_stego | fft2d_p | fft1d_p | block-test p-values"])
    for row in rows:
        if not row.get("cover_found"):
            continue
        block_values = [f"{k}={fmt(v)}" for k, v in row.items()
                        if k.startswith("block_") and (k.endswith("_max_p") or k.endswith("_variance_p"))]
        values = [row.get("image"), row.get("change_rate"), row.get("hill_auc_cover"),
                  row.get("hill_auc_stego"), row.get("fft2d_p"), row.get("fft1d_p"),
                  "; ".join(block_values) if block_values else "N/A"]
        lines.append(" | ".join(fmt(v) for v in values))

    lines.extend(["", "C. HEURISTIC DETECTOR FLAGS"])
    flag_names = [k for k in rows[0] if k.startswith("flag_")] if rows else []
    lines.append("image | " + " | ".join(flag_names))
    for row in rows:
        flags = detector_flags(row)
        lines.append(row["image"] + " | " + " | ".join(str(flags.get(k, "N/A")) for k in flag_names))

    lines.extend(["", "D. SUMMARY ACROSS IMAGES"])
    for key, value in summary.items():
        lines.append(f"{key}: {fmt(value)}")

    lines.extend([
        "",
        "INTERPRETATION NOTES",
        "- Flags are heuristic indicators, not proof of successful detection.",
        "- FFT and block tests compare against uniformly randomized change maps.",
        "- HILL AUC measures how well low-cost locations are associated with changed pixels; 0.5 is chance-level.",
        "- Blind RS and weighted-stego tests are primarily LSB-oriented and are indicative only for +/-1 embedding.",
    ])
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the selected blind and known-cover steganalysis attacks.")
    parser.add_argument("--stego", default="Stego", help="folder containing stego PNG images")
    parser.add_argument("--cover", default="images", help="folder containing original cover images; use '' to skip known-cover attacks")
    parser.add_argument("--out", default="steganalysis_res", help="output folder")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS,
                        help="randomized maps for FFT and block tests (minimum 20)")
    parser.add_argument("--block-sizes", default="8,16,32,64", help="comma-separated block sizes")
    args = parser.parse_args()

    if args.permutations < 20:
        parser.error("--permutations must be at least 20")
    try:
        block_sizes = tuple(sorted({int(v) for v in args.block_sizes.split(",") if v.strip()}))
    except ValueError:
        parser.error("--block-sizes must contain comma-separated integers")
    if not block_sizes or any(size <= 0 for size in block_sizes):
        parser.error("--block-sizes must contain positive integers")

    os.makedirs(args.out, exist_ok=True)
    stego_files = sorted(glob.glob(os.path.join(args.stego, "*.png")))
    if not stego_files:
        parser.error(f"No PNG files found in {args.stego}")

    rows = []
    for index, path in enumerate(stego_files):
        print(f"Analysing {path}")
        row = analyze(path, args.cover, args.permutations, block_sizes, seed=42 + index)
        row.update(detector_flags(row))
        rows.append(row)

    summary = {}
    for flag in [key for key in rows[0] if key.startswith("flag_")]:
        values = [row[flag] for row in rows if flag in row]
        summary[f"{flag}_rate"] = float(np.mean(values)) if values else float("nan")

    csv_path = os.path.join(args.out, "attack_results.csv")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.out, "attack_results.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"per_image": rows, "summary": summary,
                   "configuration": {"permutations": args.permutations, "block_sizes": block_sizes}},
                  file, indent=2, default=lambda value: value.item() if isinstance(value, (np.floating, np.integer, np.bool_)) else str(value))

    write_report(rows, summary, os.path.join(args.out, "attack_report.txt"))
    print(f"Results written to {args.out}/")


if __name__ == "__main__":
    main()
