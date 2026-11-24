"""Steganalysis / attack module for the HILL + STC stego images.

Run:
    python stego_attacks_extended.py --stego Stego --cover images --out results --params data.txt

Optional STC sidecars / calibration:
    --syndrome-dir stc_syndromes
    --payload-calibration payload_calibration.csv
    --permutations 200 --block-sizes 8,16,32,64

The syndrome test requires actual syndrome bit arrays exported by the STC
implementation. They cannot be reconstructed reliably from cover/stego PNGs.
Payload-length estimation requires independent calibration runs.

Outputs (in --out):
    attack_results.csv   one row per stego image (all metrics)
    attack_results.json  same, plus summary + notes
    attack_report.txt    human-readable report
    spam686/<name>.npy   (only with --spam) SPAM-686 features for later classifier use

Added evaluations:
    - Change-map FFT with permutation p-values
    - HILL selection-channel AUC (existing; reported explicitly)
    - Block-grid concentration diagnostics at configurable block sizes
    - Syndrome distribution (only when an STC syndrome .npy sidecar is supplied)
    - Payload-length estimate (only when an independent calibration CSV is supplied)

Two attacker models:
  BLIND      : uses the stego image only (chi-square, RS, WS, HCF, LSB tests, HILL costs from stego)
  KNOWN-COVER: also uses the original cover (--cover); the covers must be named <base>.<ext>
               for stego files named stego_<base>.png

Detector-flag thresholds are at the top; they are conventional values, not tuned.
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

from image_percent import HILL   # same cost function the embedder uses (Kerckhoffs)

# ---------------- flag thresholds ----------------
CHI_P_FLAG = 0.95       # chi-square p(embedding) above this -> flagged
RS_P_FLAG = 0.05        # RS estimated relative length above this -> flagged
WS_P_FLAG = 0.05
PEAK_RATIO_FLAG = 1.5   # (observed FFT peak/median) / (random-permutation null) above this -> periodic
AUC_FLAG = 0.60         # cost-map AUC above this -> selection channel predictable
SEG_P_FLAG = 0.01       # change-uniformity over raster segments below this -> non-uniform


# ---------------- added evaluation settings ----------------
DEFAULT_PERMUTATIONS = 200
DEFAULT_BLOCK_SIZES = (8, 16, 32, 64)
BLOCK_P_FLAG = 0.01


def _upper_tail_p(observed, null_values):
    null_values = np.asarray(null_values, dtype=np.float64)
    return float((1 + np.count_nonzero(null_values >= observed)) / (len(null_values) + 1))


def _robust_peak_score(spectrum, exclude=3):
    """Robust standardized maximum spectral magnitude, excluding the centered DC region."""
    a = np.asarray(spectrum, dtype=np.float64).copy()
    if a.ndim == 2:
        cy, cx = a.shape[0] // 2, a.shape[1] // 2
        a[max(0, cy - exclude):cy + exclude + 1,
          max(0, cx - exclude):cx + exclude + 1] = np.nan
    else:
        c = a.size // 2
        a[max(0, c - exclude):c + exclude + 1] = np.nan
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return float("nan")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        scale = float(np.std(vals)) + 1e-12
    return float((np.nanmax(a) - med) / scale)


def _fft_scores(change_map):
    """2D and raster-order FFT peak scores using Hann windows."""
    cm = np.asarray(change_map, dtype=np.float64)
    h, w = cm.shape
    win2 = np.outer(np.hanning(h), np.hanning(w))
    a = (cm - cm.mean()) * win2
    f2 = np.abs(np.fft.fftshift(np.fft.fft2(a)))
    score2 = _robust_peak_score(f2, exclude=3)

    flat = cm.ravel()
    win1 = np.hanning(flat.size)
    a1 = (flat - flat.mean()) * win1
    f1 = np.abs(np.fft.fftshift(np.fft.fft(a1)))
    score1 = _robust_peak_score(f1, exclude=3)
    return score2, score1


def change_map_fft_analysis(changed, rng, n_perm=DEFAULT_PERMUTATIONS):
    """Change-map FFT compared with exact-count uniformly randomized maps.

    This is a spatial-pattern diagnostic, not proof of STC-specific leakage.
    A HILL-cost-conditioned null should be added for stronger attribution.
    """
    ch = np.asarray(changed, dtype=bool)
    n = int(ch.sum())
    if n == 0:
        return {"fft2d_peak_score": float("nan"), "fft2d_p": float("nan"),
                "fft1d_peak_score": float("nan"), "fft1d_p": float("nan"),
                "fft2d_peak_excess": float("nan"), "fft1d_raster_peak_excess": float("nan")}
    obs2, obs1 = _fft_scores(ch)
    flat_n = ch.size
    null2 = np.empty(n_perm, dtype=np.float64)
    null1 = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        idx = rng.choice(flat_n, size=n, replace=False)
        rnd = np.zeros(flat_n, dtype=np.uint8)
        rnd[idx] = 1
        r2, r1 = _fft_scores(rnd.reshape(ch.shape))
        null2[i], null1[i] = r2, r1
    mean2 = float(np.mean(null2))
    mean1 = float(np.mean(null1))
    return {
        "fft2d_peak_score": float(obs2),
        "fft2d_null_mean": mean2,
        "fft2d_null_99th": float(np.quantile(null2, 0.99)),
        "fft2d_p": _upper_tail_p(obs2, null2),
        "fft2d_peak_excess": float(obs2 / mean2) if mean2 > 0 else float("nan"),
        "fft1d_peak_score": float(obs1),
        "fft1d_null_mean": mean1,
        "fft1d_null_99th": float(np.quantile(null1, 0.99)),
        "fft1d_p": _upper_tail_p(obs1, null1),
        "fft1d_raster_peak_excess": float(obs1 / mean1) if mean1 > 0 else float("nan"),
        "fft_permutations": int(n_perm),
    }


def block_structure_analysis(changed, rng, block_sizes=DEFAULT_BLOCK_SIZES,
                             n_perm=DEFAULT_PERMUTATIONS):
    """Test block-grid concentration at several scales against exact-count uniform maps.

    This is a generic block-structure diagnostic. It is not an STC decoder and
    does not identify the STC constraint length unless that length is supplied
    and independently justified.
    """
    ch = np.asarray(changed, dtype=bool)
    h, w = ch.shape
    n = int(ch.sum())
    out = {}
    if n == 0:
        return out
    flat_ids_by_size = {}
    for b in block_sizes:
        gy = (np.arange(h) // b)
        gx = (np.arange(w) // b)
        ncols = int(gx.max()) + 1
        ids = (gy[:, None] * ncols + gx[None, :]).ravel()
        observed_counts = np.bincount(ids[ch.ravel()], minlength=(int(ids.max()) + 1))
        obs_max = float(observed_counts.max())
        obs_var = float(observed_counts.var())
        null_max = np.empty(n_perm, dtype=np.float64)
        null_var = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            idx = rng.choice(ch.size, size=n, replace=False)
            counts = np.bincount(ids[idx], minlength=observed_counts.size)
            null_max[i] = counts.max()
            null_var[i] = counts.var()
        key = f"block_{b}"
        out[f"{key}_max_occupancy"] = obs_max
        out[f"{key}_occupancy_variance"] = obs_var
        out[f"{key}_max_p"] = _upper_tail_p(obs_max, null_max)
        out[f"{key}_variance_p"] = _upper_tail_p(obs_var, null_var)
        out[f"{key}_null_max_99th"] = float(np.quantile(null_max, 0.99))
    return out


def syndrome_distribution_analysis(bits):
    """Descriptive tests on an actual STC syndrome bit-vector supplied as a sidecar."""
    b = np.asarray(bits).ravel()
    if b.size < 2 or not np.all(np.isin(b, [0, 1])):
        return {"syndrome_available": False, "syndrome_error": "Expected a 1-D array of at least two binary values."}
    b = b.astype(np.uint8)
    n = int(b.size)
    ones = int(b.sum())
    p1 = ones / n
    mono_z = (p1 - 0.5) * 2.0 * np.sqrt(n)
    mono_p = float(2 * stats.norm.sf(abs(mono_z)))
    runs = int(1 + np.count_nonzero(b[1:] != b[:-1]))
    n0, n1 = n - ones, ones
    if n0 and n1 and n > 1:
        runs_mean = 1 + (2 * n0 * n1) / n
        runs_var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
        runs_z = (runs - runs_mean) / np.sqrt(runs_var) if runs_var > 0 else float("nan")
        runs_p = float(2 * stats.norm.sf(abs(runs_z))) if np.isfinite(runs_z) else float("nan")
    else:
        runs_z, runs_p = float("nan"), float("nan")
    entropy = 0.0
    for prob in (p1, 1 - p1):
        if prob > 0:
            entropy -= prob * np.log2(prob)
    corr = float(np.corrcoef(b[:-1], b[1:])[0, 1]) if np.std(b[:-1]) > 0 and np.std(b[1:]) > 0 else float("nan")
    return {
        "syndrome_available": True,
        "syndrome_length_bits": n,
        "syndrome_ones_fraction": float(p1),
        "syndrome_binary_entropy": float(entropy),
        "syndrome_monobit_p": mono_p,
        "syndrome_runs": runs,
        "syndrome_runs_p": runs_p,
        "syndrome_lag1_corr": corr,
    }


def find_syndrome_file(stego_path, syndrome_dir):
    if not syndrome_dir:
        return None
    stem = os.path.splitext(os.path.basename(stego_path))[0]
    candidates = [os.path.join(syndrome_dir, stem + ".npy")]
    if stem.startswith("stego_"):
        candidates.append(os.path.join(syndrome_dir, stem[6:] + ".npy"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_payload_calibration(path):
    """Fit a simple independent calibration: payload_bits = intercept + slope*n_changes."""
    if not path:
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append((float(row["n_changes"]), float(row["payload_bits"])))
            except (KeyError, TypeError, ValueError):
                continue
    if len(rows) < 3 or len({x for x, _ in rows}) < 2:
        raise ValueError("Payload calibration needs >=3 rows and >=2 distinct n_changes values; columns: n_changes,payload_bits")
    x = np.asarray([r[0] for r in rows], dtype=np.float64)
    y = np.asarray([r[1] for r in rows], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"intercept": float(intercept), "slope": float(slope),
            "r2_training": float(r2), "n_calibration": len(rows)}


def estimate_payload_length(n_changes, calibration):
    if calibration is None:
        return {"payload_estimation_available": False}
    estimate = calibration["intercept"] + calibration["slope"] * n_changes
    return {
        "payload_estimation_available": True,
        "payload_estimated_bits": float(max(0.0, estimate)),
        "payload_calibration_r2_training": calibration["r2_training"],
        "payload_calibration_samples": calibration["n_calibration"],
    }


def load_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


# =====================================================================
# 1. BLIND ATTACKS (stego only)
# =====================================================================
def chi_square_attack(x):
    """Westfeld-Pfitzmann chi-square on pairs of values (2k, 2k+1).
    p_embed = 1 - CDF(chi2, df): close to 1 => histogram pairs equalised => stego."""
    h = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    h0, h1 = h[0::2], h[1::2]
    e = (h0 + h1) / 2.0
    m = e >= 5
    chi = float(np.sum((h0[m] - e[m]) ** 2 / e[m]))
    df = int(m.sum()) - 1
    p = float(stats.chi2.sf(chi, df)) if df > 0 else float("nan")
    return {"chi2": chi, "chi2_df": df, "chi2_p_embed": p}


def _flip(v, m):
    """Flipping functions of Fridrich et al.: F1: 0<->1,2<->3..; F-1: shifted; F0: identity."""
    out = v.copy()
    for col, mi in enumerate(m):
        if mi == 1:
            out[:, col] = v[:, col] ^ 1
        elif mi == -1:
            out[:, col] = ((v[:, col] + 1) ^ 1) - 1
    return out


def _rs_counts(g, mask):
    f0 = np.abs(np.diff(g, axis=1)).sum(axis=1)
    gf = _flip(g, mask)
    f1 = np.abs(np.diff(gf, axis=1)).sum(axis=1)
    n = len(g)
    return (f1 > f0).sum() / n, (f1 < f0).sum() / n   # R, S


def rs_analysis(x):
    """Fridrich-Goljan-Du RS analysis. Groups of 4 horizontal pixels, masks M=[0 1 1 0], -M.
    Returns estimated relative message length p (fraction of pixels carrying a bit)."""
    h, w = x.shape
    g = x[:, : w // 4 * 4].reshape(-1, 4).astype(np.int32)
    gi = g ^ 1                                      # image with all LSBs flipped
    M, Mn = [0, 1, 1, 0], [0, -1, -1, 0]
    rm, sm = _rs_counts(g, M)
    rn, sn = _rs_counts(g, Mn)
    rm1, sm1 = _rs_counts(gi, M)
    rn1, sn1 = _rs_counts(gi, Mn)
    d0, d_0 = rm - sm, rn - sn
    d1, d_1 = rm1 - sm1, rn1 - sn1
    a = 2 * (d1 + d0)
    b = d_0 - d_1 - d1 - 3 * d0
    c = d0 - d_0
    if abs(a) < 1e-12:
        xr = -c / b if abs(b) > 1e-12 else float("nan")
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            xr = float("nan")
        else:
            r1, r2 = (-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)
            xr = r1 if abs(r1) < abs(r2) else r2
    p = float(xr / (xr - 0.5)) if np.isfinite(xr) and xr != 0.5 else float("nan")
    return {"rs_Rm": rm, "rs_Sm": sm, "rs_R-m": rn, "rs_S-m": sn,
            "rs_p": p, "rs_est_bits": p * x.size if np.isfinite(p) else float("nan")}


def ws_analysis(x):
    """Weighted-stego quantitative estimator (Fridrich-Goljan, Ker-Bohme weights/predictor)."""
    xf = x.astype(np.float64)
    F = np.array([[-1, 2, -1], [2, 0, 2], [-1, 2, -1]], dtype=np.float64) / 4.0
    pred = convolve2d(xf, F, mode="same", boundary="symm")
    var = uniform_filter(xf ** 2, 3) - uniform_filter(xf, 3) ** 2
    w = 1.0 / (5.0 + var)
    flipped = (x ^ 1).astype(np.float64)
    sl = (slice(1, -1), slice(1, -1))
    w, xs, pr, fl = w[sl], xf[sl], pred[sl], flipped[sl]
    p = float(2.0 * np.sum(w * (xs - pr) * (xs - fl)) / np.sum(w))
    return {"ws_p": p, "ws_est_bits": p * x.size}


def hcf_com(x):
    h = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    H = np.abs(np.fft.fft(h))[1:129]
    k = np.arange(1, 129)
    return float(np.sum(k * H) / np.sum(H))


def hcf_attack(x):
    """Harmsen-Pearlman HCF centre of mass with Ker's calibration (2x2 average downsample).
    ratio < 1 suggests embedding (COM drops after adding noise)."""
    h, w = x.shape
    d = x[: h // 2 * 2, : w // 2 * 2].astype(np.float64)
    d = np.rint((d[0::2, 0::2] + d[1::2, 0::2] + d[0::2, 1::2] + d[1::2, 1::2]) / 4).astype(np.uint8)
    c1, c2 = hcf_com(x), hcf_com(d)
    return {"hcf_com": c1, "hcf_com_down": c2, "hcf_ratio": c1 / c2}


def lsb_tests(x):
    """Basic LSB-plane randomness tests (naive-extraction view of the payload channel)."""
    lsb = (x & 1)
    n = lsb.size
    frac = float(lsb.mean())
    z = (frac - 0.5) * 2 * np.sqrt(n)
    p_mono = float(2 * stats.norm.sf(abs(z)))
    a, b = lsb[:, :-1].ravel().astype(float), lsb[:, 1:].ravel().astype(float)
    lag1 = float(np.corrcoef(a, b)[0, 1])
    by = np.packbits(lsb.ravel())
    cnt = np.bincount(by, minlength=256) / len(by)
    ent = float(-np.sum(cnt[cnt > 0] * np.log2(cnt[cnt > 0])))
    return {"lsb_ones_frac": frac, "lsb_monobit_p": p_mono, "lsb_lag1_corr": lag1,
            "lsb_byte_entropy": ent}


def blind_selection_channel(x):
    """HILL costs recomputed from the stego itself (attacker knows the algorithm)."""
    cost = HILL(x)
    return {"stego_hill_cost_mean": float(cost.mean())}


# =====================================================================
# 2. SPAM-686 features (Pevny et al. 2010) - for later classifier use
# =====================================================================
def _trip(x, diag, T=3):
    x = x.astype(np.int32)
    if not diag:
        d = x[:, :-1] - x[:, 1:]
        a, b, c = d[:, :-2], d[:, 1:-1], d[:, 2:]
    else:
        d = x[:-1, :-1] - x[1:, 1:]
        a, b, c = d[:-2, :-2], d[1:-1, 1:-1], d[2:, 2:]
    q = lambda t: np.clip(t, -T, T) + T
    idx = (q(a) * (2 * T + 1) + q(b)) * (2 * T + 1) + q(c)
    h = np.bincount(idx.ravel(), minlength=(2 * T + 1) ** 3).astype(np.float64)
    return h / h.sum()


def spam686(x):
    r = x[::-1, ::-1]
    hv = np.mean([_trip(x, False), _trip(r, False), _trip(x.T, False), _trip(r.T, False)], axis=0)
    fl = x[:, ::-1]
    dg = np.mean([_trip(x, True), _trip(r, True), _trip(fl, True), _trip(fl[::-1, ::-1], True)], axis=0)
    return np.concatenate([hv, dg]).astype(np.float32)


# =====================================================================
# 3. KNOWN-COVER ATTACKS
# =====================================================================
def _peak_ratio(arr, rng):
    """FFT peak / median magnitude, and the same for a random permutation (null)."""
    def pr(a):
        F = np.abs(np.fft.rfftn(a - a.mean()))
        F.flat[0] = 0
        return float(F.max() / (np.median(F) + 1e-12)), np.unravel_index(F.argmax(), F.shape)
    obs, loc = pr(arr)
    null, _ = pr(rng.permutation(arr.ravel()).reshape(arr.shape))
    return obs, null, loc


def _auc_low_cost(cost, changed):
    """AUC of 'low cost => changed' (0.5 = selection channel not predictable)."""
    flat = changed.ravel()
    n1 = int(flat.sum())
    n0 = flat.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(-cost.ravel())
    return float((ranks[flat].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def known_cover_attacks(cover, stego, rng, n_perm=DEFAULT_PERMUTATIONS, block_sizes=DEFAULT_BLOCK_SIZES):
    out = {}
    diff = stego.astype(np.int32) - cover.astype(np.int32)
    ch = diff != 0
    N = diff.size
    n_ch = int(ch.sum())
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    out.update({
        "n_changes": n_ch,
        "change_rate": n_ch / N,
        "mse": mse,
        "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else float("inf"),
        "max_abs_diff": int(np.abs(diff).max()),
        "frac_abs_gt1": float((np.abs(diff) > 1).sum() / max(n_ch, 1)),
    })

    # sign balance of +1/-1 flips
    npl, nmi = int((diff == 1).sum()), int((diff == -1).sum())
    out["n_plus1"], out["n_minus1"] = npl, nmi
    out["sign_balance_plus_frac"] = npl / max(npl + nmi, 1)
    out["sign_balance_p"] = float(stats.binomtest(npl, npl + nmi, 0.5).pvalue) if npl + nmi else float("nan")

    # parity of flips (cover LSB -> stego LSB): +1 on even cover / -1 on odd cover is the LSB-replacement signature
    even = (cover % 2 == 0)
    out["frac_plus_on_even_cover"] = float(((diff == 1) & even).sum() / max(npl, 1))
    out["frac_minus_on_odd_cover"] = float(((diff == -1) & ~even).sum() / max(nmi, 1))

    # Change-map FFT and block-grid structure diagnostics.
    if n_ch > 0:
        out.update(change_map_fft_analysis(ch, rng, n_perm=n_perm))
        out.update(block_structure_analysis(ch, rng, block_sizes=block_sizes, n_perm=n_perm))
        # Legacy raster segmentation / tail diagnostic (exploratory; sparse-count caveat applies).
        seg = np.array_split(ch.ravel(), 10)
        cnt = np.array([s.sum() for s in seg], dtype=np.float64)
        sizes = np.array([len(s) for s in seg], dtype=np.float64)
        exp = n_ch * sizes / sizes.sum()
        out["segment_change_rates"] = [float(c / s) for c, s in zip(cnt, sizes)]
        out["segment_uniformity_p"] = float(stats.chisquare(cnt, exp).pvalue)
        out["tail_rate_over_mean"] = float(cnt[-1] / sizes[-1] / (n_ch / N))

    # selection-channel predictability (maxSRM-style attacker recomputing HILL)
    out["auc_cost_cover"] = _auc_low_cost(HILL(cover), ch)
    out["auc_cost_stego"] = _auc_low_cost(HILL(stego), ch)
    return out


# =====================================================================
# Driver
# =====================================================================
def find_cover(stego_path, cover_dir):
    if not cover_dir:
        return None
    name = os.path.splitext(os.path.basename(stego_path))[0]
    base = name[6:] if name.startswith("stego_") else name
    for ext in (".png", ".bmp", ".pgm", ".tif", ".tiff", ".jpg", ".jpeg"):
        p = os.path.join(cover_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def analyze(stego_path, cover_dir, spam_dir, syndrome_dir=None, payload_calibration=None, n_perm=DEFAULT_PERMUTATIONS, block_sizes=DEFAULT_BLOCK_SIZES, seed=0):
    rng = np.random.default_rng(seed)
    x = load_gray(stego_path)
    row = {"image": os.path.basename(stego_path), "height": x.shape[0], "width": x.shape[1]}
    row.update(chi_square_attack(x))
    row.update(rs_analysis(x))
    row.update(ws_analysis(x))
    row.update(hcf_attack(x))
    row.update(lsb_tests(x))
    row.update(blind_selection_channel(x))

    # Syndrome distribution is only valid when the actual STC syndrome is exported.
    syndrome_path = find_syndrome_file(stego_path, syndrome_dir)
    if syndrome_path:
        try:
            row.update(syndrome_distribution_analysis(np.load(syndrome_path, allow_pickle=False)))
            row["syndrome_file"] = os.path.basename(syndrome_path)
        except Exception as exc:
            row.update({"syndrome_available": False, "syndrome_error": str(exc)})
    else:
        row["syndrome_available"] = False
        row["syndrome_note"] = "No STC syndrome sidecar supplied; cannot infer syndrome from PNG."

    cp = find_cover(stego_path, cover_dir)
    row["cover_found"] = cp is not None
    if cp is not None:
        cover = load_gray(cp)
        if cover.shape == x.shape:
            row.update(known_cover_attacks(cover, x, rng, n_perm=n_perm, block_sizes=block_sizes))
            row.update(estimate_payload_length(row.get("n_changes", 0), payload_calibration))
        else:
            row["cover_found"] = False

    if spam_dir:
        os.makedirs(spam_dir, exist_ok=True)
        np.save(os.path.join(spam_dir, os.path.splitext(row["image"])[0] + ".npy"), spam686(x))

    row["flag_chi2"] = bool(row["chi2_p_embed"] > CHI_P_FLAG)
    row["flag_rs"] = bool(np.isfinite(row["rs_p"]) and row["rs_p"] > RS_P_FLAG)
    row["flag_ws"] = bool(row["ws_p"] > WS_P_FLAG)
    if "fft2d_peak_excess" in row:
        row["flag_periodicity"] = bool(min(row.get("fft2d_p", 1.0), row.get("fft1d_p", 1.0)) < 0.005)
        row["flag_nonuniform_density"] = bool(row["segment_uniformity_p"] < SEG_P_FLAG)
        row["flag_selection_channel"] = bool(row["auc_cost_stego"] > AUC_FLAG)
        block_ps = [v for k, v in row.items() if k.endswith("_max_p") or k.endswith("_variance_p")]
        row["flag_block_structure"] = bool(block_ps and min(block_ps) < (BLOCK_P_FLAG / len(block_ps)))
    return row


def read_params(path):
    if not path or not os.path.exists(path):
        return None
    d = {}
    for line in open(path, encoding="utf-8"):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            d[k] = v
    return d


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_report(rows, summary, params, path):
    L = []
    L.append("STEGANALYSIS REPORT (HILL + STC, spatial domain, grayscale PNG)")
    L.append("=" * 72)
    L.append(f"images analysed : {len(rows)}")
    L.append("")
    cols_blind = ["image", "chi2_p_embed", "rs_p", "ws_p", "hcf_ratio", "lsb_monobit_p", "lsb_lag1_corr"]
    L.append("A. BLIND ATTACKS")
    L.append(" | ".join(cols_blind))
    for r in rows:
        L.append(" | ".join(fmt(r.get(c, "")) for c in cols_blind))
    L.append("")
    cols_kc = ["image", "change_rate", "psnr_db", "n_plus1", "n_minus1", "sign_balance_p",
               "fft2d_peak_excess", "fft2d_p", "fft1d_raster_peak_excess", "fft1d_p",
               "segment_uniformity_p", "tail_rate_over_mean", "auc_cost_cover", "auc_cost_stego"]
    L.append("B. KNOWN-COVER / EMBEDDING-PATTERN ANALYSIS")
    L.append(" | ".join(cols_kc))
    for r in rows:
        if r.get("cover_found"):
            L.append(" | ".join(fmt(r.get(c, "")) for c in cols_kc))
    L.append("")
    L.append("C. BLOCK-STRUCTURE DIAGNOSTICS")
    block_cols = sorted({k for r in rows for k in r if k.startswith("block_")})
    if block_cols:
        L.append("image | " + " | ".join(block_cols))
        for r in rows:
            if r.get("cover_found"):
                L.append(r["image"] + " | " + " | ".join(fmt(r.get(k, "")) for k in block_cols))
    else:
        L.append("No known-cover block diagnostics available.")
    L.append("")
    L.append("D. SYNDROME DISTRIBUTION")
    for r in rows:
        if r.get("syndrome_available"):
            L.append(f"{r['image']}: length={r.get('syndrome_length_bits')}, "
                     f"entropy={fmt(r.get('syndrome_binary_entropy'))}, "
                     f"monobit_p={fmt(r.get('syndrome_monobit_p'))}, "
                     f"runs_p={fmt(r.get('syndrome_runs_p'))}, "
                     f"lag1={fmt(r.get('syndrome_lag1_corr'))}")
        else:
            L.append(f"{r['image']}: NOT RUN ({r.get('syndrome_note', r.get('syndrome_error', 'no sidecar'))})")
    L.append("")
    L.append("E. PAYLOAD-LENGTH ESTIMATION")
    for r in rows:
        if r.get("cover_found"):
            if r.get("payload_estimation_available"):
                L.append(f"{r['image']}: estimated_bits={fmt(r.get('payload_estimated_bits'))}, "
                         f"calibration_R2={fmt(r.get('payload_calibration_r2_training'))}, "
                         f"calibration_samples={r.get('payload_calibration_samples')}")
            else:
                L.append(f"{r['image']}: NOT RUN (independent calibration CSV not supplied)")
    L.append("")
    L.append("F. DETECTOR FLAGS (heuristic flags; not proof of successful detection)")
    fl = [c for c in rows[0] if c.startswith("flag_")]
    L.append(" | ".join(["image"] + fl))
    for r in rows:
        L.append(" | ".join([r["image"]] + [fmt(r.get(c, "n/a")) for c in fl]))
    L.append("")
    L.append("G. SUMMARY ACROSS IMAGES")
    for k, v in summary.items():
        L.append(f"  {k}: {fmt(v)}")
    if params:
        L.append("")
        L.append("H. PARAMETER FILE LEAKAGE")
        L.append(f"  plaintext params file exposes: {params}")
    L.append("")
    L.append("INTERPRETATION NOTES")
    L.append("  FFT p-values use exact-count uniformly randomized change maps; this is NOT HILL-cost-conditioned.")
    L.append("  Block tests compare occupancy with uniform exact-count random maps; not proof of STC block leakage.")
    L.append("  Selection-channel AUC measures association between low HILL cost and changed locations.")
    L.append("  Syndrome tests require the actual STC syndrome bits exported by the encoder/decoder.")
    L.append("  Payload estimates require independent, representative calibration runs; report calibration error on held-out data.")
    L.append("  RS/WS are primarily LSB-oriented and are only indicative for +/-1 embedding.")
    L.append("  Sparse changes make ordinary segment chi-square approximations potentially unreliable.")
    open(path, "w", encoding="utf-8").write("\\n".join(L) + "\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stego", default="Stego")
    ap.add_argument("--cover", default="images", help="folder with original covers (known-cover attacks); '' to skip")
    ap.add_argument("--out", default="steganalysis_res")
    ap.add_argument("--params", default="data.txt")
    ap.add_argument("--spam", action="store_true", help="also dump SPAM-686 features")
    ap.add_argument("--syndrome-dir", default=None,
                    help="optional folder of actual STC syndrome .npy files, named after stego image")
    ap.add_argument("--payload-calibration", default=None,
                    help="CSV with independent calibration columns n_changes,payload_bits")
    ap.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS,
                    help="number of exact-count random maps for FFT and block tests")
    ap.add_argument("--block-sizes", default="8,16,32,64",
                    help="comma-separated block sizes in pixels")
    args = ap.parse_args()

    if args.permutations < 20:
        raise SystemExit("--permutations should be at least 20; 200+ is recommended.")
    try:
        block_sizes = tuple(sorted({int(v) for v in args.block_sizes.split(",") if v.strip()}))
    except ValueError:
        raise SystemExit("--block-sizes must be comma-separated positive integers.")
    if not block_sizes or any(v <= 0 for v in block_sizes):
        raise SystemExit("--block-sizes must contain positive integers.")

    calibration = load_payload_calibration(args.payload_calibration) if args.payload_calibration else None

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.stego, "*.png")))
    if not files:
        raise SystemExit(f"No PNG files in {args.stego}")

    spam_dir = os.path.join(args.out, "spam686") if args.spam else None
    rows = []
    for i, f in enumerate(files):
        print(f"analysing {f}")
        rows.append(analyze(
            f, args.cover, spam_dir,
            syndrome_dir=args.syndrome_dir,
            payload_calibration=calibration,
            n_perm=args.permutations,
            block_sizes=block_sizes,
            seed=42 + i
        ))

    kc = [r for r in rows if r.get("cover_found")]
    summary = {}
    if kc:
        cr = np.array([r["change_rate"] for r in kc])
        summary.update({"n_with_cover": len(kc), "change_rate_mean": float(cr.mean()),
                        "change_rate_std": float(cr.std()),
                        "change_rate_cv": float(cr.std() / cr.mean()) if cr.mean() else float("nan")})
    for fl in [c for c in rows[0] if c.startswith("flag_")]:
        vals = [r[fl] for r in rows if fl in r]
        summary[f"{fl}_rate"] = float(np.mean(vals)) if vals else float("nan")

    params = read_params(args.params)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(args.out, "attack_results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))

    with open(os.path.join(args.out, "attack_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"per_image": rows, "summary": summary, "params_file": params,
                   "configuration": {"permutations": args.permutations, "block_sizes": block_sizes,
                                     "syndrome_dir": args.syndrome_dir,
                                     "payload_calibration": args.payload_calibration}},
                  fh, indent=2, default=conv)
    write_report(rows, summary, params, os.path.join(args.out, "attack_report.txt"))
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()