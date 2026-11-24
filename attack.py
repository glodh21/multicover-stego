"""Steganalysis / attack module for the HILL + STC stego images.

Run:
    python stego_attacks.py --stego Stego --cover images --out results --params data.txt --spam

Outputs (in --out):
    attack_results.csv   one row per stego image (all metrics)
    attack_results.json  same, plus summary + notes
    attack_report.txt    human-readable report
    spam686/<name>.npy   (only with --spam) SPAM-686 features for later classifier use

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


def known_cover_attacks(cover, stego, rng):
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

    # periodicity / block structure of the change map
    cm = ch.astype(np.float64)
    if n_ch > 0:
        o2, n2, loc2 = _peak_ratio(cm, rng)
        o1, n1, loc1 = _peak_ratio(cm.ravel(), rng)
        out.update({"fft2d_peak_ratio": o2, "fft2d_null_ratio": n2,
                    "fft2d_peak_excess": o2 / n2, "fft2d_peak_loc": list(map(int, loc2)),
                    "fft1d_raster_peak_ratio": o1, "fft1d_raster_null_ratio": n1,
                    "fft1d_raster_peak_excess": o1 / n1, "fft1d_raster_peak_bin": int(loc1[0]),
                    "fft1d_raster_peak_period_px": float(N / max(loc1[0], 1))})
        # last-block / tail artifact and density uniformity along raster order
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


def analyze(stego_path, cover_dir, spam_dir, seed=0):
    rng = np.random.default_rng(seed)
    x = load_gray(stego_path)
    row = {"image": os.path.basename(stego_path), "height": x.shape[0], "width": x.shape[1]}
    row.update(chi_square_attack(x))
    row.update(rs_analysis(x))
    row.update(ws_analysis(x))
    row.update(hcf_attack(x))
    row.update(lsb_tests(x))
    row.update(blind_selection_channel(x))

    cp = find_cover(stego_path, cover_dir)
    row["cover_found"] = cp is not None
    if cp is not None:
        cover = load_gray(cp)
        if cover.shape == x.shape:
            row.update(known_cover_attacks(cover, x, rng))
        else:
            row["cover_found"] = False

    if spam_dir:
        os.makedirs(spam_dir, exist_ok=True)
        np.save(os.path.join(spam_dir, os.path.splitext(row["image"])[0] + ".npy"), spam686(x))

    row["flag_chi2"] = bool(row["chi2_p_embed"] > CHI_P_FLAG)
    row["flag_rs"] = bool(np.isfinite(row["rs_p"]) and row["rs_p"] > RS_P_FLAG)
    row["flag_ws"] = bool(row["ws_p"] > WS_P_FLAG)
    if "fft2d_peak_excess" in row:
        row["flag_periodicity"] = bool(max(row["fft2d_peak_excess"], row["fft1d_raster_peak_excess"]) > PEAK_RATIO_FLAG)
        row["flag_nonuniform_density"] = bool(row["segment_uniformity_p"] < SEG_P_FLAG)
        row["flag_selection_channel"] = bool(row["auc_cost_stego"] > AUC_FLAG)
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
               "fft2d_peak_excess", "fft1d_raster_peak_excess", "segment_uniformity_p",
               "tail_rate_over_mean", "auc_cost_cover", "auc_cost_stego"]
    L.append("B. KNOWN-COVER ATTACKS")
    L.append(" | ".join(cols_kc))
    for r in rows:
        if r.get("cover_found"):
            L.append(" | ".join(fmt(r.get(c, "")) for c in cols_kc))
    L.append("")
    L.append("C. DETECTOR FLAGS (True = attack succeeds)")
    fl = [c for c in rows[0] if c.startswith("flag_")]
    L.append(" | ".join(["image"] + fl))
    for r in rows:
        L.append(" | ".join([r["image"]] + [fmt(r.get(c, "n/a")) for c in fl]))
    L.append("")
    L.append("D. SUMMARY ACROSS IMAGES")
    for k, v in summary.items():
        L.append(f"  {k}: {fmt(v)}")
    if params:
        L.append("")
        L.append("E. PARAMETER FILE LEAKAGE")
        L.append(f"  plaintext params file exposes: {params}")
    L.append("")
    L.append("INTERPRETATION NOTES")
    L.append("  chi2_p_embed : ~1 => PoV equalisation (LSB-replacement style). Weak vs +-1 matching.")
    L.append("  rs_p / ws_p  : estimated relative payload (fraction of pixels). Designed for LSB replacement;")
    L.append("                 biased for +-1 embedding, so treat as indicative.")
    L.append("  hcf_ratio    : calibrated HCF-COM; <1 hints embedding.")
    L.append("  fft*_excess  : observed FFT peak/median divided by a random-permutation null; ~1 => no periodicity.")
    L.append("  segment_uniformity_p : chi-square of change counts over 10 raster segments (tail/last-block check).")
    L.append("  auc_cost_*   : AUC of 'low HILL cost => changed'. 0.5 = unpredictable; >0.5 = selection channel leaks.")
    L.append("  sign_balance_p : binomial test of +1/-1 flip balance; small => biased signs.")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stego", default="Stego")
    ap.add_argument("--cover", default="images", help="folder with original covers (known-cover attacks); '' to skip")
    ap.add_argument("--out", default="results")
    ap.add_argument("--params", default="data.txt")
    ap.add_argument("--spam", action="store_true", help="also dump SPAM-686 features")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.stego, "*.png")))
    if not files:
        raise SystemExit(f"No PNG files in {args.stego}")

    spam_dir = os.path.join(args.out, "spam686") if args.spam else None
    rows = []
    for f in files:
        print(f"analysing {f}")
        rows.append(analyze(f, args.cover, spam_dir))

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

    json.dump({"per_image": rows, "summary": summary, "params_file": params},
              open(os.path.join(args.out, "attack_results.json"), "w"), indent=2, default=conv)
    write_report(rows, summary, params, os.path.join(args.out, "attack_report.txt"))
    print(f"results written to {args.out}/")


if __name__ == "__main__":
    main()