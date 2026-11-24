#!/usr/bin/env python3
"""
Benchmark harness for the HILL + STC multi-image stego pipeline (main.py).

Sweeps   text size  x  number of images  x  repeats
and logs per-stage timings into two tidy ("long") CSV files:

    results/runs.csv    one row per run   (stage timings, fit info, utilisation)
    results/images.csv  one row per image per run (embed time, capacity, payload)

Both share `run_id`, so they can be joined in pandas.

Example
-------
    python benchmark.py                                   # defaults
    python benchmark.py --sizes 100 1000 10000 --counts 1 2 3 5 10 --repeats 5
    python benchmark.py --counts $(seq 1 15) --image-dir images
    python benchmark.py --text-files secret_small.txt secret_large.txt

If fewer distinct images exist than the requested count, the image pool is
cycled (img0, img1, ..., img0, img1, ...). Timing stays valid; just remember
that cycled images repeat the same cover content.
"""
import argparse
import csv
import glob
import os
import platform
import random
import secrets
import shutil
import string
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime

from PIL import Image
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from image_percent import HillCapacityCalculator
from key_exchange import encrypt_aes_key
import main as pipeline  # reuses your real functions, so timings reflect the real code

RUN_FIELDS = [
    "run_id", "timestamp", "text_bytes", "text_source", "n_images", "repeat",
    "target_bpp", "floor_fraction", "status",
    "t_capacity_s", "t_keygen_wrap_s", "t_header_s", "t_split_encrypt_s",
    "t_embed_s", "t_total_s",
    "n_fit", "n_skipped", "all_fit",
    "total_capacity_bits", "total_payload_bits", "text_bits", "utilization",
    "overhead_bits", "seq_bits", "len_bits", "capacity_pct_sum",
    "python", "platform", "cpu_count",
]

IMAGE_FIELDS = [
    "run_id", "text_bytes", "n_images", "repeat", "position", "image",
    "width", "height", "pixels", "max_capacity_bits", "capacity_percentage",
    "chunk_chars", "chunk_len_bytes", "encrypted_len_bits", "fits",
    "embedded", "t_embed_s",
]

ALPHABET = string.ascii_letters + string.digits + string.punctuation + " "


def make_text(n_bytes: int, seed: int = 1234) -> str:
    """Deterministic ASCII text of exactly n_bytes bytes (1 char = 1 byte)."""
    rng = random.Random(seed + n_bytes)
    return "".join(rng.choices(ALPHABET, k=n_bytes))


def pick_images(pool: list[str], n: int) -> list[str]:
    return [pool[i % len(pool)] for i in range(n)]


class CsvLog:
    """Append-only CSV writer that flushes every row (partial runs survive)."""

    def __init__(self, path, fields):
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        if new:
            self.w.writeheader()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


_dim_cache: dict[str, tuple[int, int]] = {}


def image_dims(path):
    if path not in _dim_cache:
        with Image.open(path) as im:
            _dim_cache[path] = im.size
    return _dim_cache[path]


def run_once(text, images, target_bpp, floor_fraction, pubkey, keep_dir=None):
    """Run the full pipeline once. Returns (timings/info dict, [per-image rows])."""
    t = {}
    info = {"status": "ok"}
    image_rows = []
    tmp = tempfile.mkdtemp(prefix="stegobench_")
    try:
        t_start = time.perf_counter()

        # 1) HILL cost maps + capacity per image
        s = time.perf_counter()
        calc = HillCapacityCalculator(target_bpp=target_bpp,
                                      floor_fraction=floor_fraction)
        capacities = calc.calculate_image_capacities(images)
        t["t_capacity_s"] = time.perf_counter() - s

        capacities.sort(key=lambda c: c["max_capacity_bits"])

        # 2) AES key + RSA envelope
        s = time.perf_counter()
        key = AESGCM.generate_key(bit_length=256)
        encrypt_aes_key(key, public_key_path=pubkey,
                        envelope_path=os.path.join(tmp, "key_envelope.json"))
        t["t_keygen_wrap_s"] = time.perf_counter() - s

        # 3) header widths
        s = time.perf_counter()
        seq_bits, len_bits = pipeline.header_field_sizes(capacities)
        t["t_header_s"] = time.perf_counter() - s

        # 4) split + encrypt
        s = time.perf_counter()
        chunks = pipeline.split_and_encrypt(text, capacities, seq_bits, len_bits, key)
        t["t_split_encrypt_s"] = time.perf_counter() - s

        # 5) embed (timed per image, one call per chunk -> same total work)
        seed = secrets.randbelow(2 ** 31)
        out_dir = keep_dir or os.path.join(tmp, "Stego")
        embed_total = 0.0
        for pos, c in enumerate(chunks):
            s = time.perf_counter()
            res = pipeline.embed_chunks([c], seed, out_dir)
            dt = time.perf_counter() - s
            embed_total += dt
            w, h = image_dims(c["path"])
            cap_pct = capacities[pos].get("capacity_percentage")
            image_rows.append({
                "position": pos,
                "image": os.path.basename(c["path"]),
                "width": w, "height": h, "pixels": w * h,
                "max_capacity_bits": c["max_capacity_bits"],
                "capacity_percentage": cap_pct,
                "chunk_chars": len(c["chunk"]),
                "chunk_len_bytes": c["chunk_len_bytes"],
                "encrypted_len_bits": c["encrypted_len_bits"],
                "fits": c["fits"],
                "embedded": res[0]["stego_path"] is not None,
                "t_embed_s": dt,
            })
        t["t_embed_s"] = embed_total
        t["t_total_s"] = time.perf_counter() - t_start

        n_fit = sum(1 for c in chunks if c["fits"])
        total_cap = sum(c["max_capacity_bits"] for c in chunks)
        total_payload = sum(c["encrypted_len_bits"] for c in chunks)
        text_bits = len(text.encode("utf-8")) * 8
        info.update({
            "n_fit": n_fit,
            "n_skipped": len(chunks) - n_fit,
            "all_fit": n_fit == len(chunks),
            "total_capacity_bits": total_cap,
            "total_payload_bits": total_payload,
            "text_bits": text_bits,
            "utilization": total_payload / total_cap if total_cap else None,
            "overhead_bits": total_payload - text_bits,
            "seq_bits": seq_bits,
            "len_bits": len_bits,
            "capacity_pct_sum": sum((c.get("capacity_percentage") or 0)
                                    for c in capacities),
        })
    except Exception as e:  # keep sweeping even if one config blows up
        info["status"] = f"error:{type(e).__name__}:{e}"[:200]
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {**t, **info}, image_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100, 500, 1_000, 5_000, 10_000, 50_000],
                    help="synthetic text sizes in bytes")
    ap.add_argument("--text-files", nargs="*", default=[],
                    help="real text files to benchmark in addition to --sizes")
    ap.add_argument("--counts", type=int, nargs="+", default=list(range(1, 11)),
                    help="number of images to spread the text across")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--image-dir", default="images")
    ap.add_argument("--image-glob", default="*.png")
    ap.add_argument("--target-bpp", type=float, default=0.4)
    ap.add_argument("--floor-fraction", type=float, default=0.0)
    ap.add_argument("--pubkey", default="keys/receiver_public.pem")
    ap.add_argument("--out", default="results")
    ap.add_argument("--keep-stego", action="store_true",
                    help="keep stego PNGs of every run under <out>/stego_runs/")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    pool = sorted(glob.glob(os.path.join(args.image_dir, args.image_glob)))
    if not pool:
        sys.exit(f"No images found in {args.image_dir}/{args.image_glob}")
    print(f"Image pool: {len(pool)} images; counts above that will cycle the pool.")

    # Build the list of (label, text) workloads
    workloads = [(f"synthetic", make_text(n)) for n in args.sizes]
    for p in args.text_files:
        with open(p, encoding="utf-8") as f:
            workloads.append((os.path.basename(p), f.read()))

    os.makedirs(args.out, exist_ok=True)
    runs_log = CsvLog(os.path.join(args.out, "runs.csv"), RUN_FIELDS)
    imgs_log = CsvLog(os.path.join(args.out, "images.csv"), IMAGE_FIELDS)

    if not args.no_warmup:
        print("Warm-up run (not logged)...")
        run_once(make_text(100), pick_images(pool, 1),
                 args.target_bpp, args.floor_fraction, args.pubkey)

    total = len(workloads) * len(args.counts) * args.repeats
    done = 0
    meta = {"python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count()}

    try:
        for source, text in workloads:
            text_bytes = len(text.encode("utf-8"))
            for n in args.counts:
                images = pick_images(pool, n)
                for rep in range(args.repeats):
                    done += 1
                    run_id = uuid.uuid4().hex[:12]
                    keep = (os.path.join(args.out, "stego_runs", run_id)
                            if args.keep_stego else None)
                    res, image_rows = run_once(text, images, args.target_bpp,
                                               args.floor_fraction, args.pubkey, keep)
                    runs_log.write({
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "text_bytes": text_bytes, "text_source": source,
                        "n_images": n, "repeat": rep,
                        "target_bpp": args.target_bpp,
                        "floor_fraction": args.floor_fraction,
                        **meta, **res,
                    })
                    for r in image_rows:
                        imgs_log.write({"run_id": run_id, "text_bytes": text_bytes,
                                        "n_images": n, "repeat": rep, **r})
                    total_s = res.get("t_total_s")
                    print(f"[{done}/{total}] {text_bytes:>8,} B  x {n:>2} img  "
                          f"rep {rep}  ->  "
                          + (f"{total_s:.2f}s  fit={res.get('n_fit')}/{n}"
                             if total_s is not None else res["status"]))
    finally:
        runs_log.close()
        imgs_log.close()
    print(f"\nDone. Logs in {args.out}/runs.csv and {args.out}/images.csv")


if __name__ == "__main__":
    main()