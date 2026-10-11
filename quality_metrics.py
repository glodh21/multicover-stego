
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

# Original cover images
COVER_DIR = Path("images")

# Generated stego images
STEGO_DIR = Path("Stego")

OUTPUT_CSV = Path("quality_results.csv")

IMAGE_FILES = [
    "high_bike.png",
    "low-sky.png",
    "mid_house.png",
    "mid_door.png",
    "high_kayak.png",
]


def load_grayscale(path):
    """Load an image as an 8-bit grayscale NumPy array."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def evaluate_pair(cover_path, stego_path):
    """Calculate distortion and structural similarity metrics."""
    cover = load_grayscale(cover_path)
    stego = load_grayscale(stego_path)

    if cover.shape != stego.shape:
        raise ValueError(
            f"Image dimensions differ: "
            f"{cover.shape} vs {stego.shape}"
        )

    mse = mean_squared_error(cover, stego)

    psnr = peak_signal_noise_ratio(
        cover, stego, data_range=255
    )

    ssim = structural_similarity(
        cover, stego, data_range=255
    )

    return {
        "image": cover_path.name,
        "width": cover.shape[1],
        "height": cover.shape[0],
        "mse": float(mse),
        "psnr_db": float(psnr),
        "ssim": float(ssim),
    }


def main():
    results = []

    for filename in IMAGE_FILES:
        cover_path = COVER_DIR / filename
        stego_path = STEGO_DIR / f"stego_{Path(filename).stem}.png"

        print(f"Evaluating: {filename}")

        if not cover_path.exists():
            print(f"  ERROR: Cover image not found: {cover_path}")
            continue

        if not stego_path.exists():
            print(f"  ERROR: Stego image not found: {stego_path}")
            continue

        try:
            result = evaluate_pair(cover_path, stego_path)
            results.append(result)

            print(f"  MSE : {result['mse']:.6f}")
            print(f"  PSNR: {result['psnr_db']:.4f} dB")
            print(f"  SSIM: {result['ssim']:.6f}\n")

        except Exception as exc:
            print(f"  ERROR: {exc}\n")

    if not results:
        print("No image pairs could be evaluated.")
        return

    fieldnames = [
        "image", "width", "height",
        "mse", "psnr_db", "ssim"
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Successfully evaluated: {len(results)}/{len(IMAGE_FILES)}")

    print(
        "Mean PSNR: "
        f"{np.mean([r['psnr_db'] for r in results]):.4f} dB"
    )
    print(
        "Mean SSIM: "
        f"{np.mean([r['ssim'] for r in results]):.6f}"
    )
    print(
        "Mean MSE : "
        f"{np.mean([r['mse'] for r in results]):.6f}"
    )

    print(f"\nDetailed results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()