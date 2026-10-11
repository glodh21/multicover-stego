import csv
from pathlib import Path

import numpy as np
from PIL import Image

COVER_DIR = Path("images")
STEGO_DIR = Path("Stego")
OUTPUT_CSV = Path("steganalysis_results.csv")

IMAGE_FILES = [
    "high_bike.png",
    "low-sky.png",
    "mid_house.png",
    "mid_door.png",
    "high_kayak.png",
]


def load_grayscale(path):
    """Load an image as an 8-bit grayscale array."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def histogram_metrics(cover, stego):
    """Compare normalized grayscale histograms."""
    cover_hist = np.bincount(
        cover.ravel(), minlength=256
    ).astype(np.float64)

    stego_hist = np.bincount(
        stego.ravel(), minlength=256
    ).astype(np.float64)

    cover_hist /= cover.size
    stego_hist /= stego.size

    # Total variation distance: 0 means identical distributions.
    histogram_distance = 0.5 * np.abs(
        cover_hist - stego_hist
    ).sum()

    # Mean absolute difference between normalized bins.
    histogram_mae = np.abs(
        cover_hist - stego_hist
    ).mean()

    return histogram_distance, histogram_mae


def adjacent_difference_metrics(image):
    """Measure neighboring-pixel differences."""
    horizontal = np.abs(
        image[:, 1:].astype(np.int16)
        - image[:, :-1].astype(np.int16)
    )

    vertical = np.abs(
        image[1:, :].astype(np.int16)
        - image[:-1, :].astype(np.int16)
    )

    return {
        "horizontal_mean_abs_diff": float(horizontal.mean()),
        "vertical_mean_abs_diff": float(vertical.mean()),
        "horizontal_diff_variance": float(horizontal.var()),
        "vertical_diff_variance": float(vertical.var()),
    }


def analyze_pair(cover_path, stego_path):
    cover = load_grayscale(cover_path)
    stego = load_grayscale(stego_path)

    if cover.shape != stego.shape:
        raise ValueError(
            f"Dimension mismatch: {cover.shape} vs {stego.shape}"
        )

    cover_i = cover.astype(np.int16)
    stego_i = stego.astype(np.int16)
    difference = stego_i - cover_i

    changed_pixels = int(np.count_nonzero(difference))
    total_pixels = int(cover.size)

    hist_distance, hist_mae = histogram_metrics(cover, stego)

    cover_adj = adjacent_difference_metrics(cover)
    stego_adj = adjacent_difference_metrics(stego)

    return {
        "image": cover_path.name,
        "width": int(cover.shape[1]),
        "height": int(cover.shape[0]),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_pixels_percent": (
            100.0 * changed_pixels / total_pixels
        ),
        "max_abs_pixel_change": int(np.abs(difference).max()),
        "mean_abs_pixel_change": float(np.abs(difference).mean()),
        "histogram_total_variation": float(hist_distance),
        "histogram_bin_mae": float(hist_mae),
        "cover_horizontal_mean_diff":
            cover_adj["horizontal_mean_abs_diff"],
        "stego_horizontal_mean_diff":
            stego_adj["horizontal_mean_abs_diff"],
        "cover_vertical_mean_diff":
            cover_adj["vertical_mean_abs_diff"],
        "stego_vertical_mean_diff":
            stego_adj["vertical_mean_abs_diff"],
        "cover_horizontal_diff_variance":
            cover_adj["horizontal_diff_variance"],
        "stego_horizontal_diff_variance":
            stego_adj["horizontal_diff_variance"],
        "cover_vertical_diff_variance":
            cover_adj["vertical_diff_variance"],
        "stego_vertical_diff_variance":
            stego_adj["vertical_diff_variance"],
    }


def main():
    results = []

    for filename in IMAGE_FILES:
        cover_path = COVER_DIR / filename
        stego_path = STEGO_DIR / f"stego_{Path(filename).stem}.png"

        if not cover_path.exists():
            print(f"Missing cover: {cover_path}")
            continue

        if not stego_path.exists():
            print(f"Missing stego image: {stego_path}")
            continue

        try:
            result = analyze_pair(cover_path, stego_path)
            results.append(result)

            print(f"\n{filename}")
            print(
                f"  Changed pixels: {result['changed_pixels']:,} "
                f"({result['changed_pixels_percent']:.6f}%)"
            )
            print(
                "  Histogram total variation: "
                f"{result['histogram_total_variation']:.10f}"
            )
            print(
                "  Horizontal mean difference: "
                f"{result['cover_horizontal_mean_diff']:.6f} -> "
                f"{result['stego_horizontal_mean_diff']:.6f}"
            )
            print(
                "  Vertical mean difference: "
                f"{result['cover_vertical_mean_diff']:.6f} -> "
                f"{result['stego_vertical_mean_diff']:.6f}"
            )

        except Exception as exc:
            print(f"Error analyzing {filename}: {exc}")

    if not results:
        print("No image pairs were analyzed.")
        return

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAnalyzed {len(results)} image pairs.")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()