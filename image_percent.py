"""Image capacity estimator based on GLCM texture entropy (Liao et al., 2021),
with an added capacity floor.

Usage:
    from image_capacity import ImageCapacityCalculator

    calc = ImageCapacityCalculator(floor_fraction=0.03)
    results = calc.calculate_image_capacities(["a.png", "b.png"])
"""
import math

import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from skimage.feature import graycomatrix


class ImageCapacityCalculator:
    """Computes per-image embedding capacity and each image's share of the total."""

    # 3x3 KerBöhme high-pass filter (Eq. 16)
    KERBOHME_KERNEL = np.array([[-1,  2, -1],
                                [ 2, -4,  2],
                                [-1,  2, -1]], dtype=np.float64)

    ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    def __init__(self, floor_fraction: float = 0.03):
        """
        floor_fraction: every image gets at least this fraction of its pixel
        count as capacity. (Our extension, not part of Liao et al., 2021.)
        """
        self.floor_fraction = floor_fraction

    # ------------------------------------------------------------------
    # Step 1: filtering
    # ------------------------------------------------------------------
    def apply_kerbohme_filter(self, image_np: np.ndarray) -> np.ndarray:
        """Apply the KerBöhme high-pass filter with mirror padding (Eq. 15).

        GLCM needs non-negative integers, so the signed response is shifted
        by a fixed +127 and clipped to [0, 255] (no per-image stretching,
        which would destroy the contrast information Eq. 20 depends on).
        """
        img_float = image_np.astype(np.float64)
        filtered = convolve(img_float, self.KERBOHME_KERNEL, mode="mirror")
        return np.clip(filtered + 127, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 2: texture entropy
    # ------------------------------------------------------------------
    def compute_texture_entropy(self, filtered_img: np.ndarray) -> float:
        """GLCM at d=1 over 4 angles, then mean directional Shannon entropy
        (Eq. 18-19)."""
        glcm = graycomatrix(
            filtered_img,
            distances=[1],
            angles=self.ANGLES,
            levels=256,
            symmetric=True,
            normed=True,
        )

        entropies = []
        for angle_idx in range(len(self.ANGLES)):
            p = glcm[:, :, 0, angle_idx]
            p_nz = p[p > 0]
            entropies.append(-np.sum(p_nz * np.log2(p_nz)))

        return float(np.mean(entropies))

    # ------------------------------------------------------------------
    # Step 3: capacities
    # ------------------------------------------------------------------
    def calculate_image_capacities(self, image_paths: list[str]) -> list[dict]:
        """For each image compute hbar, then capacity via Eq. (20):
            c_i = r_i * s_i * (hbar_i - hbar_min) / (hbar_max - hbar_min)
        with the floor applied, plus each image's percentage of total capacity.
        """
        results = []
        for path in image_paths:
            img_np = np.array(Image.open(path).convert("L"))
            r_i, s_i = img_np.shape

            filtered = self.apply_kerbohme_filter(img_np)
            hbar_i = self.compute_texture_entropy(filtered)

            results.append({
                "path": path,
                "dimensions": (r_i, s_i),
                "num_pixels": r_i * s_i,
                "hbar": hbar_i,
                "max_capacity_bits": 0,
            })

        entropies = [res["hbar"] for res in results]
        h_min, h_max = min(entropies), max(entropies)

        for res in results:
            if math.isclose(h_max, h_min):
                # Eq. (20) is undefined when all entropies are equal.
                raw_c_i = float(res["num_pixels"])
            else:
                raw_c_i = res["num_pixels"] * (
                    (res["hbar"] - h_min) / (h_max - h_min)
                )

            floor_bits = self.floor_fraction * res["num_pixels"]
            c_i = max(raw_c_i, floor_bits)

            res["raw_capacity_bits"] = int(np.floor(raw_c_i))
            res["max_capacity_bits"] = int(np.floor(c_i))

        total_cap = sum(res["max_capacity_bits"] for res in results)
        for res in results:
            res["capacity_percentage"] = (
                res["max_capacity_bits"] / total_cap * 100.0
                if total_cap > 0 else 0.0
            )

        return results

    # ------------------------------------------------------------------
    # Convenience: pretty-print results
    # ------------------------------------------------------------------
    @staticmethod
    def print_results(results: list[dict]) -> None:
        print()
        print(f"{'Image Path':<35} | {'Entropy (hbar)':<14} | "
              f"{'Raw Cap (bits)':<16} | {'Floored Cap (bits)':<20} | "
              f"{'% of Total Cap':<14}")
        print("-" * 115)
        for img in results:
            print(f"{img['path']:<35} | {img['hbar']:<14.4f} | "
                  f"{img['raw_capacity_bits']:<16,d} | "
                  f"{img['max_capacity_bits']:<20,d} | "
                  f"{img['capacity_percentage']:<14.2f}")