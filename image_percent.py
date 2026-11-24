"""HILL-based image capacity estimator (light version).

Replaces the GLCM texture-entropy score (Liao et al., 2021) with a score
derived from the same HILL cost map that the embedder uses.
"""
import numpy as np
from PIL import Image
from scipy import signal


# ----------------------------------------------------------------------
# HILL cost function (Li et al., 2014) -- moved here from main.py so the
# allocator and the embedder use exactly the same costs.
# ----------------------------------------------------------------------
def HILL(image: np.ndarray) -> np.ndarray:
    H = np.array([[-1,  2, -1],
                  [ 2, -4,  2],
                  [-1,  2, -1]], dtype=np.float64)
    L1 = np.ones((3, 3), dtype=np.float64) / (3 ** 2)
    L2 = np.ones((15, 15), dtype=np.float64) / (15 ** 2)

    image = image.astype(np.float64)
    costs = np.abs(signal.convolve2d(image, H, mode="same"))
    costs = signal.convolve2d(costs, L1, mode="same")
    costs = 1.0 / np.maximum(costs, 1e-5)
    costs = signal.convolve2d(costs, L2, mode="same")
    return costs.astype(np.float64)


def payload_bits(cost: np.ndarray, lam: float) -> float:
    """Theoretical payload (bits) an image can carry at Lagrange multiplier lam.

    Ternary (+1/-1) embedding, matching pystc.hide(..., costs, costs, ...):
        beta = exp(-lam*rho) / (1 + 2*exp(-lam*rho))   (prob. of each +-1 change)
        H    = -2*beta*log2(beta) - (1-2*beta)*log2(1-2*beta)
    Payload is the sum of H over all pixels.
    """
    e = np.exp(-lam * cost)
    beta = np.clip(e / (1.0 + 2.0 * e), 1e-12, 1.0 / 3.0)
    stay = 1.0 - 2.0 * beta
    h = -2.0 * beta * np.log2(beta) - stay * np.log2(stay)
    return float(h.sum())


class HillCapacityCalculator:
    """Same interface and output keys as ImageCapacityCalculator."""

    def __init__(self, lam: float | None = None, target_bpp: float = 0.4,
                 safety: float = 0.9, floor_fraction: float = 0.0):
        """
        lam            : fixed Lagrange multiplier. If None, it is calibrated
                         on the batch so the average payload is target_bpp.
        target_bpp     : only used when lam is None (calibration target).
        safety         : real STC reaches ~90-95% of the theoretical bound.
        floor_fraction : optional floor (default 0: not needed, see notes).
        """
        self.lam = lam
        self.target_bpp = target_bpp
        self.safety = safety
        self.floor_fraction = floor_fraction

    def _calibrate_lambda(self, costs: list[np.ndarray]) -> float:
        total_pixels = sum(c.size for c in costs)
        target = self.target_bpp * total_pixels
        lo, hi = 1e-8, 1e8                       # payload falls as lam grows
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            if sum(payload_bits(c, mid) for c in costs) > target:
                lo = mid
            else:
                hi = mid
        return float(np.sqrt(lo * hi))

    def calculate_image_capacities(self, image_paths: list[str]) -> list[dict]:
        results, costs = [], []
        for path in image_paths:
            img = np.array(Image.open(path).convert("L"))
            r, s = img.shape
            costs.append(HILL(img))
            results.append({"path": path, "dimensions": (r, s),
                            "num_pixels": r * s})

        lam = self.lam if self.lam is not None else self._calibrate_lambda(costs)

        for res, cost in zip(results, costs):
            raw = payload_bits(cost, lam)
            floored = max(raw, self.floor_fraction * res["num_pixels"])
            res["hbar"] = raw / res["num_pixels"]          # bits per pixel (score)
            res["lambda"] = lam
            res["raw_capacity_bits"] = int(np.floor(raw * self.safety))
            res["max_capacity_bits"] = int(np.floor(floored * self.safety))
            res["cost_map"] = cost

        total = sum(r["max_capacity_bits"] for r in results)
        for res in results:
            res["capacity_percentage"] = (
                res["max_capacity_bits"] / total * 100.0 if total > 0 else 0.0
            )
        return results