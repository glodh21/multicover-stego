import os

import numpy as np
from PIL import Image
from scipy import signal
import pystc  # install with: pip install pystcstego


# --- 1. HILL cost function (Li et al., 2014) ---
def HILL(image: np.ndarray) -> np.ndarray:
    H = np.array([[-1,  2, -1],
                  [ 2, -4,  2],
                  [-1,  2, -1]], dtype=np.float64)
    L1 = np.ones((3, 3), dtype=np.float64) / (3 ** 2)
    L2 = np.ones((15, 15), dtype=np.float64) / (15 ** 2)

    image = image.astype(np.float64)

    # Residual -> abs -> 3x3 average
    costs = np.abs(signal.convolve2d(image, H, mode="same"))
    costs = signal.convolve2d(costs, L1, mode="same")

    # Invert BEFORE the 15x15 smoothing (this is the order HILL defines)
    costs = 1.0 / np.maximum(costs, 1e-5)   # avoid inf in perfectly flat areas
    costs = signal.convolve2d(costs, L2, mode="same")

    return costs.astype(np.float64)


# --- 2. Configuration ---
INPUT_IMAGE = "images/high_kayak.png"
OUTPUT_DIR = "output"
SECRET_TEXT = "My secret message for STC test!"
SEED = 12345  # shared key between sender and receiver

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. Embed ---
print("[1] Loading image and calculating costs...")
cover = np.array(Image.open(INPUT_IMAGE).convert("L"), dtype=np.uint8)
costs = HILL(cover)

print(f"[2] Embedding: '{SECRET_TEXT}'")
message_bytes = SECRET_TEXT.encode("utf-8")

# hide(message, cover, cost_plus1, cost_minus1, seed, mx, mn)
stego = pystc.hide(message_bytes, cover, costs, costs, SEED, mx=255, mn=0)
stego = np.clip(np.asarray(stego), 0, 255).astype(np.uint8)

output_path = os.path.join(OUTPUT_DIR, "stego_" + os.path.basename(INPUT_IMAGE))
Image.fromarray(stego, mode="L").save(output_path)  # PNG = lossless, required
print(f"    Stego image saved to: {output_path}")

# --- 4. Extract ---
print("[3] Extracting from the saved stego image...")
stego_loaded = np.array(Image.open(output_path).convert("L"), dtype=np.uint8)
extracted_bytes = pystc.unhide(stego_loaded, SEED)

# --- 5. Verify ---
extracted_text = extracted_bytes.decode("utf-8")
print(f"    Extracted: '{extracted_text}'")

if extracted_text == SECRET_TEXT:
    print("\n[SUCCESS] Round-trip verified.")
else:
    print("\n[FAILED] Extracted text does not match original.")