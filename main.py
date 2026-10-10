import os
import secrets

import numpy as np
from PIL import Image
from scipy import signal
import pystc  # install with: pip install pystcstego
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from image_percent import ImageCapacityCalculator

NONCE_BYTES = 12   # standard GCM nonce size
TAG_BYTES = 16     # GCM auth tag, appended to ciphertext by AESGCM
STEGO_DIR = "Stego"


def header_field_sizes(capacities: list[dict]) -> tuple[int, int]:
    """Smallest bit widths that can hold the header fields.

    seq : enough bits for the highest sequence number (0 .. n-1).
    len : enough bits for the largest chunk length in BYTES any image could
          hold (max_capacity_bits // 8).
    """
    n_images = len(capacities)
    seq_bits = max(1, (n_images - 1).bit_length())

    max_len_bytes = max(c["max_capacity_bits"] for c in capacities) // 8
    len_bits = max(1, max_len_bytes.bit_length())
    return seq_bits, len_bits


def pack_payload(seq: int, chunk_bytes: bytes,
                 seq_bits: int, len_bits: int) -> tuple[bytes, str]:
    """Build [seq][len][chunk] as a bit string, zero-pad to a whole number of
    bytes (AES-GCM works on bytes), and return (plaintext_bytes, header_bits).

    The len field tells the decoder exactly how many chunk bytes to read, so
    the trailing padding bits are simply ignored after decryption.
    """
    header_bits = f"{seq:0{seq_bits}b}" + f"{len(chunk_bytes):0{len_bits}b}"
    data_bits = "".join(f"{b:08b}" for b in chunk_bytes)
    bits = header_bits + data_bits
    bits += "0" * ((-len(bits)) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big"), header_bits


def split_and_encrypt(text: str, capacity_results: list[dict],
                      seq_bits: int, len_bits: int, key: bytes) -> list[dict]:
    """Split `text` by capacity percentage, then encrypt each
    (seq + len + chunk) payload with AES-256-GCM using the same key.

    Images must already be sorted so the highest-capacity one is LAST: it
    receives whatever is left over after rounding.

    Encrypted blob layout (what you embed, and later decrypt):
        nonce (12 bytes) || ciphertext || GCM tag (16 bytes)
    """
    aesgcm = AESGCM(key)
    n_chars = len(text)
    chunks = []
    cumulative_pct = 0.0
    start = 0

    for i, res in enumerate(capacity_results):
        cumulative_pct += res["capacity_percentage"]
        end = n_chars if i == len(capacity_results) - 1 \
            else round(cumulative_pct / 100.0 * n_chars)

        chunk = text[start:end]
        chunk_bytes = chunk.encode("utf-8")

        plaintext, header_bits = pack_payload(i, chunk_bytes, seq_bits, len_bits)

        # Same key for every chunk, so the nonce MUST be unique per chunk.
        nonce = os.urandom(NONCE_BYTES)
        blob = nonce + aesgcm.encrypt(nonce, plaintext, None)

        blob_bits = len(blob) * 8
        chunks.append({
            "path": res["path"],
            "max_capacity_bits": res["max_capacity_bits"],
            "seq": i,
            "chunk": chunk,
            "chunk_len_bytes": len(chunk_bytes),
            "header_bits": header_bits,
            "encrypted_blob": blob,                 # bytes: use for embedding
            "encrypted_hex": blob.hex(),            # same data, as text
            "encrypted_len_bits": blob_bits,
            "fits": (blob_bits <= res["max_capacity_bits"]
                     and len(chunk_bytes) < 2 ** len_bits),
        })
        start = end

    return chunks


def save_key(key: bytes, path: str = "data.txt") -> None:
    """Store the AES-256 key (hex) for later decryption."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"KEY={key.hex()}\n")


def append_stego_params(seed: int, seq_bits: int, len_bits: int,
                        path: str = "data.txt") -> None:
    """Append the STC seed (shared stego key) and the header widths needed at
    decode time to data.txt, after the AES key line."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"SEED={seed}\n")
        f.write(f"SEQ_BITS={seq_bits}\n")
        f.write(f"LEN_BITS={len_bits}\n")


# --- Steganography: HILL cost function (Li et al., 2014) ---
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


def embed_chunks(chunks: list[dict], seed: int,
                 out_dir: str = STEGO_DIR) -> list[dict]:
    """Embed each chunk's encrypted blob into its image using HILL + STC.
    Stego images are saved as lossless PNG in `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for c in chunks:
        if not c["fits"]:
            print(f"  SKIPPED (does not fit): {c['path']}")
            results.append({"path": c["path"], "stego_path": None})
            continue

        cover = np.array(Image.open(c["path"]).convert("L"), dtype=np.uint8)
        costs = HILL(cover)

        # hide(message, cover, cost_plus1, cost_minus1, seed, mx, mn)
        stego = pystc.hide(c["encrypted_blob"], cover, costs, costs,
                           seed, mx=255, mn=0)
        stego = np.clip(np.asarray(stego), 0, 255).astype(np.uint8)

        base = os.path.splitext(os.path.basename(c["path"]))[0]
        stego_path = os.path.join(out_dir, f"stego_{base}.png")
        Image.fromarray(stego, mode="L").save(stego_path)  # PNG = lossless, required

        results.append({"path": c["path"], "stego_path": stego_path})

    return results


def main():
    my_images = [
        "images/high_bike.png",
        "images/low-sky.png",
        "images/mid_house.png",
        "images/mid_door.png",
        "images/high_kayak.png",
    ]

    secret_text = (
        "The Mahabharata is one of the greatest Indian epics, traditionally attributed to the sage Ved Vyasa."
    )
    # secret_text = open("secret.txt", encoding="utf-8").read()

    # 1) Compute capacities
    calculator = ImageCapacityCalculator(floor_fraction=0.03)
    capacities = calculator.calculate_image_capacities(my_images)

    # 2) Sort ascending by capacity -> highest-capacity image is LAST
    capacities.sort(key=lambda c: c["max_capacity_bits"])

    # 3) One random AES-256 key for all chunks, saved to data.txt
    key = AESGCM.generate_key(bit_length=256)
    save_key(key, "data.txt")

    # 4) Header widths, split, encrypt
    seq_bits, len_bits = header_field_sizes(capacities)
    chunks = split_and_encrypt(secret_text, capacities, seq_bits, len_bits, key)

    # 5) Output
    print(f"seq_bits={seq_bits}, len_bits={len_bits}  (needed again at decode time)\n")
    for c in chunks:
        print(f"{c['path']} | max capacity: {c['max_capacity_bits']:,} bits")
        print(f"  plain     : [{c['header_bits']}] {c['chunk']}")
        print(f"  encrypted : {c['encrypted_hex']}")
        print(f"  enc. size : {c['encrypted_len_bits']:,} bits")
        if not c["fits"]:
            print("  WARNING: encrypted payload exceeds this image's capacity")
        print()

    # 6) Steganography: embed each encrypted blob into its image (HILL + STC)
    seed = secrets.randbelow(2 ** 31)          # shared stego key (sender/receiver)
    append_stego_params(seed, seq_bits, len_bits, "data.txt")

    print(f"Embedding into images (seed saved to data.txt) -> '{STEGO_DIR}/'")
    stego_results = embed_chunks(chunks, seed, STEGO_DIR)
    for r in stego_results:
        if r["stego_path"]:
            print(f"  {r['path']}  ->  {r['stego_path']}")

    return capacities, chunks, stego_results


if __name__ == "__main__":
    main()