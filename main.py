import os
import secrets

import numpy as np
from PIL import Image
import pystc  # install with: pip install pystcstego

from encryption import create_sender_session, split_and_encrypt
from image_percent import HillCapacityCalculator

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


def save_stego_params(seed: int, seq_bits: int, len_bits: int,
                      crypto_metadata: dict[str, str],
                      path: str = "data.txt") -> None:
    """Store STC extraction values and the public key-derivation metadata."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"SEED={seed}\n")
        f.write(f"SEQ_BITS={seq_bits}\n")
        f.write(f"LEN_BITS={len_bits}\n")
        for name, value in crypto_metadata.items():
            f.write(f"{name}={value}\n")


def embed_chunks(chunks: list[dict], seed: int,
                 out_dir: str = STEGO_DIR) -> list[dict]:
    """Embed each chunk's encrypted blob into its image using the HILL cost
    map computed in image_percent (HILL + STC).
    Stego images are saved as lossless PNG in `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for c in chunks:
        if not c["fits"]:
            print(f"  SKIPPED (does not fit): {c['path']}")
            results.append({"path": c["path"], "stego_path": None})
            continue

        cover = np.array(Image.open(c["path"]).convert("L"), dtype=np.uint8)
        costs = c["cost_map"]          # reused, not recomputed

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

    '''secret_text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    )'''
    secret_text = open("secret.txt", encoding="utf-8").read()

    calculator = HillCapacityCalculator(target_bpp=0.4, floor_fraction=0.0)
    capacities = calculator.calculate_image_capacities(my_images)

    # Print a summary only (cost maps are huge arrays)
    for c in capacities:
        print({k: v for k, v in c.items() if k != "cost_map"})

    # 2) Sort ascending by capacity -> highest-capacity image is LAST
    capacities.sort(key=lambda c: c["max_capacity_bits"])

    # 3) Establish an AES-256 key with the receiver's trusted X25519 public key.
    key, crypto_metadata = create_sender_session(
        receiver_public_key_path="keys/receiver_public.pem"
    )

    # 4) Header widths, split, encrypt
    seq_bits, len_bits = header_field_sizes(capacities)
    chunks = split_and_encrypt(
        secret_text, capacities, seq_bits, len_bits, key
    )

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
    save_stego_params(seed, seq_bits, len_bits, crypto_metadata, "data.txt")

    print(f"Embedding into images (seed saved to data.txt) -> '{STEGO_DIR}/'")
    stego_results = embed_chunks(chunks, seed, STEGO_DIR)
    for r in stego_results:
        if r["stego_path"]:
            print(f"  {r['path']}  ->  {r['stego_path']}")

    return capacities, chunks, stego_results


if __name__ == "__main__":
    main()
