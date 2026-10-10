
import os
import secrets

import numpy as np
from PIL import Image
from scipy import signal
import pystc
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from image_percent import ImageCapacityCalculator
from key_exchange import encrypt_aes_key

NONCE_BYTES = 12
TAG_BYTES = 16
STEGO_DIR = "Stego"


def header_field_sizes(capacities: list[dict]) -> tuple[int, int]:
    n_images = len(capacities)
    if not capacities:
        raise ValueError("At least one cover image is required.")

    seq_bits = max(1, (len(capacities) - 1).bit_length())
    max_len_bytes = max(
        c["max_capacity_bits"] for c in capacities
    ) // 8
    len_bits = max(1, max_len_bytes.bit_length())

    return seq_bits, len_bits


def pack_payload(
    seq: int,
    chunk_bytes: bytes,
    seq_bits: int,
    len_bits: int,
) -> tuple[bytes, str]:
    if seq >= 2**seq_bits:
        raise ValueError("Sequence number does not fit the header.")
    if len(chunk_bytes) >= 2**len_bits:
        raise ValueError("Chunk length does not fit the header.")

    header_bits = (
        f"{seq:0{seq_bits}b}"
        + f"{len(chunk_bytes):0{len_bits}b}"
    )
    data_bits = "".join(f"{b:08b}" for b in chunk_bytes)
    bits = header_bits + data_bits
    bits += "0" * ((-len(bits)) % 8)

    return int(bits, 2).to_bytes(len(bits) // 8, "big"), header_bits


def split_and_encrypt(
    text: str,
    capacity_results: list[dict],
    seq_bits: int,
    len_bits: int,
    key: bytes,
) -> list[dict]:
    aesgcm = AESGCM(key)
    n_chars = len(text)
    chunks = []
    cumulative_pct = 0.0
    start = 0

    for i, res in enumerate(capacity_results):
        cumulative_pct += res["capacity_percentage"]
        end = (
            n_chars
            if i == len(capacity_results) - 1
            else round(cumulative_pct / 100.0 * n_chars)
        )

        chunk = text[start:end]
        chunk_bytes = chunk.encode("utf-8")
        plaintext, header_bits = pack_payload(
            i, chunk_bytes, seq_bits, len_bits
        )

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
            "encrypted_blob": blob,
            "encrypted_hex": blob.hex(),
            "encrypted_len_bits": blob_bits,
            "fits": (
                blob_bits <= res["max_capacity_bits"]
                and len(chunk_bytes) < 2**len_bits
            ),
        })
        start = end

    return chunks


def HILL(image: np.ndarray) -> np.ndarray:
    H = np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [-1, 2, -1],
    ], dtype=np.float64)

    L1 = np.ones((3, 3), dtype=np.float64) / 9
    L2 = np.ones((15, 15), dtype=np.float64) / 225

    image = image.astype(np.float64)

    costs = np.abs(signal.convolve2d(image, H, mode="same"))
    costs = signal.convolve2d(costs, L1, mode="same")
    costs = 1.0 / np.maximum(costs, 1e-5)
    costs = signal.convolve2d(costs, L2, mode="same")

    return costs.astype(np.float64)


def embed_chunks(
    chunks: list[dict],
    seed: int,
    out_dir: str = STEGO_DIR,
) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for c in chunks:
        if not c["fits"]:
            print(f"  SKIPPED (does not fit): {c['path']}")
            results.append({
                "path": c["path"],
                "stego_path": None,
            })
            continue

        cover = np.array(
            Image.open(c["path"]).convert("L"),
            dtype=np.uint8,
        )
        costs = HILL(cover)

        stego = pystc.hide(
            c["encrypted_blob"],
            cover,
            costs,
            costs,
            seed,
            mx=255,
            mn=0,
        )
        stego = np.clip(
            np.asarray(stego), 0, 255
        ).astype(np.uint8)

        base = os.path.splitext(os.path.basename(c["path"]))[0]
        stego_path = os.path.join(out_dir, f"stego_{base}.png")

        Image.fromarray(stego).save(stego_path)

        results.append({
            "path": c["path"],
            "stego_path": stego_path,
        })

    return results


def save_stego_params(
    seed: int,
    seq_bits: int,
    len_bits: int,
    path: str = "data.txt",
) -> None:
    """Save extraction parameters only; never store the AES key here."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"SEED={seed}\n")
        f.write(f"SEQ_BITS={seq_bits}\n")
        f.write(f"LEN_BITS={len_bits}\n")


def main():
    my_images = [
        "images/high_bike.png",
        "images/low-sky.png",
        "images/mid_house.png",
        "images/mid_door.png",
        "images/high_kayak.png",
    ]

    secret_text = (
        "The Mahabharata is one of the greatest Indian epics, "
        "traditionally attributed to the sage Ved Vyasa."
    )

    # 1. Estimate ES-ITC capacities.
    calculator = ImageCapacityCalculator(floor_fraction=0.03)
    capacities = calculator.calculate_image_capacities(my_images)
    capacities.sort(key=lambda c: c["max_capacity_bits"])

    # 2. Generate a fresh AES-256 key.
    key = AESGCM.generate_key(bit_length=256)

    # 3. Encrypt the AES key using the receiver's RSA public key.
    # The receiver must have generated this key pair beforehand.
    encrypt_aes_key(
        aes_key=key,
        public_key_path="keys/receiver_public.pem",
        envelope_path="key_envelope.json",
    )
    print("Encrypted AES key saved to key_envelope.json")

    # 4. Split the message and encrypt each chunk with AES-GCM.
    seq_bits, len_bits = header_field_sizes(capacities)
    chunks = split_and_encrypt(
        secret_text, capacities, seq_bits, len_bits, key
    )

    print(f"seq_bits={seq_bits}, len_bits={len_bits}\n")

    for c in chunks:
        print(
            f"{c['path']} | "
            f"estimated capacity: {c['max_capacity_bits']:,} bits"
        )
        print(f"  plain     : [{c['header_bits']}] {c['chunk']}")
        print(f"  encrypted : {c['encrypted_hex']}")
        print(f"  enc. size : {c['encrypted_len_bits']:,} bits")

        if not c["fits"]:
            print("  WARNING: payload exceeds estimated capacity")
        print()

    # Stop rather than silently producing an incomplete image set.
    oversized = [c["path"] for c in chunks if not c["fits"]]
    if oversized:
        raise RuntimeError(
            "Some encrypted chunks exceed estimated capacities: "
            + ", ".join(oversized)
        )

    # 5. Generate the STC seed and save extraction parameters.
    seed = secrets.randbelow(2**31)
    save_stego_params(seed, seq_bits, len_bits)

    # 6. Embed and save stego images.
    print(f"Embedding into images -> '{STEGO_DIR}/'")
    stego_results = embed_chunks(chunks, seed, STEGO_DIR)

    for result in stego_results:
        if result["stego_path"]:
            print(
                f"  {result['path']} -> {result['stego_path']}"
            )

    return capacities, chunks, stego_results


if __name__ == "__main__":
    main()
