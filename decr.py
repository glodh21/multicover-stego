import os

import numpy as np
from PIL import Image
import pystc  # install with: pip install pystcstego
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

NONCE_BYTES = 12   # must match main.py
TAG_BYTES = 16
STEGO_DIR = "Stego"
DATA_FILE = "data.txt"


def load_params(path: str = DATA_FILE) -> dict:
    """Read KEY, SEED, SEQ_BITS, LEN_BITS from data.txt."""
    raw = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                raw[k.strip()] = v.strip()
    return {
        "key": bytes.fromhex(raw["KEY"]),
        "seed": int(raw["SEED"]),
        "seq_bits": int(raw["SEQ_BITS"]),
        "len_bits": int(raw["LEN_BITS"]),
    }


def unpack_payload(plaintext: bytes, seq_bits: int, len_bits: int):
    """Plaintext layout: [seq][len][chunk bytes][zero padding].
    Returns (seq, chunk_text)."""
    bits = "".join(f"{b:08b}" for b in plaintext)
    seq = int(bits[:seq_bits], 2)
    chunk_len = int(bits[seq_bits:seq_bits + len_bits], 2)   # in BYTES
    start = seq_bits + len_bits
    chunk_bits = bits[start:start + chunk_len * 8]
    chunk_bytes = int(chunk_bits, 2).to_bytes(chunk_len, "big") if chunk_len else b""
    return seq, chunk_bytes.decode("utf-8")


def main():
    p = load_params(DATA_FILE)
    aesgcm = AESGCM(p["key"])

    files = sorted(
        f for f in os.listdir(STEGO_DIR) if f.lower().endswith(".png")
    )
    if not files:
        print(f"No PNG files found in '{STEGO_DIR}/'")
        return

    for name in files:
        path = os.path.join(STEGO_DIR, name)
        try:
            # 1) Extract the embedded blob (HILL + STC, same seed as embedding)
            stego = np.array(Image.open(path).convert("L"), dtype=np.uint8)
            blob = bytes(pystc.unhide(stego, p["seed"]))

            # 2) Decrypt: nonce (12) || ciphertext || tag (16)
            nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
            plaintext = aesgcm.decrypt(nonce, ct, None)

            # 3) Split header (seq, len) from chunk text
            seq, text = unpack_payload(plaintext, p["seq_bits"], p["len_bits"])

            print(f"Image : {name}")
            print(f"Seq   : {seq}")
            print(f"Text  : {text}\n")
        except InvalidTag:
            print(f"Image : {name}")
            print("  ERROR: decryption failed (wrong key/seed or corrupted image)\n")
        except Exception as e:
            print(f"Image : {name}")
            print(f"  ERROR: {e}\n")


if __name__ == "__main__":
    main()