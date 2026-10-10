import os

import numpy as np
from PIL import Image
import pystc
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

from key_exchange import decrypt_aes_key

NONCE_BYTES = 12
TAG_BYTES = 16
STEGO_DIR = "Stego"
DATA_FILE = "data.txt"


def load_params(path: str = DATA_FILE) -> dict:
    """Read the STC seed and payload header sizes from data.txt."""
    raw = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                raw[k.strip()] = v.strip()

    return {
        "seed": int(raw["SEED"]),
        "seq_bits": int(raw["SEQ_BITS"]),
        "len_bits": int(raw["LEN_BITS"]),
    }


def unpack_payload(plaintext: bytes, seq_bits: int, len_bits: int):
    """Extract sequence number and UTF-8 text from the decrypted payload."""
    bits = "".join(f"{b:08b}" for b in plaintext)

    header_size = seq_bits + len_bits
    if len(bits) < header_size:
        raise ValueError("Decrypted payload is too short for its header.")

    seq = int(bits[:seq_bits], 2)
    chunk_len = int(bits[seq_bits:header_size], 2)  # Length in bytes

    start = header_size
    end = start + chunk_len * 8

    if end > len(bits):
        raise ValueError("Declared chunk length exceeds the payload size.")

    chunk_bits = bits[start:end]
    chunk_bytes = (
        int(chunk_bits, 2).to_bytes(chunk_len, "big")
        if chunk_len else b""
    )

    return seq, chunk_bytes.decode("utf-8")


def main():
    # 1. Load extraction parameters.
    params = load_params(DATA_FILE)

    # 2. Recover the AES key using the receiver's RSA private key.
    aes_key = decrypt_aes_key(
        private_key_path="keys/receiver_private.pem",
        envelope_path="key_envelope.json",
    )
    aesgcm = AESGCM(aes_key)

    if not os.path.isdir(STEGO_DIR):
        print(f"Stego directory '{STEGO_DIR}/' not found.")
        return

    files = sorted(
        f for f in os.listdir(STEGO_DIR)
        if f.lower().endswith(".png")
    )

    if not files:
        print(f"No PNG files found in '{STEGO_DIR}/'.")
        return

    recovered_chunks = []
    failed_images = []

    for name in files:
        path = os.path.join(STEGO_DIR, name)

        try:
            # 3. Extract the embedded encrypted blob using STC.
            stego = np.array(
                Image.open(path).convert("L"),
                dtype=np.uint8,
            )
            blob = bytes(pystc.unhide(stego, params["seed"]))

            # 4. Validate and decrypt nonce || ciphertext || GCM tag.
            if len(blob) < NONCE_BYTES + TAG_BYTES:
                raise ValueError("Extracted blob is too short.")

            nonce = blob[:NONCE_BYTES]
            ciphertext_and_tag = blob[NONCE_BYTES:]

            plaintext = aesgcm.decrypt(nonce, ciphertext_and_tag, None)

            # 5. Recover the sequence number and text chunk.
            seq, text = unpack_payload(
                plaintext,
                params["seq_bits"],
                params["len_bits"],
            )

            recovered_chunks.append((seq, text))
            print(f"Image : {name}")
            print(f"Seq   : {seq}")
            print(f"Text  : {text}\n")

        except InvalidTag:
            failed_images.append(name)
            print(f"Image : {name}")
            print("ERROR: AES-GCM authentication failed. Check the key,")
            print("       extraction parameters, and image integrity.\n")

        except Exception as e:
            failed_images.append(name)
            print(f"Image : {name}")
            print(f"ERROR: {e}\n")

    # 6. Reconstruct the original message in sequence order.
    if recovered_chunks:
        recovered_chunks.sort(key=lambda item: item[0])
        recovered_text = "".join(text for _, text in recovered_chunks)

        print("=" * 50)
        print("RECONSTRUCTED MESSAGE")
        print("=" * 50)
        print(recovered_text)
        print("=" * 50)

        if failed_images:
            print(
                "\nWARNING: Some images failed extraction or decryption. "
                "The reconstructed message may be incomplete."
            )
    else:
        print("No chunks were successfully recovered.")


if __name__ == "__main__":
    main()