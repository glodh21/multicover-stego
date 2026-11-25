import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

NONCE_BYTES = 12
HKDF_SALT_BYTES = 32
HKDF_INFO = b"multicover-stego/aes-256-gcm/v1"


def derive_aes_key(shared_secret: bytes, salt: bytes) -> bytes:
    """Derive the AES-256-GCM key for one transmission."""
    if len(shared_secret) != 32 or shared_secret == b"\x00" * 32:
        raise ValueError("Invalid X25519 shared secret.")
    if len(salt) != HKDF_SALT_BYTES:
        raise ValueError("HKDF salt must be 32 bytes.")

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=HKDF_INFO,
    ).derive(shared_secret)


def create_sender_session(
    receiver_public_key_path: str = "keys/receiver_public.pem",
) -> tuple[bytes, dict[str, str]]:
    """Create an ephemeral X25519 session key and its public metadata."""
    receiver_public_key = serialization.load_pem_public_key(
        Path(receiver_public_key_path).read_bytes()
    )
    if not isinstance(receiver_public_key, X25519PublicKey):
        raise TypeError("The supplied public key is not an X25519 key.")

    ephemeral_private_key = X25519PrivateKey.generate()
    shared_secret = ephemeral_private_key.exchange(receiver_public_key)
    salt = os.urandom(HKDF_SALT_BYTES)
    aes_key = derive_aes_key(shared_secret, salt)

    ephemeral_public_key = ephemeral_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    metadata = {
        "EPHEMERAL_PUBLIC_KEY": base64.b64encode(
            ephemeral_public_key
        ).decode("ascii"),
        "HKDF_SALT": base64.b64encode(salt).decode("ascii"),
    }
    return aes_key, metadata


def recover_receiver_key(
    private_key_path: str,
    ephemeral_public_key: str,
    hkdf_salt: str,
    password: bytes,
) -> bytes:
    """Recreate the sender's AES key from receiver and transmission metadata."""
    private_key = serialization.load_pem_private_key(
        Path(private_key_path).read_bytes(),
        password=password,
    )
    if not isinstance(private_key, X25519PrivateKey):
        raise TypeError("The supplied private key is not an X25519 key.")

    try:
        public_key_bytes = base64.b64decode(
            ephemeral_public_key, validate=True
        )
        salt = base64.b64decode(hkdf_salt, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid X25519 transmission metadata.") from exc

    if len(public_key_bytes) != 32:
        raise ValueError("Sender's ephemeral X25519 public key must be 32 bytes.")

    sender_public_key = X25519PublicKey.from_public_bytes(public_key_bytes)
    shared_secret = private_key.exchange(sender_public_key)
    return derive_aes_key(shared_secret, salt)


def pack_payload(
    seq: int,
    chunk_bytes: bytes,
    seq_bits: int,
    len_bits: int,
) -> tuple[bytes, str]:
    """Build [seq][len][chunk], then zero-pad to a whole number of bytes."""
    header_bits = f"{seq:0{seq_bits}b}" + f"{len(chunk_bytes):0{len_bits}b}"
    data_bits = "".join(f"{byte:08b}" for byte in chunk_bytes)
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
    """Split text by capacity and AES-GCM encrypt each chunk."""
    aesgcm = AESGCM(key)
    n_chars = len(text)
    chunks = []
    cumulative_pct = 0.0
    start = 0

    for i, result in enumerate(capacity_results):
        cumulative_pct += result["capacity_percentage"]
        end = n_chars if i == len(capacity_results) - 1 \
            else round(cumulative_pct / 100.0 * n_chars)

        chunk = text[start:end]
        chunk_bytes = chunk.encode("utf-8")
        plaintext, header_bits = pack_payload(
            i, chunk_bytes, seq_bits, len_bits
        )

        nonce = os.urandom(NONCE_BYTES)
        blob = nonce + aesgcm.encrypt(nonce, plaintext, None)

        blob_bits = len(blob) * 8
        chunks.append({
            "path": result["path"],
            "cost_map": result["cost_map"],
            "max_capacity_bits": result["max_capacity_bits"],
            "seq": i,
            "chunk": chunk,
            "chunk_len_bytes": len(chunk_bytes),
            "header_bits": header_bits,
            "encrypted_blob": blob,
            "encrypted_hex": blob.hex(),
            "encrypted_len_bits": blob_bits,
            "fits": (blob_bits <= result["max_capacity_bits"]
                     and len(chunk_bytes) < 2 ** len_bits),
        })
        start = end

    return chunks
