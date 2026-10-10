
import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def encrypt_aes_key(aes_key: bytes, public_key_path: str,
                    envelope_path: str = "key_envelope.json") -> None:
    public_key = serialization.load_pem_public_key(
        Path(public_key_path).read_bytes()
    )

    if not isinstance(public_key, rsa.RSAPublicKey):
        raise TypeError("The supplied public key is not an RSA key.")

    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    envelope = {
        "version": 1,
        "key_encryption": "RSA-OAEP-SHA256",
        "content_encryption": "AES-256-GCM",
        "encrypted_aes_key": base64.b64encode(
            encrypted_key
        ).decode("ascii"),
    }

    Path(envelope_path).write_text(
        json.dumps(envelope, indent=2),
        encoding="utf-8",
    )


def decrypt_aes_key(private_key_path: str,
                    envelope_path: str = "key_envelope.json") -> bytes:
    private_key = serialization.load_pem_private_key(
        Path(private_key_path).read_bytes(),
        password=None,
    )

    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise TypeError("The supplied private key is not an RSA key.")

    envelope = json.loads(
        Path(envelope_path).read_text(encoding="utf-8")
    )

    if envelope.get("version") != 1:
        raise ValueError("Unsupported key-envelope version.")

    if envelope.get("key_encryption") != "RSA-OAEP-SHA256":
        raise ValueError("Unexpected key-encryption algorithm.")

    encrypted_key = base64.b64decode(
        envelope["encrypted_aes_key"],
        validate=True,
    )

    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    if len(aes_key) != 32:
        raise ValueError("Recovered AES key is not 256 bits.")

    return aes_key
