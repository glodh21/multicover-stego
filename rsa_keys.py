
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

KEY_DIR = Path("keys")


def generate_receiver_keys():
    KEY_DIR.mkdir(parents=True, exist_ok=True)

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    (KEY_DIR / "receiver_private.pem").write_bytes(private_pem)
    (KEY_DIR / "receiver_public.pem").write_bytes(public_pem)

    print("RSA key pair generated.")
    print(f"Public key : {KEY_DIR / 'receiver_public.pem'}")
    print(f"Private key: {KEY_DIR / 'receiver_private.pem'}")
    print("Share only the public key with the sender.")


if __name__ == "__main__":
    generate_receiver_keys()
