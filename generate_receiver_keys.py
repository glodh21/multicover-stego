
from getpass import getpass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

KEY_DIR = Path("keys")


def generate_receiver_keys():
    KEY_DIR.mkdir(parents=True, exist_ok=True)

    passphrase = getpass("Set receiver private-key passphrase: ")
    if not passphrase:
        raise ValueError("A non-empty private-key passphrase is required.")
    confirmation = getpass("Confirm receiver private-key passphrase: ")
    if passphrase != confirmation:
        raise ValueError("Passphrases do not match.")

    private_key = X25519PrivateKey.generate()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.BestAvailableEncryption(
            passphrase.encode("utf-8")
        ),
    )

    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    (KEY_DIR / "receiver_private.pem").write_bytes(private_pem)
    (KEY_DIR / "receiver_public.pem").write_bytes(public_pem)

    print("X25519 key pair generated.")
    print(f"Public key : {KEY_DIR / 'receiver_public.pem'}")
    print(f"Private key: {KEY_DIR / 'receiver_private.pem'}")
    print("Share only the public key with the sender.")


if __name__ == "__main__":
    generate_receiver_keys()
