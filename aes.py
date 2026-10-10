import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # pip install cryptography

text = input("Enter text: ")
plaintext = text.encode("utf-8")

# 1. Calculate input sizes
char_count = len(text)
plaintext_bytes = len(plaintext)
plaintext_bits = plaintext_bytes * 8

print("\n--- INPUT DATA ---")
print(f"Original text  : {char_count} characters")
print(f"Plaintext size : {plaintext_bytes} bytes ({plaintext_bits} bits)")

# 2. Cryptographic generation
key = AESGCM.generate_key(bit_length=256)   # 256-bit key
nonce = os.urandom(12)                      # 96-bit nonce (standard for GCM)

# ct_with_tag contains: [ciphertext] + [128-bit authentication tag]
ct_with_tag = AESGCM(key).encrypt(nonce, plaintext, None)  

# 3. Calculate proper bits from the encrypted output
total_encrypted_bytes = len(ct_with_tag)
tag_bytes = 16                              # AES-GCM tags are strictly 16 bytes
ciphertext_bytes = total_encrypted_bytes - tag_bytes

ciphertext_bits = ciphertext_bytes * 8
tag_bits = tag_bytes * 8
nonce_bits = len(nonce) * 8
total_stored_bits = (len(nonce) + total_encrypted_bytes) * 8

print("\n--- ENCRYPTION PIECES ---")
print(f"Nonce size     : {len(nonce)} bytes ({nonce_bits} bits)")
print(f"Ciphertext size: {ciphertext_bytes} bytes ({ciphertext_bits} bits) -> Matches plaintext byte size")
print(f"Auth tag size  : {tag_bytes} bytes ({tag_bits} bits)")

print("\n--- FINAL OUTPUT SIZES ---")
print(f"Total payload  : {total_encrypted_bytes} bytes ({total_encrypted_bytes * 8} bits) -> [ciphertext + tag]")
print(f"Total to store : {len(nonce) + total_encrypted_bytes} bytes ({total_stored_bits} bits) -> [nonce + ciphertext + tag]")
