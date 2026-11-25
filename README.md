
Estimates per-image embedding capacity from image texture, following Liao et al., *"Adaptive Payload Distribution in Multiple Images Steganography Based on Image Texture Features,"* IEEE TDSC, 2021 (ES-ITC strategy, Eq. 15–20).

**extension:** a 3% capacity floor so no image receives zero payload.

Output Format
<img width="1040" height="174" alt="Screenshot 2026-10-10 121516" src="https://github.com/user-attachments/assets/4be4afef-6186-4322-a835-5a301bd37c36" />

python3 main.py

to run attack.py
python attack.py --stego Stego --cover images --params data.txt --spam



python benchmark.py                          # sizes 100B-50KB, 1-10 images, 3 repeats
python benchmark.py --sizes 100 1000 10000 100000 --counts 1 2 3 5 8 12 15 --repeats 5
python benchmark.py --text-files secret.txt  # also test a real file
python analyze.py

## Receiver key setup and transmission

Run once on the receiver to create a long-term X25519 key pair:

```powershell
python generate_receiver_keys.py
```

The script encrypts `keys/receiver_private.pem` with the passphrase you enter.
Keep that private key and passphrase on the receiver. Transfer only
`keys/receiver_public.pem` to the sender through a trusted channel, and verify
that the sender has the receiver's authentic public key.

On the sender, place the trusted public key at
`keys/receiver_public.pem`, then run:

```powershell
python main.py
```

For each transmission, `main.py` creates a fresh ephemeral X25519 key pair,
derives an AES-256 key with HKDF-SHA-256, and writes the ephemeral public key
and HKDF salt to `data.txt`. The AES-GCM nonce, ciphertext, and tag remain
together inside each STC-embedded chunk. Transfer `Stego/` and `data.txt` to
the receiver; transfer `data.txt` through an authenticated trusted channel so
the ephemeral public key cannot be silently replaced.

On the receiver, place the files in the expected locations and run:

```powershell
python decr.py
```

Enter the private-key passphrase when prompted. The receiver uses its private
key and the transmission's ephemeral public key to derive the same AES key.
The metadata channel must be trusted; this protocol does not authenticate the
sender, and use of the receiver's long-term private key means it does not
provide forward secrecy. The existing `SEED`, `SEQ_BITS`, and `LEN_BITS`
metadata and STC seed behavior are unchanged.