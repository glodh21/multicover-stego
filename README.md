
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