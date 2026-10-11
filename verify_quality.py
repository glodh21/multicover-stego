from pathlib import Path
import numpy as np
from PIL import Image

pairs = [
    ("high_bike.png", "stego_high_bike.png"),
    ("low-sky.png", "stego_low-sky.png"),
    ("mid_house.png", "stego_mid_house.png"),
    ("mid_door.png", "stego_mid_door.png"),
    ("high_kayak.png", "stego_high_kayak.png"),
]

for cover_name, stego_name in pairs:
    cover_path = Path("images") / cover_name
    stego_path = Path("Stego") / stego_name

    if not cover_path.exists() or not stego_path.exists():
        print(f"Missing pair: {cover_name}")
        continue

    with Image.open(cover_path) as im:
        cover = np.asarray(im.convert("L"), dtype=np.int16)

    with Image.open(stego_path) as im:
        stego = np.asarray(im.convert("L"), dtype=np.int16)

    if cover.shape != stego.shape:
        print(f"{cover_name}: DIMENSION MISMATCH")
        continue

    diff = np.abs(cover - stego)
    changed = np.count_nonzero(diff)
    total = diff.size
    mse = np.mean((cover - stego) ** 2)

    print(f"\n{cover_name}")
    print(f"  Pixels changed: {changed:,}/{total:,}")
    print(f"  Changed pixels (%): {100 * changed / total:.6f}")
    print(f"  Maximum pixel difference: {diff.max()}")
    print(f"  Independently calculated MSE: {mse:.10f}")