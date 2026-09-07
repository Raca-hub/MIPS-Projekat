"""
Kopira mali, nasumican podskup vec konvertovanih slika/maski iz data/raw/
u data/demo_raw/ - za brz, zivi demo treninga na prezentaciji
(config_demo.yaml, par epoha, gotovo za 1-2 minuta).

Ovo NE dira tvoje prave podatke niti pravi model - samo kopira (ne premesta).

Upotreba:
    python sample_demo_data.py [broj_slika]

Podrazumevano: 40 slika (dovoljno za par brzih epoha, nedovoljno za dobar mIoU
- namerno, cilj demoa je da pokaze da trening RADI, ne da nauci dobar model).
"""
import os
import sys
import random
import shutil
from pathlib import Path

SRC_IMAGES = "data/raw/images"
SRC_MASKS = "data/raw/masks"
DST_IMAGES = "data/demo_raw/images"
DST_MASKS = "data/demo_raw/masks"


def sample_demo_data(n=40, seed=42):
    if not os.path.exists(SRC_IMAGES):
        print(f"Greška: {SRC_IMAGES} ne postoji. Prvo pokreni convert_deepglobe_masks.py.")
        return

    Path(DST_IMAGES).mkdir(parents=True, exist_ok=True)
    Path(DST_MASKS).mkdir(parents=True, exist_ok=True)

    all_images = [
        f for f in os.listdir(SRC_IMAGES)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
    ]

    if not all_images:
        print(f"Nema slika u {SRC_IMAGES}.")
        return

    random.seed(seed)
    n = min(n, len(all_images))
    sample = random.sample(all_images, n)

    copied = 0
    for img_file in sample:
        base_name = os.path.splitext(img_file)[0]
        mask_file = base_name + ".png"
        src_mask_path = os.path.join(SRC_MASKS, mask_file)

        if not os.path.exists(src_mask_path):
            continue

        shutil.copy2(os.path.join(SRC_IMAGES, img_file), os.path.join(DST_IMAGES, img_file))
        shutil.copy2(src_mask_path, os.path.join(DST_MASKS, mask_file))
        copied += 1

    print(f"Demo podskup spreman: {copied} parova kopirano u {DST_IMAGES} / {DST_MASKS}")
    print("\nSledeci koraci za demo:")
    print("  python prepare_data.py config_demo.yaml")
    print("  python train.py config_demo.yaml")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sample_demo_data(n)
