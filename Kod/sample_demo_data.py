"""
Bira mali, nasumican podskup PAROVA direktno iz sirove DeepGlobe arhive
(data/deepglobe_raw/train/, format XXX_sat.jpg + XXX_mask.png, RGB maske) i
konvertuje SAMO njih (ne svih 803) u data/demo_raw/ - za brz, zivi demo
treninga na prezentaciji (config_demo.yaml, par epoha, gotovo za 1-2 minuta).

Ovo je brze od uzorkovanja iz vec konvertovanog data/raw/, jer ne zahteva
da prethodno pokrenes convert_deepglobe_masks.py na svih 803 slike.

Ovo NE dira tvoje prave podatke niti pravi model.

Upotreba:
    python sample_demo_data.py [broj_slika] [putanja_do_deepglobe_train]

Podrazumevano: 40 slika, izvor "data/deepglobe_raw/train" (dovoljno za par
brzih epoha, nedovoljno za dobar mIoU - namerno, cilj demoa je da pokaze
da trening RADI, ne da nauci dobar model).
"""
import os
import sys
import random
from pathlib import Path

import cv2

from convert_deepglobe_masks import remap_mask

SRC_DIR_DEFAULT = "data/deepglobe_raw/train"
DST_IMAGES = "data/demo_raw/images"
DST_MASKS = "data/demo_raw/masks"


def sample_demo_data(n=40, src_dir=SRC_DIR_DEFAULT, seed=42):
    if not os.path.exists(src_dir):
        print(f"Greška: {src_dir} ne postoji. Proveri da si raspakovao DeepGlobe "
              f"arhivu (train deo, sa XXX_sat.jpg + XXX_mask.png parovima) tu.")
        return

    Path(DST_IMAGES).mkdir(parents=True, exist_ok=True)
    Path(DST_MASKS).mkdir(parents=True, exist_ok=True)

    sat_files = sorted([f for f in os.listdir(src_dir) if f.endswith("_sat.jpg")])

    if not sat_files:
        print(f"Nema '_sat.jpg' fajlova u {src_dir}.")
        return

    random.seed(seed)
    n = min(n, len(sat_files))
    sample = random.sample(sat_files, n)

    converted = 0
    for sat_file in sample:
        base_id = sat_file[:-len("_sat.jpg")]
        mask_file = f"{base_id}_mask.png"
        mask_path = os.path.join(src_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Upozorenje: nedostaje maska za {sat_file}, preskačem.")
            continue

        # Kopiraj sliku (samo promeni ime, bez konverzije)
        src_img_path = os.path.join(src_dir, sat_file)
        img = cv2.imread(src_img_path)
        cv2.imwrite(os.path.join(DST_IMAGES, f"{base_id}.jpg"), img)

        # Konvertuj RGB DeepGlobe masku u grayscale ID-klase (isti postupak
        # kao convert_deepglobe_masks.py, ali samo za ovih N izabranih parova)
        mask_bgr = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        class_mask = remap_mask(mask_rgb)
        cv2.imwrite(os.path.join(DST_MASKS, f"{base_id}.png"), class_mask)

        converted += 1

    print(f"Demo podskup spreman: {converted} parova konvertovano u {DST_IMAGES} / {DST_MASKS}")
    print("\nSledeci koraci za demo:")
    print("  python prepare_data.py config_demo.yaml")
    print("  python train.py config_demo.yaml")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    src = sys.argv[2] if len(sys.argv) > 2 else SRC_DIR_DEFAULT
    sample_demo_data(n, src_dir=src)
