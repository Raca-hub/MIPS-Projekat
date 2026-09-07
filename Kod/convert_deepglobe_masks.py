"""
Konvertuje DeepGlobe Land Cover dataset (RGB masks + '_sat'/'_mask' sufiksi)
u format koji ocekuje ovaj projekat: grayscale maska sa ID klase po pikselu
(0=nepoznato, 1=suma, 2=beton, 3=polje), i slika+maska sa ISTIM imenom fajla.

Upotreba:
    python convert_deepglobe_masks.py <ulazni_folder> <data/raw/images> <data/raw/masks>

<ulazni_folder> treba da sadrzi parove: XXX_sat.jpg + XXX_mask.png
"""
import os
import sys
import shutil
import numpy as np
import cv2
from pathlib import Path

# DeepGlobe RGB boja -> ID klase u ovom projektu (config.yaml: 0=nepoznato, 1=suma, 2=beton, 3=polje)
# Format boje: (R, G, B)
DEEPGLOBE_TO_PROJECT = {
    (0, 255, 255): 2,    # Urban land       -> Beton
    (255, 255, 0): 3,    # Agriculture land -> Polje
    (255, 0, 255): 3,    # Rangeland        -> Polje
    (0, 255, 0):   1,    # Forest land      -> Suma
    (0, 0, 255):   0,    # Water            -> Nepoznato
    (255, 255, 255): 0,  # Barren land      -> Nepoznato
    (0, 0, 0):     0,    # Unknown          -> Nepoznato
}


def remap_mask(mask_rgb, tolerance=30):
    """
    Pretvara RGB DeepGlobe masku u grayscale masku ID klasa.
    Koristi najblizu boju (po Euklidskoj udaljenosti) da bi izdrzao
    kompresione artefakte na ivicama poligona (anti-aliasing).
    """
    h, w = mask_rgb.shape[:2]
    output = np.zeros((h, w), dtype=np.uint8)

    palette_colors = np.array(list(DEEPGLOBE_TO_PROJECT.keys()), dtype=np.int32)
    palette_classes = np.array(list(DEEPGLOBE_TO_PROJECT.values()), dtype=np.uint8)

    pixels = mask_rgb.reshape(-1, 3).astype(np.int32)

    # Za svaki piksel nadji najblizu paletnu boju (vektorizovano, brzo i za velike slike)
    # distances shape: (num_pixels, num_palette_colors)
    distances = np.sum((pixels[:, None, :] - palette_colors[None, :, :]) ** 2, axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    output_flat = palette_classes[nearest_idx]

    return output_flat.reshape(h, w)


def convert_folder(input_dir, out_images_dir, out_masks_dir):
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    sat_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_sat.jpg")])

    if not sat_files:
        print(f"Nema '_sat.jpg' fajlova u {input_dir}")
        return

    converted = 0
    for sat_file in sat_files:
        base_id = sat_file[:-len("_sat.jpg")]  # npr. "119_sat.jpg" -> "119"
        mask_file = f"{base_id}_mask.png"
        mask_path = os.path.join(input_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Upozorenje: nedostaje maska za {sat_file}, preskačem.")
            continue

        # Kopiraj sliku pod novim, usklađenim imenom
        src_img_path = os.path.join(input_dir, sat_file)
        dst_img_path = os.path.join(out_images_dir, f"{base_id}.jpg")
        shutil.copy2(src_img_path, dst_img_path)

        # Učitaj RGB masku (cv2 čita kao BGR, pa konvertujemo)
        mask_bgr = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        class_mask = remap_mask(mask_rgb)

        dst_mask_path = os.path.join(out_masks_dir, f"{base_id}.png")
        cv2.imwrite(dst_mask_path, class_mask)

        converted += 1
        if converted % 50 == 0:
            print(f"  Konvertovano {converted}/{len(sat_files)}...")

    print(f"\nGotovo: {converted} parova konvertovano.")
    print(f"Slike:  {out_images_dir}/")
    print(f"Maske:  {out_masks_dir}/ (grayscale, ID klase 0-3)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Upotreba: python convert_deepglobe_masks.py <ulazni_folder> <data/raw/images> <data/raw/masks>")
        sys.exit(1)

    convert_folder(sys.argv[1], sys.argv[2], sys.argv[3])
