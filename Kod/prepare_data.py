import cv2
import numpy as np
import os
import yaml
import shutil
from pathlib import Path

def create_folder_structure(config_path="config.yaml"):
    """
    Kreira kompletnu strukturu foldera projekta ako ne postoji.
    """
    folders = [
        "data/raw/images",
        "data/raw/masks",
        "data/processed/images",
        "data/processed/masks",
        "models",
        "results",
        "logs",
        "src"
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Kreiranje __init__.py u src/ da bi Python video modul
    init_path = Path("src/__init__.py")
    if not init_path.exists():
        init_path.touch()

    print("Struktura foldera kreirana:")
    for folder in folders:
        print(f"  ✓ {folder}/")
    print("  ✓ src/__init__.py")


def slice_image_to_tiles(image_path, mask_path, output_images_dir,
                          output_masks_dir, tile_size=256, overlap=32):
    """
    Seče jednu veliku sliku (i njenu masku) na tile-ove fiksne veličine.

    Args:
        image_path: Putanja do drone snimka (original)
        mask_path:  Putanja do annotovane maske (isti naziv fajla, .png)
        overlap:    Preklapanje između tile-ova u pikselima (smanjuje artefakte na ivicama)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Greška: Slika nije pronađena: {image_path}")
        return 0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
    if mask_path and mask is None:
        print(f"Greška: Maska nije pronađena: {mask_path}")
        return 0

    h, w = image.shape[:2]
    base_name = Path(image_path).stem
    step = tile_size - overlap  # Korak sa preklapanjem
    tile_count = 0

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile_img = image[y:y + tile_size, x:x + tile_size]

            # Preskačemo tile-ove koji su previše tamni (oblaci, senke, nepotpune ivice)
            if _is_tile_valid(tile_img):
                tile_name = f"{base_name}_y{y:05d}_x{x:05d}.jpg"
                cv2.imwrite(os.path.join(output_images_dir, tile_name), tile_img)

                if mask is not None:
                    tile_mask = mask[y:y + tile_size, x:x + tile_size]
                    mask_name = f"{base_name}_y{y:05d}_x{x:05d}.png"
                    cv2.imwrite(os.path.join(output_masks_dir, mask_name), tile_mask)

                tile_count += 1

    return tile_count


def _is_tile_valid(tile, min_brightness=20, max_black_ratio=0.3):
    """
    Odbacuje tile-ove koji su uglavnom crni (ivice slike, oblaci).
    Vraća True ako je tile validan za trening.
    """
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < min_brightness)
    black_ratio = black_pixels / gray.size
    return black_ratio < max_black_ratio


def prepare_all(config_path="config.yaml"):
    """
    Glavni tok: čita konfiguraciju i procesira sve slike u data/raw/.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    raw_images_dir = cfg['data']['raw_images_dir']
    raw_masks_dir = cfg['data']['raw_masks_dir']
    out_images_dir = cfg['data']['tiles_images_dir']
    out_masks_dir = cfg['data']['tiles_masks_dir']
    tile_size = cfg['data']['tile_size']
    overlap = cfg['data']['tile_overlap']

    # Čišćenje starih tile-ova
    for d in [out_images_dir, out_masks_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # Procesiranje svakog snimka
    image_files = [
        f for f in os.listdir(raw_images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
    ]

    if not image_files:
        print(f"Nema slika u {raw_images_dir}")
        print("Stavi drone snimke u data/raw/images/ i maske u data/raw/masks/")
        return

    total_tiles = 0
    for img_file in image_files:
        img_path = os.path.join(raw_images_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        mask_path = os.path.join(raw_masks_dir, base_name + ".png")

        if not os.path.exists(mask_path):
            print(f"Upozorenje: Maska nije pronađena za {img_file}, preskačem.")
            continue

        count = slice_image_to_tiles(
            img_path, mask_path,
            out_images_dir, out_masks_dir,
            tile_size=tile_size, overlap=overlap
        )
        total_tiles += count
        print(f"  {img_file} -> {count} tile-ova")

    print(f"\nUkupno generisano: {total_tiles} tile-ova")
    print(f"Slike:  {out_images_dir}/")
    print(f"Maske:  {out_masks_dir}/")
    print("\nSledeći korak: python train.py")


if __name__ == "__main__":
    print("=== Priprema strukture foldera ===")
    create_folder_structure()

    print("\n=== Sekanje slika na tile-ove ===")
    prepare_all()
