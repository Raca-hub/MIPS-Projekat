import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import yaml

class LandDataset(Dataset):
    """
    PyTorch Dataset za učitavanje parova (slika, maska) iz tile foldera.
    Očekuje strukturu:
        data/processed/images/  -> tile_001.jpg, tile_002.jpg, ...
        data/processed/masks/   -> tile_001.png, tile_002.png, ...
    """

    def __init__(self, images_dir, masks_dir, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment

        # Automatski pronalazimo sve parove slika i maski
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
        ])

        # Proveravamo da svaka slika ima odgovarajuću masku
        valid_pairs = []
        for img_name in self.image_files:
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + ".png"  # Maske su uvek .png
            mask_path = os.path.join(masks_dir, mask_name)
            if os.path.exists(mask_path):
                valid_pairs.append((img_name, mask_name))
            else:
                print(f"Upozorenje: Maska nije pronađena za {img_name}, preskačem.")

        self.pairs = valid_pairs
        print(f"Dataset učitan: {len(self.pairs)} validnih parova (slika, maska).")

        # Standardizacija slike (ImageNet norme - konzistentno sa segmentation.py)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]

        # Učitavanje slike
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Učitavanje maske (grayscale - svaki piksel je ID klase: 0, 1, 2, 3)
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Augmentacija (samo za trening skup)
        if self.augment:
            image, mask = self._augment(image, mask)

        # Konvertovanje u tenzore
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = self.normalize(image_tensor)

        # Maska mora biti LongTensor (int64) za CrossEntropyLoss
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor

    def _augment(self, image, mask):
        """Ručne augmentacije konzistentne na slici i masci."""
        # Horizontalni flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Vertikalni flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Rotacija (90, 180 ili 270 stepeni)
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)

        # Promena osvetljenosti i kontrasta (samo na slici, ne masci)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # kontrast
            beta = np.random.randint(-20, 20)     # osvetljenost
            image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return image, mask


def create_dataloaders(config_path="config.yaml"):
    """
    Kreira train i validation DataLoader-e na osnovu config.yaml.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    images_dir = cfg['data']['tiles_images_dir']
    masks_dir = cfg['data']['tiles_masks_dir']
    train_split = cfg['data']['train_split']
    seed = cfg['data']['seed']
    batch_size = cfg['training']['batch_size']
    num_workers = cfg['training']['num_workers']
    pin_memory = cfg['training']['pin_memory']

    # Ceo dataset (bez augmentacije za početak, dodajemo posle splita)
    full_dataset = LandDataset(images_dir, masks_dir, augment=False)

    # Split na train i validation
    total = len(full_dataset)
    train_size = int(total * train_split)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Uključujemo augmentaciju samo za trening skup
    # (direktno modifikujemo dataset objekat u subsets-u)
    train_subset.dataset = LandDataset(images_dir, masks_dir, augment=True)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Train: {train_size} tile-ova | Validacija: {val_size} tile-ova")
    return train_loader, val_loader


if __name__ == "__main__":
    # Brzi test - proveravamo da li DataLoader radi
    train_loader, val_loader = create_dataloaders()
    images, masks = next(iter(train_loader))
    print(f"Batch slika: {images.shape}  (batch, channels, H, W)")
    print(f"Batch maski: {masks.shape}   (batch, H, W)")
    print(f"Vrednosti klasa u masci: {torch.unique(masks).tolist()}")
