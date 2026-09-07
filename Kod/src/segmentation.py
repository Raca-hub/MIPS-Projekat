import os
import torch
import cv2
import numpy as np
import yaml
import segmentation_models_pytorch as smp
from torchvision import transforms

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


class LandSegmentation:

    def __init__(self, model_path=None, config_path=_DEFAULT_CONFIG_PATH):
        # Arhitektura se sada čita iz config.yaml (WBS 1.2 / 4.1) umesto da bude
        # hardkodirana, tako da promena "encoder" u config.yaml stvarno ima efekta.
        # Za slab/bez GPU-a preporučeno: encoder: "mobilenet_v2" (3-5x manje parametara
        # i memorije od resnet34, uz malo niži mIoU).
        m_cfg = {"encoder": "resnet34", "encoder_weights": "imagenet", "in_channels": 3, "num_classes": 4}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                full_cfg = yaml.safe_load(f)
            m_cfg.update(full_cfg.get("model", {}))

        self.model = smp.Unet(
            encoder_name=m_cfg["encoder"],
            encoder_weights=m_cfg["encoder_weights"],
            in_channels=m_cfg["in_channels"],
            classes=m_cfg["num_classes"],
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Transformacije za ulaznu sliku
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)), # Veličina zavisi od WBS 2.5 (tiling)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def process_large_image(self, image_path, tile_size=256):
        """
        Deli veliku sliku na manje delove (tiles), 
        procesira svaki i spaja ih nazad u jednu masku.
        """
        full_img = cv2.imread(image_path)
        h, w, _ = full_img.shape
        
        # Kreiramo praznu masku istih dimenzija (samo visina i širina)
        full_mask = np.zeros((h, w), dtype=np.uint8)

        # Prolazimo kroz sliku u koracima veličine tile_size
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Određivanje granica isečka (pazimo na ivice slike)
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = full_img[y:y_end, x:x_end]
                
                # Ako je isečak manji od 256x256 (na ivicama), dopunimo ga (padding) ili resize
                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    tile_resized = cv2.resize(tile, (tile_size, tile_size))
                else:
                    tile_resized = tile

                # Predikcija za taj konkretan isečak
                # (Koristimo tvoju postojeću transformaciju i model)
                input_tensor = self.transform(tile_resized).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(input_tensor)
                    tile_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                # Vraćanje isečka na originalnu veličinu ako je bilo resizing-a
                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    tile_mask = cv2.resize(tile_mask, (x_end - x, y_end - y), interpolation=cv2.INTER_NEAREST)

                # Upisivanje u veliku masku
                full_mask[y:y_end, x:x_end] = tile_mask

        return full_mask

    def predict(self, image_path):
        """
        Generisanje maske za jedan fragment (tile) snimka (WBS 4.2).
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Priprema tenzora
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            # Uzimamo klasu sa najvećom verovatnoćom za svaki piksel
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        return mask

    def colorize_mask(self, mask):
        """
        Pretvaranje numeričke maske u boju radi vizuelne provere.
        """
        # Boje su u BGR redosledu (OpenCV konvencija) jer se rezultat čuva preko
        # cv2.imwrite - RGB žuta [255,255,0] bi se inače prikazala kao cijan.
        color_map = {
            0: [0, 0, 0],       # Nepoznato - Crno
            1: [0, 255, 0],     # Šuma - Zeleno (simetrično u BGR/RGB)
            2: [128, 128, 128], # Beton/Put - Sivo (simetrično u BGR/RGB)
            3: [0, 255, 255]    # Polje - Žuto (BGR: B=0, G=255, R=255)
        }
        
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in color_map.items():
            colored_mask[mask == class_id] = color
            
        return colored_mask

if __name__ == "__main__":
    segmentor = LandSegmentation()
    
    # Testiranje na jednom fragmentu (WBS 2.5)
    mask = segmentor.predict("tile_01.jpg")
    colored = segmentor.colorize_mask(mask)
    
    cv2.imwrite("mask_result.png", colored)
    print("Maska generisana i sačuvana.")