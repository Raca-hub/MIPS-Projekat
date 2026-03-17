import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms

class LandSegmentation:
    
    def __init__(self, model_path=None):
        # Definišemo arhitekturu (WBS 1.2 / 4.1)
        # U-Net sa ResNet34 backbone-om je odličan balans brzine i preciznosti
        self.model = smp.Unet(
            encoder_name="resnet34",        
            encoder_weights="imagenet",     
            in_channels=3,                  
            classes=4,  # npr. 0: nepoznato, 1: šuma, 2: beton, 3: trava (WBS 1.1)
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
        color_map = {
            0: [0, 0, 0],       # Nepoznato - Crno
            1: [0, 255, 0],     # Šuma - Zeleno
            2: [128, 128, 128], # Beton/Put - Sivo
            3: [255, 255, 0]    # Polje - Žuto
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