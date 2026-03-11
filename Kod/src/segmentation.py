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