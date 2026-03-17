import onnxruntime as ort
import numpy as np
import cv2

class LandSegmentationLite:
    def __init__(self, model_path="models/best_model.onnx"):
        # Pokretanje sesije (na Jetsonu će koristiti CUDA ako je instaliran onnxruntime-gpu)
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path):
        # Učitavanje i preprocesiranje kao u tvom originalnom kodu
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (256, 256))
        
        # Normalizacija (ImageNet standard koji koristiš u segmentation.py)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_data = (resized / 255.0 - mean) / std
        
        # Formatiranje za ONNX (Batch, Channel, H, W)
        input_data = input_data.transpose(2, 0, 1).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        # Inferencija
        outputs = self.session.run(None, {self.input_name: input_data})
        mask = np.argmax(outputs[0], axis=1).squeeze()
        
        return mask

    def colorize_mask(self, mask):
        # Kopiramo tvoju logiku bojenja iz originalnog fajla
        color_map = {
            0: [0, 0, 0], 1: [0, 255, 0], 2: [128, 128, 128], 3: [255, 255, 0]
        }
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in color_map.items():
            colored[mask == label] = color
        return colored