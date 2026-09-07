import onnxruntime as ort
import numpy as np
import cv2

class LandSegmentationLite:
    def __init__(self, model_path="models/best_model.onnx"):
        # Pokretanje sesije (na Jetsonu će koristiti CUDA ako je instaliran onnxruntime-gpu)
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def _predict_tile(self, tile_bgr, tile_size=256):
        """Predikcija za JEDAN tile (256x256 ili manji, uvek se skalira na tile_size)."""
        image_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (tile_size, tile_size))

        # Normalizacija (ImageNet standard koji koristiš u segmentation.py)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_data = (resized / 255.0 - mean) / std

        # Formatiranje za ONNX (Batch, Channel, H, W)
        input_data = input_data.transpose(2, 0, 1).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        outputs = self.session.run(None, {self.input_name: input_data})
        mask = np.argmax(outputs[0], axis=1).squeeze()
        return mask

    def predict(self, image_path):
        """
        Predikcija za MALU sliku (cela slika se samo skalira na 256x256).
        Za velike slike (T1/T2 snimci u main_onnx.py) koristi process_large_image
        umesto ove metode - ovde bi resize cele slike na 256x256 uništio detalje.
        """
        image = cv2.imread(image_path)
        return self._predict_tile(image)

    def process_large_image(self, image_path, tile_size=256):
        """
        ONNX ekvivalent LandSegmentation.process_large_image iz segmentation.py -
        deli veliku sliku na tile-ove, predikuje svaki, spaja nazad u punu masku.
        Identična logika kao PyTorch verzija, samo koristi ONNX Runtime sesiju.
        """
        full_img = cv2.imread(image_path)
        h, w, _ = full_img.shape
        full_mask = np.zeros((h, w), dtype=np.uint8)

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)

                tile = full_img[y:y_end, x:x_end]
                tile_mask = self._predict_tile(tile, tile_size)

                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    tile_mask = cv2.resize(
                        tile_mask.astype(np.uint8), (x_end - x, y_end - y),
                        interpolation=cv2.INTER_NEAREST
                    )

                full_mask[y:y_end, x:x_end] = tile_mask

        return full_mask

    def colorize_mask(self, mask):
        # Boje u BGR redosledu (za cv2.imwrite) - RGB zuta [255,255,0] bi se
        # inace prikazala kao cijan.
        color_map = {
            0: [0, 0, 0], 1: [0, 255, 0], 2: [128, 128, 128], 3: [0, 255, 255]
        }
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in color_map.items():
            colored[mask == label] = color
        return colored