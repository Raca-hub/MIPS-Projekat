import cv2
import os
import sys
from datetime import datetime

# Uvoz tvojih modula
from src.registration import ImageRegistration
from src.segmentation import LandSegmentation
from src.change_analysis import detect_changes
from src.visualization import create_visual_report
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def initialize_engine(model_path):
    if HAS_TORCH and model_path.endswith(".pth"):
        print("Korišćenje PyTorch engine-a (Desktop mode)")
        from src.segmentation import LandSegmentation
        return LandSegmentation(model_path)
    else:
        print("Korišćenje ONNX Lite engine-a (ARM/Mobile mode)")
        from src.inference_lite import LandSegmentationLite
        onnx_path = model_path.replace(".pth", ".onnx")
        return LandSegmentationLite(onnx_path)

def run_pipeline(img_path_old, img_path_new, output_dir="results"):
    """
    Glavni procesni pipeline: WBS 3.0 -> 4.0 -> 5.0
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Inicijalizacija sistema...")
    
    # 1. Inicijalizacija (WBS 1.3)
    reg_engine = ImageRegistration()
    seg_engine = LandSegmentation(model_path="models/best_model.pth")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 3: Registracija i poravnanje...")
    # 2. Registracija (WBS 3.3 - 3.4)
    aligned_old, validation_view = reg_engine.register(img_path_old, img_path_new)
    
    if aligned_old is None:
        print("Greška: Registracija nije uspela. Proverite kvalitet snimaka.")
        return

    # Učitavanje nove slike za dalju obradu
    img_new = cv2.imread(img_path_new)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 4: Segmentacija i detekcija promena...")
    # 3. Analiza promena (WBS 4.3)
    # Koristimo SSIM ili tvoj AI model za masku promena
    change_mask, change_percent = detect_changes(aligned_old, img_new)

    # 4. Generisanje maski tipova zemljišta (WBS 4.2)
    # Opciono: mozes segmentisati obe slike da vidiš ŠTA se tačno promenilo (npr. šuma u beton)
    land_mask_new = seg_engine.predict(img_path_new)
    colored_land_mask = seg_engine.colorize_mask(land_mask_new)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 5: Generisanje izveštaja...")
    # 5. Vizuelizacija i čuvanje (WBS 4.4 / 5.5)
    final_report = create_visual_report(aligned_old, img_new, change_mask, change_percent)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"report_{timestamp}.jpg")
    cv2.imwrite(output_path, final_report)
    cv2.imwrite(os.path.join(output_dir, f"mask_{timestamp}.png"), colored_land_mask)

    print("-" * 30)
    print(f"ANALIZA ZAVRŠENA!")
    print(f"Procenat promene: {change_percent:.2f}%")
    print(f"Rezultat sačuvan na: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    # Primer pokretanja sa komandne linije: python main.py staro.jpg novo.jpg
    if len(sys.argv) < 3:
        print("Upotreba: python main.py <putanja_stara_slika> <putanja_nova_slika>")
    else:
        run_pipeline(sys.argv[1], sys.argv[2])