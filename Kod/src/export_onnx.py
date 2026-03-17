import torch
import os
import sys

# Dodajemo koren projekta u putanju da bi Python video 'src' modul
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.segmentation import LandSegmentation

def export_to_onnx():
    """
    Konvertuje istrenirani PyTorch model (.pth) u ONNX format 
    radi optimizacije za ARM platforme (Jetson, Raspberry Pi).
    """
    # Definišemo putanje (relativno u odnosu na koren projekta)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, "models", "best_model.pth")
    onnx_path = os.path.join(base_dir, "models", "best_model.onnx")
    
    # Provera da li folder za modele postoji
    models_dir = os.path.dirname(onnx_path)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Napravljen folder: {models_dir}")

    # Provera da li .pth fajl postoji pre učitavanja
    if not os.path.exists(model_path):
        print(f"GRESKA: Model nije pronađen na putanji: {model_path}")
        return

    # 1. Inicijalizacija i učitavanje modela
    print("Učitavam PyTorch model...")
    seg_engine = LandSegmentation(model_path=model_path)
    model = seg_engine.model
    model.eval()

    # 2. Kreiranje testnog (dummy) ulaza
    # Standardna veličina za tvoj model je 256x256 (prema segmentation.py)
    dummy_input = torch.randn(1, 3, 256, 256).to(seg_engine.device)

    # 3. Eksportovanje
    print(f"Eksportujem model u {onnx_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,      # Čuva istrenirane težine unutar fajla
            opset_version=11,        # Verzija kompatibilna sa većinom ARM runtime-ova
            do_constant_folding=True, # Optimizacija modela tokom eksporta
            input_names=['input'],   # Ime ulaznog čvora
            output_names=['output'], # Ime izlaznog čvora
            dynamic_axes={           # Dozvoljava promenu batch size-a tokom rada
                'input': {0: 'batch_size'}, 
                'output': {0: 'batch_size'}
            }
        )
        print("✅ Eksport završen uspešno!")
    except Exception as e:
        print(f"❌ Došlo je do greške tokom eksporta: {e}")

if __name__ == "__main__":
    export_to_onnx()