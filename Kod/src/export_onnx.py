import torch
import os
import sys
import yaml

# Dodajemo koren projekta u putanju da bi Python video 'src' modul
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.segmentation import LandSegmentation

def export_to_onnx(config_path=None):
    """
    Konvertuje istrenirani PyTorch model (.pth) u ONNX format
    radi optimizacije za ARM platforme (Jetson, Raspberry Pi).

    config_path: putanja do config fajla (npr. "config_demo.yaml" za demo model).
    Podrazumevano "config.yaml" (glavni, pun model) ako se ne prosledi.
    Arhitektura (ResNet34/MobileNetV2) se čita iz ISTOG config-a kao putanje,
    da se izbegne neusklađenost (npr. pokušaj učitavanja MobileNetV2 težina
    u ResNet34 arhitekturu).
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if config_path is None:
        config_path = os.path.join(base_dir, "config.yaml")
    elif not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_path = os.path.join(base_dir, cfg['paths']['best_model_pth'])
    onnx_path = os.path.join(base_dir, cfg['paths']['best_model_onnx'])

    # Provera da li folder za modele postoji
    models_dir = os.path.dirname(onnx_path)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Napravljen folder: {models_dir}")

    # Provera da li .pth fajl postoji pre učitavanja
    if not os.path.exists(model_path):
        print(f"GRESKA: Model nije pronađen na putanji: {model_path}")
        return

    # 1. Inicijalizacija i učitavanje modela (arhitektura iz ISTOG config-a)
    print(f"Učitavam PyTorch model ({model_path}, config: {os.path.basename(config_path)})...")
    seg_engine = LandSegmentation(model_path=model_path, config_path=config_path)
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
    # python export_onnx.py             -> izvozi glavni model (config.yaml)
    # python export_onnx.py config_demo.yaml -> izvozi demo model
    arg_config = sys.argv[1] if len(sys.argv) > 1 else None
    export_to_onnx(config_path=arg_config)