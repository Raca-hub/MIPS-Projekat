import cv2
import os
import sys
import json
import argparse
from datetime import datetime

# Uvoz tvojih modula
from src.registration import ImageRegistration
from src.segmentation import LandSegmentation
from src.change_analysis import detect_changes, change_metrics, load_class_names
from src.visualization import create_visual_report, create_transition_heatmap, plot_transition_matrix
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

def run_full_setup(config_path="config.yaml", force_prepare=False, force_train=False,
                    force_export=False, skip_evaluate=False):
    """
    Orkestracija celog toka PRE analize: priprema podataka -> trening -> evaluacija -> ONNX export.

    Svaki korak se PRESKAČE ako je već urađen (osim ako se eksplicitno ne zatraži
    ponavljanje sa --force-prepare / --force-train / --force-export) - trening ume
    da traje satima, ne treba ga slučajno pokrenuti ponovo svaki put kad se main.py zove.
    """
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    tiles_dir = cfg['data']['tiles_images_dir']
    model_path = cfg['paths']['best_model_pth']
    onnx_path = cfg['paths']['best_model_onnx']

    # 1. Priprema podataka (tile-ovanje)
    tiles_exist = os.path.exists(tiles_dir) and len(os.listdir(tiles_dir)) > 0
    if force_prepare or not tiles_exist:
        print(f"\n{'='*50}\n[1/4] PRIPREMA PODATAKA\n{'='*50}")
        from prepare_data import create_folder_structure, prepare_all
        create_folder_structure(config_path)
        prepare_all(config_path)
    else:
        print(f"[1/4] Priprema podataka preskočena (tile-ovi već postoje u {tiles_dir})")

    # 2. Trening
    model_exists = os.path.exists(model_path)
    if force_train or not model_exists:
        print(f"\n{'='*50}\n[2/4] TRENING MODELA\n{'='*50}")
        from train import train
        train(config_path=config_path)
    else:
        print(f"[2/4] Trening preskočen (model već postoji: {model_path})")

    # 3. Evaluacija
    if not skip_evaluate:
        print(f"\n{'='*50}\n[3/4] EVALUACIJA\n{'='*50}")
        from evaluate import evaluate
        evaluate(config_path=config_path, split="val")
    else:
        print("[3/4] Evaluacija preskočena (--skip-evaluate)")

    # 4. Export u ONNX
    onnx_exists = os.path.exists(onnx_path)
    if force_export or not onnx_exists:
        print(f"\n{'='*50}\n[4/4] EXPORT U ONNX\n{'='*50}")
        from src.export_onnx import export_to_onnx
        export_to_onnx()
    else:
        print(f"[4/4] Export preskočen (ONNX model već postoji: {onnx_path})")

    print(f"\n{'='*50}\nSETUP ZAVRŠEN - prelazim na analizu T1/T2\n{'='*50}\n")


def run_pipeline(img_path_old, img_path_new, output_dir="results", skip_registration=False):
    """
    Glavni procesni pipeline: WBS 3.0 -> 4.0 -> 5.0

    Faza 4 sada radi DVE stvari:
    1. Generičku SSIM proveru (brza, orijentaciona - i provera kvaliteta poravnanja)
    2. Praćenje promene TIPA zemljišta - segmentacija T1 i T2 slike, pa poređenje
       KLASA piksel po piksel (class_transition_matrix / change_metrics), što je
       suština teme projekta.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Inicijalizacija sistema...")

    # 1. Inicijalizacija (WBS 1.3)
    reg_engine = ImageRegistration()
    seg_engine = LandSegmentation(model_path="models/best_model.pth")
    class_names = load_class_names()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 3: Registracija i poravnanje...")
    # 2. Registracija (WBS 3.3 - 3.4)
    if skip_registration:
        print("  (--skip-registration: slike se tretiraju kao već poravnate, npr. "
              "Google Earth istorijski snimci sa zaključanim pogledom)")
        aligned_old = cv2.imread(img_path_old)
        if aligned_old is None:
            print(f"Greška: Nije moguće učitati {img_path_old}")
            return
    else:
        aligned_old, validation_view = reg_engine.register(img_path_old, img_path_new)

        if aligned_old is None:
            print(f"Greška: Registracija nije uspela. Razlog: {validation_view}")
            print("Savet: ako su T1/T2 slike već poravnate po konstrukciji (npr. Google "
                  "Earth sa zaključanim pogledom, samo menjana godina), probaj "
                  "--skip-registration da preskočiš SIFT poravnanje.")
            return

    # Učitavanje nove slike za dalju obradu
    img_new = cv2.imread(img_path_new)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 4a: SSIM provera (orijentaciono)...")
    # 3a. Generička promena (SSIM) - brza provera, ne govori ŠTA se promenilo
    change_mask, change_percent = detect_changes(aligned_old, img_new)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 4b: Segmentacija T1/T2 i praćenje promene tipa...")
    # 3b. Segmentacija OBE slike (poravnata stara T1 i nova T2)
    aligned_old_path = os.path.join(output_dir, f"_tmp_aligned_old_{timestamp}.jpg")
    cv2.imwrite(aligned_old_path, aligned_old)
    mask_t1 = seg_engine.process_large_image(aligned_old_path, tile_size=256)
    mask_t2 = seg_engine.process_large_image(img_path_new, tile_size=256)
    os.remove(aligned_old_path)

    colored_land_mask = seg_engine.colorize_mask(mask_t2)

    # 3c. Matrica prelaza klasa + konkretne, imenovane metrike promene tipa
    metrics = change_metrics(mask_t1, mask_t2, class_names=class_names)
    transition_heatmap = create_transition_heatmap(mask_t1, mask_t2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 5: Generisanje izveštaja...")
    # 4. Vizuelizacija i čuvanje (WBS 4.4 / 5.5)
    final_report = create_visual_report(aligned_old, img_new, change_mask, change_percent)

    output_path = os.path.join(output_dir, f"report_{timestamp}.jpg")
    cv2.imwrite(output_path, final_report)
    cv2.imwrite(os.path.join(output_dir, f"mask_{timestamp}.png"), colored_land_mask)
    cv2.imwrite(os.path.join(output_dir, f"transition_heatmap_{timestamp}.png"), transition_heatmap)
    plot_transition_matrix(
        metrics["transition_matrix"], class_names,
        save_path=os.path.join(output_dir, f"transition_matrix_{timestamp}.png")
    )

    # Sačuvaj i mašinski čitljiv izveštaj (za dokumentaciju/dalju obradu)
    report_json = {k: v for k, v in metrics.items() if k != "transition_matrix"}
    report_json["transition_matrix"] = metrics["transition_matrix"].tolist()
    report_json["ssim_change_percent"] = round(float(change_percent), 2)
    with open(os.path.join(output_dir, f"change_report_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    print("-" * 30)
    print(f"ANALIZA ZAVRŠENA!")
    print(f"SSIM (generička) promena: {change_percent:.2f}%")
    print(f"Promena tipa zemljišta:   {metrics['change_ratio_percent']}%")
    if metrics["transitions"]:
        top = metrics["transitions"][0]
        print(f"Najveća tranzicija: {top['from']} -> {top['to']} ({top['percentage']}%)")
    print(f"Rezultat sačuvan na: {output_path}")
    print(f"Izveštaj (JSON):      {output_dir}/change_report_{timestamp}.json")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prati promenu tipa zemljista izmedju dve slike (T1/T2)."
    )
    parser.add_argument("img_old", help="Putanja do starije (T1) slike")
    parser.add_argument("img_new", help="Putanja do novije (T2) slike")
    parser.add_argument("--all", action="store_true",
                         help="Pre analize, pokreni ceo setup: priprema podataka -> trening -> evaluacija -> ONNX export "
                              "(svaki korak se preskace ako je vec uradjen)")
    parser.add_argument("--config", default="config.yaml", help="Putanja do config fajla (podrazumevano config.yaml)")
    parser.add_argument("--force-prepare", action="store_true", help="Ponovo iseci tile-ove i ako vec postoje")
    parser.add_argument("--force-train", action="store_true", help="Ponovo treniraj model i ako vec postoji")
    parser.add_argument("--force-export", action="store_true", help="Ponovo eksportuj ONNX i ako vec postoji")
    parser.add_argument("--skip-registration", action="store_true",
                         help="Preskoci SIFT poravnanje - koristi kad su T1/T2 slike vec poravnate "
                              "po konstrukciji (npr. Google Earth istorijski snimci sa zakljucanim pogledom)")
    parser.add_argument("--skip-evaluate", action="store_true", help="Preskoci evaluate.py korak u --all rezimu")

    args = parser.parse_args()

    if args.all:
        run_full_setup(
            config_path=args.config,
            force_prepare=args.force_prepare,
            force_train=args.force_train,
            force_export=args.force_export,
            skip_evaluate=args.skip_evaluate,
        )

    run_pipeline(args.img_old, args.img_new, skip_registration=args.skip_registration)