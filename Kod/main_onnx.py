import cv2
import os
import sys
import json
import argparse
from datetime import datetime

# Uvoz modula - ISTI registration/change_analysis/visualization kao main.py
# (ne zavise od PyTorch-a, samo cv2/numpy), ali ONNX engine umesto PyTorch modela
from src.registration import ImageRegistration
from src.inference_lite import LandSegmentationLite
from src.change_analysis import detect_changes, change_metrics, load_class_names
from src.visualization import create_visual_report, create_transition_heatmap, plot_transition_matrix


def run_pipeline_onnx(img_path_old, img_path_new, onnx_model_path="models/best_model.onnx",
                       output_dir="results", skip_registration=False):
    """
    ONNX ekvivalent run_pipeline iz main.py - ISTI tok (registracija -> segmentacija
    T1/T2 -> transition matrica -> heat-mapa/izvestaj), ali koristi ONNX Runtime
    umesto PyTorch-a. Koristan za:
    1) testiranje da ONNX export stvarno daje iste/slicne rezultate kao PyTorch model
    2) simulaciju tacnog izvrsnog okruzenja kakvo bi bilo na Raspberry Pi/ARM uredjaju
       (bez GPU-a, bez pune PyTorch instalacije)
    """
    if not os.path.exists(onnx_model_path):
        print(f"Greška: {onnx_model_path} ne postoji. Prvo pokreni: python src\\export_onnx.py")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Inicijalizacija (ONNX Runtime)...")
    reg_engine = ImageRegistration()
    seg_engine = LandSegmentationLite(model_path=onnx_model_path)
    class_names = load_class_names()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 3: Registracija i poravnanje...")
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

    img_new = cv2.imread(img_path_new)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 4a: SSIM provera (orijentaciono)...")
    change_mask, change_percent = detect_changes(aligned_old, img_new)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 4b: ONNX segmentacija T1/T2 i praćenje promene tipa...")
    aligned_old_path = os.path.join(output_dir, f"_tmp_aligned_old_onnx_{timestamp}.jpg")
    cv2.imwrite(aligned_old_path, aligned_old)

    t_start = datetime.now()
    mask_t1 = seg_engine.process_large_image(aligned_old_path, tile_size=256)
    mask_t2 = seg_engine.process_large_image(img_path_new, tile_size=256)
    inference_seconds = (datetime.now() - t_start).total_seconds()
    os.remove(aligned_old_path)

    colored_land_mask = seg_engine.colorize_mask(mask_t2)

    metrics = change_metrics(mask_t1, mask_t2, class_names=class_names)
    transition_heatmap = create_transition_heatmap(mask_t1, mask_t2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faza 5: Generisanje izveštaja...")
    final_report = create_visual_report(aligned_old, img_new, change_mask, change_percent)

    output_path = os.path.join(output_dir, f"report_onnx_{timestamp}.jpg")
    cv2.imwrite(output_path, final_report)
    cv2.imwrite(os.path.join(output_dir, f"mask_onnx_{timestamp}.png"), colored_land_mask)
    cv2.imwrite(os.path.join(output_dir, f"transition_heatmap_onnx_{timestamp}.png"), transition_heatmap)
    plot_transition_matrix(
        metrics["transition_matrix"], class_names,
        save_path=os.path.join(output_dir, f"transition_matrix_onnx_{timestamp}.png")
    )

    report_json = {k: v for k, v in metrics.items() if k != "transition_matrix"}
    report_json["transition_matrix"] = metrics["transition_matrix"].tolist()
    report_json["ssim_change_percent"] = round(float(change_percent), 2)
    report_json["engine"] = "onnx_runtime"
    report_json["inference_seconds_total"] = round(inference_seconds, 3)
    with open(os.path.join(output_dir, f"change_report_onnx_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    print("-" * 30)
    print("ANALIZA ZAVRŠENA (ONNX Runtime)!")
    print(f"SSIM (generička) promena: {change_percent:.2f}%")
    print(f"Promena tipa zemljišta:   {metrics['change_ratio_percent']}%")
    print(f"Vreme ONNX inferecne (T1+T2): {inference_seconds:.2f}s")
    if metrics["transitions"]:
        top = metrics["transitions"][0]
        print(f"Najveća tranzicija: {top['from']} -> {top['to']} ({top['percentage']}%)")
    print(f"Rezultat sačuvan na: {output_path}")
    print(f"Izveštaj (JSON):      {output_dir}/change_report_onnx_{timestamp}.json")
    print("-" * 30)
    print("\nNapomena: vreme inferecne iznad je mereno na OVOM računaru (CPU), ne na")
    print("stvarnom Raspberry Pi-ju - koristi ga kao relativno poređenje sa PyTorch")
    print("verzijom (main.py), ne kao apsolutnu procenu brzine na ARM uređaju.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONNX Runtime verzija main.py - prati promenu tipa zemljista koristeci "
                     "ONNX model umesto PyTorch-a (simulira ARM/Raspberry Pi izvrsno okruzenje)."
    )
    parser.add_argument("img_old", help="Putanja do starije (T1) slike")
    parser.add_argument("img_new", help="Putanja do novije (T2) slike")
    parser.add_argument("--model", default="models/best_model.onnx",
                         help="Putanja do ONNX modela (podrazumevano models/best_model.onnx)")
    parser.add_argument("--skip-registration", action="store_true",
                         help="Preskoci SIFT poravnanje - koristi kad su T1/T2 slike vec poravnate "
                              "po konstrukciji (npr. Google Earth istorijski snimci sa zakljucanim pogledom)")

    args = parser.parse_args()
    run_pipeline_onnx(args.img_old, args.img_new, onnx_model_path=args.model,
                       skip_registration=args.skip_registration)
