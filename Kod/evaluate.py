import os
import sys
import yaml
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.segmentation import LandSegmentation
from dataset import LandDataset
from torch.utils.data import DataLoader


def evaluate(config_path="config.yaml", split="val"):
    """
    Evaluira model na validacionom ili test skupu.
    Generiše:
        - mIoU (mean Intersection over Union)
        - Per-class IoU, Precision, Recall, F1
        - Confusion matrix
        - Vizuelni primeri predikcija
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg['model']['num_classes']
    class_names = [cfg['classes'][i] for i in range(num_classes)]
    model_path = cfg['paths']['best_model_pth']
    results_dir = cfg['paths']['results_dir']
    images_dir = cfg['data']['tiles_images_dir']
    masks_dir = cfg['data']['tiles_masks_dir']

    os.makedirs(results_dir, exist_ok=True)

    # Učitavanje modela
    if not os.path.exists(model_path):
        print(f"Greška: Model nije pronađen na {model_path}")
        print("Prvo pokreni: python train.py")
        return

    print(f"Učitavam model: {model_path}")
    seg_engine = LandSegmentation(model_path=model_path)
    device = seg_engine.device

    # Dataset i loader
    dataset = LandDataset(images_dir, masks_dir, augment=False)
    train_size = int(len(dataset) * cfg['data']['train_split'])
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg['data']['seed'])
    )

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Evaluiram na {len(val_dataset)} tile-ova...\n")

    # Inicijalizacija akumulatora
    all_preds = []
    all_targets = []
    seg_engine.model.eval()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = seg_engine.model(images)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            targets = masks.squeeze(0).numpy()
            all_preds.append(preds.flatten())
            all_targets.append(targets.flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # ─────────────────────────────────────────
    # Per-class metrike
    # ─────────────────────────────────────────
    print("=" * 55)
    print(f"{'Klasa':<15} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 55)

    iou_per_class = []
    for c in range(num_classes):
        pred_c = (all_preds == c)
        target_c = (all_targets == c)

        tp = (pred_c & target_c).sum()
        fp = (pred_c & ~target_c).sum()
        fn = (~pred_c & target_c).sum()
        intersection = tp
        union = tp + fp + fn

        iou = intersection / union if union > 0 else float('nan')
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        iou_per_class.append(iou)
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
        print(f"{class_names[c]:<15} {iou_str:>8} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f}")

    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    print("-" * 55)
    print(f"{'mIoU':<15} {mean_iou:>8.4f}")
    print("=" * 55)

    # ─────────────────────────────────────────
    # Pixel Accuracy
    # ─────────────────────────────────────────
    pixel_acc = (all_preds == all_targets).mean()
    print(f"\nPixel Accuracy: {pixel_acc:.4f} ({pixel_acc*100:.2f}%)")

    # ─────────────────────────────────────────
    # Confusion Matrix
    # ─────────────────────────────────────────
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    # Normalizujemo po redu (true label) da vidimo recall po klasi
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Apsolutne vrednosti
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp1.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Confusion Matrix (apsolutno)', fontsize=13)
    axes[0].tick_params(axis='x', rotation=30)

    # Normalizovane vrednosti
    disp2 = ConfusionMatrixDisplay(confusion_matrix=np.round(cm_normalized, 2), display_labels=class_names)
    disp2.plot(ax=axes[1], colorbar=False, cmap='Blues')
    axes[1].set_title('Confusion Matrix (normalizovano)', fontsize=13)
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix sačuvana: {cm_path}")

    # ─────────────────────────────────────────
    # Vizuelni primeri predikcija
    # ─────────────────────────────────────────
    _save_visual_examples(
        loader=loader,
        seg_engine=seg_engine,
        device=device,
        cfg=cfg,
        results_dir=results_dir,
        num_examples=5
    )

    # ─────────────────────────────────────────
    # Čuvanje sažetog izveštaja
    # ─────────────────────────────────────────
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Evaluacioni izveštaj\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Broj tile-ova: {len(val_dataset)}\n\n")
        f.write(f"mIoU: {mean_iou:.4f}\n")
        f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n\n")
        f.write(f"{'Klasa':<15} {'IoU':>8}\n")
        f.write("-" * 25 + "\n")
        for i, name in enumerate(class_names):
            iou = iou_per_class[i]
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
            f.write(f"{name:<15} {iou_str:>8}\n")

    print(f"Izveštaj sačuvan: {report_path}")
    return mean_iou


def _save_visual_examples(loader, seg_engine, device, cfg, results_dir, num_examples=5):
    """Čuva primere: originalna slika | tačna maska | predikcija."""
    color_map = cfg['class_colors']
    example_count = 0

    seg_engine.model.eval()
    with torch.no_grad():
        for images, masks in loader:
            if example_count >= num_examples:
                break

            images_dev = images.to(device)
            outputs = seg_engine.model(images_dev)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Originalna slika (denormalizacija)
            img_np = images.squeeze(0).permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = np.clip((img_np * std + mean) * 255, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Bojenje maski
            gt_mask = masks.squeeze(0).numpy().astype(np.uint8)
            gt_colored = _colorize(gt_mask, color_map)
            pred_colored = _colorize(preds, color_map)

            # Kombinovani prikaz
            combined = np.hstack([img_bgr, gt_colored, pred_colored])
            out_path = os.path.join(results_dir, f"example_{example_count+1:02d}.png")
            cv2.imwrite(out_path, combined)
            example_count += 1

    print(f"Vizuelni primeri sačuvani u: {results_dir}/example_XX.png")
    print("Format: [Originalna slika | Tačna maska | Predikcija]")


def _colorize(mask, color_map):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        colored[mask == int(label)] = color
    return colored


if __name__ == "__main__":
    evaluate()
