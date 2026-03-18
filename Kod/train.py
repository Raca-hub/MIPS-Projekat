import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.segmentation import LandSegmentation
from dataset import create_dataloaders


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(self.num_classes):
            pred = probs[:, c]
            target = (targets == c).float()
            intersection = (pred * target).sum()
            dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
            loss += (1.0 - dice)
        return loss / self.num_classes


class ComboLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        return self.dice_weight * self.dice(logits, targets) + \
               self.ce_weight * self.ce(logits, targets)


def compute_iou_per_class(preds, targets, num_classes):
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


def compute_mean_iou(preds, targets, num_classes):
    ious = compute_iou_per_class(preds, targets, num_classes)
    valid = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid) if valid else 0.0


def train_one_epoch(seg_engine, loader, optimizer, criterion, device):
    seg_engine.model.train()
    total_loss = 0.0
    total_miou = 0.0
    num_classes = seg_engine.model.segmentation_head[0].out_channels

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = seg_engine.model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            miou = compute_mean_iou(preds.cpu(), masks.cpu(), num_classes)
            total_miou += miou

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f} | mIoU: {miou:.4f}")

    return total_loss / len(loader), total_miou / len(loader)


def validate(seg_engine, loader, criterion, device):
    seg_engine.model.eval()
    total_loss = 0.0
    total_miou = 0.0
    num_classes = seg_engine.model.segmentation_head[0].out_channels

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = seg_engine.model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            miou = compute_mean_iou(preds.cpu(), masks.cpu(), num_classes)
            total_miou += miou

    return total_loss / len(loader), total_miou / len(loader)


def save_log(log_path, epoch, train_loss, val_loss, train_miou, val_miou, lr):
    with open(log_path, 'a') as f:
        if epoch == 1:
            f.write("epoch,train_loss,val_loss,train_miou,val_miou,lr\n")
        f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_miou:.6f},{val_miou:.6f},{lr:.8f}\n")


def train(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    t_cfg = cfg['training']
    p_cfg = cfg['paths']
    num_classes = cfg['model']['num_classes']

    os.makedirs(p_cfg['models_dir'], exist_ok=True)
    os.makedirs(p_cfg['logs_dir'], exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(p_cfg['logs_dir'], f"train_{timestamp}.csv")
    best_model_path = p_cfg['best_model_pth']

    print("Inicijalizacija modela...")
    seg_engine = LandSegmentation()
    device = seg_engine.device
    print(f"Uredjaj: {device}")

    train_loader, val_loader = create_dataloaders(config_path)

    loss_fn_name = t_cfg['loss_function']
    if loss_fn_name == "combo":
        criterion = ComboLoss(num_classes, t_cfg['dice_weight'], t_cfg['ce_weight'])
    elif loss_fn_name == "dice":
        criterion = DiceLoss(num_classes)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        seg_engine.model.parameters(),
        lr=t_cfg['learning_rate'],
        weight_decay=t_cfg['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=t_cfg['scheduler_patience'],
        factor=t_cfg['scheduler_factor'],
        min_lr=t_cfg['min_lr']
    )

    best_val_miou = 0.0
    patience_counter = 0
    early_stop_patience = t_cfg['early_stopping_patience']

    print(f"\n{'='*50}")
    print(f"Pocinjem trening: {t_cfg['epochs']} epoha, batch={t_cfg['batch_size']}")
    print(f"{'='*50}\n")

    for epoch in range(1, t_cfg['epochs'] + 1):
        print(f"Epoha [{epoch}/{t_cfg['epochs']}]")

        train_loss, train_miou = train_one_epoch(seg_engine, train_loader, optimizer, criterion, device)
        val_loss, val_miou = validate(seg_engine, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   mIoU: {val_miou:.4f}")
        print(f"  LR: {current_lr:.8f}")

        save_log(log_path, epoch, train_loss, val_loss, train_miou, val_miou, current_lr)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(seg_engine.model.state_dict(), best_model_path)
            print(f"  Novi best model sacuvan! Val mIoU: {best_val_miou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Bez poboljsanja ({patience_counter}/{early_stop_patience})")

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping aktiviran u epohi {epoch}.")
            break

        print()

    print(f"\n{'='*50}")
    print(f"TRENING ZAVRSEN!")
    print(f"Najbolji Val mIoU: {best_val_miou:.4f}")
    print(f"Model sacuvan na: {best_model_path}")
    print(f"Log sacuvan na:   {log_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    train()