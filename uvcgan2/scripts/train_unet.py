#!/usr/bin/env python3
"""
U-Net baseline for Carvana binary segmentation.

Drop-in comparison against LinkNet, DeepLabV3+, and seGANmentation.
Uses segmentation_models_pytorch with a ResNet-34 backbone.

Expected data layout (same as the seGANmentation project):
    data/Carvana_resized/
        train/
            images/   <-- RGB car photos
            masks/    <-- binary masks
        val/
            images/
            masks/

Outputs:
    outdir/unet/
        best_model.pth          – best checkpoint (by val Dice)
        training_log.csv        – per-epoch metrics
        predictions/            – saved val predictions for evaluate_segmentation.py
"""

import os
import csv
import argparse
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------
class CarvanaSegDataset(Dataset):
    """Paired (image, mask) dataset for Carvana.

    Reads from:
        <root>/<split>/images/
        <root>/<split>/masks/
    Image filenames: 0cdf5b5d0ce1_01.jpg.png
    Mask  filenames: 0cdf5b5d0ce1_01_mask.gif.png
    """

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, root, split='train', img_size=256, augment=False):
        super().__init__()
        self.img_dir  = os.path.join(root, split, 'images')
        self.mask_dir = os.path.join(root, split, 'masks')
        self.img_size = img_size

        # Collect image filenames (sorted for reproducibility)
        self.filenames = sorted([
            f for f in os.listdir(self.img_dir)
            if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS
        ])

        # ---- transforms ----
        # Shared spatial transforms (applied identically to image & mask)
        self.augment = augment

        # Image-only normalisation (ImageNet stats for pretrained backbone)
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),   # -> [0, 1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        image = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        
        # Deduce mask filename for Carvana (e.g. 6d375..._16.jpg.png -> 6d375..._16_mask.gif.png)
        mask_fname = fname
        if fname.endswith('.jpg.png'):
            base = fname[:-8]
            candidates = [base + '_mask.gif.png', base + '_mask.png', base + '_mask.gif']
            for cand in candidates:
                if os.path.exists(os.path.join(self.mask_dir, cand)):
                    mask_fname = cand
                    break
        elif fname.endswith('.jpg'):
            base = fname[:-4]
            candidates = [base + '_mask.gif', base + '_mask.png']
            for cand in candidates:
                if os.path.exists(os.path.join(self.mask_dir, cand)):
                    mask_fname = cand
                    break

        mask  = Image.open(os.path.join(self.mask_dir, mask_fname)).convert('L')

        # Random horizontal flip (same coin for image & mask)
        if self.augment and torch.rand(1).item() > 0.5:
            image = T.functional.hflip(image)
            mask  = T.functional.hflip(mask)

        image = self.img_transform(image)

        mask = self.mask_transform(mask)
        mask = (mask > 0).float()  # binarise

        return image, mask, fname


# ---------------------------------------------------------------------------
#  Metrics  (match evaluate_segmentation.py exactly)
# ---------------------------------------------------------------------------
def dice_coefficient(pred, target, eps=1e-6):
    """Dice over a batch – matches the project's evaluate_segmentation.py."""
    pred   = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred, target, eps=1e-6):
    """IoU over a batch – matches the project's evaluate_segmentation.py."""
    pred   = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)


def accuracy_score(pred, target):
    """Pixel accuracy – matches the project's evaluate_segmentation.py."""
    correct = (pred == target).sum()
    total   = target.numel()
    return correct.float() / total


def precision_score(pred, target, eps=1e-6):
    """Precision over a batch."""
    pred   = pred.view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, target, eps=1e-6):
    """Recall over a batch."""
    pred   = pred.view(-1)
    target = target.view(-1)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)


# ---------------------------------------------------------------------------
#  Training & Validation loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou  = 0.0
    running_acc  = 0.0
    running_prec = 0.0
    running_rec  = 0.0
    n_batches    = 0

    for images, masks, _ in loader:
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)              # (B, 1, H, W)
        loss   = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics on binarised predictions
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            running_loss += loss.item()
            running_dice += dice_coefficient(preds, masks).item()
            running_iou  += iou_score(preds, masks).item()
            running_acc  += accuracy_score(preds, masks).item()
            running_prec += precision_score(preds, masks).item()
            running_rec  += recall_score(preds, masks).item()
            n_batches    += 1

    return {
        'loss':     running_loss / n_batches,
        'dice':     running_dice / n_batches,
        'iou':      running_iou  / n_batches,
        'accuracy': running_acc  / n_batches,
        'precision': running_prec / n_batches,
        'recall':   running_rec  / n_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, save_dir=None):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou  = 0.0
    running_acc  = 0.0
    running_prec = 0.0
    running_rec  = 0.0
    n_batches    = 0

    for images, masks, fnames in loader:
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        loss   = criterion(logits, masks)

        preds = (torch.sigmoid(logits) > 0.5).float()
        running_loss += loss.item()
        running_dice += dice_coefficient(preds, masks).item()
        running_iou  += iou_score(preds, masks).item()
        running_acc  += accuracy_score(preds, masks).item()
        running_prec += precision_score(preds, masks).item()
        running_rec  += recall_score(preds, masks).item()
        n_batches    += 1

        # Optionally save predictions as images (for evaluate_segmentation.py)
        if save_dir is not None:
            preds_np = preds.squeeze(1).cpu().numpy()   # (B, H, W)
            for i, fname in enumerate(fnames):
                pred_img = (preds_np[i] * 255).astype(np.uint8)
                Image.fromarray(pred_img).save(os.path.join(save_dir, fname))

    return {
        'loss':     running_loss / n_batches,
        'dice':     running_dice / n_batches,
        'iou':      running_iou  / n_batches,
        'accuracy': running_acc  / n_batches,
        'precision': running_prec / n_batches,
        'recall':   running_rec  / n_batches,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Train U-Net on Carvana (binary segmentation)')

    p.add_argument('--data-root', type=str, default='data/Carvana_resized',
                   help='Root of the Carvana dataset')
    p.add_argument('--outdir', type=str, default='outdir/unet',
                   help='Where to save checkpoints and logs')
    p.add_argument('--img-size', type=int, default=256,
                   help='Input image size (square)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-5,
                   help='Learning rate (Adam)')
    p.add_argument('--workers', type=int, default=4,
                   help='DataLoader num_workers')
    p.add_argument('--save-preds', action='store_true',
                   help='Save val predictions after the final epoch')
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pred_dir = os.path.join(args.outdir, 'predictions')

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    # ---- Data ----
    train_ds = CarvanaSegDataset(
        args.data_root, split='train', img_size=args.img_size, augment=True)
    val_ds = CarvanaSegDataset(
        args.data_root, split='val', img_size=args.img_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(f'[INFO] Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}')

    # ---- Model (U-Net with ResNet-34 backbone) ----
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None,        # raw logits → BCEWithLogitsLoss
    )
    model = model.to(device)
    print(f'[INFO] U-Net (resnet34) loaded  –  '
          f'{sum(p.numel() for p in model.parameters()):,} params')

    # ---- Optimiser & Loss ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # ---- Training ----
    best_dice = 0.0
    log_path  = os.path.join(args.outdir, 'training_log.csv')

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss', 'train_dice', 'train_iou', 'train_accuracy', 'train_precision', 'train_recall',
            'val_loss',   'val_dice',   'val_iou',   'val_accuracy', 'val_precision', 'val_recall',
            'epoch_time_s',
        ])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        # Validate (save preds only on the last epoch if requested)
        save_now = args.save_preds and (epoch == args.epochs)
        if save_now:
            os.makedirs(pred_dir, exist_ok=True)

        val_metrics = validate(
            model, val_loader, criterion, device,
            save_dir=pred_dir if save_now else None)

        elapsed = time.time() - t0

        # Log
        print(
            f'Epoch {epoch:>3d}/{args.epochs}  |  '
            f'Train Loss {train_metrics["loss"]:.4f}  Dice {train_metrics["dice"]:.4f}  '
            f'IoU {train_metrics["iou"]:.4f}  Acc {train_metrics["accuracy"]:.4f}  '
            f'Prec {train_metrics["precision"]:.4f}  Rec {train_metrics["recall"]:.4f}  |  '
            f'Val Loss {val_metrics["loss"]:.4f}  Dice {val_metrics["dice"]:.4f}  '
            f'IoU {val_metrics["iou"]:.4f}  Acc {val_metrics["accuracy"]:.4f}  '
            f'Prec {val_metrics["precision"]:.4f}  Rec {val_metrics["recall"]:.4f}  |  '
            f'{elapsed:.1f}s'
        )

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{train_metrics["loss"]:.6f}',
                f'{train_metrics["dice"]:.6f}',
                f'{train_metrics["iou"]:.6f}',
                f'{train_metrics["accuracy"]:.6f}',
                f'{train_metrics["precision"]:.6f}',
                f'{train_metrics["recall"]:.6f}',
                f'{val_metrics["loss"]:.6f}',
                f'{val_metrics["dice"]:.6f}',
                f'{val_metrics["iou"]:.6f}',
                f'{val_metrics["accuracy"]:.6f}',
                f'{val_metrics["precision"]:.6f}',
                f'{val_metrics["recall"]:.6f}',
                f'{elapsed:.1f}',
            ])

        # Checkpoint best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            ckpt_path = os.path.join(args.outdir, 'best_model.pth')
            torch.save({
                'epoch':      epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice':  best_dice,
            }, ckpt_path)
            print(f'  ✓ New best model saved (Dice={best_dice:.4f})')

    # ---- Final summary ----
    print('\n' + '=' * 70)
    print(f'Training complete.  Best val Dice: {best_dice:.4f}')
    print(f'  Checkpoint : {os.path.join(args.outdir, "best_model.pth")}')
    print(f'  Log        : {log_path}')
    if args.save_preds:
        print(f'  Predictions: {pred_dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
