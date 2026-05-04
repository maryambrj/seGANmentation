#!/usr/bin/env python3
"""
FID evaluation for generated segmentation masks.

Computes FID between predicted and ground-truth binary mask directories.
Masks are auto-converted to 3-channel RGB for InceptionV3 feature extraction.

Usage:
    python evaluate_fid_masks.py --gt-folder <GT> --pred-folder <PRED>

Requirements:
    pip install torch-fidelity
"""

import argparse
import os
import sys
import shutil
import tempfile
import numpy as np
from PIL import Image

try:
    import torch_fidelity
except ImportError:
    print("[ERROR] torch_fidelity required: pip install torch-fidelity",
          file=sys.stderr)
    sys.exit(1)


def convert_masks_to_rgb(src_folder, dst_folder):
    """Convert single-channel binary masks to 3-channel RGB images."""
    os.makedirs(dst_folder, exist_ok=True)
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif'}
    n = 0
    for fname in sorted(os.listdir(src_folder)):
        if os.path.splitext(fname)[1].lower() not in exts:
            continue
        img = np.array(Image.open(os.path.join(src_folder, fname)))
        if img.ndim == 3:
            img = img[:, :, 0]
        img = ((img > 0).astype(np.uint8)) * 255
        rgb = np.stack([img] * 3, axis=-1)
        out_name = os.path.splitext(fname)[0] + '.png'
        Image.fromarray(rgb).save(os.path.join(dst_folder, out_name))
        n += 1
    return n


def compute_fid_for_masks(gt_folder, pred_folder, kid=False, kid_size=100,
                          use_gpu=True, cleanup=True):
    """Compute FID between GT and predicted mask directories."""
    tmp_dir = tempfile.mkdtemp(prefix='fid_masks_')
    gt_rgb = os.path.join(tmp_dir, 'gt_rgb')
    pred_rgb = os.path.join(tmp_dir, 'pred_rgb')
    try:
        print("[INFO] Converting GT masks to 3-channel RGB...")
        n_gt = convert_masks_to_rgb(gt_folder, gt_rgb)
        print(f"  -> {n_gt} GT masks converted")
        print("[INFO] Converting predicted masks to 3-channel RGB...")
        n_pred = convert_masks_to_rgb(pred_folder, pred_rgb)
        print(f"  -> {n_pred} predicted masks converted")
        if n_gt == 0 or n_pred == 0:
            print("[ERROR] No images in one or both dirs.", file=sys.stderr)
            return None
        print("[INFO] Computing FID (InceptionV3 features)...")
        metrics = torch_fidelity.calculate_metrics(
            input1=pred_rgb, input2=gt_rgb,
            cuda=use_gpu, isc=False, fid=True, kid=kid,
            verbose=False, kid_subset_size=kid_size if kid else 100,
        )
        return metrics
    finally:
        if cleanup and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description='Compute FID for generated segmentation masks')
    p.add_argument('--gt-folder', type=str, required=True,
                   help='Directory of ground-truth binary masks')
    p.add_argument('--pred-folder', type=str, required=True,
                   help='Directory of predicted binary masks')
    p.add_argument('--kid', action='store_true',
                   help='Also compute KID')
    p.add_argument('--kid-size', type=int, default=100,
                   help='KID subset size (default: 100)')
    p.add_argument('--no-gpu', action='store_true',
                   help='Disable GPU')
    p.add_argument('--save-csv', type=str, default=None,
                   help='Optional path to save metrics CSV')
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.gt_folder):
        print(f"[ERROR] GT folder not found: {args.gt_folder}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.pred_folder):
        print(f"[ERROR] Pred folder not found: {args.pred_folder}", file=sys.stderr)
        sys.exit(1)

    metrics = compute_fid_for_masks(
        gt_folder=args.gt_folder, pred_folder=args.pred_folder,
        kid=args.kid, kid_size=args.kid_size, use_gpu=not args.no_gpu,
    )
    if metrics is None:
        sys.exit(1)

    print(f"\n{'='*60}")
    fid = metrics.get('frechet_inception_distance', float('nan'))
    print(f"  FID (masks)  =  {fid:.4f}")
    if args.kid and 'kernel_inception_distance_mean' in metrics:
        kid_mean = metrics['kernel_inception_distance_mean']
        kid_std = metrics['kernel_inception_distance_std']
        print(f"  KID (masks)  =  {kid_mean:.6f} +/- {kid_std:.6f}")
    print(f"{'='*60}\n")

    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for k, v in metrics.items():
                writer.writerow([k, f'{v:.6f}'])
        print(f"Metrics saved to: {args.save_csv}")


if __name__ == '__main__':
    main()
