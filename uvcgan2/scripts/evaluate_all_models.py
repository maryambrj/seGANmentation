#!/usr/bin/env python3
"""
Unified evaluation runner for all segmentation models.

Runs Dice, IoU, Accuracy, AP@0.5, and FID across all four model families:
  - seGANmentation (GAN-based)
  - DeepLabV3+
  - LinkNet
  - SegFormer
  - U-Net

Usage:
    python evaluate_all_models.py --data-root data/Carvana_resized

The script auto-discovers prediction directories under outdir/.
You can also specify custom paths via --models JSON config.
"""

import argparse
import csv
import json
import os
import sys

# Add scripts dir to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_segmentation import evaluate_segmentation
from evaluate_ap import evaluate_ap


# ---------------------------------------------------------------------------
#  Default model configurations
# ---------------------------------------------------------------------------
DEFAULT_MODELS = {
    'seGANmentation': {
        'gt_folder': (
            './outdir/Carvana_resized/I2L/'
            '7-model_m(uvcgan2)_d(basic)_g(vit-modnet)_uvcgan2-bn_'
            '(False:0.0:5.0:1.0:1e-08)/evals/final/images_eval-val/real_b/'
        ),
        'pred_folder': (
            './outdir/Carvana_resized/I2L/'
            '7-model_m(uvcgan2)_d(basic)_g(vit-modnet)_uvcgan2-bn_'
            '(False:0.0:5.0:1.0:1e-08)/evals/final/images_eval-val/fake_b/'
        ),
    },
    'DeepLabV3+': {
        'gt_folder': 'data/Carvana_resized/val/masks/',
        'pred_folder': 'outdir/deeplabv3plus/predictions/',
    },
    'LinkNet': {
        'gt_folder': 'data/Carvana_resized/val/masks/',
        'pred_folder': 'outdir/linknet/predictions/',
    },
    'SegFormer': {
        'gt_folder': 'data/Carvana_resized/val/masks/',
        'pred_folder': 'outdir/segformer/predictions/',
    },
    'U-Net': {
        'gt_folder': 'data/Carvana_resized/val/masks/',
        'pred_folder': 'outdir/unet/predictions/',
    },
}


def try_fid(gt_folder, pred_folder, use_gpu=True):
    """Attempt to compute FID; returns None on failure."""
    try:
        from evaluate_fid_masks import compute_fid_for_masks
        metrics = compute_fid_for_masks(
            gt_folder, pred_folder, kid=False, use_gpu=use_gpu, cleanup=True
        )
        if metrics and 'frechet_inception_distance' in metrics:
            return metrics['frechet_inception_distance']
    except Exception as e:
        print(f"  [WARN] FID computation failed: {e}")
    return None


def evaluate_model(name, gt_folder, pred_folder, use_gpu=True):
    """Run all metrics for a single model."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  GT:   {gt_folder}")
    print(f"  Pred: {pred_folder}")
    print(f"{'='*60}")

    result = {'model': name}

    if not os.path.isdir(gt_folder):
        print(f"  [SKIP] GT folder not found: {gt_folder}")
        return None
    if not os.path.isdir(pred_folder):
        print(f"  [SKIP] Pred folder not found: {pred_folder}")
        return None

    # 1. Dice / IoU / Accuracy / Precision / Recall
    print("  [1/3] Computing Dice, IoU, Accuracy, Precision, Recall...")
    try:
        dice, iou, acc, prec, rec = evaluate_segmentation(gt_folder, pred_folder)
        result['dice'] = dice
        result['iou'] = iou
        result['accuracy'] = acc
        result['precision'] = prec
        result['recall'] = rec
        print(f"    Dice={dice:.4f}  IoU={iou:.4f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    except Exception as e:
        print(f"    [WARN] Segmentation metrics failed: {e}")
        result['dice'] = result['iou'] = result['accuracy'] = result['precision'] = result['recall'] = None

    # 2. AP@0.5
    print("  [2/3] Computing AP@0.5...")
    try:
        ap, stats = evaluate_ap(gt_folder, pred_folder, iou_threshold=0.5)
        result['ap_at_0.5'] = ap
        print(f"    AP@0.5={ap:.4f}  "
              f"(TP={stats['total_tp']}  FP={stats['total_fp']}  "
              f"FN={stats['total_fn']})")
    except Exception as e:
        print(f"    [WARN] AP computation failed: {e}")
        result['ap_at_0.5'] = None

    # 3. FID
    print("  [3/3] Computing FID...")
    fid = try_fid(gt_folder, pred_folder, use_gpu=use_gpu)
    result['fid'] = fid
    if fid is not None:
        print(f"    FID={fid:.4f}")
    else:
        print("    FID=N/A")

    return result


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print(f"\n\n{'='*80}")
    print("  COMPARISON TABLE — All Models")
    print(f"{'='*80}")

    header = f"{'Model':<16} {'Dice':>8} {'IoU':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'AP@0.5':>8} {'FID':>10}"
    print(header)
    print('-' * len(header))

    for r in results:
        def fmt(v, decimal=4):
            return f"{v:.{decimal}f}" if v is not None else "N/A"

        line = (
            f"{r['model']:<16} "
            f"{fmt(r.get('dice')):>8} "
            f"{fmt(r.get('iou')):>8} "
            f"{fmt(r.get('accuracy')):>8} "
            f"{fmt(r.get('precision')):>8} "
            f"{fmt(r.get('recall')):>8} "
            f"{fmt(r.get('ap_at_0.5')):>8} "
            f"{fmt(r.get('fid'), 2):>10}"
        )
        print(line)

    print(f"{'='*80}\n")


def save_results_csv(results, path):
    """Save results to CSV."""
    if not results:
        return
    fieldnames = ['model', 'dice', 'iou', 'accuracy', 'precision', 'recall', 'ap_at_0.5', 'fid']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            for k in fieldnames[1:]:
                if row[k] is not None and row[k] != '':
                    row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
    print(f"Results saved to: {path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Run all evaluation metrics across all models')
    p.add_argument('--models-json', type=str, default=None,
                   help='JSON file with custom model configs '
                        '(overrides defaults)')
    p.add_argument('--output-csv', type=str,
                   default='evaluation_results.csv',
                   help='Output CSV path (default: evaluation_results.csv)')
    p.add_argument('--no-gpu', action='store_true',
                   help='Disable GPU for FID computation')
    p.add_argument('--skip-fid', action='store_true',
                   help='Skip FID computation (much faster)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.models_json and os.path.isfile(args.models_json):
        with open(args.models_json) as f:
            models = json.load(f)
        print(f"[INFO] Loaded model configs from {args.models_json}")
    else:
        models = DEFAULT_MODELS
        print("[INFO] Using default model configurations")

    results = []
    for name, cfg in models.items():
        r = evaluate_model(
            name, cfg['gt_folder'], cfg['pred_folder'],
            use_gpu=not args.no_gpu
        )
        if r is not None:
            if args.skip_fid:
                r['fid'] = None
            results.append(r)

    if results:
        print_comparison_table(results)
        save_results_csv(results, args.output_csv)
    else:
        print("\n[WARN] No models were successfully evaluated.")


if __name__ == '__main__':
    main()
