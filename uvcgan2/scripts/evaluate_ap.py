#!/usr/bin/env python3
"""
Average Precision at IoU >= 0.5 (AP@0.5) for binary segmentation masks.

Evaluates predicted masks against ground-truth masks using an instance-level
matching approach:
  1. Extract connected components from both GT and predicted masks.
  2. Build an IoU matrix between predicted and GT components.
  3. Greedily match predicted → GT components (highest IoU first).
  4. A match counts as TP if IoU >= 0.5; unmatched predictions are FP;
     unmatched GT instances are FN.
  5. Compute precision-recall curve and 11-point interpolated AP.

Usage:
    python evaluate_ap.py --gt-folder <GT_DIR> --pred-folder <PRED_DIR>
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage


# ---------------------------------------------------------------------------
#  Image loading (matches evaluate_segmentation.py)
# ---------------------------------------------------------------------------
def load_binary_mask(path):
    """Load an image and binarise it (foreground = 1, background = 0)."""
    image = np.array(Image.open(path))
    if image.ndim == 3:
        image = image[:, :, 0]
    return (image > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
#  Connected-component extraction
# ---------------------------------------------------------------------------
def extract_instances(binary_mask):
    """Return a list of boolean masks, one per connected component."""
    labelled, n_components = ndimage.label(binary_mask)
    instances = []
    for i in range(1, n_components + 1):
        instances.append(labelled == i)
    return instances


# ---------------------------------------------------------------------------
#  IoU helpers
# ---------------------------------------------------------------------------
def instance_iou(mask_a, mask_b):
    """IoU between two boolean masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


# ---------------------------------------------------------------------------
#  Per-image matching
# ---------------------------------------------------------------------------
def match_instances(gt_instances, pred_instances, iou_threshold=0.5):
    """
    Greedily match predicted instances to GT instances.

    Returns
    -------
    tp : int   – number of true positive matches (IoU >= threshold)
    fp : int   – number of unmatched predictions
    fn : int   – number of unmatched GT instances
    ious : list[float] – IoU values for each TP match
    """
    if len(pred_instances) == 0:
        return 0, 0, len(gt_instances), []

    if len(gt_instances) == 0:
        return 0, len(pred_instances), 0, []

    # Build IoU matrix: (n_pred, n_gt)
    iou_matrix = np.zeros((len(pred_instances), len(gt_instances)))
    for i, pred in enumerate(pred_instances):
        for j, gt in enumerate(gt_instances):
            iou_matrix[i, j] = instance_iou(pred, gt)

    tp = 0
    fp = 0
    matched_gt = set()
    match_ious = []

    # Greedy matching: iterate over (pred, gt) pairs sorted by IoU descending
    flat_indices = np.argsort(-iou_matrix.ravel())
    matched_pred = set()

    for flat_idx in flat_indices:
        i = flat_idx // len(gt_instances)
        j = flat_idx % len(gt_instances)

        if i in matched_pred or j in matched_gt:
            continue

        if iou_matrix[i, j] < iou_threshold:
            break  # remaining pairs are all below threshold

        tp += 1
        match_ious.append(iou_matrix[i, j])
        matched_pred.add(i)
        matched_gt.add(j)

    fp = len(pred_instances) - len(matched_pred)
    fn = len(gt_instances) - len(matched_gt)

    return tp, fp, fn, match_ious


# ---------------------------------------------------------------------------
#  AP computation
# ---------------------------------------------------------------------------
def compute_ap(precisions, recalls):
    """11-point interpolated Average Precision (VOC-style)."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        # Precision at recall >= t
        prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        if prec_at_recall:
            ap += max(prec_at_recall)
    return ap / 11.0


# ---------------------------------------------------------------------------
#  Main evaluation
# ---------------------------------------------------------------------------
def evaluate_ap(gt_folder, pred_folder, iou_threshold=0.5):
    """
    Compute AP@{iou_threshold} across all images.

    Parameters
    ----------
    gt_folder : str
        Directory with ground-truth binary masks.
    pred_folder : str
        Directory with predicted binary masks.
    iou_threshold : float
        IoU threshold for a match to count as TP.

    Returns
    -------
    ap : float
        Average Precision at the given IoU threshold.
    stats : dict
        Aggregate TP / FP / FN counts and per-image breakdown.
    """
    gt_files = sorted(os.listdir(gt_folder))
    pred_files_set = set(os.listdir(pred_folder))

    # Accumulate over all images
    all_tp = 0
    all_fp = 0
    all_fn = 0
    per_image = []

    for gt_file in gt_files:
        if gt_file not in pred_files_set:
            print(f"[WARN] No prediction found for {gt_file}")
            continue

        gt_mask = load_binary_mask(os.path.join(gt_folder, gt_file))
        pred_mask = load_binary_mask(os.path.join(pred_folder, gt_file))

        gt_instances = extract_instances(gt_mask)
        pred_instances = extract_instances(pred_mask)

        tp, fp, fn, ious = match_instances(
            gt_instances, pred_instances, iou_threshold
        )

        all_tp += tp
        all_fp += fp
        all_fn += fn

        per_image.append({
            'file': gt_file,
            'tp': tp, 'fp': fp, 'fn': fn,
            'n_gt': len(gt_instances),
            'n_pred': len(pred_instances),
        })

    # Build cumulative precision-recall curve
    # (sorted by confidence — here all detections have equal confidence,
    #  so we accumulate image-by-image)
    cum_tp = 0
    cum_fp = 0
    total_gt = all_tp + all_fn

    precisions = []
    recalls = []

    for img in per_image:
        cum_tp += img['tp']
        cum_fp += img['fp']
        precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
        recall = cum_tp / total_gt if total_gt > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    ap = compute_ap(precisions, recalls) if precisions else 0.0

    stats = {
        'total_tp': all_tp,
        'total_fp': all_fp,
        'total_fn': all_fn,
        'total_gt_instances': total_gt,
        'n_images': len(per_image),
        'per_image': per_image,
    }

    return ap, stats


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Compute Average Precision at IoU >= 0.5 for binary masks')

    p.add_argument('--gt-folder', type=str, required=True,
                   help='Directory of ground-truth binary masks')
    p.add_argument('--pred-folder', type=str, required=True,
                   help='Directory of predicted binary masks')
    p.add_argument('--iou-threshold', type=float, default=0.5,
                   help='IoU threshold for TP matching (default: 0.5)')
    p.add_argument('--save-csv', type=str, default=None,
                   help='Optional path to save per-image results CSV')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.gt_folder):
        print(f"[ERROR] GT folder not found: {args.gt_folder}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.pred_folder):
        print(f"[ERROR] Pred folder not found: {args.pred_folder}", file=sys.stderr)
        sys.exit(1)

    ap, stats = evaluate_ap(args.gt_folder, args.pred_folder, args.iou_threshold)

    print(f"\n{'='*60}")
    print(f"  AP@{args.iou_threshold:.2f}  =  {ap:.4f}")
    print(f"  Total TP: {stats['total_tp']}  |  FP: {stats['total_fp']}  |  "
          f"FN: {stats['total_fn']}")
    print(f"  GT instances: {stats['total_gt_instances']}  |  "
          f"Images evaluated: {stats['n_images']}")
    print(f"{'='*60}\n")

    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'file', 'tp', 'fp', 'fn', 'n_gt', 'n_pred'])
            writer.writeheader()
            writer.writerows(stats['per_image'])
        print(f"Per-image results saved to: {args.save_csv}")


if __name__ == '__main__':
    main()
