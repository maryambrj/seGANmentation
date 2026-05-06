import torch
import os
from PIL import Image
import numpy as np


def load_image(path):
    image = Image.open(path).convert('L')
    image = np.array(image)
    image = (image > 127).astype(np.uint8)  # Binarize the image (white=1, black=0)
    return torch.tensor(image, dtype=torch.float32)


def dice_coefficient(pred, target, reduce_batch_first=False):
    assert pred.size() == target.size()
    assert pred.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    intersection = 2*(pred * target).sum(dim=sum_dim)
    sets_sum = pred.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, intersection, sets_sum)

    epsilon = 1e-6
    dice = (intersection + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def iou(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    # print("Intersection:", intersection)
    # print("Union:", union)
    return intersection / (union + 1e-6)

def accuracy(pred, target):
    correct = (pred == target).sum()
    total = pred.numel()
    return correct.float() / total


# def evaluate_segmentation(gt_folder, est_folder):
#     gt_files = sorted(os.listdir(gt_folder))
#     est_files = sorted(os.listdir(est_folder))

#     # Convert list of estimated files to a set for faster lookup
#     est_files_set = set(est_files)

#     dice_scores = []
#     iou_scores = []

    # for gt_file in gt_files:
    #     if gt_file in est_files_set:
    #         gt_path = os.path.join(gt_folder, gt_file)
    #         est_path = os.path.join(est_folder, gt_file)

    #         gt_image = load_image(gt_path)
    #         est_image = load_image(est_path)

    #         dice_score = dice_coefficient(est_image, gt_image)
    #         iou_score = iou(est_image, gt_image)

    #         dice_scores.append(dice_score)
    #         iou_scores.append(iou_score)
    #     else:
    #         print(f"No corresponding prediction found for {gt_file}")

    # if dice_scores and iou_scores:
    #     avg_dice = sum(dice_scores) / len(dice_scores)
    #     avg_iou = sum(iou_scores) / len(iou_scores)
    #     return avg_dice.item(), avg_iou.item()
    # else:
    #     return None, None
def get_pred_filename(gt_file, pred_files_set):
    if gt_file in pred_files_set:
        return gt_file
    base = gt_file
    for suffix in ['_mask.gif.png', '_mask.gif', '_mask.png', '.png', '.jpg']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    candidates = [
        f"{base}.jpg.png", f"{base}.jpg", f"{base}.png",
        f"{base}_mask.gif.png", f"{base}_mask.gif", f"{base}_mask.png",
    ]
    for cand in candidates:
        if cand in pred_files_set:
            return cand
    return None

def evaluate_segmentation(gt_folder, est_folder):
    gt_files = sorted([f for f in os.listdir(gt_folder) if os.path.isfile(os.path.join(gt_folder, f)) and not f.startswith('.')])
    est_files = sorted([f for f in os.listdir(est_folder) if not f.startswith('.')])

    est_files_set = set(est_files)

    dice_scores = []
    iou_scores = []
    accuracy_scores = []

    for i, gt_file in enumerate(gt_files):
        pred_file = get_pred_filename(gt_file, est_files_set)
        if pred_file is None:
            fallback = f"sample_{i}.png"
            if fallback in est_files_set:
                pred_file = fallback

        if pred_file is not None:
            gt_path = os.path.join(gt_folder, gt_file)
            est_path = os.path.join(est_folder, pred_file)

            gt_image = load_image(gt_path)
            est_image = load_image(est_path)

            dice_score = dice_coefficient(est_image, gt_image)
            iou_score = iou(est_image, gt_image)
            acc_score = accuracy(est_image, gt_image)

            dice_scores.append(dice_score)
            iou_scores.append(iou_score)
            accuracy_scores.append(acc_score)
        else:
            print(f"No corresponding prediction found for {gt_file}")

    if dice_scores and iou_scores and accuracy_scores:
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_iou = sum(iou_scores) / len(iou_scores)
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        return avg_dice.item(), avg_iou.item(), avg_accuracy.item()
    else:
        return None, None, None


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description='Evaluate binary segmentation masks (Dice, IoU, Accuracy)')
    p.add_argument(
        '--gt-folder', type=str,
        default='./outdir/Carvana_resized/I2L/7-model_m(uvcgan2)_d(basic)_g(vit-modnet)_uvcgan2-bn_(False:0.0:5.0:1.0:1e-08)/evals/final/images_eval-val/real_b/',
        help='Directory of ground-truth binary masks')
    p.add_argument(
        '--pred-folder', type=str,
        default='./outdir/Carvana_resized/I2L/7-model_m(uvcgan2)_d(basic)_g(vit-modnet)_uvcgan2-bn_(False:0.0:5.0:1.0:1e-08)/evals/final/images_eval-val/fake_b/',
        help='Directory of predicted binary masks')
    p.add_argument(
        '--save-csv', type=str, default=None,
        help='Optional path to save summary results CSV')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    avg_dice, avg_iou, avg_accuracy = evaluate_segmentation(
        args.gt_folder, args.pred_folder)
    print(f"Average Dice Coefficient: {avg_dice}")
    print(f"Average IoU: {avg_iou}")
    print(f"Average Accuracy: {avg_accuracy}")

    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Dice', f"{avg_dice:.4f}" if avg_dice else "None"])
            writer.writerow(['IoU', f"{avg_iou:.4f}" if avg_iou else "None"])
            writer.writerow(['Accuracy', f"{avg_accuracy:.4f}" if avg_accuracy else "None"])
        print(f"Summary results saved to: {args.save_csv}")

