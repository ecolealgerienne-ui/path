"""
CellPose Validation on SIPaKMeD — Quality Check

Ce script valide CellPose pré-entraîné sur SIPaKMeD preprocessed:
1. Load CellPose model (pré-entraîné 'nuclei')
2. Prédictions sur images 224×224
3. Comparer vs masques ground truth (-d.bmp)
4. Calculer métriques (IoU, Dice, Precision, Recall, AP50)
5. Générer rapport + visualisations

Author: V14 Cytology Branch
Date: 2026-01-19
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from cellpose import models
import cv2


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Intersection over Union

    Args:
        pred: Binary mask prediction (H, W)
        gt: Binary mask ground truth (H, W)

    Returns:
        IoU score [0, 1]
    """
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice coefficient (F1 score)

    Args:
        pred: Binary mask prediction
        gt: Binary mask ground truth

    Returns:
        Dice score [0, 1]
    """
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2.0 * intersection) / total


def compute_precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """
    Precision and Recall

    Args:
        pred: Binary mask prediction
        gt: Binary mask ground truth

    Returns:
        precision, recall
    """
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


# ═════════════════════════════════════════════════════════════════════════════
#  CELLPOSE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_cellpose_on_dataset(
    data_dir: str,
    split: str = 'val',
    diameter: float = 35,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    n_samples: int = None
) -> Dict:
    """
    Valide CellPose sur dataset preprocessed

    Args:
        data_dir: data/processed/sipakmed/
        split: 'train' or 'val'
        diameter: CellPose diameter parameter
        flow_threshold: Flow threshold
        cellprob_threshold: Cell probability threshold
        n_samples: Nombre samples à tester (None = tous)

    Returns:
        results: Dict with metrics
    """
    split_dir = os.path.join(data_dir, split)
    metadata_path = os.path.join(split_dir, 'metadata.json')

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if n_samples is not None:
        metadata = metadata[:n_samples]

    print(f"\n🔬 Validating CellPose on {len(metadata)} samples ({split} split)")
    print(f"   Model: nuclei (pré-entraîné)")
    print(f"   Diameter: {diameter}")
    print(f"   Flow threshold: {flow_threshold}")
    print(f"   Cellprob threshold: {cellprob_threshold}")

    # Load CellPose model (API v4.x: pretrained_model, not model_type)
    model = models.CellposeModel(pretrained_model='nuclei', gpu=True)

    # Metrics storage
    ious = []
    dices = []
    precisions = []
    recalls = []
    class_metrics = {cls: {'iou': [], 'dice': []} for cls in set(m['class_name'] for m in metadata)}

    # Process samples
    for sample in tqdm(metadata, desc="  Validating"):
        # Load image
        image_path = os.path.join(split_dir, sample['image_path'])
        image = np.array(Image.open(image_path))

        # Load ground truth mask
        mask_path = os.path.join(split_dir, sample['mask_path'])
        mask_gt = np.array(Image.open(mask_path))
        mask_gt_binary = (mask_gt > 0).astype(np.uint8)

        # CellPose prediction (API v4.x: returns 3 values)
        try:
            masks_pred, flows, styles = model.eval(
                image,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold
            )

            # Convert to binary
            mask_pred_binary = (masks_pred > 0).astype(np.uint8)

        except Exception as e:
            print(f"⚠️  CellPose failed on {sample['filename']}: {e}")
            continue

        # Compute metrics
        iou = compute_iou(mask_pred_binary, mask_gt_binary)
        dice = compute_dice(mask_pred_binary, mask_gt_binary)
        precision, recall = compute_precision_recall(mask_pred_binary, mask_gt_binary)

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)

        # Per-class metrics
        class_name = sample['class_name']
        class_metrics[class_name]['iou'].append(iou)
        class_metrics[class_name]['dice'].append(dice)

    # Aggregate results
    results = {
        'n_samples': len(metadata),
        'n_processed': len(ious),
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'mean_dice': np.mean(dices),
        'std_dice': np.std(dices),
        'mean_precision': np.mean(precisions),
        'std_precision': np.std(precisions),
        'mean_recall': np.mean(recalls),
        'std_recall': np.std(recalls),
        'class_metrics': {}
    }

    # Per-class aggregation
    for cls_name, metrics in class_metrics.items():
        if len(metrics['iou']) > 0:
            results['class_metrics'][cls_name] = {
                'n_samples': len(metrics['iou']),
                'mean_iou': np.mean(metrics['iou']),
                'mean_dice': np.mean(metrics['dice'])
            }

    return results, ious, dices


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_validation_results(results: Dict, ious: List[float], dices: List[float], output_path: str):
    """
    Générer visualisations résultats validation

    Args:
        results: Dict from validate_cellpose_on_dataset
        ious: List of IoU scores
        dices: List of Dice scores
        output_path: Chemin sauvegarde figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: IoU distribution
    axes[0, 0].hist(ious, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(results['mean_iou'], color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {results['mean_iou']:.3f}")
    axes[0, 0].axvline(0.85, color='green', linestyle='--', linewidth=2,
                       label="Target: 0.85")
    axes[0, 0].set_xlabel('IoU Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('IoU Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Dice distribution
    axes[0, 1].hist(dices, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(results['mean_dice'], color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {results['mean_dice']:.3f}")
    axes[0, 1].set_xlabel('Dice Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Dice Coefficient Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Per-class IoU
    class_names = list(results['class_metrics'].keys())
    class_ious = [results['class_metrics'][cls]['mean_iou'] for cls in class_names]

    axes[1, 0].barh(class_names, class_ious, color='steelblue', edgecolor='black')
    axes[1, 0].axvline(0.85, color='green', linestyle='--', linewidth=2, label='Target')
    axes[1, 0].set_xlabel('Mean IoU')
    axes[1, 0].set_title('IoU by Class')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='x')

    # Plot 4: Summary metrics
    metrics_names = ['IoU', 'Dice', 'Precision', 'Recall']
    metrics_values = [
        results['mean_iou'],
        results['mean_dice'],
        results['mean_precision'],
        results['mean_recall']
    ]

    axes[1, 1].bar(metrics_names, metrics_values, color=['steelblue', 'coral', 'lightgreen', 'gold'],
                   edgecolor='black', alpha=0.7)
    axes[1, 1].axhline(0.85, color='green', linestyle='--', linewidth=2, label='Target IoU')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Summary Metrics')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Visualization saved: {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═════════════════════════════════════════════════════════════════════════════

def print_validation_report(results: Dict):
    """
    Affiche rapport validation formaté

    Args:
        results: Dict from validate_cellpose_on_dataset
    """
    print("\n" + "=" * 80)
    print(" CELLPOSE VALIDATION REPORT (SIPaKMeD)")
    print("=" * 80)

    print(f"\nDataset: {results['n_processed']}/{results['n_samples']} images processed")
    print(f"Model:   CellPose 'nuclei' (pré-entraîné)")

    print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Status':<10}")
    print("-" * 80)

    iou_status = "✅ PASS" if results['mean_iou'] >= 0.85 else "❌ FAIL"
    print(f"{'IoU':<20} {results['mean_iou']:<10.3f} {results['std_iou']:<10.3f} {iou_status}")
    print(f"{'Dice':<20} {results['mean_dice']:<10.3f} {results['std_dice']:<10.3f}")
    print(f"{'Precision':<20} {results['mean_precision']:<10.3f} {results['std_precision']:<10.3f}")
    print(f"{'Recall':<20} {results['mean_recall']:<10.3f} {results['std_recall']:<10.3f}")

    print("\n" + "-" * 80)
    print("Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<25} {'N':<8} {'IoU':<10} {'Dice':<10}")
    print("-" * 80)

    for cls_name, metrics in sorted(results['class_metrics'].items()):
        print(f"{cls_name:<25} {metrics['n_samples']:<8} "
              f"{metrics['mean_iou']:<10.3f} {metrics['mean_dice']:<10.3f}")

    print("\n" + "=" * 80)

    if results['mean_iou'] >= 0.85:
        print("✅ VALIDATION PASSED — CellPose quality sufficient")
        print("   → Continue to next step: Feature extraction")
    else:
        print("❌ VALIDATION FAILED — CellPose quality insufficient")
        print(f"   → IoU {results['mean_iou']:.3f} < 0.85 (target)")
        print("   → Consider adjusting CellPose parameters (diameter, thresholds)")

    print("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate CellPose on preprocessed SIPaKMeD"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/sipakmed',
        help='Preprocessed data directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val'],
        help='Dataset split to validate on'
    )
    parser.add_argument(
        '--diameter',
        type=float,
        default=35,
        help='CellPose diameter parameter (cervical cells ~35)'
    )
    parser.add_argument(
        '--flow_threshold',
        type=float,
        default=0.4,
        help='Flow threshold (default: 0.4)'
    )
    parser.add_argument(
        '--cellprob_threshold',
        type=float,
        default=0.0,
        help='Cell probability threshold (default: 0.0)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of samples to test (None = all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/cellpose_validation',
        help='Output directory for report and plots'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CELLPOSE VALIDATION — V14 Cytology")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Split:          {args.split}")
    print(f"Diameter:       {args.diameter}")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run validation
    results, ious, dices = validate_cellpose_on_dataset(
        data_dir=args.data_dir,
        split=args.split,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        n_samples=args.n_samples
    )

    # Print report
    print_validation_report(results)

    # Save results JSON
    results_path = os.path.join(args.output_dir, 'validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved: {results_path}")

    # Generate plots
    plot_path = os.path.join(args.output_dir, 'validation_plots.png')
    plot_validation_results(results, ious, dices, plot_path)

    # Next steps
    print("\n" + "=" * 80)
    if results['mean_iou'] >= 0.85:
        print("Next step:")
        print("  python scripts/cytology/01_generate_cellpose_masks.py")
    else:
        print("Adjust CellPose parameters and retry:")
        print(f"  python scripts/cytology/00b_validate_cellpose.py --diameter 40 --flow_threshold 0.3")
    print("=" * 80)


if __name__ == '__main__':
    main()
