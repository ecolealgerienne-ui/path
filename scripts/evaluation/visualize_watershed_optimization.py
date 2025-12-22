#!/usr/bin/env python3
"""
Visualize watershed optimization results.

Creates a comparison image showing:
- Ground Truth instances
- Best watershed result
- Overlay comparison

Usage:
    python scripts/evaluation/visualize_watershed_optimization.py \
        --results_file results/watershed_optimization/best_watershed_params.npz \
        --npz_file data/evaluation/pannuke_fold2_converted/image_00000.npz \
        --output results/watershed_optimization/comparison.png
"""
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    parser = argparse.ArgumentParser(description="Visualize watershed optimization")
    parser.add_argument("--results_file", type=Path, required=True)
    parser.add_argument("--npz_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/watershed_optimization/comparison.png"))
    args = parser.parse_args()

    # Load data
    results = np.load(args.results_file)
    data = np.load(args.npz_file)

    image = data['image']
    gt_inst = results['gt_inst']
    pred_inst = results['instance_map']

    edge_t = float(results['edge_threshold'])
    dist_t = int(results['dist_threshold'])
    min_s = int(results['min_size'])
    n_pred = int(results['n_instances'])
    n_gt = gt_inst.max()

    print(f"=" * 70)
    print(f"WATERSHED OPTIMIZATION VISUALIZATION")
    print(f"=" * 70)
    print(f"\nBest parameters:")
    print(f"  edge_threshold: {edge_t}")
    print(f"  dist_threshold: {dist_t}")
    print(f"  min_size: {min_s}")
    print(f"\nInstances:")
    print(f"  GT: {n_gt}")
    print(f"  Pred: {n_pred}")
    print(f"  Error: {abs(n_pred - n_gt)}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Row 1, Col 1: Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Row 1, Col 2: GT instances
    axes[0, 1].imshow(gt_inst, cmap='nipy_spectral')
    axes[0, 1].set_title(f"GT Instances\n{n_gt} instances", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Row 2, Col 1: Predicted instances
    axes[1, 0].imshow(pred_inst, cmap='nipy_spectral')
    axes[1, 0].set_title(
        f"Predicted Instances\n{n_pred} instances (edge={edge_t}, dist={dist_t}, min={min_s})",
        fontsize=14,
        fontweight='bold'
    )
    axes[1, 0].axis('off')

    # Row 2, Col 2: Overlay comparison
    overlay = image.copy()

    # GT contours in green
    import cv2
    for inst_id in range(1, gt_inst.max() + 1):
        mask = (gt_inst == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Pred contours in red
    for inst_id in range(1, pred_inst.max() + 1):
        mask = (pred_inst == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Overlay: GT (green) vs Pred (red)", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Legend
    green_patch = mpatches.Patch(color='green', label=f'GT ({n_gt} instances)')
    red_patch = mpatches.Patch(color='red', label=f'Pred ({n_pred} instances)')
    axes[1, 1].legend(handles=[green_patch, red_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')

    print(f"\n✅ Saved: {args.output}")


if __name__ == "__main__":
    main()
