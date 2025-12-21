#!/usr/bin/env python3
"""
Test different watershed parameters to improve instance separation.

This script tests various combinations of:
- edge_threshold: Threshold for HV gradient edges
- dist_threshold: Threshold for distance transform markers

Usage:
    python scripts/evaluation/test_watershed_params.py \
        --npz_file data/evaluation/pannuke_fold2_converted/image_00002.npz \
        --checkpoint_dir models/checkpoints
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scipy import ndimage
from skimage.segmentation import watershed


def test_watershed_params(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    edge_thresholds: list = [0.1, 0.2, 0.3, 0.4],
    dist_thresholds: list = [1, 2, 3, 4]
) -> dict:
    """Test different watershed parameters."""
    results = {}

    for edge_thresh in edge_thresholds:
        for dist_thresh in dist_thresholds:
            # Apply watershed with these params
            inst_map = apply_watershed(np_pred, hv_pred, edge_thresh, dist_thresh)

            # Count instances
            inst_ids = np.unique(inst_map)
            n_instances = len(inst_ids[inst_ids > 0])

            results[(edge_thresh, dist_thresh)] = {
                'inst_map': inst_map,
                'n_instances': n_instances
            }

    return results


def apply_watershed(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    edge_threshold: float,
    dist_threshold: float
) -> np.ndarray:
    """Apply watershed with specified parameters."""
    binary_mask = np_pred > 0.5

    if not binary_mask.any():
        return np.zeros_like(np_pred, dtype=np.int32)

    # Compute HV gradients
    h_grad = np.abs(cv2.Sobel(hv_pred[0], cv2.CV_64F, 1, 0, ksize=3))
    v_grad = np.abs(cv2.Sobel(hv_pred[1], cv2.CV_64F, 0, 1, ksize=3))

    edge = h_grad + v_grad
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

    # Create markers
    markers = np_pred.copy()
    markers[edge > edge_threshold] = 0
    markers = (markers > 0.7).astype(np.uint8)

    # Distance transform
    dist = ndimage.distance_transform_edt(binary_mask)
    markers = ndimage.label(markers * (dist > dist_threshold))[0]

    # Watershed
    if markers.max() > 0:
        instance_map = watershed(-dist, markers, mask=binary_mask)
    else:
        instance_map = ndimage.label(binary_mask)[0]

    return instance_map


def main():
    parser = argparse.ArgumentParser(description="Test watershed parameters")
    parser.add_argument("--npz_file", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/watershed_tests"))

    args = parser.parse_args()

    # Load GT
    print(f"📥 Loading: {args.npz_file.name}")
    data = np.load(args.npz_file)
    image = data['image']
    gt_inst = data['inst_map']

    gt_n_instances = len(np.unique(gt_inst)[np.unique(gt_inst) > 0])
    print(f"   GT instances: {gt_n_instances}")

    # Load model and predict
    print(f"🤖 Loading model...")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily

    model = OptimusGateInferenceMultiFamily(checkpoint_dir=str(args.checkpoint_dir))

    print(f"🔮 Running prediction...")
    result = model.predict(image)

    # Get NP and HV predictions (before watershed)
    np_pred = result.get('np_pred', None)
    hv_pred = result.get('hv_pred', None)

    if np_pred is None or hv_pred is None:
        print("❌ Model doesn't return np_pred/hv_pred - cannot test watershed params")
        return

    # Test different parameters
    print(f"\n🧪 Testing watershed parameters...")
    edge_thresholds = [0.1, 0.2, 0.3, 0.4]
    dist_thresholds = [1, 2, 3, 4]

    results = test_watershed_params(np_pred, hv_pred, edge_thresholds, dist_thresholds)

    # Create visualization grid
    fig, axes = plt.subplots(len(edge_thresholds), len(dist_thresholds), figsize=(16, 16))

    for i, edge_thresh in enumerate(edge_thresholds):
        for j, dist_thresh in enumerate(dist_thresholds):
            result_data = results[(edge_thresh, dist_thresh)]
            inst_map = result_data['inst_map']
            n_inst = result_data['n_instances']

            ax = axes[i, j]
            ax.imshow(inst_map, cmap='nipy_spectral')
            ax.set_title(f'edge={edge_thresh}, dist={dist_thresh}\nn={n_inst}')
            ax.axis('off')

            # Highlight the best one (closest to GT)
            if abs(n_inst - gt_n_instances) <= 2:
                ax.spines['bottom'].set_color('green')
                ax.spines['top'].set_color('green')
                ax.spines['left'].set_color('green')
                ax.spines['right'].set_color('green')
                ax.spines['bottom'].set_linewidth(5)
                ax.spines['top'].set_linewidth(5)
                ax.spines['left'].set_linewidth(5)
                ax.spines['right'].set_linewidth(5)

    plt.suptitle(f'Watershed Parameter Sweep (GT instances: {gt_n_instances})', fontsize=16)
    plt.tight_layout()

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{args.npz_file.stem}_watershed_sweep.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_file}")

    # Print best parameters
    print(f"\n📊 Results Summary:")
    print(f"GT instances: {gt_n_instances}")
    print(f"\nClosest results:")
    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['n_instances'] - gt_n_instances))
    for (edge_t, dist_t), data in sorted_results[:5]:
        n = data['n_instances']
        diff = n - gt_n_instances
        print(f"  edge={edge_t}, dist={dist_t} → {n} instances (Δ={diff:+d})")


if __name__ == "__main__":
    main()
