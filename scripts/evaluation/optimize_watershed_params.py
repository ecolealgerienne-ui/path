#!/usr/bin/env python3
"""
Optimize watershed parameters to reduce over-segmentation.

Tests multiple combinations of edge_threshold, dist_threshold, and min_size
to find the best match with Ground Truth instance count.

Usage:
    python scripts/evaluation/optimize_watershed_params.py \
        --npz_file data/evaluation/pannuke_fold2_converted/image_00000.npz \
        --checkpoint_dir models/checkpoints_FIXED
"""
import argparse
import numpy as np
from pathlib import Path
import sys
import cv2
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily


def compute_gradient_magnitude(hv_map: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude from HV maps.

    Args:
        hv_map: (2, H, W) horizontal and vertical distance maps

    Returns:
        Gradient magnitude (H, W)
    """
    h_grad = cv2.Sobel(hv_map[0], cv2.CV_64F, 1, 0, ksize=3)
    v_grad = cv2.Sobel(hv_map[1], cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(h_grad**2 + v_grad**2)

    return gradient


def watershed_post_process(
    np_mask: np.ndarray,
    hv_map: np.ndarray,
    edge_threshold: float = 0.3,
    dist_threshold: int = 2,
    min_size: int = 10,
) -> np.ndarray:
    """
    Post-process NP mask and HV maps to separate instances.

    Args:
        np_mask: Binary nuclei presence mask (H, W)
        hv_map: Horizontal-Vertical distance maps (2, H, W)
        edge_threshold: Threshold for HV gradient (higher = fewer instances)
        dist_threshold: Threshold for distance transform (higher = fewer instances)
        min_size: Minimum instance size in pixels

    Returns:
        Instance map (H, W) with labeled instances
    """
    # Gradient magnitude from HV maps
    gradient = compute_gradient_magnitude(hv_map)

    # Threshold to get edges
    edges = gradient > edge_threshold

    # Distance transform on inverted edges
    dist = ndimage.distance_transform_edt(~edges)

    # Find local maxima as markers
    local_max = peak_local_max(
        dist,
        min_distance=dist_threshold,
        labels=np_mask.astype(int),
        exclude_border=False,
    )

    # Create markers
    markers = np.zeros_like(np_mask, dtype=int)
    markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

    # Watershed
    instance_map = watershed(-dist, markers, mask=np_mask)

    # Remove small instances
    if min_size > 0:
        for inst_id in range(1, instance_map.max() + 1):
            if (instance_map == inst_id).sum() < min_size:
                instance_map[instance_map == inst_id] = 0

        # Re-label to remove gaps
        instance_map, _ = ndimage.label(instance_map > 0)

    return instance_map


def evaluate_params(
    np_mask: np.ndarray,
    hv_map: np.ndarray,
    gt_n_instances: int,
    edge_threshold: float,
    dist_threshold: int,
    min_size: int,
) -> dict:
    """
    Evaluate a single parameter combination.

    Returns:
        Dict with metrics and instance map
    """
    instance_map = watershed_post_process(
        np_mask, hv_map,
        edge_threshold=edge_threshold,
        dist_threshold=dist_threshold,
        min_size=min_size,
    )

    n_instances = instance_map.max()
    error = abs(n_instances - gt_n_instances)
    ratio = n_instances / gt_n_instances if gt_n_instances > 0 else float('inf')

    return {
        'edge_threshold': edge_threshold,
        'dist_threshold': dist_threshold,
        'min_size': min_size,
        'n_instances': n_instances,
        'gt_n_instances': gt_n_instances,
        'error': error,
        'ratio': ratio,
        'instance_map': instance_map,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize watershed parameters")
    parser.add_argument("--npz_file", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints_FIXED"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/watershed_optimization"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WATERSHED PARAMETER OPTIMIZATION")
    print("=" * 70)

    # Load GT
    data = np.load(args.npz_file)
    image = data['image']
    gt_inst = data['inst_map']
    gt_n_instances = gt_inst.max()

    print(f"\nGround Truth:")
    print(f"  Instances: {gt_n_instances}")

    # Load model and predict
    print(f"\n🤖 Loading model: {args.checkpoint_dir}")
    model = OptimusGateInferenceMultiFamily(
        checkpoint_dir=str(args.checkpoint_dir),
        device='cuda',
    )

    result = model.predict(image)

    # Extract NP mask and HV map
    if 'multifamily_result' in result:
        mf_result = result['multifamily_result']
        np_mask = mf_result.np_mask
        hv_map = mf_result.hv_map
    else:
        print("❌ No multifamily_result found!")
        return

    print(f"\nNP Mask shape: {np_mask.shape}")
    print(f"HV Map shape: {hv_map.shape}")
    print(f"HV Map range: [{hv_map.min():.3f}, {hv_map.max():.3f}]")

    # Define parameter grid
    edge_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dist_thresholds = [1, 2, 3, 4, 5, 7, 10]
    min_sizes = [5, 10, 20, 30, 50]

    print(f"\n🔍 Testing parameter grid:")
    print(f"  edge_threshold: {edge_thresholds}")
    print(f"  dist_threshold: {dist_thresholds}")
    print(f"  min_size: {min_sizes}")
    print(f"  Total combinations: {len(edge_thresholds) * len(dist_thresholds) * len(min_sizes)}")

    # Test all combinations
    results = []

    for edge_t in edge_thresholds:
        for dist_t in dist_thresholds:
            for min_s in min_sizes:
                res = evaluate_params(
                    np_mask, hv_map, gt_n_instances,
                    edge_t, dist_t, min_s
                )
                results.append(res)

    # Sort by error (ascending)
    results.sort(key=lambda x: (x['error'], abs(x['ratio'] - 1.0)))

    # Display top 10 results
    print(f"\n📊 Top 10 Parameter Combinations (by instance count match):")
    print(f"{'Rank':<5} {'Edge':<6} {'Dist':<6} {'MinSz':<6} {'Pred':<6} {'GT':<6} {'Error':<6} {'Ratio':<7}")
    print("-" * 70)

    for i, res in enumerate(results[:10], 1):
        print(f"{i:<5} {res['edge_threshold']:<6.2f} {res['dist_threshold']:<6} "
              f"{res['min_size']:<6} {res['n_instances']:<6} {res['gt_n_instances']:<6} "
              f"{res['error']:<6} {res['ratio']:<7.2f}")

    # Best result
    best = results[0]

    print(f"\n✅ BEST PARAMETERS:")
    print(f"  edge_threshold: {best['edge_threshold']}")
    print(f"  dist_threshold: {best['dist_threshold']}")
    print(f"  min_size: {best['min_size']}")
    print(f"  Predicted instances: {best['n_instances']}")
    print(f"  GT instances: {best['gt_n_instances']}")
    print(f"  Error: {best['error']} instances")
    print(f"  Ratio: {best['ratio']:.2f}x")

    # Save best instance map
    np.savez_compressed(
        args.output_dir / "best_watershed_params.npz",
        instance_map=best['instance_map'],
        edge_threshold=best['edge_threshold'],
        dist_threshold=best['dist_threshold'],
        min_size=best['min_size'],
        n_instances=best['n_instances'],
        gt_inst=gt_inst,
    )

    print(f"\n💾 Saved: {args.output_dir / 'best_watershed_params.npz'}")

    # Save full results table
    import json
    with open(args.output_dir / "all_results.json", "w") as f:
        # Remove instance_map (not JSON serializable)
        results_clean = [
            {k: v for k, v in r.items() if k != 'instance_map'}
            for r in results
        ]
        json.dump(results_clean, f, indent=2)

    print(f"💾 Saved: {args.output_dir / 'all_results.json'}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if best['error'] == 0:
        print(f"  🎯 Perfect match found! Use these parameters in production.")
    elif best['error'] <= 2:
        print(f"  ✅ Excellent match (±{best['error']} instances).")
    elif best['error'] <= 5:
        print(f"  ⚠️ Good match but some over/under-segmentation remains.")
        print(f"     Consider manual review or further tuning.")
    else:
        print(f"  ❌ Poor match ({best['error']} instances off).")
        print(f"     Possible causes:")
        print(f"     1. HV gradients too weak (check HV MSE during training)")
        print(f"     2. GT annotations incomplete")
        print(f"     3. Watershed may not be the right post-processing method")

    # Check if over-segmentation or under-segmentation
    if best['n_instances'] > gt_n_instances:
        delta = best['n_instances'] - gt_n_instances
        print(f"\n  📌 Over-segmentation by {delta} instances")
        print(f"     → Try INCREASING edge_threshold or dist_threshold")
        print(f"     → Try INCREASING min_size to filter small fragments")
    elif best['n_instances'] < gt_n_instances:
        delta = gt_n_instances - best['n_instances']
        print(f"\n  📌 Under-segmentation by {delta} instances")
        print(f"     → Try DECREASING edge_threshold or dist_threshold")
        print(f"     → Try DECREASING min_size")


if __name__ == "__main__":
    main()
