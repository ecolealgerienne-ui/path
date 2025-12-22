#!/usr/bin/env python3
"""
Diagnose watershed parameters across multiple images.

Tests the same parameters on 10 random images to see if they generalize.
"""
import argparse
import numpy as np
from pathlib import Path
import sys
import cv2
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily


def compute_gradient_magnitude(hv_map: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude from HV maps."""
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
    """Watershed with given parameters."""
    gradient = compute_gradient_magnitude(hv_map)
    edges = gradient > edge_threshold
    dist = ndimage.distance_transform_edt(~edges)

    local_max = peak_local_max(
        dist,
        min_distance=dist_threshold,
        labels=np_mask.astype(int),
        exclude_border=False,
    )

    markers = np.zeros_like(np_mask, dtype=int)
    if len(local_max) > 0:
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

    if markers.max() > 0:
        instance_map = watershed(-dist, markers, mask=np_mask)
    else:
        instance_map = ndimage.label(np_mask)[0]

    # Remove small instances
    if min_size > 0:
        for inst_id in range(1, instance_map.max() + 1):
            if (instance_map == inst_id).sum() < min_size:
                instance_map[instance_map == inst_id] = 0
        instance_map, _ = ndimage.label(instance_map > 0)

    return instance_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--force_family", type=str, default="glandular")
    args = parser.parse_args()

    # Load model
    print("🚀 Loading model...")
    model = OptimusGateInferenceMultiFamily(
        checkpoint_dir=str(args.checkpoint_dir),
        device='cuda'
    )

    # Get NPZ files
    npz_files = sorted(args.dataset_dir.glob("*.npz"))[:args.num_samples]
    print(f"📁 Testing on {len(npz_files)} images\n")

    # Current parameters from optimization
    CURRENT_PARAMS = {'edge_threshold': 0.2, 'dist_threshold': 1, 'min_size': 10}

    # Alternative parameter sets to test
    PARAM_SETS = [
        {'edge_threshold': 0.2, 'dist_threshold': 1, 'min_size': 10, 'name': 'CURRENT (optimized for 1 image)'},
        {'edge_threshold': 0.3, 'dist_threshold': 2, 'min_size': 10, 'name': 'CONSERVATIVE (less sensitive)'},
        {'edge_threshold': 0.1, 'dist_threshold': 1, 'min_size': 5, 'name': 'AGGRESSIVE (more sensitive)'},
        {'edge_threshold': 0.15, 'dist_threshold': 2, 'min_size': 8, 'name': 'BALANCED'},
    ]

    results_by_params = {p['name']: [] for p in PARAM_SETS}

    print("="*70)
    print("TESTING PARAMETER SETS")
    print("="*70)

    for npz_file in tqdm(npz_files, desc="Processing"):
        # Load GT
        data = np.load(npz_file)
        image = data['image']
        gt_inst = data['inst_map']
        gt_n = gt_inst.max()

        # Predict
        result = model.predict(image, force_family=args.force_family)
        mf_result = result['multifamily_result']
        np_mask = mf_result.np_mask
        hv_map = mf_result.hv_map

        # Test each parameter set
        for params in PARAM_SETS:
            instance_map = watershed_post_process(
                np_mask, hv_map,
                edge_threshold=params['edge_threshold'],
                dist_threshold=params['dist_threshold'],
                min_size=params['min_size']
            )

            pred_n = instance_map.max()
            error = abs(pred_n - gt_n)

            results_by_params[params['name']].append({
                'gt_n': gt_n,
                'pred_n': pred_n,
                'error': error,
                'ratio': pred_n / gt_n if gt_n > 0 else 0
            })

    # Aggregate results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    for param_set in PARAM_SETS:
        name = param_set['name']
        results = results_by_params[name]

        total_gt = sum(r['gt_n'] for r in results)
        total_pred = sum(r['pred_n'] for r in results)
        total_error = sum(r['error'] for r in results)
        avg_ratio = np.mean([r['ratio'] for r in results])

        print(f"\n{name}")
        print(f"  Params: edge={param_set['edge_threshold']}, dist={param_set['dist_threshold']}, min_size={param_set['min_size']}")
        print(f"  Total GT:    {total_gt} instances")
        print(f"  Total Pred:  {total_pred} instances")
        print(f"  Total Error: {total_error} instances")
        print(f"  Avg Ratio:   {avg_ratio:.2f}x")
        print(f"  Detection Rate: {total_pred/total_gt*100:.1f}%")

    # Find best
    best_name = min(results_by_params.keys(),
                   key=lambda n: sum(r['error'] for r in results_by_params[n]))

    print("\n" + "="*70)
    print(f"✅ BEST PARAMETERS: {best_name}")
    print("="*70)

    best_params = next(p for p in PARAM_SETS if p['name'] == best_name)
    print(f"\nRecommended settings:")
    print(f"  edge_threshold: {best_params['edge_threshold']}")
    print(f"  dist_threshold: {best_params['dist_threshold']}")
    print(f"  min_size: {best_params['min_size']}")


if __name__ == "__main__":
    main()
