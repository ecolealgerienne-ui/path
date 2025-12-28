#!/usr/bin/env python3
"""
Optimisation des paramètres Watershed pour améliorer l'AJI.

Ce script effectue un grid search sur les paramètres du watershed post-processing
pour trouver la configuration optimale qui maximise l'AJI.

Paramètres optimisés:
- beta: Poids de la magnitude HV dans l'énergie (0.5 à 2.0)
- min_size: Taille minimale des instances (10 à 50 pixels)
- dist_threshold: Seuil de distance pour les markers (0.3 à 0.7)

Usage:
    python scripts/evaluation/optimize_watershed_aji.py \
        --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_best.pth \
        --family epidermal \
        --n_samples 50
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
import json
from datetime import datetime
from itertools import product
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder


def compute_aji(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    """
    Compute Aggregated Jaccard Index (AJI).

    AJI = sum(|P_i ∩ G_σ(i)|) / sum(|P_i ∪ G_σ(i)|) + sum(|G_j - ∪P|)

    Where σ(i) is the GT instance with maximum IoU for prediction i.
    """
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]

    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0

    if len(pred_ids) == 0:
        return 0.0

    # For each prediction, find best matching GT
    used_gt = set()
    inter_sum = 0
    union_sum = 0

    for pred_id in pred_ids:
        pred_mask = pred_inst == pred_id

        best_iou = 0
        best_gt_id = None
        best_inter = 0
        best_union = 0

        for gt_id in gt_ids:
            if gt_id in used_gt:
                continue
            gt_mask = gt_inst == gt_id
            inter = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = inter / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
                best_inter = inter
                best_union = union

        if best_gt_id is not None:
            used_gt.add(best_gt_id)
            inter_sum += best_inter
            union_sum += best_union
        else:
            # No match - add pred to union only
            union_sum += pred_mask.sum()

    # Add unmatched GT instances
    for gt_id in gt_ids:
        if gt_id not in used_gt:
            gt_mask = gt_inst == gt_id
            union_sum += gt_mask.sum()

    return inter_sum / union_sum if union_sum > 0 else 0.0


def hv_to_instances(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    beta: float = 1.5,
    min_size: int = 40,
    dist_threshold: float = 0.5
) -> np.ndarray:
    """
    Convert NP mask and HV maps to instance segmentation using watershed.

    Args:
        np_pred: Binary nuclear presence mask (H, W)
        hv_pred: HV maps (2, H, W) with H and V components
        beta: Weight for HV magnitude in energy (higher = more separation)
        min_size: Minimum instance size in pixels
        dist_threshold: Threshold for distance transform (fraction of max)

    Returns:
        Instance segmentation map (H, W) with unique IDs per instance
    """
    if np_pred.sum() == 0:
        return np.zeros_like(np_pred, dtype=np.int32)

    # Binary mask
    binary_mask = (np_pred > 0.5).astype(np.uint8)

    # Distance transform
    dist = distance_transform_edt(binary_mask)

    # HV magnitude
    h_map = hv_pred[0]
    v_map = hv_pred[1]
    hv_mag = np.sqrt(h_map**2 + v_map**2)

    # Normalize HV magnitude to [0, 1]
    hv_mag_norm = (hv_mag - hv_mag.min()) / (hv_mag.max() - hv_mag.min() + 1e-8)

    # Energy: combination of distance and HV magnitude
    # High energy at centers (high dist, low HV mag)
    # Low energy at boundaries (low dist, high HV mag)
    energy = dist * (1 - hv_mag_norm ** beta)

    # Find markers (local maxima of energy within mask)
    markers_mask = (energy > dist_threshold * energy.max()) & (binary_mask > 0)
    markers, n_markers = label(markers_mask)

    if n_markers == 0:
        # Fallback: use connected components
        markers, n_markers = label(binary_mask)

    # Watershed
    inst_map = watershed(-energy, markers, mask=binary_mask)

    # Remove small objects
    if min_size > 0:
        inst_map = remove_small_objects(inst_map.astype(np.int32), min_size=min_size)
        # Relabel to sequential IDs
        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]
        new_inst_map = np.zeros_like(inst_map)
        for new_id, old_id in enumerate(unique_ids, start=1):
            new_inst_map[inst_map == old_id] = new_id
        inst_map = new_inst_map

    return inst_map


def load_model_and_data(checkpoint_path: str, family: str, device: str = "cuda"):
    """Load model and validation data with auto-detection of architecture."""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Auto-detect architecture from checkpoint
    has_ruifrok = any('ruifrok' in k for k in state_dict.keys())
    has_h_projection = any('h_projection' in k for k in state_dict.keys())

    # Check head input dimension
    if 'np_head.head.0.weight' in state_dict:
        head_in_channels = state_dict['np_head.head.0.weight'].shape[1]
    else:
        head_in_channels = 64

    # Determine configuration
    if head_in_channels == 80:
        use_hybrid = True
        print(f"  Auto-detected: V3 Hybrid (16 H-channels)")
    elif head_in_channels == 65:
        use_hybrid = True
        print(f"  Auto-detected: V2 Hybrid (1 H-channel)")
    elif has_ruifrok or has_h_projection:
        use_hybrid = True
        print(f"  Auto-detected: Hybrid mode ({head_in_channels} input)")
    else:
        use_hybrid = False
        print(f"  Auto-detected: Non-hybrid ({head_in_channels} input)")

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=5,
        use_hybrid=use_hybrid
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # Load validation data
    data_dir = Path("data/family_data_v13_smart_crops")
    val_file = data_dir / f"{family}_val_v13_smart_crops.npz"

    if not val_file.exists():
        raise FileNotFoundError(f"Validation data not found: {val_file}")

    val_data = np.load(val_file, allow_pickle=True)

    # Load features
    features_dir = Path("data/cache/family_data")
    rgb_features_file = features_dir / f"{family}_rgb_features_v13_smart_crops_val.npz"

    if not rgb_features_file.exists():
        raise FileNotFoundError(f"RGB features not found: {rgb_features_file}")

    rgb_features = np.load(rgb_features_file, allow_pickle=True)['features']

    return model, val_data, rgb_features, use_hybrid


def run_inference(model, rgb_features, images, idx, device, use_hybrid):
    """Run inference on a single sample."""

    # Prepare input
    features = torch.from_numpy(rgb_features[idx:idx+1]).float().to(device)

    # For hybrid model, need images
    if use_hybrid:
        image = images[idx]
        # Ensure correct format (C, H, W) and normalize
        if image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
        if image_tensor.max() > 1:
            image_tensor = image_tensor / 255.0
    else:
        image_tensor = None

    # Run inference
    with torch.no_grad():
        outputs = model(features, images_rgb=image_tensor)

    # Extract predictions
    np_pred = torch.sigmoid(outputs['np']).cpu().numpy()[0, 0]
    hv_pred = outputs['hv'].cpu().numpy()[0]

    return np_pred, hv_pred


def grid_search(
    model,
    val_data,
    rgb_features,
    use_hybrid,
    n_samples: int,
    device: str,
    beta_range: list = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    min_size_range: list = [10, 20, 30, 40, 50],
    dist_threshold_range: list = [0.3, 0.4, 0.5, 0.6, 0.7]
):
    """
    Grid search over watershed parameters.

    Returns:
        Best parameters and full results grid
    """

    images = val_data['images']
    inst_maps = val_data['inst_maps']

    n_to_eval = min(n_samples, len(images))

    print(f"\n{'='*70}")
    print(f"GRID SEARCH WATERSHED PARAMETERS")
    print(f"{'='*70}")
    print(f"Samples: {n_to_eval}")
    print(f"Beta values: {beta_range}")
    print(f"Min size values: {min_size_range}")
    print(f"Dist threshold values: {dist_threshold_range}")
    print(f"Total configurations: {len(beta_range) * len(min_size_range) * len(dist_threshold_range)}")
    print(f"{'='*70}\n")

    # First, run inference on all samples and cache predictions
    print("Running inference on validation samples...")
    predictions = []
    for i in range(n_to_eval):
        np_pred, hv_pred = run_inference(model, rgb_features, images, i, device, use_hybrid)
        predictions.append((np_pred, hv_pred))
        if (i + 1) % 10 == 0:
            print(f"  Inference: {i+1}/{n_to_eval}")

    print(f"\nEvaluating {len(beta_range) * len(min_size_range) * len(dist_threshold_range)} configurations...")

    # Grid search
    results = []
    best_aji = 0
    best_params = None

    total_configs = len(beta_range) * len(min_size_range) * len(dist_threshold_range)
    config_idx = 0

    for beta, min_size, dist_thresh in product(beta_range, min_size_range, dist_threshold_range):
        config_idx += 1

        ajis = []
        n_pred_instances = []
        n_gt_instances = []

        for i in range(n_to_eval):
            np_pred, hv_pred = predictions[i]
            gt_inst = inst_maps[i]

            # Convert to instances with current parameters
            pred_inst = hv_to_instances(np_pred, hv_pred, beta, min_size, dist_thresh)

            # Compute AJI
            aji = compute_aji(pred_inst, gt_inst)
            ajis.append(aji)

            n_pred_instances.append(len(np.unique(pred_inst)) - 1)  # Exclude background
            n_gt_instances.append(len(np.unique(gt_inst)) - 1)

        mean_aji = np.mean(ajis)
        std_aji = np.std(ajis)
        mean_n_pred = np.mean(n_pred_instances)
        mean_n_gt = np.mean(n_gt_instances)
        over_seg_ratio = mean_n_pred / mean_n_gt if mean_n_gt > 0 else 0

        results.append({
            'beta': beta,
            'min_size': min_size,
            'dist_threshold': dist_thresh,
            'aji_mean': mean_aji,
            'aji_std': std_aji,
            'n_pred': mean_n_pred,
            'n_gt': mean_n_gt,
            'over_seg_ratio': over_seg_ratio
        })

        if mean_aji > best_aji:
            best_aji = mean_aji
            best_params = {'beta': beta, 'min_size': min_size, 'dist_threshold': dist_thresh}

        if config_idx % 20 == 0:
            print(f"  Progress: {config_idx}/{total_configs} - Current best AJI: {best_aji:.4f}")

    return best_params, results


def main():
    parser = argparse.ArgumentParser(description="Optimize watershed parameters for AJI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--family", type=str, default="epidermal", help="Family to evaluate")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="results/watershed_optimization",
                        help="Output directory for results")

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model and data
    print(f"\nLoading model from: {args.checkpoint}")
    model, val_data, rgb_features, use_hybrid = load_model_and_data(
        args.checkpoint, args.family, args.device
    )
    print(f"Model loaded. Hybrid mode: {use_hybrid}")
    print(f"Validation samples: {len(val_data['images'])}")

    # Run grid search
    best_params, results = grid_search(
        model, val_data, rgb_features, use_hybrid,
        args.n_samples, args.device
    )

    # Sort results by AJI
    results_sorted = sorted(results, key=lambda x: x['aji_mean'], reverse=True)

    # Print top 10 configurations
    print(f"\n{'='*70}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Beta':<8} {'MinSize':<10} {'DistThr':<10} {'AJI Mean':<12} {'AJI Std':<10} {'OverSeg':<10}")
    print(f"{'-'*70}")

    for i, r in enumerate(results_sorted[:10]):
        print(f"{i+1:<6} {r['beta']:<8.2f} {r['min_size']:<10} {r['dist_threshold']:<10.2f} "
              f"{r['aji_mean']:<12.4f} {r['aji_std']:<10.4f} {r['over_seg_ratio']:<10.2f}")

    # Print best configuration
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Beta:            {best_params['beta']}")
    print(f"  Min Size:        {best_params['min_size']}")
    print(f"  Dist Threshold:  {best_params['dist_threshold']}")
    print(f"  AJI Mean:        {results_sorted[0]['aji_mean']:.4f} ± {results_sorted[0]['aji_std']:.4f}")
    print(f"  Over-seg Ratio:  {results_sorted[0]['over_seg_ratio']:.2f}")
    print(f"{'='*70}")

    # Compare with default parameters
    default_result = next((r for r in results if r['beta'] == 1.5 and r['min_size'] == 40 and r['dist_threshold'] == 0.5), None)
    if default_result:
        improvement = (results_sorted[0]['aji_mean'] - default_result['aji_mean']) / default_result['aji_mean'] * 100
        print(f"\nComparison with default (beta=1.5, min_size=40, dist_thresh=0.5):")
        print(f"  Default AJI:   {default_result['aji_mean']:.4f}")
        print(f"  Optimized AJI: {results_sorted[0]['aji_mean']:.4f}")
        print(f"  Improvement:   {improvement:+.1f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"watershed_optimization_{args.family}_{timestamp}.json"

    output_data = {
        'family': args.family,
        'n_samples': args.n_samples,
        'best_params': best_params,
        'best_aji': results_sorted[0]['aji_mean'],
        'all_results': results_sorted,
        'timestamp': timestamp
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print usage recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print(f"Update hv_to_instances() in test_v13_smart_crops_aji.py with:")
    print(f"  beta={best_params['beta']}")
    print(f"  min_size={best_params['min_size']}")
    print(f"  dist_threshold={best_params['dist_threshold']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
