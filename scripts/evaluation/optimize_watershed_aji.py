#!/usr/bin/env python3
"""
Optimisation des paramètres Watershed pour améliorer l'AJI.

Ce script effectue un grid search sur les paramètres du watershed post-processing
pour trouver la configuration optimale qui maximise l'AJI.

L'algorithme hv_to_instances est IDENTIQUE à celui de test_v13_smart_crops_aji.py
pour garantir la cohérence des résultats.

Paramètres optimisés:
- beta: Exposant de la magnitude HV pour suppression frontières (expert: ~2.0)
- min_size: Taille minimale des instances en pixels (expert: ~50)
- np_threshold: Seuil de binarisation NP (default: 0.35)
- min_distance: Distance minimale entre pics pour peak_local_max (default: 3)

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
    beta: float = 0.5,
    min_size: int = 40,
    np_threshold: float = 0.35,
    min_distance: int = 3,
    debug: bool = False
) -> np.ndarray:
    """
    HV-guided watershed for instance segmentation.

    IMPORTANT: This function matches test_v13_smart_crops_aji.py exactly.

    Args:
        np_pred: Nuclear presence mask (H, W) in [0, 1]
        hv_pred: HV maps (2, H, W) in [-1, 1]
        beta: HV magnitude exponent for boundary suppression
        min_size: Minimum instance size in pixels
        np_threshold: Threshold for NP binarization
        min_distance: Minimum distance between peaks
        debug: Print debug info for sanity checking

    Returns:
        Instance segmentation map (H, W) with unique IDs per instance
    """
    from skimage.feature import peak_local_max
    from skimage.measure import label as skimage_label

    # Threshold NP to get binary mask
    np_binary = (np_pred > np_threshold).astype(np.uint8)

    if np_binary.sum() == 0:
        return np.zeros_like(np_pred, dtype=np.int32)

    # Distance transform
    dist = distance_transform_edt(np_binary)

    # HV magnitude (range [0, sqrt(2)]) - NO NORMALIZATION
    hv_h = hv_pred[0]
    hv_v = hv_pred[1]
    hv_magnitude = np.sqrt(hv_h**2 + hv_v**2)

    # HV-guided marker energy
    marker_energy = dist * (1 - hv_magnitude ** beta)

    # Find local maxima as markers using peak_local_max
    markers_coords = peak_local_max(
        marker_energy,
        min_distance=min_distance,
        threshold_abs=0.1,
        exclude_border=False
    )

    if debug:
        print(f"    peak_local_max found {len(markers_coords)} markers")

    # Create markers map with sequential IDs
    markers = np.zeros_like(np_binary, dtype=np.int32)
    for i, (y, x) in enumerate(markers_coords, start=1):
        markers[y, x] = i

    # If no markers found, return empty
    if markers.max() == 0:
        if debug:
            print("    WARNING: No markers found!")
        return np.zeros_like(np_pred, dtype=np.int32)

    # Watershed - propagates marker IDs to all connected pixels
    inst_map = watershed(-dist, markers, mask=np_binary)

    if debug:
        unique_before = np.unique(inst_map)
        print(f"    After watershed: {len(unique_before)} unique values: {unique_before[:10]}...")

    # Remove small objects - requires labeled input
    if min_size > 0:
        # Ensure proper labeling before remove_small_objects
        inst_map = skimage_label(inst_map)
        inst_map = remove_small_objects(inst_map.astype(np.int32), min_size=min_size)
        # Relabel to ensure consecutive IDs after removal
        inst_map = skimage_label(inst_map)

    if debug:
        unique_after = np.unique(inst_map)
        n_instances = len(unique_after) - 1  # Exclude background (0)
        print(f"    After remove_small_objects: {n_instances} instances")
        if n_instances <= 1:
            print(f"    WARNING: Only {n_instances} instance(s)! unique={unique_after}")

    return inst_map.astype(np.int32)


def load_model_and_data(checkpoint_path: str, family: str, device: str = "cuda"):
    """Load model and validation data - use checkpoint metadata for hybrid mode."""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get use_hybrid and use_fpn_chimique from checkpoint metadata
    use_hybrid = checkpoint.get('use_hybrid', False)
    use_fpn_chimique = checkpoint.get('use_fpn_chimique', False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    print(f"  Checkpoint metadata: use_hybrid={use_hybrid}, use_fpn_chimique={use_fpn_chimique}")

    # Also check head dimensions for additional info
    if 'np_head.head.0.weight' in state_dict:
        head_in_channels = state_dict['np_head.head.0.weight'].shape[1]
        print(f"  Head input channels: {head_in_channels}")

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=5,
        dropout=0.1,
        use_hybrid=use_hybrid,
        use_fpn_chimique=use_fpn_chimique
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    if use_fpn_chimique:
        print(f"  ✅ Mode FPN CHIMIQUE activé: injection H-channel multi-échelle (5 niveaux)")
    elif use_hybrid:
        print(f"  ✅ Mode HYBRID activé: injection H-channel via RuifrokExtractor")

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

    # Extract predictions - handle both dict and tuple returns
    if isinstance(outputs, dict):
        np_out = outputs['np']
        hv_out = outputs['hv']
    else:
        # Tuple: (np_out, hv_out, nt_out)
        np_out, hv_out, nt_out = outputs

    # CRITICAL: Use SOFTMAX (not sigmoid!) - NP output is 2-channel for CrossEntropyLoss
    # Channel 0 = Background, Channel 1 = Nuclei
    np_probs = torch.softmax(np_out, dim=1).cpu().numpy()[0]  # (2, 224, 224)
    np_pred = np_probs[1]  # Canal 1 = Noyaux (224, 224)
    hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

    return np_pred, hv_pred


def grid_search(
    model,
    val_data,
    rgb_features,
    use_hybrid,
    n_samples: int,
    device: str,
    beta_range: list = [0.5, 1.0, 1.5, 2.0, 2.5],
    min_size_range: list = [20, 30, 40, 50, 60],
    np_threshold_range: list = [0.3, 0.35, 0.4, 0.45],
    min_distance_range: list = [2, 3, 4, 5]
):
    """
    Grid search over watershed parameters.

    Parameters optimized (matching test_v13_smart_crops_aji.py):
    - beta: HV magnitude exponent for boundary suppression (expert suggests ~2.0)
    - min_size: Minimum instance size in pixels (expert suggests ~50)
    - np_threshold: NP binarization threshold
    - min_distance: Minimum distance between peak markers

    Returns:
        Best parameters and full results grid
    """

    images = val_data['images']
    inst_maps = val_data['inst_maps']

    n_to_eval = min(n_samples, len(images))

    total_configs = len(beta_range) * len(min_size_range) * len(np_threshold_range) * len(min_distance_range)

    print(f"\n{'='*70}")
    print(f"GRID SEARCH WATERSHED PARAMETERS")
    print(f"{'='*70}")
    print(f"Samples: {n_to_eval}")
    print(f"Beta values: {beta_range}")
    print(f"Min size values: {min_size_range}")
    print(f"NP threshold values: {np_threshold_range}")
    print(f"Min distance values: {min_distance_range}")
    print(f"Total configurations: {total_configs}")
    print(f"{'='*70}\n")

    # First, run inference on all samples and cache predictions
    print("Running inference on validation samples...")
    predictions = []
    for i in range(n_to_eval):
        np_pred, hv_pred = run_inference(model, rgb_features, images, i, device, use_hybrid)
        predictions.append((np_pred, hv_pred))
        if (i + 1) % 10 == 0:
            print(f"  Inference: {i+1}/{n_to_eval}")

    # Sanity check: verify labeling works on first 3 samples
    print("\n🔍 SANITY CHECK: Verifying instance labeling on first 3 samples...")
    for i in range(min(3, n_to_eval)):
        np_pred, hv_pred = predictions[i]
        gt_inst = inst_maps[i]
        n_gt = len(np.unique(gt_inst)) - 1

        # Run with debug=True
        pred_inst = hv_to_instances(
            np_pred, hv_pred,
            beta=1.5, min_size=40, np_threshold=0.35, min_distance=3,
            debug=True
        )
        n_pred = len(np.unique(pred_inst)) - 1

        print(f"  Sample {i}: GT={n_gt} instances, Pred={n_pred} instances")

        if n_pred == 0 or n_pred == 1:
            print(f"  ⚠️  WARNING: Very few instances detected! Check labeling.")

    print("✅ Sanity check complete\n")

    print(f"\nEvaluating {total_configs} configurations...")

    # Grid search
    results = []
    best_aji = 0
    best_params = None

    config_idx = 0

    for beta, min_size, np_thresh, min_dist in product(beta_range, min_size_range, np_threshold_range, min_distance_range):
        config_idx += 1

        ajis = []
        n_pred_instances = []
        n_gt_instances = []

        for i in range(n_to_eval):
            np_pred, hv_pred = predictions[i]
            gt_inst = inst_maps[i]

            # Convert to instances with current parameters
            pred_inst = hv_to_instances(
                np_pred, hv_pred,
                beta=beta,
                min_size=min_size,
                np_threshold=np_thresh,
                min_distance=min_dist
            )

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
            'np_threshold': np_thresh,
            'min_distance': min_dist,
            'aji_mean': mean_aji,
            'aji_std': std_aji,
            'n_pred': mean_n_pred,
            'n_gt': mean_n_gt,
            'over_seg_ratio': over_seg_ratio
        })

        if mean_aji > best_aji:
            best_aji = mean_aji
            best_params = {
                'beta': beta,
                'min_size': min_size,
                'np_threshold': np_thresh,
                'min_distance': min_dist
            }

        if config_idx % 50 == 0:
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
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Beta':<8} {'MinSize':<10} {'NP_Thr':<10} {'MinDist':<10} {'AJI Mean':<12} {'OverSeg':<10}")
    print(f"{'-'*80}")

    for i, r in enumerate(results_sorted[:10]):
        print(f"{i+1:<6} {r['beta']:<8.2f} {r['min_size']:<10} {r['np_threshold']:<10.2f} "
              f"{r['min_distance']:<10} {r['aji_mean']:<12.4f} {r['over_seg_ratio']:<10.2f}")

    # Print best configuration
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Beta:            {best_params['beta']}")
    print(f"  Min Size:        {best_params['min_size']}")
    print(f"  NP Threshold:    {best_params['np_threshold']}")
    print(f"  Min Distance:    {best_params['min_distance']}")
    print(f"  AJI Mean:        {results_sorted[0]['aji_mean']:.4f} ± {results_sorted[0]['aji_std']:.4f}")
    print(f"  Over-seg Ratio:  {results_sorted[0]['over_seg_ratio']:.2f}")
    print(f"{'='*80}")

    # Compare with default parameters (from test_v13_smart_crops_aji.py)
    default_result = next((r for r in results if r['beta'] == 0.5 and r['min_size'] == 40
                           and r['np_threshold'] == 0.35 and r['min_distance'] == 3), None)
    if default_result:
        improvement = (results_sorted[0]['aji_mean'] - default_result['aji_mean']) / default_result['aji_mean'] * 100
        print(f"\nComparison with default (beta=0.5, min_size=40, np_threshold=0.35, min_distance=3):")
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
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    print(f"Update hv_guided_watershed() in test_v13_smart_crops_aji.py with:")
    print(f"  beta={best_params['beta']}")
    print(f"  min_size={best_params['min_size']}")
    print(f"  np_threshold={best_params['np_threshold']}")
    print(f"  min_distance={best_params['min_distance']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
