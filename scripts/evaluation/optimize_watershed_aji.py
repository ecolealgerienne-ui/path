#!/usr/bin/env python3
"""
Optimisation des paramètres Watershed pour améliorer l'AJI.

Ce script effectue un grid search sur les paramètres du watershed post-processing
pour trouver la configuration optimale qui maximise l'AJI.

UTILISE LE MODULE PARTAGÉ src.evaluation pour garantir la cohérence
avec test_v13_smart_crops_aji.py (single source of truth).

Usage:
    python scripts/evaluation/optimize_watershed_aji.py \
        --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_best.pth \
        --family epidermal \
        --n_samples 50
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from itertools import product
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder
from src.evaluation import run_inference, evaluate_batch_with_params
from src.postprocessing import hv_guided_watershed  # Pour sanity check seulement


def load_model_and_data(checkpoint_path: str, family: str, device: str = "cuda", data_prefix: str = None):
    """Load model and validation data - use checkpoint metadata for hybrid mode."""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get use_hybrid and use_fpn_chimique from checkpoint metadata
    use_hybrid = checkpoint.get('use_hybrid', False)
    use_fpn_chimique = checkpoint.get('use_fpn_chimique', False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    print(f"  Checkpoint metadata: use_hybrid={use_hybrid}, use_fpn_chimique={use_fpn_chimique}")

    # Detect use_h_alpha from checkpoint (check if h_alphas params exist)
    use_h_alpha = any('h_alphas' in k for k in state_dict.keys())

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=5,
        dropout=0.1,
        use_hybrid=use_hybrid,
        use_fpn_chimique=use_fpn_chimique,
        use_h_alpha=use_h_alpha
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    if use_fpn_chimique:
        print(f"  ✅ Mode FPN CHIMIQUE activé: injection H-channel multi-échelle (5 niveaux)")
    if use_h_alpha:
        print(f"  ✅ Mode H-Alpha activé: facteur α learnable par niveau")
    elif use_hybrid:
        print(f"  ✅ Mode HYBRID activé: injection H-channel via RuifrokExtractor")

    # Load validation data (utilise data_prefix si spécifié)
    prefix = data_prefix if data_prefix else family
    data_dir = Path("data/family_data_v13_smart_crops")
    val_file = data_dir / f"{prefix}_val_v13_smart_crops.npz"

    if not val_file.exists():
        raise FileNotFoundError(f"Validation data not found: {val_file}")

    val_data = np.load(val_file, allow_pickle=True)

    # Load features (utilise prefix)
    features_dir = Path("data/cache/family_data")
    rgb_features_file = features_dir / f"{prefix}_rgb_features_v13_smart_crops_val.npz"

    if not rgb_features_file.exists():
        raise FileNotFoundError(f"RGB features not found: {rgb_features_file}")

    rgb_features = np.load(rgb_features_file, allow_pickle=True)['features']

    return model, val_data, rgb_features, use_hybrid


def cache_predictions(model, val_data, rgb_features, use_hybrid, n_samples, device):
    """Run inference once and cache all predictions."""
    images = val_data['images']
    n_to_eval = min(n_samples, len(images))

    print("Running inference on validation samples...")
    predictions = []

    for i in range(n_to_eval):
        # Prepare inputs
        features = torch.from_numpy(rgb_features[i:i+1]).float().to(device)

        if use_hybrid:
            image = images[i]
            if image.shape[-1] == 3:
                image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
            if image_tensor.max() > 1:
                image_tensor = image_tensor / 255.0
        else:
            image_tensor = None

        # Use shared inference function
        np_pred, hv_pred = run_inference(model, features, image_tensor, device)
        predictions.append((np_pred, hv_pred))

        if (i + 1) % 10 == 0:
            print(f"  Inference: {i+1}/{n_to_eval}")

    return predictions


def grid_search(
    predictions,
    gt_instances,
    beta_range: list = [0.5, 1.0, 1.5, 2.0, 2.5],
    min_size_range: list = [20, 30, 40, 50, 60],
    np_threshold_range: list = [0.3, 0.35, 0.4, 0.45],
    min_distance_range: list = [2, 3, 4, 5]
):
    """
    Grid search over watershed parameters using shared evaluation module.

    Uses evaluate_batch_with_params from src.evaluation (single source of truth).
    """
    total_configs = len(beta_range) * len(min_size_range) * len(np_threshold_range) * len(min_distance_range)

    print(f"\n{'='*70}")
    print(f"GRID SEARCH WATERSHED PARAMETERS")
    print(f"{'='*70}")
    print(f"Samples: {len(predictions)}")
    print(f"Beta values: {beta_range}")
    print(f"Min size values: {min_size_range}")
    print(f"NP threshold values: {np_threshold_range}")
    print(f"Min distance values: {min_distance_range}")
    print(f"Total configurations: {total_configs}")
    print(f"{'='*70}\n")

    # Sanity check
    print("🔍 SANITY CHECK: Verifying instance labeling on first 3 samples...")
    for i in range(min(3, len(predictions))):
        np_pred, hv_pred = predictions[i]
        gt_inst = gt_instances[i]
        n_gt = len(np.unique(gt_inst)) - 1

        pred_inst = hv_guided_watershed(
            np_pred, hv_pred,
            beta=1.5, min_size=40, np_threshold=0.35, min_distance=3,
            debug=True
        )
        n_pred = len(np.unique(pred_inst)) - 1
        print(f"  Sample {i}: GT={n_gt} instances, Pred={n_pred} instances")

    print("✅ Sanity check complete\n")

    print(f"Evaluating {total_configs} configurations...")

    # Grid search using shared evaluation module
    results = []
    best_aji = 0
    best_params = None
    config_idx = 0

    for beta, min_size, np_thresh, min_dist in product(beta_range, min_size_range, np_threshold_range, min_distance_range):
        config_idx += 1

        # Use shared evaluation function (single source of truth)
        metrics = evaluate_batch_with_params(
            predictions=predictions,
            gt_instances=gt_instances,
            np_threshold=np_thresh,
            beta=beta,
            min_size=min_size,
            min_distance=min_dist
        )

        results.append({
            'beta': beta,
            'min_size': min_size,
            'np_threshold': np_thresh,
            'min_distance': min_dist,
            'aji_mean': metrics['aji_mean'],
            'aji_std': metrics['aji_std'],
            'n_pred': metrics['n_pred_mean'],
            'n_gt': metrics['n_gt_mean'],
            'over_seg_ratio': metrics['over_seg_ratio']
        })

        if metrics['aji_mean'] > best_aji:
            best_aji = metrics['aji_mean']
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
    parser.add_argument("--organ", type=str, default=None,
                        help="Specific organ (optional). If specified, loads {organ}_val.npz instead of {family}_val.npz")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="results/watershed_optimization",
                        help="Output directory for results")

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Préfixe pour les fichiers de données (organe ou famille)
    data_prefix = args.organ.lower() if args.organ else args.family

    # Load model and data
    print(f"\nLoading model from: {args.checkpoint}")
    if args.organ:
        print(f"  Organ filter: {args.organ}")
        print(f"  Data prefix: {data_prefix}")
    model, val_data, rgb_features, use_hybrid = load_model_and_data(
        args.checkpoint, args.family, args.device, data_prefix=data_prefix
    )
    print(f"Model loaded. Hybrid mode: {use_hybrid}")
    print(f"Validation samples: {len(val_data['images'])}")

    # Cache predictions (inference runs once)
    predictions = cache_predictions(
        model, val_data, rgb_features, use_hybrid,
        args.n_samples, args.device
    )

    # Get GT instances
    gt_instances = list(val_data['inst_maps'][:len(predictions)])

    # Run grid search using shared evaluation module
    best_params, results = grid_search(predictions, gt_instances)

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

    # Compare with default parameters
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
    results_file = output_dir / f"watershed_optimization_{data_prefix}_{timestamp}.json"

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
    print(f"Use these parameters in test_v13_smart_crops_aji.py:")
    print(f"  --beta {best_params['beta']}")
    print(f"  --min_size {best_params['min_size']}")
    print(f"  --np_threshold {best_params['np_threshold']}")
    print(f"  --min_distance {best_params['min_distance']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
