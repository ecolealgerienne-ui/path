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


def load_model_and_data(checkpoint_path: str, family: str, device: str = "cuda", organ: str = None):
    """Load model and validation data - use checkpoint metadata for hybrid mode.

    Args:
        checkpoint_path: Path to model checkpoint
        family: Family name (respiratory, digestive, etc.)
        device: Device to use (string like "cuda" or "cpu")
        organ: Optional organ filter. If specified, only samples from this organ are returned.
               Uses dynamic filtering from family data (no regeneration needed).
    """
    # Convert to torch.device for explicit GPU handling
    device = torch.device(device)
    print(f"  🖥️  Device: {device} (CUDA available: {torch.cuda.is_available()})")

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

    # Load validation data (always from family file)
    data_dir = Path("data/family_data_v13_smart_crops")
    val_file = data_dir / f"{family}_val_v13_smart_crops.npz"

    if not val_file.exists():
        raise FileNotFoundError(f"Validation data not found: {val_file}")

    val_data_raw = np.load(val_file, allow_pickle=True)

    # Load features (always from family file)
    features_dir = Path("data/cache/family_data")
    rgb_features_file = features_dir / f"{family}_rgb_features_v13_smart_crops_val.npz"

    if not rgb_features_file.exists():
        raise FileNotFoundError(f"RGB features not found: {rgb_features_file}")

    rgb_features_raw = np.load(rgb_features_file, allow_pickle=True)['features']

    # =========================================================================
    # FILTRAGE DYNAMIQUE PAR ORGANE (sans régénération des données)
    # =========================================================================
    if organ:
        organ_names = val_data_raw.get('organ_names', None)
        if organ_names is None:
            raise ValueError(f"organ_names not found in {val_file}. Regenerate data with latest prepare_v13_smart_crops.py")

        # Normaliser les noms (gérer bytes vs str)
        organ_names = np.array([
            name.decode('utf-8') if isinstance(name, bytes) else name
            for name in organ_names
        ])

        # Créer le masque de filtre
        organ_mask = organ_names == organ
        n_organ = organ_mask.sum()

        if n_organ == 0:
            available_organs = np.unique(organ_names)
            raise ValueError(f"No samples found for organ '{organ}'. Available: {available_organs}")

        print(f"  🔬 Filtrage organe: {organ} → {n_organ}/{len(organ_names)} samples")

        # Appliquer le masque sur toutes les données
        val_data = {
            'images': val_data_raw['images'][organ_mask],
            'inst_maps': val_data_raw['inst_maps'][organ_mask],
            'organ_names': organ_names[organ_mask],
        }
        # Copier les autres clés si présentes
        for key in ['hv_targets', 'nt_targets', 'source_image_ids']:
            if key in val_data_raw.files:
                val_data[key] = val_data_raw[key][organ_mask]

        rgb_features = rgb_features_raw[organ_mask]
    else:
        # Mode famille: utiliser toutes les données
        val_data = val_data_raw
        rgb_features = rgb_features_raw

    return model, val_data, rgb_features, use_hybrid, device


def cache_predictions(model, val_data, rgb_features, use_hybrid, n_samples, device, batch_size=16):
    """Run inference once and cache all predictions using batch processing on GPU."""
    images = val_data['images']
    n_to_eval = min(n_samples, len(images))

    # Ensure device is torch.device
    if isinstance(device, str):
        device = torch.device(device)

    print(f"\n🔥 Running batch inference on {device} (batch_size={batch_size})...")
    predictions = []

    # Process in batches for GPU efficiency
    for batch_start in range(0, n_to_eval, batch_size):
        batch_end = min(batch_start + batch_size, n_to_eval)
        batch_indices = range(batch_start, batch_end)
        current_batch_size = len(batch_indices)

        # Prepare batch on CPU, then transfer once to GPU
        features_batch = torch.from_numpy(rgb_features[batch_start:batch_end]).float().to(device)

        if use_hybrid:
            # Prepare images batch
            images_list = []
            for i in batch_indices:
                image = images[i]
                if image.shape[-1] == 3:
                    image = np.transpose(image, (2, 0, 1))
                images_list.append(image)

            images_batch = torch.from_numpy(np.stack(images_list)).float().to(device)
            if images_batch.max() > 1:
                images_batch = images_batch / 255.0
        else:
            images_batch = None

        # Batch inference on GPU
        with torch.no_grad():
            outputs = model(features_batch, images_rgb=images_batch)

            if isinstance(outputs, dict):
                np_out = outputs['np']
                hv_out = outputs['hv']
            else:
                np_out, hv_out, _ = outputs

            # Softmax for NP
            np_probs = torch.softmax(np_out, dim=1)[:, 1].cpu().numpy()  # (B, 224, 224)
            hv_preds = hv_out.cpu().numpy()  # (B, 2, 224, 224)

        # Store predictions
        for j in range(current_batch_size):
            predictions.append((np_probs[j], hv_preds[j]))

        print(f"  Batch {batch_start//batch_size + 1}/{(n_to_eval + batch_size - 1)//batch_size}: {batch_end}/{n_to_eval} samples")

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
                        help="Specific organ (optional). Dynamic filtering from family data (no regeneration needed). Ex: Lung, Colon, Breast")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for GPU inference (default: 16)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="results/watershed_optimization",
                        help="Output directory for results")

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model and data (with optional organ filter - dynamic, no regeneration needed)
    print(f"\nLoading model from: {args.checkpoint}")
    print(f"  Family: {args.family}")
    if args.organ:
        print(f"  Organ filter: {args.organ} (dynamic filtering)")
    model, val_data, rgb_features, use_hybrid, device = load_model_and_data(
        args.checkpoint, args.family, args.device, organ=args.organ
    )
    print(f"Model loaded. Hybrid mode: {use_hybrid}")
    print(f"Validation samples: {len(val_data['images'])}")
    print(f"  🚀 Model on: {next(model.parameters()).device}")

    # Cache predictions (batch inference on GPU - much faster)
    predictions = cache_predictions(
        model, val_data, rgb_features, use_hybrid,
        args.n_samples, device, batch_size=args.batch_size
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

    # Nom du fichier: organe si spécifié, sinon famille
    output_prefix = args.organ.lower() if args.organ else args.family
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"watershed_optimization_{output_prefix}_{timestamp}.json"

    output_data = {
        'family': args.family,
        'organ': args.organ,  # None si mode famille
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
