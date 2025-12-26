#!/usr/bin/env python3
"""
Optimize Watershed Parameters for V13-Hybrid AJI.

Tests different beta and min_size values to find optimal parameters
that maximize AJI metric while reducing over-segmentation.

Usage:
    python scripts/evaluation/optimize_watershed_params.py \
        --checkpoint models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_best.pth \
        --family epidermal \
        --n_samples 50

Author: CellViT-Optimus Project
Date: 2025-12-26
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, label
from skimage.feature import peak_local_max
from typing import Dict
import json
from datetime import datetime
from tqdm import tqdm

from src.models.hovernet_decoder_hybrid import HoVerNetDecoderHybrid
from src.metrics.ground_truth_metrics import compute_aji

# Import H-channel CNN
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocessing'))
from extract_h_features_v13 import HChannelCNN


def hv_guided_watershed(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    beta: float = 1.0,
    min_size: int = 20
) -> np.ndarray:
    """HV-guided watershed for instance segmentation."""
    binary_mask = (np_pred > 0.5).astype(np.uint8)
    
    h_map = hv_pred[0]
    v_map = hv_pred[1]
    hv_magnitude = np.sqrt(h_map**2 + v_map**2)
    hv_magnitude = np.clip(hv_magnitude / np.sqrt(2), 0, 1)
    
    dist = distance_transform_edt(binary_mask)
    marker_energy = dist * (1 - hv_magnitude ** beta)
    
    threshold = 0.3 * marker_energy.max()
    markers_binary = (marker_energy > threshold).astype(np.uint8)
    markers = label(markers_binary)  # skimage.morphology.label returns only labeled array

    elevation = -marker_energy
    inst_map = watershed(elevation, markers, mask=binary_mask)
    inst_map = remove_small_objects(inst_map, min_size=min_size).astype(np.int32)
    
    return inst_map


def evaluate_params(model, rgb_features, h_features, gt_inst, beta, min_size, device):
    """Evaluate AJI for given watershed parameters."""
    n_samples = len(rgb_features)
    aji_scores = []
    n_pred_instances = []
    n_gt_instances = []
    
    with torch.no_grad():
        for i in range(n_samples):
            rgb_feat = torch.from_numpy(rgb_features[i]).unsqueeze(0).to(device)
            h_feat = torch.from_numpy(h_features[i]).unsqueeze(0).to(device)
            patch_tokens = rgb_feat[:, 5:261, :]
            
            output = model(patch_tokens, h_feat)
            result = output.to_numpy(apply_activations=True)
            
            np_pred = result['np'][0, 1]
            hv_pred = result['hv'][0]
            
            np_pred_256 = cv2.resize(np_pred, (256, 256), interpolation=cv2.INTER_LINEAR)
            hv_pred_256 = np.zeros((2, 256, 256), dtype=np.float32)
            hv_pred_256[0] = cv2.resize(hv_pred[0], (256, 256), interpolation=cv2.INTER_LINEAR)
            hv_pred_256[1] = cv2.resize(hv_pred[1], (256, 256), interpolation=cv2.INTER_LINEAR)
            
            pred_inst = hv_guided_watershed(np_pred_256, hv_pred_256, beta=beta, min_size=min_size)
            
            gt_inst_i = gt_inst[i]
            if gt_inst_i.shape != (256, 256):
                gt_inst_256 = cv2.resize(gt_inst_i.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            else:
                gt_inst_256 = gt_inst_i
            
            aji = compute_aji(pred_inst, gt_inst_256)
            aji_scores.append(aji)
            n_pred_instances.append(len(np.unique(pred_inst)) - 1)
            n_gt_instances.append(len(np.unique(gt_inst_256)) - 1)
    
    return {
        'aji_mean': np.mean(aji_scores),
        'aji_std': np.std(aji_scores),
        'aji_median': np.median(aji_scores),
        'n_pred': np.mean(n_pred_instances),
        'n_gt': np.mean(n_gt_instances),
        'over_seg_ratio': np.mean(n_pred_instances) / (np.mean(n_gt_instances) + 1e-8)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--family', type=str, default='epidermal')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("=" * 80)
    print("WATERSHED PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Load model
    print("\n🔧 Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = HoVerNetDecoderHybrid(
        embed_dim=1536, h_dim=256, n_classes=5,
        dropout=checkpoint['args']['dropout'] if 'args' in checkpoint else 0.1
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✅ Model loaded (epoch {checkpoint['epoch']})")
    
    # Load data
    print("\n📂 Loading test samples...")
    hybrid_data_path = Path(f"data/family_data_v13_hybrid/{args.family}_data_v13_hybrid.npz")
    h_features_path = Path(f"data/cache/family_data/{args.family}_h_features_v13.npz")
    rgb_features_path = Path(f"data/cache/family_data/{args.family}_rgb_features_v13.npz")

    hybrid_data = np.load(hybrid_data_path)
    h_data = np.load(h_features_path)
    rgb_data = np.load(rgb_features_path)

    # Use SAME split logic as training (based on source_image_ids, not simple slice)
    fold_ids = hybrid_data['fold_ids']
    source_image_ids = hybrid_data['source_image_ids']

    unique_source_ids = np.unique(source_image_ids)
    n_unique = len(unique_source_ids)
    n_train_unique = int(0.8 * n_unique)

    np.random.seed(42)  # Same seed as training
    shuffled_ids = np.random.permutation(unique_source_ids)

    train_source_ids = shuffled_ids[:n_train_unique]
    val_source_ids = shuffled_ids[n_train_unique:]

    val_mask = np.isin(source_image_ids, val_source_ids)
    val_indices = np.where(val_mask)[0]

    print(f"  Validation samples available: {len(val_indices)}")

    n_to_load = min(args.n_samples, len(val_indices))
    selected_indices = val_indices[:n_to_load]

    print(f"  Loading {n_to_load} samples for evaluation")

    rgb_features = rgb_data['features'][selected_indices]
    h_features = h_data['h_features'][selected_indices]
    np_targets = hybrid_data['np_targets'][selected_indices]
    
    # Generate GT
    print(f"  🔧 Generating GT instance maps...")
    gt_inst_maps = []
    for i in range(n_to_load):
        binary_mask = (np_targets[i] > 0.5).astype(np.uint8)
        dist = distance_transform_edt(binary_mask)
        local_maxi = peak_local_max(dist, min_distance=5, labels=binary_mask, exclude_border=False)
        markers_gt = np.zeros_like(dist, dtype=int)
        for idx, (y, x) in enumerate(local_maxi):
            markers_gt[y, x] = idx + 1
        gt_inst = watershed(-dist, markers_gt, mask=binary_mask)
        gt_inst_maps.append(gt_inst)
    gt_inst = np.array(gt_inst_maps)
    print(f"  ✅ Loaded {n_to_load} samples")
    
    # Parameter grid
    beta_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    min_size_values = [10, 20, 30, 40]
    
    print(f"\n🔬 Testing {len(beta_values)} × {len(min_size_values)} = {len(beta_values) * len(min_size_values)} combinations...")
    
    results = []
    for beta in tqdm(beta_values, desc="Beta"):
        for min_size in min_size_values:
            metrics = evaluate_params(model, rgb_features, h_features, gt_inst, beta, min_size, args.device)
            results.append({'beta': beta, 'min_size': min_size, **metrics})
    
    # Sort by AJI
    results_sorted = sorted(results, key=lambda x: x['aji_mean'], reverse=True)
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print("\n🏆 TOP 5 CONFIGURATIONS:\n")
    print(f"{'Rank':<5} {'Beta':<6} {'MinSize':<8} {'AJI':<10} {'OverSeg':<10} {'N_Pred':<8} {'N_GT':<8}")
    print("-" * 80)
    
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"{i:<5} {r['beta']:<6.2f} {r['min_size']:<8} {r['aji_mean']:<10.4f} {r['over_seg_ratio']:<10.2f} {r['n_pred']:<8.1f} {r['n_gt']:<8.1f}")
    
    best = results_sorted[0]
    print(f"\n🎯 BEST CONFIGURATION:")
    print(f"  Beta:            {best['beta']:.2f}")
    print(f"  Min Size:        {best['min_size']}")
    print(f"  AJI Mean:        {best['aji_mean']:.4f} ± {best['aji_std']:.4f}")
    print(f"  AJI Median:      {best['aji_median']:.4f}")
    print(f"  Over-seg Ratio:  {best['over_seg_ratio']:.2f}× (Pred {best['n_pred']:.1f} / GT {best['n_gt']:.1f})")
    
    baseline = [r for r in results if r['beta'] == 1.0 and r['min_size'] == 20][0]
    improvement = ((best['aji_mean'] - baseline['aji_mean']) / baseline['aji_mean']) * 100
    
    print(f"\n📊 IMPROVEMENT vs BASELINE (beta=1.0, min_size=20):")
    print(f"  Baseline AJI:    {baseline['aji_mean']:.4f}")
    print(f"  Optimized AJI:   {best['aji_mean']:.4f}")
    print(f"  Improvement:     {improvement:+.1f}%")
    
    # Save
    output_dir = Path("results/watershed_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"optimization_{args.family}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': results_sorted,
            'best': best,
            'baseline': baseline,
            'improvement_pct': improvement
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
