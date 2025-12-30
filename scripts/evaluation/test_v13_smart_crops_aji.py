#!/usr/bin/env python3
"""
Evaluate V13 Smart Crops Model on AJI Metric.

This script evaluates the trained V13 Smart Crops model on validation samples
and computes instance segmentation metrics (AJI, PQ) to verify improvement:
- V13 POC baseline: AJI = 0.57 (measured on TRAIN data - invalidated)
- V13 Smart Crops target: AJI ≥ 0.68 (+18% improvement on VAL data)

The script uses HV-guided watershed for instance segmentation post-processing
with optimized parameters (beta=1.0 for softmax, min_size=40).

Usage:
    python scripts/evaluation/test_v13_smart_crops_aji.py \
        --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_best.pth \
        --family epidermal \
        --n_samples 50

Author: CellViT-Optimus Project
Date: 2025-12-27
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
from typing import Dict
import json
from datetime import datetime
from tqdm import tqdm

from src.models.hovernet_decoder import HoVerNetDecoder
from src.metrics.ground_truth_metrics import compute_aji  # Centralized AJI
from src.postprocessing import hv_guided_watershed  # Single source of truth


def compute_pq(pred_inst: np.ndarray, gt_inst: np.ndarray, iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) metric.

    PQ = Detection Quality (DQ) × Segmentation Quality (SQ)
    """
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]

    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0:
        return {"PQ": 1.0 if len(pred_ids) == 0 else 0.0, "DQ": 1.0, "SQ": 1.0}

    if len(pred_ids) == 0:
        return {"PQ": 0.0, "DQ": 0.0, "SQ": 0.0}

    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))

    for i, gt_id in enumerate(gt_ids):
        gt_mask = (gt_inst == gt_id)

        for j, pred_id in enumerate(pred_ids):
            pred_mask = (pred_inst == pred_id)

            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()

            if union > 0:
                iou_matrix[i, j] = intersection / union

    # Match GT and predicted instances
    TP = 0
    sum_iou = 0.0
    matched_gt = set()
    matched_pred = set()

    for i in range(len(gt_ids)):
        for j in range(len(pred_ids)):
            if iou_matrix[i, j] >= iou_threshold:
                if i not in matched_gt and j not in matched_pred:
                    TP += 1
                    sum_iou += iou_matrix[i, j]
                    matched_gt.add(i)
                    matched_pred.add(j)

    FP = len(pred_ids) - TP
    FN = len(gt_ids) - TP

    # Detection Quality
    if TP + 0.5 * FP + 0.5 * FN == 0:
        DQ = 0.0
    else:
        DQ = TP / (TP + 0.5 * FP + 0.5 * FN)

    # Segmentation Quality
    SQ = (sum_iou / TP) if TP > 0 else 0.0

    # Panoptic Quality
    PQ = DQ * SQ

    return {
        "PQ": PQ,
        "DQ": DQ,
        "SQ": SQ,
        "TP": TP,
        "FP": FP,
        "FN": FN
    }


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient for binary masks."""
    pred_binary = (pred > 0.5).astype(bool)
    gt_binary = (gt > 0.5).astype(bool)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()

    if union == 0:
        return 1.0

    return (2.0 * intersection / union)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate V13 Smart Crops model on AJI metric"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to trained V13 Smart Crops checkpoint"
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"],
        help="Family to evaluate"
    )
    parser.add_argument(
        "--organ",
        type=str,
        default=None,
        help="Specific organ (optional). If specified, loads {organ}_val.npz instead of {family}_val.npz"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of validation samples to evaluate"
    )
    # === WATERSHED PARAMETERS (Optimized via grid search 2025-12-29) ===
    # Respiratory optimal: np_threshold=0.40, min_size=30, min_distance=5
    # Epidermal optimal:   np_threshold=0.45, min_size=40, min_distance=5
    parser.add_argument(
        "--np_threshold",
        type=float,
        default=0.40,
        help="NP binarization threshold (Respiratory=0.40, Epidermal=0.45)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.50,
        help="HV magnitude exponent (optimal=0.50 for all families)"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=30,
        help="Minimum instance size in pixels (Respiratory=30, Epidermal=40)"
    )
    parser.add_argument(
        "--min_distance",
        type=int,
        default=5,
        help="Minimum distance between peak markers (optimal=5 for all families)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"]
    )
    parser.add_argument(
        "--use_hybrid",
        action="store_true",
        help="Use hybrid mode (RGB+H-channel injection)"
    )
    parser.add_argument(
        "--use_fpn_chimique",
        action="store_true",
        help="Use FPN Chimique (multi-scale H-injection at 5 levels)"
    )
    parser.add_argument(
        "--use_h_alpha",
        action="store_true",
        help="Use learnable alpha for H-channel amplification"
    )
    parser.add_argument(
        "--diagnose_outliers",
        action="store_true",
        help="Enable detailed outlier analysis for samples with low AJI"
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=0.50,
        help="AJI threshold below which samples are considered outliers (default: 0.50)"
    )
    args = parser.parse_args()

    # Préfixe pour les fichiers de données (organe ou famille)
    data_prefix = args.organ.lower() if args.organ else args.family

    device = torch.device(args.device)
    n_classes = 5

    print("=" * 80)
    print("EVALUATION V13 SMART CROPS - AJI METRIC")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Family: {args.family}")
    print(f"  N samples: {args.n_samples}")
    print(f"  NP threshold: {args.np_threshold}")
    print(f"  Watershed beta: {args.beta}")
    print(f"  Watershed min_size: {args.min_size}")
    print(f"  Watershed min_distance: {args.min_distance}")
    print(f"  Hybrid mode: {args.use_hybrid} (H-channel injection)")
    print(f"  FPN Chimique: {args.use_fpn_chimique} (multi-scale H @ 16,32,64,112,224)")
    print(f"  Device: {args.device}")
    if args.organ:
        print(f"  Organ filter: {args.organ}")
        print(f"  Data prefix: {data_prefix}")
    if args.diagnose_outliers:
        print(f"  Outlier diagnosis: ENABLED (threshold={args.outlier_threshold})")

    # Load model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    # Check checkpoint for hybrid/fpn mode consistency
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_use_hybrid = checkpoint.get('use_hybrid', False)
    checkpoint_use_fpn = checkpoint.get('use_fpn_chimique', False)

    if args.use_hybrid != checkpoint_use_hybrid:
        print(f"  ⚠️  WARNING: Checkpoint trained with use_hybrid={checkpoint_use_hybrid}")
        print(f"  ⚠️  But evaluation requested use_hybrid={args.use_hybrid}")
        print(f"  ⚠️  Using checkpoint setting: use_hybrid={checkpoint_use_hybrid}")
        args.use_hybrid = checkpoint_use_hybrid

    if args.use_fpn_chimique != checkpoint_use_fpn:
        print(f"  ⚠️  WARNING: Checkpoint trained with use_fpn_chimique={checkpoint_use_fpn}")
        print(f"  ⚠️  But evaluation requested use_fpn_chimique={args.use_fpn_chimique}")
        print(f"  ⚠️  Using checkpoint setting: use_fpn_chimique={checkpoint_use_fpn}")
        args.use_fpn_chimique = checkpoint_use_fpn

    # Detect use_h_alpha from checkpoint (check if h_alphas params exist)
    checkpoint_use_h_alpha = any('h_alphas' in k for k in checkpoint['model_state_dict'].keys())
    if checkpoint_use_h_alpha:
        print(f"  ✅ Checkpoint contains h_alphas parameters (use_h_alpha=True)")
        args.use_h_alpha = True

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=n_classes,
        dropout=0.1,
        use_hybrid=args.use_hybrid,
        use_fpn_chimique=args.use_fpn_chimique,
        use_h_alpha=args.use_h_alpha
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✅ Checkpoint loaded (epoch {checkpoint['epoch']})")
    print(f"  Best Dice: {checkpoint.get('best_dice', 'N/A')}")
    if args.use_hybrid:
        print(f"  ✅ Mode HYBRID activé: injection H-channel via RuifrokExtractor")
    if args.use_fpn_chimique:
        print(f"  ✅ Mode FPN CHIMIQUE activé: injection H multi-échelle (16→32→64→112→224)")

    # Load validation data (targets + inst_maps)
    print("\n" + "=" * 80)
    print("LOADING VALIDATION DATA")
    print("=" * 80)

    val_data_path = Path(f"data/family_data_v13_smart_crops/{data_prefix}_val_v13_smart_crops.npz")
    if not val_data_path.exists():
        print(f"❌ ERROR: {val_data_path} not found")
        print(f"\nRun first:")
        if args.organ:
            print(f"  python scripts/preprocessing/prepare_v13_smart_crops.py --family {args.family} --organ {args.organ}")
        else:
            print(f"  python scripts/preprocessing/prepare_v13_smart_crops.py --family {args.family}")
        return 1

    print(f"Loading {val_data_path.name}...")
    val_data = np.load(val_data_path)

    np_targets = val_data['np_targets']  # (N_val, 224, 224)
    hv_targets = val_data['hv_targets']  # (N_val, 2, 224, 224)
    inst_maps = val_data['inst_maps']  # ✅ (N_val, 224, 224) int32 - VRAIES instances!

    # Images RGB pour mode hybride (injection H-channel)
    if args.use_hybrid:
        if 'images' in val_data:
            all_images = val_data['images']  # (N_val, 224, 224, 3) uint8
            print(f"  ✅ Images RGB chargées: shape {all_images.shape}, dtype {all_images.dtype}")
        else:
            print(f"❌ ERROR: Mode hybride activé mais 'images' non trouvées dans {val_data_path}")
            print(f"  Régénérez les données avec prepare_v13_smart_crops.py")
            return 1
    else:
        all_images = None

    # Load PRE-EXTRACTED features (same as training!)
    features_path = Path(f"data/cache/family_data/{data_prefix}_rgb_features_v13_smart_crops_val.npz")
    if not features_path.exists():
        print(f"❌ ERROR: {features_path} not found")
        print(f"\nRun first:")
        if args.organ:
            print(f"  python scripts/preprocessing/extract_features_v13_smart_crops.py --family {args.family} --organ {args.organ} --split val")
        else:
            print(f"  python scripts/preprocessing/extract_features_v13_smart_crops.py --family {args.family} --split val")
        return 1

    print(f"Loading {features_path.name}...")
    features_data = np.load(features_path)
    all_features = features_data['features']  # (N_val, 261, 1536)

    n_total = len(all_features)
    n_to_eval = min(args.n_samples, n_total)

    print(f"  → {n_total} validation crops available")
    print(f"  → Evaluating first {n_to_eval} samples")
    print(f"  ✅ Using PRE-EXTRACTED features (same as training)")
    print(f"  ✅ Using TRUE instance maps (not watershed reconstruction)")

    # Use true instance maps from data
    gt_instances = inst_maps[:n_to_eval]

    print(f"\n✅ Data loaded: {len(gt_instances)} samples")

    # Evaluation
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    all_dice = []
    all_aji = []
    all_aji_fair = []  # Fair AJI using same watershed for both
    all_pq = []
    n_pred_instances = []
    n_gt_instances = []

    # Per-sample details for outlier analysis
    sample_details = []

    for i in tqdm(range(n_to_eval), desc="Evaluating"):
        gt_inst = gt_instances[i]

        # Use PRE-EXTRACTED features (same as training!)
        features = torch.from_numpy(all_features[i]).unsqueeze(0).float().to(device)  # (1, 261, 1536)

        # Image RGB pour mode hybride
        if args.use_hybrid:
            # Convertir image HWC uint8 → CHW float32 [0, 255]
            image = torch.from_numpy(all_images[i]).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            image = None

        # Forward through decoder
        with torch.no_grad():
            np_out, hv_out, nt_out = model(features, images_rgb=image)  # (1, 2, 224, 224), (1, 2, 224, 224), (1, 5, 224, 224)

            # Convert to numpy - USE SOFTMAX (not sigmoid!) for CrossEntropyLoss
            np_probs = torch.softmax(np_out, dim=1).cpu().numpy()[0]  # (2, 224, 224)
            np_pred = np_probs[1]  # Canal 1 = Noyaux (224, 224)
            hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

        # Watershed post-processing
        pred_inst = hv_guided_watershed(
            np_pred,
            hv_pred,
            np_threshold=args.np_threshold,
            beta=args.beta,
            min_size=args.min_size,
            min_distance=args.min_distance
        )

        # Metrics
        dice = compute_dice(np_pred, np_targets[i])
        aji = compute_aji(pred_inst, gt_inst)
        pq_metrics = compute_pq(pred_inst, gt_inst)

        # DEBUG: Compute "fair AJI" using same method for both
        # Reconstruct GT instances using SAME watershed method
        gt_hv = hv_targets[i]  # (2, 224, 224)
        gt_inst_watershed = hv_guided_watershed(
            np_targets[i],  # Use GT NP as binary mask
            gt_hv,          # Use GT HV for guidance
            np_threshold=0.5,  # GT is already binary
            beta=args.beta,
            min_size=args.min_size,
            min_distance=args.min_distance
        )
        aji_fair = compute_aji(pred_inst, gt_inst_watershed)

        # Count instances
        n_pred = len(np.unique(pred_inst)) - 1  # -1 for background
        n_gt = len(np.unique(gt_inst)) - 1

        # Compute HV magnitude on predicted nuclei regions
        hv_magnitude = np.sqrt(hv_pred[0]**2 + hv_pred[1]**2)
        nuclei_mask = np_pred > args.np_threshold
        hv_mag_mean = float(hv_magnitude[nuclei_mask].mean()) if nuclei_mask.sum() > 0 else 0.0

        # Nuclear coverage (% of image that is nuclei)
        np_coverage = float(nuclei_mask.sum()) / (224 * 224) * 100

        all_dice.append(dice)
        all_aji.append(aji)
        all_aji_fair.append(aji_fair)
        all_pq.append(pq_metrics['PQ'])
        n_pred_instances.append(n_pred)
        n_gt_instances.append(n_gt)

        # Store sample details for outlier analysis
        sample_details.append({
            'index': i,
            'aji': aji,
            'dice': dice,
            'pq': pq_metrics['PQ'],
            'n_gt': n_gt,
            'n_pred': n_pred,
            'hv_mag_mean': hv_mag_mean,
            'np_coverage': np_coverage,
            'tp': pq_metrics['TP'],
            'fp': pq_metrics['FP'],
            'fn': pq_metrics['FN']
        })

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    mean_aji = np.mean(all_aji)
    std_aji = np.std(all_aji)
    median_aji = np.median(all_aji)
    mean_aji_fair = np.mean(all_aji_fair)
    std_aji_fair = np.std(all_aji_fair)
    mean_pq = np.mean(all_pq)
    std_pq = np.std(all_pq)
    mean_n_pred = np.mean(n_pred_instances)
    mean_n_gt = np.mean(n_gt_instances)
    over_seg_ratio = mean_n_pred / mean_n_gt if mean_n_gt > 0 else 0.0

    print(f"\n📊 MÉTRIQUES GLOBALES (n={n_to_eval}):")
    print(f"\n  Dice:        {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"  AJI:         {mean_aji:.4f} ± {std_aji:.4f}")
    print(f"  AJI Median:  {median_aji:.4f}")
    print(f"  AJI FAIR:    {mean_aji_fair:.4f} ± {std_aji_fair:.4f}  ← Same watershed for GT & Pred")
    print(f"  PQ:          {mean_pq:.4f} ± {std_pq:.4f}")
    print(f"\n  Instances pred: {mean_n_pred:.1f}")
    print(f"  Instances GT:   {mean_n_gt:.1f}")
    print(f"  Over-seg ratio: {over_seg_ratio:.2f}×")

    # Diagnostic
    if mean_aji_fair > mean_aji + 0.05:
        print(f"\n⚠️  DIAGNOSTIC: AJI FAIR (+{mean_aji_fair - mean_aji:.2f}) > AJI standard")
        print(f"   → Le problème vient de la MÉTHODE de construction des GT instances")
        print(f"   → scipy.ndimage.label() produit des frontières différentes du watershed")
        print(f"   → Considérez utiliser watershed pour les GT aussi (cohérence)")

    # Outlier Analysis (if enabled)
    if args.diagnose_outliers:
        print("\n" + "=" * 80)
        print("🔍 OUTLIER ANALYSIS")
        print("=" * 80)

        # Find outliers (samples below threshold)
        outliers = [s for s in sample_details if s['aji'] < args.outlier_threshold]
        outliers_sorted = sorted(outliers, key=lambda x: x['aji'])

        print(f"\nThreshold: AJI < {args.outlier_threshold}")
        print(f"Outliers found: {len(outliers)} / {n_to_eval} samples ({len(outliers)/n_to_eval*100:.1f}%)")

        if outliers:
            # Compute impact on mean
            non_outlier_ajis = [s['aji'] for s in sample_details if s['aji'] >= args.outlier_threshold]
            mean_without_outliers = np.mean(non_outlier_ajis) if non_outlier_ajis else 0
            print(f"Mean AJI without outliers: {mean_without_outliers:.4f} (vs {mean_aji:.4f} with)")

            print(f"\n{'─'*80}")
            print(f"{'Idx':<5} {'AJI':<7} {'Dice':<6} {'GT':<5} {'Pred':<5} {'Δ':<6} {'Type':<12} {'HV_mag':<7} {'NP%':<6}")
            print(f"{'─'*80}")

            for s in outliers_sorted[:15]:  # Show top 15 worst
                delta = s['n_pred'] - s['n_gt']
                if delta < -2:
                    seg_type = "UNDER-SEG"
                elif delta > 2:
                    seg_type = "OVER-SEG"
                else:
                    seg_type = "boundary"

                print(f"{s['index']:<5} {s['aji']:<7.4f} {s['dice']:<6.3f} {s['n_gt']:<5} {s['n_pred']:<5} {delta:+<5} {seg_type:<12} {s['hv_mag_mean']:<7.3f} {s['np_coverage']:<6.1f}")

            print(f"{'─'*80}")

            # Analyze patterns
            under_seg = sum(1 for s in outliers if s['n_pred'] < s['n_gt'] - 2)
            over_seg = sum(1 for s in outliers if s['n_pred'] > s['n_gt'] + 2)
            boundary = len(outliers) - under_seg - over_seg

            print(f"\n📊 PATTERN ANALYSIS:")
            print(f"  Under-segmentation (pred << gt): {under_seg} ({under_seg/len(outliers)*100:.0f}%)")
            print(f"  Over-segmentation (pred >> gt):  {over_seg} ({over_seg/len(outliers)*100:.0f}%)")
            print(f"  Boundary issues (|Δ| ≤ 2):       {boundary} ({boundary/len(outliers)*100:.0f}%)")

            # HV magnitude analysis
            outlier_hv_mags = [s['hv_mag_mean'] for s in outliers]
            good_samples = [s for s in sample_details if s['aji'] >= 0.65]
            good_hv_mags = [s['hv_mag_mean'] for s in good_samples] if good_samples else [0]

            print(f"\n📊 HV MAGNITUDE COMPARISON:")
            print(f"  Outliers mean:     {np.mean(outlier_hv_mags):.4f}")
            print(f"  Good samples mean: {np.mean(good_hv_mags):.4f}")

            if np.mean(outlier_hv_mags) < np.mean(good_hv_mags) * 0.8:
                print(f"  ⚠️  Outliers have weak HV gradients → Watershed struggles to separate")

            # Nuclear density analysis
            outlier_coverage = [s['np_coverage'] for s in outliers]
            good_coverage = [s['np_coverage'] for s in good_samples] if good_samples else [0]

            print(f"\n📊 NUCLEAR DENSITY:")
            print(f"  Outliers mean coverage:     {np.mean(outlier_coverage):.1f}%")
            print(f"  Good samples mean coverage: {np.mean(good_coverage):.1f}%")

            if np.mean(outlier_coverage) > np.mean(good_coverage) * 1.3:
                print(f"  ⚠️  Outliers are dense patches → Consider reducing min_distance")

        else:
            print(f"\n✅ No outliers found! All samples have AJI ≥ {args.outlier_threshold}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    target_aji = 0.68
    if mean_aji >= target_aji:
        verdict = "✅ OBJECTIF ATTEINT"
        print(f"\n{verdict}")
        print(f"  AJI = {mean_aji:.4f} ≥ {target_aji} (+{((mean_aji / 0.57) - 1) * 100:.1f}% vs baseline 0.57)")
    else:
        verdict = "⚠️ OBJECTIF NON ATTEINT"
        gap = target_aji - mean_aji
        print(f"\n{verdict}")
        print(f"  AJI = {mean_aji:.4f} < {target_aji} (écart: -{gap:.4f})")
        print(f"  Progress: {(mean_aji / target_aji) * 100:.1f}% of target")

    if median_aji > 0.80:
        print(f"\n✅ Note positive: Médiane AJI = {median_aji:.4f} démontre excellente capacité")

    # Save results
    results = {
        "checkpoint": str(args.checkpoint),
        "family": args.family,
        "organ": args.organ,
        "n_samples": n_to_eval,
        "use_hybrid": args.use_hybrid,
        "watershed_params": {
            "np_threshold": args.np_threshold,
            "beta": args.beta,
            "min_size": args.min_size,
            "min_distance": args.min_distance
        },
        "metrics": {
            "dice_mean": float(mean_dice),
            "dice_std": float(std_dice),
            "aji_mean": float(mean_aji),
            "aji_std": float(std_aji),
            "aji_median": float(median_aji),
            "pq_mean": float(mean_pq),
            "pq_std": float(std_pq),
            "over_seg_ratio": float(over_seg_ratio),
            "n_pred_mean": float(mean_n_pred),
            "n_gt_mean": float(mean_n_gt)
        },
        "verdict": verdict,
        "target_aji": target_aji,
        "timestamp": datetime.now().isoformat()
    }

    # Add outlier analysis to results if enabled
    if args.diagnose_outliers:
        outliers = [s for s in sample_details if s['aji'] < args.outlier_threshold]
        results["outlier_analysis"] = {
            "threshold": args.outlier_threshold,
            "n_outliers": len(outliers),
            "outlier_indices": [s['index'] for s in outliers],
            "outliers": sorted(outliers, key=lambda x: x['aji']),
            "all_samples": sample_details
        }

    results_dir = Path("results/v13_smart_crops")
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_hybrid" if args.use_hybrid else ""
    results_file = results_dir / f"{data_prefix}{suffix}_aji_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved: {results_file}")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
