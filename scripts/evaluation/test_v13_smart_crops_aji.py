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
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from typing import Dict
import json
from datetime import datetime
from tqdm import tqdm

from src.models.hovernet_decoder import HoVerNetDecoder
from src.metrics.ground_truth_metrics import compute_aji  # Centralized AJI


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


def hv_guided_watershed(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    np_threshold: float = 0.45,  # Optimized via grid search (2025-12-28)
    beta: float = 0.5,   # Optimal: centres nets mais gradients HV bruités
    min_size: int = 50,  # Optimized via grid search - filtre bruit
    min_distance: int = 5  # Optimized via grid search - évite sur-segmentation
) -> np.ndarray:
    """
    HV-guided watershed for instance segmentation.

    Uses HV magnitude to suppress markers at cell boundaries where
    HV gradients are strong.

    Args:
        np_pred: Nuclear presence probability map (H, W) in [0, 1]
        hv_pred: HV maps (2, H, W) in [-1, 1]
        np_threshold: Threshold for NP binarization (default: 0.45, grid-search optimized)
        beta: HV magnitude exponent (default: 0.5, optimal pour centres nets/gradients bruités)
        min_size: Minimum instance size in pixels (default: 50, grid-search optimized)
        min_distance: Minimum distance between peaks (default: 5, grid-search optimized)

    Returns:
        Instance map (H, W) with instance IDs starting from 1
    """
    # Threshold NP to get binary mask (lowered threshold = larger masks = better IoU)
    np_binary = (np_pred > np_threshold).astype(np.uint8)

    if np_binary.sum() == 0:
        return np.zeros_like(np_pred, dtype=np.int32)

    # Distance transform
    dist = distance_transform_edt(np_binary)

    # HV magnitude (range [0, sqrt(2)])
    hv_h = hv_pred[0]
    hv_v = hv_pred[1]
    hv_magnitude = np.sqrt(hv_h**2 + hv_v**2)

    # HV-guided marker energy
    # Higher HV magnitude → lower marker energy → suppress markers at boundaries
    marker_energy = dist * (1 - hv_magnitude ** beta)

    # Find local maxima as markers
    from skimage.feature import peak_local_max
    markers_coords = peak_local_max(
        marker_energy,
        min_distance=min_distance,  # Use parameter (default: 3 for dense nuclei)
        threshold_abs=0.1,
        exclude_border=False
    )

    # Create markers map
    markers = np.zeros_like(np_binary, dtype=np.int32)
    for i, (y, x) in enumerate(markers_coords, start=1):
        markers[y, x] = i

    # If no markers found, return empty
    if markers.max() == 0:
        return np.zeros_like(np_pred, dtype=np.int32)

    # Label markers
    markers = label(markers)[0]

    # Watershed (use distance as elevation map)
    instances = watershed(-dist, markers, mask=np_binary)

    # Remove small objects
    instances = remove_small_objects(instances, min_size=min_size)

    # Relabel to ensure consecutive IDs
    instances = label(instances)[0]

    return instances.astype(np.int32)


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
        "--n_samples",
        type=int,
        default=50,
        help="Number of validation samples to evaluate"
    )
    parser.add_argument(
        "--np_threshold",
        type=float,
        default=0.35,
        help="NP binarization threshold (default: 0.35 for larger masks)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="HV magnitude exponent (default: 0.5 for dense Epidermal tissues)"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=40,
        help="Minimum instance size (default: 40 optimized)"
    )
    parser.add_argument(
        "--min_distance",
        type=int,
        default=3,
        help="Minimum distance between peak markers (default: 3 for dense nuclei)"
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
    args = parser.parse_args()

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

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=n_classes,
        dropout=0.1,
        use_hybrid=args.use_hybrid,
        use_fpn_chimique=args.use_fpn_chimique
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

    val_data_path = Path(f"data/family_data_v13_smart_crops/{args.family}_val_v13_smart_crops.npz")
    if not val_data_path.exists():
        print(f"❌ ERROR: {val_data_path} not found")
        print(f"\nRun first:")
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
    features_path = Path(f"data/cache/family_data/{args.family}_rgb_features_v13_smart_crops_val.npz")
    if not features_path.exists():
        print(f"❌ ERROR: {features_path} not found")
        print(f"\nRun first:")
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

        all_dice.append(dice)
        all_aji.append(aji)
        all_aji_fair.append(aji_fair)
        all_pq.append(pq_metrics['PQ'])
        n_pred_instances.append(len(np.unique(pred_inst)) - 1)
        n_gt_instances.append(len(np.unique(gt_inst)) - 1)

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

    results_dir = Path("results/v13_smart_crops")
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_hybrid" if args.use_hybrid else ""
    results_file = results_dir / f"{args.family}{suffix}_aji_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved: {results_file}")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
