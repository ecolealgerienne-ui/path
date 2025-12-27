#!/usr/bin/env python3
"""
Evaluate V13 Smart Crops Model on AJI Metric.

This script evaluates the trained V13 Smart Crops model on validation samples
and computes instance segmentation metrics (AJI, PQ) to verify improvement:
- V13 POC baseline: AJI = 0.57 (measured on TRAIN data - invalidated)
- V13 Smart Crops target: AJI ≥ 0.68 (+18% improvement on VAL data)

The script uses HV-guided watershed for instance segmentation post-processing
with optimized parameters (beta=1.50, min_size=40).

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
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform
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
    beta: float = 1.50,  # Optimized value from Phase 5a
    min_size: int = 40   # Optimized value from Phase 5a
) -> np.ndarray:
    """
    HV-guided watershed for instance segmentation.

    Uses HV magnitude to suppress markers at cell boundaries where
    HV gradients are strong.

    Args:
        np_pred: Nuclear presence probability map (H, W) in [0, 1]
        hv_pred: HV maps (2, H, W) in [-1, 1]
        beta: HV magnitude exponent (higher = stronger boundary suppression)
              Optimized value: 1.50 (from grid search)
        min_size: Minimum instance size in pixels
                  Optimized value: 40 (from grid search)

    Returns:
        Instance map (H, W) with instance IDs starting from 1
    """
    # Threshold NP to get binary mask
    np_binary = (np_pred > 0.5).astype(np.uint8)

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
        min_distance=5,
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
        "--beta",
        type=float,
        default=1.50,
        help="HV magnitude exponent (default: 1.50 optimized)"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=40,
        help="Minimum instance size (default: 40 optimized)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"]
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
    print(f"  Watershed beta: {args.beta}")
    print(f"  Watershed min_size: {args.min_size}")
    print(f"  Device: {args.device}")

    # Load model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=n_classes,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✅ Checkpoint loaded (epoch {checkpoint['epoch']})")
    print(f"  Best Dice: {checkpoint.get('best_dice', 'N/A')}")

    # Load H-optimus-0 backbone
    print("\nLoading H-optimus-0 backbone...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)  # Pass string, not torch.device
    backbone.eval()
    print("  ✅ Backbone loaded")

    # Load validation data
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

    images = val_data['images']  # (N_val, 224, 224, 3)
    np_targets = val_data['np_targets']  # (N_val, 224, 224)
    hv_targets = val_data['hv_targets']  # (N_val, 2, 224, 224)
    source_image_ids = val_data['source_image_ids']

    n_total = len(images)
    n_to_eval = min(args.n_samples, n_total)

    print(f"  → {n_total} validation crops available")
    print(f"  → Evaluating first {n_to_eval} samples")

    # Create ground truth instance maps
    print("\nCreating GT instance maps from targets...")
    gt_instances = []

    for i in tqdm(range(n_to_eval), desc="GT instances"):
        np_gt = np_targets[i]
        hv_gt = hv_targets[i]

        # Use watershed on GT to create instance map
        gt_inst = hv_guided_watershed(
            np_gt,
            hv_gt,
            beta=args.beta,
            min_size=args.min_size
        )
        gt_instances.append(gt_inst)

    # Evaluation
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    transform = create_hoptimus_transform()

    all_dice = []
    all_aji = []
    all_pq = []
    n_pred_instances = []
    n_gt_instances = []

    for i in tqdm(range(n_to_eval), desc="Evaluating"):
        img = images[i]  # (224, 224, 3) uint8
        gt_inst = gt_instances[i]

        # Preprocess
        tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        # Extract features
        with torch.no_grad():
            features = backbone.forward_features(tensor)  # (1, 261, 1536)

            # Forward through decoder
            np_out, hv_out, nt_out = model(features)  # (1, 2, 224, 224), (1, 2, 224, 224), (1, 5, 224, 224)

            # Convert to numpy
            np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 1]  # (224, 224)
            hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

        # Watershed post-processing
        pred_inst = hv_guided_watershed(
            np_pred,
            hv_pred,
            beta=args.beta,
            min_size=args.min_size
        )

        # Metrics
        dice = compute_dice(np_pred, np_targets[i])
        aji = compute_aji(pred_inst, gt_inst)
        pq_metrics = compute_pq(pred_inst, gt_inst)

        all_dice.append(dice)
        all_aji.append(aji)
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
    mean_pq = np.mean(all_pq)
    std_pq = np.std(all_pq)
    mean_n_pred = np.mean(n_pred_instances)
    mean_n_gt = np.mean(n_gt_instances)
    over_seg_ratio = mean_n_pred / mean_n_gt if mean_n_gt > 0 else 0.0

    print(f"\n📊 MÉTRIQUES GLOBALES (n={n_to_eval}):")
    print(f"\n  Dice:        {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"  AJI:         {mean_aji:.4f} ± {std_aji:.4f}")
    print(f"  AJI Median:  {median_aji:.4f}")
    print(f"  PQ:          {mean_pq:.4f} ± {std_pq:.4f}")
    print(f"\n  Instances pred: {mean_n_pred:.1f}")
    print(f"  Instances GT:   {mean_n_gt:.1f}")
    print(f"  Over-seg ratio: {over_seg_ratio:.2f}×")

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
        "watershed_params": {
            "beta": args.beta,
            "min_size": args.min_size
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

    results_file = results_dir / f"{args.family}_aji_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved: {results_file}")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
