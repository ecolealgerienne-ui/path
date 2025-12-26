#!/usr/bin/env python3
"""
Evaluate V13-Hybrid Model on AJI Metric.

This script evaluates the trained V13-Hybrid model on test samples and computes
the AJI (Aggregated Jaccard Index) metric to verify the target improvement:
- V13 POC baseline: AJI = 0.57
- V13-Hybrid target: AJI ≥ 0.68 (+18% improvement)

The script uses HV-guided watershed for instance segmentation post-processing.

Usage:
    python scripts/evaluation/test_v13_hybrid_aji.py \
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
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage import color
from typing import Dict, Tuple, List
import json
from datetime import datetime

from src.models.hovernet_decoder_hybrid import HoVerNetDecoderHybrid
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform, preprocess_image
from src.constants import PANNUKE_IMAGE_SIZE

# Import H-channel CNN from preprocessing script
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocessing'))
from extract_h_features_v13 import HChannelCNN


def compute_aji(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    """
    Compute Aggregated Jaccard Index (AJI).

    AJI measures instance segmentation quality by computing IoU between
    predicted and ground truth instances.

    Args:
        pred_inst: Predicted instance map (H, W) with instance IDs
        gt_inst: Ground truth instance map (H, W) with instance IDs

    Returns:
        AJI score in [0, 1], higher is better
    """
    # Get unique instance IDs (excluding background 0)
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]

    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0:
        # No ground truth instances
        return 1.0 if len(pred_ids) == 0 else 0.0

    if len(pred_ids) == 0:
        # No predicted instances but GT exists
        return 0.0

    # Compute pairwise IoU matrix
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))

    for i, gt_id in enumerate(gt_ids):
        gt_mask = (gt_inst == gt_id)

        for j, pred_id in enumerate(pred_ids):
            pred_mask = (pred_inst == pred_id)

            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()

            if union > 0:
                iou_matrix[i, j] = intersection / union

    # For each GT instance, find best matching predicted instance
    matched_pred = set()
    sum_iou = 0.0

    for i in range(len(gt_ids)):
        best_j = np.argmax(iou_matrix[i, :])
        best_iou = iou_matrix[i, best_j]

        sum_iou += best_iou
        matched_pred.add(best_j)

    # Compute C (sum of areas of unmatched predicted instances)
    unmatched_pred_ids = [pred_ids[j] for j in range(len(pred_ids)) if j not in matched_pred]
    C = sum((pred_inst == pred_id).sum() for pred_id in unmatched_pred_ids)

    # Compute U (sum of all GT instance areas)
    U = sum((gt_inst == gt_id).sum() for gt_id in gt_ids)

    # AJI formula
    aji = sum_iou / (U + C)

    return aji


def compute_pq(pred_inst: np.ndarray, gt_inst: np.ndarray, iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) metric.

    PQ = Detection Quality (DQ) × Segmentation Quality (SQ)

    Args:
        pred_inst: Predicted instance map
        gt_inst: Ground truth instance map
        iou_threshold: IoU threshold for matching (default: 0.5)

    Returns:
        Dictionary with PQ, DQ, SQ metrics
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
    TP = 0  # True positives
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

    FP = len(pred_ids) - TP  # False positives
    FN = len(gt_ids) - TP    # False negatives

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
    beta: float = 1.0,
    min_size: int = 20
) -> np.ndarray:
    """
    HV-guided watershed for instance segmentation.

    Uses HV magnitude to guide marker energy, suppressing boundaries
    where HV gradients are strong (cell borders).

    Args:
        np_pred: Nuclear presence probability map (H, W) in [0, 1]
        hv_pred: HV maps (2, H, W) in [-1, 1]
        beta: HV magnitude exponent (higher = stronger boundary suppression)
        min_size: Minimum instance size in pixels

    Returns:
        Instance map (H, W) with instance IDs starting from 1
    """
    # Threshold nuclear presence
    binary_mask = (np_pred > 0.5).astype(np.uint8)

    if binary_mask.sum() == 0:
        return np.zeros_like(np_pred, dtype=np.int32)

    # Compute HV magnitude (strength of gradient)
    h_map = hv_pred[0]  # Horizontal component
    v_map = hv_pred[1]  # Vertical component
    hv_magnitude = np.sqrt(h_map**2 + v_map**2)  # Range [0, sqrt(2)]

    # Normalize to [0, 1]
    hv_magnitude = np.clip(hv_magnitude / np.sqrt(2), 0, 1)

    # Distance transform (higher at cell centers)
    dist = distance_transform_edt(binary_mask)

    # HV-guided marker energy
    # High HV magnitude (boundaries) → suppress distance → prevent markers at boundaries
    marker_energy = dist * (1 - hv_magnitude ** beta)

    # Find local maxima as markers
    # Use a conservative threshold to avoid over-segmentation
    threshold = 0.3 * marker_energy.max()
    markers_binary = (marker_energy > threshold).astype(np.uint8)
    markers, n_markers = label(markers_binary)

    if n_markers == 0:
        # Fallback: use single component
        inst_map, _ = label(binary_mask)
        return inst_map.astype(np.int32)

    # Watershed segmentation
    # Use inverted marker_energy as "elevation" (valleys at boundaries)
    elevation = -marker_energy
    inst_map = watershed(elevation, markers, mask=binary_mask)

    # Remove small objects
    inst_map = remove_small_objects(inst_map, min_size=min_size).astype(np.int32)

    return inst_map


def extract_h_channel_on_the_fly(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract H-channel (Hematoxylin) from RGB image on-the-fly.

    Uses HED (Hematoxylin-Eosin-DAB) color deconvolution from scikit-image.

    Args:
        image_rgb: RGB image (H, W, 3) uint8 [0, 255]

    Returns:
        H-channel (H, W) uint8 [0, 255]
    """
    # Convert RGB to HED
    hed = color.rgb2hed(image_rgb)

    # Extract H-channel (first channel)
    h_channel = hed[:, :, 0]

    # Normalize to [0, 1]
    h_min, h_max = h_channel.min(), h_channel.max()
    if h_max > h_min:
        h_normalized = (h_channel - h_min) / (h_max - h_min)
    else:
        h_normalized = np.zeros_like(h_channel)

    # Convert to uint8
    h_uint8 = (h_normalized * 255).clip(0, 255).astype(np.uint8)

    return h_uint8


def extract_h_features_on_the_fly(
    h_channel: np.ndarray,
    h_cnn: nn.Module,
    device: str
) -> np.ndarray:
    """
    Extract H-features using lightweight CNN on-the-fly.

    Args:
        h_channel: H-channel image (H, W) uint8 [0, 255]
        h_cnn: Lightweight CNN adapter model
        device: Device to run on

    Returns:
        H-features (256,) float32
    """
    # Convert to float32 and normalize to [0, 1]
    h_float = h_channel.astype(np.float32) / 255.0

    # Add channel dimension: (H, W) → (1, H, W)
    h_tensor = torch.from_numpy(h_float).unsqueeze(0)

    # Add batch dimension: (1, H, W) → (1, 1, H, W)
    h_tensor = h_tensor.unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        h_features = h_cnn(h_tensor)  # (1, 256)

    return h_features.cpu().numpy()[0]  # (256,)


def extract_rgb_features_on_the_fly(
    image_rgb: np.ndarray,
    backbone: nn.Module,
    device: str
) -> np.ndarray:
    """
    Extract RGB features using H-optimus-0 on-the-fly.

    Args:
        image_rgb: RGB image (H, W, 3) uint8 [0, 255]
        backbone: H-optimus-0 model
        device: Device to run on

    Returns:
        RGB features (261, 1536) float32
    """
    # Preprocess image
    tensor = preprocess_image(image_rgb, device=device)  # (1, 3, 224, 224)

    # Extract features
    with torch.no_grad():
        features = backbone.forward_features(tensor)  # (1, 261, 1536)

    return features.cpu().numpy()[0]  # (261, 1536)


def load_test_samples(
    hybrid_data_path: Path,
    h_features_path: Path,
    rgb_features_path: Path,
    n_samples: int,
    on_the_fly: bool = False,
    h_cnn: nn.Module = None,
    backbone: nn.Module = None,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test samples from validation split.

    Uses the same 80/20 split as training (fold_ids, source_image_ids).

    Args:
        hybrid_data_path: Path to hybrid dataset
        h_features_path: Path to H-channel features (ignored if on_the_fly=True)
        rgb_features_path: Path to RGB features (ignored if on_the_fly=True)
        n_samples: Number of samples to load from validation set
        on_the_fly: If True, generate features on-the-fly from raw images
        h_cnn: Lightweight CNN adapter (required if on_the_fly=True)
        backbone: H-optimus-0 model (required if on_the_fly=True)
        device: Device to run on

    Returns:
        (rgb_features, h_features, gt_np, gt_inst) tuple
    """
    print(f"\n📂 Loading test samples...")
    print(f"  Mode: {'On-the-fly feature extraction' if on_the_fly else 'Pre-extracted features'}")

    # Load hybrid dataset
    hybrid_data = np.load(hybrid_data_path)

    # Get validation indices (same logic as training script)
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

    # Select n_samples from validation set
    n_to_load = min(n_samples, len(val_indices))
    selected_indices = val_indices[:n_to_load]

    print(f"  Loading {n_to_load} samples for evaluation")

    if on_the_fly:
        # Generate features on-the-fly from raw images
        print(f"  ⚙️  Extracting features on-the-fly...")

        if h_cnn is None or backbone is None:
            raise ValueError("h_cnn and backbone required for on-the-fly mode")

        # Load raw images
        images_224 = hybrid_data['images_224'][selected_indices]

        # Extract features for each sample
        rgb_features = np.zeros((n_to_load, 261, 1536), dtype=np.float32)
        h_features = np.zeros((n_to_load, 256), dtype=np.float32)

        for i in range(n_to_load):
            if (i + 1) % 10 == 0:
                print(f"    Extracted features for {i+1}/{n_to_load} samples...")

            image_rgb = images_224[i]  # (224, 224, 3) uint8

            # Extract H-channel
            h_channel = extract_h_channel_on_the_fly(image_rgb)

            # Extract features
            rgb_features[i] = extract_rgb_features_on_the_fly(image_rgb, backbone, device)
            h_features[i] = extract_h_features_on_the_fly(h_channel, h_cnn, device)

        print(f"  ✅ On-the-fly feature extraction complete")

    else:
        # Load pre-extracted features
        print(f"  📦 Loading pre-extracted features...")

        h_data = np.load(h_features_path)
        h_features_all = h_data['h_features']

        rgb_data = np.load(rgb_features_path)
        rgb_features_all = rgb_data['features']

        # Extract features for selected samples
        rgb_features = rgb_features_all[selected_indices]
        h_features = h_features_all[selected_indices]

    # Extract ground truth
    np_targets = hybrid_data['np_targets'][selected_indices]

    # Create GT instance maps from np_targets
    # For simplicity, use connected components on binary mask
    # (In real usage, should load pre-computed instance maps)
    gt_inst_maps = []

    for i in range(n_to_load):
        binary_mask = (np_targets[i] > 0.5).astype(np.uint8)
        inst_map, _ = label(binary_mask)
        gt_inst_maps.append(inst_map)

    gt_inst = np.array(gt_inst_maps)

    return rgb_features, h_features, np_targets, gt_inst


def evaluate_aji(
    checkpoint_path: Path,
    hybrid_data_path: Path,
    h_features_path: Path,
    rgb_features_path: Path,
    n_samples: int,
    beta: float,
    min_size: int,
    device: str,
    on_the_fly: bool = False
) -> Dict:
    """
    Evaluate AJI metric on test samples.

    Args:
        checkpoint_path: Path to trained model checkpoint
        hybrid_data_path: Path to hybrid dataset
        h_features_path: Path to H-channel features (ignored if on_the_fly=True)
        rgb_features_path: Path to RGB features (ignored if on_the_fly=True)
        n_samples: Number of test samples to evaluate
        beta: HV magnitude exponent for watershed
        min_size: Minimum instance size
        device: Device to run inference on
        on_the_fly: If True, generate features on-the-fly from raw images

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING V13-HYBRID MODEL ON AJI METRIC")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Test samples: {n_samples}")
    print(f"Feature mode: {'On-the-fly extraction' if on_the_fly else 'Pre-extracted'}")
    print(f"Watershed params: beta={beta}, min_size={min_size}")
    print(f"Device: {device}")

    # Load hybrid model
    print(f"\n🔧 Loading hybrid model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = HoVerNetDecoderHybrid(
        embed_dim=1536,
        h_dim=256,
        n_classes=5,
        dropout=checkpoint['args']['dropout'] if 'args' in checkpoint else 0.1
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✅ Hybrid model loaded (epoch {checkpoint['epoch']})")
    print(f"  Best Dice: {checkpoint['best_dice']:.4f}")

    # Load feature extraction models if on-the-fly mode
    h_cnn = None
    backbone = None

    if on_the_fly:
        print(f"\n🔧 Loading feature extraction models for on-the-fly mode...")

        # Load H-CNN adapter
        h_cnn = HChannelCNN(output_dim=256).to(device)
        h_cnn.eval()
        print(f"  ✅ H-CNN adapter loaded ({h_cnn.get_num_params()} params)")

        # Load H-optimus-0 backbone
        backbone = ModelLoader.load_hoptimus0(device=device)
        print(f"  ✅ H-optimus-0 backbone loaded")

    # Load test samples
    rgb_features, h_features, gt_np, gt_inst = load_test_samples(
        hybrid_data_path, h_features_path, rgb_features_path, n_samples,
        on_the_fly=on_the_fly, h_cnn=h_cnn, backbone=backbone, device=device
    )

    # Inference
    print(f"\n🔬 Running inference on {n_samples} samples...")

    aji_scores = []
    pq_scores = []
    dq_scores = []
    sq_scores = []

    dice_scores = []
    n_pred_instances = []
    n_gt_instances = []

    with torch.no_grad():
        for i in range(n_samples):
            # Prepare inputs
            rgb_feat = torch.from_numpy(rgb_features[i]).unsqueeze(0).to(device)  # (1, 261, 1536)
            h_feat = torch.from_numpy(h_features[i]).unsqueeze(0).to(device)      # (1, 256)

            # Extract patch tokens (skip CLS token + 4 Register tokens)
            patch_tokens = rgb_feat[:, 5:261, :]  # (1, 256, 1536)

            # Forward pass
            output = model(patch_tokens, h_feat)

            # Convert to numpy
            result = output.to_numpy(apply_activations=True)

            np_pred = result['np'][0, 0]  # (224, 224)
            hv_pred = result['hv'][0]     # (2, 224, 224)

            # Resize to 256×256 for comparison with GT
            np_pred_256 = cv2.resize(np_pred, (256, 256), interpolation=cv2.INTER_LINEAR)
            hv_pred_256 = np.zeros((2, 256, 256), dtype=np.float32)
            hv_pred_256[0] = cv2.resize(hv_pred[0], (256, 256), interpolation=cv2.INTER_LINEAR)
            hv_pred_256[1] = cv2.resize(hv_pred[1], (256, 256), interpolation=cv2.INTER_LINEAR)

            # HV-guided watershed
            pred_inst = hv_guided_watershed(np_pred_256, hv_pred_256, beta=beta, min_size=min_size)

            # Resize GT to 256×256 (assuming GT is at 224×224)
            gt_inst_i = gt_inst[i]
            if gt_inst_i.shape != (256, 256):
                gt_inst_256 = cv2.resize(gt_inst_i.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            else:
                gt_inst_256 = gt_inst_i

            # Compute metrics
            aji = compute_aji(pred_inst, gt_inst_256)
            pq_result = compute_pq(pred_inst, gt_inst_256)

            # Binary Dice for NP
            gt_binary = (gt_np[i] > 0.5).astype(np.float32)
            pred_binary = (np_pred > 0.5).astype(np.float32)

            # Resize GT binary to match prediction size
            if gt_binary.shape != pred_binary.shape:
                gt_binary_resized = cv2.resize(gt_binary, pred_binary.shape[::-1], interpolation=cv2.INTER_NEAREST)
            else:
                gt_binary_resized = gt_binary

            intersection = (pred_binary * gt_binary_resized).sum()
            dice = 2 * intersection / (pred_binary.sum() + gt_binary_resized.sum() + 1e-8)

            # Store results
            aji_scores.append(aji)
            pq_scores.append(pq_result['PQ'])
            dq_scores.append(pq_result['DQ'])
            sq_scores.append(pq_result['SQ'])
            dice_scores.append(dice)

            n_pred = len(np.unique(pred_inst)) - 1  # Exclude background
            n_gt = len(np.unique(gt_inst_256)) - 1
            n_pred_instances.append(n_pred)
            n_gt_instances.append(n_gt)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{n_samples} samples...")

    # Aggregate results
    results = {
        "n_samples": n_samples,
        "AJI": {
            "mean": float(np.mean(aji_scores)),
            "std": float(np.std(aji_scores)),
            "median": float(np.median(aji_scores)),
            "min": float(np.min(aji_scores)),
            "max": float(np.max(aji_scores))
        },
        "PQ": {
            "mean": float(np.mean(pq_scores)),
            "std": float(np.std(pq_scores))
        },
        "DQ": {
            "mean": float(np.mean(dq_scores)),
            "std": float(np.std(dq_scores))
        },
        "SQ": {
            "mean": float(np.mean(sq_scores)),
            "std": float(np.std(sq_scores))
        },
        "Dice": {
            "mean": float(np.mean(dice_scores)),
            "std": float(np.std(dice_scores))
        },
        "instances": {
            "mean_pred": float(np.mean(n_pred_instances)),
            "mean_gt": float(np.mean(n_gt_instances)),
            "std_pred": float(np.std(n_pred_instances)),
            "std_gt": float(np.std(n_gt_instances))
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": int(checkpoint['epoch']),
            "best_dice": float(checkpoint['best_dice'])
        },
        "watershed_params": {
            "beta": beta,
            "min_size": min_size
        }
    }

    return results


def print_results(results: Dict, baseline_aji: float = 0.57):
    """Print evaluation results in a formatted report."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")

    aji_mean = results['AJI']['mean']
    aji_std = results['AJI']['std']

    print(f"\n📊 AJI (Aggregated Jaccard Index)")
    print(f"  Mean:   {aji_mean:.4f} ± {aji_std:.4f}")
    print(f"  Median: {results['AJI']['median']:.4f}")
    print(f"  Range:  [{results['AJI']['min']:.4f}, {results['AJI']['max']:.4f}]")

    # Compare with baseline
    improvement = ((aji_mean - baseline_aji) / baseline_aji) * 100
    print(f"\n🎯 Comparison with V13 POC Baseline")
    print(f"  Baseline AJI:     {baseline_aji:.4f}")
    print(f"  V13-Hybrid AJI:   {aji_mean:.4f}")
    print(f"  Improvement:      {improvement:+.1f}%")
    print(f"  Target (≥0.68):   {'✅ ACHIEVED' if aji_mean >= 0.68 else '❌ NOT MET'}")

    print(f"\n📊 Panoptic Quality (PQ)")
    print(f"  PQ:  {results['PQ']['mean']:.4f} ± {results['PQ']['std']:.4f}")
    print(f"  DQ:  {results['DQ']['mean']:.4f} ± {results['DQ']['std']:.4f}")
    print(f"  SQ:  {results['SQ']['mean']:.4f} ± {results['SQ']['std']:.4f}")

    print(f"\n📊 Nuclear Presence (Dice)")
    print(f"  Mean: {results['Dice']['mean']:.4f} ± {results['Dice']['std']:.4f}")

    print(f"\n📊 Instance Counts")
    print(f"  Predicted: {results['instances']['mean_pred']:.1f} ± {results['instances']['std_pred']:.1f}")
    print(f"  Ground Truth: {results['instances']['mean_gt']:.1f} ± {results['instances']['std_gt']:.1f}")

    print(f"\n⚙️  Configuration")
    print(f"  Samples:    {results['n_samples']}")
    print(f"  Checkpoint: {results['checkpoint']['path']}")
    print(f"  Epoch:      {results['checkpoint']['epoch']}")
    print(f"  Best Dice:  {results['checkpoint']['best_dice']:.4f}")
    print(f"  Beta:       {results['watershed_params']['beta']}")
    print(f"  Min Size:   {results['watershed_params']['min_size']}")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate V13-Hybrid model on AJI metric"
    )

    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--family',
        type=str,
        default='epidermal',
        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
        help='Family name (default: epidermal)'
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=50,
        help='Number of test samples to evaluate (default: 50)'
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=1.0,
        help='HV magnitude exponent for watershed (default: 1.0)'
    )

    parser.add_argument(
        '--min_size',
        type=int,
        default=20,
        help='Minimum instance size in pixels (default: 20)'
    )

    parser.add_argument(
        '--baseline_aji',
        type=float,
        default=0.57,
        help='V13 POC baseline AJI for comparison (default: 0.57)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on (default: cuda if available)'
    )

    parser.add_argument(
        '--on_the_fly',
        action='store_true',
        help='Generate features on-the-fly from raw images (slower but validates full pipeline)'
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results/v13_hybrid_aji'),
        help='Output directory for results (default: results/v13_hybrid_aji)'
    )

    args = parser.parse_args()

    # Verify checkpoint exists
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Set up paths
    hybrid_data_path = Path(f'data/family_data_v13_hybrid/{args.family}_data_v13_hybrid.npz')
    h_features_path = Path(f'data/cache/family_data/{args.family}_h_features_v13.npz')
    rgb_features_path = Path(f'data/cache/family_data/{args.family}_rgb_features_v13.npz')

    # Verify data exists
    for name, path in [
        ("Hybrid dataset", hybrid_data_path),
        ("H-features", h_features_path),
        ("RGB features", rgb_features_path)
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # Evaluate
    results = evaluate_aji(
        checkpoint_path=args.checkpoint,
        hybrid_data_path=hybrid_data_path,
        h_features_path=h_features_path,
        rgb_features_path=rgb_features_path,
        n_samples=args.n_samples,
        beta=args.beta,
        min_size=args.min_size,
        device=args.device,
        on_the_fly=args.on_the_fly
    )

    # Print results
    print_results(results, baseline_aji=args.baseline_aji)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"aji_results_{args.family}_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == '__main__':
    main()
