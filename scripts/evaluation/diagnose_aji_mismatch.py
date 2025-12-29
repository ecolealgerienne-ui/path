#!/usr/bin/env python3
"""
Diagnostic pour comprendre pourquoi AJI est faible malgré bon Dice et over-seg ratio.

Ce script visualise quelques samples pour identifier le problème:
1. GT instances vs Predicted instances
2. IoU distribution par instance
3. Boundaries alignment

Usage:
    python scripts/evaluation/diagnose_aji_mismatch.py --family epidermal --n_samples 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max
from tqdm import tqdm

from src.models.hovernet_decoder import HoVerNetDecoder


def compute_instance_ious(pred_inst, gt_inst):
    """Compute IoU for each GT instance with its best matching prediction."""
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]

    ious = []
    matched = []
    unmatched_gt = []

    for gt_id in gt_ids:
        gt_mask = gt_inst == gt_id

        best_iou = 0
        best_pred_id = None

        for pred_id in pred_ids:
            pred_mask = pred_inst == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()

            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_id

        if best_pred_id is not None and best_iou > 0:
            ious.append(best_iou)
            matched.append((gt_id, best_pred_id, best_iou))
        else:
            unmatched_gt.append(gt_id)

    return ious, matched, unmatched_gt


def hv_guided_watershed(np_pred, hv_pred, np_threshold=0.45, beta=0.5, min_size=50, min_distance=5):
    """Same as in evaluation script."""
    np_binary = (np_pred > np_threshold).astype(np.uint8)

    if np_binary.sum() == 0:
        return np.zeros_like(np_pred, dtype=np.int32)

    dist = distance_transform_edt(np_binary)

    hv_h = hv_pred[0]
    hv_v = hv_pred[1]
    hv_magnitude = np.sqrt(hv_h**2 + hv_v**2)

    marker_energy = dist * (1 - hv_magnitude ** beta)

    markers_coords = peak_local_max(
        marker_energy,
        min_distance=min_distance,
        threshold_abs=0.1,
        exclude_border=False
    )

    markers = np.zeros_like(np_binary, dtype=np.int32)
    for i, (y, x) in enumerate(markers_coords, start=1):
        markers[y, x] = i

    if markers.max() == 0:
        return np.zeros_like(np_pred, dtype=np.int32)

    markers = label(markers)[0]
    instances = watershed(-dist, markers, mask=np_binary)
    instances = remove_small_objects(instances, min_size=min_size)
    instances = label(instances)[0]

    return instances.astype(np.int32)


def visualize_sample(idx, gt_inst, pred_inst, np_pred, hv_pred, np_target, save_path):
    """Create diagnostic visualization for a single sample."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Basic comparison
    axes[0, 0].imshow(np_target, cmap='gray')
    axes[0, 0].set_title(f'NP Target (GT Binary)')

    axes[0, 1].imshow(np_pred, cmap='gray')
    axes[0, 1].set_title(f'NP Prediction (Probs)')

    axes[0, 2].imshow(gt_inst, cmap='nipy_spectral')
    axes[0, 2].set_title(f'GT Instances ({gt_inst.max()} nuclei)')

    axes[0, 3].imshow(pred_inst, cmap='nipy_spectral')
    axes[0, 3].set_title(f'Pred Instances ({pred_inst.max()} nuclei)')

    # Row 2: HV analysis
    hv_magnitude = np.sqrt(hv_pred[0]**2 + hv_pred[1]**2)

    axes[1, 0].imshow(hv_pred[0], cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 0].set_title('HV Horizontal')

    axes[1, 1].imshow(hv_pred[1], cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('HV Vertical')

    axes[1, 2].imshow(hv_magnitude, cmap='hot')
    axes[1, 2].set_title('HV Magnitude')

    # Overlay: GT boundaries on pred instances
    from skimage.segmentation import find_boundaries
    gt_boundaries = find_boundaries(gt_inst, mode='thick')
    pred_boundaries = find_boundaries(pred_inst, mode='thick')

    overlay = np.zeros((*gt_inst.shape, 3))
    overlay[gt_inst > 0] = [0.3, 0.3, 0.3]  # Gray for GT
    overlay[pred_inst > 0] = [0.5, 0.5, 0.5]  # Lighter gray for pred
    overlay[gt_boundaries] = [0, 1, 0]  # Green for GT boundaries
    overlay[pred_boundaries] = [1, 0, 0]  # Red for pred boundaries

    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title('Boundaries: GT(green) vs Pred(red)')

    # Compute IoU distribution
    ious, matched, unmatched = compute_instance_ious(pred_inst, gt_inst)

    plt.suptitle(f'Sample {idx}: Mean IoU = {np.mean(ious):.3f}, Matched = {len(matched)}/{gt_inst.max()}', fontsize=14)

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return ious, matched, unmatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_fpn_best.pth"))
    parser.add_argument("--family", default="epidermal")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--use_hybrid", action="store_true", default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path("results/aji_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AJI MISMATCH DIAGNOSTIC")
    print("=" * 80)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    use_fpn = checkpoint.get('use_fpn_chimique', False)
    use_hybrid = checkpoint.get('use_hybrid', True)

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"  use_hybrid: {use_hybrid}")
    print(f"  use_fpn_chimique: {use_fpn}")

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=5,
        dropout=0.1,
        use_hybrid=use_hybrid,
        use_fpn_chimique=use_fpn
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    val_data_path = Path(f"data/family_data_v13_smart_crops/{args.family}_val_v13_smart_crops.npz")
    val_data = np.load(val_data_path)

    np_targets = val_data['np_targets']
    inst_maps = val_data['inst_maps']
    all_images = val_data['images'] if 'images' in val_data else None

    features_path = Path(f"data/cache/family_data/{args.family}_rgb_features_v13_smart_crops_val.npz")
    features_data = np.load(features_path)
    all_features = features_data['features']

    print(f"\nLoaded {len(all_features)} validation samples")

    # Analyze samples
    all_ious = []

    for i in tqdm(range(min(args.n_samples, len(all_features))), desc="Diagnosing"):
        gt_inst = inst_maps[i]
        np_target = np_targets[i]

        features = torch.from_numpy(all_features[i]).unsqueeze(0).float().to(device)

        if use_hybrid and all_images is not None:
            image = torch.from_numpy(all_images[i]).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            image = None

        with torch.no_grad():
            np_out, hv_out, nt_out = model(features, images_rgb=image)
            np_probs = torch.softmax(np_out, dim=1).cpu().numpy()[0]
            np_pred = np_probs[1]
            hv_pred = hv_out.cpu().numpy()[0]

        pred_inst = hv_guided_watershed(np_pred, hv_pred)

        save_path = output_dir / f"sample_{i:03d}.png"
        ious, matched, unmatched = visualize_sample(i, gt_inst, pred_inst, np_pred, hv_pred, np_target, save_path)

        all_ious.extend(ious)

        print(f"\n  Sample {i}:")
        print(f"    GT instances: {gt_inst.max()}, Pred instances: {pred_inst.max()}")
        print(f"    Mean IoU: {np.mean(ious):.3f}, Min IoU: {np.min(ious):.3f}, Max IoU: {np.max(ious):.3f}")
        print(f"    Unmatched GT: {len(unmatched)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nIoU Distribution across all instances:")
    print(f"  Mean:   {np.mean(all_ious):.4f}")
    print(f"  Median: {np.median(all_ious):.4f}")
    print(f"  Std:    {np.std(all_ious):.4f}")
    print(f"  Min:    {np.min(all_ious):.4f}")
    print(f"  Max:    {np.max(all_ious):.4f}")

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_ious, bins=20, edgecolor='black')
    plt.xlabel('IoU per instance')
    plt.ylabel('Count')
    plt.title(f'IoU Distribution (n={len(all_ious)} instances)')
    plt.axvline(x=0.5, color='r', linestyle='--', label='IoU threshold for TP')
    plt.legend()
    plt.savefig(output_dir / 'iou_distribution.png', dpi=150)
    plt.close()

    # IoU bins analysis
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    hist, _ = np.histogram(all_ious, bins=bins)
    print(f"\nIoU bins:")
    for i in range(len(bins)-1):
        pct = hist[i] / len(all_ious) * 100
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {hist[i]:3d} ({pct:.1f}%)")

    print(f"\n✅ Visualizations saved to {output_dir}/")
    print(f"\n🔍 Key insight:")
    if np.mean(all_ious) < 0.6:
        print("   Low mean IoU indicates BOUNDARY MISALIGNMENT")
        print("   → Watershed parameters might need tuning")
        print("   → OR HV predictions have wrong gradients at boundaries")
    else:
        print("   IoU distribution looks OK, issue might be elsewhere")


if __name__ == "__main__":
    main()
