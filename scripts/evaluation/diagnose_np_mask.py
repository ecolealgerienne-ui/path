#!/usr/bin/env python3
"""
Diagnostic: Pourquoi le NP mask prédit est si petit (4.8%)?

Compare la prédiction NP avec le Ground Truth pour identifier le problème.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily


def diagnose_np_mask(npz_file: Path, checkpoint_dir: Path, output_dir: Path):
    """Compare NP prediction vs GT."""

    print("=" * 70)
    print("DIAGNOSTIC NP MASK")
    print("=" * 70)

    # Load GT
    data = np.load(npz_file)
    image = data['image']
    inst_map_gt = data['inst_map']
    type_map_gt = data['type_map']

    # Create GT binary mask (union of all instances)
    np_mask_gt = (inst_map_gt > 0).astype(np.float32)

    print(f"\n📥 Ground Truth:")
    print(f"   Image shape: {image.shape}")
    print(f"   GT NP mask coverage: {np_mask_gt.sum() / np_mask_gt.size * 100:.2f}%")
    print(f"   GT instances: {len(np.unique(inst_map_gt)) - 1}")

    # Load model and predict
    print(f"\n🤖 Loading model...")
    model = OptimusGateInferenceMultiFamily(checkpoint_dir=str(checkpoint_dir), device='cuda')

    print(f"🔮 Running prediction...")
    predictions, probabilities = model.predict(image)

    # Get raw NP prediction
    print(f"\n📊 Model predictions:")
    print(f"   Organ: {predictions['organ_name']}")
    print(f"   Family: {predictions.get('family', 'N/A')}")

    # Extract raw NP probabilities
    # We need to run inference manually to get raw NP output
    tensor = model.preprocess_image(image)
    features = model.extract_features(tensor)
    cls_token = features[:, 0, :]
    patch_tokens = features[:, 1:257, :]

    # Get organ and family
    pred_idx, probs = model.model.organ_head.predict(cls_token)
    organ_idx = pred_idx[0].item()
    organ_name = model.model.organ_head.organ_names[organ_idx]
    from src.inference.optimus_gate_inference_multifamily import get_family
    family = get_family(organ_name)

    # Get HoVer-Net raw output
    hovernet = model.model.hovernet_decoders[family]
    with torch.no_grad():
        np_logits, hv_pred, nt_logits = hovernet(patch_tokens)

    # Apply sigmoid to get probabilities
    np_pred = torch.sigmoid(np_logits).cpu().numpy()[0, 0]  # (224, 224)

    print(f"\n📊 NP Prediction (raw):")
    print(f"   Shape: {np_pred.shape}")
    print(f"   Range: [{np_pred.min():.3f}, {np_pred.max():.3f}]")
    print(f"   Mean: {np_pred.mean():.3f}")
    print(f"   Median: {np.median(np_pred):.3f}")

    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n🎯 Coverage at different thresholds:")
    for thresh in thresholds:
        mask = (np_pred > thresh).astype(np.float32)
        coverage = mask.sum() / mask.size * 100
        print(f"   Threshold {thresh:.1f}: {coverage:6.2f}% ({int(mask.sum())} pixels)")

    # Resize GT to 224x224 for fair comparison
    import cv2
    np_mask_gt_resized = cv2.resize(np_mask_gt, (224, 224), interpolation=cv2.INTER_NEAREST)
    gt_coverage_224 = np_mask_gt_resized.sum() / np_mask_gt_resized.size * 100

    print(f"\n📏 GT Coverage after resize to 224x224: {gt_coverage_224:.2f}%")

    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Row 1: Original, GT 256x256, GT 224x224
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np_mask_gt, cmap='gray')
    axes[0, 1].set_title(f"GT NP Mask (256x256)\nCoverage: {np_mask_gt.sum() / np_mask_gt.size * 100:.2f}%")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np_mask_gt_resized, cmap='gray')
    axes[0, 2].set_title(f"GT NP Mask (224x224)\nCoverage: {gt_coverage_224:.2f}%")
    axes[0, 2].axis('off')

    # Row 2: NP raw prediction + thresholds
    axes[1, 0].imshow(np_pred, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f"NP Prediction (raw)\nRange: [{np_pred.min():.3f}, {np_pred.max():.3f}]")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np_pred > 0.3, cmap='gray')
    coverage_03 = (np_pred > 0.3).sum() / np_pred.size * 100
    axes[1, 1].set_title(f"NP > 0.3\nCoverage: {coverage_03:.2f}%")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np_pred > 0.5, cmap='gray')
    coverage_05 = (np_pred > 0.5).sum() / np_pred.size * 100
    axes[1, 2].set_title(f"NP > 0.5 (current)\nCoverage: {coverage_05:.2f}%")
    axes[1, 2].axis('off')

    # Row 3: More thresholds + comparison
    axes[2, 0].imshow(np_pred > 0.7, cmap='gray')
    coverage_07 = (np_pred > 0.7).sum() / np_pred.size * 100
    axes[2, 0].set_title(f"NP > 0.7\nCoverage: {coverage_07:.2f}%")
    axes[2, 0].axis('off')

    # Overlay comparison: GT vs Pred (0.5)
    overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    overlay[np_mask_gt_resized > 0, 0] = 255  # GT in red
    overlay[np_pred > 0.5, 1] = 255  # Pred in green (overlap = yellow)
    axes[2, 1].imshow(overlay)
    axes[2, 1].set_title("Overlay: Red=GT, Green=Pred(0.5)")
    axes[2, 1].axis('off')

    # Histogram of NP probabilities
    axes[2, 2].hist(np_pred.flatten(), bins=50, color='blue', alpha=0.7)
    axes[2, 2].axvline(0.3, color='green', linestyle='--', label='0.3')
    axes[2, 2].axvline(0.5, color='orange', linestyle='--', label='0.5')
    axes[2, 2].axvline(0.7, color='red', linestyle='--', label='0.7')
    axes[2, 2].set_xlabel('NP Probability')
    axes[2, 2].set_ylabel('Pixel Count')
    axes[2, 2].set_title('Distribution of NP Probabilities')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{npz_file.stem}_np_diagnostic.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")

    # Calculate IoU at different thresholds
    print(f"\n🎯 IoU (Intersection over Union) with GT:")
    for thresh in thresholds:
        pred_mask = (np_pred > thresh).astype(bool)
        gt_mask = (np_mask_gt_resized > 0).astype(bool)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / (union + 1e-10)

        print(f"   Threshold {thresh:.1f}: IoU = {iou:.4f}")

    # Final diagnosis
    print(f"\n" + "=" * 70)
    print("🔍 DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Check if GT coverage is also low
    if gt_coverage_224 < 10:
        print("⚠️  GT coverage is ALSO low (<10%)")
        print("    → This image might have very few cells")
        print("    → OR connectedComponents is creating small disconnected regions")
    else:
        print(f"✅ GT coverage is normal ({gt_coverage_224:.2f}%)")

    # Check if prediction is close to GT at any threshold
    best_thresh = None
    best_iou = 0
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_mask = (np_pred > thresh).astype(bool)
        gt_mask = (np_mask_gt_resized > 0).astype(bool)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / (union + 1e-10)
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh

    print(f"\n🏆 Best threshold: {best_thresh:.1f} (IoU: {best_iou:.4f})")

    if best_iou < 0.5:
        print("❌ PROBLEM: Even best threshold gives IoU < 0.5")
        print("    → Model predictions are POOR")
        print("    → Likely cause: Training data mismatch")
        print("    → Solution: Retrain with correct instance segmentation")
    elif best_thresh != 0.5:
        print(f"⚠️  Best threshold ({best_thresh}) ≠ current (0.5)")
        print(f"    → Adjusting threshold to {best_thresh} might improve results")
        print("    → BUT this is a band-aid, not a real fix")
    else:
        print("✅ Current threshold (0.5) is optimal")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Diagnose NP mask prediction")
    parser.add_argument("--npz_file", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/np_diagnostic"))

    args = parser.parse_args()
    diagnose_np_mask(args.npz_file, args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
