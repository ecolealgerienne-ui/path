#!/usr/bin/env python3
"""
Diagnostic script to visualize model predictions vs GT.

This script helps debug why detection metrics are low:
- Shows predicted instance map
- Shows GT instance map
- Compares predicted vs GT instance counts
- Visualizes overlaps

Usage:
    python scripts/evaluation/diagnose_predictions.py \
        --npz_file data/evaluation/pannuke_fold2_converted/image_00002.npz \
        --checkpoint_dir models/checkpoints
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily

PANNUKE_CLASSES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial"
}


def diagnose_prediction(
    npz_file: Path,
    checkpoint_dir: Path,
    output_dir: Path = None
) -> None:
    """Diagnose a single prediction."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSING PREDICTION")
    print(f"{'='*70}\n")

    # Load GT
    print(f"📥 Loading GT: {npz_file}")
    data = np.load(npz_file)
    image = data['image']
    gt_inst = data['inst_map']
    gt_type = data['type_map']

    gt_inst_ids = np.unique(gt_inst)
    gt_inst_ids = gt_inst_ids[gt_inst_ids > 0]

    print(f"   Image: {image.shape}, dtype={image.dtype}")
    print(f"   GT instances: {len(gt_inst_ids)}")

    # Count GT types
    gt_type_counts = {}
    for inst_id in gt_inst_ids:
        inst_mask = gt_inst == inst_id
        types_in_inst = gt_type[inst_mask]
        majority_type = np.bincount(types_in_inst).argmax()
        gt_type_counts[majority_type] = gt_type_counts.get(majority_type, 0) + 1

    print(f"\n   GT Type Distribution:")
    for type_id in sorted(gt_type_counts.keys()):
        if type_id > 0:
            class_name = PANNUKE_CLASSES.get(type_id, f"Unknown_{type_id}")
            print(f"     {type_id} ({class_name:12s}): {gt_type_counts[type_id]:3d} instances")

    # Load model
    print(f"\n🤖 Loading model...")
    model = OptimusGateInferenceMultiFamily(checkpoint_dir=str(checkpoint_dir))

    # Predict
    print(f"\n🔮 Running prediction...")
    result = model.predict(image)

    pred_inst = result['instance_map']
    pred_type_raw = result['nt_mask']

    # Convert nt_mask to type_map format
    pred_type = np.zeros_like(pred_inst, dtype=np.uint8)
    for inst_id in range(1, pred_inst.max() + 1):
        inst_mask = pred_inst == inst_id
        if inst_mask.sum() == 0:
            continue
        types_in_inst = pred_type_raw[inst_mask]
        types_valid = types_in_inst[types_in_inst >= 0]
        if len(types_valid) > 0:
            inst_type = int(np.bincount(types_valid).argmax())
            pred_type[inst_mask] = inst_type + 1

    # Resize predictions to match GT
    if pred_inst.shape != gt_inst.shape:
        import cv2
        h, w = gt_inst.shape
        pred_inst = cv2.resize(
            pred_inst.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        pred_type = cv2.resize(
            pred_type.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

    pred_inst_ids = np.unique(pred_inst)
    pred_inst_ids = pred_inst_ids[pred_inst_ids > 0]

    print(f"   Predicted instances: {len(pred_inst_ids)}")
    print(f"   Detected organ: {result['organ'].organ_name}")
    print(f"   Family: {result['family']}")

    # Count predicted types
    pred_type_counts = {}
    for inst_id in pred_inst_ids:
        inst_mask = pred_inst == inst_id
        types_in_inst = pred_type[inst_mask]
        majority_type = np.bincount(types_in_inst).argmax()
        pred_type_counts[majority_type] = pred_type_counts.get(majority_type, 0) + 1

    print(f"\n   Predicted Type Distribution:")
    for type_id in sorted(pred_type_counts.keys()):
        if type_id > 0:
            class_name = PANNUKE_CLASSES.get(type_id, f"Unknown_{type_id}")
            print(f"     {type_id} ({class_name:12s}): {pred_type_counts[type_id]:3d} instances")

    # Visualize
    print(f"\n📊 Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original, GT inst, Pred inst
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_inst, cmap='nipy_spectral')
    axes[0, 1].set_title(f'GT Instances ({len(gt_inst_ids)})')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_inst, cmap='nipy_spectral')
    axes[0, 2].set_title(f'Predicted Instances ({len(pred_inst_ids)})')
    axes[0, 2].axis('off')

    # Row 2: GT type, Pred type, Overlay
    axes[1, 0].imshow(gt_type, cmap='tab10', vmin=0, vmax=5)
    axes[1, 0].set_title('GT Types')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_type, cmap='tab10', vmin=0, vmax=5)
    axes[1, 1].set_title('Predicted Types')
    axes[1, 1].axis('off')

    # Overlay: GT in red, Pred in green, overlap in yellow
    overlay = np.zeros((*gt_inst.shape, 3), dtype=np.uint8)
    overlay[gt_inst > 0] = [255, 0, 0]  # GT in red
    overlay[pred_inst > 0, 1] = 255  # Pred in green (overlap becomes yellow)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Overlay (Red=GT, Green=Pred, Yellow=Overlap)')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{npz_file.stem}_diagnosis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved: {output_file}")
    else:
        plt.show()

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose model predictions")
    parser.add_argument("--npz_file", type=Path, required=True, help="Path to .npz file")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints"), help="Checkpoint directory")
    parser.add_argument("--output_dir", type=Path, default=Path("results/diagnosis"), help="Output directory for visualizations")

    args = parser.parse_args()

    diagnose_prediction(args.npz_file, args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
