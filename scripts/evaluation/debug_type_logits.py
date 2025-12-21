#!/usr/bin/env python3
"""
Debug script to inspect raw type logits and probabilities.

This helps determine if the model is genuinely predicting the wrong class
or if there's a bug in the mapping/display logic.

Usage:
    python scripts/evaluation/debug_type_logits.py \
        --npz_file data/evaluation/pannuke_fold2_converted/image_00000.npz \
        --checkpoint_dir models/checkpoints_FIXED
"""
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily

CELL_TYPES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']


def main():
    parser = argparse.ArgumentParser(description="Debug type logits")
    parser.add_argument("--npz_file", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints_FIXED"))
    args = parser.parse_args()

    # Load GT
    data = np.load(args.npz_file)
    image = data['image']
    gt_type = data['type_map']

    print("=" * 70)
    print("DEBUG TYPE LOGITS & PROBABILITIES")
    print("=" * 70)

    # GT distribution
    print(f"\n📊 GT Type Distribution:")
    for i, name in enumerate(CELL_TYPES, 1):
        count = (gt_type == i).sum()
        pct = count / gt_type.size * 100
        print(f"  {i}. {name:15s}: {count:5d} pixels ({pct:5.2f}%)")

    gt_dominant = np.bincount(gt_type[gt_type > 0]).argmax()
    print(f"\n  🎯 GT Dominant Type: {gt_dominant} ({CELL_TYPES[gt_dominant-1]})")

    # Load model
    print(f"\n🤖 Loading model: {args.checkpoint_dir}")
    model = OptimusGateInferenceMultiFamily(
        checkpoint_dir=str(args.checkpoint_dir),
        device='cuda',
    )

    # Predict
    result = model.predict(image)

    # Extract type_probs
    if 'multifamily_result' in result:
        mf_result = result['multifamily_result']

        print(f"\n🔬 Type Probabilities Analysis:")
        print(f"  Shape: {mf_result.type_probs.shape}")  # (5, H, W) or (6, H, W)?

        # Average probability per class across whole image
        class_means = mf_result.type_probs.mean(axis=(1, 2))

        print(f"\n  Average Probability per Class (model output [0-{len(class_means)-1}]):")
        for i, (cls, mean) in enumerate(zip(CELL_TYPES, class_means)):
            bar = '█' * int(mean * 50)
            print(f"    {i}. {cls:15s}: {mean:.4f} {bar}")

        # Dominant class
        dominant_idx = class_means.argmax()
        print(f"\n  🎯 Predicted Dominant Class:")
        print(f"     Model indexing [0-4]: {dominant_idx} ({CELL_TYPES[dominant_idx]})")
        print(f"     PanNuke labels [1-5]: {dominant_idx+1} ({CELL_TYPES[dominant_idx]})")

        # Pixel-level distribution
        pred_type = mf_result.type_map
        print(f"\n📊 Predicted Type Distribution (from type_map):")
        print(f"  Unique values in type_map: {np.unique(pred_type[pred_type > 0])}")

        for i, name in enumerate(CELL_TYPES, 1):
            count = (pred_type == i).sum()
            pct = count / pred_type.size * 100
            if count > 0:
                print(f"  {i}. {name:15s}: {count:5d} pixels ({pct:5.2f}%)")

        # Comparison
        print(f"\n🔍 Comparison:")
        print(f"  GT Dominant:   {gt_dominant} ({CELL_TYPES[gt_dominant-1]})")
        print(f"  Pred Dominant: {dominant_idx+1} ({CELL_TYPES[dominant_idx]})")

        if dominant_idx + 1 == gt_dominant:
            print(f"\n  ✅ MATCH! Model predicts correct dominant class")
        else:
            print(f"\n  ❌ MISMATCH! Model predicts wrong dominant class")
            print(f"\n  🔴 This is a REAL classification error, not a display bug!")

        # Sample region analysis (center 10x10)
        h, w = pred_type.shape
        cy, cx = h // 2, w // 2
        sample = pred_type[cy-5:cy+5, cx-5:cx+5]
        print(f"\n🔬 Sample 10x10 region from center:")
        print(f"  Unique types: {np.unique(sample[sample > 0])}")
        print(f"  Sample:\n{sample}")

    else:
        print("❌ No multifamily_result found in prediction!")


if __name__ == "__main__":
    main()
