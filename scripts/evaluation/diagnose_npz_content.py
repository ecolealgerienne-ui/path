#!/usr/bin/env python3
"""
Diagnostic script to inspect .npz file contents.

This script checks:
1. Image dtype (should be uint8)
2. Type_map distribution (should have all 5 classes, not just one)
3. Instance_map integrity

Usage:
    python scripts/evaluation/diagnose_npz_content.py \
        --npz_file data/evaluation/pannuke_fold2_converted/image_00000.npz
"""

import argparse
import numpy as np
from pathlib import Path

PANNUKE_CLASSES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial"
}


def diagnose_npz(npz_file: Path) -> None:
    """Diagnose a single .npz file."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {npz_file.name}")
    print(f"{'='*70}\n")

    data = np.load(npz_file)

    # Check keys
    print("📦 Keys in .npz:")
    for key in data.keys():
        print(f"   - {key}: {data[key].shape}, dtype={data[key].dtype}")

    # Check image
    if 'image' in data:
        image = data['image']
        print(f"\n🖼️  Image:")
        print(f"   Shape: {image.shape}")
        print(f"   Dtype: {image.dtype}")
        print(f"   Range: [{image.min()}, {image.max()}]")

        if image.dtype != np.uint8:
            print(f"   ❌ ERROR: Image is {image.dtype}, expected uint8!")
        else:
            print(f"   ✅ Image is uint8")

    # Check inst_map
    inst_map = data['inst_map']
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    print(f"\n🔢 Instance Map:")
    print(f"   Shape: {inst_map.shape}")
    print(f"   Dtype: {inst_map.dtype}")
    print(f"   Num instances: {len(inst_ids)}")
    print(f"   Instance IDs: {inst_ids[:10]}{'...' if len(inst_ids) > 10 else ''}")

    # Check type_map
    type_map = data['type_map']
    type_values = np.unique(type_map)

    print(f"\n🏷️  Type Map:")
    print(f"   Shape: {type_map.shape}")
    print(f"   Dtype: {type_map.dtype}")
    print(f"   Unique values: {type_values}")

    # Count pixels per class
    print(f"\n📊 Type Distribution (pixels):")
    for type_id in range(6):
        count = np.sum(type_map == type_id)
        class_name = PANNUKE_CLASSES.get(type_id, f"Unknown_{type_id}")
        print(f"   {type_id} ({class_name:12s}): {count:6d} pixels")

    # Count instances per class
    print(f"\n🧬 Type Distribution (instances):")
    type_counts = {}
    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        types_in_inst = type_map[inst_mask]
        # Majority vote
        majority_type = np.bincount(types_in_inst).argmax()
        type_counts[majority_type] = type_counts.get(majority_type, 0) + 1

    for type_id in range(6):
        count = type_counts.get(type_id, 0)
        class_name = PANNUKE_CLASSES.get(type_id, f"Unknown_{type_id}")
        print(f"   {type_id} ({class_name:12s}): {count:3d} instances")

    # Check for problems
    print(f"\n⚠️  Issues:")
    issues = []

    if 'image' in data and data['image'].dtype != np.uint8:
        issues.append("Image is not uint8 (normalization will be incorrect)")

    non_zero_types = type_values[type_values > 0]
    if len(non_zero_types) == 1:
        issues.append(f"Only ONE cell type found: {PANNUKE_CLASSES[non_zero_types[0]]}")

    if len(non_zero_types) == 0:
        issues.append("NO cell types found (all background)")

    if len(issues) == 0:
        print("   ✅ No issues detected")
    else:
        for issue in issues:
            print(f"   ❌ {issue}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose .npz file contents")
    parser.add_argument("--npz_file", type=Path, help="Path to .npz file")
    parser.add_argument("--npz_dir", type=Path, help="Directory with .npz files (diagnose first 5)")
    parser.add_argument("--num_files", type=int, default=5, help="Number of files to diagnose if using --npz_dir")

    args = parser.parse_args()

    if args.npz_file:
        diagnose_npz(args.npz_file)
    elif args.npz_dir:
        npz_files = sorted(args.npz_dir.glob("*.npz"))[:args.num_files]
        print(f"\n🔍 Diagnosing {len(npz_files)} files from {args.npz_dir}")
        for npz_file in npz_files:
            diagnose_npz(npz_file)
    else:
        parser.print_help()
        print("\n❌ Error: Provide --npz_file or --npz_dir")


if __name__ == "__main__":
    main()
