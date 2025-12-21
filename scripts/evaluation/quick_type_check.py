#!/usr/bin/env python3
"""
Quick check of type values from diagnostic results.
"""
import numpy as np
from pathlib import Path

# Load GT
npz_file = Path("data/evaluation/pannuke_fold2_converted/image_00000.npz")
data = np.load(npz_file)
gt_type = data['type_map']

print("=" * 70)
print("TYPE VALUES CHECK")
print("=" * 70)

print(f"\nGT Type Map:")
print(f"  Shape: {gt_type.shape}")
print(f"  Dtype: {gt_type.dtype}")
print(f"  Unique values: {np.unique(gt_type)}")

for i in range(1, 6):
    count = (gt_type == i).sum()
    pct = count / gt_type.size * 100
    print(f"  Type {i}: {count:5d} pixels ({pct:5.2f}%)")

# Load the diagnostic image predictions from the saved result
# The diagnostic script should have saved intermediate results
result_file = Path("results/diagnostic_gt/debug_result.npz")

if result_file.exists():
    result_data = np.load(result_file)
    pred_type = result_data['pred_type']

    print(f"\nPred Type Map:")
    print(f"  Shape: {pred_type.shape}")
    print(f"  Dtype: {pred_type.dtype}")
    print(f"  Unique values: {np.unique(pred_type)}")

    for i in range(1, 6):
        count = (pred_type == i).sum()
        pct = count / pred_type.size * 100
        print(f"  Type {i}: {count:5d} pixels ({pct:5.2f}%)")

    # Check for systematic offset
    print(f"\nChecking for systematic offset...")
    diff = pred_type.astype(int) - gt_type.astype(int)
    diff_nonzero = diff[gt_type > 0]
    unique_diffs, counts = np.unique(diff_nonzero, return_counts=True)

    print(f"  Difference distribution (Pred - GT):")
    for d, c in zip(unique_diffs, counts):
        pct = c / diff_nonzero.size * 100
        print(f("    Δ{d:+2d}: {c:5d} pixels ({pct:5.2f}%)")

    if len(unique_diffs) == 1:
        offset = unique_diffs[0]
        print(f"\n  ✅ SYSTEMATIC OFFSET DETECTED: Pred = GT + {offset}")
    else:
        print(f"\n  ⚠️ No single systematic offset")
else:
    print("\n⚠️ No saved debug result found. Run diagnose_gt_failure.py first with --save_debug flag.")
