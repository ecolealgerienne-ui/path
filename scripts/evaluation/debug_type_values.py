#!/usr/bin/env python3
"""
Debug script to print exact type values from GT and Pred.
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily

# Load GT
npz_file = Path("data/evaluation/pannuke_fold2_converted/image_00000.npz")
data = np.load(npz_file)
image = data['image']
gt_type = data['type_map']

print("=" * 70)
print("DEBUG TYPE VALUES")
print("=" * 70)

print(f"\nGT Type Map:")
print(f"  Shape: {gt_type.shape}")
print(f"  Dtype: {gt_type.dtype}")
print(f"  Unique values: {np.unique(gt_type)}")

for i in range(1, 6):
    count = (gt_type == i).sum()
    pct = count / gt_type.size * 100
    print(f"  Type {i}: {count:5d} pixels ({pct:5.2f}%)")

# Load model
print(f"\nLoading model...")
model = OptimusGateInferenceMultiFamily(
    checkpoint_dir="models/checkpoints_FIXED",
    device='cuda',
)

# Predict
result = model.predict(image)

# Extract type_map
if 'multifamily_result' in result:
    pred_type = result['multifamily_result'].type_map
else:
    pred_type = result.get('type_map', result.get('nt_mask'))

print(f"\nPred Type Map:")
print(f"  Shape: {pred_type.shape}")
print(f"  Dtype: {pred_type.dtype}")
print(f"  Unique values: {np.unique(pred_type)}")

for i in range(1, 6):
    count = (pred_type == i).sum()
    pct = count / pred_type.size * 100
    print(f"  Type {i}: {count:5d} pixels ({pct:5.2f}%)")

# Find most common type in GT
gt_most_common = np.bincount(gt_type.flatten())[1:].argmax() + 1
pred_most_common = np.bincount(pred_type.flatten())[1:].argmax() + 1

print(f"\nMost common types:")
print(f"  GT:   Type {gt_most_common} ({(gt_type == gt_most_common).sum()} pixels)")
print(f"  Pred: Type {pred_most_common} ({(pred_type == pred_most_common).sum()} pixels)")

# Sample 10x10 region from center
h, w = gt_type.shape
center_y, center_x = h // 2, w // 2
sample_size = 10

print(f"\nSample 10x10 region from center ({center_y-5}:{center_y+5}, {center_x-5}:{center_x+5}):")
print(f"\nGT Type Values:")
print(gt_type[center_y-5:center_y+5, center_x-5:center_x+5])

print(f"\nPred Type Values:")
print(pred_type[center_y-5:center_y+5, center_x-5:center_x+5])

# Check if there's a systematic offset
print(f"\nChecking for systematic offset...")
diff = pred_type.astype(int) - gt_type.astype(int)
diff_nonzero = diff[gt_type > 0]
unique_diffs, counts = np.unique(diff_nonzero, return_counts=True)

print(f"  Difference distribution (Pred - GT):")
for d, c in zip(unique_diffs, counts):
    pct = c / diff_nonzero.size * 100
    print(f"    Δ{d:+2d}: {c:5d} pixels ({pct:5.2f}%)")

if len(unique_diffs) == 1:
    offset = unique_diffs[0]
    print(f"\n  ✅ SYSTEMATIC OFFSET DETECTED: Pred = GT + {offset}")
    print(f"  → All predictions are shifted by {offset} compared to GT")
else:
    print(f"\n  ⚠️ No single systematic offset - predictions vary")
