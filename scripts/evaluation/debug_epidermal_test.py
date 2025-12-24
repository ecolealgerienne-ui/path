#!/usr/bin/env python3
"""
Debug script to understand why AJI=0.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load data
# ⚠️ FIX GHOST PATH BUG: Chercher UN SEUL endroit (source de vérité)
# AVANT: Cherchait dans data/cache/family_data/ (ancien cache, peut être corrompu)
# APRÈS: Cherche UNIQUEMENT dans data/family_FIXED/ (dernière version v4)
data_file = Path("data/family_FIXED/epidermal_data_FIXED.npz")
data = np.load(data_file)

print("=" * 80)
print("DEBUG EPIDERMAL DATA")
print("=" * 80)

print("\nKeys in .npz file:")
for key in data.keys():
    value = data[key]
    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    if key in ['fold_ids', 'image_ids']:
        print(f"    → unique values: {np.unique(value)}")
        print(f"    → min={value.min()}, max={value.max()}")

print("\nFirst 10 samples:")
if 'fold_ids' in data and 'image_ids' in data:
    fold_ids = data['fold_ids']
    image_ids = data['image_ids']
    for i in range(min(10, len(fold_ids))):
        print(f"  Sample {i}: fold={fold_ids[i]}, img_id={image_ids[i]}")
else:
    print("  ❌ fold_ids or image_ids not found!")

# Test loading a GT mask
print("\nTesting GT mask loading:")
pannuke_dir = Path("/home/amar/data/PanNuke")

if 'fold_ids' in data and 'image_ids' in data:
    fold_id = int(fold_ids[0])
    img_id = int(image_ids[0])

    print(f"  Loading fold{fold_id}/masks.npy, index {img_id}...")

    try:
        masks = np.load(pannuke_dir / f"fold{fold_id}" / "masks.npy", mmap_mode='r')
        print(f"  ✅ Masks loaded: shape={masks.shape}")

        gt_mask = masks[img_id]
        print(f"  ✅ GT mask: shape={gt_mask.shape}, dtype={gt_mask.dtype}")
        print(f"     → Unique values in channels:")
        for c in range(gt_mask.shape[2]):
            unique = np.unique(gt_mask[:, :, c])
            print(f"       Channel {c}: {len(unique)} unique values (max={unique.max()})")

    except Exception as e:
        print(f"  ❌ Error: {e}")
else:
    print("  ❌ Cannot test - fold_ids/image_ids missing")

print("\n" + "=" * 80)
