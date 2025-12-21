#!/usr/bin/env python3
"""
Diagnostic script to inspect PanNuke data structure.

This will help us understand why the type_map is not being extracted correctly.
"""

import numpy as np
import sys
from pathlib import Path

def inspect_pannuke_structure(masks_file: str):
    """Inspect the structure of PanNuke masks."""

    print("="*70)
    print("INSPECTING PANNUKE DATA STRUCTURE")
    print("="*70)

    # Load masks
    masks = np.load(masks_file, mmap_mode='r')
    print(f"\n📦 Masks array:")
    print(f"   Shape: {masks.shape}")
    print(f"   Dtype: {masks.dtype}")
    print(f"   Format: (N_images, Height, Width, 6)")

    # Examine first few images
    for img_idx in [0, 1, 2]:
        print(f"\n" + "="*70)
        print(f"IMAGE {img_idx}")
        print("="*70)

        mask = masks[img_idx]

        # Check each channel
        print("\nChannel Analysis:")
        for ch in range(6):
            channel = mask[:, :, ch]
            unique_vals = np.unique(channel)
            n_unique = len(unique_vals)
            max_val = channel.max()

            channel_name = [
                "Neoplastic",
                "Inflammatory",
                "Connective",
                "Dead",
                "Epithelial",
                "Instance Map"
            ][ch]

            print(f"  Channel {ch} ({channel_name}):")
            print(f"    Unique values: {n_unique}")
            print(f"    Max value: {max_val}")

            if n_unique <= 20:
                print(f"    Values: {unique_vals}")

            # Count instances (values > 0)
            if max_val > 0:
                n_instances = n_unique - 1  # Exclude background (0)
                print(f"    Instances: {n_instances}")

        # Try to understand the relationship between channels
        print("\n🔍 Type Mapping Analysis:")

        # Method 1: Channel 5 as global instance map
        inst_map_global = mask[:, :, 5]
        inst_ids_global = np.unique(inst_map_global)
        inst_ids_global = inst_ids_global[inst_ids_global > 0]

        print(f"  Global instance map (ch 5): {len(inst_ids_global)} instances")

        # For each instance in global map, find which class it belongs to
        if len(inst_ids_global) > 0:
            print("\n  Instance → Class mapping:")
            for inst_id in inst_ids_global[:5]:  # First 5 instances
                inst_mask = inst_map_global == inst_id

                # Check which class channels contain this instance
                found_in = []
                for class_id in range(5):
                    class_channel = mask[:, :, class_id]
                    if (class_channel[inst_mask] > 0).any():
                        found_in.append(class_id)

                class_names = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
                found_names = [class_names[c] for c in found_in]

                print(f"    Instance {inst_id}: found in channels {found_in} ({', '.join(found_names)})")

        # Method 2: Check if class channels contain instance IDs
        print("\n  Alternative: Class channels as separate instance maps?")
        for class_id in range(5):
            class_channel = mask[:, :, class_id]
            class_inst_ids = np.unique(class_channel)
            class_inst_ids = class_inst_ids[class_inst_ids > 0]

            class_names = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
            print(f"    {class_names[class_id]}: {len(class_inst_ids)} instances")

def main():
    if len(sys.argv) > 1:
        masks_file = sys.argv[1]
    else:
        masks_file = "/home/amar/data/PanNuke/fold2/masks.npy"

    if not Path(masks_file).exists():
        print(f"❌ Error: File not found: {masks_file}")
        print(f"\nUsage: python {sys.argv[0]} <path/to/masks.npy>")
        sys.exit(1)

    inspect_pannuke_structure(masks_file)

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nBased on this analysis, we need to determine:")
    print("1. Is channel 5 a global instance map?")
    print("2. Or are channels 0-4 separate instance maps per class?")
    print("3. How to correctly build a type_map?")

if __name__ == "__main__":
    main()
