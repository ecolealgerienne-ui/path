#!/usr/bin/env python3
"""
Inspect raw PanNuke data structure to understand how instances are stored.

This script examines the raw masks.npy to understand:
1. What each channel contains
2. How instances are encoded
3. How to properly extract instance maps

Usage:
    python scripts/evaluation/inspect_pannuke_raw.py \
        --masks_file /home/amar/data/PanNuke/fold2/masks.npy \
        --image_index 0
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


def inspect_mask(masks_file: Path, image_index: int) -> None:
    """Inspect a single mask from PanNuke."""
    print(f"\n{'='*70}")
    print(f"INSPECTING PANNUKE RAW DATA")
    print(f"{'='*70}\n")

    # Load masks
    print(f"📥 Loading: {masks_file}")
    masks = np.load(masks_file, mmap_mode='r')
    print(f"   Shape: {masks.shape}")
    print(f"   Dtype: {masks.dtype}")

    # Get single mask
    mask = masks[image_index]
    print(f"\n🎯 Analyzing image index {image_index}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask dtype: {mask.dtype}")

    # Analyze each channel
    print(f"\n📊 CHANNEL ANALYSIS:")
    print(f"{'='*70}")

    for channel in range(6):
        channel_data = mask[:, :, channel]
        unique_values = np.unique(channel_data)
        num_nonzero = np.sum(channel_data > 0)

        print(f"\nChannel {channel}:")
        print(f"  Unique values: {unique_values[:20]}{'...' if len(unique_values) > 20 else ''}")
        print(f"  Num unique: {len(unique_values)}")
        print(f"  Non-zero pixels: {num_nonzero}")
        print(f"  Range: [{channel_data.min()}, {channel_data.max()}]")
        print(f"  Dtype: {channel_data.dtype}")

        # If values look like instance IDs (many unique values > 1)
        if len(unique_values) > 2 and channel_data.max() > 1:
            print(f"  ⚠️  This looks like an INSTANCE MAP (many unique IDs)")
        elif len(unique_values) == 2 and set(unique_values) == {0, 1}:
            print(f"  ℹ️  This looks like a BINARY MASK")
        elif len(unique_values) <= 2:
            print(f"  ℹ️  This looks like a SEMANTIC MASK (few classes)")

    # Special analysis for channel 5
    print(f"\n{'='*70}")
    print(f"🔍 DETAILED ANALYSIS: CHANNEL 5 (Instance Map)")
    print(f"{'='*70}")

    inst_channel = mask[:, :, 5]
    inst_ids = np.unique(inst_channel)
    inst_ids = inst_ids[inst_ids > 0]

    print(f"\nInstance IDs in channel 5:")
    print(f"  Count: {len(inst_ids)}")
    print(f"  IDs: {inst_ids[:30]}{'...' if len(inst_ids) > 30 else ''}")

    # Count pixels per instance
    if len(inst_ids) > 0:
        print(f"\nPixels per instance (first 10):")
        for inst_id in inst_ids[:10]:
            count = np.sum(inst_channel == inst_id)
            print(f"  Instance {int(inst_id):3d}: {count:5d} pixels")

    # Compare with union of class channels
    print(f"\n{'='*70}")
    print(f"🔍 COMPARISON: Class Union vs Channel 5")
    print(f"{'='*70}")

    # Union of channels 1-5 (or 0-4?)
    print(f"\nTrying: Union of channels 1-5")
    class_union_1to5 = mask[:, :, 1:6].sum(axis=-1) > 0
    num_pixels_1to5 = np.sum(class_union_1to5)
    print(f"  Non-zero pixels (channels 1-5): {num_pixels_1to5}")

    print(f"\nTrying: Union of channels 0-4")
    class_union_0to4 = mask[:, :, 0:5].sum(axis=-1) > 0
    num_pixels_0to4 = np.sum(class_union_0to4)
    print(f"  Non-zero pixels (channels 0-4): {num_pixels_0to4}")

    num_pixels_ch5 = np.sum(inst_channel > 0)
    print(f"  Non-zero pixels (channel 5): {num_pixels_ch5}")

    # Check if any channels contain instance IDs
    print(f"\n{'='*70}")
    print(f"🔍 CHECKING FOR INSTANCE IDS IN CLASS CHANNELS")
    print(f"{'='*70}")

    for c in range(5):
        channel_data = mask[:, :, c]
        unique = np.unique(channel_data)
        if len(unique) > 10:  # Likely instance IDs
            print(f"\nChannel {c} ({PANNUKE_CLASSES.get(c+1, 'Unknown')}):")
            print(f"  ⚠️  Contains {len(unique)} unique values (possible instance IDs)")
            print(f"  Values: {unique[:20]}...")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect raw PanNuke masks")
    parser.add_argument("--masks_file", type=Path, required=True, help="Path to masks.npy")
    parser.add_argument("--image_index", type=int, default=0, help="Image index to inspect")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to inspect")

    args = parser.parse_args()

    for i in range(args.image_index, args.image_index + args.num_images):
        inspect_mask(args.masks_file, i)


if __name__ == "__main__":
    main()
