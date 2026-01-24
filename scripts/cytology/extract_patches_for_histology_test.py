#!/usr/bin/env python3
"""
Extract 224x224 patches from cytology images for testing with histology module.

This script extracts patches that likely contain cells (not empty background)
for testing the V13 histology module on cytology data.

Usage:
    python scripts/cytology/extract_patches_for_histology_test.py \
        --image path/to/cytology_image.jpg \
        --output patches_output/ \
        --max_patches 20

Author: V15.3 Cytology Branch
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np


def has_cell_content(patch: np.ndarray, threshold: float = 0.15) -> tuple:
    """
    Check if a patch likely contains cells (not just empty background).

    Uses multiple heuristics:
    1. Not too white (background is usually light)
    2. Has some color variation (cells have nuclei = dark spots)
    3. Has enough non-background pixels

    Args:
        patch: RGB image (224, 224, 3)
        threshold: Minimum ratio of "interesting" pixels

    Returns:
        (has_content: bool, score: float)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # Check if too white (empty background)
    mean_intensity = np.mean(gray)
    if mean_intensity > 240:  # Almost white
        return False, 0.0

    # Check for dark regions (nuclei are dark)
    dark_pixels = np.sum(gray < 150) / gray.size

    # Check for color variation (stained cells have color)
    std_intensity = np.std(gray)

    # Compute score
    score = dark_pixels * 0.5 + (std_intensity / 50) * 0.5

    return score > threshold, score


def extract_patches(
    image_path: str,
    output_dir: str,
    patch_size: int = 224,
    stride: int = 112,
    max_patches: int = 20,
    min_score: float = 0.15
) -> list:
    """
    Extract patches with cell content from an image.

    Args:
        image_path: Path to input image
        output_dir: Directory to save patches
        patch_size: Size of patches (default 224)
        stride: Stride between patches (default 112 = 50% overlap)
        max_patches: Maximum number of patches to extract
        min_score: Minimum content score to keep patch

    Returns:
        List of saved patch paths
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract all patches with scores
    patches_with_scores = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]

            has_content, score = has_cell_content(patch, threshold=min_score)

            if has_content:
                patches_with_scores.append({
                    'patch': patch,
                    'x': x,
                    'y': y,
                    'score': score
                })

    # Sort by score (best patches first)
    patches_with_scores.sort(key=lambda p: p['score'], reverse=True)

    # Take top N patches
    selected = patches_with_scores[:max_patches]

    # Save patches
    image_name = Path(image_path).stem
    saved_paths = []

    for i, p in enumerate(selected):
        patch_filename = f"{image_name}_patch_{i:03d}_x{p['x']}_y{p['y']}_score{p['score']:.2f}.png"
        patch_path = output_path / patch_filename

        # Save as PNG (lossless)
        cv2.imwrite(str(patch_path), cv2.cvtColor(p['patch'], cv2.COLOR_RGB2BGR))
        saved_paths.append(str(patch_path))

        print(f"  [{i+1}/{len(selected)}] Saved: {patch_filename}")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract 224x224 patches from cytology images for histology module testing"
    )

    parser.add_argument("--image", type=str, required=True,
                        help="Path to cytology image")
    parser.add_argument("--output", type=str, default="patches_for_histology_test",
                        help="Output directory for patches")
    parser.add_argument("--patch_size", type=int, default=224,
                        help="Patch size (default: 224)")
    parser.add_argument("--stride", type=int, default=112,
                        help="Stride between patches (default: 112)")
    parser.add_argument("--max_patches", type=int, default=20,
                        help="Maximum patches to extract (default: 20)")
    parser.add_argument("--min_score", type=float, default=0.15,
                        help="Minimum content score (default: 0.15)")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  EXTRACT PATCHES FOR HISTOLOGY MODULE TEST")
    print("=" * 60)
    print(f"  Image: {args.image}")
    print(f"  Output: {args.output}")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  Max patches: {args.max_patches}")
    print("-" * 60)

    try:
        saved = extract_patches(
            args.image,
            args.output,
            patch_size=args.patch_size,
            stride=args.stride,
            max_patches=args.max_patches,
            min_score=args.min_score
        )

        print("-" * 60)
        print(f"  Total patches extracted: {len(saved)}")
        print(f"  Output directory: {args.output}")
        print("=" * 60)

    except Exception as e:
        print(f"  [ERROR] {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
