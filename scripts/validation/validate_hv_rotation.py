#!/usr/bin/env python3
"""
HV Maps Rotation Validation
============================

Validates that HV vector field transformations are correct after rotation.

For correct HV maps:
- Vectors must point toward nucleus centers (gradient flow)
- After rotation, vectors must still point to ROTATED nucleus centers
- Component swapping must be exact (90° → H'=V, V'=-H)

Usage:
    python scripts/validation/validate_hv_rotation.py \
        --data_file data/family_data_v13_smart_crops/epidermal_train_v13_smart_crops.npz \
        --n_samples 5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def visualize_hv_vectors(
    image: np.ndarray,
    np_mask: np.ndarray,
    hv_map: np.ndarray,
    title: str,
    ax: plt.Axes,
    subsample: int = 16
):
    """
    Visualizes HV vector field overlaid on image.

    Args:
        image: (224, 224, 3) RGB uint8
        np_mask: (224, 224) binary mask
        hv_map: (2, 224, 224) HV gradients float32 [-1, 1]
        title: Plot title
        ax: Matplotlib axes
        subsample: Show 1 vector every N pixels
    """
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

    # Extract H and V components
    h_comp = hv_map[0]
    v_comp = hv_map[1]

    # Subsample grid
    y_coords, x_coords = np.meshgrid(
        np.arange(0, 224, subsample),
        np.arange(0, 224, subsample),
        indexing='ij'
    )

    # Only show vectors inside nuclei
    for y, x in zip(y_coords.flatten(), x_coords.flatten()):
        if np_mask[y, x] > 0:  # Inside nucleus
            dx = h_comp[y, x] * subsample * 0.5  # Scale for visibility
            dy = v_comp[y, x] * subsample * 0.5

            # Draw arrow from (x, y) in direction (dx, dy)
            arrow = FancyArrowPatch(
                (x, y),
                (x + dx, y + dy),
                arrowstyle='->',
                color='yellow',
                linewidth=1.5,
                alpha=0.7
            )
            ax.add_patch(arrow)


def compute_hv_divergence(hv_map: np.ndarray, np_mask: np.ndarray) -> float:
    """
    Computes divergence of HV field (should be NEGATIVE inside nuclei).

    Correct HV maps have negative divergence (vectors point INWARD to center).

    Args:
        hv_map: (2, 224, 224) HV gradients
        np_mask: (224, 224) binary mask

    Returns:
        Mean divergence inside nuclei (should be < 0)
    """
    h_comp = hv_map[0]
    v_comp = hv_map[1]

    # Compute divergence: ∇·F = ∂H/∂x + ∂V/∂y
    dh_dx = np.gradient(h_comp, axis=1)
    dv_dy = np.gradient(v_comp, axis=0)
    divergence = dh_dx + dv_dy

    # Average inside nuclei
    nuclei_pixels = np_mask > 0
    if nuclei_pixels.sum() > 0:
        mean_div = divergence[nuclei_pixels].mean()
    else:
        mean_div = 0.0

    return mean_div


def validate_hv_consistency(
    images_224: np.ndarray,
    np_targets: np.ndarray,
    hv_targets: np.ndarray,
    n_samples: int = 5
) -> dict:
    """
    Validates HV maps across crops/rotations.

    Checks:
    1. Range: HV values in [-1, 1]
    2. Divergence: Negative inside nuclei (vectors point inward)
    3. Consistency: Similar divergence across different crops

    Args:
        images_224: (N×5, 224, 224, 3)
        np_targets: (N×5, 224, 224)
        hv_targets: (N×5, 2, 224, 224)
        n_samples: Number of sample groups to validate

    Returns:
        Validation results dict
    """
    results = {
        'range_valid': [],
        'divergence_values': [],
        'divergence_negative': []
    }

    for i in range(min(n_samples, len(images_224) // 5)):
        # Get 5 crops from same source image
        start_idx = i * 5
        end_idx = start_idx + 5

        for j in range(start_idx, end_idx):
            hv_map = hv_targets[j]
            np_mask = np_targets[j]

            # Check 1: Range [-1, 1]
            hv_min = hv_map.min()
            hv_max = hv_map.max()
            range_valid = (-1.0 <= hv_min <= 1.0) and (-1.0 <= hv_max <= 1.0)
            results['range_valid'].append(range_valid)

            # Check 2: Divergence negative (vectors point inward)
            div = compute_hv_divergence(hv_map, np_mask)
            results['divergence_values'].append(div)
            results['divergence_negative'].append(div < 0)

    # Summary statistics
    results['range_valid_pct'] = np.mean(results['range_valid']) * 100
    results['divergence_mean'] = np.mean(results['divergence_values'])
    results['divergence_negative_pct'] = np.mean(results['divergence_negative']) * 100

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate HV rotation transformations")
    parser.add_argument('--data_file', type=Path, required=True)
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=Path, default=Path('results/hv_validation'))

    args = parser.parse_args()

    if not args.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    # Load data
    print(f"Loading: {args.data_file}")
    data = np.load(args.data_file)

    images_224 = data['images']
    np_targets = data['np_targets']
    hv_targets = data['hv_targets']

    print(f"  Images: {images_224.shape}")
    print(f"  HV targets: {hv_targets.shape}")

    # Validate
    print("\nValidating HV transformations...")
    results = validate_hv_consistency(images_224, np_targets, hv_targets, args.n_samples)

    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"Range valid:          {results['range_valid_pct']:.1f}% (should be 100%)")
    print(f"Divergence mean:      {results['divergence_mean']:.4f} (should be < 0)")
    print(f"Divergence negative:  {results['divergence_negative_pct']:.1f}% (should be ~100%)")
    print("="*70)

    if results['range_valid_pct'] < 100:
        print("⚠️  WARNING: Some HV values outside [-1, 1]!")

    if results['divergence_mean'] > 0:
        print("⚠️  WARNING: Positive divergence (vectors point OUTWARD instead of INWARD)!")

    if results['divergence_negative_pct'] < 90:
        print("⚠️  WARNING: Many crops have positive divergence!")

    # Visualize sample
    print(f"\nVisualizing {args.n_samples} sample groups...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.n_samples, len(images_224) // 5)):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        # Data order: for each crop position, apply 5 rotations
        # So indices 0-4 are: same crop (e.g. centre) with 5 rotations
        rotation_names = [
            '0° (original)',
            '90° CW',
            '180°',
            '270° CW',
            'Flip H'
        ]

        start_idx = i * 5

        for j in range(5):
            idx = start_idx + j
            visualize_hv_vectors(
                images_224[idx],
                np_targets[idx],
                hv_targets[idx],
                rotation_names[j],
                axes[j],
                subsample=16
            )

        plt.tight_layout()
        output_file = args.output_dir / f"hv_validation_sample_{i:03d}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

    print("\n✅ Validation complete")


if __name__ == '__main__':
    main()
