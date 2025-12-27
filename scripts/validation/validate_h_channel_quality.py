#!/usr/bin/env python3
"""
Validation H-Channel Quality - 3 Checks
========================================

1. Test du "Négatif": Visualize H-channel images
2. Alignement Macenko: Check for NaN/failures
3. Poids du Fichier: Verify file size reasonable

Usage: python validate_h_channel_quality.py --family epidermal
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def check_1_visualize_h_channels(data_file: Path, n_samples: int = 4):
    """Check 1: Visualize H-channel images to verify quality."""
    print("="*80)
    print("CHECK 1: VISUALISATION H-CHANNELS")
    print("="*80)

    data = np.load(data_file)

    # Load data
    images_224 = data['images_224']  # (N, 224, 224, 3) RGB
    h_channels_224 = data['h_channels_224']  # (N, 224, 224) H-channel

    n_total = len(images_224)
    print(f"\nTotal samples: {n_total}")

    # Select random samples
    np.random.seed(42)
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    # Create visualization
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples * 3))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        rgb = images_224[idx]
        h_channel = h_channels_224[idx]

        # RGB image
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Sample {idx}: RGB (Original)")
        axes[i, 0].axis('off')

        # H-channel (inverted colormap - darker = more hematoxylin)
        axes[i, 1].imshow(h_channel, cmap='gray')
        axes[i, 1].set_title(f"Sample {idx}: H-Channel (uint8)")
        axes[i, 1].axis('off')

        # Statistics
        h_mean = h_channel.mean()
        h_std = h_channel.std()
        h_min, h_max = h_channel.min(), h_channel.max()

        print(f"\nSample {idx}:")
        print(f"  H-channel stats: mean={h_mean:.1f}, std={h_std:.1f}, range=[{h_min}, {h_max}]")

        # Quality checks
        if h_std < 10:
            print(f"  ⚠️  WARNING: Very low std ({h_std:.1f}) - image might be too uniform")
        elif h_std > 100:
            print(f"  ⚠️  WARNING: Very high std ({h_std:.1f}) - check normalization")
        else:
            print(f"  ✅ Std OK")

        if h_min == h_max:
            print(f"  ❌ FAIL: Constant image (all pixels = {h_min})")
        elif h_max - h_min < 50:
            print(f"  ⚠️  WARNING: Low contrast (range = {h_max - h_min})")
        else:
            print(f"  ✅ Contrast OK")

    plt.tight_layout()
    output_file = Path("results/h_channel_validation.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_file}")

    data.close()
    return True


def check_2_macenko_alignment(data_file: Path):
    """Check 2: Verify no NaN/Inf values from Macenko failures."""
    print("\n" + "="*80)
    print("CHECK 2: ALIGNEMENT MACENKO (NaN/Inf)")
    print("="*80)

    data = np.load(data_file)

    images_224 = data['images_224']
    h_channels_224 = data['h_channels_224']

    # Check for NaN/Inf
    has_nan_img = np.isnan(images_224).any()
    has_inf_img = np.isinf(images_224).any()
    has_nan_h = np.isnan(h_channels_224).any()
    has_inf_h = np.isinf(h_channels_224).any()

    print(f"\nRGB Images (224×224):")
    print(f"  Contains NaN: {has_nan_img}")
    print(f"  Contains Inf: {has_inf_img}")

    print(f"\nH-Channels (224×224):")
    print(f"  Contains NaN: {has_nan_h}")
    print(f"  Contains Inf: {has_inf_h}")

    if has_nan_img or has_inf_img or has_nan_h or has_inf_h:
        print("\n❌ FAIL: NaN/Inf detected - Macenko normalization failed on some images")

        # Find problematic samples
        if has_nan_h or has_inf_h:
            bad_samples = []
            for i in range(len(h_channels_224)):
                if np.isnan(h_channels_224[i]).any() or np.isinf(h_channels_224[i]).any():
                    bad_samples.append(i)

            print(f"  Problematic samples: {len(bad_samples)}/{len(h_channels_224)}")
            if len(bad_samples) <= 10:
                print(f"  Indices: {bad_samples}")

        return False
    else:
        print("\n✅ PASS: No NaN/Inf - Macenko alignment OK")

    # Check metadata if available
    if 'macenko_applied' in data.keys():
        macenko_applied = data['macenko_applied']
        print(f"\nMacenko normalization applied: {macenko_applied}")

        if macenko_applied:
            h_std_mean = data.get('h_channel_std_mean', None)
            h_std_range = data.get('h_channel_std_range', None)

            if h_std_mean is not None:
                print(f"  H-channel std mean: {h_std_mean:.4f}")
            if h_std_range is not None:
                print(f"  H-channel std range: [{h_std_range[0]:.4f}, {h_std_range[1]:.4f}]")

    data.close()
    return True


def check_3_file_weight(data_file: Path, family: str):
    """Check 3: Verify file size is reasonable."""
    print("\n" + "="*80)
    print("CHECK 3: POIDS DU FICHIER")
    print("="*80)

    file_size_bytes = data_file.stat().st_size
    file_size_mb = file_size_bytes / (1024 ** 2)

    print(f"\nFile: {data_file.name}")
    print(f"Size: {file_size_mb:.2f} MB")

    # Load data to check dtypes
    data = np.load(data_file)

    print(f"\nData types:")
    for key in data.keys():
        arr = data[key]
        if hasattr(arr, 'shape'):
            mem_mb = arr.nbytes / (1024 ** 2)
            print(f"  {key}: {arr.dtype}, shape {arr.shape}, {mem_mb:.2f} MB")

    data.close()

    # Expected sizes (rough estimates)
    # Epidermal: ~571 samples
    # Images (224, 224, 3) uint8: ~571 * 224 * 224 * 3 = ~86 MB
    # H-channels (224, 224) uint8: ~571 * 224 * 224 = ~29 MB
    # Targets: similar sizes
    # Total expected: ~100-150 MB (with compression)

    expected_range = (50, 200)  # MB

    if file_size_mb < expected_range[0]:
        print(f"\n⚠️  WARNING: File suspiciously small ({file_size_mb:.2f} MB < {expected_range[0]} MB)")
        print(f"   Check if data is complete")
    elif file_size_mb > expected_range[1]:
        print(f"\n⚠️  WARNING: File suspiciously large ({file_size_mb:.2f} MB > {expected_range[1]} MB)")
        print(f"   Possible issues:")
        print(f"   - Using float64 instead of uint8/float32")
        print(f"   - Duplicate data stored")
        print(f"   - Compression not applied")
    else:
        print(f"\n✅ PASS: File size reasonable ({file_size_mb:.2f} MB in expected range)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate H-channel quality")
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
                        help='Family name')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of samples to visualize (default: 4)')

    args = parser.parse_args()

    # Locate data file
    data_file = Path(f"data/family_data_v13_hybrid/{args.family}_data_v13_hybrid.npz")

    if not data_file.exists():
        print(f"❌ ERROR: Data file not found: {data_file}")
        return 1

    print("\n" + "🔬" * 40)
    print("H-CHANNEL QUALITY VALIDATION")
    print("🔬" * 40)
    print(f"\nFamily: {args.family}")
    print(f"File: {data_file}")

    # Run 3 checks
    results = []

    try:
        results.append(("Visualize H-Channels", check_1_visualize_h_channels(data_file, args.n_samples)))
    except Exception as e:
        print(f"\n❌ CHECK 1 FAILED: {e}")
        results.append(("Visualize H-Channels", False))

    try:
        results.append(("Macenko Alignment", check_2_macenko_alignment(data_file)))
    except Exception as e:
        print(f"\n❌ CHECK 2 FAILED: {e}")
        results.append(("Macenko Alignment", False))

    try:
        results.append(("File Weight", check_3_file_weight(data_file, args.family)))
    except Exception as e:
        print(f"\n❌ CHECK 3 FAILED: {e}")
        results.append(("File Weight", False))

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} — {check_name}")

    n_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {n_passed}/{len(results)} checks passed")

    if n_passed == len(results):
        print("\n🎉 ALL CHECKS PASSED! Data is ready for training.")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED - Review warnings before training.")
        return 1


if __name__ == '__main__':
    exit(main())
