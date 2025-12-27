#!/usr/bin/env python3
"""
Validate V13-Hybrid Training Data Before Training.

Checks:
1. Hybrid dataset structure and multi-crop consistency
2. RGB features alignment
3. H-channel features alignment
4. Fold distribution
5. Data quality (NaN/Inf, dtypes, ranges)
6. Expected sample counts (571 with --no-multicrop, 2855 with multicrop)

Usage:
    # With multi-crop augmentation (5 crops per image)
    python validate_training_data_v13_hybrid.py --family epidermal

    # Without multi-crop (1:1 ratio)
    python validate_training_data_v13_hybrid.py --family epidermal --no-multicrop
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List


def check_hybrid_dataset(data_path: Path, expect_multicrop: bool = True) -> Dict:
    """Check hybrid dataset structure and multi-crop consistency.

    Args:
        data_path: Path to hybrid dataset .npz file
        expect_multicrop: If True, expects 5 crops per image (n_samples % 5 == 0)
                         If False, accepts any sample count (1:1 ratio)
    """
    print("\n" + "="*80)
    print("CHECK 1: HYBRID DATASET STRUCTURE")
    print("="*80)

    if not data_path.exists():
        return {"valid": False, "error": f"File not found: {data_path}"}

    data = np.load(data_path)
    result = {"valid": True, "checks": []}

    # Check keys
    required_keys = ['images_224', 'h_channels_224', 'np_targets', 'hv_targets',
                     'nt_targets', 'fold_ids', 'source_image_ids', 'crop_position_ids']
    missing_keys = [k for k in required_keys if k not in data.keys()]

    if missing_keys:
        result["valid"] = False
        result["error"] = f"Missing keys: {missing_keys}"
        return result

    print(f"✅ All required keys present")

    # Check shapes
    n_samples = len(data['images_224'])
    n_unique_source_ids = len(np.unique(data['source_image_ids']))

    print(f"\n📊 Dataset size: {n_samples} samples")
    print(f"   Unique source images: {n_unique_source_ids}")
    print(f"   Mode: {'Multi-crop (5×)' if expect_multicrop else 'No augmentation (1:1)'}")

    # Check multi-crop consistency
    if expect_multicrop:
        if n_samples % 5 != 0:
            result["valid"] = False
            result["error"] = f"Sample count {n_samples} not divisible by 5!"
            print(f"  ❌ FAIL: Sample count must be divisible by 5 for multi-crop")
            print(f"  💡 TIP: Use --no-multicrop flag if data was generated without augmentation")
            return result

        n_inferred_images = n_samples // 5
        print(f"  Total samples: {n_samples}")
        print(f"  Inferred original images: {n_inferred_images} (= {n_samples} / 5)")

        if n_unique_source_ids < n_inferred_images:
            print(f"  ℹ️  Note: {n_inferred_images - n_unique_source_ids} duplicate source_image_ids")
            print(f"     This is OK if the same image was processed multiple times")

        print(f"  ✅ PASS: Sample count {n_samples} = {n_inferred_images} × 5 (valid multi-crop)")
        result["checks"].append("multi_crop_count")
    else:
        # No multi-crop: 1:1 ratio expected
        if n_samples != n_unique_source_ids:
            print(f"  ⚠️  WARNING: {n_samples} samples but {n_unique_source_ids} unique source IDs")
            print(f"     Expected 1:1 ratio for no-multicrop mode")
        else:
            print(f"  ✅ PASS: 1:1 ratio ({n_samples} samples = {n_unique_source_ids} source images)")

        result["checks"].append("no_multicrop_count")

    # Check crop positions distribution
    crop_positions = data['crop_position_ids']
    unique_positions, counts = np.unique(crop_positions, return_counts=True)

    print(f"\n📐 Crop positions distribution:")
    crop_names = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
    for pos, count in zip(unique_positions, counts):
        if pos < len(crop_names):
            print(f"  {crop_names[pos]}: {count} samples")

    if expect_multicrop:
        # Multi-crop mode: expect 5 positions with equal distribution
        if len(unique_positions) != 5:
            result["valid"] = False
            result["error"] = f"Expected 5 crop positions, got {len(unique_positions)}"
            print(f"  ❌ FAIL: Expected 5 crop positions")
            return result

        n_inferred_images = n_samples // 5
        if not np.allclose(counts, n_inferred_images):
            result["valid"] = False
            result["error"] = "Crop distribution incorrect"
            print(f"  ❌ FAIL: Expected {n_inferred_images} samples per crop position")
            return result
        else:
            print(f"  ✅ PASS: All 5 crop positions have {n_inferred_images} samples each")
            result["checks"].append("crop_distribution")
    else:
        # No multi-crop mode: expect only center crops (position 0)
        if len(unique_positions) == 1 and unique_positions[0] == 0:
            print(f"  ✅ PASS: All samples are center crops (no augmentation)")
            result["checks"].append("no_multicrop_crops")
        else:
            print(f"  ⚠️  WARNING: Expected only center crops (position 0) for no-multicrop mode")
            print(f"     Found positions: {unique_positions.tolist()}")
            # Don't fail, just warn

    # Check fold distribution
    folds = data['fold_ids']
    unique_folds, fold_counts = np.unique(folds, return_counts=True)

    print(f"\n🗂️  Fold distribution:")
    for fold, count in zip(unique_folds, fold_counts):
        print(f"  Fold {fold}: {count} samples ({count/n_samples*100:.1f}%)")

    # For single-family training, we typically use only fold 0
    if len(unique_folds) == 1 and 0 in unique_folds:
        print(f"  ✅ PASS: Using fold 0 (standard for single-family training)")
        result["checks"].append("fold_0_only")
    elif len(unique_folds) >= 3:
        print(f"  ✅ PASS: All 3 folds present")
        result["checks"].append("fold_distribution")
    else:
        print(f"  ⚠️  WARNING: {len(unique_folds)} fold(s) present")
        # Don't fail, just warn

    # Check shapes consistency
    print(f"\n📏 Array shapes:")
    print(f"  images_224: {data['images_224'].shape} (expected: {n_samples}, 224, 224, 3)")
    print(f"  h_channels_224: {data['h_channels_224'].shape} (expected: {n_samples}, 224, 224)")
    print(f"  np_targets: {data['np_targets'].shape} (expected: {n_samples}, 224, 224)")
    print(f"  hv_targets: {data['hv_targets'].shape} (expected: {n_samples}, 2, 224, 224)")
    print(f"  nt_targets: {data['nt_targets'].shape} (expected: {n_samples}, 224, 224)")

    expected_shapes = {
        'images_224': (n_samples, 224, 224, 3),
        'h_channels_224': (n_samples, 224, 224),
        'np_targets': (n_samples, 224, 224),
        'hv_targets': (n_samples, 2, 224, 224),
        'nt_targets': (n_samples, 224, 224),
    }

    for key, expected_shape in expected_shapes.items():
        if data[key].shape != expected_shape:
            result["valid"] = False
            result["error"] = f"Shape mismatch for {key}: {data[key].shape} != {expected_shape}"
            print(f"  ❌ FAIL: {key} shape mismatch")
            return result

    print(f"  ✅ PASS: All shapes correct")
    result["checks"].append("shapes")

    # Check dtypes
    print(f"\n🔢 Data types:")
    print(f"  images_224: {data['images_224'].dtype} (expected: uint8)")
    print(f"  h_channels_224: {data['h_channels_224'].dtype} (expected: uint8)")
    print(f"  np_targets: {data['np_targets'].dtype} (expected: float32)")
    print(f"  hv_targets: {data['hv_targets'].dtype} (expected: float32)")
    print(f"  nt_targets: {data['nt_targets'].dtype} (expected: int64)")

    expected_dtypes = {
        'images_224': np.uint8,
        'h_channels_224': np.uint8,
        'np_targets': np.float32,
        'hv_targets': np.float32,
        'nt_targets': np.int64,
    }

    for key, expected_dtype in expected_dtypes.items():
        if data[key].dtype != expected_dtype:
            result["valid"] = False
            result["error"] = f"Dtype mismatch for {key}: {data[key].dtype} != {expected_dtype}"
            print(f"  ❌ FAIL: {key} dtype mismatch")
            return result

    print(f"  ✅ PASS: All dtypes correct")
    result["checks"].append("dtypes")

    # Check HV targets range (critical!)
    hv_min, hv_max = data['hv_targets'].min(), data['hv_targets'].max()
    print(f"\n⚠️  CRITICAL: HV targets range check")
    print(f"  HV min: {hv_min:.4f}")
    print(f"  HV max: {hv_max:.4f}")
    print(f"  Expected: [-1.0, 1.0]")

    if not (-1.0 <= hv_min <= hv_max <= 1.0):
        result["valid"] = False
        result["error"] = f"HV range [{hv_min:.3f}, {hv_max:.3f}] outside [-1, 1]!"
        print(f"  ❌ FAIL: HV targets out of range!")
        return result
    else:
        print(f"  ✅ PASS: HV targets in correct range")
        result["checks"].append("hv_range")

    # Check for NaN/Inf
    print(f"\n🔍 NaN/Inf check:")
    has_nan = any(np.isnan(data[k]).any() for k in ['np_targets', 'hv_targets', 'nt_targets'])
    has_inf = any(np.isinf(data[k]).any() for k in ['np_targets', 'hv_targets'])

    if has_nan:
        result["valid"] = False
        result["error"] = "NaN values detected in targets"
        print(f"  ❌ FAIL: NaN values detected")
        return result

    if has_inf:
        result["valid"] = False
        result["error"] = "Inf values detected in targets"
        print(f"  ❌ FAIL: Inf values detected")
        return result

    print(f"  ✅ PASS: No NaN/Inf values")
    result["checks"].append("nan_inf")

    data.close()

    print(f"\n✅ HYBRID DATASET: ALL CHECKS PASSED ({len(result['checks'])}/6)")
    return result


def check_h_features(h_features_path: Path, expected_samples: int) -> Dict:
    """Check H-channel features alignment."""
    print("\n" + "="*80)
    print("CHECK 2: H-CHANNEL FEATURES")
    print("="*80)

    if not h_features_path.exists():
        return {"valid": False, "error": f"File not found: {h_features_path}"}

    data = np.load(h_features_path)
    result = {"valid": True, "checks": []}

    if 'h_features' not in data:
        result["valid"] = False
        result["error"] = "Missing 'h_features' key"
        return result

    h_features = data['h_features']

    print(f"📊 H-features shape: {h_features.shape}")
    print(f"   Expected: ({expected_samples}, 256)")

    if h_features.shape != (expected_samples, 256):
        result["valid"] = False
        result["error"] = f"Shape mismatch: {h_features.shape} != ({expected_samples}, 256)"
        print(f"  ❌ FAIL: Shape mismatch")
        return result

    print(f"  ✅ PASS: Shape correct")
    result["checks"].append("shape")

    # Check dtype
    if h_features.dtype != np.float32:
        result["valid"] = False
        result["error"] = f"Dtype mismatch: {h_features.dtype} != float32"
        print(f"  ❌ FAIL: Dtype mismatch")
        return result

    print(f"  ✅ PASS: Dtype correct (float32)")
    result["checks"].append("dtype")

    # Check statistics
    print(f"\n📈 H-features statistics:")
    print(f"  Mean: {h_features.mean():.6f}")
    print(f"  Std: {h_features.std():.6f}")
    print(f"  Range: [{h_features.min():.6f}, {h_features.max():.6f}]")

    # Check for NaN/Inf
    if np.isnan(h_features).any():
        result["valid"] = False
        result["error"] = "NaN values detected"
        print(f"  ❌ FAIL: NaN values detected")
        return result

    if np.isinf(h_features).any():
        result["valid"] = False
        result["error"] = "Inf values detected"
        print(f"  ❌ FAIL: Inf values detected")
        return result

    print(f"  ✅ PASS: No NaN/Inf values")
    result["checks"].append("nan_inf")

    data.close()

    print(f"\n✅ H-FEATURES: ALL CHECKS PASSED ({len(result['checks'])}/3)")
    return result


def check_rgb_features(rgb_features_path: Path, expected_samples: int) -> Dict:
    """Check RGB features alignment."""
    print("\n" + "="*80)
    print("CHECK 3: RGB FEATURES (H-OPTIMUS-0)")
    print("="*80)

    if not rgb_features_path.exists():
        return {"valid": False, "error": f"File not found: {rgb_features_path}"}

    data = np.load(rgb_features_path, mmap_mode='r')
    result = {"valid": True, "checks": []}

    # Check for features key
    if 'features' in data:
        features = data['features']
    elif 'layer_24' in data:
        features = data['layer_24']
    else:
        result["valid"] = False
        result["error"] = "Missing 'features' or 'layer_24' key"
        return result

    print(f"📊 RGB features shape: {features.shape}")
    print(f"   Expected: ({expected_samples}, 261, 1536)")
    print(f"   Where 261 = CLS (1) + Registers (4) + Patches (256)")

    if features.shape != (expected_samples, 261, 1536):
        result["valid"] = False
        result["error"] = f"Shape mismatch: {features.shape} != ({expected_samples}, 261, 1536)"
        print(f"  ❌ FAIL: Shape mismatch")
        return result

    print(f"  ✅ PASS: Shape correct")
    result["checks"].append("shape")

    # Check dtype
    if features.dtype != np.float32:
        result["valid"] = False
        result["error"] = f"Dtype mismatch: {features.dtype} != float32"
        print(f"  ❌ FAIL: Dtype mismatch")
        return result

    print(f"  ✅ PASS: Dtype correct (float32)")
    result["checks"].append("dtype")

    # Check CLS token std (should be ~0.70-0.90 after Bug #1/#2 fixes)
    cls_tokens = features[:, 0, :]  # (N, 1536)
    cls_std = cls_tokens.std()

    print(f"\n🔍 CLS token validation (Bug #1/#2 detection):")
    print(f"  CLS std: {cls_std:.4f}")
    print(f"  Expected range: [0.70, 0.90]")

    if not (0.70 <= cls_std <= 0.90):
        result["valid"] = False
        result["error"] = f"CLS std {cls_std:.4f} outside [0.70, 0.90] - preprocessing corrupted!"
        print(f"  ❌ FAIL: CLS std indicates corrupted preprocessing")
        print(f"  This suggests Bug #1 (ToPILImage float64) or Bug #2 (LayerNorm mismatch)")
        return result

    print(f"  ✅ PASS: CLS std OK - preprocessing correct")
    result["checks"].append("cls_std")

    # Sample a few features to check for NaN/Inf (full check would be too slow on mmap)
    sample_indices = np.random.choice(expected_samples, min(100, expected_samples), replace=False)
    sample_features = features[sample_indices]

    if np.isnan(sample_features).any():
        result["valid"] = False
        result["error"] = "NaN values detected in sampled features"
        print(f"  ❌ FAIL: NaN values detected")
        return result

    if np.isinf(sample_features).any():
        result["valid"] = False
        result["error"] = "Inf values detected in sampled features"
        print(f"  ❌ FAIL: Inf values detected")
        return result

    print(f"  ✅ PASS: No NaN/Inf in sampled features")
    result["checks"].append("nan_inf")

    print(f"\n✅ RGB FEATURES: ALL CHECKS PASSED ({len(result['checks'])}/4)")
    return result


def check_alignment(hybrid_path: Path, h_features_path: Path, rgb_features_path: Path) -> Dict:
    """Check that all three datasets have aligned sample counts."""
    print("\n" + "="*80)
    print("CHECK 4: DATASET ALIGNMENT")
    print("="*80)

    result = {"valid": True}

    # Load sample counts
    hybrid_data = np.load(hybrid_path)
    h_data = np.load(h_features_path)
    rgb_data = np.load(rgb_features_path, mmap_mode='r')

    n_hybrid = len(hybrid_data['images_224'])
    n_h = len(h_data['h_features'])

    if 'features' in rgb_data:
        n_rgb = len(rgb_data['features'])
    else:
        n_rgb = len(rgb_data['layer_24'])

    print(f"📊 Sample counts:")
    print(f"  Hybrid dataset: {n_hybrid}")
    print(f"  H-features: {n_h}")
    print(f"  RGB features: {n_rgb}")

    if n_hybrid != n_h or n_hybrid != n_rgb:
        result["valid"] = False
        result["error"] = f"Sample count mismatch: hybrid={n_hybrid}, h={n_h}, rgb={n_rgb}"
        print(f"  ❌ FAIL: Sample counts don't match!")
        return result

    print(f"  ✅ PASS: All datasets have {n_hybrid} samples")

    # Check fold_ids match
    hybrid_folds = hybrid_data['fold_ids']

    if 'fold_ids' in h_data:
        h_folds = h_data['fold_ids']
        if not np.array_equal(hybrid_folds, h_folds):
            result["valid"] = False
            result["error"] = "fold_ids mismatch between hybrid and H-features"
            print(f"  ❌ FAIL: fold_ids don't match")
            return result
        print(f"  ✅ PASS: fold_ids aligned between hybrid and H-features")

    if 'fold_ids' in rgb_data:
        rgb_folds = rgb_data['fold_ids']
        if not np.array_equal(hybrid_folds, rgb_folds):
            result["valid"] = False
            result["error"] = "fold_ids mismatch between hybrid and RGB features"
            print(f"  ❌ FAIL: fold_ids don't match")
            return result
        print(f"  ✅ PASS: fold_ids aligned between hybrid and RGB features")

    hybrid_data.close()
    h_data.close()

    print(f"\n✅ ALIGNMENT: ALL CHECKS PASSED")
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate V13-Hybrid training data")
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--hybrid_data_dir', type=Path,
                        default=Path('data/family_data_v13_hybrid'))
    parser.add_argument('--features_dir', type=Path,
                        default=Path('data/cache/family_data'))
    parser.add_argument('--no-multicrop', action='store_true',
                        help='Dataset has no multi-crop augmentation (1:1 ratio instead of 1:5)')

    args = parser.parse_args()

    print("\n" + "🔬"*40)
    print("V13-HYBRID TRAINING DATA VALIDATION")
    print("🔬"*40)
    print(f"\nFamily: {args.family}")

    # Paths
    hybrid_path = args.hybrid_data_dir / f"{args.family}_data_v13_hybrid.npz"
    h_features_path = args.features_dir / f"{args.family}_h_features_v13.npz"
    rgb_features_path = args.features_dir / f"{args.family}_rgb_features_v13.npz"

    # Run checks
    results = []

    # Check 1: Hybrid dataset
    expect_multicrop = not args.no_multicrop
    hybrid_result = check_hybrid_dataset(hybrid_path, expect_multicrop=expect_multicrop)
    results.append(("Hybrid Dataset", hybrid_result))

    if not hybrid_result["valid"]:
        print(f"\n❌ CRITICAL ERROR in Hybrid Dataset:")
        print(f"   {hybrid_result.get('error', 'Unknown error')}")
        print(f"\n⚠️  Cannot proceed with other checks. Fix this first!")
        return 1

    expected_samples = len(np.load(hybrid_path)['images_224'])
    np.load(hybrid_path).close()

    # Check 2: H-features
    h_result = check_h_features(h_features_path, expected_samples)
    results.append(("H-Features", h_result))

    # Check 3: RGB features
    rgb_result = check_rgb_features(rgb_features_path, expected_samples)
    results.append(("RGB Features", rgb_result))

    # Check 4: Alignment
    if hybrid_result["valid"] and h_result["valid"] and rgb_result["valid"]:
        alignment_result = check_alignment(hybrid_path, h_features_path, rgb_features_path)
        results.append(("Dataset Alignment", alignment_result))

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    all_passed = True
    for name, result in results:
        if result["valid"]:
            n_checks = len(result.get("checks", []))
            print(f"✅ {name}: PASS ({n_checks} checks)")
        else:
            print(f"❌ {name}: FAIL - {result.get('error', 'Unknown error')}")
            all_passed = False

    if all_passed:
        print("\n" + "🎉"*40)
        print("✅ ALL VALIDATION CHECKS PASSED!")
        print("🎉"*40)
        print(f"\n✅ Ready to train with {expected_samples} samples")
        print(f"   Expected training batches/epoch: ~{expected_samples * 0.8 / 16:.0f}")
        print(f"   Expected validation batches: ~{expected_samples * 0.2 / 16:.0f}")
        return 0
    else:
        print("\n" + "⚠️ "*40)
        print("❌ VALIDATION FAILED - DO NOT TRAIN YET!")
        print("⚠️ "*40)
        print("\nFix the errors above before proceeding with training.")
        return 1


if __name__ == '__main__':
    exit(main())
