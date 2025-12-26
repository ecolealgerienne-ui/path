#!/usr/bin/env python3
"""
Verify Clean Split (Grouped Split) Integrity

This script validates that the Clean Split implementation correctly prevents
data leakage between train and validation sets.

Usage:
    python scripts/validation/verify_clean_split.py \
        --data_file data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz

Author: Claude AI (CellViT-Optimus Development)
Date: 2025-12-26
"""

import argparse
import numpy as np
from pathlib import Path


def verify_clean_split(data_path: Path) -> bool:
    """
    Verify the integrity of a Clean Split dataset.

    Checks:
    1. split_types field exists
    2. split_types values are 0 or 1
    3. No source ID appears in both train and val
    4. All source IDs are assigned
    5. Split ratio is approximately 80/20

    Args:
        data_path: Path to .npz file

    Returns:
        True if all checks pass, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"🔍 CLEAN SPLIT INTEGRITY VERIFICATION")
    print(f"{'='*80}\n")

    print(f"📂 Loading dataset: {data_path}")

    try:
        data = np.load(data_path)
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        return False

    print(f"   ✅ Dataset loaded successfully\n")

    # List all fields
    print(f"📋 Dataset Fields:")
    for key in sorted(data.keys()):
        if key in ['images_224', 'h_channels_224', 'np_targets', 'hv_targets', 'nt_targets']:
            print(f"   • {key}: {data[key].shape}, {data[key].dtype}")
        elif key in ['source_image_ids', 'fold_ids', 'crop_position_ids', 'split_types']:
            print(f"   • {key}: {data[key].shape}, {data[key].dtype}")
        else:
            # Metadata
            print(f"   • {key}: {data[key]}")

    print()

    all_checks_passed = True

    # ============================================================================
    # CHECK 1: split_types field exists
    # ============================================================================
    print(f"{'─'*80}")
    print(f"CHECK 1: split_types field exists")
    print(f"{'─'*80}")

    if 'split_types' not in data:
        print(f"❌ CRITICAL: 'split_types' field is MISSING!")
        print(f"   This dataset was NOT created with Clean Split.")
        print(f"   Re-run prepare_v13_hybrid_dataset.py with updated script.")
        return False

    print(f"✅ PASS: 'split_types' field present\n")

    split_types = data['split_types']
    source_ids = data['source_image_ids']

    # ============================================================================
    # CHECK 2: split_types values are 0 or 1
    # ============================================================================
    print(f"{'─'*80}")
    print(f"CHECK 2: split_types encoding is correct")
    print(f"{'─'*80}")

    unique_splits = np.unique(split_types)
    if not set(unique_splits) == {0, 1}:
        print(f"❌ FAIL: Invalid split_types values: {unique_splits}")
        print(f"   Expected: {{0, 1}}")
        print(f"   Got: {set(unique_splits)}")
        all_checks_passed = False
    else:
        print(f"✅ PASS: split_types contains only 0 (Train) and 1 (Val)\n")

    # ============================================================================
    # CHECK 3: No source ID overlap
    # ============================================================================
    print(f"{'─'*80}")
    print(f"CHECK 3: No source ID appears in both train and val")
    print(f"{'─'*80}")

    train_sources = np.unique(source_ids[split_types == 0])
    val_sources = np.unique(source_ids[split_types == 1])
    overlap = np.intersect1d(train_sources, val_sources)

    if len(overlap) > 0:
        print(f"❌ CRITICAL FAIL: {len(overlap)} source IDs appear in BOTH train and val!")
        print(f"   This is DATA LEAKAGE!")
        print(f"   Overlapping source IDs: {overlap[:10]}...")
        all_checks_passed = False
    else:
        print(f"✅ PASS: No source ID overlap (0 sources in both train and val)\n")

    # ============================================================================
    # CHECK 4: All source IDs assigned
    # ============================================================================
    print(f"{'─'*80}")
    print(f"CHECK 4: All source IDs are assigned to train or val")
    print(f"{'─'*80}")

    all_sources = np.unique(source_ids)
    assigned_sources = np.union1d(train_sources, val_sources)

    if len(all_sources) != len(assigned_sources):
        missing = len(all_sources) - len(assigned_sources)
        print(f"❌ FAIL: {missing} source IDs were NOT assigned to train or val!")
        all_checks_passed = False
    else:
        print(f"✅ PASS: All {len(all_sources)} source IDs are assigned\n")

    # ============================================================================
    # CHECK 5: Split ratio is approximately 80/20
    # ============================================================================
    print(f"{'─'*80}")
    print(f"CHECK 5: Split ratio is approximately 80/20")
    print(f"{'─'*80}")

    n_train_sources = len(train_sources)
    n_val_sources = len(val_sources)
    n_total_sources = len(all_sources)

    train_pct = n_train_sources / n_total_sources * 100
    val_pct = n_val_sources / n_total_sources * 100

    print(f"Source Image Split:")
    print(f"   Train: {n_train_sources}/{n_total_sources} ({train_pct:.1f}%)")
    print(f"   Val:   {n_val_sources}/{n_total_sources} ({val_pct:.1f}%)")

    # Allow 5% tolerance for small datasets
    if not (75 <= train_pct <= 85):
        print(f"⚠️  WARNING: Train split is {train_pct:.1f}% (expected ~80%)")
        print(f"   This may be OK for small datasets.")
    else:
        print(f"✅ PASS: Train split is within expected range (75-85%)")

    print()

    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================
    print(f"{'='*80}")
    print(f"📊 SPLIT STATISTICS SUMMARY")
    print(f"{'='*80}\n")

    n_train_crops = np.sum(split_types == 0)
    n_val_crops = np.sum(split_types == 1)
    n_total_crops = len(split_types)

    print(f"Crop-Level Split:")
    print(f"   Total crops:  {n_total_crops}")
    print(f"   Train crops:  {n_train_crops} ({n_train_crops/n_total_crops*100:.1f}%)")
    print(f"   Val crops:    {n_val_crops} ({n_val_crops/n_total_crops*100:.1f}%)")

    print(f"\nSource-Level Split:")
    print(f"   Total sources: {n_total_sources}")
    print(f"   Train sources: {n_train_sources} ({train_pct:.1f}%)")
    print(f"   Val sources:   {n_val_sources} ({val_pct:.1f}%)")

    print(f"\nData Leakage Check:")
    print(f"   Source ID overlap: {len(overlap)} (must be 0)")
    print(f"   Unassigned sources: {n_total_sources - len(assigned_sources)} (must be 0)")

    print()

    # ============================================================================
    # FINAL VERDICT
    # ============================================================================
    print(f"{'='*80}")
    if all_checks_passed and len(overlap) == 0:
        print(f"✅ ALL CHECKS PASSED - Clean Split is VALID!")
        print(f"{'='*80}\n")
        print(f"🎉 This dataset is safe to use for training and validation.")
        print(f"   No data leakage detected.\n")
        return True
    else:
        print(f"❌ SOME CHECKS FAILED - Clean Split is INVALID!")
        print(f"{'='*80}\n")
        print(f"⚠️  This dataset has integrity issues.")
        print(f"   Please re-run prepare_v13_hybrid_dataset.py\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify Clean Split (Grouped Split) integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Verify epidermal family dataset
    python scripts/validation/verify_clean_split.py \\
        --data_file data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz

    # Verify all families
    for family in glandular digestive urologic epidermal respiratory; do
        python scripts/validation/verify_clean_split.py \\
            --data_file data/family_data_v13_hybrid/${family}_data_v13_hybrid.npz
    done
        """
    )

    parser.add_argument(
        '--data_file',
        type=Path,
        required=True,
        help='Path to .npz dataset file'
    )

    args = parser.parse_args()

    # Check file exists
    if not args.data_file.exists():
        print(f"❌ ERROR: File not found: {args.data_file}")
        print(f"   Please check the path and try again.\n")
        return 1

    # Run verification
    success = verify_clean_split(args.data_file)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
