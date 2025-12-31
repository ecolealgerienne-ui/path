#!/usr/bin/env python3
"""
Migration script: Add organ_names to existing V13 Smart Crops .npz files.

This script adds the organ_names field to .npz files that were generated
before this field was added to prepare_v13_smart_crops.py.

Usage:
    python scripts/preprocessing/migrate_add_organ_names.py \
        --family epidermal \
        --pannuke_dir /path/to/PanNuke

The script will:
1. Load existing train/val .npz files
2. Look up organ names from PanNuke types.npy using source_image_ids
3. Add organ_names array and resave the files
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# PanNuke organ mapping (index -> name)
PANNUKE_ORGANS = [
    "Breast", "Colon", "Lung", "Kidney", "Prostate",
    "Bladder", "Ovarian", "Esophagus", "Thyroid", "Stomach",
    "Liver", "Uterus", "HeadNeck", "Bile-duct", "Testis",
    "Cervix", "Adrenal_gland", "Skin", "Pancreatic", "Unknown"
]


def load_pannuke_types(pannuke_dir: Path):
    """Load organ types from all PanNuke folds."""
    all_types = {}

    for fold_idx in range(3):
        fold_dir = pannuke_dir / f"fold{fold_idx}"
        types_file = fold_dir / "types.npy"

        if not types_file.exists():
            raise FileNotFoundError(f"Types file not found: {types_file}")

        types = np.load(types_file, allow_pickle=True)

        # Store as (fold_idx, image_idx) -> organ_name
        for img_idx, organ_type in enumerate(types):
            # types.npy can contain strings directly or indices
            if isinstance(organ_type, (str, np.str_)):
                organ_name = str(organ_type)
            elif isinstance(organ_type, (int, np.integer)):
                organ_name = PANNUKE_ORGANS[organ_type] if organ_type < len(PANNUKE_ORGANS) else "Unknown"
            else:
                organ_name = str(organ_type)
            all_types[(fold_idx, img_idx)] = organ_name

    return all_types


def migrate_npz_file(npz_path: Path, pannuke_types: dict, dry_run: bool = False):
    """Add organ_names to a single .npz file."""
    print(f"\nProcessing: {npz_path}")

    # Load existing data
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    print(f"  Existing keys: {keys}")

    # Check if already has organ_names
    if 'organ_names' in keys:
        print(f"  ✅ Already has organ_names, skipping")
        return True

    # Need source_image_ids to look up organs
    if 'source_image_ids' not in keys:
        print(f"  ❌ Missing source_image_ids, cannot migrate")
        return False

    source_ids = data['source_image_ids']
    n_samples = len(source_ids)
    print(f"  Samples: {n_samples}")

    # Build organ_names array
    organ_names = []
    missing_count = 0

    for sid in source_ids:
        # source_image_id format: "fold{fold_idx}_{image_idx}" or similar
        # Parse it to get fold_idx and image_idx
        sid_str = sid.decode('utf-8') if isinstance(sid, bytes) else str(sid)

        try:
            # Try format: "fold0_123" or "0_123"
            if sid_str.startswith('fold'):
                parts = sid_str.replace('fold', '').split('_')
                fold_idx = int(parts[0])
                img_idx = int(parts[1])
            else:
                parts = sid_str.split('_')
                fold_idx = int(parts[0])
                img_idx = int(parts[1])

            key = (fold_idx, img_idx)
            if key in pannuke_types:
                organ_names.append(pannuke_types[key])
            else:
                organ_names.append("Unknown")
                missing_count += 1
        except (ValueError, IndexError):
            organ_names.append("Unknown")
            missing_count += 1

    organ_names = np.array(organ_names, dtype=object)

    # Show distribution
    unique, counts = np.unique(organ_names, return_counts=True)
    print(f"  Organ distribution:")
    for organ, count in zip(unique, counts):
        print(f"    {organ}: {count}")

    if missing_count > 0:
        print(f"  ⚠️  {missing_count} samples with unknown organ")

    if dry_run:
        print(f"  🔍 DRY RUN - would add organ_names")
        return True

    # Create new dict with all existing data + organ_names
    new_data = {key: data[key] for key in keys}
    new_data['organ_names'] = organ_names

    # Save (overwrite)
    np.savez(npz_path, **new_data)
    print(f"  ✅ Saved with organ_names")

    return True


def main():
    parser = argparse.ArgumentParser(description="Add organ_names to existing V13 Smart Crops files")
    parser.add_argument("--family", type=str, required=True,
                        help="Family to migrate (epidermal, digestive, etc.)")
    parser.add_argument("--pannuke_dir", type=str, required=True,
                        help="Path to PanNuke dataset directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be done without modifying files")

    args = parser.parse_args()

    pannuke_dir = Path(args.pannuke_dir)
    data_dir = Path("data/family_data_v13_smart_crops")

    print(f"{'='*60}")
    print(f"MIGRATION: Add organ_names to V13 Smart Crops")
    print(f"{'='*60}")
    print(f"Family: {args.family}")
    print(f"PanNuke dir: {pannuke_dir}")
    print(f"Data dir: {data_dir}")
    if args.dry_run:
        print(f"Mode: DRY RUN (no changes)")

    # Load PanNuke types
    print(f"\nLoading PanNuke organ types...")
    pannuke_types = load_pannuke_types(pannuke_dir)
    print(f"  Loaded {len(pannuke_types)} image->organ mappings")

    # Find files to migrate
    train_file = data_dir / f"{args.family}_train_v13_smart_crops.npz"
    val_file = data_dir / f"{args.family}_val_v13_smart_crops.npz"

    success = True

    for npz_file in [train_file, val_file]:
        if npz_file.exists():
            if not migrate_npz_file(npz_file, pannuke_types, args.dry_run):
                success = False
        else:
            print(f"\n⚠️  File not found: {npz_file}")

    print(f"\n{'='*60}")
    if success:
        print("✅ Migration complete!")
    else:
        print("❌ Migration had errors")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
