#!/usr/bin/env python3
"""
Verify Cytology Datasets

Checks if datasets are properly downloaded and structured.
Validates file counts, formats, and annotations.

Usage:
    python scripts/datasets/verify_datasets.py
    python scripts/datasets/verify_datasets.py --dataset tb_panda
"""

import os
import sys
from pathlib import Path
import argparse
import json
from collections import defaultdict

# Dataset configurations
DATASETS_CONFIG = {
    "tb_panda": {
        "name": "TB-PANDA (Thyroid FNA)",
        "path": "data/raw/tb_panda",
        "expected_samples": 10000,
        "min_samples": 8000,
        "required_dirs": ["images"],
        "image_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        "organ": "Thyroid",
        "classes": ["Bethesda_I", "Bethesda_II", "Bethesda_III", "Bethesda_IV", "Bethesda_V", "Bethesda_VI"]
    },
    "herlev": {
        "name": "Herlev (Cervical Pap Smear)",
        "path": "data/raw/herlev",
        "expected_samples": 917,
        "min_samples": 900,
        "required_dirs": ["images"],
        "image_extensions": [".bmp", ".png", ".jpg"],
        "organ": "Cervix",
        "classes": ["normal", "light_dysplasia", "moderate_dysplasia", "severe_dysplasia", "carcinoma"]
    },
    "sipakmed": {
        "name": "SIPaKMeD (Cervical Pap Smear)",
        "path": "data/raw/sipakmed",
        "expected_samples": 4049,
        "min_samples": 4000,
        "required_dirs": ["images"],
        "image_extensions": [".bmp", ".png", ".jpg"],
        "organ": "Cervix",
        "classes": ["im_Superficial-Intermediate", "im_Parabasal", "im_Koilocytotic", "im_Dyskeratotic", "im_Metaplastic"]
    },
    "isbi_2014": {
        "name": "ISBI 2014 (Breast Mitoses)",
        "path": "data/raw/isbi_2014_mitoses",
        "expected_samples": 1200,
        "min_samples": 1000,
        "required_dirs": ["images"],
        "image_extensions": [".tif", ".tiff", ".png"],
        "organ": "Breast",
        "classes": ["mitosis", "no_mitosis"]
    }
}

def count_images(directory, extensions):
    """Count images in directory recursively"""
    count = 0
    if not directory.exists():
        return 0

    for ext in extensions:
        count += len(list(directory.rglob(f"*{ext}")))

    return count

def verify_dataset(dataset_name, config):
    """Verify a single dataset"""
    print(f"\n{'='*60}")
    print(f"📂 {config['name']}")
    print(f"{'='*60}")

    dataset_path = Path(config['path'])

    # Check if dataset directory exists
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print(f"💡 Follow download instructions in: docs/DATASET_ACQUISITION_GUIDE.md")
        return False

    print(f"✅ Dataset directory found: {dataset_path}")

    # Check required directories
    missing_dirs = []
    for req_dir in config['required_dirs']:
        req_path = dataset_path / req_dir
        if not req_path.exists():
            missing_dirs.append(req_dir)
        else:
            print(f"✅ Found directory: {req_dir}/")

    if missing_dirs:
        print(f"⚠️  Missing directories: {', '.join(missing_dirs)}")

    # Count images
    images_dir = dataset_path / "images"
    num_images = count_images(dataset_path, config['image_extensions'])

    print(f"\n📊 Image Statistics:")
    print(f"   Found: {num_images:,} images")
    print(f"   Expected: ~{config['expected_samples']:,} images")
    print(f"   Minimum: {config['min_samples']:,} images")

    # Validation
    if num_images >= config['min_samples']:
        status = "✅ VALID"
        percentage = (num_images / config['expected_samples']) * 100
        print(f"   Status: {status} ({percentage:.1f}%)")
        is_valid = True
    elif num_images > 0:
        status = "⚠️  PARTIAL"
        percentage = (num_images / config['min_samples']) * 100
        print(f"   Status: {status} ({percentage:.1f}% of minimum)")
        is_valid = False
    else:
        status = "❌ EMPTY"
        print(f"   Status: {status}")
        is_valid = False

    # Check for class subdirectories
    if images_dir.exists():
        subdirs = [d.name for d in images_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"\n📁 Class Subdirectories Found:")
            class_counts = {}
            for subdir in subdirs:
                subdir_path = images_dir / subdir
                count = count_images(subdir_path, config['image_extensions'])
                class_counts[subdir] = count
                status_icon = "✅" if count > 0 else "❌"
                print(f"   {status_icon} {subdir}: {count:,} images")

            # Check if expected classes are present
            expected_classes = set(config.get('classes', []))
            found_classes = set(subdirs)
            missing_classes = expected_classes - found_classes
            extra_classes = found_classes - expected_classes

            if missing_classes:
                print(f"\n⚠️  Missing expected classes: {', '.join(missing_classes)}")
            if extra_classes:
                print(f"\n💡 Extra classes found: {', '.join(extra_classes)}")

    # Check for annotations
    annotation_files = list(dataset_path.rglob("*.csv")) + list(dataset_path.rglob("*.json")) + list(dataset_path.rglob("*.xml"))
    if annotation_files:
        print(f"\n📝 Annotation Files Found:")
        for ann_file in annotation_files[:5]:  # Show first 5
            print(f"   ✅ {ann_file.relative_to(dataset_path)}")
        if len(annotation_files) > 5:
            print(f"   ... and {len(annotation_files) - 5} more")
    else:
        print(f"\n⚠️  No annotation files found (CSV, JSON, XML)")

    # Dataset-specific checks
    if dataset_name == "tb_panda":
        print(f"\n🔬 TB-PANDA Specific Checks:")
        readme = dataset_path / "README.md"
        if readme.exists():
            print(f"   ✅ README.md found")
        else:
            print(f"   ⚠️  README.md not found")

    elif dataset_name == "herlev":
        print(f"\n🔬 Herlev Specific Checks:")
        smeardb = dataset_path / "smeardb.bmp"
        if smeardb.exists():
            print(f"   ✅ Original database file found")

    return is_valid

def print_summary(results):
    """Print summary of all datasets"""
    print(f"\n{'='*60}")
    print(f"📊 VERIFICATION SUMMARY")
    print(f"{'='*60}\n")

    total_datasets = len(results)
    valid_datasets = sum(1 for v in results.values() if v['valid'])
    total_images = sum(v['count'] for v in results.values())

    print(f"Datasets Checked: {total_datasets}")
    print(f"Valid Datasets: {valid_datasets}/{total_datasets}")
    print(f"Total Images Found: {total_images:,}\n")

    print("| Dataset | Status | Images |")
    print("|---------|--------|--------|")
    for name, info in results.items():
        status_icon = "✅" if info['valid'] else ("⚠️" if info['count'] > 0 else "❌")
        print(f"| {info['name']:<30} | {status_icon} | {info['count']:>7,} |")

    print(f"\n📁 Data directory: data/raw/")
    print(f"📖 Download guide: docs/DATASET_ACQUISITION_GUIDE.md")

    if valid_datasets == total_datasets:
        print(f"\n✅ ALL DATASETS READY FOR PREPROCESSING")
        print(f"\n💡 Next step:")
        print(f"   python scripts/datasets/preprocess_cytology.py --all")
    elif valid_datasets > 0:
        print(f"\n⚠️  SOME DATASETS MISSING")
        print(f"\n💡 Next steps:")
        print(f"   1. Download missing datasets (see guide)")
        print(f"   2. Re-run verification")
        print(f"   3. Preprocess available datasets")
    else:
        print(f"\n❌ NO DATASETS FOUND")
        print(f"\n💡 Next step:")
        print(f"   Follow download instructions in: docs/DATASET_ACQUISITION_GUIDE.md")

def main():
    parser = argparse.ArgumentParser(
        description="Verify cytology datasets are properly downloaded"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS_CONFIG.keys()),
        help="Verify specific dataset only"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )

    args = parser.parse_args()

    print("="*60)
    print("🔍 CYTOLOGY DATASETS VERIFICATION")
    print("="*60)

    results = {}

    if args.dataset:
        # Verify single dataset
        config = DATASETS_CONFIG[args.dataset]
        is_valid = verify_dataset(args.dataset, config)
        num_images = count_images(Path(config['path']), config['image_extensions'])
        results[args.dataset] = {
            'name': config['name'],
            'valid': is_valid,
            'count': num_images
        }
    else:
        # Verify all datasets
        for dataset_name, config in DATASETS_CONFIG.items():
            is_valid = verify_dataset(dataset_name, config)
            num_images = count_images(Path(config['path']), config['image_extensions'])
            results[dataset_name] = {
                'name': config['name'],
                'valid': is_valid,
                'count': num_images
            }

    # Print summary
    print_summary(results)

    # Exit code
    if all(r['valid'] for r in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
