#!/usr/bin/env python3
"""
Preprocess Cytology Datasets

Converts raw datasets to unified format for training.
- Resize images to 512×512
- Convert to PNG RGB
- Extract annotations to unified JSON format
- Create train/val splits (80/20 stratified)
- Quality control filtering

Usage:
    python scripts/datasets/preprocess_cytology.py --all
    python scripts/datasets/preprocess_cytology.py --dataset tb_panda
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

# Output directory
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Target image size
TARGET_SIZE = 512

def resize_image(img, target_size=TARGET_SIZE):
    """
    Resize image preserving aspect ratio

    Args:
        img: PIL Image
        target_size: Target size (will be max dimension)

    Returns:
        Resized PIL Image
    """
    # Get current size
    w, h = img.size

    # Calculate scaling factor
    if w > h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))

    # Resize
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create canvas and paste
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    return canvas

def validate_image(img_path):
    """
    Check if image is valid and not corrupted

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        img = Image.open(img_path)
        img.verify()  # Verify image integrity

        # Re-open for actual check (verify() closes the file)
        img = Image.open(img_path)

        # Check dimensions
        w, h = img.size
        if w < 64 or h < 64:
            return False

        # Check if image can be converted to RGB
        img.convert('RGB')

        return True
    except Exception as e:
        print(f"⚠️  Invalid image {img_path}: {e}")
        return False

def preprocess_tb_panda():
    """
    Preprocess TB-PANDA (Thyroid FNA) dataset

    Expected structure:
        data/raw/tb_panda/
        ├── images/
        │   ├── Bethesda_I/
        │   ├── Bethesda_II/
        │   ...
    """
    print(f"\n{'='*60}")
    print(f"📥 PREPROCESSING TB-PANDA (Thyroid)")
    print(f"{'='*60}")

    raw_dir = Path("data/raw/tb_panda")
    output_dir = PROCESSED_DIR / "tb_panda"

    if not raw_dir.exists():
        print(f"❌ Dataset not found: {raw_dir}")
        return False

    # Create output directories
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)

    # Collect all images by class
    images_by_class = defaultdict(list)

    images_dir = raw_dir / "images"
    if images_dir.exists():
        for class_dir in images_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                        if validate_image(img_path):
                            images_by_class[class_name].append(img_path)

    if not images_by_class:
        print(f"❌ No valid images found in {images_dir}")
        return False

    print(f"📊 Found {sum(len(imgs) for imgs in images_by_class.values())} valid images")
    print(f"📁 Classes: {list(images_by_class.keys())}")

    # Stratified split
    all_annotations = []
    train_annotations = []
    val_annotations = []

    image_id = 0

    for class_name, img_paths in images_by_class.items():
        print(f"\n🔄 Processing class: {class_name} ({len(img_paths)} images)")

        # Split train/val (80/20)
        train_paths, val_paths = train_test_split(
            img_paths,
            test_size=0.2,
            random_state=42
        )

        # Process train images
        for img_path in tqdm(train_paths, desc=f"  Train"):
            try:
                # Load and preprocess
                img = Image.open(img_path).convert('RGB')
                img_processed = resize_image(img)

                # Save
                output_path = output_dir / "train" / "images" / f"{image_id:06d}.png"
                img_processed.save(output_path, "PNG")

                # Annotation
                annotation = {
                    "image_id": f"tb_panda_{image_id:06d}",
                    "file_name": f"{image_id:06d}.png",
                    "organ": "Thyroid",
                    "classification_system": "Bethesda",
                    "diagnosis": class_name,
                    "width": TARGET_SIZE,
                    "height": TARGET_SIZE,
                    "source_dataset": "TB-PANDA",
                    "source_file": str(img_path.name)
                }
                train_annotations.append(annotation)
                image_id += 1

            except Exception as e:
                print(f"⚠️  Error processing {img_path}: {e}")

        # Process val images
        for img_path in tqdm(val_paths, desc=f"  Val"):
            try:
                # Load and preprocess
                img = Image.open(img_path).convert('RGB')
                img_processed = resize_image(img)

                # Save
                output_path = output_dir / "val" / "images" / f"{image_id:06d}.png"
                img_processed.save(output_path, "PNG")

                # Annotation
                annotation = {
                    "image_id": f"tb_panda_{image_id:06d}",
                    "file_name": f"{image_id:06d}.png",
                    "organ": "Thyroid",
                    "classification_system": "Bethesda",
                    "diagnosis": class_name,
                    "width": TARGET_SIZE,
                    "height": TARGET_SIZE,
                    "source_dataset": "TB-PANDA",
                    "source_file": str(img_path.name)
                }
                val_annotations.append(annotation)
                image_id += 1

            except Exception as e:
                print(f"⚠️  Error processing {img_path}: {e}")

    # Save annotations
    with open(output_dir / "train" / "annotations.json", 'w') as f:
        json.dump(train_annotations, f, indent=2)

    with open(output_dir / "val" / "annotations.json", 'w') as f:
        json.dump(val_annotations, f, indent=2)

    print(f"\n✅ TB-PANDA preprocessing complete")
    print(f"   Train: {len(train_annotations)} images")
    print(f"   Val: {len(val_annotations)} images")
    print(f"   Output: {output_dir}")

    return True

def preprocess_herlev():
    """
    Preprocess Herlev (Cervical Pap Smear) dataset
    """
    print(f"\n{'='*60}")
    print(f"📥 PREPROCESSING HERLEV (Cervix)")
    print(f"{'='*60}")

    raw_dir = Path("data/raw/herlev")
    output_dir = PROCESSED_DIR / "herlev"

    if not raw_dir.exists():
        print(f"❌ Dataset not found: {raw_dir}")
        return False

    # Create output directories
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)

    # Collect images
    images_by_class = defaultdict(list)

    images_dir = raw_dir / "images"
    if images_dir.exists():
        for class_dir in images_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.bmp', '.png', '.jpg', '.jpeg']:
                        if validate_image(img_path):
                            images_by_class[class_name].append(img_path)
    else:
        # Try flat structure
        for img_path in raw_dir.rglob("*"):
            if img_path.suffix.lower() in ['.bmp', '.png', '.jpg', '.jpeg']:
                if validate_image(img_path):
                    # Infer class from filename if possible
                    images_by_class["unknown"].append(img_path)

    if not images_by_class:
        print(f"❌ No valid images found")
        return False

    print(f"📊 Found {sum(len(imgs) for imgs in images_by_class.values())} valid images")

    # Process similar to TB-PANDA
    train_annotations = []
    val_annotations = []
    image_id = 0

    for class_name, img_paths in images_by_class.items():
        print(f"\n🔄 Processing class: {class_name} ({len(img_paths)} images)")

        train_paths, val_paths = train_test_split(img_paths, test_size=0.2, random_state=42)

        # Process train
        for img_path in tqdm(train_paths, desc="  Train"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_processed = resize_image(img)

                output_path = output_dir / "train" / "images" / f"{image_id:06d}.png"
                img_processed.save(output_path, "PNG")

                annotation = {
                    "image_id": f"herlev_{image_id:06d}",
                    "file_name": f"{image_id:06d}.png",
                    "organ": "Cervix",
                    "classification_system": "CIN",
                    "diagnosis": class_name,
                    "width": TARGET_SIZE,
                    "height": TARGET_SIZE,
                    "source_dataset": "Herlev",
                    "source_file": str(img_path.name)
                }
                train_annotations.append(annotation)
                image_id += 1
            except Exception as e:
                print(f"⚠️  Error: {e}")

        # Process val
        for img_path in tqdm(val_paths, desc="  Val"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_processed = resize_image(img)

                output_path = output_dir / "val" / "images" / f"{image_id:06d}.png"
                img_processed.save(output_path, "PNG")

                annotation = {
                    "image_id": f"herlev_{image_id:06d}",
                    "file_name": f"{image_id:06d}.png",
                    "organ": "Cervix",
                    "classification_system": "CIN",
                    "diagnosis": class_name,
                    "width": TARGET_SIZE,
                    "height": TARGET_SIZE,
                    "source_dataset": "Herlev",
                    "source_file": str(img_path.name)
                }
                val_annotations.append(annotation)
                image_id += 1
            except Exception as e:
                print(f"⚠️  Error: {e}")

    # Save annotations
    with open(output_dir / "train" / "annotations.json", 'w') as f:
        json.dump(train_annotations, f, indent=2)

    with open(output_dir / "val" / "annotations.json", 'w') as f:
        json.dump(val_annotations, f, indent=2)

    print(f"\n✅ Herlev preprocessing complete")
    print(f"   Train: {len(train_annotations)} images")
    print(f"   Val: {len(val_annotations)} images")

    return True

def preprocess_sipakmed():
    """Preprocess SIPaKMeD dataset (same structure as Herlev)"""
    print(f"\n{'='*60}")
    print(f"📥 PREPROCESSING SIPaKMeD (Cervix)")
    print(f"{'='*60}")

    # Similar to Herlev
    print("💡 Implementation similar to Herlev - adapt paths")
    return False

def preprocess_isbi_2014():
    """Preprocess ISBI 2014 Mitosis dataset"""
    print(f"\n{'='*60}")
    print(f"📥 PREPROCESSING ISBI 2014 (Breast Mitoses)")
    print(f"{'='*60}")

    print("💡 Implementation needed - histology format")
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess cytology datasets to unified format"
    )
    parser.add_argument(
        "--dataset",
        choices=["tb_panda", "herlev", "sipakmed", "isbi_2014", "all"],
        help="Dataset to preprocess"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Preprocess all available datasets"
    )

    args = parser.parse_args()

    print("="*60)
    print("🔄 CYTOLOGY DATASETS PREPROCESSING")
    print("="*60)

    if args.all or args.dataset == "all":
        datasets = ["tb_panda", "herlev", "sipakmed", "isbi_2014"]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("❌ Please specify --dataset or --all")
        sys.exit(1)

    results = {}

    for dataset in datasets:
        if dataset == "tb_panda":
            results[dataset] = preprocess_tb_panda()
        elif dataset == "herlev":
            results[dataset] = preprocess_herlev()
        elif dataset == "sipakmed":
            results[dataset] = preprocess_sipakmed()
        elif dataset == "isbi_2014":
            results[dataset] = preprocess_isbi_2014()

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 PREPROCESSING SUMMARY")
    print(f"{'='*60}")

    for dataset, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {dataset}")

    successful = sum(1 for s in results.values() if s)
    print(f"\nSuccessful: {successful}/{len(results)}")

    if successful > 0:
        print(f"\n💡 Next steps:")
        print(f"   1. Verify processed data:")
        print(f"      ls -la data/processed/")
        print(f"   2. Review annotations:")
        print(f"      cat data/processed/tb_panda/train/annotations.json | head -50")
        print(f"   3. Configure organ settings:")
        print(f"      config/cytology_organ_config.json")
        print(f"   4. Start training CellPose models")

if __name__ == "__main__":
    main()
