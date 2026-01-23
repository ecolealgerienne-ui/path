"""
Convert APCData YOLO Labels to Detection-Only Format

Ce script convertit les labels APCData de 6 classes Bethesda vers 1 classe unique "cell".
Cela permet d'entraîner YOLO uniquement pour la détection, laissant la classification
à des heads spécialisées (Binary → Severity → Fine-grained).

Conversion:
    Classes 0-5 (NILM, ASCUS, ASCH, LSIL, HSIL, SCC) → Classe 0 (cell)

Usage:
    python scripts/cytology/04_convert_to_detection_only.py \
        --input_dir data/raw/apcdata/APCData_YOLO \
        --output_dir data/raw/apcdata/APCData_YOLO_Detection

Author: V15 Cytology Branch
Date: 2026-01-23
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_info(message: str):
    print(f"  [INFO] {message}")


def print_success(message: str):
    print(f"  [OK] {message}")


def print_warning(message: str):
    print(f"  [WARN] {message}")


# Original Bethesda classes
BETHESDA_CLASSES = {
    0: "NILM",
    1: "ASCUS",
    2: "ASCH",
    3: "LSIL",
    4: "HSIL",
    5: "SCC"
}

# Target: single class for detection
DETECTION_CLASSES = {
    0: "cell"
}


def convert_label_file(
    input_path: Path,
    output_path: Path
) -> Tuple[int, Dict[int, int]]:
    """
    Convert a single label file from 6 classes to 1 class.

    Args:
        input_path: Path to input label file (6 classes)
        output_path: Path to output label file (1 class)

    Returns:
        Tuple of (num_cells, class_distribution)
    """
    class_counts = {i: 0 for i in range(6)}
    converted_lines = []

    with open(input_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                original_class = int(parts[0])
                x_center = parts[1]
                y_center = parts[2]
                width = parts[3]
                height = parts[4]

                # Track original class distribution
                if original_class in class_counts:
                    class_counts[original_class] += 1

                # Convert to class 0 (cell)
                converted_line = f"0 {x_center} {y_center} {width} {height}\n"
                converted_lines.append(converted_line)

    # Write converted file
    with open(output_path, 'w') as f:
        f.writelines(converted_lines)

    return len(converted_lines), class_counts


def create_data_yaml(output_dir: Path, has_split: bool = True) -> None:
    """Create YOLO data.yaml for detection-only dataset."""

    if has_split:
        yaml_content = f"""# APCData YOLO Detection-Only Configuration
# V15 Cytology Pipeline — Cell Detection (Single Class)
# Converted from 6 Bethesda classes to 1 "cell" class

path: {output_dir.absolute()}
train: train/images
val: val/images

# Single class for detection
names:
  0: cell

nc: 1

# Note: Original classes were NILM, ASCUS, ASCH, LSIL, HSIL, SCC
# Classification is done by specialized heads after detection
"""
    else:
        yaml_content = f"""# APCData YOLO Detection-Only Configuration
# V15 Cytology Pipeline — Cell Detection (Single Class)
# Converted from 6 Bethesda classes to 1 "cell" class

path: {output_dir.absolute()}
train: images
val: images

# Single class for detection
names:
  0: cell

nc: 1

# Note: Original classes were NILM, ASCUS, ASCH, LSIL, HSIL, SCC
# Classification is done by specialized heads after detection
"""

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print_success(f"Created {yaml_path}")


def convert_split(
    input_dir: Path,
    output_dir: Path,
    split: str
) -> Dict[str, int]:
    """Convert a single split (train or val)."""

    input_labels = input_dir / split / "labels"
    input_images = input_dir / split / "images"
    output_labels = output_dir / split / "labels"
    output_images = output_dir / split / "images"

    if not input_labels.exists():
        return None

    # Create output directories
    output_labels.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    total_cells = 0
    total_class_counts = {i: 0 for i in range(6)}
    num_files = 0

    label_files = list(input_labels.glob("*.txt"))
    label_files = [f for f in label_files if "Zone.Identifier" not in f.name]

    for label_file in label_files:
        # Convert label
        output_label = output_labels / label_file.name
        num_cells, class_counts = convert_label_file(label_file, output_label)

        total_cells += num_cells
        for cls_id, count in class_counts.items():
            total_class_counts[cls_id] += count
        num_files += 1

        # Copy corresponding image
        basename = label_file.stem
        for ext in [".jpg", ".png", ".jpeg"]:
            src_img = input_images / f"{basename}{ext}"
            if src_img.exists():
                dst_img = output_images / f"{basename}{ext}"
                shutil.copy2(src_img, dst_img)
                break

    return {
        "num_files": num_files,
        "total_cells": total_cells,
        "class_counts": total_class_counts
    }


def convert_flat_structure(
    input_dir: Path,
    output_dir: Path
) -> Dict[str, int]:
    """Convert flat structure (images/ and labels/ without train/val split)."""

    input_labels = input_dir / "labels"
    input_images = input_dir / "images"
    output_labels = output_dir / "labels"
    output_images = output_dir / "images"

    # Create output directories
    output_labels.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    total_cells = 0
    total_class_counts = {i: 0 for i in range(6)}
    num_files = 0

    label_files = list(input_labels.glob("*.txt"))
    label_files = [f for f in label_files if "Zone.Identifier" not in f.name]

    for label_file in label_files:
        # Convert label
        output_label = output_labels / label_file.name
        num_cells, class_counts = convert_label_file(label_file, output_label)

        total_cells += num_cells
        for cls_id, count in class_counts.items():
            total_class_counts[cls_id] += count
        num_files += 1

        # Copy corresponding image
        basename = label_file.stem
        for ext in [".jpg", ".png", ".jpeg"]:
            src_img = input_images / f"{basename}{ext}"
            if src_img.exists():
                dst_img = output_images / f"{basename}{ext}"
                shutil.copy2(src_img, dst_img)
                break

    return {
        "num_files": num_files,
        "total_cells": total_cells,
        "class_counts": total_class_counts
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert APCData YOLO labels to detection-only format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/apcdata/APCData_YOLO",
        help="Path to input APCData_YOLO directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/apcdata/APCData_YOLO_Detection",
        help="Path to output detection-only directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if exists"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("\n" + "=" * 80)
    print("  APCDATA YOLO → DETECTION-ONLY CONVERSION")
    print("  V15 Cytology Pipeline")
    print("=" * 80)

    # Verify input
    print_header("STEP 1: VERIFY INPUT")

    if not input_dir.exists():
        print(f"  [ERROR] Input directory not found: {input_dir}")
        return 1

    print_success(f"Input directory: {input_dir}")

    # Check structure (split or flat)
    has_split = (input_dir / "train").exists() and (input_dir / "val").exists()
    has_flat = (input_dir / "images").exists() and (input_dir / "labels").exists()

    if has_split:
        print_info("Structure: train/val split detected")
    elif has_flat:
        print_info("Structure: flat (images/ + labels/)")
    else:
        print(f"  [ERROR] Invalid structure. Expected train/val or images/labels")
        return 1

    # Handle output directory
    print_header("STEP 2: PREPARE OUTPUT")

    if output_dir.exists():
        if args.force:
            print_warning(f"Removing existing output: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print_warning(f"Output directory exists: {output_dir}")
            print_info("Use --force to overwrite")
            return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"Output directory: {output_dir}")

    # Convert
    print_header("STEP 3: CONVERT LABELS")

    all_stats = {}

    if has_split:
        # Convert train and val splits
        for split in ["train", "val"]:
            print_info(f"Converting {split}/...")
            stats = convert_split(input_dir, output_dir, split)
            if stats:
                all_stats[split] = stats
                print_success(f"  {split}: {stats['num_files']} files, {stats['total_cells']} cells")
    else:
        # Convert flat structure
        print_info("Converting flat structure...")
        stats = convert_flat_structure(input_dir, output_dir)
        all_stats["all"] = stats
        print_success(f"  Converted: {stats['num_files']} files, {stats['total_cells']} cells")

    # Create data.yaml
    print_header("STEP 4: CREATE DATA.YAML")
    create_data_yaml(output_dir, has_split=has_split)

    # Summary
    print_header("CONVERSION SUMMARY")

    print_info("Original classes distribution:")
    total_by_class = {i: 0 for i in range(6)}
    for split, stats in all_stats.items():
        for cls_id, count in stats["class_counts"].items():
            total_by_class[cls_id] += count

    for cls_id, count in total_by_class.items():
        cls_name = BETHESDA_CLASSES[cls_id]
        print(f"    {cls_id} ({cls_name}): {count}")

    total_cells = sum(total_by_class.values())
    print_info(f"Total cells: {total_cells} → All converted to class 0 (cell)")

    print("\n" + "=" * 80)
    print("  CONVERSION COMPLETED")
    print("=" * 80)
    print(f"\n  Output: {output_dir}")
    print(f"\n  Next step:")
    print(f"    python scripts/cytology/03_train_yolo26_apcdata.py \\")
    print(f"        --data {output_dir}/data.yaml \\")
    print(f"        --model yolo26n.pt \\")
    print(f"        --epochs 100")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
