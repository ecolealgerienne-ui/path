"""
Prepare APCData for YOLO26 Training — Train/Val Split

Ce script prépare APCData pour l'entraînement YOLO26:
1. Split stratifié train/val (80/20)
2. Copie les images et labels dans la structure YOLO standard
3. Met à jour le fichier data.yaml

Structure finale:
    APCData_YOLO/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml

Usage:
    python scripts/cytology/02_prepare_apcdata_split.py \
        --data_dir data/raw/apcdata/APCData_YOLO \
        --val_ratio 0.2

Author: V15 Cytology Branch
Date: 2026-01-22
"""

import os
import sys
import argparse
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def print_step(step_num: int, description: str):
    """Print formatted step header"""
    print("\n" + "=" * 80)
    print(f"  STEP {step_num}: {description}")
    print("=" * 80)


def print_info(message: str):
    print(f"  ℹ️  {message}")


def print_success(message: str):
    print(f"  ✅ {message}")


def print_warning(message: str):
    print(f"  ⚠️  {message}")


# ═════════════════════════════════════════════════════════════════════════════
#  YOLO CLASS MAPPING
# ═════════════════════════════════════════════════════════════════════════════

YOLO_CLASS_MAPPING = {
    0: "NILM",
    1: "ASCUS",
    2: "ASCH",
    3: "LSIL",
    4: "HSIL",
    5: "SCC"
}


def get_image_class(label_path: Path) -> int:
    """
    Get the dominant class for an image (for stratified split).

    Strategy: Use the most severe class present in the image.
    Severity order: SCC > HSIL > ASCH > LSIL > ASCUS > NILM
    """
    severity_order = {5: 6, 4: 5, 2: 4, 3: 3, 1: 2, 0: 1}  # SCC=6, HSIL=5, etc.

    max_severity = 0
    dominant_class = 0

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                severity = severity_order.get(class_id, 0)
                if severity > max_severity:
                    max_severity = severity
                    dominant_class = class_id

    return dominant_class


def stratified_split(
    data_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Perform stratified train/val split based on dominant class.

    Returns:
        train_files: List of file basenames for training
        val_files: List of file basenames for validation
    """
    random.seed(seed)

    labels_dir = data_dir / "labels"

    # Group files by dominant class
    class_files: Dict[int, List[str]] = defaultdict(list)

    for label_file in labels_dir.glob("*.txt"):
        if "Zone.Identifier" in label_file.name:
            continue

        basename = label_file.stem
        dominant_class = get_image_class(label_file)
        class_files[dominant_class].append(basename)

    # Stratified split
    train_files = []
    val_files = []

    for class_id, files in sorted(class_files.items()):
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio))  # At least 1 for validation

        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])

        class_name = YOLO_CLASS_MAPPING.get(class_id, f"class_{class_id}")
        print_info(f"  Class {class_id} ({class_name}): {len(files)} total → "
                   f"{len(files) - n_val} train, {n_val} val")

    return train_files, val_files


def create_split_structure(
    data_dir: Path,
    train_files: List[str],
    val_files: List[str]
) -> None:
    """Create train/val directory structure and copy files."""

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    # Create directories
    for split in ["train", "val"]:
        (data_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (data_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_files(files: List[str], split: str):
        for basename in files:
            # Find image (could be .jpg or .png)
            for ext in [".jpg", ".png", ".jpeg"]:
                src_img = images_dir / f"{basename}{ext}"
                if src_img.exists():
                    dst_img = data_dir / split / "images" / f"{basename}{ext}"
                    shutil.copy2(src_img, dst_img)
                    break

            # Copy label
            src_label = labels_dir / f"{basename}.txt"
            if src_label.exists():
                dst_label = data_dir / split / "labels" / f"{basename}.txt"
                shutil.copy2(src_label, dst_label)

    print_info(f"Copying {len(train_files)} files to train/...")
    copy_files(train_files, "train")

    print_info(f"Copying {len(val_files)} files to val/...")
    copy_files(val_files, "val")


def create_data_yaml(data_dir: Path) -> None:
    """Create YOLO data.yaml file."""

    yaml_content = f"""# APCData YOLO Configuration — Auto-generated
# V15 Cytology Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

# Dataset paths
path: {data_dir.absolute()}
train: train/images
val: val/images

# Classes (Bethesda System)
names:
  0: NILM    # Negative for Intraepithelial Lesion or Malignancy
  1: ASCUS   # Atypical Squamous Cells of Undetermined Significance
  2: ASCH    # Atypical Squamous Cells, cannot exclude HSIL
  3: LSIL    # Low-grade Squamous Intraepithelial Lesion
  4: HSIL    # High-grade Squamous Intraepithelial Lesion
  5: SCC     # Squamous Cell Carcinoma

nc: 6
"""

    yaml_path = data_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print_success(f"Created {yaml_path}")


def verify_split(data_dir: Path) -> Dict:
    """Verify the split structure."""

    stats = {}

    for split in ["train", "val"]:
        images = list((data_dir / split / "images").glob("*"))
        labels = list((data_dir / split / "labels").glob("*.txt"))

        # Filter out Zone.Identifier
        labels = [l for l in labels if "Zone.Identifier" not in l.name]

        stats[split] = {
            "images": len(images),
            "labels": len(labels)
        }

        print_info(f"{split}: {len(images)} images, {len(labels)} labels")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare APCData for YOLO training")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/apcdata/APCData_YOLO",
        help="Path to APCData_YOLO directory"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing train/val directories"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("\n" + "=" * 80)
    print("  APCDATA TRAIN/VAL SPLIT PREPARATION")
    print("  V15 Cytology Pipeline")
    print("=" * 80)

    # Step 1: Check existing structure
    print_step(1, "CHECK EXISTING STRUCTURE")

    if not data_dir.exists():
        print(f"  ❌ Data directory not found: {data_dir}")
        return 1

    train_exists = (data_dir / "train").exists()
    val_exists = (data_dir / "val").exists()

    if train_exists or val_exists:
        if args.force:
            print_warning("Removing existing train/val directories...")
            if train_exists:
                shutil.rmtree(data_dir / "train")
            if val_exists:
                shutil.rmtree(data_dir / "val")
        else:
            print_warning("train/ or val/ already exists. Use --force to overwrite.")
            print_info("Verifying existing split...")
            verify_split(data_dir)
            return 0

    print_success(f"Data directory: {data_dir}")

    # Step 2: Stratified split
    print_step(2, f"STRATIFIED SPLIT (val_ratio={args.val_ratio})")

    train_files, val_files = stratified_split(
        data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print_success(f"Train: {len(train_files)} images")
    print_success(f"Val: {len(val_files)} images")

    # Step 3: Create structure
    print_step(3, "CREATE DIRECTORY STRUCTURE")

    create_split_structure(data_dir, train_files, val_files)

    # Step 4: Create data.yaml
    print_step(4, "CREATE DATA.YAML")

    create_data_yaml(data_dir)

    # Step 5: Verify
    print_step(5, "VERIFY SPLIT")

    stats = verify_split(data_dir)

    # Summary
    print("\n" + "=" * 80)
    print("  SPLIT COMPLETED")
    print("=" * 80)
    print(f"\n  Structure created:")
    print(f"    {data_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/ ({stats['train']['images']} files)")
    print(f"    │   └── labels/ ({stats['train']['labels']} files)")
    print(f"    ├── val/")
    print(f"    │   ├── images/ ({stats['val']['images']} files)")
    print(f"    │   └── labels/ ({stats['val']['labels']} files)")
    print(f"    └── data.yaml")
    print(f"\n  Next step:")
    print(f"    python scripts/cytology/03_train_yolo26_apcdata.py \\")
    print(f"        --data {data_dir}/data.yaml \\")
    print(f"        --epochs 100")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
