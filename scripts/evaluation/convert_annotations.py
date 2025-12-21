#!/usr/bin/env python3
"""
Convert dataset annotations to unified format for evaluation.

Converts from various formats (.mat, .npy, HuggingFace) to standardized .npz:
- instance_map: (H, W) with values 0=background, 1..N=instances
- type_map: (H, W) with values 0=background, 1..K=classes
- centroids: (N, 2) with [x, y] coordinates

Handles class mapping:
- CoNSeP (7 classes) → PanNuke (5 classes)
- MoNuSAC (4 classes) → PanNuke (5 classes)

Usage:
    # Convert CoNSeP to unified format
    python scripts/evaluation/convert_annotations.py \
        --dataset consep \
        --input_dir data/evaluation/consep \
        --output_dir data/evaluation/consep_converted

    # Convert PanNuke (already in correct format, just unify)
    python scripts/evaluation/convert_annotations.py \
        --dataset pannuke \
        --input_dir data/evaluation/pannuke/Fold\ 2 \
        --output_dir data/evaluation/pannuke_fold2_converted
"""

import argparse
import numpy as np
import scipy.io as sio
import cv2
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm

# Class mappings
PANNUKE_CLASSES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial"
}

# CoNSeP original classes (from HoVer-Net paper)
CONSEP_CLASSES = {
    0: "Background",
    1: "Other",           # → 3 (Connective)
    2: "Inflammatory",    # → 2 (Inflammatory)
    3: "Epithelial",      # → 5 (Epithelial)
    4: "Spindle-shaped",  # → 3 (Connective)
}

# Mapping CoNSeP → PanNuke
CONSEP_TO_PANNUKE = {
    0: 0,  # Background
    1: 3,  # Other → Connective
    2: 2,  # Inflammatory → Inflammatory
    3: 5,  # Epithelial → Epithelial
    4: 3,  # Spindle-shaped → Connective
}

# MoNuSAC classes
MONUSAC_CLASSES = {
    0: "Background",
    1: "Epithelial",   # → 5 (Epithelial)
    2: "Lymphocyte",   # → 2 (Inflammatory)
    3: "Neutrophil",   # → 2 (Inflammatory)
    4: "Macrophage"    # → 2 (Inflammatory)
}

# Mapping MoNuSAC → PanNuke
MONUSAC_TO_PANNUKE = {
    0: 0,  # Background
    1: 5,  # Epithelial → Epithelial
    2: 2,  # Lymphocyte → Inflammatory
    3: 2,  # Neutrophil → Inflammatory
    4: 2   # Macrophage → Inflammatory
}


def load_mat_annotation(mat_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load .mat annotation file (CoNSeP, Lizard format).

    Args:
        mat_file: Path to .mat file

    Returns:
        Tuple of (inst_map, type_map, centroids)
    """
    data = sio.loadmat(str(mat_file))

    # Extract data
    inst_map = data.get('inst_map', None)
    type_map = data.get('type_map', None)
    centroids = data.get('inst_centroid', None)

    if inst_map is None:
        raise ValueError(f"No 'inst_map' found in {mat_file}")

    # ⚠️ CRITICAL: .mat indexing starts at 1, not 0
    # Background is 0, instances are 1..N
    # This is already correct in HoVer-Net format

    return inst_map, type_map, centroids


def load_pannuke_annotation(
    images_file: Path,
    masks_file: Path,
    index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PanNuke annotation (already in .npy format).

    Args:
        images_file: Path to images.npy
        masks_file: Path to masks.npy
        index: Image index in the arrays

    Returns:
        Tuple of (image, inst_map, type_map, centroids)
    """
    images = np.load(images_file, mmap_mode='r')
    masks = np.load(masks_file, mmap_mode='r')

    image = images[index]  # (256, 256, 3)
    mask = masks[index]    # (256, 256, 6)

    # CRITICAL: Convert float64 to uint8 BEFORE saving
    # PanNuke images are float64 [0, 255]
    # ToPILImage() in inference expects uint8, otherwise it multiplies by 255 → overflow!
    # See CLAUDE.md section "BUG #1: ToPILImage avec float64"
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)

    # Extract instance map from PanNuke structure
    # CRITICAL: PanNuke channels 1-4 contain INSTANCE IDs (not binary masks!)
    # Channel 5 (Epithelial) is BINARY and needs connectedComponents
    #
    # PanNuke structure:
    #   Channel 0: Background instances (ignored)
    #   Channel 1: Neoplastic instances (with unique IDs)
    #   Channel 2: Inflammatory instances (with unique IDs)
    #   Channel 3: Connective instances (with unique IDs)
    #   Channel 4: Dead instances (with unique IDs)
    #   Channel 5: Epithelial BINARY mask (needs connectedComponents)

    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Process channels 1-4 (instance IDs already present)
    for c in range(1, 5):  # Channels 1-4 (Neoplastic, Inflammatory, Connective, Dead)
        class_instances = mask[:, :, c]
        inst_ids = np.unique(class_instances)
        inst_ids = inst_ids[inst_ids > 0]  # Skip background (0)

        for inst_id in inst_ids:
            inst_mask = class_instances == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Process channel 5 (Epithelial - binary mask)
    epithelial_mask = mask[:, :, 5] > 0
    if epithelial_mask.any():
        _, epi_instances = cv2.connectedComponents(epithelial_mask.astype(np.uint8))
        epi_inst_ids = np.unique(epi_instances)
        epi_inst_ids = epi_inst_ids[epi_inst_ids > 0]

        for epi_inst_id in epi_inst_ids:
            inst_mask_local = epi_instances == epi_inst_id
            inst_map[inst_mask_local] = instance_counter
            instance_counter += 1

    # Create type map from class channels (1-5)
    # Following the same logic as prepare_family_data.py (training pipeline)
    # Channels 1-5 correspond to classes 1-5 (Neoplastic, Inflammatory, Connective, Dead, Epithelial)
    type_map = np.zeros((256, 256), dtype=np.uint8)

    for c in range(5):  # Iterate over 5 classes
        class_mask = mask[:, :, c + 1] > 0  # Channels 1-5 (indices 1, 2, 3, 4, 5)
        type_map[class_mask] = c + 1  # Assign class IDs 1-5

    # Compute centroids
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    centroids = []
    for inst_id in inst_ids:
        y, x = np.where(inst_map == inst_id)
        if len(x) > 0:
            centroid = [x.mean(), y.mean()]
            centroids.append(centroid)

    centroids = np.array(centroids) if centroids else np.zeros((0, 2))

    return image, inst_map, type_map, centroids


def apply_class_mapping(
    type_map: np.ndarray,
    mapping: Dict[int, int]
) -> np.ndarray:
    """
    Apply class mapping to convert between taxonomies.

    Args:
        type_map: Original type map
        mapping: Dictionary {original_class: new_class}

    Returns:
        Mapped type map
    """
    mapped = np.zeros_like(type_map)

    for orig_class, new_class in mapping.items():
        mapped[type_map == orig_class] = new_class

    return mapped


def convert_consep(input_dir: Path, output_dir: Path) -> None:
    """Convert CoNSeP dataset to unified format."""
    print("\n" + "="*70)
    print("CONVERTING CONSEP TO UNIFIED FORMAT")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .mat files
    mat_files = sorted(input_dir.rglob("*.mat"))

    if len(mat_files) == 0:
        print(f"❌ No .mat files found in {input_dir}")
        return

    print(f"Found {len(mat_files)} .mat files")

    for mat_file in tqdm(mat_files, desc="Converting"):
        try:
            # Load annotation
            inst_map, type_map, centroids = load_mat_annotation(mat_file)

            # Apply class mapping CoNSeP → PanNuke
            if type_map is not None:
                type_map = apply_class_mapping(type_map, CONSEP_TO_PANNUKE)

            # Save to .npz
            output_file = output_dir / f"{mat_file.stem}.npz"
            np.savez_compressed(
                output_file,
                inst_map=inst_map,
                type_map=type_map if type_map is not None else np.zeros_like(inst_map),
                centroids=centroids if centroids is not None else np.zeros((0, 2))
            )

        except Exception as e:
            print(f"⚠️ Error processing {mat_file.name}: {e}")

    print(f"\n✅ Converted {len(mat_files)} files to: {output_dir}")


def convert_pannuke(input_dir: Path, output_dir: Path) -> None:
    """Convert PanNuke dataset to unified format."""
    print("\n" + "="*70)
    print("CONVERTING PANNUKE TO UNIFIED FORMAT")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images.npy and masks.npy
    images_file = input_dir / "images.npy"
    masks_file = input_dir / "masks.npy"
    types_file = input_dir / "types.npy"  # Organ types

    if not images_file.exists() or not masks_file.exists():
        print(f"❌ Missing images.npy or masks.npy in {input_dir}")
        return

    images = np.load(images_file, mmap_mode='r')
    n_images = len(images)

    print(f"Found {n_images} images in PanNuke")

    for i in tqdm(range(n_images), desc="Converting"):
        try:
            image, inst_map, type_map, centroids = load_pannuke_annotation(
                images_file, masks_file, i
            )

            # Save to .npz
            output_file = output_dir / f"image_{i:05d}.npz"
            np.savez_compressed(
                output_file,
                image=image,
                inst_map=inst_map,
                type_map=type_map,
                centroids=centroids
            )

        except Exception as e:
            print(f"⚠️ Error processing image {i}: {e}")

    print(f"\n✅ Converted {n_images} images to: {output_dir}")


def convert_monusac(input_dir: Path, output_dir: Path) -> None:
    """Convert MoNuSAC dataset to unified format."""
    print("\n" + "="*70)
    print("CONVERTING MONUSAC TO UNIFIED FORMAT")
    print("="*70)

    print("⚠️ MoNuSAC conversion requires HuggingFace datasets library")
    print("This is a placeholder - implement if needed")

    # TODO: Implement MoNuSAC conversion
    # Will require loading from HuggingFace dataset format


def verify_conversion(npz_file: Path) -> None:
    """Verify a converted .npz file."""
    data = np.load(npz_file)

    required_keys = ["inst_map", "type_map"]
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing key '{key}' in {npz_file}")
            return

    inst_map = data["inst_map"]
    type_map = data["type_map"]

    # Check instance IDs start at 1
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    if len(inst_ids) > 0:
        if inst_ids.min() != 1:
            print(f"⚠️ Warning: Instance IDs don't start at 1 in {npz_file}")

    # Check type values are in valid range [0, 5]
    type_values = np.unique(type_map)
    if type_values.max() > 5:
        print(f"⚠️ Warning: Type values > 5 found in {npz_file}")

    print(f"✅ Valid: {npz_file.name}")
    print(f"   Instances: {len(inst_ids)}, Type range: [{type_values.min()}, {type_values.max()}]")


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset annotations to unified format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert CoNSeP
  python scripts/evaluation/convert_annotations.py \\
      --dataset consep \\
      --input_dir data/evaluation/consep/Test \\
      --output_dir data/evaluation/consep_converted

  # Convert PanNuke Fold 2
  python scripts/evaluation/convert_annotations.py \\
      --dataset pannuke \\
      --input_dir data/evaluation/pannuke/Fold\\ 2 \\
      --output_dir data/evaluation/pannuke_fold2_converted

  # Verify converted files
  python scripts/evaluation/convert_annotations.py \\
      --verify data/evaluation/consep_converted/image_001.npz
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["consep", "pannuke", "monusac"],
        help="Dataset type"
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Input directory with original annotations"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for converted .npz files"
    )

    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify a single .npz file"
    )

    args = parser.parse_args()

    if args.verify:
        verify_conversion(args.verify)
        return

    if not args.dataset or not args.input_dir or not args.output_dir:
        parser.print_help()
        print("\n❌ Error: --dataset, --input_dir, and --output_dir are required")
        return

    # Convert based on dataset type
    if args.dataset == "consep":
        convert_consep(args.input_dir, args.output_dir)
    elif args.dataset == "pannuke":
        convert_pannuke(args.input_dir, args.output_dir)
    elif args.dataset == "monusac":
        convert_monusac(args.input_dir, args.output_dir)

    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE")
    print("="*70)
    print(f"\nConverted annotations saved to: {args.output_dir.absolute()}")
    print("\nNext step:")
    print("  Run evaluation: python scripts/evaluation/evaluate_ground_truth.py")


if __name__ == "__main__":
    main()
