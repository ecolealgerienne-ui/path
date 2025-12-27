#!/usr/bin/env python3
"""
V13 Smart Crops Data Preparation
=================================

Strategy (CTO validated):
1. Split FIRST by patient (80/20) to prevent data leakage
2. Apply 5 strategic crops to EACH dataset separately:
   - Centre crop 224×224 → Rotation 0°
   - Coin Haut-Gauche → Rotation 90°
   - Coin Haut-Droit → Rotation 180°
   - Coin Bas-Gauche → Rotation 270°
   - Coin Bas-Droit → Flip horizontal

This ensures NO rotated version of train image appears in val.

Requirements:
    pip install albumentations

Usage:
    python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import albumentations as A
from typing import Tuple, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE
from src.data.preprocessing import validate_targets, TargetFormat


# ============================================================================
# STRATEGIC CROPS CONFIGURATION
# ============================================================================

def get_crop_configs() -> List[Dict]:
    """
    Returns 5 strategic crop configurations.

    Each config specifies:
    - name: Descriptive name
    - crop_coords: (y_start, x_start) for 224×224 crop in 256×256 image
    - rotation: Albumentations transform to apply AFTER crop

    Returns:
        List of 5 crop configurations
    """
    configs = [
        {
            'name': 'centre_0deg',
            'crop_coords': (16, 16),  # Centre: (256-224)/2 = 16
            'transform': A.NoOp()  # No rotation
        },
        {
            'name': 'top_left_90deg',
            'crop_coords': (0, 0),  # Coin Haut-Gauche
            'transform': A.Rotate(limit=(90, 90), p=1.0, border_mode=0)
        },
        {
            'name': 'top_right_180deg',
            'crop_coords': (0, 32),  # Coin Haut-Droit (256-224=32)
            'transform': A.Rotate(limit=(180, 180), p=1.0, border_mode=0)
        },
        {
            'name': 'bottom_left_270deg',
            'crop_coords': (32, 0),  # Coin Bas-Gauche
            'transform': A.Rotate(limit=(270, 270), p=1.0, border_mode=0)
        },
        {
            'name': 'bottom_right_hflip',
            'crop_coords': (32, 32),  # Coin Bas-Droit
            'transform': A.HorizontalFlip(p=1.0)
        }
    ]

    return configs


# ============================================================================
# HV MAPS ROTATION HANDLING
# ============================================================================

def correct_hv_after_rotation(hv_map: np.ndarray, rotation_angle: int) -> np.ndarray:
    """
    Corrects HV component swapping after rotation.

    HV maps are vector fields encoding (H, V) distance to nucleus center.
    Rotation requires component transformation:

    - 90° clockwise:  H' = V,  V' = -H
    - 180°:           H' = -H, V' = -V
    - 270° clockwise: H' = -V, V' = H

    Args:
        hv_map: (2, H, W) or (H, W, 2) array in range [-1, 1]
        rotation_angle: 0, 90, 180, or 270

    Returns:
        Corrected HV map with same shape
    """
    if rotation_angle == 0:
        return hv_map

    # Handle both (2, H, W) and (H, W, 2) formats
    if hv_map.shape[0] == 2:
        h_comp = hv_map[0]
        v_comp = hv_map[1]
        axis = 0
    else:
        h_comp = hv_map[:, :, 0]
        v_comp = hv_map[:, :, 1]
        axis = 2

    if rotation_angle == 90:
        new_h = v_comp
        new_v = -h_comp
    elif rotation_angle == 180:
        new_h = -h_comp
        new_v = -v_comp
    elif rotation_angle == 270:
        new_h = -v_comp
        new_v = h_comp
    else:
        raise ValueError(f"Invalid rotation angle: {rotation_angle}")

    if axis == 0:
        return np.stack([new_h, new_v], axis=0)
    else:
        return np.stack([new_h, new_v], axis=2)


def correct_hv_after_hflip(hv_map: np.ndarray) -> np.ndarray:
    """
    Corrects HV maps after horizontal flip.

    Horizontal flip inverts H component (left↔right) but NOT V component.

    Args:
        hv_map: (2, H, W) or (H, W, 2) array in range [-1, 1]

    Returns:
        Corrected HV map
    """
    if hv_map.shape[0] == 2:
        h_comp = hv_map[0]
        v_comp = hv_map[1]
        return np.stack([-h_comp, v_comp], axis=0)
    else:
        h_comp = hv_map[:, :, 0]
        v_comp = hv_map[:, :, 1]
        return np.stack([-h_comp, v_comp], axis=2)


# ============================================================================
# CROP + ROTATION APPLICATION
# ============================================================================

def apply_strategic_crop(
    image: np.ndarray,
    np_target: np.ndarray,
    hv_target: np.ndarray,
    nt_target: np.ndarray,
    crop_config: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies crop + rotation to image and all targets synchronously.

    Args:
        image: (256, 256, 3) RGB uint8
        np_target: (256, 256) binary uint8
        hv_target: (2, 256, 256) float32 [-1, 1]
        nt_target: (256, 256) int64
        crop_config: Dict with 'crop_coords', 'transform', 'name'

    Returns:
        (image_224, np_224, hv_224, nt_224) all at 224×224
    """
    y_start, x_start = crop_config['crop_coords']

    # Step 1: Crop all targets to 224×224
    image_crop = image[y_start:y_start+224, x_start:x_start+224]
    np_crop = np_target[y_start:y_start+224, x_start:x_start+224]
    nt_crop = nt_target[y_start:y_start+224, x_start:x_start+224]

    # HV needs special handling (convert to H, W, 2 for Albumentations)
    hv_crop = hv_target[:, y_start:y_start+224, x_start:x_start+224]
    hv_crop = np.transpose(hv_crop, (1, 2, 0))  # (2, 224, 224) → (224, 224, 2)

    # Step 2: Apply rotation using Albumentations
    transform = A.Compose([
        crop_config['transform']
    ], additional_targets={
        'mask_np': 'mask',
        'mask_hv': 'image',  # Treat as image to preserve float values
        'mask_nt': 'mask'
    })

    transformed = transform(
        image=image_crop,
        mask_np=np_crop,
        mask_hv=hv_crop,
        mask_nt=nt_crop
    )

    image_rot = transformed['image']
    np_rot = transformed['mask_np']
    hv_rot = transformed['mask_hv']  # Still (224, 224, 2)
    nt_rot = transformed['mask_nt']

    # Step 3: Correct HV component swapping
    if 'deg' in crop_config['name']:
        # Extract rotation angle from name (e.g., "90deg")
        angle_str = crop_config['name'].split('_')[-1].replace('deg', '')
        rotation_angle = int(angle_str)
        hv_rot = correct_hv_after_rotation(hv_rot, rotation_angle)
    elif 'hflip' in crop_config['name']:
        hv_rot = correct_hv_after_hflip(hv_rot)

    # Convert HV back to (2, 224, 224)
    hv_rot = np.transpose(hv_rot, (2, 0, 1))

    return image_rot, np_rot, hv_rot, nt_rot


# ============================================================================
# DATASET SPLIT + AMPLIFICATION
# ============================================================================

def split_by_patient(
    images: np.ndarray,
    masks: np.ndarray,
    source_image_ids: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    Splits dataset by patient (source image ID) to prevent data leakage.

    Args:
        images: (N, 256, 256, 3)
        masks: (N, 256, 256, 6)
        source_image_ids: (N,) array of source image indices
        train_ratio: Fraction for train split (default 0.8)
        seed: Random seed for reproducibility

    Returns:
        train_data: {'images': ..., 'masks': ..., 'source_ids': ...}
        val_data: {'images': ..., 'masks': ..., 'source_ids': ...}
    """
    unique_source_ids = np.unique(source_image_ids)
    n_total_unique = len(unique_source_ids)
    n_train_unique = int(n_total_unique * train_ratio)

    # Shuffle source IDs
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(unique_source_ids)

    train_source_ids = shuffled_ids[:n_train_unique]
    val_source_ids = shuffled_ids[n_train_unique:]

    # Create masks for train/val
    train_mask = np.isin(source_image_ids, train_source_ids)
    val_mask = np.isin(source_image_ids, val_source_ids)

    train_data = {
        'images': images[train_mask],
        'masks': masks[train_mask],
        'source_ids': source_image_ids[train_mask]
    }

    val_data = {
        'images': images[val_mask],
        'masks': masks[val_mask],
        'source_ids': source_image_ids[val_mask]
    }

    return train_data, val_data


def amplify_with_crops(
    images: np.ndarray,
    np_targets: np.ndarray,
    hv_targets: np.ndarray,
    nt_targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies 5 strategic crops to dataset (amplifies by 5×).

    Args:
        images: (N, 256, 256, 3)
        np_targets: (N, 256, 256)
        hv_targets: (N, 2, 256, 256)
        nt_targets: (N, 256, 256)

    Returns:
        (images_224, np_224, hv_224, nt_224) with shape (N×5, 224, 224, ...)
    """
    crop_configs = get_crop_configs()
    n_samples = len(images)
    n_crops = len(crop_configs)

    # Preallocate output arrays
    images_out = np.zeros((n_samples * n_crops, 224, 224, 3), dtype=np.uint8)
    np_out = np.zeros((n_samples * n_crops, 224, 224), dtype=np.uint8)
    hv_out = np.zeros((n_samples * n_crops, 2, 224, 224), dtype=np.float32)
    nt_out = np.zeros((n_samples * n_crops, 224, 224), dtype=np.int64)

    idx = 0
    for i in range(n_samples):
        for crop_config in crop_configs:
            img_crop, np_crop, hv_crop, nt_crop = apply_strategic_crop(
                images[i],
                np_targets[i],
                hv_targets[i],
                nt_targets[i],
                crop_config
            )

            images_out[idx] = img_crop
            np_out[idx] = np_crop
            hv_out[idx] = hv_crop
            nt_out[idx] = nt_crop
            idx += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")

    return images_out, np_out, hv_out, nt_out


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="V13 Smart Crops Data Preparation")
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--source_data_dir', type=Path, default=Path('data/family_FIXED'))
    parser.add_argument('--output_dir', type=Path, default=Path('data/family_data_v13_smart_crops'))
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # ========================================================================
    # STEP 1: Load source data
    # ========================================================================

    source_file = args.source_data_dir / f"{args.family}_data_FIXED.npz"

    if not source_file.exists():
        raise FileNotFoundError(
            f"Source data not found: {source_file}\n"
            f"Run prepare_family_data_FIXED.py first."
        )

    print(f"Loading source data: {source_file}")
    data = np.load(source_file)

    images_256 = data['images_224']  # Actually 256×256 before crop
    np_targets_256 = data['np_targets']
    hv_targets_256 = data['hv_targets']
    nt_targets_256 = data['nt_targets']
    source_image_ids = data.get('source_image_ids', np.arange(len(images_256)))

    n_total = len(images_256)
    print(f"Loaded {n_total} samples for family '{args.family}'")

    # Validate HV targets (prevent Bug #3)
    print("Validating HV targets...")
    try:
        validate_targets(
            np_targets_256[0],
            hv_targets_256[0],
            nt_targets_256[0],
            strict=True
        )
        print("✅ HV targets validated (float32, range [-1, 1])")
    except ValueError as e:
        raise ValueError(f"HV validation failed: {e}")

    # ========================================================================
    # STEP 2: Split by patient FIRST (80/20)
    # ========================================================================

    print(f"\nSplitting by patient ({args.train_ratio:.0%} train / {1-args.train_ratio:.0%} val)...")
    train_data, val_data = split_by_patient(
        images_256,
        np.stack([np_targets_256, hv_targets_256.transpose(0, 2, 3, 1), nt_targets_256], axis=-1),  # Temporary stack
        source_image_ids,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    # Unpack masks
    train_images = train_data['images']
    train_np = train_data['masks'][:, :, :, 0]
    train_hv = train_data['masks'][:, :, :, 1:3].transpose(0, 3, 1, 2)  # Back to (N, 2, 256, 256)
    train_nt = train_data['masks'][:, :, :, 3].astype(np.int64)

    val_images = val_data['images']
    val_np = val_data['masks'][:, :, :, 0]
    val_hv = val_data['masks'][:, :, :, 1:3].transpose(0, 3, 1, 2)
    val_nt = val_data['masks'][:, :, :, 3].astype(np.int64)

    print(f"  Train: {len(train_images)} samples")
    print(f"  Val:   {len(val_images)} samples")

    # ========================================================================
    # STEP 3: Apply 5 strategic crops to EACH dataset separately
    # ========================================================================

    print("\nApplying 5 strategic crops to TRAIN dataset...")
    train_images_224, train_np_224, train_hv_224, train_nt_224 = amplify_with_crops(
        train_images, train_np, train_hv, train_nt
    )
    print(f"  Train amplified: {len(train_images_224)} crops")

    print("\nApplying 5 strategic crops to VAL dataset...")
    val_images_224, val_np_224, val_hv_224, val_nt_224 = amplify_with_crops(
        val_images, val_np, val_hv, val_nt
    )
    print(f"  Val amplified: {len(val_images_224)} crops")

    # ========================================================================
    # STEP 4: Save train and val datasets
    # ========================================================================

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_file = args.output_dir / f"{args.family}_train_v13_smart_crops.npz"
    val_file = args.output_dir / f"{args.family}_val_v13_smart_crops.npz"

    print(f"\nSaving train dataset: {train_file}")
    np.savez_compressed(
        train_file,
        images_224=train_images_224,
        np_targets=train_np_224,
        hv_targets=train_hv_224,
        nt_targets=train_nt_224,
        metadata={
            'family': args.family,
            'split': 'train',
            'n_samples': len(train_images_224),
            'amplification': '5x strategic crops',
            'seed': args.seed
        }
    )

    print(f"Saving val dataset: {val_file}")
    np.savez_compressed(
        val_file,
        images_224=val_images_224,
        np_targets=val_np_224,
        hv_targets=val_hv_224,
        nt_targets=val_nt_224,
        metadata={
            'family': args.family,
            'split': 'val',
            'n_samples': len(val_images_224),
            'amplification': '5x strategic crops',
            'seed': args.seed
        }
    )

    print("\n" + "="*70)
    print("✅ V13 SMART CROPS DATA PREPARATION COMPLETE")
    print("="*70)
    print(f"Family:       {args.family}")
    print(f"Train:        {len(train_images_224)} crops (from {len(train_images)} sources)")
    print(f"Val:          {len(val_images_224)} crops (from {len(val_images)} sources)")
    print(f"Amplification: 5× (centre + 4 corners with rotations)")
    print(f"Data leakage: PREVENTED (split-first-then-rotate)")
    print("="*70)


if __name__ == '__main__':
    main()
