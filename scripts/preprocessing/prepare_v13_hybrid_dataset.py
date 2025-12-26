#!/usr/bin/env python3
"""
Prepare V13-Hybrid Dataset with Macenko Normalization + H-Channel Extraction.

Pipeline:
1. Load V13 Multi-Crop data
2. Macenko stain normalization (stabilize inter-slide coloration)
3. RGB → HED deconvolution (Ruifrok & Johnston)
4. Extract H-channel (Hematoxylin)
5. Save hybrid dataset with validation

Author: CellViT-Optimus Team
Date: 2025-12-26
"""

import argparse
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm

# HED deconvolution
from skimage.color import rgb2hed
import cv2


def extract_5_crops_from_256(
    image_256: np.ndarray,
    np_target_256: np.ndarray,
    hv_target_256: np.ndarray,
    nt_target_256: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 5 crops (224×224) from 256×256 images: center + 4 corners.

    This implements the V13 Multi-Crop strategy for 5× data augmentation.

    Args:
        image_256: (256, 256, 3) uint8
        np_target_256: (256, 256) float32
        hv_target_256: (2, 256, 256) float32
        nt_target_256: (256, 256) int64

    Returns:
        Tuple of 5 crops for each input:
        - images_crops: (5, 224, 224, 3) uint8
        - np_crops: (5, 224, 224) float32
        - hv_crops: (5, 2, 224, 224) float32
        - nt_crops: (5, 224, 224) int64
    """
    # Crop positions: [top_left_y, top_left_x]
    # From 256×256, we can extract 224×224 with offset up to 32 pixels
    positions = [
        (16, 16),   # Center crop
        (0, 0),     # Top-left corner
        (0, 32),    # Top-right corner
        (32, 0),    # Bottom-left corner
        (32, 32),   # Bottom-right corner
    ]

    images_crops = np.zeros((5, 224, 224, 3), dtype=np.uint8)
    np_crops = np.zeros((5, 224, 224), dtype=np.float32)
    hv_crops = np.zeros((5, 2, 224, 224), dtype=np.float32)
    nt_crops = np.zeros((5, 224, 224), dtype=np.int64)

    for i, (y, x) in enumerate(positions):
        # Extract 224×224 crop
        images_crops[i] = image_256[y:y+224, x:x+224, :]
        np_crops[i] = np_target_256[y:y+224, x:x+224]
        hv_crops[i] = hv_target_256[:, y:y+224, x:x+224]
        nt_crops[i] = nt_target_256[y:y+224, x:x+224]

    return images_crops, np_crops, hv_crops, nt_crops


class MacenkoNormalizer:
    """
    Macenko stain normalization implementation.

    Based on: "A method for normalizing histology slides for quantitative analysis"
    Macenko et al., 2009
    """

    def __init__(self):
        self.target_stains = None
        self.target_concentrations = None
        self.maxC_target = None

    def fit(self, target: np.ndarray):
        """Fit normalizer on target image."""
        self.target_stains = self._get_stain_matrix(target)
        target_concentrations = self._get_concentrations(target, self.target_stains)
        self.maxC_target = np.percentile(target_concentrations, 99, axis=1, keepdims=True)

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Normalize source image to match target."""
        if self.target_stains is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        source_stains = self._get_stain_matrix(source)
        source_concentrations = self._get_concentrations(source, source_stains)
        maxC_source = np.percentile(source_concentrations, 99, axis=1, keepdims=True)

        # Normalize concentrations
        source_concentrations *= (self.maxC_target / maxC_source)

        # Recreate image
        normalized = 255 * np.exp(-self.target_stains @ source_concentrations)
        normalized = normalized.T.reshape(source.shape).astype(np.uint8)

        return normalized

    @staticmethod
    def _get_stain_matrix(I: np.ndarray, beta=0.15, alpha=1):
        """Extract stain matrix using Macenko method."""
        h, w, c = I.shape
        I = I.reshape(-1, 3).astype(np.float32)

        # Optical density
        OD = -np.log((I + 1) / 256.0)

        # Remove transparent pixels
        ODhat = OD[~np.any(OD < beta, axis=1)]

        if ODhat.shape[0] < 2:
            # Fallback: return identity-like matrix
            return np.array([[0.65, 0.70], [0.07, 0.99]])

        # Compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        # Project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        That = ODhat @ eigvecs[:, 1:3]

        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        vMin = eigvecs[:, 1:3] @ np.array([np.cos(minPhi), np.sin(minPhi)])
        vMax = eigvecs[:, 1:3] @ np.array([np.cos(maxPhi), np.sin(maxPhi)])

        # A heuristic to make the vector corresponding to hematoxylin first
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T

        return HE

    @staticmethod
    def _get_concentrations(I: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Get concentrations of each stain."""
        h, w, c = I.shape
        I = I.reshape(-1, 3).astype(np.float32)

        # Optical density
        OD = -np.log((I + 1) / 256.0)

        # Concentrations
        C = np.linalg.lstsq(stain_matrix, OD.T, rcond=None)[0]

        return C

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.constants import PANNUKE_IMAGE_SIZE
from src.models.organ_families import ORGAN_TO_FAMILY


def normalize_macenko(image: np.ndarray, normalizer: MacenkoNormalizer) -> np.ndarray:
    """
    Apply Macenko stain normalization.

    Args:
        image: RGB image (H, W, 3) uint8 [0, 255]
        normalizer: MacenkoNormalizer fitted on reference image

    Returns:
        Normalized image (H, W, 3) uint8 [0, 255]
    """
    # ⚠️ BUG #1 PREVENTION: Ensure uint8
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)

    try:
        normalized = normalizer.transform(image)
        return normalized.astype(np.uint8)
    except Exception as e:
        warnings.warn(f"Macenko normalization failed: {e}. Using original image.")
        return image


def extract_h_channel(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract Hematoxylin channel via HED deconvolution.

    Args:
        image_rgb: RGB image (H, W, 3) uint8 [0, 255]

    Returns:
        h_channel: Hematoxylin channel (H, W) uint8 [0, 255]
    """
    # Convert to float [0, 1] for skimage
    image_float = image_rgb.astype(np.float32) / 255.0

    # HED deconvolution
    hed = rgb2hed(image_float)
    h_channel_float = hed[:, :, 0]  # Hematoxylin channel

    # Normalize to [0, 255] and convert to uint8
    h_min, h_max = h_channel_float.min(), h_channel_float.max()

    # Avoid division by zero
    if h_max - h_min < 1e-6:
        warnings.warn("H-channel has no contrast. Using zeros.")
        return np.zeros_like(h_channel_float, dtype=np.uint8)

    h_normalized = ((h_channel_float - h_min) / (h_max - h_min) * 255.0)
    h_uint8 = h_normalized.clip(0, 255).astype(np.uint8)

    return h_uint8


def validate_h_channel_quality(h_channels: np.ndarray) -> Dict[str, float]:
    """
    Validate H-channel quality statistics.

    Expected std range: [0.15, 0.35] (based on PanNuke empirical data)

    Args:
        h_channels: (N, H, W) uint8 array

    Returns:
        stats: {'mean_std': float, 'min_std': float, 'max_std': float}
    """
    stds = []
    for i in range(h_channels.shape[0]):
        h_std = h_channels[i].std() / 255.0  # Normalize to [0, 1]
        stds.append(h_std)

    stats = {
        'mean_std': np.mean(stds),
        'min_std': np.min(stds),
        'max_std': np.max(stds),
        'n_valid': np.sum((np.array(stds) >= 0.15) & (np.array(stds) <= 0.35)),
        'total': len(stds)
    }

    return stats


def prepare_hybrid_dataset(
    family: str,
    v13_data_file: Path,
    output_dir: Path,
    use_macenko: bool = True
) -> None:
    """
    Prepare hybrid dataset with H-channel extraction.

    Args:
        family: Family name (e.g., 'epidermal')
        v13_data_file: Path to V13 data .npz file
        output_dir: Output directory
        use_macenko: Whether to apply Macenko normalization
    """
    print(f"\n{'='*80}")
    print(f"PREPARING V13-HYBRID DATASET: {family.upper()}")
    print(f"{'='*80}\n")

    # Load V13 data
    print(f"📂 Loading V13 data: {v13_data_file}")
    data = np.load(v13_data_file)

    # Load data (may be 256×256 or 224×224 depending on source)
    images_raw = data['images']  # (N, H, W, 3) uint8
    np_targets_raw = data['np_targets']  # (N, H, W) float32
    hv_targets_raw = data['hv_targets']  # (N, 2, H, W) float32
    nt_targets_raw = data['nt_targets']  # (N, H, W) int64

    # Handle metadata (different key names in FIXED vs V13 files)
    if 'image_ids' in data.keys():
        source_image_ids = data['image_ids']
    elif 'source_image_ids' in data.keys():
        source_image_ids = data['source_image_ids']
    else:
        source_image_ids = np.arange(len(images_raw), dtype=np.int32)

    if 'fold_ids' in data.keys():
        fold_ids = data['fold_ids']
    else:
        fold_ids = np.zeros(len(images_raw), dtype=np.int32)

    n_samples = images_raw.shape[0]
    orig_size = images_raw.shape[1]
    print(f"  ✅ Loaded {n_samples} samples (original images)")
    print(f"  Original size: {orig_size}×{orig_size}")
    print(f"  Images: {images_raw.shape}, {images_raw.dtype}")
    print(f"  NP targets: {np_targets_raw.shape}, {np_targets_raw.dtype}")
    print(f"  HV targets: {hv_targets_raw.shape}, {hv_targets_raw.dtype}")
    print(f"  NT targets: {nt_targets_raw.shape}, {nt_targets_raw.dtype}")

    # Multi-crop: Extract 5 crops (224×224) from each 256×256 image
    if orig_size == 256:
        print(f"\n✂️  Extracting 5 crops per image (center + 4 corners)...")
        print(f"   Input: {n_samples} images → Output: {n_samples * 5} crops")

        # Allocate arrays for 5× augmented data
        n_crops = n_samples * 5
        images_224 = np.zeros((n_crops, 224, 224, 3), dtype=np.uint8)
        np_targets = np.zeros((n_crops, 224, 224), dtype=np.float32)
        hv_targets = np.zeros((n_crops, 2, 224, 224), dtype=np.float32)
        nt_targets = np.zeros((n_crops, 224, 224), dtype=np.int64)

        # Replicate metadata 5× with crop position markers
        crop_positions = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
        new_fold_ids = np.repeat(fold_ids, 5)
        new_source_image_ids = np.repeat(source_image_ids, 5)
        crop_position_ids = np.tile(np.arange(5, dtype=np.int32), n_samples)

        # Extract crops
        for i in range(n_samples):
            # Extract 5 crops from this image
            img_crops, np_crops, hv_crops, nt_crops = extract_5_crops_from_256(
                images_raw[i],
                np_targets_raw[i],
                hv_targets_raw[i],
                nt_targets_raw[i]
            )

            # Store in output arrays
            start_idx = i * 5
            end_idx = start_idx + 5
            images_224[start_idx:end_idx] = img_crops
            np_targets[start_idx:end_idx] = np_crops
            hv_targets[start_idx:end_idx] = hv_crops
            nt_targets[start_idx:end_idx] = nt_crops

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{n_samples} images → {(i+1)*5}/{n_crops} crops")

        print(f"  ✅ Multi-crop complete: {n_crops} total crops")

        # Update metadata
        fold_ids = new_fold_ids
        source_image_ids = new_source_image_ids
        n_final = n_crops  # Track final number for H-channel extraction

    elif orig_size == 224:
        # Already correct size, no cropping needed
        print(f"\n✅ Images already 224×224, no cropping needed")
        images_224 = images_raw
        np_targets = np_targets_raw
        hv_targets = hv_targets_raw
        nt_targets = nt_targets_raw
        crop_position_ids = np.zeros(n_samples, dtype=np.int32)  # All center
        n_final = n_samples  # Track final number for H-channel extraction
    else:
        raise ValueError(f"Unsupported image size: {orig_size}×{orig_size}. Expected 256×256 or 224×224.")

    # ⚠️ BUG #3 PREVENTION: Validate HV targets
    print("\n🔍 Validating HV targets...")
    if hv_targets.dtype != np.float32:
        raise ValueError(
            f"❌ HV dtype is {hv_targets.dtype} instead of float32! "
            f"This causes MSE ~4681 instead of ~0.01. "
            f"Regenerate data with correct dtype."
        )

    hv_min, hv_max = hv_targets.min(), hv_targets.max()
    if not (-1.0 <= hv_min <= hv_max <= 1.0):
        raise ValueError(
            f"❌ HV range [{hv_min:.3f}, {hv_max:.3f}] is not in [-1, 1]!"
        )

    print(f"  ✅ HV dtype: {hv_targets.dtype}")
    print(f"  ✅ HV range: [{hv_min:.4f}, {hv_max:.4f}]")

    # Initialize Macenko normalizer
    normalizer = None
    if use_macenko:
        print("\n🎨 Initializing Macenko normalizer...")
        try:
            normalizer = MacenkoNormalizer()
            # Fit on first image as reference
            ref_image = images_224[0]
            normalizer.fit(ref_image)
            print("  ✅ Macenko normalizer fitted")
        except Exception as e:
            warnings.warn(f"Macenko fitting failed: {e}. Skipping normalization.")
            normalizer = None

    # Extract H-channels
    print(f"\n🔬 Extracting H-channels from {n_final} samples...")
    h_channels_224 = np.zeros((n_final, 224, 224), dtype=np.uint8)

    for i in tqdm(range(n_final), desc="Processing samples"):
        image = images_224[i]

        # Macenko normalization
        if normalizer is not None:
            image = normalize_macenko(image, normalizer)

        # Extract H-channel
        h_channel = extract_h_channel(image)
        h_channels_224[i] = h_channel

    print(f"  ✅ H-channels extracted: {h_channels_224.shape}, {h_channels_224.dtype}")

    # Validate H-channel quality
    print("\n📊 Validating H-channel quality...")
    h_stats = validate_h_channel_quality(h_channels_224)

    print(f"  H-channel std (normalized [0, 1]):")
    print(f"    Mean: {h_stats['mean_std']:.4f}")
    print(f"    Range: [{h_stats['min_std']:.4f}, {h_stats['max_std']:.4f}]")
    print(f"    Valid samples (std ∈ [0.15, 0.35]): {h_stats['n_valid']}/{h_stats['total']} ({h_stats['n_valid']/h_stats['total']*100:.1f}%)")

    if h_stats['n_valid'] / h_stats['total'] < 0.80:
        warnings.warn(
            f"⚠️  Only {h_stats['n_valid']/h_stats['total']*100:.1f}% samples have valid H-channel std. "
            f"Expected >80%. Check stain normalization."
        )
    else:
        print(f"  ✅ H-channel quality OK ({h_stats['n_valid']/h_stats['total']*100:.1f}% valid)")

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{family}_data_v13_hybrid.npz"

    # Final validation before save
    print("\n🔍 Final validation before save...")
    print(f"  Images 224: {images_224.shape}, {images_224.dtype}")
    print(f"  H-channels 224: {h_channels_224.shape}, {h_channels_224.dtype}")
    print(f"  NP targets: {np_targets.shape}, {np_targets.dtype}, range [{np_targets.min():.3f}, {np_targets.max():.3f}]")
    print(f"  HV targets: {hv_targets.shape}, {hv_targets.dtype}, range [{hv_targets.min():.3f}, {hv_targets.max():.3f}]")
    print(f"  NT targets: {nt_targets.shape}, {nt_targets.dtype}, unique {len(np.unique(nt_targets))} classes")
    print(f"  Source IDs: {source_image_ids.shape}, {source_image_ids.dtype}")
    print(f"  Fold IDs: {fold_ids.shape}, {fold_ids.dtype}")

    # Save
    print(f"\n💾 Saving to: {output_file}")
    np.savez_compressed(
        output_file,
        images_224=images_224,
        h_channels_224=h_channels_224,
        np_targets=np_targets,
        hv_targets=hv_targets,
        nt_targets=nt_targets,
        source_image_ids=source_image_ids,
        fold_ids=fold_ids,
        crop_position_ids=crop_position_ids,  # 0=center, 1=TL, 2=TR, 3=BL, 4=BR
        # Metadata
        macenko_applied=use_macenko and normalizer is not None,
        h_channel_std_mean=h_stats['mean_std'],
        h_channel_std_range=(h_stats['min_std'], h_stats['max_std'])
    )

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"  ✅ Saved: {file_size_mb:.2f} MB")

    print(f"\n{'='*80}")
    print(f"✅ V13-HYBRID DATASET PREPARATION COMPLETE: {family.upper()}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare V13-Hybrid dataset with H-channel extraction")
    parser.add_argument('--family', type=str, default='epidermal',
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
                        help='Family to process')
    parser.add_argument('--source_data_dir', type=Path, default=Path('data/family_FIXED'),
                        help='Directory containing source family data (FIXED version)')
    parser.add_argument('--output_dir', type=Path, default=Path('data/family_data_v13_hybrid'),
                        help='Output directory for hybrid data')
    parser.add_argument('--no_macenko', action='store_true',
                        help='Disable Macenko normalization')

    args = parser.parse_args()

    # Locate source data file
    v13_data_file = args.source_data_dir / f"{args.family}_data_FIXED.npz"

    if not v13_data_file.exists():
        raise FileNotFoundError(f"Source data file not found: {v13_data_file}")

    # Prepare hybrid dataset
    prepare_hybrid_dataset(
        family=args.family,
        v13_data_file=v13_data_file,
        output_dir=args.output_dir,
        use_macenko=not args.no_macenko
    )


if __name__ == '__main__':
    main()
