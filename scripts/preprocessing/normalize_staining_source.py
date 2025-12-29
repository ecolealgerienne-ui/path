#!/usr/bin/env python3
"""
Macenko Stain Normalization for PanNuke Images.

This script normalizes the color variability of H&E stained images using
the Macenko method. It should be run BEFORE prepare_v13_smart_crops.py.

Key Benefits:
1. Stabilizes inter-slide coloration variability
2. Reduces eosin "bleeding" on nucleus edges → cleaner AJI
3. Prepares for H-channel extraction (Ruifrok) in the next phase
4. Zero modification to downstream pipeline

Algorithm (Macenko 2009):
1. RGB → Optical Density: OD = -log10(I/I0)
2. Filter transparent pixels (OD < threshold)
3. SVD to find stain vectors (Hematoxylin/Eosin axes)
4. Map source concentrations to target (reference image)

Usage:
    python scripts/preprocessing/normalize_staining_source.py --family epidermal

    # With custom reference image:
    python scripts/preprocessing/normalize_staining_source.py \
        --family epidermal \
        --reference_idx 42

    # Save to new folder instead of overwriting:
    python scripts/preprocessing/normalize_staining_source.py \
        --family epidermal \
        --output_suffix _normalized

Author: CellViT-Optimus Team
Date: 2025-12-28
"""

import argparse
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MacenkoNormalizer:
    """
    Macenko stain normalization implementation.

    Based on: "A method for normalizing histology slides for quantitative analysis"
    Macenko et al., ISBI 2009

    The method:
    1. Converts RGB to Optical Density (OD) space
    2. Uses SVD to find the principal stain vectors
    3. Maps source concentrations to match target (reference) image
    """

    def __init__(self):
        self.target_stains = None
        self.target_concentrations = None
        self.maxC_target = None

    def fit(self, target: np.ndarray):
        """
        Fit normalizer on target (reference) image.

        Args:
            target: Reference image (H, W, 3) uint8, representing the
                   "Gold Standard" coloration we want all images to match.
        """
        self.target_stains = self._get_stain_matrix(target)
        target_concentrations = self._get_concentrations(target, self.target_stains)
        self.maxC_target = np.percentile(target_concentrations, 99, axis=1, keepdims=True)

    def transform(self, source: np.ndarray) -> np.ndarray:
        """
        Normalize source image to match target colorimetry.

        Args:
            source: Image to normalize (H, W, 3) uint8

        Returns:
            Normalized image (H, W, 3) uint8
        """
        if self.target_stains is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        source_stains = self._get_stain_matrix(source)
        source_concentrations = self._get_concentrations(source, source_stains)
        maxC_source = np.percentile(source_concentrations, 99, axis=1, keepdims=True)

        # Avoid division by zero
        maxC_source = np.maximum(maxC_source, 1e-6)

        # Normalize concentrations to match target
        source_concentrations *= (self.maxC_target / maxC_source)

        # Recreate image from normalized concentrations
        normalized = 255 * np.exp(-self.target_stains @ source_concentrations)
        normalized = np.clip(normalized.T.reshape(source.shape), 0, 255).astype(np.uint8)

        return normalized

    @staticmethod
    def _get_stain_matrix(I: np.ndarray, beta: float = 0.15, alpha: float = 1.0) -> np.ndarray:
        """
        Extract stain matrix using Macenko method.

        Args:
            I: RGB image (H, W, 3)
            beta: OD threshold for filtering transparent pixels
            alpha: Percentile for angle calculation

        Returns:
            Stain matrix (3, 2) representing H and E vectors
        """
        h, w, c = I.shape
        I = I.reshape(-1, 3).astype(np.float32)

        # RGB → Optical Density
        # OD = -log10(I/I0), with I0=255 (white point)
        OD = -np.log((I + 1) / 256.0)

        # Filter transparent pixels (low OD = no tissue)
        ODhat = OD[~np.any(OD < beta, axis=1)]

        if ODhat.shape[0] < 10:
            # Fallback for images with very little tissue
            # Standard H&E matrix from literature
            return np.array([[0.65, 0.70, 0.29],
                            [0.07, 0.99, 0.11]]).T

        # Compute eigenvectors of covariance matrix
        try:
            _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        except np.linalg.LinAlgError:
            return np.array([[0.65, 0.70, 0.29],
                            [0.07, 0.99, 0.11]]).T

        # Project on plane spanned by two largest eigenvectors
        That = ODhat @ eigvecs[:, 1:3]

        # Find angles
        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        # Get stain vectors
        vMin = eigvecs[:, 1:3] @ np.array([np.cos(minPhi), np.sin(minPhi)])
        vMax = eigvecs[:, 1:3] @ np.array([np.cos(maxPhi), np.sin(maxPhi)])

        # Heuristic: Hematoxylin has higher first component
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T

        return HE

    @staticmethod
    def _get_concentrations(I: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """
        Get concentration of each stain at each pixel.

        Args:
            I: RGB image (H, W, 3)
            stain_matrix: Stain vectors (3, 2)

        Returns:
            Concentrations (2, H*W)
        """
        h, w, c = I.shape
        I = I.reshape(-1, 3).astype(np.float32)

        # RGB → Optical Density
        OD = -np.log((I + 1) / 256.0)

        # Solve for concentrations: OD = stain_matrix @ C
        C = np.linalg.lstsq(stain_matrix, OD.T, rcond=None)[0]

        return C


def select_reference_image(images: np.ndarray, method: str = "balanced") -> Tuple[int, np.ndarray]:
    """
    Automatically select a reference image for Macenko normalization.

    Args:
        images: Array of images (N, H, W, 3) uint8
        method: Selection method:
            - "balanced": Select image with best H/E balance (default)
            - "median": Select image closest to median intensity
            - "first": Use first image

    Returns:
        (index, image): Selected reference image
    """
    if method == "first":
        return 0, images[0]

    if method == "median":
        # Select image closest to median intensity
        intensities = images.mean(axis=(1, 2, 3))
        median_idx = np.argmin(np.abs(intensities - np.median(intensities)))
        return int(median_idx), images[median_idx]

    if method == "balanced":
        # Select image with best H/E balance
        # Good H&E: Purple nuclei (low R, medium G, high B) + Pink tissue (high R, medium G, low B)
        # We look for images with good violet-pink contrast

        best_idx = 0
        best_score = -np.inf

        for i, img in enumerate(images):
            # Convert to float
            r, g, b = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)

            # Detect nuclei (high blue, low red relative to blue)
            nuclei_mask = (b > r) & (b > 100)
            nuclei_count = nuclei_mask.sum()

            # Detect eosin/tissue (high red, low blue)
            tissue_mask = (r > b) & (r > 100)
            tissue_count = tissue_mask.sum()

            # Good images have both nuclei and tissue
            if nuclei_count > 100 and tissue_count > 100:
                # Score based on balance and contrast
                balance = min(nuclei_count, tissue_count) / max(nuclei_count, tissue_count)

                # Contrast: difference between nuclei blue and tissue red
                nuclei_blue = b[nuclei_mask].mean() if nuclei_count > 0 else 0
                tissue_red = r[tissue_mask].mean() if tissue_count > 0 else 0
                contrast = abs(nuclei_blue - tissue_red)

                score = balance * contrast

                if score > best_score:
                    best_score = score
                    best_idx = i

        return best_idx, images[best_idx]

    raise ValueError(f"Unknown method: {method}")


def normalize_images(
    images: np.ndarray,
    reference_idx: Optional[int] = None,
    selection_method: str = "balanced"
) -> Tuple[np.ndarray, int]:
    """
    Normalize all images using Macenko method.

    Args:
        images: Array of images (N, H, W, 3) uint8
        reference_idx: Index of reference image (auto-select if None)
        selection_method: Method for auto-selection

    Returns:
        (normalized_images, reference_idx)
    """
    n_images = len(images)

    # Select reference image
    if reference_idx is None:
        reference_idx, reference_img = select_reference_image(images, method=selection_method)
        print(f"  Auto-selected reference image: index {reference_idx}")
    else:
        if reference_idx < 0 or reference_idx >= n_images:
            raise ValueError(f"reference_idx {reference_idx} out of range [0, {n_images})")
        reference_img = images[reference_idx]
        print(f"  Using specified reference image: index {reference_idx}")

    # Fit normalizer on reference
    normalizer = MacenkoNormalizer()
    normalizer.fit(reference_img)

    # Normalize all images
    normalized = np.zeros_like(images)
    failed_count = 0

    for i in tqdm(range(n_images), desc="  Normalizing"):
        try:
            normalized[i] = normalizer.transform(images[i])
        except Exception as e:
            # Keep original if normalization fails
            normalized[i] = images[i]
            failed_count += 1
            if failed_count <= 5:
                warnings.warn(f"Image {i} normalization failed: {e}")

    if failed_count > 0:
        print(f"  Warning: {failed_count}/{n_images} images failed normalization (kept original)")

    return normalized, reference_idx


def main():
    parser = argparse.ArgumentParser(
        description="Apply Macenko stain normalization to family images"
    )
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"],
        help="Family to normalize"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/family_FIXED"),
        help="Directory containing family_data_FIXED.npz files"
    )
    parser.add_argument(
        "--reference_idx",
        type=int,
        default=None,
        help="Index of reference image (auto-select if not specified)"
    )
    parser.add_argument(
        "--selection_method",
        type=str,
        default="balanced",
        choices=["balanced", "median", "first"],
        help="Method for auto-selecting reference image"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix for output file (empty = overwrite original)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without modifying files"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MACENKO STAIN NORMALIZATION")
    print("=" * 80)
    print(f"Family:           {args.family}")
    print(f"Data directory:   {args.data_dir}")
    print(f"Reference image:  {'auto-select' if args.reference_idx is None else args.reference_idx}")
    print(f"Selection method: {args.selection_method}")
    print(f"Output suffix:    {args.output_suffix or '(overwrite original)'}")
    print(f"Dry run:          {args.dry_run}")
    print()

    # Load source data
    input_file = args.data_dir / f"{args.family}_data_FIXED.npz"

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run prepare_family_data_FIXED.py first.")
        return 1

    print(f"Loading {input_file}...")
    data = np.load(input_file, allow_pickle=True)

    # Get images
    images = data["images"]  # (N, 256, 256, 3) uint8
    n_images = len(images)
    print(f"  Loaded {n_images} images, shape: {images.shape}, dtype: {images.dtype}")

    # Ensure uint8
    if images.dtype != np.uint8:
        print(f"  Converting images from {images.dtype} to uint8...")
        images = np.clip(images, 0, 255).astype(np.uint8)

    if args.dry_run:
        print("\n[DRY RUN] Would normalize images and save to:")
        output_file = args.data_dir / f"{args.family}_data_FIXED{args.output_suffix}.npz"
        print(f"  {output_file}")
        return 0

    # Normalize images
    print("\nApplying Macenko normalization...")
    normalized_images, ref_idx = normalize_images(
        images,
        reference_idx=args.reference_idx,
        selection_method=args.selection_method
    )

    # Compute stats before/after
    print("\nNormalization statistics:")
    print(f"  Reference image index: {ref_idx}")

    mean_before = images.mean(axis=(1, 2, 3))
    mean_after = normalized_images.mean(axis=(1, 2, 3))
    std_before = mean_before.std()
    std_after = mean_after.std()

    print(f"  Mean intensity std (before): {std_before:.2f}")
    print(f"  Mean intensity std (after):  {std_after:.2f}")
    print(f"  Variability reduction:       {100 * (1 - std_after / std_before):.1f}%")

    # Prepare output data
    output_data = dict(data)  # Copy all fields
    output_data["images"] = normalized_images
    output_data["macenko_reference_idx"] = ref_idx

    # Save
    output_file = args.data_dir / f"{args.family}_data_FIXED{args.output_suffix}.npz"

    print(f"\nSaving to {output_file}...")
    np.savez_compressed(output_file, **output_data)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved: {file_size_mb:.1f} MB")

    print("\n" + "=" * 80)
    print("NORMALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Run prepare_v13_smart_crops.py on normalized data:")
    print(f"  python scripts/preprocessing/prepare_v13_smart_crops.py --family {args.family}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
