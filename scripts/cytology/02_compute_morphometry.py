"""
Morphometry Features Computation — V14 Cytology

Ce script calcule les 20 features morphométriques pour le pipeline V14:
1. Charge embeddings + masques GT depuis l'étape 01
2. Calcule features géométriques (area, perimeter, circularity, etc.)
3. Calcule features d'intensité (mean, std sur H-channel Ruifrok)
4. Calcule features de texture (Haralick)
5. Fusionne avec CLS tokens et sauvegarde

Author: V14 Cytology Branch
Date: 2026-01-20
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
import cv2


# =============================================================================
#  RUIFROK STAIN DECONVOLUTION (H-channel extraction)
# =============================================================================

# Ruifrok stain vectors (Beer-Lambert law)
RUIFROK_STAIN_MATRIX = np.array([
    [0.650, 0.704, 0.286],  # Hematoxylin
    [0.268, 0.570, 0.776],  # Eosin
    [0.578, 0.421, 0.698]   # DAB (residual)
])


def extract_h_channel(image: np.ndarray) -> np.ndarray:
    """
    Extrait le canal Hématoxyline via déconvolution Ruifrok

    Args:
        image: RGB image (H, W, 3), uint8 [0, 255]

    Returns:
        h_channel: Hematoxylin channel (H, W), float [0, 1]
    """
    # Convert to optical density (OD)
    image_float = image.astype(np.float32) / 255.0
    image_float = np.clip(image_float, 1e-6, 1.0)  # Avoid log(0)
    od = -np.log(image_float)

    # Reshape for matrix multiplication
    od_flat = od.reshape(-1, 3)

    # Inverse of stain matrix
    stain_matrix_inv = np.linalg.inv(RUIFROK_STAIN_MATRIX)

    # Deconvolve
    stain_concentrations = od_flat @ stain_matrix_inv.T

    # Extract H channel (first column)
    h_channel = stain_concentrations[:, 0].reshape(image.shape[:2])

    # Normalize to [0, 1]
    h_channel = np.clip(h_channel, 0, None)
    if h_channel.max() > 0:
        h_channel = h_channel / h_channel.max()

    return h_channel.astype(np.float32)


# =============================================================================
#  GEOMETRIC FEATURES
# =============================================================================

def compute_geometric_features(mask: np.ndarray) -> Dict[str, float]:
    """
    Calcule les features géométriques à partir d'un masque binaire

    Args:
        mask: Binary mask (H, W), values 0 or 1

    Returns:
        features: Dict with geometric features
    """
    features = {
        'area': 0.0,
        'perimeter': 0.0,
        'circularity': 0.0,
        'eccentricity': 0.0,
        'solidity': 0.0,
        'extent': 0.0,
        'major_axis': 0.0,
        'minor_axis': 0.0,
        'orientation': 0.0,
        'equivalent_diameter': 0.0,
    }

    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return features

    # Use largest contour
    contour = max(contours, key=cv2.contourArea)

    # Area
    area = cv2.contourArea(contour)
    features['area'] = area

    if area < 10:  # Too small
        return features

    # Perimeter
    perimeter = cv2.arcLength(contour, closed=True)
    features['perimeter'] = perimeter

    # Circularity (4 * pi * area / perimeter^2)
    if perimeter > 0:
        features['circularity'] = (4 * np.pi * area) / (perimeter ** 2)

    # Fit ellipse (need at least 5 points)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)

        features['major_axis'] = major_axis
        features['minor_axis'] = minor_axis
        features['orientation'] = angle

        # Eccentricity
        if major_axis > 0:
            features['eccentricity'] = np.sqrt(1 - (minor_axis / major_axis) ** 2)

    # Solidity (area / convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features['solidity'] = area / hull_area

    # Extent (area / bounding rect area)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    if rect_area > 0:
        features['extent'] = area / rect_area

    # Equivalent diameter (diameter of circle with same area)
    features['equivalent_diameter'] = np.sqrt(4 * area / np.pi)

    return features


# =============================================================================
#  INTENSITY FEATURES
# =============================================================================

def compute_intensity_features(
    h_channel: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calcule les features d'intensité sur le canal H (Ruifrok)

    Args:
        h_channel: Hematoxylin channel (H, W), float [0, 1]
        mask: Binary mask (H, W)

    Returns:
        features: Dict with intensity features
    """
    features = {
        'h_mean': 0.0,
        'h_std': 0.0,
        'h_min': 0.0,
        'h_max': 0.0,
        'h_range': 0.0,
    }

    # Extract pixels inside mask
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return features

    h_values = h_channel[mask_bool]

    features['h_mean'] = float(np.mean(h_values))
    features['h_std'] = float(np.std(h_values))
    features['h_min'] = float(np.min(h_values))
    features['h_max'] = float(np.max(h_values))
    features['h_range'] = features['h_max'] - features['h_min']

    return features


# =============================================================================
#  TEXTURE FEATURES (Haralick)
# =============================================================================

def compute_texture_features(
    h_channel: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calcule les features de texture Haralick sur le canal H

    Args:
        h_channel: Hematoxylin channel (H, W), float [0, 1]
        mask: Binary mask (H, W)

    Returns:
        features: Dict with texture features
    """
    features = {
        'haralick_contrast': 0.0,
        'haralick_homogeneity': 0.0,
        'haralick_energy': 0.0,
        'haralick_correlation': 0.0,
        'haralick_entropy': 0.0,
    }

    # Mask the H channel
    mask_bool = mask.astype(bool)
    if mask_bool.sum() < 100:  # Need enough pixels
        return features

    # Quantize H channel to 8 levels for GLCM
    h_quantized = (h_channel * 7).astype(np.uint8)

    # Apply mask (set background to 0)
    h_masked = h_quantized.copy()
    h_masked[~mask_bool] = 0

    # Find bounding box of mask
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not rows.any() or not cols.any():
        return features

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to ROI
    roi = h_masked[rmin:rmax+1, cmin:cmax+1]

    if roi.shape[0] < 4 or roi.shape[1] < 4:
        return features

    try:
        # Compute GLCM
        glcm = graycomatrix(
            roi,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=8,
            symmetric=True,
            normed=True
        )

        # Compute Haralick features (average over angles)
        features['haralick_contrast'] = float(graycoprops(glcm, 'contrast').mean())
        features['haralick_homogeneity'] = float(graycoprops(glcm, 'homogeneity').mean())
        features['haralick_energy'] = float(graycoprops(glcm, 'energy').mean())
        features['haralick_correlation'] = float(graycoprops(glcm, 'correlation').mean())

        # Entropy (manual calculation)
        glcm_flat = glcm.flatten()
        glcm_flat = glcm_flat[glcm_flat > 0]
        features['haralick_entropy'] = float(-np.sum(glcm_flat * np.log2(glcm_flat + 1e-10)))

    except Exception as e:
        pass  # Return default features

    return features


# =============================================================================
#  COMBINED MORPHOMETRY
# =============================================================================

def compute_all_morphometry_features(
    image: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calcule toutes les features morphométriques (20 dims)

    Args:
        image: RGB image (H, W, 3), uint8
        mask: Binary mask (H, W)

    Returns:
        features: Dict with 20 features
    """
    # Extract H channel
    h_channel = extract_h_channel(image)

    # Compute all features
    geom_features = compute_geometric_features(mask)
    intensity_features = compute_intensity_features(h_channel, mask)
    texture_features = compute_texture_features(h_channel, mask)

    # Combine all features
    all_features = {}
    all_features.update(geom_features)      # 10 features
    all_features.update(intensity_features)  # 5 features
    all_features.update(texture_features)    # 5 features

    return all_features


def get_feature_names() -> List[str]:
    """
    Retourne la liste ordonnée des noms de features (20 dims)
    """
    return [
        # Geometric (10)
        'area', 'perimeter', 'circularity', 'eccentricity', 'solidity',
        'extent', 'major_axis', 'minor_axis', 'orientation', 'equivalent_diameter',
        # Intensity (5)
        'h_mean', 'h_std', 'h_min', 'h_max', 'h_range',
        # Texture (5)
        'haralick_contrast', 'haralick_homogeneity', 'haralick_energy',
        'haralick_correlation', 'haralick_entropy',
    ]


# =============================================================================
#  MAIN PROCESSING
# =============================================================================

def compute_morphometry_for_split(
    data_dir: str,
    embeddings_dir: str,
    output_dir: str,
    split: str
):
    """
    Calcule les features morphométriques pour un split

    Args:
        data_dir: data/processed/sipakmed/
        embeddings_dir: data/embeddings/sipakmed/
        output_dir: Output directory
        split: 'train' or 'val'
    """
    print(f"\n{'='*80}")
    print(f"COMPUTING MORPHOMETRY — {split.upper()} SPLIT")
    print(f"{'='*80}")

    # Load embeddings (contains masks and metadata)
    embeddings_path = os.path.join(embeddings_dir, f'sipakmed_{split}_embeddings.pt')
    print(f"Loading embeddings from: {embeddings_path}")

    data = torch.load(embeddings_path)
    masks = data['masks'].numpy()  # (N, 224, 224)
    labels = data['labels'].numpy()  # (N,)
    filenames = data['filenames']
    class_names = data['class_names']
    cls_tokens = data['cls_tokens']  # (N, 1536)

    n_samples = len(filenames)
    print(f"Loaded {n_samples} samples")

    # Load images from processed directory
    split_dir = os.path.join(data_dir, split)

    # Load metadata for image paths
    metadata_path = os.path.join(split_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create filename to image path mapping
    filename_to_path = {m['filename']: m['image_path'] for m in metadata}

    # Compute features for each sample
    feature_names = get_feature_names()
    n_features = len(feature_names)

    all_features = np.zeros((n_samples, n_features), dtype=np.float32)

    print(f"\nComputing {n_features} morphometry features...")

    for i, (filename, mask) in enumerate(tqdm(zip(filenames, masks), total=n_samples, desc=f"  {split}")):
        # Load image
        image_path = os.path.join(split_dir, filename_to_path[filename])
        image = np.array(Image.open(image_path).convert('RGB'))

        # Compute features
        features_dict = compute_all_morphometry_features(image, mask)

        # Convert to array
        for j, name in enumerate(feature_names):
            all_features[i, j] = features_dict[name]

    # Normalize features (z-score)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0) + 1e-8
    all_features_normalized = (all_features - mean) / std

    print(f"\nFeatures computed:")
    print(f"  Raw shape:        {all_features.shape}")
    print(f"  Normalized shape: {all_features_normalized.shape}")
    print(f"  Feature names:    {feature_names[:5]}... ({n_features} total)")

    # Fuse with CLS tokens
    cls_tokens_np = cls_tokens.numpy()
    fused_features = np.concatenate([cls_tokens_np, all_features_normalized], axis=1)

    print(f"\nFused features:")
    print(f"  CLS tokens:  {cls_tokens_np.shape}")
    print(f"  Morphometry: {all_features_normalized.shape}")
    print(f"  Fused:       {fused_features.shape} (1536 + {n_features} = {1536 + n_features})")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'sipakmed_{split}_features.pt')

    torch.save({
        'fused_features': torch.from_numpy(fused_features),  # (N, 1556)
        'cls_tokens': cls_tokens,  # (N, 1536)
        'morphometry': torch.from_numpy(all_features_normalized),  # (N, 20)
        'morphometry_raw': torch.from_numpy(all_features),  # (N, 20) non-normalized
        'labels': torch.from_numpy(labels),  # (N,)
        'class_names': class_names,
        'filenames': filenames,
        'feature_names': feature_names,
        'normalization': {
            'mean': mean.tolist(),
            'std': std.tolist(),
        },
        'metadata': {
            'n_cls_features': 1536,
            'n_morpho_features': n_features,
            'n_fused_features': 1536 + n_features,
        }
    }, output_path)

    print(f"\n  Saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    return fused_features, labels


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute morphometry features for V14 Cytology"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/sipakmed',
        help='Preprocessed data directory'
    )
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        default='data/embeddings/sipakmed',
        help='Embeddings directory (from step 01)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/features/sipakmed',
        help='Output directory for features'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='both',
        choices=['train', 'val', 'both'],
        help='Which split to process'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MORPHOMETRY FEATURES COMPUTATION — V14 Cytology")
    print("=" * 80)
    print(f"Data directory:       {args.data_dir}")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Output directory:     {args.output_dir}")
    print(f"Split:                {args.split}")
    print("")
    print("Features (20 dims):")
    print("  - Geometric (10): area, perimeter, circularity, eccentricity, ...")
    print("  - Intensity (5):  h_mean, h_std, h_min, h_max, h_range")
    print("  - Texture (5):    haralick_contrast, homogeneity, energy, ...")
    print("=" * 80)

    # Process splits
    splits = ['train', 'val'] if args.split == 'both' else [args.split]

    for split in splits:
        compute_morphometry_for_split(
            data_dir=args.data_dir,
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            split=split
        )

    # Summary
    print("\n" + "=" * 80)
    print("MORPHOMETRY COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"Features saved to: {args.output_dir}")
    print(f"\nFused features: 1536 (CLS) + 20 (morpho) = 1556 dims")
    print("\nNext step:")
    print("  python scripts/cytology/03_train_mlp_classifier.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
