#!/usr/bin/env python3
"""
Morphological Feature Extraction for Local Nucleus Classification.

Extracts context-independent features from individual nuclei for stable
WSI visualization. These features are based on biological properties that
pathologists understand and that remain consistent regardless of patch context.

Features extracted (13 total):
GEOMETRY (4):
1. area - Nuclear area in pixels
2. circularity - 4π × area / perimeter² (1.0 = perfect circle)
3. eccentricity - Ellipse fit eccentricity (0 = circle, 1 = line)
4. solidity - Area / convex hull area (shape regularity)

INTENSITY H-CHANNEL (4):
5. h_mean - Mean H-channel intensity (chromatin density)
6. h_std - Std of H-channel intensity (chromatin texture)
7. h_skewness - Asymmetry of H distribution (neg=dark tail, pos=light tail)
8. h_kurtosis - "Peakedness" of H distribution (high=concentrated)

COLOR RGB (3):
9. rgb_mean_r - Mean red channel (eosinophilia indicator)
10. rgb_mean_g - Mean green channel
11. rgb_mean_b - Mean blue channel (hematoxylin absorption)

TEXTURE (2):
12. boundary_ratio - Edge intensity / core intensity (membrane staining)
13. intensity_range - Max-Min H intensity (heterogeneity)

Reference:
- Expert recommendation: use morphological features + RF for stable cell-level visualization
- Literature: QuPath, PathAI, Hologic all use similar approaches

Author: CellViT-Optimus
Date: 2026-01-28
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
from scipy import ndimage

# Import Ruifrok H-channel extraction
from src.preprocessing.stain_separation import ruifrok_extract_h_channel

logger = logging.getLogger(__name__)


@dataclass
class MorphologicalFeatures:
    """Morphological features for a single nucleus (13 features)."""

    # Geometry (4)
    area: float              # Area in pixels
    circularity: float       # 4π × area / perimeter²
    eccentricity: float      # Ellipse eccentricity (0=circle, 1=line)
    solidity: float          # Area / convex hull area

    # Intensity H-channel (4)
    h_mean: float            # Mean H-channel intensity [0-255]
    h_std: float             # Std of H-channel intensity
    h_skewness: float        # Skewness of H distribution
    h_kurtosis: float        # Kurtosis of H distribution

    # Color RGB (3)
    rgb_mean_r: float        # Mean red channel
    rgb_mean_g: float        # Mean green channel
    rgb_mean_b: float        # Mean blue channel

    # Texture (2)
    boundary_ratio: float    # Edge intensity / core intensity
    intensity_range: float   # Max - Min H intensity

    # Metadata (not used for classification)
    centroid: Tuple[int, int] = (0, 0)  # (y, x) position

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for classifier (13 features)."""
        return np.array([
            # Geometry
            self.area / 1000.0,           # Normalize (typical: 100-2000 px)
            self.circularity,              # Already 0-1
            self.eccentricity,             # Already 0-1
            self.solidity,                 # Already 0-1
            # H-channel
            self.h_mean / 255.0,           # Normalize to 0-1
            self.h_std / 128.0,            # Normalize (typical: 0-128)
            self.h_skewness / 3.0,         # Normalize (typical: -3 to 3)
            self.h_kurtosis / 10.0,        # Normalize (typical: 0-10)
            # RGB
            self.rgb_mean_r / 255.0,       # Normalize to 0-1
            self.rgb_mean_g / 255.0,       # Normalize to 0-1
            self.rgb_mean_b / 255.0,       # Normalize to 0-1
            # Texture
            self.boundary_ratio,           # Already ~0-2 range
            self.intensity_range / 255.0,  # Normalize to 0-1
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Return feature names for interpretability."""
        return [
            # Geometry
            "area_norm",
            "circularity",
            "eccentricity",
            "solidity",
            # H-channel
            "h_mean_norm",
            "h_std_norm",
            "h_skewness_norm",
            "h_kurtosis_norm",
            # RGB
            "rgb_r_norm",
            "rgb_g_norm",
            "rgb_b_norm",
            # Texture
            "boundary_ratio",
            "intensity_range_norm",
        ]


def _compute_skewness(values: np.ndarray) -> float:
    """Compute skewness (third standardized moment)."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-6:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def _compute_kurtosis(values: np.ndarray) -> float:
    """Compute excess kurtosis (fourth standardized moment - 3)."""
    n = len(values)
    if n < 4:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-6:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 4) - 3.0)


def extract_nucleus_features(
    image_rgb: np.ndarray,
    instance_map: np.ndarray,
    nucleus_id: int,
    h_channel: Optional[np.ndarray] = None,
) -> Optional[MorphologicalFeatures]:
    """
    Extract morphological features for a single nucleus (13 features).

    Args:
        image_rgb: RGB image (H, W, 3)
        instance_map: Instance segmentation map (H, W)
        nucleus_id: ID of the nucleus in instance_map
        h_channel: Pre-computed H-channel (optional, computed if None)

    Returns:
        MorphologicalFeatures or None if extraction fails
    """
    # Get nucleus mask
    mask = (instance_map == nucleus_id).astype(np.uint8)

    if mask.sum() < 10:  # Too small
        return None

    # Compute H-channel if not provided
    if h_channel is None:
        h_channel = ruifrok_extract_h_channel(image_rgb)

    # === GEOMETRY (4 features) ===

    # Area
    area = float(mask.sum())

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    # Perimeter and circularity
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
        circularity = min(circularity, 1.0)  # Cap at 1.0
    else:
        circularity = 0.0

    # Eccentricity via ellipse fit
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            (_, axes, _) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0.0
        except cv2.error:
            eccentricity = 0.0
    else:
        eccentricity = 0.0

    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # === INTENSITY H-CHANNEL (4 features) ===

    h_values = h_channel[mask > 0]
    if len(h_values) == 0:
        return None

    h_mean = float(np.mean(h_values))
    h_std = float(np.std(h_values))
    h_skewness = _compute_skewness(h_values)
    h_kurtosis = _compute_kurtosis(h_values)

    # === COLOR RGB (3 features) ===

    rgb_values = image_rgb[mask > 0]  # (N, 3)
    rgb_mean_r = float(np.mean(rgb_values[:, 0]))
    rgb_mean_g = float(np.mean(rgb_values[:, 1]))
    rgb_mean_b = float(np.mean(rgb_values[:, 2]))

    # === TEXTURE (2 features) ===

    # Boundary ratio: compare edge vs core intensity
    # Erode mask to get core
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded

    boundary_h = h_channel[boundary > 0]
    core_h = h_channel[eroded > 0]

    if len(core_h) > 0 and np.mean(core_h) > 1e-6:
        boundary_ratio = float(np.mean(boundary_h) / np.mean(core_h)) if len(boundary_h) > 0 else 1.0
    else:
        boundary_ratio = 1.0

    # Intensity range (heterogeneity)
    intensity_range = float(np.max(h_values) - np.min(h_values))

    # === CENTROID ===

    coords = np.where(mask > 0)
    centroid = (int(np.mean(coords[0])), int(np.mean(coords[1])))  # (y, x)

    return MorphologicalFeatures(
        area=area,
        circularity=circularity,
        eccentricity=eccentricity,
        solidity=solidity,
        h_mean=h_mean,
        h_std=h_std,
        h_skewness=h_skewness,
        h_kurtosis=h_kurtosis,
        rgb_mean_r=rgb_mean_r,
        rgb_mean_g=rgb_mean_g,
        rgb_mean_b=rgb_mean_b,
        boundary_ratio=boundary_ratio,
        intensity_range=intensity_range,
        centroid=centroid,
    )


def extract_all_nuclei_features(
    image_rgb: np.ndarray,
    instance_map: np.ndarray,
) -> Tuple[List[int], np.ndarray]:
    """
    Extract features for all nuclei in an instance map.

    Args:
        image_rgb: RGB image (H, W, 3)
        instance_map: Instance segmentation map (H, W)

    Returns:
        (nucleus_ids, feature_matrix) where feature_matrix is (N, 13)
    """
    # Pre-compute H-channel once for efficiency
    h_channel = ruifrok_extract_h_channel(image_rgb)

    # Get all nucleus IDs
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]  # Exclude background

    nucleus_ids = []
    features_list = []

    for nid in unique_ids:
        feat = extract_nucleus_features(
            image_rgb, instance_map, int(nid), h_channel
        )
        if feat is not None:
            nucleus_ids.append(int(nid))
            features_list.append(feat.to_vector())

    n_features = len(MorphologicalFeatures.feature_names())  # 13
    if not features_list:
        return [], np.zeros((0, n_features), dtype=np.float32)

    feature_matrix = np.stack(features_list, axis=0)

    return nucleus_ids, feature_matrix
