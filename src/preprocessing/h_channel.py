#!/usr/bin/env python3
"""
H-Channel Extraction and Statistics for V15.3 Cytology Pipeline

This module provides H-Channel based features for cytology analysis:
1. H-Channel extraction using Ruifrok deconvolution
2. H-Statistics computation for patches (mean, std, nuclei_count, area_ratio)
3. Nuclei detection via Otsu/Adaptive thresholding

The H-Channel (Hematoxylin) highlights nuclear structures and is used to:
- Validate Cell Triage predictions (confidence boosting)
- Detect nuclei for cell-level visualization
- Compute cell density metrics

Reference:
- Ruifrok AC, Johnston DA. "Quantification of histochemical staining by color deconvolution."
  Analytical and Quantitative Cytology and Histology, 2001.

Author: CellViT-Optimus V15.3
Date: 2026-01-24
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import existing Ruifrok implementation
from .stain_separation import (
    ruifrok_extract_h_channel,
    RUIFROK_H_VECTOR,
)


# =========================================================================
# CONSTANTS
# =========================================================================

# Nucleus size constraints (in pixels for 224x224 patches at 0.5 MPP)
MIN_NUCLEUS_AREA = 50    # Minimum nucleus area (filters noise)
MAX_NUCLEUS_AREA = 5000  # Maximum nucleus area (filters debris/clusters)

# Morphological kernel size for cleanup
MORPH_KERNEL_SIZE = 3


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass
class NucleusInfo:
    """Information about a detected nucleus"""
    centroid: Tuple[int, int]  # (x, y) coordinates
    area: int                   # Area in pixels
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box
    contour: Optional[np.ndarray] = None  # OpenCV contour


@dataclass
class HChannelStats:
    """H-Channel statistics for a patch"""
    h_mean: float              # Mean H-channel intensity [0-255]
    h_std: float               # Std deviation of H-channel
    nuclei_count: int          # Number of detected nuclei
    nuclei_area_ratio: float   # Total nuclei area / patch area
    nuclei_details: List[NucleusInfo]  # Per-nucleus information

    def to_features(self) -> np.ndarray:
        """Convert to feature vector for Cell Triage v2"""
        return np.array([
            self.h_mean / 255.0,       # Normalize to [0, 1]
            self.h_std / 255.0,        # Normalize to [0, 1]
            min(self.nuclei_count / 20.0, 1.0),  # Cap at 20 nuclei
            self.nuclei_area_ratio
        ], dtype=np.float32)


# =========================================================================
# H-CHANNEL EXTRACTION
# =========================================================================

def extract_h_channel_ruifrok(
    rgb_image: np.ndarray,
    output_range: str = "uint8"
) -> np.ndarray:
    """
    Extract Hematoxylin channel using Ruifrok deconvolution.

    This is a wrapper around the existing ruifrok_extract_h_channel()
    with additional options for output format.

    Ruifrok uses FIXED physical constants (Beer-Lambert law):
    - Hematoxylin vector: [0.650, 0.704, 0.286]
    - Eosin vector: [0.072, 0.990, 0.105]

    Args:
        rgb_image: RGB image (H, W, 3) in range [0, 255]
        output_range: Output format:
            - "uint8": [0, 255] uint8 (for visualization/thresholding)
            - "float01": [0, 1] float32 (for neural network input)
            - "raw": Raw OD projection values

    Returns:
        h_channel: Hematoxylin density map (H, W)

    Example:
        >>> import cv2
        >>> image = cv2.imread("cell.png")
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> h_channel = extract_h_channel_ruifrok(image)
        >>> h_channel.shape
        (224, 224)
        >>> h_channel.dtype
        dtype('uint8')
    """
    # Use existing Ruifrok implementation (normalized to [0, 1])
    h_channel = ruifrok_extract_h_channel(rgb_image, normalize=True)

    if output_range == "uint8":
        # Convert to uint8 for OpenCV operations
        h_channel = (h_channel * 255).astype(np.uint8)
    elif output_range == "float01":
        # Keep as float32 [0, 1]
        h_channel = h_channel.astype(np.float32)
    elif output_range == "raw":
        # Re-extract without normalization
        h_channel = ruifrok_extract_h_channel(rgb_image, normalize=False)
    else:
        raise ValueError(f"Invalid output_range: {output_range}. "
                        f"Must be 'uint8', 'float01', or 'raw'")

    return h_channel


# =========================================================================
# H-STATISTICS COMPUTATION
# =========================================================================

def compute_h_stats(
    rgb_image: np.ndarray,
    h_channel: Optional[np.ndarray] = None,
    min_nucleus_area: int = MIN_NUCLEUS_AREA,
    max_nucleus_area: int = MAX_NUCLEUS_AREA,
    return_binary_mask: bool = False
) -> HChannelStats:
    """
    Compute H-Channel statistics for a patch.

    This function extracts features that characterize the nuclear content:
    - h_mean: Average H-channel intensity (higher = more nuclei)
    - h_std: Heterogeneity of staining
    - nuclei_count: Number of detected nuclei (blob count)
    - nuclei_area_ratio: Fraction of patch covered by nuclei

    These features are used by Cell Triage v2 to improve filtering.

    Args:
        rgb_image: RGB image (H, W, 3) in range [0, 255]
        h_channel: Pre-computed H-channel (optional, will be extracted if None)
        min_nucleus_area: Minimum nucleus area in pixels
        max_nucleus_area: Maximum nucleus area in pixels
        return_binary_mask: If True, also return the binary mask

    Returns:
        HChannelStats: Statistics object with h_mean, h_std, nuclei_count, etc.

    Example:
        >>> stats = compute_h_stats(patch_rgb)
        >>> print(f"Nuclei: {stats.nuclei_count}, Area ratio: {stats.nuclei_area_ratio:.2%}")
        Nuclei: 5, Area ratio: 12.34%
        >>> features = stats.to_features()  # For Cell Triage v2
    """
    # Extract H-channel if not provided
    if h_channel is None:
        h_channel = extract_h_channel_ruifrok(rgb_image, output_range="uint8")

    # Ensure uint8 for OpenCV operations
    if h_channel.dtype != np.uint8:
        if h_channel.max() <= 1.0:
            h_channel = (h_channel * 255).astype(np.uint8)
        else:
            h_channel = h_channel.astype(np.uint8)

    # Basic statistics
    h_mean = float(np.mean(h_channel))
    h_std = float(np.std(h_channel))

    # Otsu thresholding for nuclei detection
    _, binary = cv2.threshold(
        h_channel, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Filter nuclei by size
    nuclei_details = []
    total_nuclei_area = 0

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]

        if min_nucleus_area <= area <= max_nucleus_area:
            nucleus = NucleusInfo(
                centroid=(int(centroids[i][0]), int(centroids[i][1])),
                area=area,
                bbox=(
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT]
                )
            )
            nuclei_details.append(nucleus)
            total_nuclei_area += area

    # Compute area ratio
    patch_area = h_channel.shape[0] * h_channel.shape[1]
    nuclei_area_ratio = total_nuclei_area / patch_area

    result = HChannelStats(
        h_mean=h_mean,
        h_std=h_std,
        nuclei_count=len(nuclei_details),
        nuclei_area_ratio=nuclei_area_ratio,
        nuclei_details=nuclei_details
    )

    if return_binary_mask:
        return result, binary

    return result


def compute_h_stats_batch(
    images: np.ndarray,
    min_nucleus_area: int = MIN_NUCLEUS_AREA,
    max_nucleus_area: int = MAX_NUCLEUS_AREA
) -> List[HChannelStats]:
    """
    Compute H-Channel statistics for a batch of patches.

    Args:
        images: Batch of RGB images (N, H, W, 3) in range [0, 255]
        min_nucleus_area: Minimum nucleus area in pixels
        max_nucleus_area: Maximum nucleus area in pixels

    Returns:
        List of HChannelStats, one per image
    """
    return [
        compute_h_stats(
            images[i],
            min_nucleus_area=min_nucleus_area,
            max_nucleus_area=max_nucleus_area
        )
        for i in range(len(images))
    ]


# =========================================================================
# NUCLEI DETECTION FOR VISUALIZATION
# =========================================================================

def detect_nuclei_for_visualization(
    rgb_image: np.ndarray,
    predicted_class: str = "UNKNOWN",
    use_adaptive_threshold: bool = True,
    min_nucleus_area: int = MIN_NUCLEUS_AREA,
    max_nucleus_area: int = MAX_NUCLEUS_AREA
) -> List[Dict]:
    """
    Detect nuclei in a patch for cell-level visualization.

    This function returns nuclei with contours that can be drawn
    on the image for cell-level visualization (V15.3).

    Each nucleus inherits the predicted class from its parent patch.

    Args:
        rgb_image: RGB image (H, W, 3) in range [0, 255]
        predicted_class: Bethesda class to assign to detected nuclei
        use_adaptive_threshold: Use adaptive threshold (better for clusters)
        min_nucleus_area: Minimum nucleus area in pixels
        max_nucleus_area: Maximum nucleus area in pixels

    Returns:
        List of nuclei dicts with keys:
            - 'contour': OpenCV contour array
            - 'centroid': (x, y) coordinates
            - 'area': Area in pixels
            - 'class': Inherited Bethesda class

    Example:
        >>> nuclei = detect_nuclei_for_visualization(patch, predicted_class="HSIL")
        >>> for n in nuclei:
        ...     cv2.drawContours(image, [n['contour']], -1, (0, 0, 255), 2)
    """
    # Extract H-channel
    h_channel = extract_h_channel_ruifrok(rgb_image, output_range="uint8")

    if use_adaptive_threshold:
        # Adaptive threshold (better for varying illumination and clusters)
        binary = cv2.adaptiveThreshold(
            h_channel, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21,
            C=5
        )
        # Invert (nuclei are dark in H-channel → high values)
        binary = 255 - binary
    else:
        # Otsu threshold
        _, binary = cv2.threshold(
            h_channel, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Distance transform for watershed-style separation
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform,
        0.5 * dist_transform.max(),
        255,
        0
    )
    sure_fg = np.uint8(sure_fg)

    # Find contours
    contours, _ = cv2.findContours(
        sure_fg,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter and build result
    nuclei = []
    for contour in contours:
        area = cv2.contourArea(contour)

        if min_nucleus_area < area < max_nucleus_area:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                nuclei.append({
                    'contour': contour,
                    'centroid': (cx, cy),
                    'area': area,
                    'class': predicted_class
                })

    return nuclei


# =========================================================================
# CONFIDENCE BOOSTING
# =========================================================================

def apply_confidence_boosting(
    prediction: Dict,
    h_stats: HChannelStats
) -> Dict:
    """
    Adjust prediction confidence based on H-Channel validation.

    Rules:
    1. Abnormal prediction + 0 nuclei detected → reduce confidence (likely false positive)
    2. Normal prediction + high nuclei density → flag for review
    3. High H-std + HSIL/SCC → boost confidence (cluster pattern)

    Args:
        prediction: Dict with keys 'class', 'confidence'
        h_stats: H-Channel statistics for the patch

    Returns:
        Updated prediction dict with adjusted 'confidence' and optional 'flag'

    Example:
        >>> pred = {'class': 'HSIL', 'confidence': 0.85}
        >>> stats = compute_h_stats(patch)
        >>> pred = apply_confidence_boosting(pred, stats)
        >>> if 'flag' in pred:
        ...     print(f"Review required: {pred['flag']}")
    """
    result = prediction.copy()
    confidence = result['confidence']
    predicted_class = result['class']

    # Rule 1: Abnormal prediction without nuclei = suspect
    if predicted_class != 'NILM' and h_stats.nuclei_count == 0:
        result['confidence'] = confidence * 0.5
        result['flag'] = 'LOW_CONFIDENCE_NO_NUCLEI'

    # Rule 2: Normal prediction with high density = review
    elif predicted_class == 'NILM' and h_stats.nuclei_count > 10:
        result['flag'] = 'REVIEW_HIGH_DENSITY'

    # Rule 3: High variance + HSIL/SCC = boost (cluster pattern)
    elif h_stats.h_std > 50 and predicted_class in ['HSIL', 'SCC']:
        result['confidence'] = min(confidence * 1.2, 0.99)

    return result


# =========================================================================
# VISUALIZATION UTILITIES
# =========================================================================

# Bethesda class colors (BGR for OpenCV)
BETHESDA_COLORS = {
    'NILM': (0, 200, 0),      # Green
    'ASCUS': (0, 255, 255),   # Yellow
    'ASCH': (0, 128, 255),    # Orange
    'LSIL': (0, 200, 255),    # Yellow-Orange
    'HSIL': (0, 0, 255),      # Red
    'SCC': (128, 0, 128),     # Purple
    'UNKNOWN': (200, 200, 200) # Gray
}


def render_nuclei_overlay(
    image: np.ndarray,
    nuclei: List[Dict],
    alpha: float = 0.4
) -> np.ndarray:
    """
    Render nuclei contours on image with class-based colors.

    Args:
        image: RGB or BGR image (H, W, 3)
        nuclei: List of nuclei from detect_nuclei_for_visualization()
        alpha: Transparency for filled contours (0=invisible, 1=opaque)

    Returns:
        Image with nuclei overlay

    Example:
        >>> nuclei = detect_nuclei_for_visualization(patch, "HSIL")
        >>> vis = render_nuclei_overlay(patch, nuclei)
        >>> cv2.imwrite("visualization.png", vis)
    """
    overlay = image.copy()

    for nucleus in nuclei:
        color = BETHESDA_COLORS.get(nucleus['class'], (200, 200, 200))
        contour = nucleus['contour']

        # Draw contour outline
        cv2.drawContours(overlay, [contour], -1, color, 2)

        # Fill with semi-transparent color
        cv2.drawContours(overlay, [contour], -1, color, -1)

    # Blend with original
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return result


# =========================================================================
# EXPORTS
# =========================================================================

__all__ = [
    # Constants
    'MIN_NUCLEUS_AREA',
    'MAX_NUCLEUS_AREA',
    'BETHESDA_COLORS',
    # Data classes
    'NucleusInfo',
    'HChannelStats',
    # Main functions
    'extract_h_channel_ruifrok',
    'compute_h_stats',
    'compute_h_stats_batch',
    'detect_nuclei_for_visualization',
    'apply_confidence_boosting',
    'render_nuclei_overlay',
]


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔬 H-CHANNEL MODULE FOR V15.3 CYTOLOGY PIPELINE")
    print("=" * 70)
    print("\nFunctions available:")
    print("  - extract_h_channel_ruifrok(): Extract Hematoxylin channel")
    print("  - compute_h_stats(): Compute H-channel statistics")
    print("  - detect_nuclei_for_visualization(): Detect nuclei with contours")
    print("  - apply_confidence_boosting(): Adjust confidence based on H-stats")
    print("  - render_nuclei_overlay(): Draw nuclei on image")
    print("\nUsage:")
    print("  >>> from src.preprocessing.h_channel import compute_h_stats")
    print("  >>> stats = compute_h_stats(patch_rgb)")
    print("  >>> print(f'Nuclei: {stats.nuclei_count}')")
    print("\n" + "=" * 70)
