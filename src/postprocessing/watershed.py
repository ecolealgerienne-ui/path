"""
HV-guided Watershed for Instance Segmentation.

This module provides the SINGLE SOURCE OF TRUTH for the watershed algorithm.
Both evaluation scripts (test_v13_smart_crops_aji.py and optimize_watershed_aji.py)
MUST import from here to ensure consistent results.

Algorithm:
    1. Threshold NP prediction to get binary mask
    2. Compute distance transform
    3. Compute HV magnitude and marker energy: dist * (1 - hv_magnitude^beta)
    4. Find local maxima as seed markers
    5. Label markers with scipy.ndimage.label
    6. Run watershed with distance as elevation
    7. Remove small objects and relabel

Author: CellViT-Optimus Project
Date: 2025-12-29
"""

import numpy as np
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max


def hv_guided_watershed(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    np_threshold: float = 0.40,
    beta: float = 0.50,
    min_size: int = 30,
    min_distance: int = 5,
    debug: bool = False
) -> np.ndarray:
    """
    HV-guided watershed for instance segmentation.

    Uses HV magnitude to suppress markers at cell boundaries where
    HV gradients are strong, improving instance separation.

    Args:
        np_pred: Nuclear presence probability map (H, W) in [0, 1]
        hv_pred: HV maps (2, H, W) in [-1, 1]
        np_threshold: Threshold for NP binarization
            - Respiratory optimal: 0.40
            - Epidermal optimal: 0.45
        beta: HV magnitude exponent (0.50 optimal for all families)
        min_size: Minimum instance size in pixels
            - Respiratory optimal: 30
            - Epidermal optimal: 40
        min_distance: Minimum distance between peak markers (5 optimal)
        debug: Print debug info for sanity checking

    Returns:
        Instance map (H, W) with instance IDs starting from 1

    Example:
        >>> from src.postprocessing import hv_guided_watershed
        >>> pred_inst = hv_guided_watershed(
        ...     np_pred, hv_pred,
        ...     np_threshold=0.45, min_size=40  # Epidermal params
        ... )
    """
    # Threshold NP to get binary mask
    np_binary = (np_pred > np_threshold).astype(np.uint8)

    if np_binary.sum() == 0:
        return np.zeros_like(np_pred, dtype=np.int32)

    # Distance transform
    dist = distance_transform_edt(np_binary)

    # HV magnitude (range [0, sqrt(2)])
    hv_h = hv_pred[0]
    hv_v = hv_pred[1]
    hv_magnitude = np.sqrt(hv_h**2 + hv_v**2)

    # HV-guided marker energy
    # Higher HV magnitude -> lower marker energy -> suppress markers at boundaries
    marker_energy = dist * (1 - hv_magnitude ** beta)

    # Find local maxima as markers
    markers_coords = peak_local_max(
        marker_energy,
        min_distance=min_distance,
        threshold_abs=0.1,
        exclude_border=False
    )

    if debug:
        print(f"    peak_local_max found {len(markers_coords)} markers")

    # Create markers map with sequential IDs
    markers = np.zeros_like(np_binary, dtype=np.int32)
    for i, (y, x) in enumerate(markers_coords, start=1):
        markers[y, x] = i

    # If no markers found, return empty
    if markers.max() == 0:
        if debug:
            print("    WARNING: No markers found!")
        return np.zeros_like(np_pred, dtype=np.int32)

    # CRITICAL: Label markers BEFORE watershed
    # Uses scipy.ndimage.label for consistency
    markers = label(markers)[0]

    # Watershed (use distance as elevation map)
    instances = watershed(-dist, markers, mask=np_binary)

    if debug:
        unique_before = np.unique(instances)
        print(f"    After watershed: {len(unique_before)} unique values")

    # Remove small objects (skip if only 0 or 1 label to avoid warning)
    if min_size > 0:
        n_labels = len(np.unique(instances)) - 1  # Exclude background
        if n_labels > 1:
            instances = remove_small_objects(instances, min_size=min_size)

    # Relabel to ensure consecutive IDs
    instances = label(instances)[0]

    if debug:
        unique_after = np.unique(instances)
        n_instances = len(unique_after) - 1  # Exclude background (0)
        print(f"    After remove_small_objects: {n_instances} instances")

    return instances.astype(np.int32)
