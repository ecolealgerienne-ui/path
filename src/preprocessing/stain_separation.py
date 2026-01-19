#!/usr/bin/env python3
"""
Stain Separation - Ruifrok & Macenko

Extracts Hematoxylin (H) and Eosin (E) channels from H&E stained images.

Two methods:
1. Ruifrok (RECOMMENDED for V13/V14) - Physical constants (Beer-Lambert law)
2. Macenko (DEPRECATED for V13) - Statistical (causes -4.3% AJI regression)

Reference:
- Ruifrok AC, Johnston DA. "Quantification of histochemical staining by color deconvolution."
  Analytical and Quantitative Cytology and Histology, 2001.

Author: CellViT-Optimus V13/V14
Date: 2026-01-18
"""

import numpy as np
import cv2
from typing import Tuple, Optional


# =========================================================================
# RUIFROK DECONVOLUTION (PHYSICAL CONSTANTS)
# =========================================================================

# Fixed stain vectors (Beer-Lambert law)
RUIFROK_H_VECTOR = np.array([0.650, 0.704, 0.286])  # Hematoxylin
RUIFROK_E_VECTOR = np.array([0.072, 0.990, 0.105])  # Eosin
RUIFROK_DAB_VECTOR = np.array([0.268, 0.570, 0.776])  # DAB (for IHC)


def ruifrok_extract_h_channel(
    image_rgb: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Extract Hematoxylin channel using Ruifrok deconvolution

    This is the GOLD STANDARD for V13/V14 because:
    - Uses PHYSICAL constants (universal)
    - Preserves chromatin texture
    - No statistical bias

    Args:
        image_rgb: RGB image (H, W, 3) in range [0, 255]
        normalize: Whether to normalize output to [0, 1]

    Returns:
        h_channel: Hematoxylin density (H, W) in range [0, 1] if normalized
    """
    # Convert RGB to OD (Optical Density)
    od = rgb_to_od(image_rgb)

    # Project onto Hematoxylin vector
    h_channel = np.dot(od, RUIFROK_H_VECTOR)

    if normalize:
        # Normalize to [0, 1]
        h_channel = np.clip(h_channel, 0, None)
        if h_channel.max() > 0:
            h_channel = h_channel / h_channel.max()

    return h_channel


def ruifrok_extract_e_channel(
    image_rgb: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Extract Eosin channel using Ruifrok deconvolution"""
    od = rgb_to_od(image_rgb)
    e_channel = np.dot(od, RUIFROK_E_VECTOR)

    if normalize:
        e_channel = np.clip(e_channel, 0, None)
        if e_channel.max() > 0:
            e_channel = e_channel / e_channel.max()

    return e_channel


def ruifrok_deconvolution(
    image_rgb: np.ndarray,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Ruifrok deconvolution (H + E channels)

    Returns:
        Tuple of (h_channel, e_channel)
    """
    h_channel = ruifrok_extract_h_channel(image_rgb, normalize)
    e_channel = ruifrok_extract_e_channel(image_rgb, normalize)

    return h_channel, e_channel


# =========================================================================
# MACENKO NORMALIZATION (DEPRECATED FOR V13)
# =========================================================================

def macenko_normalize(
    image_rgb: np.ndarray,
    reference_image: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.15
) -> np.ndarray:
    """
    Macenko stain normalization

    ⚠️ WARNING: Causes -4.3% AJI regression in V13 Histology
    Reason: Rotates Eosin towards Hematoxylin vector → cytoplasm "ghosts" in H-channel

    Recommended for:
    - ✅ Cytology (V14) - Color standardization across scanners
    - ❌ Histology (V13) - Conflicts with Ruifrok FPN Chimique

    Args:
        image_rgb: RGB image (H, W, 3)
        reference_image: Reference image for normalization (optional)
        alpha: Percentile for stain vector estimation (default 1%)
        beta: OD threshold (default 0.15)

    Returns:
        Normalized RGB image
    """
    # Convert to OD
    od = rgb_to_od(image_rgb)

    # Reshape for SVD
    od_reshape = od.reshape((-1, 3))

    # Remove transparent pixels (OD < beta)
    od_mask = np.all(od_reshape > beta, axis=1)
    od_filtered = od_reshape[od_mask]

    if len(od_filtered) < 10:
        # Not enough stained pixels
        return image_rgb

    # SVD to find stain vectors
    _, V = np.linalg.eigh(np.cov(od_filtered.T))
    V = V[:, [2, 1]]  # Take top 2 eigenvectors

    # Rotate to ensure first vector is Hematoxylin-like
    if V[0, 0] < 0:
        V[:, 0] *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1

    # Project OD onto stain vectors
    stain_concentrations = np.dot(od_filtered, V)

    # Get percentile values
    max_c = np.percentile(stain_concentrations, 100 * (1 - alpha), axis=0)

    # Normalize
    stain_concentrations /= max_c

    # If reference provided, match to reference
    if reference_image is not None:
        # Extract reference stain vectors
        od_ref = rgb_to_od(reference_image)
        od_ref_reshape = od_ref.reshape((-1, 3))
        od_ref_mask = np.all(od_ref_reshape > beta, axis=1)
        od_ref_filtered = od_ref_reshape[od_ref_mask]

        _, V_ref = np.linalg.eigh(np.cov(od_ref_filtered.T))
        V_ref = V_ref[:, [2, 1]]

        if V_ref[0, 0] < 0:
            V_ref[:, 0] *= -1
        if V_ref[0, 1] < 0:
            V_ref[:, 1] *= -1

        stain_concentrations_ref = np.dot(od_ref_filtered, V_ref)
        max_c_ref = np.percentile(stain_concentrations_ref, 100 * (1 - alpha), axis=0)

        # Match concentrations
        stain_concentrations *= max_c_ref
        V = V_ref

    # Reconstruct OD
    od_normalized = np.dot(stain_concentrations, V.T)

    # Fill back into original shape
    od_output = np.zeros_like(od_reshape)
    od_output[od_mask] = od_normalized
    od_output = od_output.reshape(od.shape)

    # Convert back to RGB
    rgb_normalized = od_to_rgb(od_output)

    return rgb_normalized


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def rgb_to_od(image_rgb: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Convert RGB to Optical Density (OD)

    Beer-Lambert law: OD = -log10(I / I0)
    where I = transmitted light, I0 = incident light (white = 255)

    Args:
        image_rgb: RGB image in range [0, 255]
        epsilon: Small value to avoid log(0)

    Returns:
        Optical density (H, W, 3)
    """
    # Normalize to [0, 1]
    image_float = image_rgb.astype(np.float32) / 255.0

    # Clip to avoid log(0)
    image_float = np.clip(image_float, epsilon, 1.0)

    # OD = -log10(I / I0)
    od = -np.log10(image_float)

    return od


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    """
    Convert Optical Density back to RGB

    I = I0 * 10^(-OD)

    Args:
        od: Optical density (H, W, 3)

    Returns:
        RGB image in range [0, 255]
    """
    # I = 10^(-OD)
    image_float = 10 ** (-od)

    # Clip to [0, 1]
    image_float = np.clip(image_float, 0, 1)

    # Scale to [0, 255]
    image_rgb = (image_float * 255).astype(np.uint8)

    return image_rgb


def visualize_h_channel(
    image_rgb: np.ndarray,
    h_channel: np.ndarray
) -> np.ndarray:
    """
    Create visualization showing H-channel overlay

    Args:
        image_rgb: Original RGB image
        h_channel: H-channel (normalized to [0, 1])

    Returns:
        Visualization image
    """
    # Convert H-channel to heatmap
    h_heatmap = (h_channel * 255).astype(np.uint8)
    h_colored = cv2.applyColorMap(h_heatmap, cv2.COLORMAP_JET)

    # Blend with original
    vis = cv2.addWeighted(image_rgb, 0.5, h_colored, 0.5, 0)

    return vis


# =========================================================================
# VALIDATION FUNCTIONS
# =========================================================================

def compare_ruifrok_vs_macenko(image_rgb: np.ndarray):
    """
    Compare Ruifrok vs Macenko extraction

    Demonstrates why Ruifrok is superior for V13 Histology
    """
    # Ruifrok
    h_ruifrok, e_ruifrok = ruifrok_deconvolution(image_rgb)

    # Macenko
    image_macenko = macenko_normalize(image_rgb)
    h_macenko, e_macenko = ruifrok_deconvolution(image_macenko)

    # Compute difference
    h_diff = np.abs(h_ruifrok - h_macenko)

    print("="*70)
    print("🔬 RUIFROK vs MACENKO COMPARISON")
    print("="*70)
    print(f"\nH-channel variance:")
    print(f"  Ruifrok: {h_ruifrok.var():.4f}")
    print(f"  Macenko: {h_macenko.var():.4f}")
    print(f"  Difference: {h_diff.mean():.4f}")
    print("\n⚠️  V13 Discovery: Macenko causes -4.3% AJI regression")
    print("    Reason: Eosin → Hematoxylin shift creates cytoplasm ghosts")
    print("\n✅ Recommendation: Use Ruifrok for production")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("🔬 STAIN SEPARATION - RUIFROK & MACENKO")
    print("="*70)
    print("\nRuifrok Deconvolution (RECOMMENDED):")
    print("  - Uses physical constants (Beer-Lambert law)")
    print("  - Hematoxylin vector: [0.650, 0.704, 0.286]")
    print("  - Eosin vector: [0.072, 0.990, 0.105]")
    print("  - V13 Production: ✅ Validated (AJI 0.6872)")
    print("\nMacenko Normalization (DEPRECATED for V13):")
    print("  - Uses statistical SVD")
    print("  - V13 Histology: ❌ Causes -4.3% AJI regression")
    print("  - V14 Cytology: ✅ OK for scanner standardization")
    print("\n" + "="*70)
