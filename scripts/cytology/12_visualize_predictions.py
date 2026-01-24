"""
Visualisation des Predictions — Cell Triage + MultiHead Bethesda

Ce script genere des images annotees avec les predictions du pipeline:

V15.2 (Patch-Level):
- Rectangles colores par severite (Vert=NILM, Jaune=Low-grade, Rouge=High-grade)

V15.3 (Cell-Level) [--cell_level]:
- Contours de noyaux detectes via H-Channel (Ruifrok)
- Chaque noyau herite la classe du patch parent
- Plus interpretable pour les pathologistes

Usage:
    # Patch-level (V15.2)
    python scripts/cytology/12_visualize_predictions.py \
        --image path/to/image.jpg \
        --output results/visualizations/

    # Cell-level (V15.3)
    python scripts/cytology/12_visualize_predictions.py \
        --image path/to/image.jpg \
        --output results/visualizations/ \
        --cell_level

    # Process all images in directory
    python scripts/cytology/12_visualize_predictions.py \
        --input_dir data/raw/apcdata/APCData_YOLO/val/images \
        --output results/visualizations/

Author: V15.2/V15.3 Cytology Branch
Date: 2026-01-24
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import importlib.util

# Import H-Channel functions for V15.3 cell-level visualization
from src.preprocessing.h_channel import (
    detect_nuclei_for_visualization,
    render_nuclei_overlay,
    compute_h_stats,
    apply_confidence_boosting,
    BETHESDA_COLORS as H_CHANNEL_COLORS
)

# Dynamic import from 11_unified_inference.py
def _import_unified_inference():
    """Import classes from unified inference module"""
    module_path = Path(__file__).parent / "11_unified_inference.py"
    spec = importlib.util.spec_from_file_location("unified_inference", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_unified = _import_unified_inference()
UnifiedInferencePipeline = _unified.UnifiedInferencePipeline
ImageDiagnosis = _unified.ImageDiagnosis
PatchResult = _unified.PatchResult
BETHESDA_CLASSES = _unified.BETHESDA_CLASSES


# =============================================================================
#  CONFIGURATION
# =============================================================================

# Colors BGR (OpenCV format)
SEVERITY_COLORS = {
    "Normal": (0, 200, 0),      # Green
    "Low-grade": (0, 200, 255),  # Yellow-Orange
    "High-grade": (0, 0, 255),   # Red
    "Empty": (180, 180, 180)     # Gray
}

# Class colors for fine-grained display
CLASS_COLORS = {
    "NILM": (0, 200, 0),        # Green
    "ASCUS": (0, 255, 255),     # Yellow
    "ASCH": (0, 128, 255),      # Orange
    "LSIL": (0, 200, 255),      # Light orange
    "HSIL": (0, 0, 255),        # Red
    "SCC": (128, 0, 128),       # Purple (cancer)
    "EMPTY": (180, 180, 180)    # Gray
}

# Legend position and styling
LEGEND_FONT = cv2.FONT_HERSHEY_SIMPLEX
LEGEND_FONT_SCALE = 0.5
LEGEND_THICKNESS = 1
LEGEND_LINE_HEIGHT = 25
LEGEND_MARGIN = 10
LEGEND_BOX_SIZE = 15

# YOLO class mapping (from APCData)
YOLO_CLASSES = {
    0: "NILM",
    1: "ASCUS",
    2: "ASCH",
    3: "LSIL",
    4: "HSIL",
    5: "SCC"
}


# =============================================================================
#  GROUND TRUTH FUNCTIONS
# =============================================================================

def load_yolo_annotations(label_path: Path, img_width: int, img_height: int):
    """
    Load YOLO format annotations and convert to pixel coordinates.

    Args:
        label_path: Path to .txt label file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of (class_id, class_name, x1, y1, x2, y2) tuples
    """
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                class_name = YOLO_CLASSES.get(class_id, f"Class_{class_id}")
                annotations.append((class_id, class_name, x1, y1, x2, y2))

    return annotations


def draw_ground_truth(
    image: np.ndarray,
    annotations: list,
    border_width: int = 2,
    show_labels: bool = True
) -> np.ndarray:
    """
    Draw ground truth bounding boxes on image.

    Args:
        image: Image to annotate (RGB)
        annotations: List from load_yolo_annotations
        border_width: Line thickness
        show_labels: Show class labels on boxes

    Returns:
        Annotated image
    """
    annotated = image.copy()

    for class_id, class_name, x1, y1, x2, y2 in annotations:
        # Get color for this class
        color = CLASS_COLORS.get(class_name, (200, 200, 200))
        color_bgr = (color[2], color[1], color[0])

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, border_width)

        # Draw label
        if show_labels:
            label = class_name
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Background for text
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 4),
                (x1 + text_width + 4, y1),
                color_bgr,
                -1
            )

            # Text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )

    return annotated


def create_comparison_image(
    image: np.ndarray,
    diagnosis: ImageDiagnosis,
    annotations: list,
    tile_size: int = 224,
    alpha: float = 0.25
) -> np.ndarray:
    """
    Create side-by-side comparison: GT (left) vs Predictions (right).

    Args:
        image: Original image (RGB)
        diagnosis: Predictions from pipeline
        annotations: Ground truth annotations
        tile_size: Patch size
        alpha: Overlay transparency

    Returns:
        Combined comparison image
    """
    h, w = image.shape[:2]

    # Left side: Ground Truth
    gt_image = draw_ground_truth(image.copy(), annotations, border_width=3)

    # Add GT label (Cell-Level)
    cv2.putText(
        gt_image,
        "GROUND TRUTH (Cell-Level)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3
    )
    cv2.putText(
        gt_image,
        "GROUND TRUTH (Cell-Level)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    # Count GT by class
    gt_counts = {}
    for _, class_name, _, _, _, _ in annotations:
        gt_counts[class_name] = gt_counts.get(class_name, 0) + 1

    gt_text = f"Annotated Cells: {len(annotations)}"
    cv2.putText(gt_image, gt_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(gt_image, gt_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Right side: Predictions
    pred_image = draw_patch_overlay(
        image.copy(), diagnosis,
        tile_size=tile_size, alpha=alpha,
        show_empty=False, color_mode="class"
    )

    # Add Prediction label (Patch-Level)
    cv2.putText(
        pred_image,
        "PREDICTIONS (Patch-Level)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3
    )
    cv2.putText(
        pred_image,
        "PREDICTIONS (Patch-Level)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    pred_text = f"Patches with cells: {diagnosis.patches_with_cells}"
    cv2.putText(pred_image, pred_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(pred_image, pred_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Combine side by side
    combined = np.hstack([gt_image, pred_image])

    return combined


# =============================================================================
#  VISUALIZATION FUNCTIONS
# =============================================================================

def draw_patch_overlay(
    image: np.ndarray,
    diagnosis: ImageDiagnosis,
    tile_size: int = 224,
    alpha: float = 0.3,
    border_width: int = 2,
    show_empty: bool = False,
    color_mode: str = "severity"
) -> np.ndarray:
    """
    Draw colored rectangles on patches.

    Args:
        image: Original image (RGB)
        diagnosis: ImageDiagnosis from unified pipeline
        tile_size: Size of patches (default 224)
        alpha: Transparency of fill (0=transparent, 1=opaque)
        border_width: Width of rectangle border
        show_empty: Whether to show empty patches
        color_mode: "severity" or "class"

    Returns:
        Annotated image
    """
    # Create overlay for semi-transparent fill
    overlay = image.copy()
    annotated = image.copy()

    for patch in diagnosis.patch_results:
        if not patch.has_cells and not show_empty:
            continue

        x, y = patch.x, patch.y
        x2, y2 = x + tile_size, y + tile_size

        # Clip to image bounds
        x2 = min(x2, image.shape[1])
        y2 = min(y2, image.shape[0])

        # Get color based on mode
        if color_mode == "class":
            color = CLASS_COLORS.get(patch.class_name, (180, 180, 180))
        else:
            color = SEVERITY_COLORS.get(patch.severity, (180, 180, 180))

        # Convert RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])

        # Draw filled rectangle on overlay
        if patch.has_cells:
            cv2.rectangle(overlay, (x, y), (x2, y2), color_bgr, -1)
        elif show_empty:
            cv2.rectangle(overlay, (x, y), (x2, y2), (180, 180, 180), -1)

        # Draw border
        cv2.rectangle(annotated, (x, y), (x2, y2), color_bgr, border_width)

    # Blend overlay with original
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

    return annotated


def draw_cell_level_overlay(
    image: np.ndarray,
    diagnosis: ImageDiagnosis,
    tile_size: int = 224,
    alpha: float = 0.4,
    min_nucleus_area: int = 50,
    max_nucleus_area: int = 5000
) -> tuple:
    """
    Draw nuclei contours detected via H-Channel (V15.3).

    For each patch with cells, detects nuclei using Ruifrok deconvolution
    and draws their contours with the class color inherited from the patch.

    Args:
        image: Original image (RGB)
        diagnosis: ImageDiagnosis from unified pipeline
        tile_size: Size of patches
        alpha: Transparency of contour fill (0=transparent, 1=opaque)
        min_nucleus_area: Minimum nucleus area in pixels
        max_nucleus_area: Maximum nucleus area in pixels

    Returns:
        tuple: (annotated_image, nuclei_count, nuclei_by_class)
    """
    annotated = image.copy()
    all_nuclei = []
    nuclei_by_class = {}

    for patch in diagnosis.patch_results:
        if not patch.has_cells:
            continue

        x, y = patch.x, patch.y
        x2 = min(x + tile_size, image.shape[1])
        y2 = min(y + tile_size, image.shape[0])

        # Extract patch
        patch_rgb = image[y:y2, x:x2].copy()

        # Skip if patch is too small
        if patch_rgb.shape[0] < 50 or patch_rgb.shape[1] < 50:
            continue

        # Detect nuclei in this patch
        nuclei = detect_nuclei_for_visualization(
            patch_rgb,
            predicted_class=patch.class_name,
            min_nucleus_area=min_nucleus_area,
            max_nucleus_area=max_nucleus_area
        )

        # Offset nuclei coordinates to image space
        for nucleus in nuclei:
            # Offset contour
            nucleus['contour'] = nucleus['contour'] + np.array([x, y])
            # Offset centroid
            cx, cy = nucleus['centroid']
            nucleus['centroid'] = (cx + x, cy + y)

            all_nuclei.append(nucleus)

            # Count by class
            cls = nucleus['class']
            nuclei_by_class[cls] = nuclei_by_class.get(cls, 0) + 1

    # Render all nuclei on image
    if all_nuclei:
        annotated = render_nuclei_overlay(annotated, all_nuclei, alpha=alpha)

    return annotated, len(all_nuclei), nuclei_by_class


def draw_cell_level_legend(
    image: np.ndarray,
    nuclei_by_class: dict,
    total_nuclei: int,
    position: str = "top-right"
) -> np.ndarray:
    """
    Draw legend for cell-level visualization.

    Args:
        image: Annotated image
        nuclei_by_class: Dict of {class_name: count}
        total_nuclei: Total number of nuclei detected
        position: Legend position

    Returns:
        Image with legend
    """
    h, w = image.shape[:2]
    annotated = image.copy()

    # Prepare legend items
    items = []
    for class_name in ["NILM", "ASCUS", "ASCH", "LSIL", "HSIL", "SCC"]:
        count = nuclei_by_class.get(class_name, 0)
        if count > 0:
            color = CLASS_COLORS.get(class_name, (180, 180, 180))
            items.append((class_name, count, color))

    if not items:
        items.append(("No nuclei", 0, (180, 180, 180)))

    # Calculate legend dimensions
    legend_width = 200
    legend_height = (len(items) + 1) * LEGEND_LINE_HEIGHT + 2 * LEGEND_MARGIN  # +1 for total

    # Position
    if "right" in position:
        x_start = w - legend_width - LEGEND_MARGIN
    else:
        x_start = LEGEND_MARGIN

    if "bottom" in position:
        y_start = h - legend_height - LEGEND_MARGIN
    else:
        y_start = LEGEND_MARGIN

    # Draw semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (x_start, y_start),
        (x_start + legend_width, y_start + legend_height),
        (255, 255, 255),
        -1
    )
    cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

    # Draw border
    cv2.rectangle(
        annotated,
        (x_start, y_start),
        (x_start + legend_width, y_start + legend_height),
        (0, 0, 0),
        1
    )

    # Title
    y_pos = y_start + LEGEND_MARGIN + LEGEND_LINE_HEIGHT // 2
    cv2.putText(
        annotated,
        f"Nuclei Detected: {total_nuclei}",
        (x_start + LEGEND_MARGIN, y_pos + 5),
        LEGEND_FONT,
        LEGEND_FONT_SCALE,
        (0, 0, 0),
        LEGEND_THICKNESS + 1
    )
    y_pos += LEGEND_LINE_HEIGHT

    # Draw legend items
    for name, count, color in items:
        # Color box
        color_bgr = (color[2], color[1], color[0])
        cv2.rectangle(
            annotated,
            (x_start + LEGEND_MARGIN, y_pos - LEGEND_BOX_SIZE // 2),
            (x_start + LEGEND_MARGIN + LEGEND_BOX_SIZE, y_pos + LEGEND_BOX_SIZE // 2),
            color_bgr,
            -1
        )
        cv2.rectangle(
            annotated,
            (x_start + LEGEND_MARGIN, y_pos - LEGEND_BOX_SIZE // 2),
            (x_start + LEGEND_MARGIN + LEGEND_BOX_SIZE, y_pos + LEGEND_BOX_SIZE // 2),
            (0, 0, 0),
            1
        )

        # Text
        text = f"{name}: {count}"
        cv2.putText(
            annotated,
            text,
            (x_start + LEGEND_MARGIN + LEGEND_BOX_SIZE + 10, y_pos + 5),
            LEGEND_FONT,
            LEGEND_FONT_SCALE,
            (0, 0, 0),
            LEGEND_THICKNESS
        )

        y_pos += LEGEND_LINE_HEIGHT

    return annotated


def draw_legend(
    image: np.ndarray,
    diagnosis: ImageDiagnosis,
    position: str = "top-right",
    color_mode: str = "severity"
) -> np.ndarray:
    """
    Draw legend with class distribution.

    Args:
        image: Annotated image
        diagnosis: ImageDiagnosis
        position: "top-right", "top-left", "bottom-right", "bottom-left"
        color_mode: "severity" or "class"

    Returns:
        Image with legend
    """
    h, w = image.shape[:2]
    annotated = image.copy()

    # Prepare legend items
    if color_mode == "class":
        items = []
        for class_name, count in diagnosis.class_distribution.items():
            if count > 0:
                color = CLASS_COLORS.get(class_name, (180, 180, 180))
                items.append((class_name, count, color))
        # Add empty count
        empty_count = diagnosis.total_patches - diagnosis.patches_with_cells
        if empty_count > 0:
            items.append(("Empty", empty_count, (180, 180, 180)))
    else:
        # Severity mode
        normal_count = diagnosis.class_distribution.get("NILM", 0)
        low_grade_count = (
            diagnosis.class_distribution.get("ASCUS", 0) +
            diagnosis.class_distribution.get("LSIL", 0)
        )
        high_grade_count = (
            diagnosis.class_distribution.get("ASCH", 0) +
            diagnosis.class_distribution.get("HSIL", 0) +
            diagnosis.class_distribution.get("SCC", 0)
        )
        empty_count = diagnosis.total_patches - diagnosis.patches_with_cells

        items = [
            ("Normal (NILM)", normal_count, SEVERITY_COLORS["Normal"]),
            ("Low-grade", low_grade_count, SEVERITY_COLORS["Low-grade"]),
            ("High-grade", high_grade_count, SEVERITY_COLORS["High-grade"]),
            ("Empty", empty_count, SEVERITY_COLORS["Empty"])
        ]

    # Calculate legend dimensions
    legend_width = 200
    legend_height = len(items) * LEGEND_LINE_HEIGHT + 2 * LEGEND_MARGIN

    # Position
    if "right" in position:
        x_start = w - legend_width - LEGEND_MARGIN
    else:
        x_start = LEGEND_MARGIN

    if "bottom" in position:
        y_start = h - legend_height - LEGEND_MARGIN
    else:
        y_start = LEGEND_MARGIN

    # Draw semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (x_start, y_start),
        (x_start + legend_width, y_start + legend_height),
        (255, 255, 255),
        -1
    )
    cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

    # Draw border
    cv2.rectangle(
        annotated,
        (x_start, y_start),
        (x_start + legend_width, y_start + legend_height),
        (0, 0, 0),
        1
    )

    # Draw legend items
    y_pos = y_start + LEGEND_MARGIN + LEGEND_LINE_HEIGHT // 2

    for name, count, color in items:
        # Color box
        color_bgr = (color[2], color[1], color[0])
        cv2.rectangle(
            annotated,
            (x_start + LEGEND_MARGIN, y_pos - LEGEND_BOX_SIZE // 2),
            (x_start + LEGEND_MARGIN + LEGEND_BOX_SIZE, y_pos + LEGEND_BOX_SIZE // 2),
            color_bgr,
            -1
        )
        cv2.rectangle(
            annotated,
            (x_start + LEGEND_MARGIN, y_pos - LEGEND_BOX_SIZE // 2),
            (x_start + LEGEND_MARGIN + LEGEND_BOX_SIZE, y_pos + LEGEND_BOX_SIZE // 2),
            (0, 0, 0),
            1
        )

        # Text
        text = f"{name}: {count}"
        cv2.putText(
            annotated,
            text,
            (x_start + LEGEND_MARGIN + LEGEND_BOX_SIZE + 10, y_pos + 5),
            LEGEND_FONT,
            LEGEND_FONT_SCALE,
            (0, 0, 0),
            LEGEND_THICKNESS
        )

        y_pos += LEGEND_LINE_HEIGHT

    return annotated


def draw_diagnosis_banner(
    image: np.ndarray,
    diagnosis: ImageDiagnosis,
    position: str = "bottom"
) -> np.ndarray:
    """
    Draw diagnosis banner with result and recommendation.

    Args:
        image: Annotated image
        diagnosis: ImageDiagnosis
        position: "top" or "bottom"

    Returns:
        Image with banner
    """
    h, w = image.shape[:2]

    # Banner settings
    banner_height = 80
    banner_padding = 15

    # Create banner
    if position == "bottom":
        banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
        banner[:] = (40, 40, 40)  # Dark gray background
    else:
        banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
        banner[:] = (40, 40, 40)

    # Determine colors based on result
    if diagnosis.severity_result == "High-grade":
        result_color = (0, 0, 255)  # Red
        status_text = "ABNORMAL - HIGH-GRADE"
    elif diagnosis.severity_result == "Low-grade":
        result_color = (0, 200, 255)  # Orange
        status_text = "ABNORMAL - LOW-GRADE"
    else:
        result_color = (0, 200, 0)  # Green
        status_text = "NORMAL"

    # Draw status
    cv2.putText(
        banner,
        f"DIAGNOSIS: {status_text}",
        (banner_padding, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        result_color,
        2
    )

    # Draw recommendation
    cv2.putText(
        banner,
        diagnosis.recommendation,
        (banner_padding, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    # Draw stats on right side
    stats_text = f"Patches: {diagnosis.patches_with_cells}/{diagnosis.total_patches} | Abnormal: {diagnosis.abnormal_patches}"
    text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(
        banner,
        stats_text,
        (w - text_size[0] - banner_padding, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )

    # Combine banner with image
    if position == "bottom":
        result = np.vstack([image, banner])
    else:
        result = np.vstack([banner, image])

    return result


def visualize_diagnosis(
    image_path: str,
    diagnosis: ImageDiagnosis,
    output_path: str,
    tile_size: int = 224,
    alpha: float = 0.25,
    color_mode: str = "severity",
    show_empty: bool = False,
    show_legend: bool = True,
    show_banner: bool = True,
    cell_level: bool = False
) -> dict:
    """
    Generate complete visualization for a diagnosis.

    Args:
        image_path: Path to original image
        diagnosis: ImageDiagnosis from pipeline
        output_path: Where to save the visualization
        tile_size: Patch size
        alpha: Overlay transparency
        color_mode: "severity" or "class"
        show_empty: Show empty patches
        show_legend: Add legend
        show_banner: Add diagnosis banner
        cell_level: Use V15.3 cell-level visualization (nuclei contours)

    Returns:
        dict with 'path' and optionally 'nuclei_count'
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = {'path': output_path}

    if cell_level:
        # V15.3: Cell-level visualization with nuclei contours
        annotated, nuclei_count, nuclei_by_class = draw_cell_level_overlay(
            image, diagnosis,
            tile_size=tile_size,
            alpha=alpha
        )

        result['nuclei_count'] = nuclei_count
        result['nuclei_by_class'] = nuclei_by_class

        # Add cell-level legend
        if show_legend:
            annotated = draw_cell_level_legend(
                annotated, nuclei_by_class, nuclei_count
            )
    else:
        # V15.2: Patch-level visualization with rectangles
        annotated = draw_patch_overlay(
            image, diagnosis,
            tile_size=tile_size,
            alpha=alpha,
            show_empty=show_empty,
            color_mode=color_mode
        )

        # Add legend
        if show_legend:
            annotated = draw_legend(annotated, diagnosis, color_mode=color_mode)

    # Add banner
    if show_banner:
        annotated = draw_diagnosis_banner(annotated, diagnosis)

    # Convert to BGR for saving
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # Save
    cv2.imwrite(output_path, annotated_bgr)

    return result


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Cytology Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image
    python scripts/cytology/12_visualize_predictions.py \\
        --image data/raw/apcdata/APCData_YOLO/val/images/001.jpg \\
        --output results/visualizations/

    # Directory of images
    python scripts/cytology/12_visualize_predictions.py \\
        --input_dir data/raw/apcdata/APCData_YOLO/val/images \\
        --output results/visualizations/ \\
        --max_images 10

    # Compare with ground truth (side-by-side)
    python scripts/cytology/12_visualize_predictions.py \\
        --input_dir data/raw/apcdata/APCData_YOLO/val/images \\
        --output results/visualizations/ \\
        --compare_gt \\
        --max_images 5

    # Fine-grained class colors
    python scripts/cytology/12_visualize_predictions.py \\
        --image path/to/image.jpg \\
        --color_mode class
        """
    )

    # Input
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--input_dir", type=str, help="Path to directory of images")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Max number of images to process")

    # Models
    parser.add_argument("--cell_triage", type=str,
                        default="models/cytology/cell_triage.pt",
                        help="Path to Cell Triage model")
    parser.add_argument("--bethesda", type=str,
                        default="models/cytology/multihead_bethesda_combined.pt",
                        help="Path to MultiHead Bethesda model")

    # Output
    parser.add_argument("--output", type=str, default="results/visualizations",
                        help="Output directory")

    # Visualization options
    parser.add_argument("--color_mode", type=str, default="severity",
                        choices=["severity", "class"],
                        help="Color by severity or fine-grained class")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Overlay transparency (0-1)")
    parser.add_argument("--show_empty", action="store_true",
                        help="Show empty patches (gray)")
    parser.add_argument("--no_legend", action="store_true",
                        help="Hide legend")
    parser.add_argument("--no_banner", action="store_true",
                        help="Hide diagnosis banner")
    parser.add_argument("--cell_level", action="store_true",
                        help="V15.3: Use cell-level visualization with nuclei contours (instead of patch rectangles)")

    # Pipeline options
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--triage_threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")

    # Ground truth comparison
    parser.add_argument("--compare_gt", action="store_true",
                        help="Show side-by-side comparison with ground truth")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Path to YOLO labels directory (auto-detected if not specified)")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  CYTOLOGY PREDICTION VISUALIZATION")
    if args.cell_level:
        print("  V15.3 Cell-Level (Nuclei Contours via H-Channel)")
    else:
        print("  V15.2 Patch-Level (Rectangle Overlays)")
    print("=" * 80)

    # Validate inputs
    if args.image is None and args.input_dir is None:
        print("  [ERROR] Must specify --image or --input_dir")
        return 1

    # Get image paths
    if args.image:
        image_paths = [Path(args.image)]
    else:
        input_dir = Path(args.input_dir)
        image_paths = sorted(
            list(input_dir.glob("*.jpg")) +
            list(input_dir.glob("*.png")) +
            list(input_dir.glob("*.jpeg"))
        )
        if args.max_images:
            image_paths = image_paths[:args.max_images]

    print(f"  [INFO] Found {len(image_paths)} images to process")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check model files exist
    if not Path(args.cell_triage).exists():
        print(f"  [ERROR] Cell Triage model not found: {args.cell_triage}")
        return 1
    if not Path(args.bethesda).exists():
        print(f"  [ERROR] Bethesda model not found: {args.bethesda}")
        return 1

    # Initialize pipeline
    print("\n" + "-" * 40)
    print("  Loading Models...")
    print("-" * 40)

    try:
        pipeline = UnifiedInferencePipeline(
            cell_triage_path=args.cell_triage,
            bethesda_path=args.bethesda,
            device=args.device,
            triage_threshold=args.triage_threshold
        )
    except Exception as e:
        print(f"  [ERROR] Failed to load models: {e}")
        return 1

    # Process images
    print("\n" + "-" * 40)
    print("  Processing Images...")
    print("-" * 40)

    results_summary = []

    for i, image_path in enumerate(image_paths):
        print(f"\n  [{i+1}/{len(image_paths)}] {image_path.name}")

        try:
            # Run inference
            diagnosis = pipeline.run(
                str(image_path),
                tile_size=args.tile_size,
                stride=args.stride,
                batch_size=args.batch_size
            )

            # Generate visualization
            output_filename = f"{image_path.stem}_annotated.jpg"
            output_path = output_dir / output_filename

            # Load image for comparison
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]

            # Check for ground truth comparison
            if args.compare_gt:
                # Auto-detect labels directory if not specified
                if args.labels_dir:
                    labels_dir = Path(args.labels_dir)
                else:
                    # Try to find labels/ sibling to images/
                    if image_path.parent.name == "images":
                        labels_dir = image_path.parent.parent / "labels"
                    else:
                        labels_dir = image_path.parent / "labels"

                # Load ground truth annotations
                label_path = labels_dir / f"{image_path.stem}.txt"
                annotations = load_yolo_annotations(label_path, img_w, img_h)

                # Create comparison image
                comparison = create_comparison_image(
                    image, diagnosis, annotations,
                    tile_size=args.tile_size, alpha=args.alpha
                )

                # Save comparison
                output_filename = f"{image_path.stem}_comparison.jpg"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

                print(f"    GT cells: {len(annotations)}")
                print(f"    Result: {diagnosis.binary_result} - {diagnosis.severity_result}")
                print(f"    Patches: {diagnosis.patches_with_cells}/{diagnosis.total_patches} with cells")
                print(f"    Saved: {output_path}")
            else:
                # Standard visualization (patch-level or cell-level)
                vis_result = visualize_diagnosis(
                    str(image_path),
                    diagnosis,
                    str(output_path),
                    tile_size=args.tile_size,
                    alpha=args.alpha,
                    color_mode=args.color_mode,
                    show_empty=args.show_empty,
                    show_legend=not args.no_legend,
                    show_banner=not args.no_banner,
                    cell_level=args.cell_level
                )

                print(f"    Result: {diagnosis.binary_result} - {diagnosis.severity_result}")
                print(f"    Patches: {diagnosis.patches_with_cells}/{diagnosis.total_patches} with cells")
                if args.cell_level and 'nuclei_count' in vis_result:
                    print(f"    Nuclei detected: {vis_result['nuclei_count']}")
                print(f"    Saved: {output_path}")

            results_summary.append({
                "image": image_path.name,
                "result": diagnosis.binary_result,
                "severity": diagnosis.severity_result,
                "output": str(output_path)
            })

        except Exception as e:
            print(f"    [ERROR] {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    normal_count = sum(1 for r in results_summary if r["result"] == "NORMAL")
    abnormal_count = len(results_summary) - normal_count
    high_grade_count = sum(1 for r in results_summary if r["severity"] == "High-grade")

    print(f"  Processed: {len(results_summary)} images")
    print(f"  Normal: {normal_count}")
    print(f"  Abnormal: {abnormal_count}")
    print(f"    - Low-grade: {abnormal_count - high_grade_count}")
    print(f"    - High-grade: {high_grade_count}")
    print(f"\n  Output directory: {output_dir}")

    print("\n" + "=" * 80)
    print("  VISUALIZATION COMPLETE")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
