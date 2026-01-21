"""
CellPose Validation on APCData — Detection Rate Validation

Ce script valide CellPose sur APCData (point annotations):
1. Load CellPose model (pré-entraîné 'nuclei')
2. Détection sur images complètes (2048×1532)
3. Matching détections vs GT points (distance threshold)
4. Calculer métriques (Detection Rate, Precision, F1)
5. Générer rapport + visualisations

DIFFERENCE avec 00b_validate_cellpose.py:
- APCData = multi-cellules par image (pas 1 cellule isolée)
- Annotations = points (nucleus_x, nucleus_y), pas masques complets
- Objectif = valider que CellPose détecte TOUS les noyaux

Author: V14 Cytology Branch
Date: 2026-01-21
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from cellpose import models
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# Bethesda class mapping (APCData uses "Negative" instead of "NILM")
CLASS_MAPPING = {
    "Negative": "NILM",
    "NILM": "NILM",
    "ASCUS": "ASCUS",
    "ASCH": "ASCH",
    "LSIL": "LSIL",
    "HSIL": "HSIL",
    "SCC": "SCC"
}

# Binary classification
ABNORMAL_CLASSES = {"ASCUS", "ASCH", "LSIL", "HSIL", "SCC"}


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

# YOLO class mapping (from classes.txt)
YOLO_CLASS_MAPPING = {
    0: "ASC-US",
    1: "HSIL",
    2: "LSIL",
    3: "Negative",
    4: "SCC"
}


def load_yolo_annotations(data_dir: str, image_width: int = 2048, image_height: int = 1532) -> Dict[str, List[Dict]]:
    """
    Load APCData annotations from YOLO format.

    YOLO format: class_id x_center y_center width height (normalized 0-1)

    Args:
        data_dir: Path to APCData_YOLO/ directory
        image_width: Image width for denormalization
        image_height: Image height for denormalization

    Returns:
        Dict mapping image_filename to list of cell annotations
    """
    labels_dir = os.path.join(data_dir, 'labels')
    images_dir = os.path.join(data_dir, 'images')

    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"YOLO labels directory not found: {labels_dir}")

    annotations = {}

    # Get all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt') and not f.endswith(':Zone.Identifier')]

    for label_file in label_files:
        # Match label to image (same name, different extension)
        base_name = label_file.replace('.txt', '')
        image_file = base_name + '.jpg'

        # Check if image exists
        image_path = os.path.join(images_dir, image_file)
        if not os.path.exists(image_path):
            # Try png
            image_file = base_name + '.png'
            image_path = os.path.join(images_dir, image_file)
            if not os.path.exists(image_path):
                continue

        # Read label file
        label_path = os.path.join(labels_dir, label_file)
        cells = []
        cell_id = 0

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    # width = float(parts[3])  # Not needed for point matching
                    # height = float(parts[4])

                    # Denormalize to pixel coordinates
                    nucleus_x = int(x_center * image_width)
                    nucleus_y = int(y_center * image_height)

                    bethesda_class = YOLO_CLASS_MAPPING.get(class_id, f"class_{class_id}")

                    cells.append({
                        'cell_id': cell_id,
                        'class': bethesda_class,
                        'nucleus_x': nucleus_x,
                        'nucleus_y': nucleus_y
                    })
                    cell_id += 1

        if cells:
            annotations[image_file] = cells

    return annotations


def load_apcdata_annotations(labels_dir: str) -> Dict[str, List[Dict]]:
    """
    Load APCData annotations from CSV files.

    Args:
        labels_dir: Path to APCData_points/labels/csv/

    Returns:
        Dict mapping image_filename to list of cell annotations
    """
    annotations = {}
    csv_dir = os.path.join(labels_dir, 'csv')

    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    csv_files = list(Path(csv_dir).glob("*.csv"))

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            image_filename = row['image_filename']

            if image_filename not in annotations:
                annotations[image_filename] = []

            # Map class name
            bethesda_class = CLASS_MAPPING.get(row['bethesda_system'], row['bethesda_system'])

            annotations[image_filename].append({
                'cell_id': row['cell_id'],
                'class': bethesda_class,
                'nucleus_x': row['nucleus_x'],
                'nucleus_y': row['nucleus_y']
            })

    return annotations


def load_apcdata_annotations_json(labels_dir: str) -> Dict[str, List[Dict]]:
    """
    Load APCData annotations from JSON files.

    Args:
        labels_dir: Path to APCData_points/labels/

    Returns:
        Dict mapping image_filename to list of cell annotations
    """
    annotations = {}
    json_dir = os.path.join(labels_dir, 'json')

    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    json_files = list(Path(json_dir).glob("*.json"))

    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # JSON structure: list of image entries
        for entry in data:
            image_name = entry.get('image_name', entry.get('image_filename'))

            if image_name not in annotations:
                annotations[image_name] = []

            for cell in entry.get('classifications', []):
                bethesda_class = CLASS_MAPPING.get(cell['bethesda_system'], cell['bethesda_system'])

                annotations[image_name].append({
                    'cell_id': cell['cell_id'],
                    'class': bethesda_class,
                    'nucleus_x': cell['nucleus_x'],
                    'nucleus_y': cell['nucleus_y']
                })

    return annotations


# ═════════════════════════════════════════════════════════════════════════════
#  MATCHING ALGORITHM
# ═════════════════════════════════════════════════════════════════════════════

def match_detections_to_gt(
    detected_centroids: np.ndarray,
    gt_points: np.ndarray,
    max_distance: float = 50.0
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match CellPose detections to ground truth points using Hungarian algorithm.

    Args:
        detected_centroids: (N, 2) array of detected centroids (x, y)
        gt_points: (M, 2) array of GT points (x, y)
        max_distance: Maximum distance for a valid match (pixels)

    Returns:
        matches: List of (detection_idx, gt_idx) tuples
        unmatched_detections: List of detection indices (False Positives)
        unmatched_gt: List of GT indices (False Negatives)
    """
    if len(detected_centroids) == 0:
        # No detections: all GT are missed
        return [], [], list(range(len(gt_points)))

    if len(gt_points) == 0:
        # No GT: all detections are FP
        return [], list(range(len(detected_centroids))), []

    # Compute distance matrix
    dist_matrix = cdist(detected_centroids, gt_points, metric='euclidean')

    # Hungarian algorithm for optimal assignment
    det_indices, gt_indices = linear_sum_assignment(dist_matrix)

    # Filter by max_distance
    matches = []
    unmatched_detections = set(range(len(detected_centroids)))
    unmatched_gt = set(range(len(gt_points)))

    for det_idx, gt_idx in zip(det_indices, gt_indices):
        if dist_matrix[det_idx, gt_idx] <= max_distance:
            matches.append((det_idx, gt_idx))
            unmatched_detections.discard(det_idx)
            unmatched_gt.discard(gt_idx)

    return matches, list(unmatched_detections), list(unmatched_gt)


# ═════════════════════════════════════════════════════════════════════════════
#  CELLPOSE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def run_cellpose_detection(
    model,
    image: np.ndarray,
    diameter: float = 30,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run CellPose detection on an image.

    Args:
        model: CellPose model
        image: RGB image (H, W, 3)
        diameter: Expected nucleus diameter
        flow_threshold: Flow threshold
        cellprob_threshold: Cell probability threshold

    Returns:
        masks: Labeled mask (H, W) where each value is a cell ID
        detections: List of detection info (centroid, area, etc.)
    """
    # CellPose v4.x API
    masks, flows, styles = model.eval(
        image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )

    # Extract detection info
    detections = []

    if masks.max() > 0:
        props = regionprops(masks)

        for prop in props:
            cy, cx = prop.centroid
            detections.append({
                'id': prop.label,
                'centroid_x': cx,
                'centroid_y': cy,
                'area': prop.area,
                'bbox': prop.bbox
            })

    return masks, detections


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def validate_cellpose_on_apcdata(
    data_dir: str,
    diameter: float = 30,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    max_distance: float = 50.0,
    n_samples: Optional[int] = None,
    save_visualizations: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Validate CellPose detection on APCData.

    Args:
        data_dir: Path to APCData_points/ directory
        diameter: CellPose diameter parameter
        flow_threshold: Flow threshold
        cellprob_threshold: Cell probability threshold
        max_distance: Max distance for matching (pixels)
        n_samples: Number of images to test (None = all)
        save_visualizations: Save sample visualizations
        output_dir: Output directory for visualizations

    Returns:
        results: Dict with detection metrics
    """
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Try loading annotations in order: YOLO > JSON > CSV
    annotations = None

    # Check if this is YOLO format (has labels/*.txt files)
    yolo_labels_dir = os.path.join(data_dir, 'labels')
    if os.path.exists(yolo_labels_dir):
        txt_files = [f for f in os.listdir(yolo_labels_dir) if f.endswith('.txt') and not f.endswith(':Zone.Identifier')]
        if txt_files:
            # Get image dimensions from first image
            sample_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png')) and not ':Zone' in f]
            if sample_images:
                from PIL import Image as PILImage
                sample_img = PILImage.open(os.path.join(images_dir, sample_images[0]))
                img_width, img_height = sample_img.size
                annotations = load_yolo_annotations(data_dir, img_width, img_height)
                print(f"  Loaded annotations from YOLO format (image size: {img_width}x{img_height})")

    # Fall back to JSON/CSV
    if annotations is None:
        try:
            annotations = load_apcdata_annotations_json(labels_dir)
            print("  Loaded annotations from JSON format")
        except FileNotFoundError:
            annotations = load_apcdata_annotations(labels_dir)
            print("  Loaded annotations from CSV format")

    # Filter to images that have annotations (exclude Zone.Identifier files)
    all_image_files = [f for f in os.listdir(images_dir)
                       if f.endswith(('.png', '.jpg', '.jpeg')) and ':Zone' not in f]
    image_files = [f for f in all_image_files if f in annotations]

    # Debug info if no matches
    if len(image_files) == 0 and len(all_image_files) > 0 and len(annotations) > 0:
        print(f"\n  DEBUG: No matching images found!")
        print(f"    Images in folder: {len(all_image_files)}")
        print(f"    Sample image names: {all_image_files[:3]}")
        print(f"    Annotations loaded: {len(annotations)}")
        print(f"    Sample annotation keys: {list(annotations.keys())[:3]}")

    if n_samples is not None and n_samples < len(image_files):
        image_files = image_files[:n_samples]

    print(f"\n  Validating CellPose on {len(image_files)} images")
    print(f"  Model: CellPose 'nuclei' (pré-entraîné)")
    print(f"  Diameter: {diameter}")
    print(f"  Flow threshold: {flow_threshold}")
    print(f"  Max matching distance: {max_distance} px")

    # Load CellPose model
    print("\n  Loading CellPose model...")
    model = models.CellposeModel(gpu=True)

    # Metrics accumulators
    total_gt = 0
    total_detected = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_image_results = []
    per_class_stats = {cls: {'tp': 0, 'fn': 0} for cls in CLASS_MAPPING.values()}

    match_distances = []  # For distance distribution analysis

    # Process each image
    for image_file in tqdm(image_files, desc="  Processing"):
        image_path = os.path.join(images_dir, image_file)
        image = np.array(Image.open(image_path))

        # Get GT annotations for this image
        gt_cells = annotations[image_file]
        gt_points = np.array([[c['nucleus_x'], c['nucleus_y']] for c in gt_cells])
        gt_classes = [c['class'] for c in gt_cells]

        # Run CellPose
        masks, detections = run_cellpose_detection(
            model, image, diameter, flow_threshold, cellprob_threshold
        )

        # Get detected centroids
        if len(detections) > 0:
            detected_centroids = np.array([[d['centroid_x'], d['centroid_y']] for d in detections])
        else:
            detected_centroids = np.array([]).reshape(0, 2)

        # Match detections to GT
        matches, unmatched_det, unmatched_gt = match_detections_to_gt(
            detected_centroids, gt_points, max_distance
        )

        # Compute metrics for this image
        n_gt = len(gt_points)
        n_detected = len(detections)
        n_tp = len(matches)
        n_fp = len(unmatched_det)
        n_fn = len(unmatched_gt)

        total_gt += n_gt
        total_detected += n_detected
        total_tp += n_tp
        total_fp += n_fp
        total_fn += n_fn

        # Per-class metrics (only for matched cells)
        for det_idx, gt_idx in matches:
            gt_class = gt_classes[gt_idx]
            per_class_stats[gt_class]['tp'] += 1

            # Record match distance
            dist = np.linalg.norm(detected_centroids[det_idx] - gt_points[gt_idx])
            match_distances.append(dist)

        # For missed cells
        for gt_idx in unmatched_gt:
            gt_class = gt_classes[gt_idx]
            per_class_stats[gt_class]['fn'] += 1

        # Store per-image results
        detection_rate = n_tp / n_gt if n_gt > 0 else 0.0
        precision = n_tp / n_detected if n_detected > 0 else 0.0

        per_image_results.append({
            'image': image_file,
            'n_gt': n_gt,
            'n_detected': n_detected,
            'tp': n_tp,
            'fp': n_fp,
            'fn': n_fn,
            'detection_rate': detection_rate,
            'precision': precision
        })

        # Save visualization for first few images
        if save_visualizations and output_dir and len(per_image_results) <= 10:
            save_detection_visualization(
                image, gt_points, detected_centroids, matches,
                unmatched_det, unmatched_gt, image_file, output_dir
            )

    # Aggregate metrics
    detection_rate = total_tp / total_gt if total_gt > 0 else 0.0
    precision = total_tp / total_detected if total_detected > 0 else 0.0
    f1_score = 2 * precision * detection_rate / (precision + detection_rate) if (precision + detection_rate) > 0 else 0.0

    # Per-class detection rates
    per_class_detection_rate = {}
    for cls_name, stats in per_class_stats.items():
        total = stats['tp'] + stats['fn']
        if total > 0:
            per_class_detection_rate[cls_name] = {
                'tp': stats['tp'],
                'fn': stats['fn'],
                'total': total,
                'detection_rate': stats['tp'] / total
            }

    results = {
        'n_images': len(image_files),
        'total_gt_cells': total_gt,
        'total_detections': total_detected,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'detection_rate': detection_rate,
        'precision': precision,
        'f1_score': f1_score,
        'mean_match_distance': np.mean(match_distances) if match_distances else 0.0,
        'std_match_distance': np.std(match_distances) if match_distances else 0.0,
        'per_class_stats': per_class_detection_rate,
        'per_image_results': per_image_results,
        'parameters': {
            'diameter': diameter,
            'flow_threshold': flow_threshold,
            'cellprob_threshold': cellprob_threshold,
            'max_distance': max_distance
        }
    }

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def save_detection_visualization(
    image: np.ndarray,
    gt_points: np.ndarray,
    detected_centroids: np.ndarray,
    matches: List[Tuple[int, int]],
    unmatched_det: List[int],
    unmatched_gt: List[int],
    image_name: str,
    output_dir: str
):
    """
    Save visualization of detection results.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(image)

    # Draw GT points
    if len(gt_points) > 0:
        # Matched GT (green circles)
        matched_gt_idx = [gt_idx for _, gt_idx in matches]
        matched_gt = gt_points[matched_gt_idx] if matched_gt_idx else np.array([]).reshape(0, 2)
        if len(matched_gt) > 0:
            ax.scatter(matched_gt[:, 0], matched_gt[:, 1],
                      c='lime', s=100, marker='o', linewidths=2,
                      edgecolors='darkgreen', label=f'GT Matched ({len(matched_gt)})')

        # Missed GT (red circles)
        missed_gt = gt_points[unmatched_gt] if unmatched_gt else np.array([]).reshape(0, 2)
        if len(missed_gt) > 0:
            ax.scatter(missed_gt[:, 0], missed_gt[:, 1],
                      c='red', s=100, marker='o', linewidths=2,
                      edgecolors='darkred', label=f'GT Missed ({len(missed_gt)})')

    # Draw detected centroids
    if len(detected_centroids) > 0:
        # Matched detections (blue X)
        matched_det_idx = [det_idx for det_idx, _ in matches]
        matched_det = detected_centroids[matched_det_idx] if matched_det_idx else np.array([]).reshape(0, 2)
        if len(matched_det) > 0:
            ax.scatter(matched_det[:, 0], matched_det[:, 1],
                      c='cyan', s=80, marker='x', linewidths=2,
                      label=f'Detected TP ({len(matched_det)})')

        # False positive detections (orange X)
        fp_det = detected_centroids[unmatched_det] if unmatched_det else np.array([]).reshape(0, 2)
        if len(fp_det) > 0:
            ax.scatter(fp_det[:, 0], fp_det[:, 1],
                      c='orange', s=80, marker='x', linewidths=2,
                      label=f'Detected FP ({len(fp_det)})')

    # Draw match lines
    for det_idx, gt_idx in matches:
        det_point = detected_centroids[det_idx]
        gt_point = gt_points[gt_idx]
        ax.plot([det_point[0], gt_point[0]], [det_point[1], gt_point[1]],
               'g-', linewidth=1, alpha=0.5)

    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'{image_name}\nDetection Rate: {len(matches)}/{len(gt_points)} = {len(matches)/len(gt_points)*100:.1f}%')
    ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'detection_{Path(image_name).stem}.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_validation_results(results: Dict, output_dir: str):
    """
    Generate summary visualization of validation results.
    """
    # Skip plotting if no results
    if results['n_images'] == 0:
        print("\n  Skipping plots (no data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Summary metrics
    metrics = ['Detection Rate', 'Precision', 'F1 Score']
    values = [results['detection_rate'], results['precision'], results['f1_score']]
    colors = ['steelblue' if v >= 0.9 else 'orange' if v >= 0.8 else 'red' for v in values]

    axes[0, 0].bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 0].axhline(0.90, color='green', linestyle='--', linewidth=2, label='Target (90%)')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Detection Metrics Summary')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='y')

    for i, (m, v) in enumerate(zip(metrics, values)):
        axes[0, 0].text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Plot 2: Per-class detection rate
    class_names = list(results['per_class_stats'].keys())
    class_rates = [results['per_class_stats'][c]['detection_rate'] for c in class_names]
    class_totals = [results['per_class_stats'][c]['total'] for c in class_names]

    colors = ['green' if c == 'NILM' else 'orange' for c in class_names]
    bars = axes[0, 1].barh(class_names, class_rates, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0.90, color='green', linestyle='--', linewidth=2, label='Target')
    axes[0, 1].set_xlabel('Detection Rate')
    axes[0, 1].set_title('Detection Rate by Class')
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis='x')

    for i, (rate, total) in enumerate(zip(class_rates, class_totals)):
        axes[0, 1].text(rate + 0.02, i, f'{rate*100:.1f}% (n={total})', va='center', fontsize=9)

    # Plot 3: Per-image detection rate distribution
    per_image_rates = [r['detection_rate'] for r in results['per_image_results']]
    axes[1, 0].hist(per_image_rates, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(per_image_rates), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(per_image_rates)*100:.1f}%')
    axes[1, 0].axvline(0.90, color='green', linestyle='--', linewidth=2, label='Target (90%)')
    axes[1, 0].set_xlabel('Detection Rate')
    axes[1, 0].set_ylabel('Number of Images')
    axes[1, 0].set_title('Detection Rate Distribution (per image)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: TP/FP/FN breakdown
    labels = ['True Positives', 'False Positives', 'False Negatives']
    counts = [results['total_tp'], results['total_fp'], results['total_fn']]
    colors = ['green', 'orange', 'red']

    axes[1, 1].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=(0.05, 0.05, 0.05))
    axes[1, 1].set_title(f'Detection Breakdown\n(Total GT: {results["total_gt_cells"]}, Detections: {results["total_detections"]})')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cellpose_validation_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Summary plot saved: {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═════════════════════════════════════════════════════════════════════════════

def print_validation_report(results: Dict):
    """
    Print formatted validation report.
    """
    print("\n" + "=" * 80)
    print(" CELLPOSE VALIDATION REPORT (APCData)")
    print("=" * 80)

    print(f"\n  Images:       {results['n_images']}")
    print(f"  GT Cells:     {results['total_gt_cells']}")
    print(f"  Detections:   {results['total_detections']}")

    print(f"\n  {'Metric':<25} {'Value':<15} {'Status':<10}")
    print("-" * 80)

    dr_status = "PASS" if results['detection_rate'] >= 0.90 else "FAIL"
    pr_status = "PASS" if results['precision'] >= 0.85 else "FAIL"
    f1_status = "PASS" if results['f1_score'] >= 0.87 else "FAIL"

    print(f"  {'Detection Rate (Recall)':<25} {results['detection_rate']*100:>6.1f}%        {dr_status}")
    print(f"  {'Precision':<25} {results['precision']*100:>6.1f}%        {pr_status}")
    print(f"  {'F1 Score':<25} {results['f1_score']*100:>6.1f}%        {f1_status}")

    print(f"\n  {'True Positives (TP)':<25} {results['total_tp']:>6}")
    print(f"  {'False Positives (FP)':<25} {results['total_fp']:>6}")
    print(f"  {'False Negatives (FN)':<25} {results['total_fn']:>6}")

    print(f"\n  Mean Match Distance:  {results['mean_match_distance']:.1f} px (std: {results['std_match_distance']:.1f})")

    print("\n" + "-" * 80)
    print("  Per-Class Detection Rates:")
    print("-" * 80)
    print(f"  {'Class':<15} {'TP':<8} {'FN':<8} {'Total':<8} {'Det. Rate':<10}")
    print("-" * 80)

    for cls_name in ['NILM', 'ASCUS', 'ASCH', 'LSIL', 'HSIL', 'SCC']:
        if cls_name in results['per_class_stats']:
            stats = results['per_class_stats'][cls_name]
            print(f"  {cls_name:<15} {stats['tp']:<8} {stats['fn']:<8} {stats['total']:<8} {stats['detection_rate']*100:>6.1f}%")

    print("\n" + "=" * 80)

    # Overall verdict
    if results['detection_rate'] >= 0.90:
        print("  VALIDATION PASSED - CellPose detection is sufficient")
        print("  Next step: Run end-to-end pipeline validation")
    else:
        print("  VALIDATION FAILED - CellPose detection needs improvement")
        print(f"  Detection Rate {results['detection_rate']*100:.1f}% < 90% target")
        print("  Consider:")
        print("    - Adjusting diameter parameter")
        print("    - Lowering flow_threshold")
        print("    - Using cellprob_threshold=-1.0")

    print("=" * 80)


def generate_markdown_report(results: Dict, output_dir: str):
    """
    Generate markdown validation report.
    """
    # Skip if no results
    if results['n_images'] == 0:
        print("\n  Skipping markdown report (no data)")
        return

    report_path = os.path.join(output_dir, 'cellpose_validation_report.md')

    dr_status = "PASS" if results['detection_rate'] >= 0.90 else "FAIL"
    pr_status = "PASS" if results['precision'] >= 0.85 else "FAIL"

    with open(report_path, 'w') as f:
        f.write("# CellPose Validation Report — APCData\n\n")
        f.write(f"> Generated: 2026-01-21\n")
        f.write(f"> Status: **{dr_status}**\n\n")
        f.write("---\n\n")

        f.write("## Summary\n\n")
        f.write("| Metric | Value | Target | Status |\n")
        f.write("|--------|-------|--------|--------|\n")
        f.write(f"| Detection Rate | {results['detection_rate']*100:.1f}% | >90% | {dr_status} |\n")
        f.write(f"| Precision | {results['precision']*100:.1f}% | >85% | {pr_status} |\n")
        f.write(f"| F1 Score | {results['f1_score']*100:.1f}% | >87% | {'PASS' if results['f1_score'] >= 0.87 else 'FAIL'} |\n\n")

        f.write("## Dataset\n\n")
        f.write(f"- **Images:** {results['n_images']}\n")
        f.write(f"- **GT Cells:** {results['total_gt_cells']}\n")
        f.write(f"- **Detections:** {results['total_detections']}\n\n")

        f.write("## Breakdown\n\n")
        f.write("| Count | Value |\n")
        f.write("|-------|-------|\n")
        f.write(f"| True Positives (TP) | {results['total_tp']} |\n")
        f.write(f"| False Positives (FP) | {results['total_fp']} |\n")
        f.write(f"| False Negatives (FN) | {results['total_fn']} |\n\n")

        f.write("## Per-Class Detection Rates\n\n")
        f.write("| Class | TP | FN | Total | Detection Rate |\n")
        f.write("|-------|----|----|-------|---------------|\n")

        for cls_name in ['NILM', 'ASCUS', 'ASCH', 'LSIL', 'HSIL', 'SCC']:
            if cls_name in results['per_class_stats']:
                stats = results['per_class_stats'][cls_name]
                f.write(f"| {cls_name} | {stats['tp']} | {stats['fn']} | {stats['total']} | {stats['detection_rate']*100:.1f}% |\n")

        f.write("\n## Parameters\n\n")
        f.write("```json\n")
        f.write(json.dumps(results['parameters'], indent=2))
        f.write("\n```\n\n")

        f.write("## Next Steps\n\n")
        if results['detection_rate'] >= 0.90:
            f.write("1. Run end-to-end pipeline validation (`06_end_to_end_apcdata.py`)\n")
            f.write("2. Validate classification performance on detected cells\n")
        else:
            f.write("1. Adjust CellPose parameters (diameter, thresholds)\n")
            f.write("2. Consider image preprocessing (contrast enhancement)\n")
            f.write("3. Re-run validation\n")

        f.write("\n---\n\n")
        f.write("![Summary](cellpose_validation_summary.png)\n")

    print(f"\n  Markdown report saved: {report_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate CellPose detection on APCData"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw/apcdata/APCData_points',
        help='APCData_points directory'
    )
    parser.add_argument(
        '--diameter',
        type=float,
        default=30,
        help='CellPose diameter parameter (LBC cells ~30-40)'
    )
    parser.add_argument(
        '--flow_threshold',
        type=float,
        default=0.4,
        help='Flow threshold (default: 0.4)'
    )
    parser.add_argument(
        '--cellprob_threshold',
        type=float,
        default=0.0,
        help='Cell probability threshold (default: 0.0)'
    )
    parser.add_argument(
        '--max_distance',
        type=float,
        default=50.0,
        help='Max distance for GT matching in pixels (default: 50)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of images to test (None = all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports/cellpose_apcdata_validation',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='Save detection visualizations for first 10 images'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CELLPOSE VALIDATION — APCData (V14 Cytology Production)")
    print("=" * 80)
    print(f"  Data directory: {args.data_dir}")
    print(f"  Diameter:       {args.diameter}")
    print(f"  Flow threshold: {args.flow_threshold}")
    print(f"  Max distance:   {args.max_distance} px")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run validation
    results = validate_cellpose_on_apcdata(
        data_dir=args.data_dir,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        max_distance=args.max_distance,
        n_samples=args.n_samples,
        save_visualizations=args.save_visualizations,
        output_dir=args.output_dir
    )

    # Print report
    print_validation_report(results)

    # Save results JSON
    results_json = {k: v for k, v in results.items() if k != 'per_image_results'}

    if results['per_image_results']:
        results_json['per_image_summary'] = {
            'mean_detection_rate': np.mean([r['detection_rate'] for r in results['per_image_results']]),
            'std_detection_rate': np.std([r['detection_rate'] for r in results['per_image_results']]),
            'min_detection_rate': min([r['detection_rate'] for r in results['per_image_results']]),
            'max_detection_rate': max([r['detection_rate'] for r in results['per_image_results']])
        }
    else:
        results_json['per_image_summary'] = {
            'mean_detection_rate': 0.0,
            'std_detection_rate': 0.0,
            'min_detection_rate': 0.0,
            'max_detection_rate': 0.0
        }
        print("\n  WARNING: No images found matching annotations!")
        print("  Check that image filenames in annotations match files in images/ directory")

    results_path = os.path.join(args.output_dir, 'validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Generate plots
    plot_validation_results(results, args.output_dir)

    # Generate markdown report
    generate_markdown_report(results, args.output_dir)

    # Next steps
    print("\n" + "=" * 80)
    if results['detection_rate'] >= 0.90:
        print("  Next step: Run end-to-end pipeline")
        print("  python scripts/cytology/06_end_to_end_apcdata.py --data_dir data/raw/apcdata/APCData_points")
    else:
        print("  Adjust parameters and retry:")
        print(f"  python scripts/cytology/05_validate_cellpose_apcdata.py --diameter 40 --flow_threshold 0.3")
    print("=" * 80)


if __name__ == '__main__':
    main()
