"""
End-to-End Pipeline Validation — APCData (V14 Cytology Production)

Ce script valide le pipeline complet sur APCData:
1. CellPose detection (avec params optimaux validés)
2. Extraction patches 224×224 centrés sur noyaux détectés
3. H-Optimus-0 feature extraction
4. Morphometry features (sur masques CellPose)
5. MLP classification
6. Comparaison avec GT APCData (Bethesda classes)

Mapping Bethesda (APCData) → Binary:
- NILM → Normal
- ASCUS, ASCH, LSIL, HSIL, SCC → Abnormal

Author: V14 Cytology Branch
Date: 2026-01-21
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from cellpose import models
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
#  CONSTANTS
# =============================================================================

# H-Optimus-0 normalization
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224

# CellPose optimal parameters (validated on APCData 2026-01-21)
CELLPOSE_CONFIG = {
    'diameter': 60,
    'flow_threshold': 0.4,
    'cellprob_threshold': 0.0,
    'min_area': 400,
    'max_area': 100000,
}

# Bethesda classes (APCData)
BETHESDA_CLASSES = ['NILM', 'ASCUS', 'ASCH', 'LSIL', 'HSIL', 'SCC']

# YOLO class mapping
YOLO_CLASS_MAPPING = {
    0: "NILM",
    1: "ASCUS",
    2: "ASCH",
    3: "LSIL",
    4: "HSIL",
    5: "SCC"
}

# Binary grouping for Safety First
NORMAL_CLASSES_BETHESDA = {'NILM'}
ABNORMAL_CLASSES_BETHESDA = {'ASCUS', 'ASCH', 'LSIL', 'HSIL', 'SCC'}

# Ruifrok stain vectors for H-channel extraction
RUIFROK_STAIN_MATRIX = np.array([
    [0.650, 0.704, 0.286],  # Hematoxylin
    [0.268, 0.570, 0.776],  # Eosin
    [0.578, 0.421, 0.698]   # DAB (residual)
])


# =============================================================================
#  ANNOTATION LOADING
# =============================================================================

def load_yolo_annotations(data_dir: str, image_width: int = 2048, image_height: int = 1532) -> Dict[str, List[Dict]]:
    """
    Load APCData annotations from YOLO format.
    """
    labels_dir = os.path.join(data_dir, 'labels')
    annotations = {}

    label_files = [f for f in os.listdir(labels_dir)
                   if f.endswith('.txt') and ':Zone' not in f]

    for label_file in label_files:
        base_name = label_file.replace('.txt', '')
        image_file = base_name + '.jpg'

        cells = []
        label_path = os.path.join(labels_dir, label_file)

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])

                    # Denormalize coordinates
                    nucleus_x = int(x_center * image_width)
                    nucleus_y = int(y_center * image_height)

                    cells.append({
                        'nucleus_x': nucleus_x,
                        'nucleus_y': nucleus_y,
                        'class': YOLO_CLASS_MAPPING.get(class_id, 'NILM')
                    })

        annotations[image_file] = cells

    return annotations


# =============================================================================
#  CELLPOSE DETECTION
# =============================================================================

def run_cellpose_detection(
    model,
    image: np.ndarray,
    config: Dict = CELLPOSE_CONFIG
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run CellPose detection with optimal parameters.
    """
    masks, flows, styles = model.eval(
        image,
        diameter=config['diameter'],
        flow_threshold=config['flow_threshold'],
        cellprob_threshold=config['cellprob_threshold']
    )

    # Extract detection info
    detections = []

    if masks.max() > 0:
        props = regionprops(masks)

        for prop in props:
            # Filter by area
            if config['min_area'] <= prop.area <= config['max_area']:
                cy, cx = prop.centroid
                detections.append({
                    'id': prop.label,
                    'centroid_x': cx,
                    'centroid_y': cy,
                    'area': prop.area,
                    'bbox': prop.bbox
                })

    return masks, detections


def match_detections_to_gt(
    detections: List[Dict],
    gt_cells: List[Dict],
    max_distance: float = 100.0
) -> List[Tuple[Dict, Dict]]:
    """
    Match CellPose detections to GT points using Hungarian algorithm.

    Returns:
        matches: List of (detection, gt_cell) tuples
    """
    if len(detections) == 0 or len(gt_cells) == 0:
        return []

    # Build cost matrix
    det_points = np.array([[d['centroid_x'], d['centroid_y']] for d in detections])
    gt_points = np.array([[c['nucleus_x'], c['nucleus_y']] for c in gt_cells])

    cost_matrix = cdist(det_points, gt_points)

    # Hungarian matching
    det_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # Filter by max distance
    matches = []
    for det_idx, gt_idx in zip(det_indices, gt_indices):
        if cost_matrix[det_idx, gt_idx] <= max_distance:
            matches.append((detections[det_idx], gt_cells[gt_idx]))

    return matches


# =============================================================================
#  PATCH EXTRACTION
# =============================================================================

def extract_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    patch_size: int = 224
) -> np.ndarray:
    """
    Extract a patch centered on (center_x, center_y).
    Pads with white if near image boundary.
    """
    h, w = image.shape[:2]
    half = patch_size // 2

    # Calculate crop boundaries
    x1 = int(center_x - half)
    y1 = int(center_y - half)
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    # Create white canvas
    patch = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255

    # Calculate valid regions
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Copy valid region
    patch[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    return patch


def extract_cell_mask(
    masks: np.ndarray,
    cell_id: int,
    center_x: float,
    center_y: float,
    patch_size: int = 224
) -> np.ndarray:
    """
    Extract the cell mask for a specific detection.
    """
    h, w = masks.shape
    half = patch_size // 2

    x1 = int(center_x - half)
    y1 = int(center_y - half)
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    # Create empty mask
    cell_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

    # Calculate valid regions
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Extract and binarize mask for this cell
    mask_crop = masks[src_y1:src_y2, src_x1:src_x2]
    cell_mask[dst_y1:dst_y2, dst_x1:dst_x2] = (mask_crop == cell_id).astype(np.uint8)

    return cell_mask


# =============================================================================
#  H-OPTIMUS-0 FEATURE EXTRACTION
# =============================================================================

def load_hoptimus_model(device: str = 'cuda'):
    """Load H-Optimus-0 from HuggingFace"""
    import timm

    print("  Loading H-Optimus-0...")
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False
    )

    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def preprocess_for_hoptimus(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for H-Optimus-0"""
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(HOPTIMUS_MEAN)) / np.array(HOPTIMUS_STD)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return tensor


def extract_hoptimus_features(
    model,
    patches: List[np.ndarray],
    device: str = 'cuda',
    batch_size: int = 16
) -> torch.Tensor:
    """
    Extract H-Optimus-0 CLS tokens for a list of patches.
    """
    all_features = []

    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i+batch_size]

        # Preprocess batch
        tensors = [preprocess_for_hoptimus(p) for p in batch_patches]
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            outputs = model.forward_features(batch)
            cls_tokens = outputs[:, 0, :]  # (B, 1536)

        all_features.append(cls_tokens.cpu())

    return torch.cat(all_features, dim=0)


# =============================================================================
#  MORPHOMETRY FEATURES
# =============================================================================

def extract_h_channel(image: np.ndarray) -> np.ndarray:
    """Extract Hematoxylin channel via Ruifrok deconvolution"""
    image_float = image.astype(np.float32) / 255.0
    image_float = np.clip(image_float, 1e-6, 1.0)
    od = -np.log(image_float)

    od_flat = od.reshape(-1, 3)
    stain_matrix_inv = np.linalg.inv(RUIFROK_STAIN_MATRIX)
    stain_concentrations = od_flat @ stain_matrix_inv.T

    h_channel = stain_concentrations[:, 0].reshape(image.shape[:2])
    h_channel = np.clip(h_channel, 0, None)
    if h_channel.max() > 0:
        h_channel = h_channel / h_channel.max()

    return h_channel.astype(np.float32)


def compute_morphometry_features(
    patch: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Compute 20 morphometry features for a cell.
    """
    features = np.zeros(20, dtype=np.float32)

    if mask.sum() == 0:
        return features

    # Geometric features from regionprops
    props = regionprops(mask.astype(int))
    if len(props) == 0:
        return features

    prop = props[0]

    features[0] = prop.area
    features[1] = prop.perimeter if prop.perimeter > 0 else 0
    features[2] = 4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0  # circularity
    features[3] = prop.eccentricity
    features[4] = prop.solidity
    features[5] = prop.extent
    features[6] = prop.major_axis_length
    features[7] = prop.minor_axis_length
    features[8] = prop.orientation
    features[9] = prop.equivalent_diameter

    # H-channel intensity features
    h_channel = extract_h_channel(patch)
    masked_h = h_channel[mask > 0]

    if len(masked_h) > 0:
        features[10] = np.mean(masked_h)
        features[11] = np.std(masked_h)
        features[12] = np.min(masked_h)
        features[13] = np.max(masked_h)
        features[14] = np.median(masked_h)

    # Texture features (simplified GLCM)
    try:
        h_uint8 = (h_channel * 255).astype(np.uint8)
        glcm = np.zeros((256, 256), dtype=np.float32)
        # Simplified: just use intensity histogram as proxy
        hist, _ = np.histogram(masked_h, bins=16, range=(0, 1))
        hist = hist / (hist.sum() + 1e-6)
        features[15] = -np.sum(hist * np.log(hist + 1e-6))  # entropy
        features[16] = np.sum(hist ** 2)  # energy
        features[17] = np.std(hist)  # contrast proxy
        features[18] = np.max(hist)  # homogeneity proxy
        features[19] = np.mean(hist)  # mean
    except:
        pass

    return features


# =============================================================================
#  MLP CLASSIFIER
# =============================================================================

class CytologyMLP(nn.Module):
    """MLP Classifier for cytology"""

    def __init__(
        self,
        input_dim: int = 1556,
        hidden_dims: List[int] = [512, 256, 128],
        n_classes: int = 7,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# SIPaKMeD classes for MLP output
SIPAKMED_CLASSES = [
    'normal_columnar',
    'normal_intermediate',
    'normal_superficiel',
    'light_dysplastic',
    'moderate_dysplastic',
    'severe_dysplastic',
    'carcinoma_in_situ'
]

# Mapping SIPaKMeD → Binary
SIPAKMED_NORMAL = {0, 1, 2}  # normal classes
SIPAKMED_ABNORMAL = {3, 4, 5, 6}  # dysplastic + carcinoma


def map_sipakmed_to_binary(pred_class: int) -> str:
    """Map SIPaKMeD class to binary Normal/Abnormal"""
    if pred_class in SIPAKMED_NORMAL:
        return 'Normal'
    else:
        return 'Abnormal'


def map_bethesda_to_binary(bethesda_class: str) -> str:
    """Map Bethesda class to binary Normal/Abnormal"""
    if bethesda_class in NORMAL_CLASSES_BETHESDA:
        return 'Normal'
    else:
        return 'Abnormal'


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_end_to_end_pipeline(
    data_dir: str,
    mlp_checkpoint: str,
    n_samples: Optional[int] = None,
    output_dir: str = 'reports/end_to_end_apcdata',
    device: str = 'cuda',
    max_distance: float = 100.0
):
    """
    Run complete end-to-end pipeline on APCData.
    """
    print("=" * 80)
    print("END-TO-END PIPELINE VALIDATION — APCData")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load annotations
    images_dir = os.path.join(data_dir, 'images')
    sample_images = [f for f in os.listdir(images_dir)
                     if f.endswith(('.jpg', '.png')) and ':Zone' not in f]
    if sample_images:
        sample_img = Image.open(os.path.join(images_dir, sample_images[0]))
        img_width, img_height = sample_img.size
    else:
        img_width, img_height = 2048, 1532

    annotations = load_yolo_annotations(data_dir, img_width, img_height)
    print(f"  Loaded annotations for {len(annotations)} images")

    # Filter to images with annotations
    image_files = [f for f in os.listdir(images_dir)
                   if f in annotations and ':Zone' not in f]

    if n_samples and n_samples < len(image_files):
        image_files = image_files[:n_samples]

    print(f"  Processing {len(image_files)} images")

    # Load models
    print("\n  Loading models...")

    # CellPose
    cellpose_model = models.CellposeModel(gpu=True)
    print("  ✓ CellPose loaded")

    # H-Optimus-0
    hoptimus_model = load_hoptimus_model(device)
    print("  ✓ H-Optimus-0 loaded")

    # MLP Classifier
    mlp = CytologyMLP(input_dim=1556, n_classes=7)
    checkpoint = torch.load(mlp_checkpoint, map_location=device, weights_only=False)
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp = mlp.to(device)
    mlp.eval()
    print(f"  ✓ MLP loaded from {mlp_checkpoint}")

    # Results storage
    all_gt_binary = []
    all_pred_binary = []
    all_gt_bethesda = []
    all_pred_sipakmed = []
    all_confidences = []

    per_image_results = []

    # Process each image
    print(f"\n  Running pipeline...")

    for image_file in tqdm(image_files, desc="  Processing"):
        image_path = os.path.join(images_dir, image_file)
        image = np.array(Image.open(image_path))

        # Get GT annotations
        gt_cells = annotations[image_file]

        # Step 1: CellPose detection
        masks, detections = run_cellpose_detection(cellpose_model, image)

        if len(detections) == 0:
            continue

        # Step 2: Match detections to GT
        matches = match_detections_to_gt(detections, gt_cells, max_distance)

        if len(matches) == 0:
            continue

        # Step 3: Extract patches and masks for matched cells
        patches = []
        cell_masks = []
        gt_classes = []

        for det, gt in matches:
            patch = extract_patch(image, det['centroid_x'], det['centroid_y'])
            cell_mask = extract_cell_mask(masks, det['id'], det['centroid_x'], det['centroid_y'])

            patches.append(patch)
            cell_masks.append(cell_mask)
            gt_classes.append(gt['class'])

        # Step 4: Extract H-Optimus-0 features
        cls_features = extract_hoptimus_features(hoptimus_model, patches, device)

        # Step 5: Compute morphometry features
        morph_features = []
        for patch, mask in zip(patches, cell_masks):
            mf = compute_morphometry_features(patch, mask)
            morph_features.append(mf)
        morph_features = torch.from_numpy(np.stack(morph_features))

        # Step 6: Fuse features
        fused_features = torch.cat([cls_features, morph_features], dim=1)  # (N, 1556)

        # Step 7: MLP classification
        fused_features = fused_features.to(device)
        with torch.no_grad():
            probs = mlp.predict_proba(fused_features)
            pred_classes = probs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()

        # Store results
        for i, gt_class in enumerate(gt_classes):
            gt_binary = map_bethesda_to_binary(gt_class)
            pred_binary = map_sipakmed_to_binary(pred_classes[i])

            all_gt_binary.append(gt_binary)
            all_pred_binary.append(pred_binary)
            all_gt_bethesda.append(gt_class)
            all_pred_sipakmed.append(SIPAKMED_CLASSES[pred_classes[i]])
            all_confidences.append(confidences[i])

        per_image_results.append({
            'image': image_file,
            'n_gt': len(gt_cells),
            'n_detected': len(detections),
            'n_matched': len(matches),
            'n_classified': len(pred_classes)
        })

    # Compute metrics
    print("\n" + "=" * 80)
    print(" END-TO-END VALIDATION RESULTS")
    print("=" * 80)

    n_total = len(all_gt_binary)
    print(f"\n  Total cells evaluated: {n_total}")

    if n_total == 0:
        print("  ERROR: No cells were successfully processed!")
        return

    # Binary metrics
    gt_binary_numeric = [0 if g == 'Normal' else 1 for g in all_gt_binary]
    pred_binary_numeric = [0 if p == 'Normal' else 1 for p in all_pred_binary]

    # Confusion matrix (binary)
    cm_binary = confusion_matrix(gt_binary_numeric, pred_binary_numeric)

    # Safety First metrics
    tn, fp, fn, tp = cm_binary.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / n_total

    # Cohen's Kappa
    kappa = cohen_kappa_score(gt_binary_numeric, pred_binary_numeric)

    print(f"\n  {'='*60}")
    print(f"  BINARY CLASSIFICATION (Normal vs Abnormal)")
    print(f"  {'='*60}")
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Normal  Abnormal")
    print(f"  Actual Normal    {tn:>5}    {fp:>5}")
    print(f"  Actual Abnormal  {fn:>5}    {tp:>5}")

    print(f"\n  {'Metric':<30} {'Value':>10} {'Target':>10} {'Status':>10}")
    print(f"  {'-'*60}")

    sens_status = "✅ PASS" if sensitivity >= 0.98 else "❌ FAIL"
    kappa_status = "✅ PASS" if kappa >= 0.80 else "❌ FAIL"

    print(f"  {'Sensitivity (Abnormal)':<30} {sensitivity*100:>9.1f}% {'>98%':>10} {sens_status:>10}")
    print(f"  {'Specificity':<30} {specificity*100:>9.1f}% {'>60%':>10} {'✅' if specificity >= 0.60 else '⚠️':>10}")
    print(f"  {'Accuracy':<30} {accuracy*100:>9.1f}% {'-':>10} {'-':>10}")
    print(f"  {'Cohen Kappa':<30} {kappa:>10.3f} {'>0.80':>10} {kappa_status:>10}")

    # Per-Bethesda class analysis
    print(f"\n  {'='*60}")
    print(f"  PER-BETHESDA CLASS ANALYSIS")
    print(f"  {'='*60}")

    print(f"\n  {'Class':<10} {'Total':>8} {'Pred Normal':>12} {'Pred Abnormal':>14} {'Correct':>10}")
    print(f"  {'-'*60}")

    for bethesda_class in BETHESDA_CLASSES:
        indices = [i for i, g in enumerate(all_gt_bethesda) if g == bethesda_class]
        if len(indices) == 0:
            continue

        pred_normal = sum(1 for i in indices if all_pred_binary[i] == 'Normal')
        pred_abnormal = sum(1 for i in indices if all_pred_binary[i] == 'Abnormal')

        expected = 'Normal' if bethesda_class == 'NILM' else 'Abnormal'
        correct = pred_normal if expected == 'Normal' else pred_abnormal
        correct_pct = correct / len(indices) * 100

        print(f"  {bethesda_class:<10} {len(indices):>8} {pred_normal:>12} {pred_abnormal:>14} {correct_pct:>9.1f}%")

    # Save results
    results = {
        'n_images': len(per_image_results),
        'n_cells': n_total,
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'accuracy': float(accuracy),
        'cohen_kappa': float(kappa),
        'confusion_matrix': cm_binary.tolist(),
        'mean_confidence': float(np.mean(all_confidences)),
        'per_image_results': per_image_results
    }

    results_path = os.path.join(output_dir, 'end_to_end_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'End-to-End Binary Classification\nSensitivity: {sensitivity*100:.1f}% | Kappa: {kappa:.3f}')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")

    # Final verdict
    print("\n" + "=" * 80)
    if sensitivity >= 0.98 and kappa >= 0.80:
        print("  ✅ VALIDATION PASSED — Pipeline ready for production")
    elif sensitivity >= 0.90:
        print("  ⚠️  VALIDATION ACCEPTABLE — Sensitivity >90%, proceed with caution")
    else:
        print("  ❌ VALIDATION FAILED — Pipeline needs improvement")
    print("=" * 80)

    return results


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline validation on APCData"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw/apcdata/APCData_YOLO',
        help='APCData_YOLO directory'
    )
    parser.add_argument(
        '--mlp_checkpoint',
        type=str,
        default='models/cytology/mlp_classifier_best.pth',
        help='Path to trained MLP checkpoint'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of images to process (None = all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports/end_to_end_apcdata',
        help='Output directory'
    )
    parser.add_argument(
        '--max_distance',
        type=float,
        default=100.0,
        help='Max distance for GT matching (pixels)'
    )

    args = parser.parse_args()

    run_end_to_end_pipeline(
        data_dir=args.data_dir,
        mlp_checkpoint=args.mlp_checkpoint,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        max_distance=args.max_distance
    )


if __name__ == '__main__':
    main()
