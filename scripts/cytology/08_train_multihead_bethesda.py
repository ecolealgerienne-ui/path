"""
MultiHead Bethesda Classifier Training

Ce script entraîne un classificateur multi-têtes pour la classification des
cellules cervicales selon le système Bethesda.

Architecture MultiHead:
    H-Optimus embeddings (1536D)
        ↓
    Shared MLP (1536 → 512 → 256)
        ↓
    ┌────────────────┬────────────────┬────────────────┐
    │  Binary Head   │ Severity Head  │Fine-grained Head│
    │ Normal/Abnorm  │ Low/High Grade │  6 Bethesda    │
    └────────────────┴────────────────┴────────────────┘

Classes Bethesda:
    0: NILM   (Normal)    → Binary: Normal,   Severity: N/A
    1: ASCUS  (Abnormal)  → Binary: Abnormal, Severity: Low-grade
    2: ASCH   (Abnormal)  → Binary: Abnormal, Severity: High-grade
    3: LSIL   (Abnormal)  → Binary: Abnormal, Severity: Low-grade
    4: HSIL   (Abnormal)  → Binary: Abnormal, Severity: High-grade
    5: SCC    (Abnormal)  → Binary: Abnormal, Severity: High-grade

Usage:
    python scripts/cytology/08_train_multihead_bethesda.py \
        --data_dir data/raw/apcdata/APCData_YOLO \
        --output models/cytology/multihead_bethesda.pt

Author: V15 Cytology Branch
Date: 2026-01-23
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    precision_recall_fscore_support
)
from tqdm import tqdm


# =============================================================================
#  CONFIGURATION
# =============================================================================

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224

# Bethesda classes mapping
BETHESDA_CLASSES = {
    0: "NILM",   # Negative for Intraepithelial Lesion or Malignancy
    1: "ASCUS",  # Atypical Squamous Cells of Undetermined Significance
    2: "ASCH",   # Atypical Squamous Cells, cannot exclude HSIL
    3: "LSIL",   # Low-grade Squamous Intraepithelial Lesion
    4: "HSIL",   # High-grade Squamous Intraepithelial Lesion
    5: "SCC"     # Squamous Cell Carcinoma
}

# Binary mapping: Normal (0) vs Abnormal (1)
BINARY_MAPPING = {
    0: 0,  # NILM → Normal
    1: 1,  # ASCUS → Abnormal
    2: 1,  # ASCH → Abnormal
    3: 1,  # LSIL → Abnormal
    4: 1,  # HSIL → Abnormal
    5: 1   # SCC → Abnormal
}

# Severity mapping: Low-grade (0) vs High-grade (1)
# Only for abnormal cells (NILM is N/A)
SEVERITY_MAPPING = {
    1: 0,  # ASCUS → Low-grade
    2: 1,  # ASCH → High-grade
    3: 0,  # LSIL → Low-grade
    4: 1,  # HSIL → High-grade
    5: 1   # SCC → High-grade
}


# =============================================================================
#  DATASET
# =============================================================================

class BethesdaCellDataset(Dataset):
    """
    Dataset pour les cellules individuelles avec leurs labels Bethesda.

    Extrait les crops de cellules à partir des bounding boxes YOLO.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        crop_size: int = 224,
        min_cell_size: int = 32,
        padding_ratio: float = 0.2,
        transform=None
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.crop_size = crop_size
        self.min_cell_size = min_cell_size
        self.padding_ratio = padding_ratio
        self.transform = transform

        # Collect all cell annotations
        self.cells = []
        self._load_annotations()

        # Class statistics
        self.class_counts = defaultdict(int)
        for cell in self.cells:
            self.class_counts[cell['class_id']] += 1

    def _load_annotations(self):
        """Load all cell annotations from YOLO label files"""

        label_files = list(self.labels_dir.glob("*.txt"))
        label_files = [f for f in label_files if "Zone.Identifier" not in f.name]

        for label_path in label_files:
            # Find corresponding image
            basename = label_path.stem
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = self.images_dir / f"{basename}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                continue

            # Parse YOLO annotations
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        self.cells.append({
                            'image_path': img_path,
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        cell = self.cells[idx]

        # Load image
        image = cv2.imread(str(cell['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        # Convert normalized YOLO coords to pixel coords
        cx = int(cell['x_center'] * img_w)
        cy = int(cell['y_center'] * img_h)
        w = int(cell['width'] * img_w)
        h = int(cell['height'] * img_h)

        # Add padding
        pad_w = int(w * self.padding_ratio)
        pad_h = int(h * self.padding_ratio)

        # Calculate crop bounds
        x1 = max(0, cx - w // 2 - pad_w)
        y1 = max(0, cy - h // 2 - pad_h)
        x2 = min(img_w, cx + w // 2 + pad_w)
        y2 = min(img_h, cy + h // 2 + pad_h)

        # Extract crop
        crop = image[y1:y2, x1:x2]

        # Ensure minimum size
        if crop.shape[0] < self.min_cell_size or crop.shape[1] < self.min_cell_size:
            # Center pad if too small
            new_h = max(crop.shape[0], self.min_cell_size)
            new_w = max(crop.shape[1], self.min_cell_size)
            padded = np.zeros((new_h, new_w, 3), dtype=crop.dtype)
            y_off = (new_h - crop.shape[0]) // 2
            x_off = (new_w - crop.shape[1]) // 2
            padded[y_off:y_off+crop.shape[0], x_off:x_off+crop.shape[1]] = crop
            crop = padded

        # Resize to H-Optimus input size
        crop = cv2.resize(crop, (self.crop_size, self.crop_size))

        # Normalize for H-Optimus
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - np.array(HOPTIMUS_MEAN)) / np.array(HOPTIMUS_STD)

        # To tensor (C, H, W)
        crop = torch.from_numpy(crop).permute(2, 0, 1).float()

        # Labels
        class_id = cell['class_id']
        binary_label = BINARY_MAPPING[class_id]

        # Severity label (-1 for NILM, will be masked)
        if class_id == 0:
            severity_label = -1  # N/A for normal cells
        else:
            severity_label = SEVERITY_MAPPING[class_id]

        return crop, class_id, binary_label, severity_label, str(cell['image_path'])


# =============================================================================
#  MODEL
# =============================================================================

class MultiHeadBethesdaClassifier(nn.Module):
    """
    Multi-head classifier for Bethesda cell classification.

    Heads:
        1. Binary: Normal (NILM) vs Abnormal (all others)
        2. Severity: Low-grade (ASCUS, LSIL) vs High-grade (ASCH, HSIL, SCC)
        3. Fine-grained: All 6 Bethesda classes

    Clinical Priority:
        - Binary: High sensitivity for detecting ANY abnormality
        - Severity: Prioritize detection of high-grade lesions
        - Fine-grained: Detailed classification for review
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dims: Tuple[int, int] = (512, 256),
        dropout: float = 0.3
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Binary head: Normal vs Abnormal (2 classes)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Severity head: Low-grade vs High-grade (2 classes)
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Fine-grained head: 6 Bethesda classes
        self.finegrained_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, 6)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: H-Optimus embeddings (B, 1536)

        Returns:
            Dict with logits for each head:
                - 'binary': (B, 2) Normal/Abnormal
                - 'severity': (B, 2) Low/High grade
                - 'finegrained': (B, 6) All Bethesda classes
        """
        shared_features = self.shared(x)

        return {
            'binary': self.binary_head(shared_features),
            'severity': self.severity_head(shared_features),
            'finegrained': self.finegrained_head(shared_features)
        }

    def predict(
        self,
        x: torch.Tensor,
        binary_threshold: float = 0.3,  # Low threshold for high sensitivity
        severity_threshold: float = 0.4
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with adjustable thresholds.

        Args:
            x: H-Optimus embeddings (B, 1536)
            binary_threshold: Threshold for abnormal classification (lower = higher sensitivity)
            severity_threshold: Threshold for high-grade classification

        Returns:
            Dict with predictions and probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)

            # Probabilities
            binary_probs = F.softmax(logits['binary'], dim=1)
            severity_probs = F.softmax(logits['severity'], dim=1)
            finegrained_probs = F.softmax(logits['finegrained'], dim=1)

            # Predictions with thresholds
            binary_preds = (binary_probs[:, 1] >= binary_threshold).long()  # Abnormal = class 1
            severity_preds = (severity_probs[:, 1] >= severity_threshold).long()  # High-grade = class 1
            finegrained_preds = torch.argmax(finegrained_probs, dim=1)

            return {
                'binary_pred': binary_preds,
                'binary_prob': binary_probs,
                'severity_pred': severity_preds,
                'severity_prob': severity_probs,
                'finegrained_pred': finegrained_preds,
                'finegrained_prob': finegrained_probs
            }


# =============================================================================
#  FEATURE EXTRACTION
# =============================================================================

def extract_features_hoptimus(
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract H-Optimus features for all cell crops.

    Returns:
        features: (N, 1536)
        class_labels: (N,) - 6-class labels
        binary_labels: (N,) - Normal/Abnormal
        severity_labels: (N,) - Low/High grade (-1 for NILM)
    """
    print("  [INFO] Loading H-Optimus-0...")

    try:
        import timm

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

        print(f"  [OK] H-Optimus-0 loaded on {device}")

    except Exception as e:
        print(f"  [ERROR] Failed to load H-Optimus-0: {e}")
        return None, None, None, None

    all_features = []
    all_class_labels = []
    all_binary_labels = []
    all_severity_labels = []

    print("  [INFO] Extracting features...")

    with torch.no_grad():
        for crops, class_ids, binary_labels, severity_labels, paths in tqdm(dataloader, desc="Extracting"):
            crops = crops.to(device)

            # Extract features
            features = model.forward_features(crops)

            # Get CLS token
            if len(features.shape) == 3:
                cls_tokens = features[:, 0, :]
            else:
                cls_tokens = features

            all_features.append(cls_tokens.cpu())
            all_class_labels.append(class_ids)
            all_binary_labels.append(binary_labels)
            all_severity_labels.append(severity_labels)

    features = torch.cat(all_features, dim=0)
    class_labels = torch.cat(all_class_labels, dim=0)
    binary_labels = torch.cat(all_binary_labels, dim=0)
    severity_labels = torch.cat(all_severity_labels, dim=0)

    print(f"  [OK] Extracted features: {features.shape}")

    return features, class_labels, binary_labels, severity_labels


# =============================================================================
#  TRAINING
# =============================================================================

def train_multihead_classifier(
    train_features: torch.Tensor,
    train_class_labels: torch.Tensor,
    train_binary_labels: torch.Tensor,
    train_severity_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_class_labels: torch.Tensor,
    val_binary_labels: torch.Tensor,
    val_severity_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_binary: float = 1.0,
    lambda_severity: float = 1.0,
    lambda_finegrained: float = 1.0
) -> MultiHeadBethesdaClassifier:
    """
    Train multi-head classifier with weighted losses.

    Loss = lambda_binary * L_binary + lambda_severity * L_severity + lambda_finegrained * L_finegrained
    """

    model = MultiHeadBethesdaClassifier(input_dim=train_features.shape[1])
    model = model.to(device)

    # Class weights for high sensitivity (detect abnormals)
    # Binary: weight abnormal higher
    binary_weights = torch.tensor([0.3, 1.0]).to(device)
    criterion_binary = nn.CrossEntropyLoss(weight=binary_weights)

    # Severity: weight high-grade higher (clinical priority)
    severity_weights = torch.tensor([0.5, 1.0]).to(device)
    criterion_severity = nn.CrossEntropyLoss(weight=severity_weights, ignore_index=-1)

    # Fine-grained: balanced with slight emphasis on malignant classes
    # NILM:0.3, ASCUS:0.5, ASCH:1.0, LSIL:0.5, HSIL:1.0, SCC:1.0
    finegrained_weights = torch.tensor([0.3, 0.5, 1.0, 0.5, 1.0, 1.0]).to(device)
    criterion_finegrained = nn.CrossEntropyLoss(weight=finegrained_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        train_features, train_class_labels, train_binary_labels, train_severity_labels
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_features, val_class_labels, val_binary_labels, val_severity_labels
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_binary_recall = 0.0
    best_model_state = None

    print("\n  Training multi-head classifier...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0

        for features, class_labels, binary_labels, severity_labels in train_loader:
            features = features.to(device)
            class_labels = class_labels.to(device)
            binary_labels = binary_labels.to(device)
            severity_labels = severity_labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)

            # Multi-task loss
            loss_binary = criterion_binary(outputs['binary'], binary_labels)
            loss_severity = criterion_severity(outputs['severity'], severity_labels)
            loss_finegrained = criterion_finegrained(outputs['finegrained'], class_labels)

            loss = (lambda_binary * loss_binary +
                    lambda_severity * loss_severity +
                    lambda_finegrained * loss_finegrained)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        all_binary_preds = []
        all_binary_gt = []
        all_severity_preds = []
        all_severity_gt = []
        all_finegrained_preds = []
        all_finegrained_gt = []

        with torch.no_grad():
            for features, class_labels, binary_labels, severity_labels in val_loader:
                features = features.to(device)

                preds = model.predict(features, binary_threshold=0.3, severity_threshold=0.4)

                all_binary_preds.extend(preds['binary_pred'].cpu().numpy())
                all_binary_gt.extend(binary_labels.numpy())

                # Only for abnormal cells
                mask = severity_labels >= 0
                if mask.sum() > 0:
                    all_severity_preds.extend(preds['severity_pred'][mask].cpu().numpy())
                    all_severity_gt.extend(severity_labels[mask].numpy())

                all_finegrained_preds.extend(preds['finegrained_pred'].cpu().numpy())
                all_finegrained_gt.extend(class_labels.numpy())

        # Metrics
        all_binary_preds = np.array(all_binary_preds)
        all_binary_gt = np.array(all_binary_gt)

        # Binary recall (abnormal detection)
        abnormal_mask = all_binary_gt == 1
        if abnormal_mask.sum() > 0:
            binary_recall = (all_binary_preds[abnormal_mask] == 1).mean()
        else:
            binary_recall = 0.0

        # Normal specificity
        normal_mask = all_binary_gt == 0
        if normal_mask.sum() > 0:
            binary_specificity = (all_binary_preds[normal_mask] == 0).mean()
        else:
            binary_specificity = 0.0

        # Fine-grained accuracy
        finegrained_acc = balanced_accuracy_score(all_finegrained_gt, all_finegrained_preds)

        # Save best model (based on binary recall - clinical priority)
        if binary_recall > best_binary_recall:
            best_binary_recall = binary_recall
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={train_loss/len(train_loader):.4f}, "
                  f"Binary(Recall/Spec)={binary_recall:.3f}/{binary_specificity:.3f}, "
                  f"FineAcc={finegrained_acc:.3f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n  [OK] Best Binary Recall (Abnormal): {best_binary_recall:.4f}")

    return model


def evaluate_model(
    model: MultiHeadBethesdaClassifier,
    features: torch.Tensor,
    class_labels: torch.Tensor,
    binary_labels: torch.Tensor,
    severity_labels: torch.Tensor,
    device: torch.device
) -> Dict:
    """Comprehensive model evaluation"""

    model.eval()
    features = features.to(device)

    with torch.no_grad():
        preds = model.predict(features, binary_threshold=0.3, severity_threshold=0.4)

    binary_preds = preds['binary_pred'].cpu().numpy()
    binary_gt = binary_labels.numpy()
    finegrained_preds = preds['finegrained_pred'].cpu().numpy()
    finegrained_gt = class_labels.numpy()
    severity_preds = preds['severity_pred'].cpu().numpy()
    severity_gt = severity_labels.numpy()

    results = {}

    # === Binary Metrics ===
    abnormal_mask = binary_gt == 1
    normal_mask = binary_gt == 0

    results['binary'] = {
        'recall': (binary_preds[abnormal_mask] == 1).mean() if abnormal_mask.sum() > 0 else 0,
        'specificity': (binary_preds[normal_mask] == 0).mean() if normal_mask.sum() > 0 else 0,
        'balanced_accuracy': balanced_accuracy_score(binary_gt, binary_preds)
    }

    # === Severity Metrics (only for abnormal) ===
    abnormal_indices = severity_gt >= 0
    if abnormal_indices.sum() > 0:
        sev_preds = severity_preds[abnormal_indices]
        sev_gt = severity_gt[abnormal_indices]

        high_grade_mask = sev_gt == 1
        low_grade_mask = sev_gt == 0

        results['severity'] = {
            'recall_high': (sev_preds[high_grade_mask] == 1).mean() if high_grade_mask.sum() > 0 else 0,
            'specificity_low': (sev_preds[low_grade_mask] == 0).mean() if low_grade_mask.sum() > 0 else 0,
            'balanced_accuracy': balanced_accuracy_score(sev_gt, sev_preds)
        }

    # === Fine-grained Metrics ===
    results['finegrained'] = {
        'balanced_accuracy': balanced_accuracy_score(finegrained_gt, finegrained_preds),
        'per_class': {}
    }

    for class_id, class_name in BETHESDA_CLASSES.items():
        mask = finegrained_gt == class_id
        if mask.sum() > 0:
            recall = (finegrained_preds[mask] == class_id).mean()
            results['finegrained']['per_class'][class_name] = {
                'recall': float(recall),
                'count': int(mask.sum())
            }

    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(finegrained_gt, finegrained_preds).tolist()

    return results


# =============================================================================
#  MAIN
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_info(message: str):
    print(f"  [INFO] {message}")


def print_success(message: str):
    print(f"  [OK] {message}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MultiHead Bethesda Classifier"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/apcdata/APCData_YOLO",
        help="Path to APCData YOLO directory with train/val splits"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/cytology/multihead_bethesda.pt",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Cache extracted features to disk"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  MULTIHEAD BETHESDA CLASSIFIER TRAINING")
    print("  V15 Cytology Pipeline")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print_info(f"Device: {device}")

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify input
    print_header("LOADING DATA")

    train_images = data_dir / "train" / "images"
    train_labels = data_dir / "train" / "labels"
    val_images = data_dir / "val" / "images"
    val_labels = data_dir / "val" / "labels"

    if not train_images.exists():
        print(f"  [ERROR] Train images not found: {train_images}")
        return 1

    # Create datasets
    train_dataset = BethesdaCellDataset(train_images, train_labels)
    val_dataset = BethesdaCellDataset(val_images, val_labels)

    print_success(f"Train: {len(train_dataset)} cells")
    print_success(f"Val: {len(val_dataset)} cells")

    print("\n  Class distribution (train):")
    for class_id, class_name in BETHESDA_CLASSES.items():
        count = train_dataset.class_counts[class_id]
        print(f"    {class_name}: {count}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Check for cached features
    cache_dir = data_dir / "cache_cells"
    train_features_path = cache_dir / "train_cell_features.pt"
    val_features_path = cache_dir / "val_cell_features.pt"

    if train_features_path.exists() and val_features_path.exists():
        print_header("LOADING CACHED FEATURES")
        train_data = torch.load(train_features_path, weights_only=False)
        val_data = torch.load(val_features_path, weights_only=False)
        train_features = train_data['features']
        train_class_labels = train_data['class_labels']
        train_binary_labels = train_data['binary_labels']
        train_severity_labels = train_data['severity_labels']
        val_features = val_data['features']
        val_class_labels = val_data['class_labels']
        val_binary_labels = val_data['binary_labels']
        val_severity_labels = val_data['severity_labels']
        print_success(f"Loaded cached features from {cache_dir}")
    else:
        # Extract features
        print_header("EXTRACTING FEATURES")

        train_features, train_class_labels, train_binary_labels, train_severity_labels = \
            extract_features_hoptimus(train_loader, device)
        val_features, val_class_labels, val_binary_labels, val_severity_labels = \
            extract_features_hoptimus(val_loader, device)

        if train_features is None:
            return 1

        # Cache features
        if args.cache_features:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'features': train_features,
                'class_labels': train_class_labels,
                'binary_labels': train_binary_labels,
                'severity_labels': train_severity_labels
            }, train_features_path)
            torch.save({
                'features': val_features,
                'class_labels': val_class_labels,
                'binary_labels': val_binary_labels,
                'severity_labels': val_severity_labels
            }, val_features_path)
            print_success(f"Cached features to {cache_dir}")

    # Train classifier
    print_header("TRAINING CLASSIFIER")

    model = train_multihead_classifier(
        train_features, train_class_labels, train_binary_labels, train_severity_labels,
        val_features, val_class_labels, val_binary_labels, val_severity_labels,
        device,
        epochs=args.epochs
    )

    # Evaluate
    print_header("EVALUATION")

    results = evaluate_model(
        model, val_features, val_class_labels, val_binary_labels, val_severity_labels, device
    )

    # Print results
    print("\n  === Binary Classification (Normal vs Abnormal) ===")
    print(f"    Recall (Abnormal):     {results['binary']['recall']:.4f} ({results['binary']['recall']*100:.1f}%)")
    print(f"    Specificity (Normal):  {results['binary']['specificity']:.4f} ({results['binary']['specificity']*100:.1f}%)")
    print(f"    Balanced Accuracy:     {results['binary']['balanced_accuracy']:.4f}")

    if 'severity' in results:
        print("\n  === Severity Classification (Low vs High Grade) ===")
        print(f"    Recall (High-grade):   {results['severity']['recall_high']:.4f} ({results['severity']['recall_high']*100:.1f}%)")
        print(f"    Specificity (Low):     {results['severity']['specificity_low']:.4f} ({results['severity']['specificity_low']*100:.1f}%)")
        print(f"    Balanced Accuracy:     {results['severity']['balanced_accuracy']:.4f}")

    print("\n  === Fine-grained Classification (6 Bethesda) ===")
    print(f"    Balanced Accuracy:     {results['finegrained']['balanced_accuracy']:.4f}")
    print("\n    Per-class recall:")
    for class_name, metrics in results['finegrained']['per_class'].items():
        print(f"      {class_name:6s}: {metrics['recall']:.3f} (n={metrics['count']})")

    print("\n  Confusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print("         ", "  ".join([f"{BETHESDA_CLASSES[i][:4]:>4s}" for i in range(6)]))
    for i in range(6):
        print(f"    {BETHESDA_CLASSES[i]:5s}:", "  ".join([f"{cm[i,j]:4d}" for j in range(6)]))

    # Save model
    print_header("SAVING MODEL")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': 1536,
        'hidden_dims': (512, 256),
        'class_names': BETHESDA_CLASSES,
        'binary_mapping': BINARY_MAPPING,
        'severity_mapping': SEVERITY_MAPPING,
        'val_results': results,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }

    torch.save(checkpoint, output_path)
    print_success(f"Model saved to {output_path}")

    # Final summary
    print_header("SUMMARY")

    print_info(f"Binary Recall (Abnormal): {results['binary']['recall']:.4f}")
    if 'severity' in results:
        print_info(f"Severity Recall (High-grade): {results['severity']['recall_high']:.4f}")
    print_info(f"Fine-grained Balanced Accuracy: {results['finegrained']['balanced_accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETED")
    print("=" * 80)
    print(f"\n  Model: {output_path}")
    print(f"\n  Clinical Priorities:")
    print(f"    1. Binary Detection (Abnormal):  {results['binary']['recall']*100:.1f}% sensitivity")
    if 'severity' in results:
        print(f"    2. Severity (High-grade):        {results['severity']['recall_high']*100:.1f}% sensitivity")
    print(f"    3. Fine-grained Classification:  {results['finegrained']['balanced_accuracy']*100:.1f}% balanced accuracy")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
