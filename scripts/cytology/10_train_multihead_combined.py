"""
MultiHead Bethesda Classifier — Combined APCData + SIPaKMeD Training

Ce script entraîne le classificateur MultiHead sur les features combinées
de APCData et SIPaKMeD pour améliorer les performances.

Données combinées:
    APCData:  2,932 train + 687 val (6 classes Bethesda complètes)
    SIPaKMeD: ~3,200 train + ~800 val (4 classes: NILM, LSIL, HSIL, SCC)

    Total: ~6,100 train + ~1,500 val = +108% de données!

Note: SIPaKMeD n'a pas ASCUS ni ASCH, donc ces classes restent sous-représentées.

Usage:
    python scripts/cytology/10_train_multihead_combined.py \
        --apcdata_cache data/raw/apcdata/APCData_YOLO/cache_cells \
        --sipakmed_cache data/cache/sipakmed_features \
        --output models/cytology/multihead_bethesda_combined.pt

Author: V15 Cytology Branch
Date: 2026-01-23
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm


# =============================================================================
#  CONFIGURATION
# =============================================================================

BETHESDA_CLASSES = {
    0: "NILM",
    1: "ASCUS",
    2: "ASCH",
    3: "LSIL",
    4: "HSIL",
    5: "SCC"
}

# Binary mapping
BINARY_MAPPING = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

# Severity mapping (-1 for NILM)
SEVERITY_MAPPING = {0: -1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1}


# =============================================================================
#  MODEL (Same as 08_train_multihead_bethesda.py)
# =============================================================================

class MultiHeadBethesdaClassifier(nn.Module):
    """Multi-head classifier for Bethesda cell classification."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dims: Tuple[int, int] = (512, 256),
        dropout: float = 0.3
    ):
        super().__init__()

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

        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

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
        shared_features = self.shared(x)
        return {
            'binary': self.binary_head(shared_features),
            'severity': self.severity_head(shared_features),
            'finegrained': self.finegrained_head(shared_features)
        }

    def predict(self, x: torch.Tensor, binary_threshold: float = 0.3, severity_threshold: float = 0.4):
        with torch.no_grad():
            logits = self.forward(x)
            binary_probs = F.softmax(logits['binary'], dim=1)
            severity_probs = F.softmax(logits['severity'], dim=1)
            finegrained_probs = F.softmax(logits['finegrained'], dim=1)

            binary_preds = (binary_probs[:, 1] >= binary_threshold).long()
            severity_preds = (severity_probs[:, 1] >= severity_threshold).long()
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
#  DATA LOADING
# =============================================================================

def load_features(cache_path: Path) -> Dict[str, torch.Tensor]:
    """Load cached features from .pt file"""
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    data = torch.load(cache_path, weights_only=False)
    return data


def combine_datasets(
    apcdata_train: Dict,
    apcdata_val: Dict,
    sipakmed_train: Dict = None,
    sipakmed_val: Dict = None
) -> Tuple[Dict, Dict]:
    """
    Combine APCData and SIPaKMeD features.

    Returns:
        combined_train, combined_val: Dicts with 'features', 'class_labels', etc.
    """

    # Start with APCData
    train_features = [apcdata_train['features']]
    train_class = [apcdata_train['class_labels']]
    train_binary = [apcdata_train['binary_labels']]
    train_severity = [apcdata_train['severity_labels']]

    val_features = [apcdata_val['features']]
    val_class = [apcdata_val['class_labels']]
    val_binary = [apcdata_val['binary_labels']]
    val_severity = [apcdata_val['severity_labels']]

    # Add SIPaKMeD if available
    if sipakmed_train is not None:
        train_features.append(sipakmed_train['features'])
        train_class.append(sipakmed_train['class_labels'])
        train_binary.append(sipakmed_train['binary_labels'])
        train_severity.append(sipakmed_train['severity_labels'])
        print(f"  [INFO] Added SIPaKMeD train: {len(sipakmed_train['features'])} samples")

    if sipakmed_val is not None:
        val_features.append(sipakmed_val['features'])
        val_class.append(sipakmed_val['class_labels'])
        val_binary.append(sipakmed_val['binary_labels'])
        val_severity.append(sipakmed_val['severity_labels'])
        print(f"  [INFO] Added SIPaKMeD val: {len(sipakmed_val['features'])} samples")

    # Concatenate
    combined_train = {
        'features': torch.cat(train_features, dim=0),
        'class_labels': torch.cat(train_class, dim=0),
        'binary_labels': torch.cat(train_binary, dim=0),
        'severity_labels': torch.cat(train_severity, dim=0)
    }

    combined_val = {
        'features': torch.cat(val_features, dim=0),
        'class_labels': torch.cat(val_class, dim=0),
        'binary_labels': torch.cat(val_binary, dim=0),
        'severity_labels': torch.cat(val_severity, dim=0)
    }

    return combined_train, combined_val


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
    lr: float = 1e-3
) -> MultiHeadBethesdaClassifier:
    """Train multi-head classifier."""

    model = MultiHeadBethesdaClassifier(input_dim=train_features.shape[1])
    model = model.to(device)

    # Class weights based on actual distribution
    class_counts = Counter(train_class_labels.numpy())
    total = sum(class_counts.values())

    # Compute inverse frequency weights, capped
    finegrained_weights = []
    for i in range(6):
        count = class_counts.get(i, 1)  # Avoid division by zero
        weight = min(total / (6 * count), 3.0)  # Cap at 3x
        finegrained_weights.append(weight)

    finegrained_weights = torch.tensor(finegrained_weights).float().to(device)
    print(f"  [INFO] Fine-grained weights: {finegrained_weights.tolist()}")

    # Binary: weight abnormal higher for sensitivity
    binary_weights = torch.tensor([0.3, 1.0]).to(device)
    criterion_binary = nn.CrossEntropyLoss(weight=binary_weights)

    # Severity: weight high-grade higher
    severity_weights = torch.tensor([0.5, 1.0]).to(device)
    criterion_severity = nn.CrossEntropyLoss(weight=severity_weights, ignore_index=-1)

    # Fine-grained: use computed weights
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

    print("\n  Training...")

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

            loss_binary = criterion_binary(outputs['binary'], binary_labels)
            loss_severity = criterion_severity(outputs['severity'], severity_labels)
            loss_finegrained = criterion_finegrained(outputs['finegrained'], class_labels)

            loss = loss_binary + loss_severity + loss_finegrained

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        all_binary_preds = []
        all_binary_gt = []
        all_finegrained_preds = []
        all_finegrained_gt = []

        with torch.no_grad():
            for features, class_labels, binary_labels, severity_labels in val_loader:
                features = features.to(device)

                preds = model.predict(features)

                all_binary_preds.extend(preds['binary_pred'].cpu().numpy())
                all_binary_gt.extend(binary_labels.numpy())
                all_finegrained_preds.extend(preds['finegrained_pred'].cpu().numpy())
                all_finegrained_gt.extend(class_labels.numpy())

        # Metrics
        all_binary_preds = np.array(all_binary_preds)
        all_binary_gt = np.array(all_binary_gt)

        abnormal_mask = all_binary_gt == 1
        if abnormal_mask.sum() > 0:
            binary_recall = (all_binary_preds[abnormal_mask] == 1).mean()
        else:
            binary_recall = 0.0

        normal_mask = all_binary_gt == 0
        if normal_mask.sum() > 0:
            binary_specificity = (all_binary_preds[normal_mask] == 0).mean()
        else:
            binary_specificity = 0.0

        finegrained_acc = balanced_accuracy_score(all_finegrained_gt, all_finegrained_preds)

        # Save best model
        if binary_recall > best_binary_recall:
            best_binary_recall = binary_recall
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={train_loss/len(train_loader):.4f}, "
                  f"Binary(Recall/Spec)={binary_recall:.3f}/{binary_specificity:.3f}, "
                  f"FineAcc={finegrained_acc:.3f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n  [OK] Best Binary Recall: {best_binary_recall:.4f}")

    return model


def evaluate_model(
    model: MultiHeadBethesdaClassifier,
    features: torch.Tensor,
    class_labels: torch.Tensor,
    binary_labels: torch.Tensor,
    severity_labels: torch.Tensor,
    device: torch.device
) -> Dict:
    """Evaluate model."""

    model.eval()
    features = features.to(device)

    with torch.no_grad():
        preds = model.predict(features)

    binary_preds = preds['binary_pred'].cpu().numpy()
    binary_gt = binary_labels.numpy()
    finegrained_preds = preds['finegrained_pred'].cpu().numpy()
    finegrained_gt = class_labels.numpy()
    severity_preds = preds['severity_pred'].cpu().numpy()
    severity_gt = severity_labels.numpy()

    results = {}

    # Binary
    abnormal_mask = binary_gt == 1
    normal_mask = binary_gt == 0
    results['binary'] = {
        'recall': (binary_preds[abnormal_mask] == 1).mean() if abnormal_mask.sum() > 0 else 0,
        'specificity': (binary_preds[normal_mask] == 0).mean() if normal_mask.sum() > 0 else 0,
        'balanced_accuracy': balanced_accuracy_score(binary_gt, binary_preds)
    }

    # Severity
    abnormal_indices = severity_gt >= 0
    if abnormal_indices.sum() > 0:
        sev_preds = severity_preds[abnormal_indices]
        sev_gt = severity_gt[abnormal_indices]
        high_mask = sev_gt == 1
        low_mask = sev_gt == 0
        results['severity'] = {
            'recall_high': (sev_preds[high_mask] == 1).mean() if high_mask.sum() > 0 else 0,
            'specificity_low': (sev_preds[low_mask] == 0).mean() if low_mask.sum() > 0 else 0,
            'balanced_accuracy': balanced_accuracy_score(sev_gt, sev_preds)
        }

    # Fine-grained
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

    results['confusion_matrix'] = confusion_matrix(finegrained_gt, finegrained_preds).tolist()

    return results


# =============================================================================
#  MAIN
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train MultiHead Bethesda Classifier on Combined Data"
    )
    parser.add_argument(
        "--apcdata_cache",
        type=str,
        default="data/raw/apcdata/APCData_YOLO/cache_cells",
        help="Path to APCData cached features"
    )
    parser.add_argument(
        "--sipakmed_cache",
        type=str,
        default="data/cache/sipakmed_features",
        help="Path to SIPaKMeD cached features"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/cytology/multihead_bethesda_combined.pt",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  MULTIHEAD BETHESDA — COMBINED TRAINING")
    print("  APCData + SIPaKMeD")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"  [INFO] Device: {device}")

    apcdata_cache = Path(args.apcdata_cache)
    sipakmed_cache = Path(args.sipakmed_cache)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load APCData
    print_header("LOADING APCDATA")

    apcdata_train_path = apcdata_cache / "train_cell_features.pt"
    apcdata_val_path = apcdata_cache / "val_cell_features.pt"

    if not apcdata_train_path.exists():
        print(f"  [ERROR] APCData cache not found: {apcdata_train_path}")
        print("  Run: python scripts/cytology/08_train_multihead_bethesda.py --cache_features")
        return 1

    apcdata_train = load_features(apcdata_train_path)
    apcdata_val = load_features(apcdata_val_path)
    print(f"  [OK] APCData train: {len(apcdata_train['features'])} samples")
    print(f"  [OK] APCData val: {len(apcdata_val['features'])} samples")

    # Load SIPaKMeD (optional)
    print_header("LOADING SIPAKMED")

    sipakmed_train_path = sipakmed_cache / "sipakmed_train_features.pt"
    sipakmed_val_path = sipakmed_cache / "sipakmed_val_features.pt"

    sipakmed_train = None
    sipakmed_val = None

    if sipakmed_train_path.exists():
        sipakmed_train = load_features(sipakmed_train_path)
        sipakmed_val = load_features(sipakmed_val_path)
        print(f"  [OK] SIPaKMeD train: {len(sipakmed_train['features'])} samples")
        print(f"  [OK] SIPaKMeD val: {len(sipakmed_val['features'])} samples")
    else:
        print(f"  [WARN] SIPaKMeD not found: {sipakmed_train_path}")
        print("  Training with APCData only.")
        print("  To add SIPaKMeD, run: python scripts/cytology/09_extract_sipakmed_features.py")

    # Combine datasets
    print_header("COMBINING DATASETS")

    combined_train, combined_val = combine_datasets(
        apcdata_train, apcdata_val,
        sipakmed_train, sipakmed_val
    )

    print(f"\n  Combined train: {len(combined_train['features'])} samples")
    print(f"  Combined val: {len(combined_val['features'])} samples")

    # Class distribution
    print("\n  Class distribution (train):")
    class_counts = Counter(combined_train['class_labels'].numpy())
    for class_id, class_name in BETHESDA_CLASSES.items():
        count = class_counts.get(class_id, 0)
        pct = count / len(combined_train['features']) * 100
        marker = "⚠️" if count < 200 else ""
        print(f"    {class_name}: {count} ({pct:.1f}%) {marker}")

    # Train
    print_header("TRAINING")

    model = train_multihead_classifier(
        combined_train['features'],
        combined_train['class_labels'],
        combined_train['binary_labels'],
        combined_train['severity_labels'],
        combined_val['features'],
        combined_val['class_labels'],
        combined_val['binary_labels'],
        combined_val['severity_labels'],
        device,
        epochs=args.epochs
    )

    # Evaluate
    print_header("EVALUATION")

    results = evaluate_model(
        model,
        combined_val['features'],
        combined_val['class_labels'],
        combined_val['binary_labels'],
        combined_val['severity_labels'],
        device
    )

    # Print results
    print("\n  === Binary Classification ===")
    print(f"    Recall (Abnormal):     {results['binary']['recall']:.4f}")
    print(f"    Specificity (Normal):  {results['binary']['specificity']:.4f}")
    print(f"    Balanced Accuracy:     {results['binary']['balanced_accuracy']:.4f}")

    if 'severity' in results:
        print("\n  === Severity Classification ===")
        print(f"    Recall (High-grade):   {results['severity']['recall_high']:.4f}")
        print(f"    Specificity (Low):     {results['severity']['specificity_low']:.4f}")

    print("\n  === Fine-grained Classification ===")
    print(f"    Balanced Accuracy:     {results['finegrained']['balanced_accuracy']:.4f}")
    print("\n    Per-class recall:")
    for class_name, metrics in results['finegrained']['per_class'].items():
        print(f"      {class_name:6s}: {metrics['recall']:.3f} (n={metrics['count']})")

    # Confusion matrix
    print("\n  Confusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print("         ", "  ".join([f"{BETHESDA_CLASSES[i][:4]:>4s}" for i in range(6)]))
    for i in range(6):
        if i < len(cm):
            print(f"    {BETHESDA_CLASSES[i]:5s}:", "  ".join([f"{cm[i,j]:4d}" for j in range(min(6, len(cm[i])))]))

    # Save
    print_header("SAVING")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': 1536,
        'hidden_dims': (512, 256),
        'val_results': results,
        'datasets': {
            'apcdata_train': len(apcdata_train['features']),
            'apcdata_val': len(apcdata_val['features']),
            'sipakmed_train': len(sipakmed_train['features']) if sipakmed_train else 0,
            'sipakmed_val': len(sipakmed_val['features']) if sipakmed_val else 0,
            'total_train': len(combined_train['features']),
            'total_val': len(combined_val['features'])
        }
    }

    torch.save(checkpoint, output_path)
    print(f"  [OK] Model saved to {output_path}")

    # Summary
    print_header("SUMMARY")

    print(f"  Binary Recall:           {results['binary']['recall']:.4f}")
    if 'severity' in results:
        print(f"  Severity Recall (High):  {results['severity']['recall_high']:.4f}")
    print(f"  Fine-grained Bal. Acc:   {results['finegrained']['balanced_accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
