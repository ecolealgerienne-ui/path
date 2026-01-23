"""
Cell Triage Training — Binary Classifier (Cell / No-cell)

Ce script entraîne un classificateur binaire rapide pour filtrer les patches
sans cellules avant la classification Bethesda complète.

Objectif:
- Recall > 99% (ne jamais rater une cellule)
- Filtrer 60-70% des patches vides
- Accélérer le pipeline de 3-5×

Architecture:
    H-Optimus embeddings (1536D)
        ↓
    MLP (1536 → 256 → 64 → 2)
        ↓
    Binary: Cell / No-cell

Usage:
    # Entraîner le triage classifier
    python scripts/cytology/07_train_cell_triage.py \
        --tiled_dir data/raw/apcdata/APCData_YOLO_Tiled_672 \
        --output models/cytology/cell_triage.pt

Author: V15 Cytology Branch
Date: 2026-01-23
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import json

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from tqdm import tqdm


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224


# ═════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═════════════════════════════════════════════════════════════════════════════

class CellTriageDataset(Dataset):
    """
    Dataset pour entraînement du triage binaire.

    Lit les tuiles et leurs labels YOLO pour déterminer:
    - Classe 1 (Cell): tuile avec au moins 1 annotation
    - Classe 0 (Empty): tuile sans annotation
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transform=None
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        # Collect all images
        self.image_paths = list(self.images_dir.glob("*.jpg")) + \
                          list(self.images_dir.glob("*.png"))

        # Determine labels (has cells or not)
        self.labels = []
        for img_path in self.image_paths:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            has_cells = self._has_annotations(label_path)
            self.labels.append(1 if has_cells else 0)

        # Statistics
        self.num_cells = sum(self.labels)
        self.num_empty = len(self.labels) - self.num_cells

    def _has_annotations(self, label_path: Path) -> bool:
        """Check if label file has any annotations"""
        if not label_path.exists():
            return False
        with open(label_path) as f:
            content = f.read().strip()
            return len(content) > 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if image.shape[0] != HOPTIMUS_INPUT_SIZE or image.shape[1] != HOPTIMUS_INPUT_SIZE:
            image = cv2.resize(image, (HOPTIMUS_INPUT_SIZE, HOPTIMUS_INPUT_SIZE))

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(HOPTIMUS_MEAN)) / np.array(HOPTIMUS_STD)

        # To tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, label, str(img_path)


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═════════════════════════════════════════════════════════════════════════════

class CellTriageClassifier(nn.Module):
    """
    Simple MLP pour classification binaire Cell/No-cell

    Architecture optimisée pour:
    - Recall très élevé (> 99%)
    - Inférence rapide
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dims: Tuple[int, int] = (256, 64),
        dropout: float = 0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dims[1], 2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: H-Optimus embeddings (B, 1536)

        Returns:
            logits: (B, 2) for [Empty, Cell]
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with adjustable threshold for high recall

        Args:
            x: H-Optimus embeddings (B, 1536)
            threshold: Probability threshold for "Cell" class (lower = higher recall)

        Returns:
            predictions: (B,) binary predictions
            probabilities: (B, 2) softmax probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)

            # Use threshold on Cell probability (class 1)
            predictions = (probs[:, 1] >= threshold).long()

            return predictions, probs


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_features_hoptimus(
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract H-Optimus features for all samples

    Returns:
        features: (N, 1536)
        labels: (N,)
    """
    print("  [INFO] Loading H-Optimus-0...")

    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "bioptimus/H-optimus-0",
            trust_remote_code=True
        )
        model = model.to(device)
        model.eval()

        print(f"  [OK] H-Optimus-0 loaded on {device}")

    except Exception as e:
        print(f"  [ERROR] Failed to load H-Optimus-0: {e}")
        return None, None

    all_features = []
    all_labels = []

    print("  [INFO] Extracting features...")

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)

            # Extract features
            features = model(images)

            # Get CLS token
            if len(features.shape) == 3:
                cls_tokens = features[:, 0, :]  # (B, 1536)
            else:
                cls_tokens = features

            all_features.append(cls_tokens.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"  [OK] Extracted features: {features.shape}")

    return features, labels


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    class_weight_empty: float = 0.3,  # Lower weight for empty class
    class_weight_cell: float = 1.0    # Higher weight for cell class (prioritize recall)
) -> CellTriageClassifier:
    """
    Train the triage classifier with class weighting for high recall
    """

    # Create model
    model = CellTriageClassifier(input_dim=train_features.shape[1])
    model = model.to(device)

    # Class weights for high recall on cells
    class_weights = torch.tensor([class_weight_empty, class_weight_cell]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create simple dataset from features
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_recall = 0.0
    best_model_state = None

    print("\n  Training...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                preds, _ = model.predict(features, threshold=0.3)  # Low threshold for high recall
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        cell_mask = all_labels == 1
        if cell_mask.sum() > 0:
            recall_cell = (all_preds[cell_mask] == 1).mean()
        else:
            recall_cell = 0.0

        empty_mask = all_labels == 0
        if empty_mask.sum() > 0:
            precision_empty = (all_preds[empty_mask] == 0).mean()
        else:
            precision_empty = 0.0

        bal_acc = balanced_accuracy_score(all_labels, all_preds)

        # Save best model (based on cell recall)
        if recall_cell > best_recall:
            best_recall = recall_cell
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={train_loss/len(train_loader):.4f}, "
                  f"Recall(Cell)={recall_cell:.4f}, Precision(Empty)={precision_empty:.4f}, "
                  f"BalAcc={bal_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n  [OK] Best Recall (Cell): {best_recall:.4f}")

    return model


def evaluate_thresholds(
    model: CellTriageClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate model at different thresholds to find optimal operating point
    """
    model.eval()
    features = features.to(device)

    results = {}

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        preds, probs = model.predict(features, threshold=threshold)
        preds = preds.cpu().numpy()
        labels_np = labels.numpy()

        # Metrics
        cell_mask = labels_np == 1
        empty_mask = labels_np == 0

        recall_cell = (preds[cell_mask] == 1).mean() if cell_mask.sum() > 0 else 0
        precision_empty = (preds[empty_mask] == 0).mean() if empty_mask.sum() > 0 else 0

        # How many patches filtered (predicted as empty)
        filter_rate = (preds == 0).mean()

        results[threshold] = {
            'recall_cell': recall_cell,
            'precision_empty': precision_empty,
            'filter_rate': filter_rate,
            'balanced_accuracy': balanced_accuracy_score(labels_np, preds)
        }

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

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
        description="Train Cell Triage Binary Classifier"
    )
    parser.add_argument(
        "--tiled_dir",
        type=str,
        default="data/raw/apcdata/APCData_YOLO_Tiled_672",
        help="Path to tiled APCData directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/cytology/cell_triage.pt",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
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
    print("  CELL TRIAGE CLASSIFIER TRAINING")
    print("  V15 Cytology Pipeline")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print_info(f"Device: {device}")

    tiled_dir = Path(args.tiled_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify input
    print_header("LOADING DATA")

    train_images = tiled_dir / "train" / "images"
    train_labels = tiled_dir / "train" / "labels"
    val_images = tiled_dir / "val" / "images"
    val_labels = tiled_dir / "val" / "labels"

    if not train_images.exists():
        print(f"  [ERROR] Train images not found: {train_images}")
        return 1

    # Create datasets
    train_dataset = CellTriageDataset(train_images, train_labels)
    val_dataset = CellTriageDataset(val_images, val_labels)

    print_success(f"Train: {len(train_dataset)} samples ({train_dataset.num_cells} cells, {train_dataset.num_empty} empty)")
    print_success(f"Val: {len(val_dataset)} samples ({val_dataset.num_cells} cells, {val_dataset.num_empty} empty)")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Check for cached features
    cache_dir = tiled_dir / "cache"
    train_features_path = cache_dir / "train_features.pt"
    val_features_path = cache_dir / "val_features.pt"

    if train_features_path.exists() and val_features_path.exists():
        print_header("LOADING CACHED FEATURES")
        train_data = torch.load(train_features_path)
        val_data = torch.load(val_features_path)
        train_features = train_data['features']
        train_labels_tensor = train_data['labels']
        val_features = val_data['features']
        val_labels_tensor = val_data['labels']
        print_success(f"Loaded cached features")
    else:
        # Extract features
        print_header("EXTRACTING FEATURES")

        train_features, train_labels_tensor = extract_features_hoptimus(train_loader, device)
        val_features, val_labels_tensor = extract_features_hoptimus(val_loader, device)

        if train_features is None:
            return 1

        # Cache features
        if args.cache_features:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'features': train_features, 'labels': train_labels_tensor}, train_features_path)
            torch.save({'features': val_features, 'labels': val_labels_tensor}, val_features_path)
            print_success(f"Cached features to {cache_dir}")

    # Train classifier
    print_header("TRAINING CLASSIFIER")

    model = train_classifier(
        train_features, train_labels_tensor,
        val_features, val_labels_tensor,
        device,
        epochs=args.epochs
    )

    # Evaluate at different thresholds
    print_header("THRESHOLD ANALYSIS")

    results = evaluate_thresholds(model, val_features, val_labels_tensor, device)

    print("\n  Threshold | Recall(Cell) | Filter Rate | Balanced Acc")
    print("  " + "-" * 55)

    best_threshold = 0.3
    for threshold, metrics in sorted(results.items()):
        marker = " ★" if threshold == best_threshold else ""
        print(f"    {threshold:.1f}     |    {metrics['recall_cell']:.4f}    |   {metrics['filter_rate']:.4f}    |   {metrics['balanced_accuracy']:.4f}{marker}")

    # Save model
    print_header("SAVING MODEL")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': 1536,
        'hidden_dims': (256, 64),
        'recommended_threshold': best_threshold,
        'val_metrics': results[best_threshold],
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }

    torch.save(checkpoint, output_path)
    print_success(f"Model saved to {output_path}")

    # Final summary
    print_header("SUMMARY")

    best_metrics = results[best_threshold]
    print_info(f"Recommended threshold: {best_threshold}")
    print_info(f"Recall (Cell): {best_metrics['recall_cell']:.4f} ({best_metrics['recall_cell']*100:.1f}%)")
    print_info(f"Filter rate: {best_metrics['filter_rate']:.4f} ({best_metrics['filter_rate']*100:.1f}% patches filtered)")
    print_info(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETED")
    print("=" * 80)
    print(f"\n  Model: {output_path}")
    print(f"\n  Usage in sliding window:")
    print(f"    python scripts/cytology/06_sliding_window_inference.py \\")
    print(f"        --image path/to/image.jpg \\")
    print(f"        --triage_model {output_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
