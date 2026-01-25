"""
Cell Triage v2 Training — Binary Classifier with H-Channel Augmentation

Ce script entraîne un classificateur binaire avec features augmentées:
- CLS token H-Optimus (1536D)
- H-Stats (4D): h_mean, h_std, nuclei_count, nuclei_area_ratio

Total: 1540D features

Amélioration vs v1:
- Validation structurelle via H-Channel (Ruifrok deconvolution)
- Meilleure discrimination cell/empty grâce aux features nucléaires
- Réduction des faux négatifs sur patches avec coloration atypique

Architecture:
    Image RGB (224×224)
        ↓
    ┌───────────────────┬────────────────────┐
    │                   │                    │
    ↓                   ↓                    │
    H-Optimus-0     Ruifrok Deconv           │
    (CLS: 1536D)    (H-Stats: 4D)            │
    │                   │                    │
    └─────────┬─────────┘                    │
              ↓                              │
    Concat (1540D)                           │
              ↓                              │
    MLP (1540 → 256 → 64 → 2)                │
              ↓                              │
    Binary: Cell / No-cell                   │

Usage:
    python scripts/cytology/07b_train_cell_triage_v2.py \
        --tiled_dir data/raw/apcdata/APCData_YOLO_Tiled_672 \
        --output models/cytology/cell_triage_v2.pt

Author: V15.3 Cytology Branch
Date: 2026-01-24
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

# Import H-Channel module (Phase 1)
from src.preprocessing.h_channel import compute_h_stats, HChannelStats


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224

# Feature dimensions
CLS_DIM = 1536
H_STATS_DIM = 4
TOTAL_DIM = CLS_DIM + H_STATS_DIM  # 1540


# ═════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═════════════════════════════════════════════════════════════════════════════

class CellTriageDatasetV2(Dataset):
    """
    Dataset pour entraînement du triage binaire avec H-Stats.

    Retourne:
    - image: tensor (3, 224, 224) normalisé pour H-Optimus
    - image_rgb: numpy array (224, 224, 3) pour H-Stats extraction
    - label: 0 (empty) ou 1 (cell)
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

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

        # Keep original RGB for H-Stats (uint8)
        image_rgb = image.copy()

        # Normalize for H-Optimus
        image_norm = image.astype(np.float32) / 255.0
        image_norm = (image_norm - np.array(HOPTIMUS_MEAN, dtype=np.float32)) / np.array(HOPTIMUS_STD, dtype=np.float32)

        # To tensor (C, H, W)
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()

        return image_tensor, image_rgb, label, str(img_path)


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═════════════════════════════════════════════════════════════════════════════

class CellTriageClassifierV2(nn.Module):
    """
    MLP pour classification binaire avec features augmentées.

    Input: CLS (1536D) + H-Stats (4D) = 1540D
    Output: Binary logits (2D)
    """

    def __init__(
        self,
        input_dim: int = TOTAL_DIM,  # 1540
        hidden_dims: Tuple[int, int] = (256, 64),
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim

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
            x: Augmented features (B, 1540) = CLS (1536) + H-Stats (4)

        Returns:
            logits: (B, 2) for [Empty, Cell]
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with adjustable threshold for high recall

        Args:
            x: Augmented features (B, 1540)
            threshold: Probability threshold for "Cell" class

        Returns:
            predictions: (B,) binary predictions
            probabilities: (B, 2) softmax probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            predictions = (probs[:, 1] >= threshold).long()
            return predictions, probs


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_augmented_features(
    dataloader: DataLoader,
    device: torch.device,
    use_cached_cls: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract augmented features: CLS (1536D) + H-Stats (4D)

    Args:
        dataloader: DataLoader yielding (image_tensor, image_rgb, label, path)
        device: torch device
        use_cached_cls: Optional pre-extracted CLS tokens to skip H-Optimus

    Returns:
        features: (N, 1540) augmented features
        h_stats_features: (N, 4) H-Stats only (for analysis)
        labels: (N,) labels
    """

    # Load H-Optimus if not using cached CLS
    model = None
    if use_cached_cls is None:
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
            return None, None, None

    all_cls_features = []
    all_h_stats = []
    all_labels = []

    print("  [INFO] Extracting augmented features (CLS + H-Stats)...")

    batch_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            image_tensors, image_rgbs, labels, paths = batch

            # Extract CLS tokens
            if use_cached_cls is not None:
                # Use cached CLS tokens
                start_idx = batch_idx * dataloader.batch_size
                end_idx = start_idx + len(labels)
                cls_tokens = use_cached_cls[start_idx:end_idx]
            else:
                # Extract from H-Optimus
                image_tensors = image_tensors.to(device)
                features = model.forward_features(image_tensors)

                if len(features.shape) == 3:
                    cls_tokens = features[:, 0, :].cpu()  # (B, 1536)
                else:
                    cls_tokens = features.cpu()

            # Extract H-Stats for each image in batch
            batch_h_stats = []
            for i in range(len(image_rgbs)):
                # image_rgbs[i] is numpy array (H, W, 3)
                rgb = image_rgbs[i].numpy() if isinstance(image_rgbs[i], torch.Tensor) else image_rgbs[i]
                stats = compute_h_stats(rgb)
                h_features = stats.to_features()  # (4,)
                batch_h_stats.append(h_features)

            h_stats_batch = np.stack(batch_h_stats)  # (B, 4)
            h_stats_batch = torch.from_numpy(h_stats_batch)

            all_cls_features.append(cls_tokens)
            all_h_stats.append(h_stats_batch)
            all_labels.append(labels)

            batch_idx += 1

    # Concatenate
    cls_features = torch.cat(all_cls_features, dim=0)  # (N, 1536)
    h_stats_features = torch.cat(all_h_stats, dim=0)   # (N, 4)
    labels = torch.cat(all_labels, dim=0)               # (N,)

    # Augmented features
    augmented_features = torch.cat([cls_features, h_stats_features], dim=1)  # (N, 1540)

    print(f"  [OK] Extracted features: {augmented_features.shape}")
    print(f"       CLS: {cls_features.shape}, H-Stats: {h_stats_features.shape}")

    return augmented_features, h_stats_features, labels


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
    class_weight_empty: float = 0.3,
    class_weight_cell: float = 1.0
) -> CellTriageClassifierV2:
    """
    Train the v2 triage classifier with augmented features
    """

    # Create model
    model = CellTriageClassifierV2(input_dim=train_features.shape[1])
    model = model.to(device)

    # Class weights for high recall on cells
    class_weights = torch.tensor([class_weight_empty, class_weight_cell]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_recall = 0.0
    best_model_state = None

    print("\n  Training Cell Triage v2...")

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
        all_labels_list = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                preds, _ = model.predict(features, threshold=0.3)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels_np = np.array(all_labels_list)

        # Metrics
        cell_mask = all_labels_np == 1
        recall_cell = (all_preds[cell_mask] == 1).mean() if cell_mask.sum() > 0 else 0

        empty_mask = all_labels_np == 0
        precision_empty = (all_preds[empty_mask] == 0).mean() if empty_mask.sum() > 0 else 0

        bal_acc = balanced_accuracy_score(all_labels_np, all_preds)

        # Save best model
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
    model: CellTriageClassifierV2,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate model at different thresholds
    """
    model.eval()
    features = features.to(device)

    results = {}

    for threshold in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        preds, probs = model.predict(features, threshold=threshold)
        preds = preds.cpu().numpy()
        labels_np = labels.numpy()

        cell_mask = labels_np == 1
        empty_mask = labels_np == 0

        recall_cell = (preds[cell_mask] == 1).mean() if cell_mask.sum() > 0 else 0
        precision_empty = (preds[empty_mask] == 0).mean() if empty_mask.sum() > 0 else 0
        filter_rate = (preds == 0).mean()

        results[threshold] = {
            'recall_cell': recall_cell,
            'precision_empty': precision_empty,
            'filter_rate': filter_rate,
            'balanced_accuracy': balanced_accuracy_score(labels_np, preds)
        }

    return results


def analyze_h_stats_contribution(
    h_stats_features: torch.Tensor,
    labels: torch.Tensor
) -> Dict:
    """
    Analyze how H-Stats differ between cell and empty patches
    """
    h_stats = h_stats_features.numpy()
    labels_np = labels.numpy()

    cell_mask = labels_np == 1
    empty_mask = labels_np == 0

    feature_names = ['h_mean', 'h_std', 'nuclei_count', 'nuclei_area_ratio']

    analysis = {}
    for i, name in enumerate(feature_names):
        cell_values = h_stats[cell_mask, i]
        empty_values = h_stats[empty_mask, i]

        analysis[name] = {
            'cell_mean': float(np.mean(cell_values)),
            'cell_std': float(np.std(cell_values)),
            'empty_mean': float(np.mean(empty_values)),
            'empty_std': float(np.std(empty_values)),
            'separation': float(np.abs(np.mean(cell_values) - np.mean(empty_values)))
        }

    return analysis


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
        description="Train Cell Triage v2 with H-Channel Augmentation"
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
        default="models/cytology/cell_triage_v2.pt",
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
        "--use_cached_cls",
        action="store_true",
        help="Use cached CLS features from v1 (skip H-Optimus extraction)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  CELL TRIAGE v2 TRAINING")
    print("  With H-Channel Augmentation (CLS 1536D + H-Stats 4D)")
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
    train_dataset = CellTriageDatasetV2(train_images, train_labels)
    val_dataset = CellTriageDatasetV2(val_images, val_labels)

    print_success(f"Train: {len(train_dataset)} samples ({train_dataset.num_cells} cells, {train_dataset.num_empty} empty)")
    print_success(f"Val: {len(val_dataset)} samples ({val_dataset.num_cells} cells, {val_dataset.num_empty} empty)")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Check for cached features
    cache_dir = tiled_dir / "cache"
    train_features_v2_path = cache_dir / "train_features_v2.pt"
    val_features_v2_path = cache_dir / "val_features_v2.pt"

    # Check for cached CLS from v1
    cached_cls_train = None
    cached_cls_val = None

    if args.use_cached_cls:
        train_v1_path = cache_dir / "train_features.pt"
        val_v1_path = cache_dir / "val_features.pt"

        if train_v1_path.exists() and val_v1_path.exists():
            print_info("Using cached CLS features from v1")
            cached_cls_train = torch.load(train_v1_path)['features']
            cached_cls_val = torch.load(val_v1_path)['features']
        else:
            print_info("Cached CLS not found, will extract from H-Optimus")

    # Check for v2 cached features
    if train_features_v2_path.exists() and val_features_v2_path.exists():
        print_header("LOADING CACHED V2 FEATURES")
        train_data = torch.load(train_features_v2_path)
        val_data = torch.load(val_features_v2_path)
        train_features = train_data['features']
        train_h_stats = train_data['h_stats']
        train_labels_tensor = train_data['labels']
        val_features = val_data['features']
        val_h_stats = val_data['h_stats']
        val_labels_tensor = val_data['labels']
        print_success(f"Loaded cached v2 features: {train_features.shape}")
    else:
        # Extract augmented features
        print_header("EXTRACTING AUGMENTED FEATURES")

        train_features, train_h_stats, train_labels_tensor = extract_augmented_features(
            train_loader, device, cached_cls_train
        )
        val_features, val_h_stats, val_labels_tensor = extract_augmented_features(
            val_loader, device, cached_cls_val
        )

        if train_features is None:
            return 1

        # Cache features
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'features': train_features,
            'h_stats': train_h_stats,
            'labels': train_labels_tensor
        }, train_features_v2_path)
        torch.save({
            'features': val_features,
            'h_stats': val_h_stats,
            'labels': val_labels_tensor
        }, val_features_v2_path)
        print_success(f"Cached v2 features to {cache_dir}")

    # Analyze H-Stats contribution
    print_header("H-STATS ANALYSIS")

    analysis = analyze_h_stats_contribution(train_h_stats, train_labels_tensor)

    print("\n  Feature         |  Cell (mean±std)  |  Empty (mean±std) | Separation")
    print("  " + "-" * 70)
    for name, stats in analysis.items():
        print(f"  {name:16s} | {stats['cell_mean']:.3f} ± {stats['cell_std']:.3f}    | "
              f"{stats['empty_mean']:.3f} ± {stats['empty_std']:.3f}     | {stats['separation']:.3f}")

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
        marker = " <-- recommended" if threshold == best_threshold else ""
        print(f"    {threshold:.2f}    |    {metrics['recall_cell']:.4f}    |   {metrics['filter_rate']:.4f}    |   {metrics['balanced_accuracy']:.4f}{marker}")

    # Save model
    print_header("SAVING MODEL")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'version': 'v2',
        'input_dim': TOTAL_DIM,  # 1540
        'cls_dim': CLS_DIM,      # 1536
        'h_stats_dim': H_STATS_DIM,  # 4
        'hidden_dims': (256, 64),
        'recommended_threshold': best_threshold,
        'val_metrics': results[best_threshold],
        'h_stats_analysis': analysis,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }

    torch.save(checkpoint, output_path)
    print_success(f"Model saved to {output_path}")

    # Final summary
    print_header("SUMMARY")

    best_metrics = results[best_threshold]
    print_info(f"Version: Cell Triage v2 (CLS + H-Stats)")
    print_info(f"Input dimension: {TOTAL_DIM} (CLS: {CLS_DIM}, H-Stats: {H_STATS_DIM})")
    print_info(f"Recommended threshold: {best_threshold}")
    print_info(f"Recall (Cell): {best_metrics['recall_cell']:.4f} ({best_metrics['recall_cell']*100:.1f}%)")
    print_info(f"Filter rate: {best_metrics['filter_rate']:.4f} ({best_metrics['filter_rate']*100:.1f}% patches filtered)")
    print_info(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETED")
    print("=" * 80)
    print(f"\n  Model: {output_path}")
    print(f"\n  Usage:")
    print(f"    # In unified inference, use version='v2'")
    print(f"    # The model expects 1540D input (CLS + H-Stats)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
