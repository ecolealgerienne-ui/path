"""
MLP Classifier Training — V14 Cytology

Ce script entraîne le classificateur MLP pour le pipeline V14:
1. Charge les features fusionnées (1556 dims = 1536 CLS + 20 morpho)
2. Entraîne MLP avec BatchNorm et Dropout
3. Utilise Focal Loss pour gérer le déséquilibre de classes
4. Sauvegarde le meilleur modèle (validation loss)

Architecture MLP:
    Input (1556) → Linear(512) → BN → ReLU → Dropout
                → Linear(256) → BN → ReLU → Dropout
                → Linear(128) → BN → ReLU → Dropout
                → Linear(7) → Softmax

Author: V14 Cytology Branch
Date: 2026-01-20
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import seaborn as sns


# =============================================================================
#  CONSTANTS
# =============================================================================

SIPAKMED_CLASSES = [
    'normal_columnar',
    'normal_intermediate',
    'normal_superficiel',
    'light_dysplastic',
    'moderate_dysplastic',
    'severe_dysplastic',
    'carcinoma_in_situ'
]

# Binary grouping for sensitivity calculation
NORMAL_CLASSES = [0, 1, 2]  # normal_columnar, normal_intermediate, normal_superficiel
ABNORMAL_CLASSES = [3, 4, 5, 6]  # dysplastic + carcinoma


# =============================================================================
#  FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# =============================================================================
#  DATASET
# =============================================================================

class FeaturesDataset(Dataset):
    """Dataset for fused features"""

    def __init__(self, features_path: str):
        data = torch.load(features_path, weights_only=False)
        self.features = data['fused_features'].float()  # (N, 1556)
        self.labels = data['labels'].long()  # (N,)
        self.class_names = data['class_names']
        self.filenames = data['filenames']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# =============================================================================
#  MLP MODEL
# =============================================================================

class CytologyMLP(nn.Module):
    """
    MLP Classifier with BatchNorm for cytology classification

    Architecture:
        Input (1556) → 512 → 256 → 128 → 7 classes
    """

    def __init__(
        self,
        input_dim: int = 1556,
        hidden_dims: List[int] = [512, 256, 128],
        n_classes: int = 7,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

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

        # Output layer
        layers.append(nn.Linear(prev_dim, n_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# =============================================================================
#  TRAINING
# =============================================================================

def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Compute inverse frequency weights for balanced training"""
    class_counts = torch.bincount(labels, minlength=len(SIPAKMED_CLASSES))
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(SIPAKMED_CLASSES)
    return class_weights


def compute_sample_weights(labels: torch.Tensor) -> torch.Tensor:
    """Compute sample weights for WeightedRandomSampler"""
    class_weights = compute_class_weights(labels)
    sample_weights = class_weights[labels]
    return sample_weights


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def compute_sensitivity_metrics(
    preds: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute Safety First metrics (binary: normal vs abnormal)
    """
    # Convert to binary
    preds_binary = np.isin(preds, ABNORMAL_CLASSES).astype(int)
    labels_binary = np.isin(labels, ABNORMAL_CLASSES).astype(int)

    # True positives, false negatives, etc.
    tp = np.sum((preds_binary == 1) & (labels_binary == 1))
    fn = np.sum((preds_binary == 0) & (labels_binary == 1))
    fp = np.sum((preds_binary == 1) & (labels_binary == 0))
    tn = np.sum((preds_binary == 0) & (labels_binary == 0))

    # Sensitivity (recall for abnormal class) - CRITICAL
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Cohen's Kappa
    kappa = cohen_kappa_score(labels, preds)

    return {
        'sensitivity_abnormal': sensitivity,
        'specificity': specificity,
        'cohen_kappa': kappa,
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
    }


# =============================================================================
#  VISUALIZATION
# =============================================================================

def plot_training_history(history: Dict, output_path: str):
    """Plot training and validation loss/accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    output_path: str
):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Normalized)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
#  MAIN TRAINING
# =============================================================================

def train_model(
    features_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    use_focal_loss: bool = True,
    gamma: float = 2.0,
    dropout: float = 0.3,
    patience: int = 15,
    device: str = 'cuda'
):
    """Main training function"""

    print("=" * 80)
    print("MLP CLASSIFIER TRAINING — V14 Cytology")
    print("=" * 80)

    # Load data
    train_path = os.path.join(features_dir, 'sipakmed_train_features.pt')
    val_path = os.path.join(features_dir, 'sipakmed_val_features.pt')

    train_dataset = FeaturesDataset(train_path)
    val_dataset = FeaturesDataset(val_path)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Input dim:     {train_dataset.features.shape[1]}")
    print(f"Classes:       {len(SIPAKMED_CLASSES)}")

    # Class distribution
    print("\nClass distribution (train):")
    train_labels = train_dataset.labels.numpy()
    for i, cls in enumerate(SIPAKMED_CLASSES):
        count = (train_labels == i).sum()
        pct = count / len(train_labels) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")

    # Weighted sampler for balanced training
    sample_weights = compute_sample_weights(train_dataset.labels)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    input_dim = train_dataset.features.shape[1]
    model = CytologyMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        n_classes=len(SIPAKMED_CLASSES),
        dropout=dropout
    )
    model = model.to(device)

    print(f"\nModel architecture:")
    print(f"  Input:  {input_dim}")
    print(f"  Hidden: [512, 256, 128]")
    print(f"  Output: {len(SIPAKMED_CLASSES)}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    if use_focal_loss:
        class_weights = compute_class_weights(train_dataset.labels).to(device)
        criterion = FocalLoss(alpha=class_weights, gamma=gamma)
        print(f"\nLoss: Focal Loss (gamma={gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"\nLoss: CrossEntropy")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    print(f"Optimizer: Adam (lr={learning_rate})")
    print(f"Scheduler: ReduceLROnPlateau (patience=5)")
    print(f"Early stopping: {patience} epochs")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve_count = 0

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("Training...")
    print(f"{'='*80}\n")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # Train accuracy (quick estimate)
        _, train_acc, _, _ = evaluate(model, train_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Scheduler step
        scheduler.step(val_loss)

        # Compute sensitivity metrics
        sensitivity_metrics = compute_sensitivity_metrics(val_preds, val_labels)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Sens: {sensitivity_metrics['sensitivity_abnormal']:.4f} | "
              f"Kappa: {sensitivity_metrics['cohen_kappa']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve_count = 0

            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'sensitivity_metrics': sensitivity_metrics,
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dims': [512, 256, 128],
                    'n_classes': len(SIPAKMED_CLASSES),
                    'dropout': dropout,
                },
            }, checkpoint_path)
            print(f"  → Saved best model (epoch {epoch+1})")
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Final evaluation
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final metrics
    _, final_acc, final_preds, final_labels = evaluate(model, val_loader, criterion, device)
    final_sensitivity = compute_sensitivity_metrics(final_preds, final_labels)

    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy:    {final_acc:.4f}")
    print(f"  Sensitivity: {final_sensitivity['sensitivity_abnormal']:.4f} (abnormal detection)")
    print(f"  Specificity: {final_sensitivity['specificity']:.4f}")
    print(f"  Cohen Kappa: {final_sensitivity['cohen_kappa']:.4f}")

    # Safety First check
    print(f"\n{'='*40}")
    if final_sensitivity['sensitivity_abnormal'] >= 0.98:
        print("✅ SAFETY FIRST: Sensitivity >= 0.98")
    else:
        print(f"⚠️  SAFETY FIRST: Sensitivity {final_sensitivity['sensitivity_abnormal']:.4f} < 0.98")
    if final_sensitivity['cohen_kappa'] >= 0.80:
        print("✅ Cohen's Kappa >= 0.80")
    else:
        print(f"⚠️  Cohen's Kappa {final_sensitivity['cohen_kappa']:.4f} < 0.80")
    print(f"{'='*40}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        final_labels,
        final_preds,
        target_names=SIPAKMED_CLASSES,
        digits=3
    ))

    # Save plots
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))
    plot_confusion_matrix(
        final_labels,
        final_preds,
        SIPAKMED_CLASSES,
        os.path.join(output_dir, 'confusion_matrix.png')
    )

    # Save training log
    log_path = os.path.join(output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_accuracy': final_acc,
            'final_sensitivity': final_sensitivity,
            'history': history,
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'use_focal_loss': use_focal_loss,
                'gamma': gamma,
                'dropout': dropout,
            }
        }, f, indent=2)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - best_model.pth")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix.png")
    print(f"  - training_log.json")

    print(f"\n{'='*80}")
    print("Next step:")
    print("  python scripts/cytology/04_evaluate_cytology.py")
    print(f"{'='*80}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP classifier for V14 Cytology"
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='data/features/sipakmed',
        help='Features directory (from step 02)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/checkpoints_v14_cytology',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--use_focal_loss',
        action='store_true',
        default=True,
        help='Use Focal Loss (default: True)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=2.0,
        help='Focal Loss gamma parameter'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for training'
    )

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    train_model(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_focal_loss=args.use_focal_loss,
        gamma=args.gamma,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device
    )


if __name__ == '__main__':
    main()
