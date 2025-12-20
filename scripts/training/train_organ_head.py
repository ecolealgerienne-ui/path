#!/usr/bin/env python3
"""
Entraînement de l'OrganHead pour classification d'organe.

Utilise les CLS tokens pré-extraits de H-optimus-0 pour entraîner
un classifieur d'organe sur les 19 classes PanNuke.

Usage:
    python scripts/training/train_organ_head.py --fold 0 --epochs 50
    python scripts/training/train_organ_head.py --fold 0 --epochs 50 --linear_probe
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.organ_head import OrganHead, OrganHeadLoss, PANNUKE_ORGANS


class OrganDataset(Dataset):
    """
    Dataset pour classification d'organe.

    Charge les CLS tokens pré-extraits et les labels organe.
    """

    def __init__(self, data_dir: str, fold: int = 0):
        self.data_dir = Path(data_dir)
        self.fold = fold

        # Charger les features pré-extraites
        features_path = PROJECT_ROOT / "data" / "cache" / "pannuke_features" / f"fold{fold}_features.npz"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features non trouvées: {features_path}\n"
                f"Lancez: python scripts/preprocessing/extract_features.py --fold {fold}"
            )

        print(f"Chargement features fold {fold}...")
        data = np.load(features_path)

        # Extraire les CLS tokens (premier token)
        if 'layer_24' in data:
            features = data['layer_24']  # (N, 261, 1536)
        elif 'layer_23' in data:
            features = data['layer_23']
        else:
            raise KeyError("Features layer_24 non trouvées")

        self.cls_tokens = features[:, 0, :]  # (N, 1536)
        print(f"  CLS tokens: {self.cls_tokens.shape}")

        # Charger les labels organe
        types_path = self.data_dir / f"fold{fold}" / "types.npy"
        if not types_path.exists():
            raise FileNotFoundError(f"Types non trouvés: {types_path}")

        types = np.load(types_path)  # (N,) strings
        print(f"  Types: {types.shape}")

        # Convertir les noms d'organes en indices
        self.organ_to_idx = {organ: i for i, organ in enumerate(PANNUKE_ORGANS)}
        self.labels = np.array([self._get_organ_idx(t) for t in types])

        # Statistiques
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"  Organes uniques: {len(unique)}")
        for idx, count in zip(unique, counts):
            print(f"    {PANNUKE_ORGANS[idx]:20}: {count:5} ({count/len(self.labels)*100:.1f}%)")

        self.n_samples = len(self.cls_tokens)

    def _get_organ_idx(self, organ_name: str) -> int:
        """Convertit un nom d'organe en index."""
        # Nettoyer le nom
        organ_name = str(organ_name).strip()

        # Chercher correspondance exacte
        if organ_name in self.organ_to_idx:
            return self.organ_to_idx[organ_name]

        # Chercher correspondance partielle (case insensitive)
        organ_lower = organ_name.lower()
        for organ, idx in self.organ_to_idx.items():
            if organ.lower() in organ_lower or organ_lower in organ.lower():
                return idx

        # Si non trouvé, retourner 0 (Adrenal_gland) par défaut
        print(f"  ⚠️ Organe non reconnu: '{organ_name}', utilisant index 0")
        return 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        cls_token = torch.from_numpy(self.cls_tokens[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return cls_token, label


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Entraîne une epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for cls_tokens, labels in pbar:
        cls_tokens = cls_tokens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(cls_tokens)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Valide le modèle."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for cls_tokens, labels in tqdm(dataloader, desc="Val", leave=False):
        cls_tokens = cls_tokens.to(device)
        labels = labels.to(device)

        logits = model(cls_tokens)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def compute_per_class_accuracy(preds, labels, n_classes=19):
    """Calcule l'accuracy par classe."""
    per_class_acc = {}
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean()
            per_class_acc[PANNUKE_ORGANS[i]] = (acc, mask.sum())
    return per_class_acc


def main():
    parser = argparse.ArgumentParser(description="Entraînement OrganHead")
    parser.add_argument("--data_dir", type=str, default="data/pannuke",
                       help="Répertoire des données PanNuke")
    parser.add_argument("--fold", type=int, default=0,
                       help="Fold PanNuke (0, 1 ou 2)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Nombre d'epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Taille des batches")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Dimension cachée du MLP")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Proportion de validation")
    parser.add_argument("--linear_probe", action="store_true",
                       help="Utiliser un simple classificateur linéaire")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="Label smoothing")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Charger les données
    print("\n📦 Chargement des données...")
    dataset = OrganDataset(args.data_dir, args.fold)

    # Split train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\n  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Créer le modèle
    print("\n🏗️ Création du modèle...")
    if args.linear_probe:
        print("  Mode: Linear Probe (sans couche cachée)")
        model = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, 19),
        )
    else:
        print(f"  Mode: MLP (hidden_dim={args.hidden_dim})")
        model = OrganHead(
            embed_dim=1536,
            hidden_dim=args.hidden_dim,
            n_organs=19,
            dropout=args.dropout,
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {n_params:,}")

    # Calculer les class weights
    all_labels = dataset.labels
    class_weights = OrganHeadLoss.compute_class_weights(all_labels)
    class_weights = class_weights.to(device)
    print(f"\n  Class weights calculés (min={class_weights.min():.2f}, max={class_weights.max():.2f})")

    # Loss et optimizer
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Entraînement
    print("\n" + "=" * 60)
    print("🚀 ENTRAÎNEMENT")
    print("=" * 60)

    best_val_acc = 0
    best_epoch = 0

    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("=" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            checkpoint_path = checkpoint_dir / "organ_head_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  💾 Nouveau meilleur modèle sauvé! (Acc: {val_acc:.4f})")

    # Résultats finaux
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"Meilleure Val Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Checkpoint: {checkpoint_path}")

    # Charger le meilleur modèle et calculer les métriques par classe
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, val_preds, val_labels = validate(model, val_loader, criterion, device)
    per_class = compute_per_class_accuracy(val_preds, val_labels)

    print("\nAccuracy par organe:")
    for organ, (acc, count) in sorted(per_class.items(), key=lambda x: -x[1][0]):
        bar = "█" * int(acc * 20)
        print(f"  {organ:20}: {acc:.3f} ({count:4} samples) {bar}")

    # Calibrer OOD si c'est un OrganHead
    if isinstance(model, OrganHead):
        print("\n🔧 Calibration OOD...")
        all_cls_tokens = torch.from_numpy(dataset.cls_tokens).float().to(device)
        model.fit_ood(all_cls_tokens)

        # Re-sauvegarder avec OOD calibré
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': best_val_acc,
            'val_loss': val_loss,
            'args': vars(args),
            'ood_calibrated': True,
            'cls_mean': model.cls_mean,
            'cls_cov_inv': model.cls_cov_inv,
            'mahalanobis_threshold': model.mahalanobis_threshold,
        }, checkpoint_path)
        print(f"  ✅ OOD calibré et sauvé (threshold: {model.mahalanobis_threshold:.2f})")

    print("\n✅ Entraînement terminé!")


if __name__ == "__main__":
    main()
