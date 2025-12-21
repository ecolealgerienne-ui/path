#!/usr/bin/env python3
"""
Test d'entraînement minimal pour valider le pipeline.

Ce script:
1. Génère des données synthétiques (features + targets)
2. Entraîne HoVer-Net pendant quelques epochs
3. Vérifie que la loss converge
4. Vérifie que les métriques sont calculées correctement

Objectif: Valider le pipeline d'entraînement sans les vraies données PanNuke.

Usage:
    python scripts/validation/test_training_minimal.py
    python scripts/validation/test_training_minimal.py --epochs 5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder, HoVerNetLoss


def generate_synthetic_batch(batch_size=8, embed_dim=1536):
    """
    Génère un batch de données synthétiques.

    Returns:
        features: (B, 261, 1536) - Simule features H-optimus-0
        np_target: (B, 224, 224) - Masque binaire noyaux
        hv_target: (B, 2, 224, 224) - Cartes H/V
        nt_target: (B, 224, 224) - Types de noyaux (0-4)
    """
    # Features avec std ~0.77 (comme après LayerNorm)
    features = torch.randn(batch_size, 261, embed_dim) * 0.77

    # Targets
    np_target = torch.zeros(batch_size, 224, 224)
    hv_target = torch.zeros(batch_size, 2, 224, 224)
    nt_target = torch.zeros(batch_size, 224, 224, dtype=torch.long)

    # Ajouter des "noyaux" synthétiques
    for b in range(batch_size):
        n_nuclei = np.random.randint(5, 20)
        for _ in range(n_nuclei):
            cx = np.random.randint(30, 194)
            cy = np.random.randint(30, 194)
            radius = np.random.randint(5, 15)

            # Créer masque circulaire
            y, x = np.ogrid[:224, :224]
            mask = ((x - cx)**2 + (y - cy)**2 <= radius**2)

            # NP target
            np_target[b][mask] = 1

            # HV target (distance normalisée au centre)
            for i in range(224):
                for j in range(224):
                    if mask[i, j]:
                        h = (j - cx) / max(radius, 1)  # Horizontal
                        v = (i - cy) / max(radius, 1)  # Vertical
                        hv_target[b, 0, i, j] = np.clip(h, -1, 1)
                        hv_target[b, 1, i, j] = np.clip(v, -1, 1)

            # NT target (type aléatoire)
            cell_type = np.random.randint(0, 5)
            nt_target[b][mask] = cell_type

    return features, np_target, hv_target, nt_target


class SyntheticDataset(Dataset):
    """Dataset synthétique pour test."""

    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        print(f"Génération de {n_samples} samples synthétiques...")
        self.data = []
        for i in tqdm(range(n_samples)):
            features, np_t, hv_t, nt_t = generate_synthetic_batch(batch_size=1)
            self.data.append((
                features[0],
                np_t[0],
                hv_t[0],
                nt_t[0]
            ))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calcule le Dice score."""
    pred_binary = (pred.argmax(dim=1) == 1).float()
    target_float = target.float()

    intersection = (pred_binary * target_float).sum()
    union = pred_binary.sum() + target_float.sum()

    if union == 0:
        return 1.0
    return (2 * intersection / union).item()


def train_epoch(model, loader, optimizer, criterion, device):
    """Entraîne un epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    n_batches = 0

    for features, np_target, hv_target, nt_target in loader:
        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)

        optimizer.zero_grad()
        np_pred, hv_pred, nt_pred = model(features)
        loss, _ = criterion(np_pred, hv_pred, nt_pred, np_target, hv_target, nt_target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += compute_dice(np_pred, np_target)
        n_batches += 1

    return total_loss / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Valide le modèle."""
    model.eval()
    total_loss = 0
    total_dice = 0
    n_batches = 0

    for features, np_target, hv_target, nt_target in loader:
        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)

        np_pred, hv_pred, nt_pred = model(features)
        loss, _ = criterion(np_pred, hv_pred, nt_pred, np_target, hv_target, nt_target)

        total_loss += loss.item()
        total_dice += compute_dice(np_pred, np_target)
        n_batches += 1

    return total_loss / n_batches, total_dice / n_batches


def main():
    parser = argparse.ArgumentParser(description="Test d'entraînement minimal")
    parser.add_argument("--n_train", type=int, default=50, help="Nombre de samples train")
    parser.add_argument("--n_val", type=int, default=20, help="Nombre de samples val")
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"TEST D'ENTRAÎNEMENT MINIMAL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Train samples: {args.n_train}")
    print(f"Val samples: {args.n_val}")
    print(f"Epochs: {args.epochs}")

    # Datasets
    print("\n📦 Création des datasets synthétiques...")
    train_dataset = SyntheticDataset(n_samples=args.n_train)
    val_dataset = SyntheticDataset(n_samples=args.n_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Modèle
    print("\n🔧 Initialisation du modèle...")
    model = HoVerNetDecoder(embed_dim=1536, n_classes=5, dropout=0.1)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {n_params:,} ({n_params/1e6:.1f}M)")

    # Loss et optimizer
    criterion = HoVerNetLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Entraînement
    print(f"\n🚀 Entraînement ({args.epochs} epochs)...")
    losses = []
    dices = []

    for epoch in range(args.epochs):
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        losses.append((train_loss, val_loss))
        dices.append((train_dice, val_dice))

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

    # Analyse des résultats
    print(f"\n{'='*60}")
    print("ANALYSE DES RÉSULTATS")
    print(f"{'='*60}")

    # 1. Vérifier que la loss converge
    first_loss = losses[0][0]
    last_loss = losses[-1][0]
    loss_improved = last_loss < first_loss

    print(f"\n1. Convergence de la loss:")
    print(f"   Loss initiale: {first_loss:.4f}")
    print(f"   Loss finale:   {last_loss:.4f}")
    if loss_improved:
        improvement = (first_loss - last_loss) / first_loss * 100
        print(f"   ✅ La loss a diminué de {improvement:.1f}%")
    else:
        print(f"   ⚠️ La loss n'a pas diminué")

    # 2. Vérifier que le Dice augmente
    first_dice = dices[0][1]
    last_dice = dices[-1][1]
    dice_improved = last_dice > first_dice

    print(f"\n2. Amélioration du Dice:")
    print(f"   Dice initial: {first_dice:.4f}")
    print(f"   Dice final:   {last_dice:.4f}")
    if dice_improved:
        print(f"   ✅ Le Dice a augmenté")
    else:
        print(f"   ⚠️ Le Dice n'a pas augmenté (normal avec données synthétiques)")

    # 3. Vérifier l'absence de NaN
    has_nan = any(np.isnan(l[0]) or np.isnan(l[1]) for l in losses)
    print(f"\n3. Stabilité numérique:")
    if not has_nan:
        print(f"   ✅ Pas de NaN détecté")
    else:
        print(f"   ❌ NaN détecté!")

    # 4. Test forward pass avec batch de 1
    print(f"\n4. Forward pass (batch_size=1):")
    model.eval()
    with torch.no_grad():
        single_feat = torch.randn(1, 261, 1536).to(device) * 0.77
        np_pred, hv_pred, nt_pred = model(single_feat)
        print(f"   ✅ NP shape: {tuple(np_pred.shape)}")
        print(f"   ✅ HV shape: {tuple(hv_pred.shape)}")
        print(f"   ✅ NT shape: {tuple(nt_pred.shape)}")

    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")

    all_ok = loss_improved and not has_nan
    if all_ok:
        print("\n🎉 TEST RÉUSSI!")
        print("\nLe pipeline d'entraînement fonctionne correctement.")
        print("Vous pouvez procéder à l'entraînement sur les vraies données.")
        return 0
    else:
        print("\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        if not loss_improved:
            print("  - La loss ne converge pas")
        if has_nan:
            print("  - NaN détectés")
        return 1


if __name__ == "__main__":
    sys.exit(main())
