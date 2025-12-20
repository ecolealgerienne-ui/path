#!/usr/bin/env python3
"""
Entraînement du décodeur HoVer-Net par famille d'organes.

Architecture "Y inversé" : backbone H-optimus-0 partagé, décodeurs spécialisés par famille.

IMPORTANT: Exécuter d'abord prepare_family_data.py pour préparer les données:
    python scripts/preprocessing/prepare_family_data.py --data_dir /home/amar/data/PanNuke

Usage:
    # Entraîner la famille Glandulaire (45% des données)
    python scripts/training/train_hovernet_family.py --family glandular --epochs 50 --augment

    # Entraîner la famille Digestive
    python scripts/training/train_hovernet_family.py --family digestive --epochs 50 --augment

Familles disponibles:
    - glandular: Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland (3535 samples)
    - digestive: Colon, Stomach, Esophagus, Bile-duct (2274 samples)
    - urologic: Kidney, Bladder, Testis, Ovarian, Uterus, Cervix (1153 samples)
    - epidermal: Skin, HeadNeck (574 samples)
    - respiratory: Lung, Liver (364 samples)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hovernet_decoder import HoVerNetDecoder, HoVerNetLoss
from src.models.organ_families import FAMILIES, FAMILY_TO_ORGANS, FAMILY_DESCRIPTIONS, get_organs


class FeatureAugmentation:
    """Augmentation pour features H-optimus-0 et targets."""

    def __init__(self, p_flip: float = 0.5, p_rot90: float = 0.5):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90

    def __call__(self, features, np_target, hv_target, nt_target):
        # Séparer CLS, patches, registres
        cls_token = features[0:1]
        patches = features[1:257]
        registers = features[257:261]

        # Reshape patches en grille 16x16
        patches_grid = patches.reshape(16, 16, -1)

        # Flip horizontal
        if np.random.random() < self.p_flip:
            patches_grid = np.flip(patches_grid, axis=1).copy()
            np_target = np.flip(np_target, axis=1).copy()
            hv_target = np.flip(hv_target, axis=2).copy()
            hv_target[0] = -hv_target[0]
            nt_target = np.flip(nt_target, axis=1).copy()

        # Flip vertical
        if np.random.random() < self.p_flip:
            patches_grid = np.flip(patches_grid, axis=0).copy()
            np_target = np.flip(np_target, axis=0).copy()
            hv_target = np.flip(hv_target, axis=1).copy()
            hv_target[1] = -hv_target[1]
            nt_target = np.flip(nt_target, axis=0).copy()

        # Rotation 90°
        if np.random.random() < self.p_rot90:
            k = np.random.choice([1, 2, 3])
            patches_grid = np.rot90(patches_grid, k, axes=(0, 1)).copy()
            np_target = np.rot90(np_target, k).copy()
            hv_target = np.rot90(hv_target, k, axes=(1, 2)).copy()
            nt_target = np.rot90(nt_target, k).copy()

            if k == 1:
                hv_target = np.stack([-hv_target[1], hv_target[0]])
            elif k == 2:
                hv_target = np.stack([-hv_target[0], -hv_target[1]])
            elif k == 3:
                hv_target = np.stack([hv_target[1], -hv_target[0]])

        patches = patches_grid.reshape(256, -1)
        features = np.concatenate([cls_token, patches, registers], axis=0)

        return features, np_target, hv_target, nt_target


class FamilyHoVerDataset(Dataset):
    """
    Dataset par famille d'organes avec targets pré-calculés.

    Charge tout en RAM pour un entraînement rapide.
    Les targets HV sont pré-calculés par prepare_family_data.py.
    """

    def __init__(self, family: str, cache_dir: str = None, augment: bool = False):
        self.family = family
        self.augment = augment
        self.augmenter = FeatureAugmentation() if augment else None

        # Répertoire des données pré-préparées
        if cache_dir is None:
            cache_dir = PROJECT_ROOT / "data" / "cache" / "family_data"
        else:
            cache_dir = Path(cache_dir)

        features_path = cache_dir / f"{family}_features.npz"
        targets_path = cache_dir / f"{family}_targets.npz"

        if not features_path.exists() or not targets_path.exists():
            raise FileNotFoundError(
                f"Données famille {family} non trouvées.\n"
                f"Lancez d'abord:\n"
                f"  python scripts/preprocessing/prepare_family_data.py --family {family}"
            )

        print(f"\n🏷️ Famille: {family}")
        print(f"   Organes: {', '.join(get_organs(family))}")
        print(f"   Description: {FAMILY_DESCRIPTIONS[family]}")

        # Charger tout en RAM
        print(f"\nChargement {features_path.name}...")
        features_data = np.load(features_path)
        self.features = features_data['layer_24']
        self.n_samples = len(self.features)
        print(f"  → {self.n_samples} samples, {self.features.nbytes / 1e9:.2f} GB")

        print(f"Chargement {targets_path.name}...")
        targets_data = np.load(targets_path)
        self.np_targets = targets_data['np_targets']
        # HV stocké en int8 [-127, 127] → reconvertir en float32 [-1, 1]
        hv_int8 = targets_data['hv_targets']
        self.hv_targets = hv_int8.astype(np.float32) / 127.0
        self.nt_targets = targets_data['nt_targets']
        total_targets_gb = (self.np_targets.nbytes + self.hv_targets.nbytes + self.nt_targets.nbytes) / 1e9
        print(f"  → Targets: {total_targets_gb:.2f} GB (HV reconverti int8→float32)")

        print(f"\n📊 Dataset famille {family}: {self.n_samples} samples (tout en RAM)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        features = self.features[idx].copy()
        np_target = self.np_targets[idx].copy()
        hv_target = self.hv_targets[idx].copy()
        nt_target = self.nt_targets[idx].copy()

        # Resize targets de 256 à 224
        np_target_t = torch.from_numpy(np_target)
        hv_target_t = torch.from_numpy(hv_target)
        nt_target_t = torch.from_numpy(nt_target)

        np_target_t = F.interpolate(np_target_t.unsqueeze(0).unsqueeze(0),
                                    size=(224, 224), mode='nearest').squeeze()
        hv_target_t = F.interpolate(hv_target_t.unsqueeze(0),
                                    size=(224, 224), mode='bilinear',
                                    align_corners=False).squeeze(0)
        nt_target_t = F.interpolate(nt_target_t.float().unsqueeze(0).unsqueeze(0),
                                    size=(224, 224), mode='nearest').squeeze().long()

        np_target = np_target_t.numpy()
        hv_target = hv_target_t.numpy()
        nt_target = nt_target_t.numpy()

        if self.augmenter is not None:
            features, np_target, hv_target, nt_target = self.augmenter(
                features, np_target, hv_target, nt_target
            )

        features = torch.from_numpy(features)
        np_target = torch.from_numpy(np_target.copy())
        hv_target = torch.from_numpy(hv_target.copy())
        nt_target = torch.from_numpy(nt_target.copy()).long()

        return features, np_target, hv_target, nt_target


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
    model.train()
    total_loss = 0
    losses = {'np': 0, 'hv': 0, 'nt': 0}

    pbar = tqdm(loader, desc="Train")
    for features, np_target, hv_target, nt_target in pbar:
        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)

        optimizer.zero_grad()

        np_pred, hv_pred, nt_pred = model(features)

        loss, loss_dict = criterion(
            np_pred, hv_pred, nt_pred,
            np_target, hv_target, nt_target
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k in losses:
            losses[k] += loss_dict[k]

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'np': f"{loss_dict['np']:.4f}",
        })

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    n_samples = 0

    for features, np_target, hv_target, nt_target in tqdm(loader, desc="Val"):
        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)

        np_pred, hv_pred, nt_pred = model(features)

        loss, _ = criterion(
            np_pred, hv_pred, nt_pred,
            np_target, hv_target, nt_target
        )

        total_loss += loss.item()
        total_dice += compute_dice(np_pred, np_target) * features.shape[0]
        n_samples += features.shape[0]

    return total_loss / len(loader), total_dice / n_samples


def main():
    parser = argparse.ArgumentParser(description="Entraîner HoVer-Net par famille")
    parser.add_argument('--data_dir', type=str, default='/home/amar/data/PanNuke',
                       help='Répertoire PanNuke')
    parser.add_argument('--family', type=str, required=True, choices=FAMILIES,
                       help=f'Famille à entraîner: {FAMILIES}')
    parser.add_argument('--cache_dir', type=str, default='data/cache/family_data',
                       help='Répertoire des données pré-préparées')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='models/checkpoints')
    parser.add_argument('--augment', action='store_true',
                       help='Activer data augmentation')
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT HOVERNET - FAMILLE {args.family.upper()}")
    print(f"{'='*60}")
    print(f"Description: {FAMILY_DESCRIPTIONS[args.family]}")

    # Charger le dataset (tout en RAM)
    print("\n📦 Chargement des données...")
    dataset = FamilyHoVerDataset(
        family=args.family,
        cache_dir=args.cache_dir,
        augment=args.augment
    )

    # Split train/val
    n_total = len(dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)

    # Val sans augmentation - créer dataset séparé
    val_dataset = FamilyHoVerDataset(
        family=args.family,
        cache_dir=args.cache_dir,
        augment=False
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    aug_str = " (augmenté)" if args.augment else ""
    print(f"  {n_train} train{aug_str} / {n_val} val")

    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

    # Modèle
    print("\n🔧 Initialisation du décodeur HoVer-Net...")
    model = HoVerNetDecoder(embed_dim=1536, n_classes=5, dropout=args.dropout)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {n_params:,} ({n_params/1e6:.1f}M)")

    # Loss et optimizer
    criterion = HoVerNetLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Entraînement
    print(f"\n🚀 Entraînement ({args.epochs} epochs)...")

    best_dice = 0
    best_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        train_loss, train_losses = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train - Loss: {train_loss:.4f} (NP: {train_losses['np']:.4f}, HV: {train_losses['hv']:.4f})")

        val_loss, val_dice = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        scheduler.step()

        if val_dice > best_dice:
            best_dice = val_dice
            best_loss = val_loss

            checkpoint_path = output_dir / f'hovernet_{args.family}_best.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_loss': best_loss,
                'family': args.family,
                'organs': get_organs(args.family),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  💾 Nouveau meilleur modèle sauvé (Dice: {best_dice:.4f})")

    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT TERMINÉ - FAMILLE {args.family.upper()}")
    print(f"{'='*60}")
    print(f"Meilleur Dice: {best_dice:.4f}")
    print(f"Meilleur Val Loss: {best_loss:.4f}")
    print(f"Checkpoint: {output_dir / f'hovernet_{args.family}_best.pth'}")

    if best_dice >= 0.7:
        print(f"\n✅ Dice {best_dice:.4f} >= 0.7 - Objectif POC atteint!")
    else:
        print(f"\n⚠️ Dice {best_dice:.4f} < 0.7 - Continuer l'entraînement")


if __name__ == "__main__":
    main()
