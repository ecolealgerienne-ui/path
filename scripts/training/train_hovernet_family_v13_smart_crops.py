#!/usr/bin/env python3
"""
Entraînement du décodeur HoVer-Net avec stratégie V13 Smart Crops.

Cette version utilise les données avec splits train/val pré-séparés pour prévenir
le data leakage (stratégie split-first-then-rotate validée par le CTO).

IMPORTANT: Exécuter d'abord prepare_v13_smart_crops.py et extract_features_v13_smart_crops.py:
    1. python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal
    2. python scripts/preprocessing/extract_features_v13_smart_crops.py --family epidermal --split train
    3. python scripts/preprocessing/extract_features_v13_smart_crops.py --family epidermal --split val

Usage:
    python scripts/training/train_hovernet_family_v13_smart_crops.py \
        --family epidermal \
        --epochs 30 \
        --batch_size 16

Différences avec train_hovernet_family.py:
    - Charge train/val séparés (pas de split automatique)
    - Utilise fichiers *_v13_smart_crops_*.npz
    - Data leakage prevention garanti
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
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

# Import modules du projet
from src.data.preprocessing import load_targets, resize_targets
from src.models.hovernet_decoder import HoVerNetDecoder, HoVerNetLoss
from src.models.organ_families import FAMILIES, get_organs, FAMILY_DESCRIPTIONS


class FeatureAugmentation:
    """Augmentation pour features H-optimus-0 et targets."""

    def __init__(self, p_flip: float = 0.5, p_rot90: float = 0.5):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90

    def __call__(self, features, np_target, hv_target, nt_target):
        # Structure H-optimus-0: [CLS, Registers(4), Patches(256)]
        cls_token = features[0:1]
        registers = features[1:5]
        patches = features[5:261]

        # Reshape patches en grille 16x16
        patches_grid = patches.reshape(16, 16, -1)

        # Flip horizontal
        if np.random.random() < self.p_flip:
            patches_grid = np.flip(patches_grid, axis=1).copy()
            np_target = np.flip(np_target, axis=1).copy()
            hv_target = np.flip(hv_target, axis=2).copy()
            hv_target[1] = -hv_target[1]  # Inverser composante H
            nt_target = np.flip(nt_target, axis=1).copy()

        # Rotation 90° (HV component swapping)
        if np.random.random() < self.p_rot90:
            k = np.random.choice([1, 2, 3])
            patches_grid = np.rot90(patches_grid, k, axes=(0, 1)).copy()
            np_target = np.rot90(np_target, k).copy()
            hv_target = np.rot90(hv_target, k, axes=(1, 2)).copy()
            nt_target = np.rot90(nt_target, k).copy()

            # Component swapping selon rotation
            if k == 1:  # 90° anti-horaire
                hv_target = np.stack([hv_target[1], -hv_target[0]])
            elif k == 2:  # 180°
                hv_target = np.stack([-hv_target[0], -hv_target[1]])
            elif k == 3:  # 270°
                hv_target = np.stack([-hv_target[1], hv_target[0]])

        patches = patches_grid.reshape(256, -1)
        features = np.concatenate([cls_token, registers, patches], axis=0)

        return features, np_target, hv_target, nt_target


class V13SmartCropsDataset(Dataset):
    """
    Dataset V13 Smart Crops avec split explicite (train ou val).

    Différence clé avec FamilyHoVerDataset:
    - Pas de split automatique 80/20
    - Charge fichiers train ou val explicitement
    - Data leakage prevention garanti
    """

    def __init__(self, family: str, split: str, cache_dir: str = None, augment: bool = False):
        assert split in ["train", "val"], f"Split doit être 'train' ou 'val', pas '{split}'"

        self.family = family
        self.split = split
        self.augment = augment
        self.augmenter = FeatureAugmentation() if augment else None

        # Répertoire cache
        if cache_dir is None:
            cache_dir = PROJECT_ROOT / "data/cache/family_data"
        else:
            cache_dir = Path(cache_dir)

        # Chemins fichiers V13 Smart Crops
        features_path = cache_dir / f"{family}_rgb_features_v13_smart_crops_{split}.npz"

        # Pour les targets, on utilise le fichier de prepare_v13_smart_crops.py
        targets_dir = PROJECT_ROOT / "data/family_data_v13_smart_crops"
        targets_path = targets_dir / f"{family}_{split}_v13_smart_crops.npz"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features V13 Smart Crops ({split}) non trouvées.\n"
                f"Fichier manquant: {features_path}\n\n"
                f"Lancez d'abord:\n"
                f"  python scripts/preprocessing/extract_features_v13_smart_crops.py \\\n"
                f"      --family {family} --split {split}"
            )

        if not targets_path.exists():
            raise FileNotFoundError(
                f"Targets V13 Smart Crops ({split}) non trouvées.\n"
                f"Fichier manquant: {targets_path}\n\n"
                f"Lancez d'abord:\n"
                f"  python scripts/preprocessing/prepare_v13_smart_crops.py --family {family}"
            )

        print(f"\n🏷️ Famille: {family} ({split})")
        print(f"   Organes: {', '.join(get_organs(family))}")
        print(f"   Description: {FAMILY_DESCRIPTIONS[family]}")

        # Charger features
        print(f"\nChargement {features_path.name}...")
        features_data = np.load(features_path)
        self.features = features_data['features']  # (N_crops, 261, 1536)
        self.source_image_ids = features_data['source_image_ids']
        self.crop_positions = features_data['crop_positions']

        n_crops = len(self.features)
        n_source_images = len(np.unique(self.source_image_ids))
        print(f"  → {n_crops} crops depuis {n_source_images} images sources")
        print(f"  → Amplification: {n_crops / n_source_images:.1f}×")
        print(f"  → RAM: {self.features.nbytes / 1e9:.2f} GB")

        # Charger targets
        print(f"Chargement {targets_path.name}...")
        targets_data = np.load(targets_path)
        self.np_targets = targets_data['np_targets']  # (N_crops, 224, 224)
        self.hv_targets = targets_data['hv_targets']  # (N_crops, 2, 224, 224)
        self.nt_targets = targets_data['nt_targets']  # (N_crops, 224, 224)

        # Validation HV
        print(f"\nValidation HV targets:")
        print(f"  Dtype: {self.hv_targets.dtype}")
        print(f"  Range: [{self.hv_targets.min():.3f}, {self.hv_targets.max():.3f}]")

        if self.hv_targets.dtype != np.float32:
            print(f"  ⚠️  WARNING: HV dtype devrait être float32")
        if not (-1.0 <= self.hv_targets.min() <= self.hv_targets.max() <= 1.0):
            print(f"  ⚠️  WARNING: HV range devrait être [-1, 1]")

        total_targets_gb = (
            self.np_targets.nbytes + self.hv_targets.nbytes + self.nt_targets.nbytes
        ) / 1e9
        print(f"  → Targets: {total_targets_gb:.2f} GB")

        print(f"\n📊 Dataset {family} ({split}): {n_crops} crops (tout en RAM)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx].copy()
        np_target = self.np_targets[idx].copy()
        hv_target = self.hv_targets[idx].copy()
        nt_target = self.nt_targets[idx].copy()

        # Pas de resize nécessaire (déjà à 224×224)

        if self.augmenter is not None and self.split == "train":
            features, np_target, hv_target, nt_target = self.augmenter(
                features, np_target, hv_target, nt_target
            )

        features = torch.from_numpy(features)
        np_target = torch.from_numpy(np_target.copy())
        hv_target = torch.from_numpy(hv_target.copy())
        nt_target = torch.from_numpy(nt_target.copy()).long()

        return features, np_target, hv_target, nt_target


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calcule le Dice score pour NP."""
    pred_binary = (pred.argmax(dim=1) == 1).float()
    target_float = target.float()

    intersection = (pred_binary * target_float).sum()
    union = pred_binary.sum() + target_float.sum()

    if union == 0:
        return 1.0

    return (2 * intersection / union).item()


def compute_hv_mse(hv_pred: torch.Tensor, hv_target: torch.Tensor, np_target: torch.Tensor) -> float:
    """Calcule le MSE des cartes HV sur pixels de noyaux."""
    mask = np_target.float().unsqueeze(1)

    if mask.sum() == 0:
        return 0.0

    hv_pred_masked = hv_pred * mask
    hv_target_masked = hv_target * mask

    mse = ((hv_pred_masked - hv_target_masked) ** 2).sum() / mask.sum()
    return mse.item()


def compute_nt_accuracy(nt_pred: torch.Tensor, nt_target: torch.Tensor, np_target: torch.Tensor) -> float:
    """Calcule l'accuracy de classification NT sur pixels de noyaux."""
    mask = np_target.bool()

    if mask.sum() == 0:
        return 0.0

    nt_pred_class = nt_pred.argmax(dim=1)
    correct = (nt_pred_class[mask] == nt_target[mask]).float().sum()
    total = mask.sum()

    return (correct / total).item()


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, epoch, n_classes
):
    """Train pour une epoch."""
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_hv_mse = 0.0
    total_nt_acc = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for features, np_target, hv_target, nt_target in pbar:
        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)

        # Forward
        np_out, hv_out, nt_out = model(features)

        # Loss
        loss, loss_dict = criterion(
            np_out, hv_out, nt_out,
            np_target, hv_target, nt_target
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Métriques
        dice = compute_dice(np_out, np_target)
        hv_mse = compute_hv_mse(hv_out, hv_target, np_target)
        nt_acc = compute_nt_accuracy(nt_out, nt_target, np_target)

        total_loss += loss.item()
        total_dice += dice
        total_hv_mse += hv_mse
        total_nt_acc += nt_acc
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'hv_mse': f'{hv_mse:.4f}'
        })

    return {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_batches,
        'hv_mse': total_hv_mse / n_batches,
        'nt_acc': total_nt_acc / n_batches
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, n_classes):
    """Validation."""
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_hv_mse = 0.0
    total_nt_acc = 0.0
    n_batches = 0

    for features, np_target, hv_target, nt_target in tqdm(dataloader, desc="Validation"):
        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)

        # Forward
        np_out, hv_out, nt_out = model(features)

        # Loss
        loss, loss_dict = criterion(
            np_out, hv_out, nt_out,
            np_target, hv_target, nt_target
        )

        # Métriques
        dice = compute_dice(np_out, np_target)
        hv_mse = compute_hv_mse(hv_out, hv_target, np_target)
        nt_acc = compute_nt_accuracy(nt_out, nt_target, np_target)

        total_loss += loss.item()
        total_dice += dice
        total_hv_mse += hv_mse
        total_nt_acc += nt_acc
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_batches,
        'hv_mse': total_hv_mse / n_batches,
        'nt_acc': total_nt_acc / n_batches
    }


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement HoVer-Net V13 Smart Crops"
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=FAMILIES,
        help="Famille d'organes"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_np", type=float, default=1.0)
    parser.add_argument("--lambda_hv", type=float, default=2.0)
    parser.add_argument("--lambda_nt", type=float, default=1.0)
    parser.add_argument("--lambda_magnitude", type=float, default=1.0,
                       help="Poids magnitude loss (force gradients HV)")
    parser.add_argument("--dropout", type=float, default=0.4,
                       help="Dropout rate pour régularisation")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--adaptive_loss", action="store_true",
                       help="Activer Uncertainty Weighting (poids appris)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device)
    n_classes = 5  # PanNuke: 5 classes

    print("=" * 80)
    print("ENTRAÎNEMENT HOVERNET V13 SMART CROPS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Famille: {args.family}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Lambda (NP/HV/NT/Mag): {args.lambda_np}/{args.lambda_hv}/{args.lambda_nt}/{args.lambda_magnitude}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Augmentation: {args.augment}")
    print(f"  Adaptive loss: {args.adaptive_loss}")
    print(f"  Device: {args.device}")

    # Datasets
    print("\n" + "=" * 80)
    print("CHARGEMENT DATASETS")
    print("=" * 80)

    train_dataset = V13SmartCropsDataset(
        family=args.family,
        split="train",
        augment=args.augment
    )

    val_dataset = V13SmartCropsDataset(
        family=args.family,
        split="val",
        augment=False  # Pas d'augmentation en validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Modèle
    print("\n" + "=" * 80)
    print("INITIALISATION MODÈLE")
    print("=" * 80)

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=n_classes,
        dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  → Paramètres: {n_params:,}")

    # Loss
    criterion = HoVerNetLoss(
        lambda_np=args.lambda_np,
        lambda_hv=args.lambda_hv,
        lambda_nt=args.lambda_nt,
        lambda_magnitude=args.lambda_magnitude,
        adaptive=args.adaptive_loss
    )

    # Si adaptive loss, mettre criterion sur le device (contient des paramètres apprenables)
    if args.adaptive_loss:
        criterion.to(device)

    # Optimizer (inclut les paramètres de loss si adaptive)
    if args.adaptive_loss:
        optimizer = AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=args.lr, weight_decay=1e-4
        )
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT")
    print("=" * 80)

    best_dice = 0.0
    best_combined_score = -float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_hv_mse': [],
        'val_nt_acc': []
    }

    # Checkpoints directory
    checkpoint_dir = PROJECT_ROOT / "models/checkpoints_v13_smart_crops"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, n_classes
        )

        # Validation
        val_metrics = validate(model, val_loader, criterion, device, n_classes)

        # Scheduler step
        scheduler.step()

        # Logs
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"HV MSE: {val_metrics['hv_mse']:.4f}, NT Acc: {val_metrics['nt_acc']:.4f}")

        # History
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_hv_mse'].append(val_metrics['hv_mse'])
        history['val_nt_acc'].append(val_metrics['nt_acc'])

        # Save best
        combined_score = val_metrics['dice'] - 0.5 * val_metrics['hv_mse']

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_dice = val_metrics['dice']

            checkpoint_path = checkpoint_dir / f"hovernet_{args.family}_v13_smart_crops_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'best_combined_score': best_combined_score,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)

            print(f"✅ Best model saved (Score: {combined_score:.4f})")

    # Save history
    history_path = checkpoint_dir / f"hovernet_{args.family}_v13_smart_crops_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    print(f"\nBest Dice: {best_dice:.4f}")
    print(f"Best Combined Score: {best_combined_score:.4f}")
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print("\nProchaine étape:")
    print(f"  python scripts/evaluation/test_v13_smart_crops_aji.py \\")
    print(f"      --family {args.family} --n_samples 50")

    return 0


if __name__ == "__main__":
    sys.exit(main())
