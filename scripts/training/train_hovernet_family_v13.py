#!/usr/bin/env python3
"""
Entraînement HoVer-Net V13 avec Multi-Crop Statique + AMP.

Différences vs V12:
- Multi-Crop: 5 crops par image source (preserve morphology)
- AMP: Mixed precision training pour économiser VRAM
- POC: 30 epochs au lieu de 60 (test rapide)

Usage:
    python scripts/training/train_hovernet_family_v13.py \
        --family epidermal \
        --epochs 30 \
        --augment \
        --amp
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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.data.preprocessing import load_targets, resize_targets
from src.models.hovernet_decoder import HoVerNetDecoder, HoVerNetLoss
from src.models.organ_families import FAMILIES, FAMILY_DESCRIPTIONS, get_organs
from src.constants import DEFAULT_FAMILY_DATA_DIR


class FeatureAugmentation:
    """Augmentation pour features H-optimus-0 et targets."""

    def __init__(self, p_flip: float = 0.5, p_rot90: float = 0.5):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90

    def __call__(self, features, np_target, hv_target, nt_target):
        # Séparer CLS, registers, patches
        cls_token = features[0:1]       # (1, 1536) - CLS
        registers = features[1:5]       # (4, 1536) - Registers (non-spatiaux)
        patches = features[5:261]       # (256, 1536) - Patches spatiaux

        # Reshape patches en grille 16x16
        patches_grid = patches.reshape(16, 16, -1)

        # Flip horizontal
        if np.random.random() < self.p_flip:
            patches_grid = np.flip(patches_grid, axis=1).copy()
            np_target = np.flip(np_target, axis=1).copy()
            hv_target = np.flip(hv_target, axis=2).copy()
            hv_target[1] = -hv_target[1]  # Inverser H (X)
            nt_target = np.flip(nt_target, axis=1).copy()

        # Flip vertical
        if np.random.random() < self.p_flip:
            patches_grid = np.flip(patches_grid, axis=0).copy()
            np_target = np.flip(np_target, axis=0).copy()
            hv_target = np.flip(hv_target, axis=1).copy()
            hv_target[0] = -hv_target[0]  # Inverser V (Y)
            nt_target = np.flip(nt_target, axis=0).copy()

        # Rotation 90°
        if np.random.random() < self.p_rot90:
            k = np.random.choice([1, 2, 3])
            patches_grid = np.rot90(patches_grid, k, axes=(0, 1)).copy()
            np_target = np.rot90(np_target, k).copy()
            hv_target = np.rot90(hv_target, k, axes=(1, 2)).copy()
            nt_target = np.rot90(nt_target, k).copy()

            if k == 1:  # 90° anti-horaire
                hv_target = np.stack([hv_target[1], -hv_target[0]])
            elif k == 2:  # 180°
                hv_target = np.stack([-hv_target[0], -hv_target[1]])
            elif k == 3:  # 270°
                hv_target = np.stack([-hv_target[1], hv_target[0]])

        patches = patches_grid.reshape(256, -1)
        features = np.concatenate([cls_token, registers, patches], axis=0)

        return features, np_target, hv_target, nt_target


class V13MultiCropDataset(Dataset):
    """
    Dataset V13 Multi-Crop avec metadata de position.

    Différence vs V12: Chaque crop est tracé (source_image_id, crop_position).
    """

    def __init__(self, family: str, cache_dir: str = None, augment: bool = False):
        self.family = family
        self.augment = augment
        self.augmenter = FeatureAugmentation() if augment else None

        # Répertoire features V13
        if cache_dir is None:
            cache_dir = PROJECT_ROOT / "data/cache/family_features_v13"
        else:
            cache_dir = Path(cache_dir)

        features_path = cache_dir / f"{family}_features_v13.npz"

        # Targets depuis V13 crops (même structure que V12)
        targets_dir = PROJECT_ROOT / "data/family_V13"
        targets_path = targets_dir / f"{family}_data_v13_crops.npz"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features V13 non trouvées: {features_path}\n"
                f"Lancez d'abord:\n"
                f"  python scripts/preprocessing/extract_features_from_v13.py \\\n"
                f"      --input_file {targets_path} \\\n"
                f"      --family {family}"
            )

        if not targets_path.exists():
            raise FileNotFoundError(
                f"Targets V13 non trouvés: {targets_path}\n"
                f"Lancez d'abord:\n"
                f"  python scripts/preprocessing/prepare_family_data_v13_multi_crop.py \\\n"
                f"      --family {family}"
            )

        print(f"\n🏷️ Famille: {family}")
        print(f"   Organes: {', '.join(get_organs(family))}")
        print(f"   Description: {FAMILY_DESCRIPTIONS[family]}")

        # Charger features
        print(f"\nChargement {features_path.name}...")
        features_data = np.load(features_path)
        self.features = features_data['features']
        self.source_ids = features_data['source_image_ids']
        self.crop_positions = features_data['crop_positions']
        self.n_samples = len(self.features)
        print(f"  → {self.n_samples} crops, {self.features.nbytes / 1e9:.2f} GB")

        # Charger targets
        print(f"Chargement {targets_path.name}...")
        targets_data = np.load(targets_path)
        self.np_targets = targets_data['np_targets']
        self.hv_targets = targets_data['hv_targets']
        self.nt_targets = targets_data['nt_targets']
        total_targets_gb = (self.np_targets.nbytes + self.hv_targets.nbytes + self.nt_targets.nbytes) / 1e9
        print(f"  → Targets: {total_targets_gb:.2f} GB")

        # Validation shapes (V13 crops sont déjà 224×224)
        assert self.np_targets.shape[1:] == (224, 224), f"NP shape: {self.np_targets.shape}"
        assert self.hv_targets.shape[1:] == (2, 224, 224), f"HV shape: {self.hv_targets.shape}"
        assert self.nt_targets.shape[1:] == (224, 224), f"NT shape: {self.nt_targets.shape}"

        # Statistiques crop positions
        print(f"\n📊 Répartition des crop positions:")
        unique_positions, counts = np.unique(self.crop_positions, return_counts=True)
        for pos, count in zip(unique_positions, counts):
            pct = 100 * count / self.n_samples
            print(f"  {pos:15s}: {count:4d} ({pct:5.1f}%)")

        print(f"\n📊 Dataset V13 Multi-Crop: {self.n_samples} crops (tout en RAM)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        features = self.features[idx].copy()
        np_target = self.np_targets[idx].copy()
        hv_target = self.hv_targets[idx].copy()
        nt_target = self.nt_targets[idx].copy()

        # Pas de resize (crops déjà 224×224)
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
    """Calcule le Dice score pour NP (nuclei presence)."""
    pred_binary = (pred.argmax(dim=1) == 1).float()
    target_float = target.float()

    intersection = (pred_binary * target_float).sum()
    union = pred_binary.sum() + target_float.sum()

    if union == 0:
        return 1.0

    return (2 * intersection / union).item()


def compute_hv_mse(hv_pred: torch.Tensor, hv_target: torch.Tensor, np_target: torch.Tensor) -> float:
    """Calcule le MSE des cartes HV uniquement sur les pixels de noyaux."""
    mask = np_target.float().unsqueeze(1)

    if mask.sum() == 0:
        return 0.0

    diff = (hv_pred - hv_target) ** 2
    masked_diff = diff * mask
    mse = masked_diff.sum() / (mask.sum() * 2)

    return mse.item()


def compute_nt_accuracy(nt_pred: torch.Tensor, nt_target: torch.Tensor, np_target: torch.Tensor) -> float:
    """Calcule l'accuracy de classification des types sur les pixels de noyaux."""
    mask = np_target > 0

    if mask.sum() == 0:
        return 1.0

    pred_class = nt_pred.argmax(dim=1)
    correct = (pred_class == nt_target) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


def train_epoch_amp(model, loader, optimizer, criterion, scaler, device, use_amp):
    """Epoch avec AMP support."""
    model.train()
    total_loss = 0
    losses = {'np': 0, 'hv': 0, 'nt': 0}
    total_dice = 0
    total_hv_mse = 0
    total_nt_acc = 0
    n_samples = 0

    pbar = tqdm(loader, desc="Train")
    for features, np_target, hv_target, nt_target in pbar:
        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)

        optimizer.zero_grad()

        # AMP forward pass
        with autocast(enabled=use_amp):
            np_pred, hv_pred, nt_pred = model(features)
            loss, loss_dict = criterion(
                np_pred, hv_pred, nt_pred,
                np_target, hv_target, nt_target
            )

        # AMP backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item()
        total_dice += compute_dice(np_pred, np_target) * batch_size
        total_hv_mse += compute_hv_mse(hv_pred, hv_target, np_target) * batch_size
        total_nt_acc += compute_nt_accuracy(nt_pred, nt_target, np_target) * batch_size
        n_samples += batch_size

        pbar.set_postfix({'loss': loss.item(), 'dice': compute_dice(np_pred, np_target)})

    metrics = {
        'dice': total_dice / n_samples,
        'hv_mse': total_hv_mse / n_samples,
        'nt_acc': total_nt_acc / n_samples,
    }
    return total_loss / len(loader), metrics


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    """Epoch de validation (sans AMP pour stabilité)."""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_hv_mse = 0
    total_nt_acc = 0
    n_samples = 0

    pbar = tqdm(loader, desc="Val")
    for features, np_target, hv_target, nt_target in pbar:
        features = features.to(device)
        np_target = np_target.to(device)
        hv_target = hv_target.to(device)
        nt_target = nt_target.to(device)

        np_pred, hv_pred, nt_pred = model(features)

        loss, loss_dict = criterion(
            np_pred, hv_pred, nt_pred,
            np_target, hv_target, nt_target
        )

        batch_size = features.shape[0]
        total_loss += loss.item()
        total_dice += compute_dice(np_pred, np_target) * batch_size
        total_hv_mse += compute_hv_mse(hv_pred, hv_target, np_target) * batch_size
        total_nt_acc += compute_nt_accuracy(nt_pred, nt_target, np_target) * batch_size
        n_samples += batch_size

    metrics = {
        'dice': total_dice / n_samples,
        'hv_mse': total_hv_mse / n_samples,
        'nt_acc': total_nt_acc / n_samples,
    }
    return total_loss / len(loader), metrics


def main():
    parser = argparse.ArgumentParser(description="Entraîner HoVer-Net V13 Multi-Crop + AMP")
    parser.add_argument('--family', type=str, required=True, choices=FAMILIES,
                       help='Famille à entraîner')
    parser.add_argument('--cache_dir', type=str, default='data/cache/family_features_v13',
                       help='Répertoire features V13')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Nombre d\'époques (POC V13: 30)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='models/checkpoints_v13')
    parser.add_argument('--augment', action='store_true',
                       help='Activer data augmentation')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout (v12 baseline: 0.4)')
    parser.add_argument('--amp', action='store_true',
                       help='Activer AMP (Automatic Mixed Precision)')

    # Loss weights (v12-Équilibré baseline)
    parser.add_argument('--lambda_np', type=float, default=1.5,
                       help='Poids NP phase 1')
    parser.add_argument('--lambda_hv', type=float, default=1.0,
                       help='Poids HV phase 2')
    parser.add_argument('--lambda_nt', type=float, default=0.5,
                       help='Poids NT phase 2')
    parser.add_argument('--lambda_magnitude', type=float, default=5.0,
                       help='Poids magnitude phase 2')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"AMP: {'✅ Activé' if args.amp else '❌ Désactivé'}")

    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT HOVERNET V13 - FAMILLE {args.family.upper()}")
    print(f"{'='*60}")
    print(f"Description: {FAMILY_DESCRIPTIONS[args.family]}")

    # Charger dataset V13
    print("\n📦 Chargement des données V13 Multi-Crop...")
    dataset = V13MultiCropDataset(
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

    # Val sans augmentation
    val_dataset = V13MultiCropDataset(
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
    model = HoVerNetDecoder(embed_dim=1536, n_classes=2, dropout=args.dropout)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {n_params:,} ({n_params/1e6:.1f}M)")

    # Loss (poids fixes pour POC)
    criterion = HoVerNetLoss(
        lambda_np=args.lambda_np,
        lambda_hv=args.lambda_hv,
        lambda_nt=args.lambda_nt,
        lambda_magnitude=args.lambda_magnitude,
        adaptive=False,
    )
    print(f"  Loss: Poids fixes (NP={args.lambda_np}, HV={args.lambda_hv}, NT={args.lambda_nt}, Mag={args.lambda_magnitude})")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP scaler
    scaler = GradScaler(enabled=args.amp)

    # Entraînement
    print(f"\n🚀 Entraînement V13 POC ({args.epochs} epochs)...")

    best_score = -float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        # Phased training (v12-Équilibré adapté pour 30 epochs)
        # Phase 1 (0-10): NP focus (λhv=0)
        # Phase 2 (11-30): Équilibré (λnp=2.0, λhv=1.0, λnt=0.5, λmag=5.0)
        if epoch < 10:
            criterion.lambda_np = 1.5
            criterion.lambda_hv = 0.0
            criterion.lambda_nt = 0.0
            criterion.lambda_magnitude = 0.0
            phase = "Phase 1 - NP Focus"
        else:
            criterion.lambda_np = 2.0
            criterion.lambda_hv = 1.0
            criterion.lambda_nt = 0.5
            criterion.lambda_magnitude = 5.0
            phase = "Phase 2 - Équilibré"

        print(f"📊 {phase}")
        print(f"   λnp={criterion.lambda_np}, λhv={criterion.lambda_hv}, λnt={criterion.lambda_nt}, λmag={criterion.lambda_magnitude}")

        # Train
        train_loss, train_metrics = train_epoch_amp(
            model, train_loader, optimizer, criterion, scaler, device, args.amp
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        # Afficher métriques
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        print(f"  Train HV MSE: {train_metrics['hv_mse']:.4f} | Val HV MSE: {val_metrics['hv_mse']:.4f}")
        print(f"  Train NT Acc: {train_metrics['nt_acc']:.4f} | Val NT Acc: {val_metrics['nt_acc']:.4f}")

        # Sauvegarder meilleur modèle (score combiné)
        combined_score = val_metrics['dice'] - 0.5 * val_metrics['hv_mse']

        if combined_score > best_score:
            best_score = combined_score
            checkpoint_path = output_dir / f"hovernet_{args.family}_v13_best.pth"

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_metrics['dice'],
                'val_hv_mse': val_metrics['hv_mse'],
                'val_nt_acc': val_metrics['nt_acc'],
                'combined_score': combined_score,
            }, checkpoint_path)

            print(f"💾 Best model saved (score={combined_score:.4f})")

    print(f"\n{'='*60}")
    print(f"✅ ENTRAÎNEMENT V13 COMPLÉTÉ")
    print(f"{'='*60}")
    print(f"Best Combined Score: {best_score:.4f}")


if __name__ == '__main__':
    main()
