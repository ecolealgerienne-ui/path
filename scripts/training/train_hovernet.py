#!/usr/bin/env python3
"""
Entraînement du décodeur HoVer-Net avec H-optimus-0.

Usage:
    # Entraîner sur fold 0 avec validation interne
    python scripts/training/train_hovernet.py --fold 0 --epochs 50

    # Entraîner sur fold 0, valider sur fold 1
    python scripts/training/train_hovernet.py --train_fold 0 --val_fold 1 --epochs 50
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hovernet_decoder import HoVerNetDecoder, HoVerNetLoss


class FeatureAugmentation:
    """
    Augmentation pour features H-optimus-0 et targets.

    Applique des transformations géométriques cohérentes sur:
    - Features: reshape 256 patch tokens en 16x16, augmenter, re-flatten
    - Targets: augmentation spatiale directe
    """

    def __init__(self, p_flip: float = 0.5, p_rot90: float = 0.5):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90

    def __call__(self, features, np_target, hv_target, nt_target):
        """
        Args:
            features: (261, 1536) - 1 CLS + 256 patches + 4 registers
            np_target: (224, 224)
            hv_target: (2, 224, 224)
            nt_target: (224, 224)
        """
        # Séparer CLS, patches, registres
        cls_token = features[0:1]        # (1, 1536)
        patches = features[1:257]        # (256, 1536)
        registers = features[257:261]    # (4, 1536)

        # Reshape patches en grille 16x16
        patches_grid = patches.reshape(16, 16, -1)  # (16, 16, 1536)

        # Flip horizontal
        if np.random.random() < self.p_flip:
            patches_grid = np.flip(patches_grid, axis=1).copy()
            np_target = np.flip(np_target, axis=1).copy()
            hv_target = np.flip(hv_target, axis=2).copy()
            hv_target[0] = -hv_target[0]  # Inverser la composante H
            nt_target = np.flip(nt_target, axis=1).copy()

        # Flip vertical
        if np.random.random() < self.p_flip:
            patches_grid = np.flip(patches_grid, axis=0).copy()
            np_target = np.flip(np_target, axis=0).copy()
            hv_target = np.flip(hv_target, axis=1).copy()
            hv_target[1] = -hv_target[1]  # Inverser la composante V
            nt_target = np.flip(nt_target, axis=0).copy()

        # Rotation 90°
        if np.random.random() < self.p_rot90:
            k = np.random.choice([1, 2, 3])  # 90, 180, 270 degrés
            patches_grid = np.rot90(patches_grid, k, axes=(0, 1)).copy()
            np_target = np.rot90(np_target, k).copy()
            hv_target = np.rot90(hv_target, k, axes=(1, 2)).copy()
            nt_target = np.rot90(nt_target, k).copy()

            # Ajuster les composantes H/V selon la rotation
            if k == 1:  # 90°
                hv_target = np.stack([-hv_target[1], hv_target[0]])
            elif k == 2:  # 180°
                hv_target = np.stack([-hv_target[0], -hv_target[1]])
            elif k == 3:  # 270°
                hv_target = np.stack([hv_target[1], -hv_target[0]])

        # Re-flatten patches
        patches = patches_grid.reshape(256, -1)

        # Reconstruire features
        features = np.concatenate([cls_token, patches, registers], axis=0)

        return features, np_target, hv_target, nt_target


class PanNukeHoVerDataset(Dataset):
    """
    Dataset PanNuke avec targets HoVer-Net.

    Charge les features H-optimus-0 pré-extraites et génère:
    - np_target: masque binaire noyaux
    - hv_target: cartes horizontal/vertical (normalisées)
    - nt_target: type de noyau par pixel
    """

    def __init__(self, data_dir: str, fold: int = 0, split: str = 'train', augment: bool = False):
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.split = split
        self.augment = augment

        # Augmentation (seulement pour train)
        self.augmenter = FeatureAugmentation() if augment else None

        # Charger les features pré-extraites (layer 24 seulement)
        features_path = PROJECT_ROOT / "data" / "cache" / "pannuke_features" / f"fold{fold}_features.npz"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features non trouvées: {features_path}\n"
                f"Lancez: python scripts/preprocessing/extract_features.py --fold {fold}"
            )

        print(f"Chargement features fold {fold}...")
        data = np.load(features_path)

        # On utilise SEULEMENT layer_24 (features finales)
        # Pas besoin des couches intermédiaires grâce au bottleneck partagé
        if 'layer_24' in data:
            self.features = data['layer_24']  # (N, 261, 1536)
        elif 'layer_23' in data:
            # Compatibilité avec l'ancien format (0-indexed)
            self.features = data['layer_23']
        else:
            raise KeyError("Features layer_24 non trouvées dans le fichier")
        print(f"  Features: {self.features.shape}")

        # Charger les masques PanNuke
        masks_path = self.data_dir / f"fold{fold}" / "masks.npy"
        if not masks_path.exists():
            raise FileNotFoundError(f"Masques non trouvés: {masks_path}")

        masks = np.load(masks_path)  # (N, 256, 256, 6) - 6 canaux
        print(f"  Masques: {masks.shape}")

        # Préparer les targets
        self.np_targets, self.hv_targets, self.nt_targets = self._prepare_targets(masks)

        self.n_samples = len(self.features)
        print(f"  Total samples: {self.n_samples}")

    def _prepare_targets(self, masks: np.ndarray):
        """
        Prépare les targets HoVer-Net depuis les masques PanNuke.

        PanNuke masks: (N, 256, 256, 6)
        - Channel 0: Background
        - Channel 1-5: Neoplastic, Inflammatory, Connective, Dead, Epithelial
        """
        N = masks.shape[0]

        # NP target: binaire (noyau ou non)
        np_targets = np.zeros((N, 256, 256), dtype=np.float32)

        # NT target: type de noyau (0-4)
        nt_targets = np.zeros((N, 256, 256), dtype=np.int64)

        # HV target: horizontal et vertical (2 canaux)
        hv_targets = np.zeros((N, 2, 256, 256), dtype=np.float32)

        for i in range(N):
            mask = masks[i]  # (256, 256, 6)

            # NP: union de tous les types de noyaux
            np_mask = mask[:, :, 1:].sum(axis=-1) > 0
            np_targets[i] = np_mask.astype(np.float32)

            # NT: argmax sur les canaux 1-5
            for c in range(5):
                type_mask = mask[:, :, c + 1] > 0
                nt_targets[i][type_mask] = c

            # HV: calculer les cartes H/V pour chaque noyau
            hv_targets[i] = self._compute_hv_maps(np_mask)

        return np_targets, hv_targets, nt_targets

    def _compute_hv_maps(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Calcule les cartes H/V depuis un masque binaire.

        Pour chaque pixel de noyau, calcule la distance normalisée
        au centre de masse du noyau.
        """
        import cv2

        hv = np.zeros((2, 256, 256), dtype=np.float32)

        if not binary_mask.any():
            return hv

        # Trouver les composantes connexes
        binary_uint8 = (binary_mask * 255).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(binary_uint8)

        for label_id in range(1, n_labels):
            instance_mask = labels == label_id
            coords = np.where(instance_mask)

            if len(coords[0]) == 0:
                continue

            # Centre de masse
            cy = coords[0].mean()
            cx = coords[1].mean()

            # Distances normalisées
            for y, x in zip(coords[0], coords[1]):
                # Distance au centre, normalisée par la taille de l'instance
                h_dist = (x - cx)  # Horizontal
                v_dist = (y - cy)  # Vertical

                # Normaliser par le rayon approximatif
                radius = max(np.sqrt(len(coords[0]) / np.pi), 1)
                hv[0, y, x] = h_dist / radius
                hv[1, y, x] = v_dist / radius

        # Clip pour éviter les valeurs extrêmes
        hv = np.clip(hv, -1, 1)

        return hv

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        features = self.features[idx].copy()  # (261, 1536)
        np_target = self.np_targets[idx].copy()  # (256, 256)
        hv_target = self.hv_targets[idx].copy()  # (2, 256, 256)
        nt_target = self.nt_targets[idx].copy()  # (256, 256)

        # Resize targets de 256 à 224 pour correspondre à la sortie
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

        # Convertir en numpy pour augmentation
        np_target = np_target_t.numpy()
        hv_target = hv_target_t.numpy()
        nt_target = nt_target_t.numpy()

        # Appliquer augmentation si activée
        if self.augmenter is not None:
            features, np_target, hv_target, nt_target = self.augmenter(
                features, np_target, hv_target, nt_target
            )

        # Convertir en tenseurs
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
            'hv': f"{loss_dict['hv']:.4f}",
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
    parser = argparse.ArgumentParser(description="Entraîner HoVer-Net decoder")
    parser.add_argument('--data_dir', type=str, default='/home/amar/data/PanNuke',
                       help='Répertoire PanNuke')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold unique (avec split interne)')
    parser.add_argument('--train_fold', type=int, default=None,
                       help='Fold entraînement (si validation croisée)')
    parser.add_argument('--val_fold', type=int, default=None,
                       help='Fold validation (si validation croisée)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='models/checkpoints')
    parser.add_argument('--augment', action='store_true',
                       help='Activer data augmentation (flip, rot90)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate dans le décodeur')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Charger le dataset
    print("\n📦 Chargement des données...")

    if args.train_fold is not None and args.val_fold is not None:
        # Validation croisée
        train_dataset = PanNukeHoVerDataset(args.data_dir, args.train_fold, 'train', augment=args.augment)
        val_dataset = PanNukeHoVerDataset(args.data_dir, args.val_fold, 'val', augment=False)
        print(f"  Train fold: {args.train_fold} ({len(train_dataset)} samples)")
        print(f"  Val fold: {args.val_fold} ({len(val_dataset)} samples)")
    else:
        # Split interne - charger 2 fois: avec et sans augmentation
        train_base = PanNukeHoVerDataset(args.data_dir, args.fold, 'train', augment=args.augment)
        val_base = PanNukeHoVerDataset(args.data_dir, args.fold, 'val', augment=False)

        n_total = len(train_base)
        n_val = int(n_total * args.val_split)
        n_train = n_total - n_val

        # Créer les indices pour le split
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = torch.utils.data.Subset(train_base, train_indices)
        val_dataset = torch.utils.data.Subset(val_base, val_indices)

        aug_str = " (augmenté)" if args.augment else ""
        print(f"  Fold {args.fold} split: {n_train} train{aug_str} / {n_val} val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

    # Modèle
    print("\n🔧 Initialisation du décodeur HoVer-Net...")
    model = HoVerNetDecoder(embed_dim=1536, n_classes=5, dropout=args.dropout)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {n_params:,} ({n_params/1e6:.1f}M)")
    if args.dropout > 0:
        print(f"  Dropout: {args.dropout}")
    if args.augment:
        print(f"  Augmentation: activée (flip + rot90)")

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

        # Train
        train_loss, train_losses = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train - Loss: {train_loss:.4f} (NP: {train_losses['np']:.4f}, HV: {train_losses['hv']:.4f}, NT: {train_losses['nt']:.4f})")

        # Validation
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        scheduler.step()

        # Sauvegarder le meilleur modèle
        if val_dice > best_dice:
            best_dice = val_dice
            best_loss = val_loss

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_loss': best_loss,
            }
            torch.save(checkpoint, output_dir / 'hovernet_best.pth')
            print(f"  💾 Nouveau meilleur modèle sauvé (Dice: {best_dice:.4f})")

    print(f"\n{'='*60}")
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*60}")
    print(f"Meilleur Dice: {best_dice:.4f}")
    print(f"Meilleur Val Loss: {best_loss:.4f}")
    print(f"Checkpoint: {output_dir / 'hovernet_best.pth'}")

    if best_dice >= 0.7:
        print(f"\n✅ Dice {best_dice:.4f} >= 0.7 - Objectif POC atteint!")
    else:
        print(f"\n⚠️ Dice {best_dice:.4f} < 0.7 - Continuer l'entraînement")


if __name__ == "__main__":
    main()
