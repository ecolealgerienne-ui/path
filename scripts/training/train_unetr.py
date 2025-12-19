#!/usr/bin/env python3
"""
Script d'entraînement du décodeur UNETR sur features H-optimus-0 pré-extraites.

Conforme aux specs CLAUDE.md:
- Features H-optimus-0 pré-extraites (couches 6, 12, 18, 24)
- Entraînement du décodeur UNETR uniquement
- Sorties: NP, HV, NT

Usage:
    # Étape 1: Extraire les features
    python scripts/preprocessing/extract_features.py \
        --data_dir /home/amar/data/PanNuke --fold 0

    # Étape 2: Entraîner UNETR
    python scripts/training/train_unetr.py \
        --features_dir data/cache/pannuke_features \
        --train_fold 0 --val_fold 1
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.unetr_decoder import UNETRDecoder


class PreExtractedFeaturesDataset(Dataset):
    """Dataset avec features H-optimus-0 pré-extraites."""

    def __init__(self, features_dir: str, fold: int = 0, target_size: int = 224):
        self.features_dir = Path(features_dir)
        self.target_size = target_size

        # Charger features pré-extraites
        features_path = self.features_dir / f"fold{fold}_features.npz"
        masks_path = self.features_dir / f"fold{fold}_masks.npy"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features non trouvées: {features_path}\n"
                f"Exécutez d'abord: python scripts/preprocessing/extract_features.py"
            )

        print(f"Chargement features fold{fold}...")
        data = np.load(features_path)
        self.layer_6 = data['layer_6']    # (N, 256, 1536)
        self.layer_12 = data['layer_12']
        self.layer_18 = data['layer_18']
        self.layer_24 = data['layer_24']

        print(f"Chargement masks fold{fold}...")
        self.masks = np.load(masks_path)  # (N, 256, 256, 6)

        print(f"  → {len(self.masks)} images")
        print(f"  → Features shape: {self.layer_6.shape}")
        print(f"  → Masks shape: {self.masks.shape}")

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        # Features pré-extraites (déjà en format ViT)
        f6 = torch.from_numpy(self.layer_6[idx]).float()
        f12 = torch.from_numpy(self.layer_12[idx]).float()
        f18 = torch.from_numpy(self.layer_18[idx]).float()
        f24 = torch.from_numpy(self.layer_24[idx]).float()

        # Masks
        mask = self.masks[idx]  # (256, 256, 6)

        # Resize masks vers target_size si nécessaire
        if mask.shape[0] != self.target_size:
            import cv2
            mask = cv2.resize(mask, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_NEAREST)

        mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # (6, H, W)

        # Créer les targets
        # NP: présence de noyau (binaire) - canaux 0-4 sont les types, canal 5 est instance
        np_target = (mask[:-1].sum(dim=0) > 0).long()  # (H, W)

        # NT: type de noyau (5 classes + background)
        nt_target = mask[:-1].argmax(dim=0)  # (H, W)
        nt_target[np_target == 0] = 0  # Background = 0

        return {
            'f6': f6,
            'f12': f12,
            'f18': f18,
            'f24': f24,
            'np_target': np_target,
            'nt_target': nt_target,
        }


def train_epoch(model, dataloader, optimizer, criterion_np, criterion_nt, device):
    """Entraîne une époque."""
    model.train()
    total_loss = 0
    total_loss_np = 0
    total_loss_nt = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        f6 = batch['f6'].to(device)
        f12 = batch['f12'].to(device)
        f18 = batch['f18'].to(device)
        f24 = batch['f24'].to(device)
        np_target = batch['np_target'].to(device)
        nt_target = batch['nt_target'].to(device)

        # Forward décodeur
        np_logits, hv_maps, nt_logits = model(f6, f12, f18, f24)

        # Losses
        loss_np = criterion_np(np_logits, np_target)
        loss_nt = criterion_nt(nt_logits, nt_target)
        loss = loss_np + loss_nt

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_np += loss_np.item()
        total_loss_nt += loss_nt.item()
        n_batches += 1
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'np': f'{loss_np.item():.4f}',
            'nt': f'{loss_nt.item():.4f}'
        })

    return total_loss / n_batches, total_loss_np / n_batches, total_loss_nt / n_batches


def validate(model, dataloader, criterion_np, criterion_nt, device):
    """Validation avec calcul du Dice."""
    model.eval()
    total_loss = 0
    total_dice = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            f6 = batch['f6'].to(device)
            f12 = batch['f12'].to(device)
            f18 = batch['f18'].to(device)
            f24 = batch['f24'].to(device)
            np_target = batch['np_target'].to(device)
            nt_target = batch['nt_target'].to(device)

            np_logits, hv_maps, nt_logits = model(f6, f12, f18, f24)

            loss_np = criterion_np(np_logits, np_target)
            loss_nt = criterion_nt(nt_logits, nt_target)
            loss = loss_np + loss_nt

            # Calcul Dice pour NP
            np_pred = torch.argmax(np_logits, dim=1)
            intersection = (np_pred * np_target).sum()
            dice = (2 * intersection) / (np_pred.sum() + np_target.sum() + 1e-8)

            total_loss += loss.item()
            total_dice += dice.item()
            n_batches += 1

    return total_loss / n_batches, total_dice / n_batches


def main():
    parser = argparse.ArgumentParser(description="Entraînement UNETR sur features pré-extraites")
    parser.add_argument("--features_dir", type=str, default="data/cache/pannuke_features",
                        help="Dossier des features pré-extraites")
    parser.add_argument("--train_fold", type=int, default=0,
                        help="Fold pour entraînement")
    parser.add_argument("--val_fold", type=int, default=None,
                        help="Fold pour validation (si None, split interne 80/20)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Proportion validation si single fold (défaut: 0.2)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    print("\n📦 Chargement des datasets...")

    if args.val_fold is not None:
        # Mode 2 folds séparés
        train_dataset = PreExtractedFeaturesDataset(args.features_dir, fold=args.train_fold)
        val_dataset = PreExtractedFeaturesDataset(args.features_dir, fold=args.val_fold)
        print(f"  Train: {len(train_dataset)} images (fold{args.train_fold})")
        print(f"  Val: {len(val_dataset)} images (fold{args.val_fold})")
    else:
        # Mode single fold avec split interne
        full_dataset = PreExtractedFeaturesDataset(args.features_dir, fold=args.train_fold)
        n_val = int(len(full_dataset) * args.val_split)
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"  Fold {args.train_fold} split: {n_train} train / {n_val} val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

    # Modèle UNETR (pas de backbone, features pré-extraites)
    print("\n🔧 Initialisation du décodeur UNETR...")
    decoder = UNETRDecoder(n_classes=5).to(device)

    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"  → Paramètres entraînables: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimiseur
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Losses
    criterion_np = nn.CrossEntropyLoss()
    criterion_nt = nn.CrossEntropyLoss(ignore_index=0)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n🚀 Démarrage de l'entraînement...")
    best_dice = 0.0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})")
        print(f"{'='*60}")

        train_loss, train_np, train_nt = train_epoch(
            decoder, train_loader, optimizer, criterion_np, criterion_nt, device
        )

        val_loss, val_dice = validate(
            decoder, val_loader, criterion_np, criterion_nt, device
        )

        scheduler.step()

        print(f"\nTrain - Loss: {train_loss:.4f} (NP: {train_np:.4f}, NT: {train_nt:.4f})")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        # Sauvegarder le meilleur modèle (basé sur Dice)
        if val_dice > best_dice:
            best_dice = val_dice
            best_val_loss = val_loss
            checkpoint_path = output_dir / "unetr_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
            }, checkpoint_path)
            print(f"✓ Meilleur modèle sauvegardé: {checkpoint_path} (Dice: {val_dice:.4f})")

    # Résumé final
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print(f"Meilleur Dice: {best_dice:.4f}")
    print(f"Meilleur Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {output_dir / 'unetr_best.pth'}")

    # Critère POC
    if best_dice >= 0.7:
        print("\n✅ CRITÈRE POC VALIDÉ: Dice >= 0.7")
    else:
        print(f"\n⚠️ Dice {best_dice:.4f} < 0.7 - Continuer l'entraînement")


if __name__ == "__main__":
    main()
