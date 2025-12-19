#!/usr/bin/env python3
"""
Script d'entraînement du décodeur UNETR sur PanNuke.

Conforme aux specs CLAUDE.md:
- Backbone H-optimus-0 gelé
- Entraînement du décodeur UNETR uniquement
- Sorties: NP, HV, NT

Usage:
    python scripts/training/train_unetr.py --data_dir data/raw/pannuke/Fold1
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


# Normalisation H-optimus-0 (depuis CLAUDE.md)
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


class PanNukeDataset(Dataset):
    """Dataset PanNuke pour entraînement."""

    def __init__(
        self,
        data_dir: str,
        transform=None,
        target_size: int = 224,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size

        # Charger les données
        images_path = self.data_dir / "images" / "images.npy"
        masks_path = self.data_dir / "masks" / "masks.npy"
        types_path = self.data_dir / "images" / "types.npy"

        if not images_path.exists():
            raise FileNotFoundError(f"Images non trouvées: {images_path}")

        print(f"Chargement des données depuis {self.data_dir}...")
        self.images = np.load(images_path)  # (N, 256, 256, 3)
        self.masks = np.load(masks_path)    # (N, 256, 256, 6) - 5 types + 1 instance
        self.types = np.load(types_path) if types_path.exists() else None

        print(f"  Images: {self.images.shape}")
        print(f"  Masks: {self.masks.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (256, 256, 3)
        mask = self.masks[idx]    # (256, 256, 6)

        # Redimensionner vers target_size
        import cv2
        image = cv2.resize(image, (self.target_size, self.target_size))
        mask = cv2.resize(mask, (self.target_size, self.target_size),
                         interpolation=cv2.INTER_NEAREST)

        # Normaliser l'image pour H-optimus-0
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(HOPTIMUS_MEAN)) / np.array(HOPTIMUS_STD)

        # Convertir en tenseurs
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()    # (6, H, W)

        # Créer les targets
        # NP: présence de noyau (binaire)
        np_target = (mask[:-1].sum(dim=0) > 0).long()  # (H, W)

        # NT: type de noyau (5 classes)
        # Canal 0-4: types cellulaires, Canal 5: instance ID
        nt_target = mask[:-1].argmax(dim=0)  # (H, W)
        nt_target[np_target == 0] = 0  # Background = 0

        return {
            'image': image,
            'np_target': np_target,
            'nt_target': nt_target,
            'mask': mask,
        }


class HOptimusBackbone(nn.Module):
    """Wrapper pour H-optimus-0 avec extraction de features intermédiaires."""

    def __init__(self, model_name: str = "hf-hub:bioptimus/H-optimus-0"):
        super().__init__()
        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )

        # Geler le backbone
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Couches à extraire
        self.layer_indices = [5, 11, 17, 23]
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for idx in self.layer_indices:
            self.model.blocks[idx].register_forward_hook(get_hook(f'layer_{idx}'))

    @torch.no_grad()
    def forward(self, x):
        _ = self.model.forward_features(x)
        return (
            self.features['layer_5'],
            self.features['layer_11'],
            self.features['layer_17'],
            self.features['layer_23'],
        )


def train_epoch(model, backbone, dataloader, optimizer, criterion_np, criterion_nt, device):
    """Entraîne une époque."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        np_target = batch['np_target'].to(device)
        nt_target = batch['nt_target'].to(device)

        # Extraire features du backbone
        with torch.no_grad():
            f6, f12, f18, f24 = backbone(images)

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
        n_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / n_batches


def validate(model, backbone, dataloader, criterion_np, criterion_nt, device):
    """Validation."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            np_target = batch['np_target'].to(device)
            nt_target = batch['nt_target'].to(device)

            f6, f12, f18, f24 = backbone(images)
            np_logits, hv_maps, nt_logits = model(f6, f12, f18, f24)

            loss_np = criterion_np(np_logits, np_target)
            loss_nt = criterion_nt(nt_logits, nt_target)
            loss = loss_np + loss_nt

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Entraînement UNETR sur PanNuke")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Chemin vers le dossier PanNuke Fold")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    print("\n📦 Chargement du dataset...")
    dataset = PanNukeDataset(args.data_dir, target_size=224)

    # Split train/val (80/20)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")

    # Modèles
    print("\n🔧 Initialisation des modèles...")
    backbone = HOptimusBackbone().to(device)
    decoder = UNETRDecoder(n_classes=5).to(device)

    # Optimiseur (seulement le décodeur)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Losses
    criterion_np = nn.CrossEntropyLoss()
    criterion_nt = nn.CrossEntropyLoss(ignore_index=0)  # Ignorer background

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n🚀 Démarrage de l'entraînement...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")

        train_loss = train_epoch(
            decoder, backbone, train_loader,
            optimizer, criterion_np, criterion_nt, device
        )

        val_loss = validate(
            decoder, backbone, val_loader,
            criterion_np, criterion_nt, device
        )

        scheduler.step()

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "unetr_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ Meilleur modèle sauvegardé: {checkpoint_path}")

    print("\n✅ Entraînement terminé!")
    print(f"Meilleur val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
