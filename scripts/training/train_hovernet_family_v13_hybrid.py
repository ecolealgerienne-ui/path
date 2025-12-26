#!/usr/bin/env python3
"""
Train HoVer-Net Hybrid Decoder (V13-Hybrid) for a specific family.

Architecture:
- RGB branch: H-optimus-0 features (1536-dim) → Bottleneck (256)
- H branch: H-channel CNN features (256-dim) → Bottleneck (256)
- Fusion: Additive (rgb_map + h_map)
- Decoder: Shared upsampling → NP/HV/NT branches

Loss:
- L_total = λ_np * L_np + λ_hv * L_hv + λ_nt * L_nt + λ_h_recon * L_h_recon
- Separate LR for RGB (1e-4) and H (5e-5) branches

Author: CellViT-Optimus Team
Date: 2025-12-26
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder_hybrid import HoVerNetDecoderHybrid
from src.data.preprocessing import validate_targets


class HybridDataset(Dataset):
    """
    Dataset for V13-Hybrid training.

    Loads:
    - RGB features: H-optimus-0 patch tokens (256, 1536)
    - H features: CNN features (256,)
    - Targets: NP (224, 224), HV (2, 224, 224), NT (224, 224)
    """

    def __init__(
        self,
        hybrid_data_path: Path,
        h_features_path: Path,
        rgb_features_path: Path,
        fold: int = 0,
        split: str = 'train',
        augment: bool = False
    ):
        """
        Initialize dataset.

        Args:
            hybrid_data_path: Path to hybrid data .npz (NP/HV/NT targets)
            h_features_path: Path to H-channel features .npz
            rgb_features_path: Path to RGB features .npz (fold0_features.npz)
            fold: Fold number (for split determination)
            split: 'train' or 'val'
            augment: Apply data augmentation
        """
        super().__init__()

        self.augment = augment
        self.split = split

        # Load hybrid data (targets)
        print(f"Loading hybrid data from {hybrid_data_path}...")
        hybrid_data = np.load(hybrid_data_path)

        self.np_targets = hybrid_data['np_targets']  # (N, 224, 224)
        self.hv_targets = hybrid_data['hv_targets']  # (N, 2, 224, 224)
        self.nt_targets = hybrid_data['nt_targets']  # (N, 224, 224)

        # Validate targets
        print("Validating targets...")
        validate_targets(
            self.np_targets[0],
            self.hv_targets[0],
            self.nt_targets[0],
            strict=True
        )

        # Load H-channel features
        print(f"Loading H-channel features from {h_features_path}...")
        h_data = np.load(h_features_path)
        self.h_features = h_data['h_features']  # (N, 256)

        # Load RGB features
        print(f"Loading RGB features from {rgb_features_path}...")
        rgb_data = np.load(rgb_features_path, mmap_mode='r')

        # Determine indices for train/val split
        n_total = len(self.np_targets)
        n_train = int(0.8 * n_total)

        if split == 'train':
            self.indices = list(range(0, n_train))
        else:  # val
            self.indices = list(range(n_train, n_total))

        # Extract RGB features for this split (to avoid loading all in memory)
        print(f"Extracting RGB features for {split} split ({len(self.indices)} samples)...")
        if 'features' in rgb_data:
            self.rgb_features = rgb_data['features'][self.indices].copy()  # (N, 261, 1536)
        elif 'layer_24' in rgb_data:
            self.rgb_features = rgb_data['layer_24'][self.indices].copy()
        else:
            raise ValueError(f"RGB features file missing 'features' or 'layer_24' key")

        print(f"Dataset initialized: {len(self)} samples ({split})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get item.

        Returns:
            {
                'rgb_features': (256, 1536) - Patch tokens only (indices 5-260)
                'h_features': (256,)
                'np_target': (224, 224)
                'hv_target': (2, 224, 224)
                'nt_target': (224, 224)
            }
        """
        global_idx = self.indices[idx]

        # RGB features: Extract patch tokens only (skip CLS + 4 registers)
        rgb_full = self.rgb_features[idx]  # (261, 1536)
        patch_tokens = rgb_full[5:261, :]  # (256, 1536) - Skip CLS (0) + Registers (1-4)

        # H features
        h_feats = self.h_features[global_idx]  # (256,)

        # Targets
        np_target = self.np_targets[global_idx]  # (224, 224)
        hv_target = self.hv_targets[global_idx]  # (2, 224, 224)
        nt_target = self.nt_targets[global_idx]  # (224, 224)

        # Convert to tensors
        patch_tokens = torch.from_numpy(patch_tokens).float()  # (256, 1536)
        h_feats = torch.from_numpy(h_feats).float()  # (256,)
        np_target = torch.from_numpy(np_target).long()  # (224, 224)
        hv_target = torch.from_numpy(hv_target).float()  # (2, 224, 224)
        nt_target = torch.from_numpy(nt_target).long()  # (224, 224)

        # TODO: Augmentation (flip, rotate) if self.augment
        # For now, skip augmentation to simplify initial implementation

        return {
            'rgb_features': patch_tokens,
            'h_features': h_feats,
            'np_target': np_target,
            'hv_target': hv_target,
            'nt_target': nt_target
        }


class HybridLoss(nn.Module):
    """
    Combined loss for V13-Hybrid.

    L_total = λ_np * L_np + λ_hv * L_hv + λ_nt * L_nt + λ_h_recon * L_h_recon

    Where:
    - L_np: FocalLoss for binary nuclei presence
    - L_hv: SmoothL1Loss for HV maps (masked)
    - L_nt: CrossEntropyLoss for nuclear type
    - L_h_recon: Optional reconstruction loss for H-channel
    """

    def __init__(
        self,
        lambda_np: float = 1.0,
        lambda_hv: float = 2.0,
        lambda_nt: float = 1.0,
        lambda_h_recon: float = 0.1,
        focal_alpha: float = 0.5,
        focal_gamma: float = 3.0
    ):
        super().__init__()

        self.lambda_np = lambda_np
        self.lambda_hv = lambda_hv
        self.lambda_nt = lambda_nt
        self.lambda_h_recon = lambda_h_recon

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(self, pred, target):
        """
        Focal loss for NP branch.

        Args:
            pred: (B, 2, H, W) - Logits
            target: (B, H, W) - Binary labels {0, 1}
        """
        # Convert to probabilities
        pred_softmax = F.softmax(pred, dim=1)  # (B, 2, H, W)

        # Get probability of true class
        target_one_hot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()  # (B, 2, H, W)
        pt = (pred_softmax * target_one_hot).sum(dim=1)  # (B, H, W)

        # Focal loss
        focal_weight = (1 - pt) ** self.focal_gamma
        ce_loss = F.cross_entropy(pred, target, reduction='none')  # (B, H, W)

        loss = self.focal_alpha * focal_weight * ce_loss

        return loss.mean()

    def forward(self, outputs, targets):
        """
        Compute hybrid loss.

        Args:
            outputs: HybridDecoderOutput
            targets: dict with 'np_target', 'hv_target', 'nt_target'

        Returns:
            {
                'loss': total loss,
                'loss_np': NP loss,
                'loss_hv': HV loss,
                'loss_nt': NT loss
            }
        """
        np_pred = outputs.np_out  # (B, 2, 224, 224)
        hv_pred = outputs.hv_out  # (B, 2, 224, 224)
        nt_pred = outputs.nt_out  # (B, n_classes, 224, 224)

        np_target = targets['np_target']  # (B, 224, 224)
        hv_target = targets['hv_target']  # (B, 2, 224, 224)
        nt_target = targets['nt_target']  # (B, 224, 224)

        # 1. NP Loss (Focal)
        loss_np = self.focal_loss(np_pred, np_target)

        # 2. HV Loss (SmoothL1, masked)
        # Create mask from NP target (only compute loss on nuclei pixels)
        mask = (np_target == 1).float().unsqueeze(1)  # (B, 1, 224, 224)

        hv_pred_masked = hv_pred * mask  # (B, 2, 224, 224)
        hv_target_masked = hv_target * mask  # (B, 2, 224, 224)

        n_pixels = mask.sum() + 1e-8
        loss_hv = F.smooth_l1_loss(hv_pred_masked, hv_target_masked, reduction='sum') / n_pixels

        # 3. NT Loss (CrossEntropy)
        loss_nt = F.cross_entropy(nt_pred, nt_target)

        # 4. Total Loss
        total_loss = (
            self.lambda_np * loss_np +
            self.lambda_hv * loss_hv +
            self.lambda_nt * loss_nt
        )

        return {
            'loss': total_loss,
            'loss_np': loss_np.item(),
            'loss_hv': loss_hv.item(),
            'loss_nt': loss_nt.item()
        }


def compute_metrics(outputs, targets):
    """
    Compute validation metrics.

    Returns:
        {
            'dice': NP Dice score,
            'hv_mse': HV MSE,
            'nt_acc': NT accuracy
        }
    """
    np_pred = torch.sigmoid(outputs.np_out)[:, 1, :, :]  # (B, 224, 224)
    hv_pred = outputs.hv_out  # (B, 2, 224, 224)
    nt_pred = torch.argmax(outputs.nt_out, dim=1)  # (B, 224, 224)

    np_target = targets['np_target']  # (B, 224, 224)
    hv_target = targets['hv_target']  # (B, 2, 224, 224)
    nt_target = targets['nt_target']  # (B, 224, 224)

    # 1. NP Dice
    np_pred_binary = (np_pred > 0.5).float()
    np_target_binary = (np_target == 1).float()

    intersection = (np_pred_binary * np_target_binary).sum()
    dice = (2.0 * intersection) / (np_pred_binary.sum() + np_target_binary.sum() + 1e-8)

    # 2. HV MSE (masked)
    mask = np_target_binary.unsqueeze(1)  # (B, 1, 224, 224)
    hv_pred_masked = hv_pred * mask
    hv_target_masked = hv_target * mask

    n_pixels = mask.sum() + 1e-8
    hv_mse = ((hv_pred_masked - hv_target_masked) ** 2).sum() / n_pixels

    # 3. NT Accuracy
    nt_acc = (nt_pred == nt_target).float().mean()

    return {
        'dice': dice.item(),
        'hv_mse': hv_mse.item(),
        'nt_acc': nt_acc.item()
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_np = 0.0
    total_hv = 0.0
    total_nt = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        # Move to device
        rgb_features = batch['rgb_features'].to(device)  # (B, 256, 1536)
        h_features = batch['h_features'].to(device)  # (B, 256)
        np_target = batch['np_target'].to(device)
        hv_target = batch['hv_target'].to(device)
        nt_target = batch['nt_target'].to(device)

        # Forward pass
        outputs = model(rgb_features, h_features)

        # Compute loss
        targets = {
            'np_target': np_target,
            'hv_target': hv_target,
            'nt_target': nt_target
        }

        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_np += loss_dict['loss_np']
        total_hv += loss_dict['loss_hv']
        total_nt += loss_dict['loss_nt']

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'np': f"{loss_dict['loss_np']:.4f}",
            'hv': f"{loss_dict['loss_hv']:.4f}",
            'nt': f"{loss_dict['loss_nt']:.4f}"
        })

    n_batches = len(dataloader)

    return {
        'loss': total_loss / n_batches,
        'loss_np': total_np / n_batches,
        'loss_hv': total_hv / n_batches,
        'loss_nt': total_nt / n_batches
    }


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_hv_mse = 0.0
    total_nt_acc = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for batch in pbar:
            # Move to device
            rgb_features = batch['rgb_features'].to(device)
            h_features = batch['h_features'].to(device)
            np_target = batch['np_target'].to(device)
            hv_target = batch['hv_target'].to(device)
            nt_target = batch['nt_target'].to(device)

            # Forward pass
            outputs = model(rgb_features, h_features)

            # Compute loss
            targets = {
                'np_target': np_target,
                'hv_target': hv_target,
                'nt_target': nt_target
            }

            loss_dict = criterion(outputs, targets)

            # Compute metrics
            metrics = compute_metrics(outputs, targets)

            # Accumulate
            total_loss += loss_dict['loss'].item()
            total_dice += metrics['dice']
            total_hv_mse += metrics['hv_mse']
            total_nt_acc += metrics['nt_acc']

            # Update progress bar
            pbar.set_postfix({
                'dice': f"{metrics['dice']:.4f}",
                'hv_mse': f"{metrics['hv_mse']:.4f}",
                'nt_acc': f"{metrics['nt_acc']:.4f}"
            })

    n_batches = len(dataloader)

    return {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_batches,
        'hv_mse': total_hv_mse / n_batches,
        'nt_acc': total_nt_acc / n_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Train HoVer-Net V13-Hybrid")

    # Data paths
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
                        help='Family to train')
    parser.add_argument('--hybrid_data_dir', type=Path, default=Path('data/family_data_v13_hybrid'),
                        help='Directory with hybrid data')
    parser.add_argument('--h_features_dir', type=Path, default=Path('data/cache/family_data'),
                        help='Directory with H-channel features')
    parser.add_argument('--rgb_features_dir', type=Path, default=Path('data/cache/pannuke_features'),
                        help='Directory with RGB features')

    # Training params
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr_rgb', type=float, default=1e-4,
                        help='Learning rate for RGB bottleneck')
    parser.add_argument('--lr_h', type=float, default=5e-5,
                        help='Learning rate for H bottleneck')
    parser.add_argument('--lambda_np', type=float, default=1.0,
                        help='Weight for NP loss')
    parser.add_argument('--lambda_hv', type=float, default=2.0,
                        help='Weight for HV loss')
    parser.add_argument('--lambda_nt', type=float, default=1.0,
                        help='Weight for NT loss')
    parser.add_argument('--lambda_h_recon', type=float, default=0.1,
                        help='Weight for H reconstruction loss')

    # Model params
    parser.add_argument('--n_classes', type=int, default=5,
                        help='Number of nuclear type classes')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Output
    parser.add_argument('--output_dir', type=Path, default=Path('models/checkpoints_v13_hybrid'),
                        help='Output directory for checkpoints')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"TRAINING V13-HYBRID: {args.family.upper()}")
    print(f"{'='*80}\n")

    # Paths
    hybrid_data_path = args.hybrid_data_dir / f"{args.family}_data_v13_hybrid.npz"
    h_features_path = args.h_features_dir / f"{args.family}_h_features_v13.npz"
    rgb_features_path = args.rgb_features_dir / "fold0_features.npz"

    # Verify paths exist
    for path, name in [
        (hybrid_data_path, "Hybrid data"),
        (h_features_path, "H-channel features"),
        (rgb_features_path, "RGB features")
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = HybridDataset(
        hybrid_data_path=hybrid_data_path,
        h_features_path=h_features_path,
        rgb_features_path=rgb_features_path,
        fold=0,
        split='train',
        augment=False  # TODO: Implement augmentation
    )

    val_dataset = HybridDataset(
        hybrid_data_path=hybrid_data_path,
        h_features_path=h_features_path,
        rgb_features_path=rgb_features_path,
        fold=0,
        split='val',
        augment=False
    )

    # Create dataloaders
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

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Create model
    print(f"\nInitializing model...")
    model = HoVerNetDecoderHybrid(
        embed_dim=1536,
        h_dim=256,
        n_classes=args.n_classes,
        dropout=args.dropout
    ).to(device)

    n_params = model.get_num_params(trainable_only=True)
    print(f"Model parameters: {n_params:,}")

    # Create loss
    criterion = HybridLoss(
        lambda_np=args.lambda_np,
        lambda_hv=args.lambda_hv,
        lambda_nt=args.lambda_nt,
        lambda_h_recon=args.lambda_h_recon
    )

    # Create optimizer with separate LR
    optimizer = torch.optim.AdamW([
        {'params': model.bottleneck_rgb.parameters(), 'lr': args.lr_rgb},
        {'params': model.bottleneck_h.parameters(), 'lr': args.lr_h},
        {'params': model.shared_conv1.parameters(), 'lr': args.lr_rgb},
        {'params': model.shared_conv2.parameters(), 'lr': args.lr_rgb},
        {'params': model.up1.parameters(), 'lr': args.lr_rgb},
        {'params': model.up2.parameters(), 'lr': args.lr_rgb},
        {'params': model.up3.parameters(), 'lr': args.lr_rgb},
        {'params': model.np_head.parameters(), 'lr': args.lr_rgb},
        {'params': model.hv_head.parameters(), 'lr': args.lr_rgb},
        {'params': model.nt_head.parameters(), 'lr': args.lr_rgb}
    ])

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Training loop
    best_dice = 0.0
    best_epoch = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_hv_mse': [],
        'val_nt_acc': []
    }

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val Dice:   {val_metrics['dice']:.4f}")
        print(f"  Val HV MSE: {val_metrics['hv_mse']:.4f}")
        print(f"  Val NT Acc: {val_metrics['nt_acc']:.4f}")

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_hv_mse'].append(val_metrics['hv_mse'])
        history['val_nt_acc'].append(val_metrics['nt_acc'])

        # Save best checkpoint
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch

            checkpoint_path = args.output_dir / f"hovernet_{args.family}_v13_hybrid_best.pth"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)

            print(f"  ✅ Saved best checkpoint: {checkpoint_path}")

    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Dice: {best_dice:.4f} (Epoch {best_epoch})")

    # Save history
    history_path = args.output_dir / f"hovernet_{args.family}_v13_hybrid_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"History saved: {history_path}")
    print()


if __name__ == '__main__':
    main()
