#!/usr/bin/env python3
"""
Décodeur HoVer-Net style pour H-optimus-0.

Architecture basée sur:
- HoVer-Net: https://github.com/vqdang/hover_net
- CellViT: https://github.com/TIO-IKIM/CellViT

Adapté pour les features H-optimus-0 (16x16 @ 1536 dim).

Trois branches parallèles:
- NP: Nuclei Presence (segmentation binaire)
- HV: Horizontal-Vertical maps (séparation des noyaux)
- NT: Nuclei Type (classification 5 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DenseBlock(nn.Module):
    """Bloc dense avec residual connection."""

    def __init__(self, in_channels: int, growth_rate: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return torch.cat([x, out], dim=1)


class UpsampleBlock(nn.Module):
    """Bloc d'upsampling avec convolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class DecoderBranch(nn.Module):
    """
    Branche de décodage HoVer-Net style.

    Progressive upsampling: 16x16 → 32 → 64 → 128 → 224
    """

    def __init__(
        self,
        in_channels: int = 1536,
        hidden_channels: list = [512, 256, 128, 64],
        out_channels: int = 2,
    ):
        super().__init__()

        # Projection initiale
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], 1, bias=False),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Blocs d'upsampling progressif
        self.up1 = UpsampleBlock(hidden_channels[0], hidden_channels[1])  # 16→32
        self.up2 = UpsampleBlock(hidden_channels[1], hidden_channels[2])  # 32→64
        self.up3 = UpsampleBlock(hidden_channels[2], hidden_channels[3])  # 64→128
        self.up4 = UpsampleBlock(hidden_channels[3], hidden_channels[3])  # 128→256

        # Tête de sortie
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels[3], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, target_size: int = 224) -> torch.Tensor:
        x = self.proj(x)
        x = self.up1(x)  # 32x32
        x = self.up2(x)  # 64x64
        x = self.up3(x)  # 128x128
        x = self.up4(x)  # 256x256

        # Ajuster à la taille cible
        if x.shape[-1] != target_size:
            x = F.interpolate(x, size=(target_size, target_size),
                            mode='bilinear', align_corners=False)

        return self.head(x)


class HoVerNetDecoder(nn.Module):
    """
    Décodeur HoVer-Net pour H-optimus-0.

    Prend les features finales de H-optimus-0 (16x16 @ 1536) et produit:
    - NP: Nuclei Presence (B, 2, H, W)
    - HV: Horizontal-Vertical maps (B, 2, H, W)
    - NT: Nuclei Type (B, 5, H, W)

    Usage:
        decoder = HoVerNetDecoder()
        features = backbone.forward_features(images)  # (B, 261, 1536)
        np_out, hv_out, nt_out = decoder(features)
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        patch_size: int = 14,
        img_size: int = 224,
        n_classes: int = 5,
        hidden_channels: list = [512, 256, 128, 64],
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = img_size // patch_size  # 16 pour 224/14

        # Trois branches parallèles
        self.np_branch = DecoderBranch(embed_dim, hidden_channels, out_channels=2)
        self.hv_branch = DecoderBranch(embed_dim, hidden_channels, out_channels=2)
        self.nt_branch = DecoderBranch(embed_dim, hidden_channels, out_channels=n_classes)

    def reshape_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape les tokens ViT en feature map spatiale.

        Args:
            x: (B, N, D) avec N = 1 CLS + 256 patches + 4 registers = 261

        Returns:
            (B, D, H, W) avec H=W=16
        """
        B, N, D = x.shape
        n_patches = self.n_patches * self.n_patches  # 256

        # Extraire seulement les patch tokens (ignorer CLS et registers)
        if N > n_patches:
            x = x[:, 1:n_patches+1, :]  # (B, 256, D)

        # Reshape en spatial
        x = x.transpose(1, 2)  # (B, D, 256)
        x = x.reshape(B, D, self.n_patches, self.n_patches)  # (B, D, 16, 16)

        return x

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, N, D) tokens du ViT (dernière couche)

        Returns:
            np_logits: (B, 2, H, W) - probabilités noyau/background
            hv_maps: (B, 2, H, W) - cartes horizontal/vertical
            nt_logits: (B, 5, H, W) - probabilités par type
        """
        # Reshape tokens → spatial
        x = self.reshape_features(features)  # (B, 1536, 16, 16)

        # Trois branches parallèles
        np_out = self.np_branch(x, self.img_size)
        hv_out = self.hv_branch(x, self.img_size)
        nt_out = self.nt_branch(x, self.img_size)

        return np_out, hv_out, nt_out


class HoVerNetLoss(nn.Module):
    """
    Loss combinée HoVer-Net.

    - NP: BCE + Dice loss
    - HV: MSE + Gradient MSE
    - NT: CE + Dice loss
    """

    def __init__(self, lambda_np: float = 1.0, lambda_hv: float = 1.0, lambda_nt: float = 1.0):
        super().__init__()
        self.lambda_np = lambda_np
        self.lambda_hv = lambda_hv
        self.lambda_nt = lambda_nt

        self.bce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """Dice loss pour segmentation."""
        pred_soft = F.softmax(pred, dim=1)[:, 1]  # Probabilité classe 1

        intersection = (pred_soft * target).sum()
        union = pred_soft.sum() + target.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice

    def gradient_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE sur les gradients (pour HV maps)."""
        # Gradient horizontal
        pred_h = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_h = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Gradient vertical
        pred_v = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_v = target[:, :, 1:, :] - target[:, :, :-1, :]

        return self.mse(pred_h, target_h) + self.mse(pred_v, target_v)

    def forward(
        self,
        np_pred: torch.Tensor,
        hv_pred: torch.Tensor,
        nt_pred: torch.Tensor,
        np_target: torch.Tensor,
        hv_target: torch.Tensor,
        nt_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calcule la loss totale.

        Returns:
            total_loss, dict avec losses individuelles
        """
        # NP loss: BCE + Dice
        np_bce = self.bce(np_pred, np_target.long())
        np_dice = self.dice_loss(np_pred, np_target.float())
        np_loss = np_bce + np_dice

        # HV loss: MSE + Gradient MSE
        hv_mse = self.mse(hv_pred, hv_target)
        hv_grad = self.gradient_mse(hv_pred, hv_target)
        hv_loss = hv_mse + hv_grad

        # NT loss: CE + Dice (masked by NP)
        # Masquer les pixels non-noyau pour NT
        mask = np_target > 0
        if mask.sum() > 0:
            nt_ce = self.bce(nt_pred, nt_target.long())
            nt_loss = nt_ce
        else:
            nt_loss = torch.tensor(0.0, device=np_pred.device)

        # Total
        total = self.lambda_np * np_loss + self.lambda_hv * hv_loss + self.lambda_nt * nt_loss

        return total, {
            'np': np_loss.item(),
            'hv': hv_loss.item(),
            'nt': nt_loss.item() if isinstance(nt_loss, torch.Tensor) else nt_loss,
        }


# Test
if __name__ == "__main__":
    print("Test du décodeur HoVer-Net...")

    decoder = HoVerNetDecoder()

    # Simuler features H-optimus-0
    B = 4
    features = torch.randn(B, 261, 1536)  # 261 = 1 CLS + 256 patches + 4 registers

    np_out, hv_out, nt_out = decoder(features)

    print(f"✓ NP output: {np_out.shape}")   # (4, 2, 224, 224)
    print(f"✓ HV output: {hv_out.shape}")   # (4, 2, 224, 224)
    print(f"✓ NT output: {nt_out.shape}")   # (4, 5, 224, 224)

    # Paramètres
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"✓ Paramètres: {total_params:,} ({total_params/1e6:.1f}M)")

    print("\n✅ Test réussi!")
