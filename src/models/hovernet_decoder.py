#!/usr/bin/env python3
"""
Décodeur HoVer-Net style pour H-optimus-0.

Architecture corrigée avec bottleneck partagé (tronc commun):
- Projection 1x1: 1536 → 256 (économie VRAM)
- Tronc commun partagé entre les 3 branches
- 3 branches parallèles: NP, HV, NT

Basé sur:
- HoVer-Net: https://github.com/vqdang/hover_net
- CellViT: https://github.com/TIO-IKIM/CellViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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


class DecoderHead(nn.Module):
    """
    Tête de décodage légère.

    Prend les features du tronc commun et prédit la sortie spécifique.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class HoVerNetDecoder(nn.Module):
    """
    Décodeur HoVer-Net pour H-optimus-0 avec bottleneck partagé.

    Architecture:
        H-optimus-0 features (16x16 @ 1536)
                    ↓
        Bottleneck 1x1 (1536 → 256)  ← Économie VRAM!
                    ↓
        Tronc commun (upsampling partagé)
                    ↓
           ┌────────┼────────┐
           ↓        ↓        ↓
          NP       HV       NT
        (binaire) (H+V)  (5 classes)

    Usage:
        decoder = HoVerNetDecoder()
        features = backbone.forward_features(images)  # (B, 261, 1536)
        np_out, hv_out, nt_out = decoder(features)
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        bottleneck_dim: int = 256,  # Projection 1536 → 256
        patch_size: int = 14,
        img_size: int = 224,
        n_classes: int = 5,
        dropout: float = 0.1,  # Dropout pour régularisation
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = img_size // patch_size  # 16 pour 224/14

        # ===== BOTTLENECK PARTAGÉ (économie VRAM) =====
        self.bottleneck = nn.Sequential(
            nn.Conv2d(embed_dim, bottleneck_dim, 1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        # ===== TRONC COMMUN (upsampling partagé) =====
        # 16x16 → 32 → 64 → 128 → 224
        self.up1 = UpsampleBlock(bottleneck_dim, 128)   # 16→32, 256→128
        self.up2 = UpsampleBlock(128, 64)               # 32→64, 128→64
        self.up3 = UpsampleBlock(64, 64)                # 64→128, 64→64
        self.up4 = UpsampleBlock(64, 64)                # 128→256, 64→64

        # Dropout entre les blocs d'upsampling
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # ===== TÊTES SPÉCIALISÉES (légères) =====
        self.np_head = DecoderHead(64, 2)        # Nuclei Presence (binaire)
        self.hv_head = DecoderHead(64, 2)        # Horizontal-Vertical maps
        self.nt_head = DecoderHead(64, n_classes)  # Nuclei Type (5 classes)

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

        # Bottleneck partagé (économie VRAM: 1536 → 256, + dropout)
        x = self.bottleneck(x)  # (B, 256, 16, 16)

        # Tronc commun (upsampling partagé avec dropout)
        x = self.up1(x)         # (B, 128, 32, 32)
        x = self.dropout(x)
        x = self.up2(x)         # (B, 64, 64, 64)
        x = self.dropout(x)
        x = self.up3(x)         # (B, 64, 128, 128)
        x = self.dropout(x)
        x = self.up4(x)         # (B, 64, 256, 256)

        # Ajuster à la taille cible (224x224)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                            mode='bilinear', align_corners=False)

        # Têtes spécialisées (légères, partagent les features du tronc)
        np_out = self.np_head(x)
        hv_out = self.hv_head(x)
        nt_out = self.nt_head(x)

        return np_out, hv_out, nt_out


class HoVerNetLoss(nn.Module):
    """
    Loss combinée HoVer-Net.

    - NP: BCE + Dice loss
    - HV: SmoothL1 + Gradient SmoothL1 (moins sensible aux outliers)
    - NT: CE loss
    """

    def __init__(self, lambda_np: float = 1.0, lambda_hv: float = 2.0, lambda_nt: float = 1.0):
        """
        Args:
            lambda_np: Poids branche NP (segmentation binaire)
            lambda_hv: Poids branche HV (séparation instances) - 2.0 pour focus gradients
            lambda_nt: Poids branche NT (typage cellulaire)
        """
        super().__init__()
        self.lambda_np = lambda_np
        self.lambda_hv = lambda_hv
        self.lambda_nt = lambda_nt

        self.bce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()  # Remplace MSE - moins sensible aux outliers

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """Dice loss pour segmentation."""
        pred_soft = F.softmax(pred, dim=1)[:, 1]  # Probabilité classe 1

        intersection = (pred_soft * target).sum()
        union = pred_soft.sum() + target.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """SmoothL1 sur les gradients (pour HV maps) - moins sensible aux outliers."""
        # Gradient horizontal
        pred_h = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_h = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Gradient vertical
        pred_v = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_v = target[:, :, 1:, :] - target[:, :, :-1, :]

        return self.smooth_l1(pred_h, target_h) + self.smooth_l1(pred_v, target_v)

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

        # HV loss: SmoothL1 + Gradient SmoothL1 (moins sensible aux outliers)
        hv_l1 = self.smooth_l1(hv_pred, hv_target)
        hv_grad = self.gradient_loss(hv_pred, hv_target)
        hv_loss = hv_l1 + hv_grad

        # NT loss: CE (sur tous les pixels)
        nt_loss = self.bce(nt_pred, nt_target.long())

        # Total
        total = self.lambda_np * np_loss + self.lambda_hv * hv_loss + self.lambda_nt * nt_loss

        return total, {
            'np': np_loss.item(),
            'hv': hv_loss.item(),
            'nt': nt_loss.item(),
        }


# Test
if __name__ == "__main__":
    print("Test du décodeur HoVer-Net (avec bottleneck partagé)...")

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

    # Test mémoire (simule un forward sur GPU)
    if torch.cuda.is_available():
        decoder = decoder.cuda()
        features = features.cuda()

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = decoder(features)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"✓ Pic mémoire GPU: {peak_mem:.2f} GB")

    print("\n✅ Test réussi!")
