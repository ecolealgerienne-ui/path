"""
HoVer-Net Decoder Hybrid — V13-Hybrid Production.

Dual-branch architecture combining:
- RGB features from H-optimus-0 (frozen backbone)
- H-channel features from lightweight CNN adapter

Fusion strategy: Additive fusion after bottleneck (Suggestion 4 validated).

Architecture:
    Input:
        - patch_tokens: (B, 256, 1536) RGB features from H-optimus-0
        - h_features: (B, 256) H-channel features from CNN adapter

    Bottlenecks:
        - RGB: 1536 → 256 (1x1 conv)
        - H: 256 → 256 (linear projection)

    Fusion:
        - fused = rgb_map + h_map  (B, 256, 16, 16)

    Decoder:
        - Shared upsampling (16×16 → 224×224)
        - 3 branches: NP (2 classes), HV (2 channels), NT (n_classes)

Author: CellViT-Optimus Team
Date: 2025-12-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from dataclasses import dataclass


@dataclass
class HybridDecoderOutput:
    """
    Output from Hybrid Decoder.

    Attributes:
        np_out: (B, 2, 224, 224) - Nuclear Presence logits [background, nuclei]
        hv_out: (B, 2, 224, 224) - HV maps (H, V) ∈ [-1, 1]
        nt_out: (B, n_classes, 224, 224) - Nuclear Type logits
    """
    np_out: torch.Tensor
    hv_out: torch.Tensor
    nt_out: torch.Tensor

    def to_numpy(self, apply_activations: bool = True) -> dict:
        """
        Convert to numpy with optional activations.

        Args:
            apply_activations: Apply sigmoid/softmax to outputs

        Returns:
            {
                'np': (B, 2, 224, 224) float [0, 1] if activated
                'hv': (B, 2, 224, 224) float [-1, 1]
                'nt': (B, n_classes, 224, 224) float [0, 1] if activated
            }
        """
        import numpy as np

        np_np = self.np_out.detach().cpu().numpy()
        hv_np = self.hv_out.detach().cpu().numpy()
        nt_np = self.nt_out.detach().cpu().numpy()

        if apply_activations:
            # NP: Sigmoid (2-class)
            np_np = 1 / (1 + np.exp(-np_np))
            # HV: Déjà dans [-1, 1] (pas d'activation)
            # NT: Softmax (multi-class)
            nt_exp = np.exp(nt_np - nt_np.max(axis=1, keepdims=True))
            nt_np = nt_exp / nt_exp.sum(axis=1, keepdims=True)

        return {'np': np_np, 'hv': hv_np, 'nt': nt_np}


class HoVerNetDecoderHybrid(nn.Module):
    """
    HoVer-Net Decoder with Hybrid RGB + H-channel fusion.

    Key innovation: Additive fusion after bottleneck allows gradient flow
    from both RGB (spatial context) and H (nuclear morphology) branches.
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        h_dim: int = 256,
        n_classes: int = 5,
        dropout: float = 0.1
    ):
        """
        Initialize Hybrid Decoder.

        Args:
            embed_dim: H-optimus-0 embedding dimension (1536)
            h_dim: H-channel features dimension (256)
            n_classes: Number of nuclear type classes (5 for PanNuke)
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.n_classes = n_classes

        # ========== BOTTLENECKS ==========
        # RGB branch: 1536 → 256
        self.bottleneck_rgb = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.bn_rgb = nn.BatchNorm2d(256)

        # H branch: 256 → 256 (projection)
        self.bottleneck_h = nn.Linear(h_dim, 256)
        self.bn_h = nn.BatchNorm1d(256)

        # ========== SHARED DECODER ==========
        # After fusion: 256 → 256 (refine)
        self.shared_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.shared_bn1 = nn.BatchNorm2d(256)
        self.shared_dropout1 = nn.Dropout2d(dropout)

        self.shared_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.shared_bn2 = nn.BatchNorm2d(256)
        self.shared_dropout2 = nn.Dropout2d(dropout)

        # ========== UPSAMPLING (16×16 → 224×224) ==========
        # Progression: 16 → 32 → 64 → 128 → 224
        self.up1 = self._make_upsampling_block(256, 128)  # 16 → 32
        self.up2 = self._make_upsampling_block(128, 64)   # 32 → 64
        self.up3 = self._make_upsampling_block(64, 32)    # 64 → 128
        # Final: 128 → 224 (factor 1.75, using interpolate)

        # ========== OUTPUT BRANCHES ==========
        # Nuclear Presence (2 classes: background, nuclei)
        self.np_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )

        # HV maps (2 channels: H, V)
        self.hv_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh()  # HV ∈ [-1, 1]
        )

        # Nuclear Type (n_classes)
        self.nt_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

    def _make_upsampling_block(self, in_channels: int, out_channels: int):
        """Create upsampling block (×2 spatial size)."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        h_features: torch.Tensor
    ) -> HybridDecoderOutput:
        """
        Forward pass with RGB + H fusion.

        Args:
            patch_tokens: (B, 256, 1536) - H-optimus-0 patch tokens
            h_features: (B, 256) - H-channel CNN features

        Returns:
            HybridDecoderOutput with np_out, hv_out, nt_out

        Raises:
            AssertionError: If input shapes are invalid
        """
        # ========== VALIDATION ==========
        B = patch_tokens.shape[0]
        assert patch_tokens.shape == (B, 256, self.embed_dim), \
            f"Invalid patch_tokens shape: {patch_tokens.shape}, expected (B, 256, {self.embed_dim})"
        assert h_features.shape == (B, self.h_dim), \
            f"Invalid h_features shape: {h_features.shape}, expected (B, {self.h_dim})"

        # ========== RGB BRANCH ==========
        # Reshape patches to spatial grid 16×16
        x_rgb = patch_tokens.permute(0, 2, 1)  # (B, 1536, 256)
        x_rgb = x_rgb.reshape(B, self.embed_dim, 16, 16)  # (B, 1536, 16, 16)

        # Bottleneck RGB
        rgb_map = self.bottleneck_rgb(x_rgb)  # (B, 256, 16, 16)
        rgb_map = F.relu(self.bn_rgb(rgb_map))

        # ========== H BRANCH ==========
        # Project H features
        h_emb = self.bottleneck_h(h_features)  # (B, 256)
        h_emb = F.relu(self.bn_h(h_emb))

        # Broadcast to spatial dimensions
        h_emb = h_emb.unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)
        h_map = h_emb.expand(-1, -1, 16, 16)  # (B, 256, 16, 16)

        # ========== ADDITIVE FUSION ==========
        # ✅ SUGGESTION 4: Fusion additive (permet gradient flow des 2 sources)
        fused = rgb_map + h_map  # (B, 256, 16, 16)

        # ========== SHARED DECODER ==========
        x = F.relu(self.shared_bn1(self.shared_conv1(fused)))
        x = self.shared_dropout1(x)
        x = F.relu(self.shared_bn2(self.shared_conv2(x)))
        x = self.shared_dropout2(x)  # (B, 256, 16, 16)

        # ========== UPSAMPLING ==========
        x = self.up1(x)  # (B, 128, 32, 32)
        x = self.up2(x)  # (B, 64, 64, 64)
        x = self.up3(x)  # (B, 32, 128, 128)

        # Final upsampling 128 → 224 (factor 1.75)
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)  # (B, 32, 224, 224)

        # ========== OUTPUT BRANCHES ==========
        np_out = self.np_head(x)  # (B, 2, 224, 224)
        hv_out = self.hv_head(x)  # (B, 2, 224, 224), tanh applied
        nt_out = self.nt_head(x)  # (B, n_classes, 224, 224)

        return HybridDecoderOutput(np_out=np_out, hv_out=hv_out, nt_out=nt_out)

    def get_num_params(self, trainable_only: bool = True):
        """
        Get number of parameters.

        Args:
            trainable_only: Count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
