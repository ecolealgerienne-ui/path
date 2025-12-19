#!/usr/bin/env python3
"""
Décodeur UNETR pour segmentation cellulaire.

Architecture conforme aux specs CLAUDE.md:
- Backbone: H-optimus-0 (gelé)
- Décodeur: UNETR avec skip connections
- Sorties: NP (présence), HV (séparation), NT (typage 5 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock(nn.Module):
    """Bloc convolutionnel double avec BatchNorm et ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Bloc décodeur UNETR avec upsampling et skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None):
        x = self.upsample(x)
        if skip is not None:
            # Ajuster les dimensions si nécessaire
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FeatureProjector(nn.Module):
    """Projette les features ViT vers l'espace spatial."""

    def __init__(self, embed_dim: int, out_channels: int, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, h: int, w: int):
        """
        Args:
            x: (B, N, D) features du ViT (N = h*w patches)
            h, w: dimensions en patches
        Returns:
            (B, C, H, W) features spatiales
        """
        B, N, D = x.shape
        x = self.proj(x)  # (B, N, C)
        x = x.transpose(1, 2)  # (B, C, N)
        x = x.reshape(B, -1, h, w)  # (B, C, h, w)
        return x


class UNETRDecoder(nn.Module):
    """
    Décodeur UNETR pour H-optimus-0.

    Extrait les features des couches intermédiaires du ViT
    et reconstruit les masques de segmentation.

    Architecture:
        - Skip connections depuis couches 6, 12, 18, 24
        - Décodeur progressif avec upsampling
        - 3 têtes de sortie: NP, HV, NT
    """

    def __init__(
        self,
        embed_dim: int = 1536,      # H-optimus-0 dimension
        patch_size: int = 14,        # ViT patch size
        img_size: int = 224,         # Input image size
        decoder_channels: List[int] = [512, 256, 128, 64],
        n_classes: int = 5,          # 5 types cellulaires PanNuke
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = img_size // patch_size  # 16 pour 224/14

        # Projecteurs pour chaque niveau de skip
        self.proj_24 = FeatureProjector(embed_dim, decoder_channels[0], patch_size)
        self.proj_18 = FeatureProjector(embed_dim, decoder_channels[1], patch_size)
        self.proj_12 = FeatureProjector(embed_dim, decoder_channels[2], patch_size)
        self.proj_6 = FeatureProjector(embed_dim, decoder_channels[3], patch_size)

        # Blocs décodeur
        self.decoder4 = DecoderBlock(decoder_channels[0], decoder_channels[1], decoder_channels[1])
        self.decoder3 = DecoderBlock(decoder_channels[1], decoder_channels[2], decoder_channels[2])
        self.decoder2 = DecoderBlock(decoder_channels[2], decoder_channels[3], decoder_channels[3])
        self.decoder1 = DecoderBlock(decoder_channels[3], 0, decoder_channels[3])

        # Têtes de sortie
        # NP: Nucleus Presence (binaire)
        self.head_np = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),  # 2 classes: background, nucleus
        )

        # HV: Horizontal-Vertical maps (pour séparation des noyaux)
        self.head_hv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),  # 2 canaux: H et V
        )

        # NT: Nucleus Type (5 classes PanNuke)
        self.head_nt = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1),
        )

    def forward(
        self,
        features_6: torch.Tensor,
        features_12: torch.Tensor,
        features_18: torch.Tensor,
        features_24: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features_X: (B, N, D) features de la couche X du ViT

        Returns:
            np_logits: (B, 2, H, W) logits présence noyaux
            hv_maps: (B, 2, H, W) cartes H-V
            nt_logits: (B, 5, H, W) logits types cellulaires
        """
        h = w = self.n_patches

        # Projeter les features en spatial
        z24 = self.proj_24(features_24, h, w)  # (B, 512, 16, 16)
        z18 = self.proj_18(features_18, h, w)  # (B, 256, 16, 16)
        z12 = self.proj_12(features_12, h, w)  # (B, 128, 16, 16)
        z6 = self.proj_6(features_6, h, w)     # (B, 64, 16, 16)

        # Décodage progressif avec skip connections
        d4 = self.decoder4(z24, z18)  # (B, 256, 32, 32)
        d3 = self.decoder3(d4, z12)   # (B, 128, 64, 64)
        d2 = self.decoder2(d3, z6)    # (B, 64, 128, 128)
        d1 = self.decoder1(d2)        # (B, 64, 256, 256)

        # Upscale final vers la taille d'entrée
        d1 = F.interpolate(d1, size=(self.img_size, self.img_size),
                          mode='bilinear', align_corners=False)

        # Têtes de sortie
        np_logits = self.head_np(d1)
        hv_maps = self.head_hv(d1)
        nt_logits = self.head_nt(d1)

        return np_logits, hv_maps, nt_logits


class CellSegmentationModel(nn.Module):
    """
    Modèle complet: H-optimus-0 (gelé) + UNETR Decoder.

    Usage:
        model = CellSegmentationModel()
        np_logits, hv_maps, nt_logits = model(images)
    """

    def __init__(
        self,
        backbone_name: str = "hf-hub:bioptimus/H-optimus-0",
        freeze_backbone: bool = True,
        n_classes: int = 5,
    ):
        super().__init__()

        # Charger H-optimus-0
        import timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Décodeur UNETR
        self.decoder = UNETRDecoder(
            embed_dim=1536,
            patch_size=14,
            img_size=224,
            n_classes=n_classes,
        )

        # Indices des couches à extraire (0-indexed)
        # H-optimus-0 a 40 blocs, on prend 6, 12, 18, 24
        self.layer_indices = [5, 11, 17, 23]  # 0-indexed

        # Hook pour extraire les features intermédiaires
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Enregistre des hooks pour extraire les features intermédiaires."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for idx in self.layer_indices:
            layer = self.backbone.blocks[idx]
            layer.register_forward_hook(get_hook(f'layer_{idx}'))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) images normalisées

        Returns:
            np_logits, hv_maps, nt_logits
        """
        # Forward pass backbone (extrait features via hooks)
        with torch.no_grad() if not self.training else torch.enable_grad():
            _ = self.backbone.forward_features(x)

        # Récupérer les features des couches intermédiaires
        f6 = self.features['layer_5']
        f12 = self.features['layer_11']
        f18 = self.features['layer_17']
        f24 = self.features['layer_23']

        # Décodage
        return self.decoder(f6, f12, f18, f24)


# Test rapide
if __name__ == "__main__":
    print("Test du décodeur UNETR...")

    # Test sans backbone (juste le décodeur)
    decoder = UNETRDecoder()

    # Simuler des features ViT
    B, N, D = 2, 256, 1536  # batch=2, 16x16 patches, 1536 dim
    f6 = torch.randn(B, N, D)
    f12 = torch.randn(B, N, D)
    f18 = torch.randn(B, N, D)
    f24 = torch.randn(B, N, D)

    np_out, hv_out, nt_out = decoder(f6, f12, f18, f24)

    print(f"✓ NP output: {np_out.shape}")   # (2, 2, 224, 224)
    print(f"✓ HV output: {hv_out.shape}")   # (2, 2, 224, 224)
    print(f"✓ NT output: {nt_out.shape}")   # (2, 5, 224, 224)

    # Calcul paramètres
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"✓ Paramètres décodeur: {total_params:,} ({total_params/1e6:.1f}M)")
