#!/usr/bin/env python3
"""
Architecture CellViT-256 pour chargement du checkpoint pré-entraîné.

Basé sur l'inspection du checkpoint:
- Encoder: ViT-Base/16, 384-dim embeddings
- Decoders: 4 niveaux de skip connections
- Heads: nuclei_binary_map, hv_map, nuclei_type_maps

Référence: TIO-IKIM/CellViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import timm


class ConvBlock2D(nn.Module):
    """Bloc convolutionnel 2D avec BatchNorm et ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Bloc décodeur CellViT avec skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv1 = ConvBlock2D(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = ConvBlock2D(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Module):
    """Tête de segmentation pour NP, HV ou NT."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            ConvBlock2D(in_channels, hidden_channels),
            ConvBlock2D(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.head(x)


class CellViT256(nn.Module):
    """
    Architecture CellViT-256 complète.

    Input: (B, 3, 256, 256)
    Outputs:
        - nuclei_binary_map: (B, 2, 256, 256) - Présence noyaux
        - hv_map: (B, 2, 256, 256) - Cartes H-V
        - nuclei_type_maps: (B, 6, 256, 256) - Types cellulaires
    """

    def __init__(
        self,
        img_size: int = 256,
        embed_dim: int = 384,
        patch_size: int = 16,
        num_classes: int = 6,  # 5 types + background
        depth: int = 12,
        num_heads: int = 6,
    ):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Encoder ViT (timm)
        self.encoder = timm.create_model(
            "vit_small_patch16_224",  # Base pour architecture similaire
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,  # Pas de classification head
        )

        # Decoders (skip connections)
        # decoder3: 384 -> 256
        self.decoder3 = self._make_decoder(embed_dim, 0, 256)
        # decoder2: 256 -> 128
        self.decoder2 = self._make_decoder(256, 0, 128)
        # decoder1: 128 -> 64
        self.decoder1 = self._make_decoder(128, 0, 64)
        # decoder0: 64 -> 32
        self.decoder0 = self._make_decoder(64, 0, 32)

        # Segmentation heads
        self.nuclei_binary_map_decoder = SegmentationHead(32, 2)
        self.hv_map_decoder = SegmentationHead(32, 2)
        self.nuclei_type_maps_decoder = SegmentationHead(32, num_classes)

    def _make_decoder(self, in_ch: int, skip_ch: int, out_ch: int) -> nn.Module:
        """Crée un bloc décodeur."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2),
            ConvBlock2D(in_ch // 2 + skip_ch, out_ch),
            ConvBlock2D(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, 3, 256, 256)

        Returns:
            Dict avec 'nuclei_binary_map', 'hv_map', 'nuclei_type_maps'
        """
        B = x.shape[0]

        # Encoder forward
        features = self.encoder.forward_features(x)  # (B, N+1, D)

        # Retirer le CLS token et reshape en spatial
        if hasattr(self.encoder, 'cls_token'):
            features = features[:, 1:, :]  # (B, N, D)

        # Reshape: (B, N, D) -> (B, D, H, W)
        h = w = int(features.shape[1] ** 0.5)
        features = features.transpose(1, 2).reshape(B, -1, h, w)  # (B, 384, 16, 16)

        # Decoder path
        d3 = self.decoder3(features)  # (B, 256, 32, 32)
        d2 = self.decoder2(d3)        # (B, 128, 64, 64)
        d1 = self.decoder1(d2)        # (B, 64, 128, 128)
        d0 = self.decoder0(d1)        # (B, 32, 256, 256)

        # Heads
        nuclei_binary_map = self.nuclei_binary_map_decoder(d0)
        hv_map = self.hv_map_decoder(d0)
        nuclei_type_maps = self.nuclei_type_maps_decoder(d0)

        return {
            'nuclei_binary_map': nuclei_binary_map,
            'hv_map': hv_map,
            'nuclei_type_maps': nuclei_type_maps,
        }


def load_cellvit256_from_checkpoint(checkpoint_path: str, device: str = 'cpu') -> CellViT256:
    """
    Charge CellViT-256 depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le fichier .pth
        device: 'cuda' ou 'cpu'

    Returns:
        Modèle CellViT256 chargé
    """
    print(f"Chargement CellViT-256 depuis {checkpoint_path}...")

    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Vérifier l'architecture
    arch = checkpoint.get('arch', 'unknown')
    print(f"Architecture: {arch}")

    # Extraire la config si disponible
    config = checkpoint.get('config', {})

    # Créer le modèle
    model = CellViT256(
        img_size=256,
        embed_dim=384,
        patch_size=16,
        num_classes=6,
    )

    # Charger les poids
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Adapter les clés si nécessaire
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Poids chargés avec succès (strict=True)")
    except RuntimeError as e:
        print(f"Chargement strict échoué: {e}")
        print("Tentative de chargement partiel...")

        # Charger ce qui correspond
        model_state = model.state_dict()
        loaded = 0
        for key in state_dict:
            if key in model_state and state_dict[key].shape == model_state[key].shape:
                model_state[key] = state_dict[key]
                loaded += 1

        model.load_state_dict(model_state)
        print(f"Chargé {loaded}/{len(state_dict)} paramètres")

    model.eval()
    model.to(device)

    print(f"Modèle sur {device}, {sum(p.numel() for p in model.parameters()):,} params")
    return model


# Test
if __name__ == "__main__":
    import sys

    # Test sans checkpoint
    print("Test architecture CellViT-256...")
    model = CellViT256()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    print(f"Input: {x.shape}")
    print(f"nuclei_binary_map: {out['nuclei_binary_map'].shape}")
    print(f"hv_map: {out['hv_map'].shape}")
    print(f"nuclei_type_maps: {out['nuclei_type_maps'].shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Test avec checkpoint si fourni
    if len(sys.argv) > 1:
        model = load_cellvit256_from_checkpoint(sys.argv[1])
        with torch.no_grad():
            out = model(x)
        print("Inference OK!")
