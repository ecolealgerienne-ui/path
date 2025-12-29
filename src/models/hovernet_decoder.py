#!/usr/bin/env python3
"""
Décodeur HoVer-Net style pour H-optimus-0.

Architecture corrigée avec bottleneck partagé (tronc commun):
- Projection 1x1: 1536 → 256 (économie VRAM)
- Tronc commun partagé entre les 3 branches
- 3 branches parallèles: NP, HV, NT

Mode Hybrid (V13):
- Injection du canal H (Hématoxyline) via déconvolution Ruifrok
- Le canal H fournit l'info de densité chromatinienne PURE
- Concaténé aux features après bottleneck → 257 canaux

Basé sur:
- HoVer-Net: https://github.com/vqdang/hover_net
- CellViT: https://github.com/TIO-IKIM/CellViT
- Ruifrok deconvolution: Ruifrok & Johnston, Anal Quant Cytol Histol 2001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Hu et al., 2018).

    Recalibre les canaux en fonction du contexte global.

    Architecture:
        Input (B, C, H, W)
            ↓
        Global Average Pooling → (B, C, 1, 1)
            ↓
        FC1 (C → C//reduction) + ReLU
            ↓
        FC2 (C//reduction → C) + Sigmoid
            ↓
        Scale = (B, C, 1, 1) → broadcast multiply
            ↓
        Output = Input * Scale

    Pourquoi c'est crucial pour Hybrid mode:
    - Après concat: 80 canaux (64 RGB + 16 H-channel)
    - Sans SE: le modèle ignore H-channel (passager clandestin)
    - Avec SE: le modèle apprend à augmenter le poids de H sur les bordures

    Impact attendu: AJI +10% (0.54 → 0.60+)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Nombre de canaux en entrée
            reduction: Facteur de réduction pour le bottleneck FC
        """
        super().__init__()

        # S'assurer que le bottleneck a au moins 4 canaux
        reduced_channels = max(channels // reduction, 4)

        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            (B, C, H, W) recalibrated features
        """
        B, C, H, W = x.shape

        # Squeeze: Global Average Pooling
        squeezed = self.squeeze(x).view(B, C)  # (B, C)

        # Excitation: FC layers
        scale = self.excitation(squeezed)  # (B, C)

        # Scale: broadcast multiply
        scale = scale.view(B, C, 1, 1)  # (B, C, 1, 1)

        return x * scale


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


class PixelShuffleBlock(nn.Module):
    """
    Bloc d'upsampling avec PixelShuffle (Sub-Pixel Convolution).

    Avantages vs Bilinear:
    - Apprend l'upsampling (pas d'interpolation fixe)
    - Produit des bords plus nets
    - Utilisé dans les réseaux de super-résolution

    Référence: "Real-Time Single Image and Video Super-Resolution" (Shi et al. 2016)
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        # Pour PixelShuffle(r), on a besoin de r² fois plus de canaux en entrée
        mid_channels = out_channels * (scale_factor ** 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)           # (B, out*4, H, W)
        x = self.pixel_shuffle(x)   # (B, out, H*2, W*2)
        x = self.conv2(x)           # (B, out, H*2, W*2)
        return x


class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre de classes.

    Focal Loss = -α(1-p)^γ * log(p)

    - Downweight les exemples faciles (background)
    - Focus sur les exemples difficiles (bords des noyaux)

    EXPERT FIX v12-Final-Gold (2025-12-25):
    - Alpha: 0.5 (équilibré noyau/fond → modèle "ose" dessiner noyaux complets)
    - Gamma: 3.0 (focus maximal sur les bordures difficiles)

    Référence: "Focal Loss for Dense Object Detection" (Lin et al. 2017)
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 3.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = None) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) class indices
            reduction: 'mean', 'sum', or 'none'. If None, uses self.reduction.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)  # Probabilité de la bonne classe
        focal_weight = (1 - p) ** self.gamma

        # Alpha weighting pour classe 1 (noyau)
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_weight * focal_weight * ce_loss

        # Use provided reduction or fall back to instance default
        red = reduction if reduction is not None else self.reduction

        if red == 'mean':
            return focal_loss.mean()
        elif red == 'sum':
            return focal_loss.sum()
        return focal_loss


class RuifrokExtractor(nn.Module):
    """
    Extracteur de canal Hématoxyline par déconvolution couleur Ruifrok.

    Utilise la matrice de staining standard H&E pour extraire le canal H
    (Hématoxyline) qui contient l'information de densité chromatinienne.

    L'hématoxyline colore les noyaux en violet/bleu → le canal H est
    idéal pour détecter les frontières nucléaires.

    V13-Hybrid-Final (2025-12-28):
    - Ajout InstanceNorm2d pour normaliser le H-channel en N(0, 1)
    - Résout le problème de "Passager Clandestin" où le H-channel était
      ignoré car sa magnitude statistique était trop faible vs les 64 features

    Référence: Ruifrok & Johnston, Anal Quant Cytol Histol 2001

    Usage:
        extractor = RuifrokExtractor()
        h_channel = extractor(rgb_image)  # (B, 1, 224, 224) normalisé N(0, 1)
    """

    def __init__(self):
        super().__init__()
        # Matrice staining H&E standard (composante H uniquement)
        # RGB → OD → H-channel via projection sur vecteur Hématoxyline
        # [0.650, 0.704, 0.286] = vecteur Hématoxyline normalisé
        self.register_buffer(
            'stain_matrix',
            torch.tensor([0.650, 0.704, 0.286]).view(1, 3, 1, 1)
        )

        # InstanceNorm2d pour normaliser le H-channel en N(0, 1)
        # - affine=True: paramètres apprenables (γ, β) pour fine-tuning
        # - Résout le déséquilibre de magnitude: features BatchNorm'd vs H brut
        # - En pathologie, l'intensité H varie entre échantillons (même après Macenko)
        # - InstanceNorm normalise par sample → force statistique uniforme
        self.instance_norm = nn.InstanceNorm2d(1, affine=True)

    def forward(
        self,
        rgb_input: torch.Tensor,
        target_size: int = 16
    ) -> torch.Tensor:
        """
        Extrait le canal H de l'image RGB.

        Args:
            rgb_input: Image RGB (B, 3, H, W) en [0, 255] uint8-like
            target_size: Taille de sortie pour matcher la grille ViT (default: 16)

        Returns:
            H-channel (B, 1, target_size, target_size) normalisé N(0, 1)
        """
        # Clamp pour éviter log(0) et valeurs négatives
        rgb_input = rgb_input.clamp(1e-6, 255.0)

        # RGB → Optical Density (OD)
        # OD = -log10(I / I0) avec I0 = 255 (blanc de référence)
        od = -torch.log10(rgb_input / 255.0 + 1e-6)

        # Projection sur vecteur Hématoxyline
        # h_channel = od · stain_vector
        h_channel = torch.sum(od * self.stain_matrix, dim=1, keepdim=True)

        # Resize vers taille cible
        h_channel = F.interpolate(
            h_channel,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

        # V13-Hybrid-Final: Normalisation statistique pour équilibrer avec features
        # AVANT: H-channel en [0, ~2.5] (OD brut) → noyé par features en N(0, 1)
        # APRÈS: H-channel en N(0, 1) → même "force" statistique que features
        h_channel = self.instance_norm(h_channel)

        return h_channel


class HChannelPyramid(nn.Module):
    """
    FPN Chimique: Pyramide multi-échelle du canal Hématoxyline.

    Génère le H-channel à 5 résolutions pour injection à chaque niveau du décodeur:
    - Level 0: 16×16   (après bottleneck)
    - Level 1: 32×32   (après up1)
    - Level 2: 64×64   (après up2)
    - Level 3: 112×112 (après up3)
    - Level 4: 224×224 (après up4)

    POURQUOI FPN Chimique:
    - L'injection tardive (224×224 seul) = "cécité profonde"
    - Le décodeur a déjà figé ses décisions géométriques
    - Le H-channel doit guider DEPUIS LE DÉBUT la reconstruction

    Chaque niveau a sa propre projection 1→16 canaux + InstanceNorm.

    Architecture par niveau:
        H@16×16 → Conv1x1(1→16) → InstanceNorm → ReLU
        ...
        H@224×224 → Conv1x1(1→16) → InstanceNorm → ReLU

    Usage:
        pyramid = HChannelPyramid()
        h_dict = pyramid(rgb_image)
        # h_dict = {16: (B,16,16,16), 32: (B,16,32,32), ..., 224: (B,16,224,224)}
    """

    # Résolutions de la pyramide (correspondant aux niveaux du décodeur)
    SCALES = [16, 32, 64, 112, 224]

    def __init__(self, h_projection_dim: int = 16):
        super().__init__()

        self.h_projection_dim = h_projection_dim

        # Matrice staining H&E (Ruifrok)
        self.register_buffer(
            'stain_matrix',
            torch.tensor([0.650, 0.704, 0.286]).view(1, 3, 1, 1)
        )

        # Projection et normalisation séparées pour chaque échelle
        # Cela permet au modèle d'apprendre des représentations adaptées à chaque résolution
        self.projections = nn.ModuleDict()
        for scale in self.SCALES:
            self.projections[str(scale)] = nn.Sequential(
                nn.Conv2d(1, h_projection_dim, kernel_size=1, bias=False),
                nn.InstanceNorm2d(h_projection_dim, affine=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, rgb_input: torch.Tensor) -> dict:
        """
        Extrait la pyramide H-channel multi-échelle.

        Args:
            rgb_input: Image RGB (B, 3, H, W) en [0, 255]

        Returns:
            dict: {scale: h_features} où h_features = (B, h_projection_dim, scale, scale)
        """
        # Clamp pour éviter log(0)
        rgb_input = rgb_input.clamp(1e-6, 255.0)

        # RGB → Optical Density
        od = -torch.log10(rgb_input / 255.0 + 1e-6)

        # Projection sur vecteur Hématoxyline (full resolution)
        h_full = torch.sum(od * self.stain_matrix, dim=1, keepdim=True)
        # h_full: (B, 1, H, W) où H=W=224 typiquement

        # Construire la pyramide
        pyramid = {}
        for scale in self.SCALES:
            # Resize à l'échelle cible
            h_scaled = F.interpolate(
                h_full,
                size=(scale, scale),
                mode='bilinear',
                align_corners=False
            )
            # Projection 1 → 16 canaux avec normalisation
            h_projected = self.projections[str(scale)](h_scaled)
            pyramid[scale] = h_projected

        return pyramid


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

    Architecture standard:
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

    Architecture Hybrid V2 (use_hybrid=True) - HIGH RESOLUTION INJECTION:
        H-optimus-0 features     +     RGB image
              ↓                          ↓
        Bottleneck (256)         RuifrokExtractor
              ↓                          ↓
        Tronc commun              (extraction H à 224×224)
        (upsampling)                     │
              ↓                          │
        Features 224×224 (64)            │
              ↓                          ↓
              └──────── concat ──────────┘
                          ↓
                    65 canaux (64+1)
                          ↓
                    NP / HV / NT

    POURQUOI injection haute résolution (vs bottleneck 16×16):
    1. Bottleneck Dominance: 256 features noient 1 H-channel → gradient H drowns
    2. Résolution: AJI mesure précision au pixel (membrane) → 16×16 trop grossier
    3. Skip-Connection Chimique: H guide les têtes au niveau pixel, pas latent

    Le canal H (Hématoxyline) fournit l'info de densité chromatinienne
    pour améliorer la séparation d'instances (AJI cible +18%).

    Usage:
        # Standard mode
        decoder = HoVerNetDecoder()
        np_out, hv_out, nt_out = decoder(features)

        # Hybrid mode
        decoder = HoVerNetDecoder(use_hybrid=True)
        np_out, hv_out, nt_out = decoder(features, images_rgb=images)
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        bottleneck_dim: int = 256,  # Projection 1536 → 256
        patch_size: int = 14,
        img_size: int = 224,
        n_classes: int = 5,
        dropout: float = 0.1,  # Dropout pour régularisation
        use_hybrid: bool = False,  # Activer injection H-channel (obsolète, utiliser use_fpn_chimique)
        use_se_block: bool = False,  # Activer SE-Block après fusion (Hu et al., 2018)
        use_fpn_chimique: bool = False,  # Activer FPN Chimique (injection multi-échelle)
        use_h_instance_norm: bool = False,  # Appliquer InstanceNorm sur H-channel avant fusion
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = img_size // patch_size  # 16 pour 224/14
        self.use_hybrid = use_hybrid
        self.use_se_block = use_se_block
        self.use_fpn_chimique = use_fpn_chimique
        self.use_h_instance_norm = use_h_instance_norm  # Normalise H à mean=0, std=1

        # ===== BOTTLENECK PARTAGÉ (économie VRAM) =====
        self.bottleneck = nn.Sequential(
            nn.Conv2d(embed_dim, bottleneck_dim, 1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        # Nombre de canaux pour H-channel amplifié (utilisé par hybrid et fpn_chimique)
        self.h_projection_dim = 16

        # =====================================================================
        # MODE FPN CHIMIQUE: Injection multi-échelle du H-channel
        # =====================================================================
        # Résout le problème de "Cécité Profonde":
        # - L'injection tardive (224×224 seul) = le décodeur a déjà figé ses décisions
        # - FPN Chimique injecte H à CHAQUE niveau: 16, 32, 64, 112, 224
        # - Le H-channel guide la reconstruction DEPUIS LE DÉBUT
        #
        # Architecture avec FPN Chimique:
        #   Bottleneck (256) + H@16 (16) = 272 → up1 → 128
        #   128 + H@32 (16) = 144 → up2 → 64
        #   64 + H@64 (16) = 80 → up3 → 64
        #   64 + H@112 (16) = 80 → up4 → 64
        #   64 + H@224 (16) = 80 → Heads
        # =====================================================================
        if use_fpn_chimique:
            self.h_pyramid = HChannelPyramid(h_projection_dim=self.h_projection_dim)
            self.ruifrok = None  # Non utilisé en mode FPN
            self.h_projection = None

            # Canaux ajustés pour chaque niveau (+ 16 H-channel)
            up1_in_channels = bottleneck_dim + self.h_projection_dim  # 256 + 16 = 272
            up2_in_channels = 128 + self.h_projection_dim  # 128 + 16 = 144
            up3_in_channels = 64 + self.h_projection_dim   # 64 + 16 = 80
            up4_in_channels = 64 + self.h_projection_dim   # 64 + 16 = 80
            head_in_channels = 64 + self.h_projection_dim  # 64 + 16 = 80

        elif use_hybrid:
            # ===== MODE HYBRID V3 (legacy): Injection H à 224×224 seul =====
            self.h_pyramid = None
            self.ruifrok = RuifrokExtractor()
            self.h_projection = nn.Sequential(
                nn.Conv2d(1, self.h_projection_dim, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            up1_in_channels = bottleneck_dim  # 256
            up2_in_channels = 128
            up3_in_channels = 64
            up4_in_channels = 64
            head_in_channels = 64 + self.h_projection_dim  # 80

        else:
            # ===== MODE STANDARD: Pas d'injection H =====
            self.h_pyramid = None
            self.ruifrok = None
            self.h_projection = None
            up1_in_channels = bottleneck_dim  # 256
            up2_in_channels = 128
            up3_in_channels = 64
            up4_in_channels = 64
            head_in_channels = 64

        # ===== TRONC COMMUN (upsampling partagé) =====
        # Avec FPN Chimique: les input channels sont ajustés pour inclure H-channel
        # 16x16 → 32 → 64 → 128 → 224
        self.up1 = PixelShuffleBlock(up1_in_channels, 128)  # (256 ou 272) → 128
        self.up2 = PixelShuffleBlock(up2_in_channels, 64)   # (128 ou 144) → 64
        self.up3 = PixelShuffleBlock(up3_in_channels, 64)   # (64 ou 80) → 64
        self.up4 = PixelShuffleBlock(up4_in_channels, 64)   # (64 ou 80) → 64

        # Dropout entre les blocs d'upsampling
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # ===== SE-BLOCK: Attention sur canaux après fusion finale (Hu et al., 2018) =====
        # Avec FPN Chimique: recalibre les 80 canaux finaux (64 features + 16 H@224)
        if use_se_block:
            self.se_block = SEBlock(head_in_channels, reduction=16)
        else:
            self.se_block = None

        # ===== TÊTES SPÉCIALISÉES (légères) =====
        self.np_head = DecoderHead(head_in_channels, 2)        # Nuclei Presence (binaire)
        self.hv_head = nn.Sequential(
            DecoderHead(head_in_channels, 2),
            nn.Tanh()  # OBLIGATOIRE: forcer HV dans [-1, 1] pour matcher targets
        )
        self.nt_head = DecoderHead(head_in_channels, n_classes)  # Nuclei Type (5 classes)

    def reshape_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape les tokens ViT en feature map spatiale.

        Args:
            x: (B, N, D) avec N = 1 CLS + 4 registers + 256 patches = 261

        Returns:
            (B, D, H, W) avec H=W=16

        IMPORTANT - Structure H-optimus-0 (ViT-Giant/14 avec registres):
            Index 0:     CLS token (classification globale)
            Index 1-4:   Register tokens (mémoire, SANS info spatiale!)
            Index 5-260: 256 patch tokens (grille 16×16 spatiale)

        BUG CORRIGÉ (2025-12-25):
            AVANT: x[:, 1:257, :] → Prenait Registers(1-4) + Patches(5-256)
                   → Registers n'ont PAS d'info spatiale → bruit dans la grille
                   → Manquait les 4 derniers patches (257-260)
            APRÈS: x[:, 5:, :] → Prend uniquement Patches(5-260)
                   → 256 tokens spatiaux corrects
        """
        B, N, D = x.shape
        n_patches = self.n_patches * self.n_patches  # 256

        # Extraire UNIQUEMENT les patch tokens (ignorer CLS et registers)
        if N == 261:
            # H-optimus-0: [CLS(0), Registers(1-4), Patches(5-260)]
            x = x[:, 5:, :]  # (B, 256, D) - CORRECTION: sauter les registers
        elif N > n_patches:
            # Fallback pour autres modèles (ViT standard sans registers)
            x = x[:, 1:n_patches+1, :]  # (B, 256, D)

        # Reshape en spatial
        x = x.transpose(1, 2)  # (B, D, 256)
        x = x.reshape(B, D, self.n_patches, self.n_patches)  # (B, D, 16, 16)

        return x

    def forward(
        self,
        features: torch.Tensor,
        images_rgb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, N, D) tokens du ViT (dernière couche)
            images_rgb: (B, 3, H, W) images RGB en [0, 255] (requis si use_hybrid ou use_fpn_chimique)

        Returns:
            np_logits: (B, 2, H, W) - probabilités noyau/background
            hv_maps: (B, 2, H, W) - cartes horizontal/vertical
            nt_logits: (B, 5, H, W) - probabilités par type
        """
        # =====================================================================
        # FPN CHIMIQUE: Injection multi-échelle du H-channel
        # =====================================================================
        # Le H-channel est injecté à CHAQUE niveau du décodeur:
        #   16×16 → 32×32 → 64×64 → 112×112 → 224×224
        #
        # Résout le problème de "Cécité Profonde":
        # - L'injection tardive (224 seul) = décisions déjà figées
        # - FPN Chimique guide la reconstruction DEPUIS LE DÉBUT
        #
        # GAIN ATTENDU: AJI 0.54 → 0.68 (+26%) via guidage continu
        # =====================================================================

        # Reshape tokens → spatial
        x = self.reshape_features(features)  # (B, 1536, 16, 16)

        # Bottleneck partagé (économie VRAM: 1536 → 256)
        x = self.bottleneck(x)  # (B, 256, 16, 16)

        # ===== MODE FPN CHIMIQUE: Injection à chaque niveau =====
        if self.use_fpn_chimique:
            if images_rgb is None:
                raise ValueError(
                    "images_rgb est requis quand use_fpn_chimique=True. "
                    "Passez les images RGB (B, 3, H, W) au forward()."
                )

            # Extraire la pyramide H-channel (5 niveaux)
            h_pyramid = self.h_pyramid(images_rgb)
            # h_pyramid = {16: (B,16,16,16), 32: (B,16,32,32), 64: (B,16,64,64),
            #              112: (B,16,112,112), 224: (B,16,224,224)}

            # Optionnel: Normaliser H-channel à mean=0, std=1 (rééquilibre vs features)
            if self.use_h_instance_norm:
                h_pyramid = {k: F.instance_norm(v) for k, v in h_pyramid.items()}

            # Niveau 0: Après bottleneck (16×16)
            x = torch.cat([x, h_pyramid[16]], dim=1)  # (B, 272, 16, 16)

            # Niveau 1: up1 (16→32) puis injection H@32
            x = self.up1(x)  # (B, 128, 32, 32)
            x = self.dropout(x)
            x = torch.cat([x, h_pyramid[32]], dim=1)  # (B, 144, 32, 32)

            # Niveau 2: up2 (32→64) puis injection H@64
            x = self.up2(x)  # (B, 64, 64, 64)
            x = self.dropout(x)
            x = torch.cat([x, h_pyramid[64]], dim=1)  # (B, 80, 64, 64)

            # Niveau 3: up3 (64→128) puis interpolation à 112 et injection H@112
            x = self.up3(x)  # (B, 64, 128, 128)
            x = self.dropout(x)
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
            x = torch.cat([x, h_pyramid[112]], dim=1)  # (B, 80, 112, 112)

            # Niveau 4: up4 (112→224) puis injection H@224
            x = self.up4(x)  # (B, 64, 224, 224)
            x = torch.cat([x, h_pyramid[224]], dim=1)  # (B, 80, 224, 224)

        # ===== MODE HYBRID V3 (legacy): Injection H à 224×224 seul =====
        elif self.use_hybrid:
            if images_rgb is None:
                raise ValueError(
                    "images_rgb est requis quand use_hybrid=True. "
                    "Passez les images RGB (B, 3, H, W) au forward()."
                )

            # Tronc commun standard (sans injection intermédiaire)
            x = self.up1(x)         # (B, 128, 32, 32)
            x = self.dropout(x)
            x = self.up2(x)         # (B, 64, 64, 64)
            x = self.dropout(x)
            x = self.up3(x)         # (B, 64, 128, 128)
            x = self.dropout(x)
            x = self.up4(x)         # (B, 64, 256, 256)

            # Ajuster à 224×224
            if x.shape[-1] != self.img_size:
                x = F.interpolate(x, size=(self.img_size, self.img_size),
                                mode='bilinear', align_corners=False)

            # Injection H à haute résolution (224×224)
            h_channel = self.ruifrok(images_rgb, target_size=self.img_size)
            h_channel = self.h_projection(h_channel)
            x = torch.cat([x, h_channel], dim=1)  # (B, 80, 224, 224)

        # ===== MODE STANDARD: Pas d'injection H =====
        else:
            x = self.up1(x)         # (B, 128, 32, 32)
            x = self.dropout(x)
            x = self.up2(x)         # (B, 64, 64, 64)
            x = self.dropout(x)
            x = self.up3(x)         # (B, 64, 128, 128)
            x = self.dropout(x)
            x = self.up4(x)         # (B, 64, 256, 256)

            # Ajuster à 224×224
            if x.shape[-1] != self.img_size:
                x = F.interpolate(x, size=(self.img_size, self.img_size),
                                mode='bilinear', align_corners=False)

        # ===== SE-BLOCK: Recalibration finale des canaux (Hu et al., 2018) =====
        # Avec FPN Chimique: recalibre les 80 canaux finaux (64 features + 16 H@224)
        if self.se_block is not None:
            x = self.se_block(x)

        # Têtes spécialisées
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

    Supporte deux modes:
    1. Poids fixes (lambda_np, lambda_hv, lambda_nt)
    2. Uncertainty Weighting (adaptive=True) - le modèle apprend les poids optimaux

    Référence Uncertainty Weighting:
    "Multi-Task Learning Using Uncertainty to Weigh Losses" - Kendall et al. 2018
    """

    def __init__(
        self,
        lambda_np: float = 1.0,
        lambda_hv: float = 5.0,
        lambda_nt: float = 1.0,
        lambda_magnitude: float = 5.0,
        lambda_edge: float = 0.5,
        adaptive: bool = False,
    ):
        """
        Args:
            lambda_np: Poids branche NP (segmentation binaire)
            lambda_hv: Poids branche HV (séparation instances) - 5.0 pour V13-Hybrid V2
            lambda_nt: Poids branche NT (typage cellulaire)
            lambda_magnitude: Poids magnitude loss (Expert: 5.0 pour forcer gradients forts)
            lambda_edge: Poids Edge-Aware loss (Expert 2025-12-28: force compréhension contours H)
            adaptive: Si True, utilise Uncertainty Weighting (poids appris)
        """
        super().__init__()
        self.lambda_np = lambda_np
        self.lambda_hv = lambda_hv
        self.lambda_nt = lambda_nt
        self.lambda_magnitude = lambda_magnitude
        self.lambda_edge = lambda_edge
        self.adaptive = adaptive

        # NP Loss: Focal + Dice (Phase 1 Phased Training)
        # Expert: Focal Loss ignore le déséquilibre fond/noyau
        # et se concentre sur les exemples difficiles (bords)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.smooth_l1 = nn.SmoothL1Loss()

        # Sobel kernels pour Edge-Aware Loss (Expert 2025-12-28)
        # Force le modèle à "comprendre" les contours du H-channel, pas juste les "regarder"
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # Matrice staining Ruifrok pour extraction H-channel (même que RuifrokExtractor)
        self.register_buffer('stain_h', torch.tensor([0.650, 0.704, 0.286]).view(1, 3, 1, 1))

        # Uncertainty Weighting: log(σ²) pour chaque tâche (paramètres appris)
        # Initialisés à 0 → σ² = 1 → poids effectif = 1
        if adaptive:
            self.log_var_np = nn.Parameter(torch.zeros(1))  # log(σ²_np)
            self.log_var_hv = nn.Parameter(torch.zeros(1))  # log(σ²_hv)
            self.log_var_nt = nn.Parameter(torch.zeros(1))  # log(σ²_nt)

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """Dice loss pour segmentation."""
        pred_soft = F.softmax(pred, dim=1)[:, 1]  # Probabilité classe 1

        intersection = (pred_soft * target).sum()
        union = pred_soft.sum() + target.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        MSGE (Mean Squared Gradient Error) avec opérateur Sobel pour signal amplifié.

        PROBLÈME IDENTIFIÉ (Expert externe):
        - Différences finies simples (pixel[i+1] - pixel[i]) donnent signal trop faible
        - Dans HV maps [-1, 1], différences typiques ~0.01 → gradient loss négligeable
        → Modèle n'a pas de pression pour créer frontières nettes

        SOLUTION:
        - Utiliser noyau Sobel (3×3) qui amplifie naturellement les gradients
        - Sobel = convolution avec poids [-1, 0, 1] → signal 2-3× plus fort
        → Force le modèle à créer contours nets autour des noyaux

        Args:
            pred: Prédictions HV (B, 2, H, W)
            target: Targets HV (B, 2, H, W)
            mask: Masque des noyaux (B, 1, H, W) - optionnel
        """
        # Noyaux Sobel pour gradients horizontal et vertical
        sobel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

        # Appliquer Sobel séparément sur chaque canal (H et V)
        B, C, H, W = pred.shape

        # Reshape pour convolution: (B*C, 1, H, W)
        # Utiliser reshape() au lieu de view() pour gérer tensors non-contigus
        pred_reshaped = pred.reshape(B * C, 1, H, W)
        target_reshaped = target.reshape(B * C, 1, H, W)

        # Gradients Sobel avec padding pour garder la taille
        pred_grad_h = F.conv2d(pred_reshaped, sobel_h, padding=1)
        pred_grad_v = F.conv2d(pred_reshaped, sobel_v, padding=1)

        target_grad_h = F.conv2d(target_reshaped, sobel_h, padding=1)
        target_grad_v = F.conv2d(target_reshaped, sobel_v, padding=1)

        # Reshape back: (B, C, H, W)
        pred_grad_h = pred_grad_h.reshape(B, C, H, W)
        pred_grad_v = pred_grad_v.reshape(B, C, H, W)
        target_grad_h = target_grad_h.reshape(B, C, H, W)
        target_grad_v = target_grad_v.reshape(B, C, H, W)

        if mask is not None:
            # Masquer les gradients (uniquement sur les noyaux)
            grad_loss_h = F.mse_loss(pred_grad_h * mask, target_grad_h * mask, reduction='sum')
            grad_loss_v = F.mse_loss(pred_grad_v * mask, target_grad_v * mask, reduction='sum')

            # Normaliser par le nombre de pixels masqués
            n_pixels = mask.sum() * C  # Multiply by C car 2 canaux (H, V)
            grad_loss = (grad_loss_h + grad_loss_v) / (n_pixels + 1e-8)
        else:
            # Sans masque
            grad_loss = F.mse_loss(pred_grad_h, target_grad_h) + F.mse_loss(pred_grad_v, target_grad_v)

        return grad_loss

    def magnitude_loss(
        self,
        hv_pred: torch.Tensor,
        hv_target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Force le modèle à prédire des gradients FORTS aux frontières.

        ✅ EXPERT FIX (2025-12-24):
        1. Epsilon DANS la racine (stabilise gradients → Test 3 passe)
        2. Masquage AVANT réduction (élimine dilution fond → Test 1 passe)
        3. Erreur quadratique manuelle (contrôle exact du calcul)

        PROBLÈME RÉSOLU:
        - Magnitude plafonnait à 0.022 au lieu de 0.8+ (ratio 1/40)
        - Cause: Fond (90% pixels) tirait tout vers le bas
        - Solution: Normaliser UNIQUEMENT sur pixels de cellules

        RÉSULTAT ATTENDU:
        - Magnitude: 0.02 → 0.50+ (gain ×25)
        - AJI: 0.09 → 0.60+ (gain ×7)
        - Giant Blob résolu (1 instance → 8-12 cellules séparées)

        Args:
            hv_pred: Prédictions HV (B, 2, H, W) - float [-1, 1]
            hv_target: Targets HV (B, 2, H, W) - float [-1, 1]
            mask: Masque noyaux (B, 1, H, W) - binary [0, 1]

        Returns:
            Scalar loss (MSE sur magnitudes, masqué)

        Example:
            >>> # Avant fix: magnitude faible pas pénalisée
            >>> hv_pred = torch.randn(1, 2, 224, 224) * 0.02  # Faible
            >>> hv_target = torch.randn(1, 2, 224, 224) * 0.8  # Forte
            >>> mask = torch.ones(1, 1, 224, 224)
            >>> loss_before = 0.061  # Dilué par fond
            >>>
            >>> # Après fix: magnitude faible TRÈS pénalisée
            >>> loss_after = 0.61  # Signal pur (×10 plus fort)
        """
        # 1. Calculer magnitude avec epsilon DANS la racine
        #    FIX: Évite sqrt(0) qui tue les gradients (Test 3)
        mag_pred = torch.sqrt(torch.sum(hv_pred**2, dim=1) + 1e-6)  # (B, H, W)
        mag_true = torch.sqrt(torch.sum(hv_target**2, dim=1) + 1e-6)

        # 2. Erreur quadratique MANUELLE
        #    FIX: Pas F.mse_loss qui moyenne sur tous pixels
        loss = (mag_true - mag_pred)**2  # (B, H, W)

        # 3. Application du masque AVANT la réduction
        #    FIX: Élimine la dilution par le fond (Test 1)
        if mask is not None and mask.sum() > 0:
            # Squeeze pour matcher dimensions (B, H, W)
            weighted_loss = loss * mask.squeeze(1)

            # 4. Normaliser SEULEMENT par pixels de cellules
            #    FIX: Pas par toute l'image (50k pixels) mais par cellules (~5k)
            #    Résultat: Signal magnitude ×10 plus fort
            return weighted_loss.sum() / (mask.sum() + 1e-6)
        else:
            # Fallback sans masque (ne devrait jamais arriver en pratique)
            return loss.mean()

    def edge_aware_loss(
        self,
        hv_pred: torch.Tensor,
        images_rgb: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Edge-Aware Loss: Force le modèle à "comprendre" les contours du H-channel.

        EXPERT RECOMMENDATION (2025-12-28):
        "Actuellement, le modèle 'regarde' le canal H. Avec cette perte, il doit
        'comprendre' les bords du canal H. C'est le signal ultime pour l'AJI."

        Concept:
        1. Sobel sur H-channel → edges "idéales" (contours chromatine réelle)
        2. HV magnitude → edges prédites par le modèle
        3. MSE entre les deux → force alignement précis des frontières

        Le H-channel (Hématoxyline) contient l'information de densité chromatinienne.
        Les transitions abruptes de H indiquent les vraies frontières entre noyaux.
        En forçant HV_magnitude ≈ Sobel(H), le modèle apprend des frontières
        anatomiquement correctes.

        Args:
            hv_pred: Prédictions HV (B, 2, H, W) - float [-1, 1]
            images_rgb: Images RGB originales (B, 3, H, W) - uint8-like [0, 255]
            mask: Masque noyaux (B, 1, H, W) - binary [0, 1]

        Returns:
            Scalar loss (MSE entre HV magnitude et Sobel(H-channel))

        Impact attendu: AJI +10% (Expert estimate)
        """
        # Déplacer buffers sur le même device que les images
        device = images_rgb.device
        stain_h = self.stain_h.to(device)
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # 1. Extraire H-channel via déconvolution Ruifrok
        #    RGB → Optical Density → projection sur vecteur Hématoxyline
        images_rgb = images_rgb.clamp(1e-6, 255.0)
        od = -torch.log10(images_rgb / 255.0 + 1e-6)  # RGB → OD
        h_channel = torch.sum(od * stain_h, dim=1, keepdim=True)  # (B, 1, H, W)

        # 2. Sobel sur H-channel → edges "idéales"
        #    Les gradients forts de H indiquent les vraies frontières chromatine
        h_grad_x = F.conv2d(h_channel, sobel_x, padding=1)
        h_grad_y = F.conv2d(h_channel, sobel_y, padding=1)
        h_edges = torch.sqrt(h_grad_x**2 + h_grad_y**2 + 1e-6)  # (B, 1, H, W)

        # Normaliser h_edges pour être dans une plage comparable à HV magnitude
        # HV magnitude typique: [0, sqrt(2)] ≈ [0, 1.4]
        # Sobel(H) brut peut être beaucoup plus grand
        h_edges = h_edges / (h_edges.max() + 1e-6)  # Normaliser à [0, 1]

        # 3. HV magnitude prédite
        hv_magnitude = torch.sqrt(hv_pred[:, 0:1]**2 + hv_pred[:, 1:2]**2 + 1e-6)  # (B, 1, H, W)

        # Normaliser aussi HV magnitude (important pour MSE équitable)
        hv_magnitude_norm = hv_magnitude / (hv_magnitude.max() + 1e-6)

        # 4. MSE entre les deux (masqué sur noyaux)
        if mask is not None and mask.sum() > 0:
            # MSE uniquement sur pixels de noyaux
            loss = ((hv_magnitude_norm - h_edges) ** 2) * mask
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return F.mse_loss(hv_magnitude_norm, h_edges)

    def forward(
        self,
        np_pred: torch.Tensor,
        hv_pred: torch.Tensor,
        nt_pred: torch.Tensor,
        np_target: torch.Tensor,
        hv_target: torch.Tensor,
        nt_target: torch.Tensor,
        weight_map: torch.Tensor = None,
        images_rgb: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calcule la loss totale.

        Args:
            weight_map: Carte de poids Ronneberger (B, H, W) pour sur-pondérer
                       les frontières inter-cellulaires. Si None, poids uniforme.
            images_rgb: Images RGB originales (B, 3, H, W) pour Edge-Aware Loss.
                       Si None, Edge-Aware Loss n'est pas calculée.

        Returns:
            total_loss, dict avec losses individuelles
        """
        # NP loss: Focal + Dice (Phase 1 Phased Training)
        # Expert: Focal Loss gère le déséquilibre et focus sur les bords difficiles
        # Dice Loss ignore le déséquilibre et se concentre sur l'overlap
        #
        # ✅ AJOUT Tech Lead 2025-12-28: Pondération spatiale Ronneberger
        # Les pixels aux frontières inter-cellulaires reçoivent un poids 10× plus élevé
        # pour forcer le modèle à apprendre des séparations nettes.
        if weight_map is not None:
            # ✅ FIX Tech Lead 2025-12-28: CrossEntropy PURE + Ronneberger
            #
            # DIAGNOSTIC DU CONFLIT Focal ∩ Ronneberger:
            # - FocalLoss: (1-p)^γ downweight pixels où modèle confiant (p>0.8 → coef 0.04)
            # - Ronneberger: upweight pixels frontières (coef 10×)
            # - CONFLIT: Pixel frontière bien prédit (p=0.9) → Focal 0.01 × Ronneberger 10 = 0.1×
            #   → Le modèle "arrête" d'apprendre la précision chirurgicale des frontières
            #   → AJI régresse (0.55 → 0.54) malgré Dice stable
            #
            # SOLUTION: Ronneberger fait le focus spatial EXPLICITE, pas besoin de Focal
            # - Ronneberger cible géographiquement les frontières (ce que l'AJI mesure)
            # - CrossEntropy pure laisse le gradient intact pour que Ronneberger agisse
            #
            # NOTE: FocalLoss gérait aussi le déséquilibre via alpha, mais Ronneberger
            # compense car les frontières (upweighted 10×) sont dans les zones de noyaux.
            np_focal_raw = F.cross_entropy(np_pred, np_target.long(), reduction='none')
            np_focal = (np_focal_raw * weight_map).mean()
        else:
            # Sans weight_map (VAL ou ancien mode): FocalLoss pour déséquilibre classes
            np_focal = self.focal_loss(np_pred, np_target.long())

        np_dice = self.dice_loss(np_pred, np_target.float())
        np_loss = np_focal + np_dice

        # HV loss: MSE MASQUÉ (uniquement sur pixels de noyaux)
        # Littérature (Graham et al.): MSE doit être calculé UNIQUEMENT sur les noyaux
        # Sinon le modèle apprend à prédire 0 (background domine 70-80% des pixels)
        # TEST: Changé SmoothL1 → MSE pour améliorer gradients HV (Step 3 verification)
        mask = np_target.float().unsqueeze(1)  # (B, 1, H, W)

        if mask.sum() > 0:
            # Masquer pred et target
            hv_pred_masked = hv_pred * mask
            hv_target_masked = hv_target * mask

            # MSE sur les versions masquées (vs SmoothL1: gradients 2× plus forts)
            hv_mse_sum = F.mse_loss(hv_pred_masked, hv_target_masked, reduction='sum')
            hv_l1 = hv_mse_sum / (mask.sum() * 2)  # *2 car 2 canaux (H, V)
        else:
            hv_l1 = torch.tensor(0.0, device=hv_pred.device)

        # Gradient loss (MSGE - Graham et al.): force le modèle à apprendre les variations spatiales
        hv_gradient = self.gradient_loss(hv_pred, hv_target, mask=mask)

        # Magnitude loss (NOUVEAU - 2025-12-24): force le modèle à prédire gradients FORTS
        # PROBLÈME: HV MSE plafonne à 0.16, magnitude pred 0.04 vs targets 0.77 (ratio 0.05 = 20× trop faible!)
        # CAUSE: Loss actuelle (MSE + gradient) ne RÉCOMPENSE PAS magnitude élevée
        #        → Modèle apprend à prédire HV maps LISSES (compromis MSE vs gradient)
        # SOLUTION: MSE sur magnitude pour forcer valeurs ÉLEVÉES aux frontières
        # GAIN ATTENDU: magnitude 0.04 → 0.40-0.60 (10-15×), AJI 0.09 → 0.50-0.70 (5-7×)
        hv_magnitude = self.magnitude_loss(hv_pred, hv_target, mask=mask)

        # Loss totale HV (3 termes)
        # HISTORIQUE:
        #   - Lambda_hv=2.0 (EXPERT FIX 2025-12-23): équilibré après test stress lambda_hv=10.0
        #   - Lambda_magnitude=1.0 (ANCIEN 2025-12-24): masking bugué → magnitude 0.02
        #   - Lambda_magnitude=5.0 (EXPERT FIX 2025-12-24): masking corrigé → magnitude attendue 0.5+
        #
        # FIX 2025-12-25: Réduire multiplicateurs pour équilibrer avec NP
        # Avant: 3.0×gradient + 5.0×magnitude → loss HV ~7.0 (dominait tout)
        # Après: 1.0×gradient + 1.0×magnitude → loss HV ~1.0 (équilibré)
        hv_loss = hv_l1 + 1.0 * hv_gradient + self.lambda_magnitude * hv_magnitude

        # NT loss: CE MASQUÉ (uniquement sur pixels de noyaux)
        # FIX CRITIQUE 2025-12-25: Avant, calculé sur TOUS pixels → 85% background
        # → Modèle apprenait "prédire classe 0 partout = 85% accuracy"
        # → NT Acc était 0.0002% (catastrophique)
        # SOLUTION: Masquer comme HV loss (Graham et al. 2019)
        if mask.sum() > 0:
            # Flatten: (B, C, H, W) → (B*H*W, C) et (B, H, W) → (B*H*W,)
            B, C, H, W = nt_pred.shape
            nt_pred_flat = nt_pred.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, n_classes)
            nt_target_flat = nt_target.reshape(-1)  # (B*H*W,)
            mask_flat = mask.squeeze(1).reshape(-1)  # (B*H*W,)

            # Sélectionner uniquement les pixels de noyaux
            nuclear_indices = mask_flat > 0.5
            nt_pred_masked = nt_pred_flat[nuclear_indices]  # (N_nuclear, n_classes)
            nt_target_masked = nt_target_flat[nuclear_indices]  # (N_nuclear,)

            # CrossEntropyLoss sur pixels de noyaux uniquement
            nt_loss = F.cross_entropy(nt_pred_masked, nt_target_masked.long())
        else:
            nt_loss = torch.tensor(0.0, device=nt_pred.device)

        # Edge-Aware Loss (Expert 2025-12-28): Force le modèle à aligner HV magnitude
        # avec les edges réelles du canal Hématoxyline (Sobel sur H-channel).
        # Le H-channel montre la vraie chromatine → vraies frontières nucléaires.
        # GAIN ATTENDU: +10% AJI en forçant l'alignement anatomique des prédictions.
        if images_rgb is not None and self.lambda_edge > 0:
            edge_loss = self.edge_aware_loss(hv_pred, images_rgb, mask=mask)
        else:
            edge_loss = torch.tensor(0.0, device=hv_pred.device)

        # Calcul du total selon le mode
        if self.adaptive:
            # Uncertainty Weighting: L = L_i / (2σ²_i) + log(σ_i)
            # Formule: L_total = Σ (L_i * exp(-log_var_i) + log_var_i)
            # Cela revient à: L_i / σ² + log(σ)
            total = (
                torch.exp(-self.log_var_np) * np_loss + self.log_var_np +
                torch.exp(-self.log_var_hv) * hv_loss + self.log_var_hv +
                torch.exp(-self.log_var_nt) * nt_loss + self.log_var_nt
            )
            # Ajouter Edge Loss (pas en mode adaptive, poids fixe)
            total = total + self.lambda_edge * edge_loss

            # Calculer les poids effectifs pour monitoring
            w_np = torch.exp(-self.log_var_np).item()
            w_hv = torch.exp(-self.log_var_hv).item()
            w_nt = torch.exp(-self.log_var_nt).item()

            return total, {
                'np': np_loss.item(),
                'np_focal': np_focal.item(),
                'np_dice': np_dice.item(),
                'hv': hv_loss.item(),
                'hv_l1': hv_l1.item(),
                'hv_gradient': hv_gradient.item(),
                'hv_magnitude': hv_magnitude.item(),
                'edge': edge_loss.item(),
                'nt': nt_loss.item(),
                'w_np': w_np,
                'w_hv': w_hv,
                'w_nt': w_nt,
            }
        else:
            # Poids fixes
            total = (
                self.lambda_np * np_loss +
                self.lambda_hv * hv_loss +
                self.lambda_nt * nt_loss +
                self.lambda_edge * edge_loss
            )

            return total, {
                'np': np_loss.item(),
                'np_focal': np_focal.item(),
                'np_dice': np_dice.item(),
                'hv': hv_loss.item(),
                'hv_l1': hv_l1.item(),
                'hv_gradient': hv_gradient.item(),
                'hv_magnitude': hv_magnitude.item(),
                'edge': edge_loss.item(),
                'nt': nt_loss.item(),
            }

    def get_learned_weights(self) -> dict:
        """Retourne les poids appris (mode adaptive uniquement)."""
        if not self.adaptive:
            return {
                'w_np': self.lambda_np,
                'w_hv': self.lambda_hv,
                'w_nt': self.lambda_nt,
            }

        return {
            'w_np': torch.exp(-self.log_var_np).item(),
            'w_hv': torch.exp(-self.log_var_hv).item(),
            'w_nt': torch.exp(-self.log_var_nt).item(),
            'log_var_np': self.log_var_np.item(),
            'log_var_hv': self.log_var_hv.item(),
            'log_var_nt': self.log_var_nt.item(),
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
