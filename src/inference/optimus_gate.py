#!/usr/bin/env python3
"""
Optimus-Gate: Architecture unifiée pour CellViT-Optimus.

Combine:
- Flux Global (CLS token → OrganHead → Organe + OOD)
- Flux Local (Patch tokens → HoVerNet → NP/HV/NT)
- Triple Sécurité OOD (entropie organe + Mahalanobis global + Mahalanobis local)

Usage:
    model = OptimusGate()
    model.load_checkpoint("models/checkpoints/optimus_gate.pth")

    result = model.predict(image)
    print(result.organ)       # "Prostate"
    print(result.cells)       # Liste de cellules
    print(result.is_ood)      # False
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.organ_head import OrganHead, OrganPrediction, PANNUKE_ORGANS
from src.uncertainty import UncertaintyEstimator, UncertaintyResult, ConfidenceLevel


@dataclass
class CellDetection:
    """Une cellule détectée."""
    x: int
    y: int
    type_idx: int
    type_name: str
    confidence: float


@dataclass
class OptimusGateResult:
    """Résultat complet de l'inférence Optimus-Gate."""
    # Flux Global
    organ: OrganPrediction

    # Flux Local
    np_mask: np.ndarray          # Masque binaire noyaux (H, W)
    hv_map: np.ndarray           # Cartes H/V (2, H, W)
    type_map: np.ndarray         # Carte de types (H, W)
    type_probs: np.ndarray       # Probabilités types (5, H, W)
    cells: List[CellDetection] = field(default_factory=list)

    # Incertitude
    uncertainty: Optional[UncertaintyResult] = None

    # OOD Triple Sécurité
    ood_score_global: float = 0.0    # Entropie + Mahalanobis sur CLS
    ood_score_local: float = 0.0     # Mahalanobis sur patches
    ood_score_combined: float = 0.0  # Score final
    is_ood: bool = False

    # Confiance globale
    confidence_level: ConfidenceLevel = ConfidenceLevel.FIABLE

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def cell_counts(self) -> Dict[str, int]:
        """Compte les cellules par type."""
        counts = {}
        for cell in self.cells:
            counts[cell.type_name] = counts.get(cell.type_name, 0) + 1
        return counts


class OptimusGate(nn.Module):
    """
    Architecture Optimus-Gate complète.

    Combine le backbone H-optimus-0 (gelé) avec deux têtes:
    1. OrganHead: Classification d'organe et OOD global
    2. HoVerNetDecoder: Segmentation cellulaire et typage
    """

    # Types de cellules PanNuke
    CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

    def __init__(
        self,
        embed_dim: int = 1536,
        bottleneck_dim: int = 256,
        n_organs: int = 19,
        n_cell_types: int = 5,
        dropout: float = 0.1,
        ood_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        """
        Args:
            embed_dim: Dimension des embeddings H-optimus-0
            bottleneck_dim: Dimension du bottleneck HoVer-Net
            n_organs: Nombre d'organes
            n_cell_types: Nombre de types cellulaires
            dropout: Dropout rate
            ood_weights: (entropy_weight, mahal_global_weight, mahal_local_weight)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.ood_weights = ood_weights

        # Flux Global: OrganHead (CLS token → Organe + OOD)
        self.organ_head = OrganHead(
            embed_dim=embed_dim,
            hidden_dim=256,
            n_organs=n_organs,
            dropout=dropout,
        )

        # Flux Local: HoVerNetDecoder (Patches → NP/HV/NT)
        self.hovernet = HoVerNetDecoder(
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
            n_classes=n_cell_types,
            dropout=dropout,
        )

        # Estimateur d'incertitude
        self.uncertainty_estimator = UncertaintyEstimator()

        # Pour OOD local (sur patches moyennés)
        self.local_ood_fitted = False
        self.register_buffer('patch_mean', None)
        self.register_buffer('patch_cov_inv', None)
        self.patch_threshold = None

    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Features H-optimus-0 (B, 261, 1536)
                      [CLS token (1) + Patches (256) + Registers (4)]

        Returns:
            organ_logits: (B, n_organs)
            np_out: (B, 2, H, W)
            hv_out: (B, 2, H, W)
            nt_out: (B, n_types, H, W)
        """
        # Séparer CLS et patches
        cls_token = features[:, 0, :]      # (B, 1536)
        # patches = features[:, 1:257, :]  # Utilisé par HoVerNet

        # Flux Global
        organ_logits = self.organ_head(cls_token)

        # Flux Local
        np_out, hv_out, nt_out = self.hovernet(features)

        return organ_logits, np_out, hv_out, nt_out

    def fit_ood(
        self,
        train_features: torch.Tensor,
        percentile: float = 95.0
    ):
        """
        Calibre les détecteurs OOD sur les features d'entraînement.

        Args:
            train_features: Features (N, 261, 1536)
            percentile: Percentile pour les seuils
        """
        with torch.no_grad():
            # CLS tokens pour OOD global
            cls_tokens = train_features[:, 0, :]
            self.organ_head.fit_ood(cls_tokens, percentile)

            # Patches moyennés pour OOD local
            # FIX Register Token (2025-12-25): indices 5-261 = patches spatiaux
            # AVANT: features[:, 1:257, :] incluait les 4 Registers!
            patches = train_features[:, 5:261, :]  # (N, 256, 1536)
            patch_means = patches.mean(dim=1)      # (N, 1536)

            self.patch_mean = patch_means.mean(dim=0)

            # Covariance
            tokens_np = patch_means.cpu().numpy()
            try:
                from sklearn.covariance import LedoitWolf
                cov_estimator = LedoitWolf()
                cov_estimator.fit(tokens_np)
                cov = cov_estimator.covariance_
            except ImportError:
                cov = np.cov(tokens_np, rowvar=False)
                reg = 1e-5 * np.trace(cov) / cov.shape[0]
                cov += reg * np.eye(cov.shape[0])

            cov_inv = np.linalg.pinv(cov)
            self.patch_cov_inv = torch.from_numpy(cov_inv).float().to(train_features.device)

            # Seuil
            distances = self._local_mahalanobis(patch_means)
            self.patch_threshold = float(np.percentile(
                distances.cpu().numpy(), percentile
            ))

            self.local_ood_fitted = True

    def _local_mahalanobis(self, patch_means: torch.Tensor) -> torch.Tensor:
        """Distance de Mahalanobis sur les patches moyennés."""
        if patch_means.dim() == 1:
            patch_means = patch_means.unsqueeze(0)

        diff = patch_means - self.patch_mean
        left = torch.mm(diff, self.patch_cov_inv)
        return torch.sqrt(torch.sum(left * diff, dim=1))

    def compute_triple_ood(
        self,
        features: torch.Tensor,
        organ_probs: torch.Tensor,
    ) -> Tuple[float, float, float, bool]:
        """
        Calcule le score OOD "Triple Sécurité".

        Args:
            features: Features (1, 261, 1536)
            organ_probs: Probabilités organe (1, n_organs)

        Returns:
            (ood_global, ood_local, ood_combined, is_ood)
        """
        cls_token = features[:, 0, :]

        # 1. OOD Global (entropie + Mahalanobis sur CLS)
        ood_global, _ = self.organ_head.compute_ood_score(cls_token, organ_probs)

        # 2. OOD Local (Mahalanobis sur patches)
        ood_local = 0.0
        if self.local_ood_fitted:
            # FIX Register Token (2025-12-25): indices 5-261 = patches spatiaux
            patches = features[:, 5:261, :]
            patch_mean = patches.mean(dim=1)
            mahal_local = self._local_mahalanobis(patch_mean).item()
            ood_local = min(mahal_local / (self.patch_threshold + 1e-10), 2.0) / 2.0

        # 3. Score combiné
        w_entropy, w_global, w_local = self.ood_weights
        ood_combined = (
            w_entropy * ood_global +
            w_global * 0 +  # Déjà inclus dans ood_global
            w_local * ood_local
        )
        # Simplification: ood_global contient déjà entropy + mahal global
        ood_combined = 0.6 * ood_global + 0.4 * ood_local

        # Détection (seuils relaxés pour éviter faux positifs)
        is_ood = ood_combined > 0.7 or ood_global > 0.8

        return ood_global, ood_local, ood_combined, is_ood

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        threshold_np: float = 0.5,
        threshold_type: float = 0.5,
    ) -> OptimusGateResult:
        """
        Prédiction complète.

        Args:
            features: Features H-optimus-0 (261, 1536) ou (1, 261, 1536)
            threshold_np: Seuil pour masque NP
            threshold_type: Seuil pour confiance type

        Returns:
            OptimusGateResult avec toutes les informations
        """
        self.eval()

        # Ajouter dimension batch si nécessaire
        if features.dim() == 2:
            features = features.unsqueeze(0)

        # Forward
        organ_logits, np_out, hv_out, nt_out = self.forward(features)

        # Probabilités
        organ_probs = torch.softmax(organ_logits, dim=-1)
        np_probs = torch.softmax(np_out, dim=1)
        nt_probs = torch.softmax(nt_out, dim=1)

        # Prédiction organe
        organ_pred = self.organ_head.predict_with_ood(features[:, 0, :])

        # Masques et cartes
        np_mask = (np_probs[0, 1] > threshold_np).cpu().numpy()
        hv_map = hv_out[0].cpu().numpy()
        # CORRECTIF: Model trains/outputs [0-4], PanNuke labels are [1-5] → +1 REQUIRED
        type_map = nt_probs[0].argmax(dim=0).cpu().numpy() + 1
        type_probs = nt_probs[0].cpu().numpy()

        # OOD Triple Sécurité
        ood_global, ood_local, ood_combined, is_ood = self.compute_triple_ood(
            features, organ_probs
        )

        # Incertitude
        uncertainty = self.uncertainty_estimator.estimate(
            np_probs[0].permute(1, 2, 0).cpu().numpy(),
            nt_probs[0].permute(1, 2, 0).cpu().numpy(),
            compute_map=True,
        )

        # Niveau de confiance final
        if is_ood or organ_pred.is_ood:
            confidence_level = ConfidenceLevel.HORS_DOMAINE
        elif uncertainty.level == ConfidenceLevel.A_REVOIR or organ_pred.entropy > 0.5:
            confidence_level = ConfidenceLevel.A_REVOIR
        else:
            confidence_level = ConfidenceLevel.FIABLE

        # Détecter les cellules (simplifié - centroïdes du masque)
        cells = self._extract_cells(np_mask, type_map, type_probs, threshold_type)

        return OptimusGateResult(
            organ=organ_pred,
            np_mask=np_mask,
            hv_map=hv_map,
            type_map=type_map,
            type_probs=type_probs,
            cells=cells,
            uncertainty=uncertainty,
            ood_score_global=ood_global,
            ood_score_local=ood_local,
            ood_score_combined=ood_combined,
            is_ood=is_ood or organ_pred.is_ood,
            confidence_level=confidence_level,
        )

    def _extract_cells(
        self,
        np_mask: np.ndarray,
        type_map: np.ndarray,
        type_probs: np.ndarray,
        threshold: float,
    ) -> List[CellDetection]:
        """Extrait les cellules du masque (version simplifiée)."""
        from scipy import ndimage

        cells = []

        # Labelliser les composantes connexes
        labeled, n_cells = ndimage.label(np_mask)

        for i in range(1, min(n_cells + 1, 1000)):  # Limiter à 1000 cellules
            mask = labeled == i
            if mask.sum() < 10:  # Ignorer les très petites régions
                continue

            # Centroïde
            coords = np.where(mask)
            y, x = int(coords[0].mean()), int(coords[1].mean())

            # Type majoritaire
            types_in_cell = type_map[mask]
            type_idx = int(np.bincount(types_in_cell).argmax())

            # type_idx est dans [1-5] après +1, convertir vers [0-4] pour indexer type_probs et CELL_TYPES
            if not (1 <= type_idx <= 5):
                continue

            # Confiance moyenne
            confidence = float(type_probs[type_idx - 1, mask].mean())

            if confidence >= threshold:
                cells.append(CellDetection(
                    x=x,
                    y=y,
                    type_idx=type_idx,
                    type_name=self.CELL_TYPES[type_idx - 1],
                    confidence=confidence,
                ))

        return cells

    def generate_report(self, result: OptimusGateResult) -> str:
        """Génère un rapport textuel complet."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║             RAPPORT OPTIMUS-GATE                              ║",
            "╠══════════════════════════════════════════════════════════════╣",
        ]

        # Niveau de confiance
        level_emoji = {
            ConfidenceLevel.FIABLE: "✅",
            ConfidenceLevel.A_REVOIR: "⚠️",
            ConfidenceLevel.HORS_DOMAINE: "🚫",
        }
        emoji = level_emoji[result.confidence_level]

        lines.extend([
            f"║ {emoji} NIVEAU: {result.confidence_level.value.upper():40} ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ 🔬 DIAGNOSTIC CONTEXTE (Flux Global)                         ║",
            f"║    Organe prédit: {result.organ.organ_name:20} ({result.organ.confidence:.1%}) ║",
            f"║    Entropie: {result.organ.entropy:.3f}                                        ║",
            f"║    OOD Global: {result.ood_score_global:.3f}                                   ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ 🔎 ANALYSE CELLULAIRE (Flux Local)                           ║",
            f"║    Total cellules: {result.n_cells:4}                                       ║",
        ])

        # Comptage par type
        counts = result.cell_counts()
        type_emojis = {"Neoplastic": "🔴", "Inflammatory": "🟢", "Connective": "🔵",
                       "Dead": "🟡", "Epithelial": "🩵"}

        for cell_type in self.CELL_TYPES:
            count = counts.get(cell_type, 0)
            emoji = type_emojis.get(cell_type, "•")
            lines.append(f"║      {emoji} {cell_type:15}: {count:4}                            ║")

        lines.extend([
            "╠══════════════════════════════════════════════════════════════╣",
            "║ 🛡️ TRIPLE SÉCURITÉ OOD                                        ║",
            f"║    Score Global: {result.ood_score_global:.3f}                                 ║",
            f"║    Score Local:  {result.ood_score_local:.3f}                                 ║",
            f"║    Score Combiné: {result.ood_score_combined:.3f}                              ║",
            f"║    Hors Distribution: {'OUI' if result.is_ood else 'NON':3}                              ║",
            "╚══════════════════════════════════════════════════════════════╝",
        ])

        return "\n".join(lines)

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        """Sauvegarde le modèle."""
        checkpoint = {
            'organ_head': self.organ_head.state_dict(),
            'hovernet': self.hovernet.state_dict(),
            'ood_fitted': self.organ_head.ood_fitted,
            'local_ood_fitted': self.local_ood_fitted,
        }

        if self.organ_head.ood_fitted:
            checkpoint['organ_ood'] = {
                'cls_mean': self.organ_head.cls_mean,
                'cls_cov_inv': self.organ_head.cls_cov_inv,
                'threshold': self.organ_head.mahalanobis_threshold,
            }

        if self.local_ood_fitted:
            checkpoint['local_ood'] = {
                'patch_mean': self.patch_mean,
                'patch_cov_inv': self.patch_cov_inv,
                'threshold': self.patch_threshold,
            }

        if metadata:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Charge le modèle."""
        checkpoint = torch.load(path, map_location='cpu')

        self.organ_head.load_state_dict(checkpoint['organ_head'])
        self.hovernet.load_state_dict(checkpoint['hovernet'])

        if checkpoint.get('organ_ood'):
            ood = checkpoint['organ_ood']
            self.organ_head.cls_mean = ood['cls_mean']
            self.organ_head.cls_cov_inv = ood['cls_cov_inv']
            self.organ_head.mahalanobis_threshold = ood['threshold']
            self.organ_head.ood_fitted = True

        if checkpoint.get('local_ood'):
            ood = checkpoint['local_ood']
            self.patch_mean = ood['patch_mean']
            self.patch_cov_inv = ood['patch_cov_inv']
            self.patch_threshold = ood['threshold']
            self.local_ood_fitted = True

    def load_from_separate_checkpoints(
        self,
        hovernet_path: str,
        organ_head_path: str,
        device: str = "cpu"
    ):
        """
        Charge les poids depuis les checkpoints séparés.

        Args:
            hovernet_path: Chemin vers hovernet_best.pth
            organ_head_path: Chemin vers organ_head_best.pth
            device: Device cible
        """
        print(f"Chargement des checkpoints...")

        # Charger HoVer-Net
        hovernet_ckpt = torch.load(hovernet_path, map_location=device)
        if 'model_state_dict' in hovernet_ckpt:
            self.hovernet.load_state_dict(hovernet_ckpt['model_state_dict'])
        else:
            self.hovernet.load_state_dict(hovernet_ckpt)
        print(f"  ✓ HoVer-Net chargé depuis {hovernet_path}")

        # Charger OrganHead
        organ_ckpt = torch.load(organ_head_path, map_location=device)
        if 'model_state_dict' in organ_ckpt:
            state_dict = organ_ckpt['model_state_dict']
            # Filtrer les clés OOD qui sont des buffers, pas des paramètres du classifier
            filtered_state = {k: v for k, v in state_dict.items()
                            if k not in ['cls_mean', 'cls_cov_inv']}
            self.organ_head.load_state_dict(filtered_state, strict=False)
        else:
            self.organ_head.load_state_dict(organ_ckpt, strict=False)
        print(f"  ✓ OrganHead chargé depuis {organ_head_path}")

        # Charger OOD calibration si disponible
        if organ_ckpt.get('ood_calibrated') or organ_ckpt.get('cls_mean') is not None:
            self.organ_head.cls_mean = organ_ckpt.get('cls_mean')
            self.organ_head.cls_cov_inv = organ_ckpt.get('cls_cov_inv')
            self.organ_head.mahalanobis_threshold = organ_ckpt.get('mahalanobis_threshold')
            self.organ_head.ood_fitted = True
            print(f"  ✓ OOD calibré (threshold: {self.organ_head.mahalanobis_threshold:.2f})")

        # Afficher les métriques
        if 'val_acc' in organ_ckpt:
            print(f"  📊 OrganHead Val Acc: {organ_ckpt['val_acc']:.4f}")
        if 'best_dice' in hovernet_ckpt:
            print(f"  📊 HoVer-Net Dice: {hovernet_ckpt['best_dice']:.4f}")

        self.to(device)
        self.eval()
        print(f"  ✓ Modèle prêt sur {device}")

    @classmethod
    def from_pretrained(
        cls,
        hovernet_path: str = "models/checkpoints/hovernet_best.pth",
        organ_head_path: str = "models/checkpoints/organ_head_best.pth",
        device: str = "cuda",
        **kwargs
    ) -> "OptimusGate":
        """
        Crée un OptimusGate pré-entraîné.

        Usage:
            model = OptimusGate.from_pretrained()
            result = model.predict(features)
        """
        model = cls(**kwargs)
        model.load_from_separate_checkpoints(hovernet_path, organ_head_path, device)
        return model


# Test
if __name__ == "__main__":
    print("Test OptimusGate...")
    print("=" * 60)

    # Créer le modèle
    model = OptimusGate()
    model.eval()

    print(f"✓ Modèle créé")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {n_params:,}")

    # Simuler des features H-optimus-0
    features = torch.randn(1, 261, 1536)

    # Forward
    organ_logits, np_out, hv_out, nt_out = model(features)
    print(f"\n✓ Forward pass:")
    print(f"  Organ logits: {organ_logits.shape}")
    print(f"  NP output: {np_out.shape}")
    print(f"  HV output: {hv_out.shape}")
    print(f"  NT output: {nt_out.shape}")

    # Fit OOD
    train_features = torch.randn(100, 261, 1536)
    model.fit_ood(train_features)
    print(f"\n✓ OOD calibré")

    # Prédiction complète
    result = model.predict(features)
    print(f"\n✓ Prédiction complète:")
    print(f"  Organe: {result.organ.organ_name}")
    print(f"  Cellules: {result.n_cells}")
    print(f"  OOD combiné: {result.ood_score_combined:.3f}")
    print(f"  Is OOD: {result.is_ood}")
    print(f"  Niveau: {result.confidence_level.value}")

    # Rapport
    print("\n" + model.generate_report(result))

    print("\n" + "=" * 60)
    print("✅ Tous les tests passent!")
