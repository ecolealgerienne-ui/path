#!/usr/bin/env python3
"""
Optimus-Gate Multi-Family: Routage automatique vers 5 HoVer-Net spécialisés.

Architecture:
    Image → H-optimus-0 → CLS token → OrganHead → Organe
                       → Patch tokens → Router → HoVer-Net[famille]

Usage:
    model = OptimusGateMultiFamily.from_pretrained()
    result = model.predict(features)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.organ_head import OrganHead, OrganPrediction, PANNUKE_ORGANS
from src.uncertainty import UncertaintyEstimator, UncertaintyResult, ConfidenceLevel


# Mapping organe → famille
ORGAN_TO_FAMILY = {
    # Glandulaire & Hormonale (acini, sécrétions)
    "Breast": "glandular",
    "Prostate": "glandular",
    "Thyroid": "glandular",
    "Pancreatic": "glandular",
    "Adrenal_gland": "glandular",

    # Digestive (formes tubulaires)
    "Colon": "digestive",
    "Stomach": "digestive",
    "Esophagus": "digestive",
    "Bile-duct": "digestive",

    # Urologique & Reproductif (densité nucléaire)
    "Kidney": "urologic",
    "Bladder": "urologic",
    "Testis": "urologic",
    "Ovarian": "urologic",
    "Uterus": "urologic",
    "Cervix": "urologic",

    # Respiratoire & Hépatique (structures ouvertes)
    "Lung": "respiratory",
    "Liver": "respiratory",

    # Épidermoïde (couches stratifiées)
    "Skin": "epidermal",
    "HeadNeck": "epidermal",
}

FAMILIES = ["glandular", "digestive", "urologic", "respiratory", "epidermal"]

# Familles avec HV fiable (MSE < 0.1)
RELIABLE_HV_FAMILIES = ["glandular", "digestive"]

# Types de cellules PanNuke
CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]


@dataclass
class CellDetection:
    """Une cellule détectée."""
    x: int
    y: int
    type_idx: int
    type_name: str
    confidence: float


@dataclass
class MultiFamilyResult:
    """Résultat complet avec routage par famille."""
    # Flux Global
    organ: OrganPrediction
    family: str
    family_hv_reliable: bool  # HV MSE < 0.1 ?

    # Flux Local
    np_mask: np.ndarray
    hv_map: np.ndarray
    type_map: np.ndarray
    type_probs: np.ndarray
    cells: List[CellDetection] = field(default_factory=list)

    # Incertitude
    uncertainty: Optional[UncertaintyResult] = None

    # OOD
    ood_score_global: float = 0.0
    ood_score_local: float = 0.0
    is_ood: bool = False

    # Confiance
    confidence_level: ConfidenceLevel = ConfidenceLevel.FIABLE

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def cell_counts(self) -> Dict[str, int]:
        counts = {}
        for cell in self.cells:
            counts[cell.type_name] = counts.get(cell.type_name, 0) + 1
        return counts


class OptimusGateMultiFamily(nn.Module):
    """
    Optimus-Gate avec 5 décodeurs HoVer-Net spécialisés.

    Le routage est automatique:
    1. OrganHead prédit l'organe depuis le CLS token
    2. L'organe est mappé à une famille
    3. Le décodeur spécialisé de la famille est utilisé
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        bottleneck_dim: int = 256,
        n_organs: int = 19,
        n_cell_types: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Flux Global: OrganHead
        self.organ_head = OrganHead(
            embed_dim=embed_dim,
            hidden_dim=256,
            n_organs=n_organs,
            dropout=dropout,
        )

        # Flux Local: 5 HoVer-Net spécialisés
        self.hovernet_decoders = nn.ModuleDict()
        for family in FAMILIES:
            self.hovernet_decoders[family] = HoVerNetDecoder(
                embed_dim=embed_dim,
                bottleneck_dim=bottleneck_dim,
                n_classes=n_cell_types,
                dropout=dropout,
            )

        # Décodeur par défaut (fallback si organe inconnu)
        self.default_family = "glandular"

        # Estimateur d'incertitude
        self.uncertainty_estimator = UncertaintyEstimator()

        # État de chargement
        self.loaded_families = set()

    def get_family(self, organ_name: str) -> str:
        """Obtient la famille pour un organe."""
        return ORGAN_TO_FAMILY.get(organ_name, self.default_family)

    def forward(
        self,
        features: torch.Tensor,
        family: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass avec routage.

        Args:
            features: (B, 261, 1536)
            family: Famille forcée, sinon auto-routage
        """
        cls_token = features[:, 0, :]

        # Flux Global
        organ_logits = self.organ_head(cls_token)

        # Auto-routage si famille non spécifiée
        if family is None:
            with torch.no_grad():
                organ_idx = organ_logits.argmax(dim=-1).item()
                organ_name = PANNUKE_ORGANS[organ_idx]
                family = self.get_family(organ_name)

        # Flux Local avec décodeur spécialisé
        decoder = self.hovernet_decoders[family]
        np_out, hv_out, nt_out = decoder(features)

        return organ_logits, np_out, hv_out, nt_out, family

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        threshold_np: float = 0.5,
        threshold_type: float = 0.5,
        force_family: Optional[str] = None,
    ) -> MultiFamilyResult:
        """
        Prédiction complète avec routage automatique.
        """
        self.eval()

        if features.dim() == 2:
            features = features.unsqueeze(0)

        # 1. Prédiction organe
        cls_token = features[:, 0, :]
        organ_pred = self.organ_head.predict_with_ood(cls_token)

        # 2. Déterminer la famille
        family = force_family or self.get_family(organ_pred.organ_name)
        family_hv_reliable = family in RELIABLE_HV_FAMILIES

        # 3. Forward avec le bon décodeur
        decoder = self.hovernet_decoders[family]
        np_out, hv_out, nt_out = decoder(features)

        # Probabilités
        np_probs = torch.softmax(np_out, dim=1)
        nt_probs = torch.softmax(nt_out, dim=1)

        # Masques et cartes
        np_mask = (np_probs[0, 1] > threshold_np).cpu().numpy()
        hv_map = hv_out[0].cpu().numpy()
        # CORRECTIF: Model trains/outputs [0-4], PanNuke labels are [1-5] → +1 REQUIRED
        type_map = nt_probs[0].argmax(dim=0).cpu().numpy() + 1
        type_probs = nt_probs[0].cpu().numpy()

        # Incertitude
        uncertainty = self.uncertainty_estimator.estimate(
            np_probs[0].permute(1, 2, 0).cpu().numpy(),
            nt_probs[0].permute(1, 2, 0).cpu().numpy(),
            compute_map=True,
        )

        # OOD scores
        ood_global, _ = self.organ_head.compute_ood_score(
            cls_token,
            torch.softmax(self.organ_head(cls_token), dim=-1)
        )

        # Niveau de confiance
        if organ_pred.is_ood:
            confidence_level = ConfidenceLevel.HORS_DOMAINE
        elif uncertainty.level == ConfidenceLevel.A_REVOIR:
            confidence_level = ConfidenceLevel.A_REVOIR
        elif not family_hv_reliable:
            # Familles avec HV dégradé → avertissement
            confidence_level = ConfidenceLevel.A_REVOIR
        else:
            confidence_level = ConfidenceLevel.FIABLE

        # Extraire cellules
        cells = self._extract_cells(np_mask, type_map, type_probs, threshold_type)

        return MultiFamilyResult(
            organ=organ_pred,
            family=family,
            family_hv_reliable=family_hv_reliable,
            np_mask=np_mask,
            hv_map=hv_map,
            type_map=type_map,
            type_probs=type_probs,
            cells=cells,
            uncertainty=uncertainty,
            ood_score_global=ood_global,
            is_ood=organ_pred.is_ood,
            confidence_level=confidence_level,
        )

    def _extract_cells(
        self,
        np_mask: np.ndarray,
        type_map: np.ndarray,
        type_probs: np.ndarray,
        threshold: float,
    ) -> List[CellDetection]:
        """Extrait les cellules du masque."""
        from scipy import ndimage

        cells = []
        labeled, n_cells = ndimage.label(np_mask)

        for i in range(1, min(n_cells + 1, 1000)):
            mask = labeled == i
            if mask.sum() < 10:
                continue

            coords = np.where(mask)
            y, x = int(coords[0].mean()), int(coords[1].mean())

            types_in_cell = type_map[mask]
            type_idx = int(np.bincount(types_in_cell).argmax())
            confidence = float(type_probs[type_idx, mask].mean())

            if confidence >= threshold:
                cells.append(CellDetection(
                    x=x, y=y,
                    type_idx=type_idx,
                    type_name=CELL_TYPES[type_idx],
                    confidence=confidence,
                ))

        return cells

    def generate_report(self, result: MultiFamilyResult) -> str:
        """Génère un rapport avec info famille."""
        level_emoji = {
            ConfidenceLevel.FIABLE: "✅",
            ConfidenceLevel.A_REVOIR: "⚠️",
            ConfidenceLevel.HORS_DOMAINE: "🚫",
        }
        type_emojis = {
            "Neoplastic": "🔴", "Inflammatory": "🟢",
            "Connective": "🔵", "Dead": "🟡", "Epithelial": "🩵"
        }

        emoji = level_emoji[result.confidence_level]
        hv_status = "✅ Fiable" if result.family_hv_reliable else "⚠️ Vérifier manuellement"

        lines = [
            "╔════════════════════════════════════════════════════════════════╗",
            "║           RAPPORT OPTIMUS-GATE (Multi-Famille)                 ║",
            "╠════════════════════════════════════════════════════════════════╣",
            f"║ {emoji} NIVEAU: {result.confidence_level.value.upper():46} ║",
            "╠════════════════════════════════════════════════════════════════╣",
            "║ 🔬 CONTEXTE TISSULAIRE                                         ║",
            f"║    Organe: {result.organ.organ_name:20} ({result.organ.confidence:.1%})          ║",
            f"║    Famille: {result.family.upper():18}                            ║",
            f"║    Séparation HV: {hv_status:20}                      ║",
            "╠════════════════════════════════════════════════════════════════╣",
            "║ 🔎 ANALYSE CELLULAIRE                                          ║",
            f"║    Cellules détectées: {result.n_cells:4}                                     ║",
        ]

        counts = result.cell_counts()
        total = sum(counts.values())

        if total > 0:
            for name in CELL_TYPES:
                count = counts.get(name, 0)
                pct = count / total * 100
                emoji = type_emojis.get(name, '•')
                bar_len = int(pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"║    {emoji} {name:12}: {bar} {count:3} ({pct:5.1f}%)  ║")

        lines.extend([
            "╠════════════════════════════════════════════════════════════════╣",
            "║ 🛡️ SÉCURITÉ                                                     ║",
            f"║    OOD Score: {result.ood_score_global:.3f}                                         ║",
            f"║    Hors Domaine: {'OUI 🚫' if result.is_ood else 'NON ✓':6}                                       ║",
            "╚════════════════════════════════════════════════════════════════╝",
        ])

        return "\n".join(lines)

    def load_family_checkpoint(
        self,
        family: str,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        """Charge un checkpoint pour une famille."""
        ckpt = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in ckpt:
            self.hovernet_decoders[family].load_state_dict(ckpt['model_state_dict'])
        else:
            self.hovernet_decoders[family].load_state_dict(ckpt)

        self.loaded_families.add(family)

        dice = ckpt.get('best_dice', ckpt.get('dice', 'N/A'))
        print(f"  ✓ {family.capitalize():12} chargé (Dice: {dice})")

    def load_organ_head(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        """Charge l'OrganHead."""
        ckpt = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            filtered = {k: v for k, v in state_dict.items()
                       if k not in ['cls_mean', 'cls_cov_inv']}
            self.organ_head.load_state_dict(filtered, strict=False)
        else:
            self.organ_head.load_state_dict(ckpt, strict=False)

        # Charger OOD calibration
        if ckpt.get('cls_mean') is not None:
            self.organ_head.cls_mean = ckpt['cls_mean']
            self.organ_head.cls_cov_inv = ckpt['cls_cov_inv']
            self.organ_head.mahalanobis_threshold = ckpt.get('mahalanobis_threshold')
            self.organ_head.ood_fitted = True

        acc = ckpt.get('val_acc', 'N/A')
        print(f"  ✓ OrganHead chargé (Acc: {acc})")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str = "models/checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> "OptimusGateMultiFamily":
        """
        Charge le modèle avec les 5 familles.

        Attend les fichiers:
            - organ_head_best.pth
            - hovernet_glandular_best.pth
            - hovernet_digestive_best.pth
            - hovernet_urologic_best.pth
            - hovernet_respiratory_best.pth
            - hovernet_epidermal_best.pth
        """
        checkpoint_dir = Path(checkpoint_dir)

        print(f"🚀 Chargement Optimus-Gate Multi-Famille...")

        model = cls(**kwargs)

        # Charger OrganHead
        organ_path = checkpoint_dir / "organ_head_best.pth"
        if organ_path.exists():
            model.load_organ_head(str(organ_path), device)
        else:
            print(f"  ⚠️ OrganHead non trouvé: {organ_path}")

        # Charger les 5 familles
        for family in FAMILIES:
            family_path = checkpoint_dir / f"hovernet_{family}_best.pth"
            if family_path.exists():
                model.load_family_checkpoint(family, str(family_path), device)
            else:
                print(f"  ⚠️ {family.capitalize()} non trouvé: {family_path}")

        model.to(device)
        model.eval()

        print(f"  ✅ {len(model.loaded_families)}/5 familles chargées")
        print(f"  📍 Device: {device}")

        return model


# Test
if __name__ == "__main__":
    print("Test OptimusGateMultiFamily...")
    print("=" * 60)

    model = OptimusGateMultiFamily()
    model.eval()

    print(f"✓ Modèle créé")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres totaux: {n_params:,}")

    # Test forward pour chaque famille
    features = torch.randn(1, 261, 1536)

    for family in FAMILIES:
        organ_logits, np_out, hv_out, nt_out, used_family = model(features, family=family)
        print(f"  {family}: NP {np_out.shape}, HV {hv_out.shape}")

    # Test auto-routage
    result = model.predict(features)
    print(f"\n✓ Auto-routage: {result.organ.organ_name} → {result.family}")
    print(f"  HV fiable: {result.family_hv_reliable}")
    print(f"  Niveau: {result.confidence_level.value}")

    print("\n" + model.generate_report(result))

    print("\n" + "=" * 60)
    print("✅ Tests passés!")
