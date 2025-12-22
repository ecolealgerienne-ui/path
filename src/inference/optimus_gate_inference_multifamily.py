#!/usr/bin/env python3
"""
Wrapper d'inférence pour Optimus-Gate Multi-Famille.

Pipeline: Image → H-optimus-0 → OrganHead → Router → HoVer-Net[famille]
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.optimus_gate_multifamily import (
    OptimusGateMultiFamily,
    MultiFamilyResult,
    FAMILIES,
    RELIABLE_HV_FAMILIES,
    CELL_TYPES,
)
from src.uncertainty import ConfidenceLevel

# Imports des modules centralisés (Phase 1 Refactoring)
from src.preprocessing import preprocess_image, validate_features
from src.models.loader import ModelLoader

# Couleurs pour visualisation
CELL_COLORS = {
    'Neoplastic': (255, 0, 0),
    'Inflammatory': (0, 255, 0),
    'Connective': (0, 0, 255),
    'Dead': (255, 255, 0),
    'Epithelial': (0, 255, 255),
}

CELL_EMOJIS = {
    'Neoplastic': '🔴',
    'Inflammatory': '🟢',
    'Connective': '🔵',
    'Dead': '🟡',
    'Epithelial': '🩵',
}


class OptimusGateInferenceMultiFamily:
    """
    Wrapper d'inférence Optimus-Gate avec 5 familles.

    Charge H-optimus-0 + OrganHead + 5 HoVer-Net spécialisés.

    Usage:
        model = OptimusGateInferenceMultiFamily()
        result = model.predict(image)
        vis = model.visualize(image, result)
        report = model.generate_report(result)
    """

    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        np_threshold: float = 0.5,
    ):
        self.device = device
        self.img_size = 224
        self.np_threshold = np_threshold

        print(f"🚀 Chargement Optimus-Gate Multi-Famille sur {device}...")

        # 1. Charger H-optimus-0 backbone via ModelLoader (centralisé)
        print("  ⏳ Chargement H-optimus-0 (1.1B params)...")
        self.backbone = ModelLoader.load_hoptimus0(device=device)
        print("  ✓ H-optimus-0 chargé")

        # 2. Charger Optimus-Gate Multi-Famille
        self.model = OptimusGateMultiFamily.from_pretrained(
            checkpoint_dir=checkpoint_dir,
            device=device,
        )

        print("  ✅ Optimus-Gate Multi-Famille prêt!")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les features de H-optimus-0 via forward_features().

        IMPORTANT: Utilise forward_features() qui inclut le LayerNorm final.
        Validation automatique des features pour détecter les bugs.

        Args:
            x: Tensor d'entrée (B, 3, 224, 224)

        Returns:
            Features (B, 261, 1536) - CLS token + 256 patch tokens

        Raises:
            RuntimeError: Si features corrompues (CLS std hors plage)
        """
        # forward_features() inclut le LayerNorm final
        features = self.backbone.forward_features(x)

        # Validation automatique (Phase 1 Refactoring - détection bugs)
        validation = validate_features(features)
        if not validation["valid"]:
            raise RuntimeError(
                f"❌ Features corrompues détectées!\n{validation['message']}\n\n"
                f"Cela indique un problème de preprocessing ou de chargement modèle."
            )

        return features.float()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Prétraitement de l'image pour H-optimus-0.

        Utilise le module centralisé src.preprocessing pour garantir
        la cohérence parfaite entre entraînement et inférence.

        Args:
            image: Image RGB (H, W, 3) - uint8, float [0-1], ou float [0-255]

        Returns:
            Tensor (1, 3, 224, 224) normalisé
        """
        # Utiliser la fonction centralisée (Phase 1 Refactoring)
        return preprocess_image(image, device=self.device)

    def post_process_hv(
        self,
        np_pred: np.ndarray,
        hv_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Watershed sur les cartes HV pour séparer les instances.

        Utilise peak_local_max avec paramètres GLOBALEMENT optimaux:
        - edge_threshold: 0.3 (CONSERVATIVE - batch diagnostic sur 10 images)
        - dist_threshold: 2 (min_distance pour peak_local_max)
        - min_size: 10 pixels (filtrage post-processing)

        Batch diagnostic results (10 images, glandular family):
        - CONSERVATIVE (0.3, 2, 10): Error=38, Detection=141% ✅ BEST
        - vs OLD (0.2, 1, 10): Error=161, Detection=512% ❌ (5x over-seg)

        Improvement: 76% error reduction vs single-image optimized params.
        """
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        # Binary mask
        binary_mask = (np_pred > self.np_threshold).astype(np.uint8)

        if not binary_mask.any():
            return np.zeros_like(np_pred, dtype=np.int32)

        # 1. Compute gradient magnitude from HV maps
        h_grad = cv2.Sobel(hv_pred[0], cv2.CV_64F, 1, 0, ksize=3)
        v_grad = cv2.Sobel(hv_pred[1], cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(h_grad**2 + v_grad**2)

        # 2. Threshold to get edges (GLOBALLY OPTIMAL - batch diagnostic)
        edge_threshold = 0.3  # CONSERVATIVE: 76% error reduction vs edge=0.2
        edges = gradient > edge_threshold

        # 3. Distance transform on INVERTED edges
        dist = ndimage.distance_transform_edt(~edges)

        # 4. Find local maxima as markers (GLOBALLY OPTIMAL - batch diagnostic)
        dist_threshold = 2  # CONSERVATIVE: prevents over-segmentation
        local_max = peak_local_max(
            dist,
            min_distance=dist_threshold,
            labels=binary_mask.astype(int),
            exclude_border=False,
        )

        # 5. Create markers from local maxima
        markers = np.zeros_like(binary_mask, dtype=int)
        if len(local_max) > 0:
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

        # 6. Watershed segmentation
        if markers.max() > 0:
            instance_map = watershed(-dist, markers, mask=binary_mask)
        else:
            # Fallback si aucun marker trouvé
            instance_map = ndimage.label(binary_mask)[0]

        # 7. Remove small instances (min_size optimisé)
        min_size = 10
        for inst_id in range(1, instance_map.max() + 1):
            if (instance_map == inst_id).sum() < min_size:
                instance_map[instance_map == inst_id] = 0

        # 8. Re-label to remove gaps
        instance_map, _ = ndimage.label(instance_map > 0)

        return instance_map

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        force_family: Optional[str] = None,
    ) -> Dict:
        """
        Prédit avec routage automatique vers la famille appropriée.

        Args:
            image: Image RGB (H, W, 3)
            force_family: Forcer une famille spécifique (optionnel)

        Returns:
            Dict avec organ, family, cellules, etc.
        """
        original_size = image.shape[:2]

        x = self.preprocess(image)
        # Extraire features via forward_features() (avec LayerNorm final)
        features = self.extract_features(x)

        result = self.model.predict(features, force_family=force_family)

        # Instance segmentation
        hv_pred = result.hv_map
        np_prob = result.np_mask.astype(np.float32)
        instance_map = self.post_process_hv(np_prob, hv_pred)

        # Type par instance
        nt_mask = np.zeros_like(instance_map, dtype=np.int32) - 1
        type_probs = result.type_probs

        for inst_id in range(1, instance_map.max() + 1):
            inst_mask = instance_map == inst_id
            if inst_mask.sum() == 0:
                continue

            type_votes = np.zeros(5)
            for t in range(5):
                type_votes[t] = type_probs[t][inst_mask].mean()
            # CORRECTIF: Model outputs [0-4], PanNuke labels are [1-5] → +1 REQUIRED
            nt_mask[inst_mask] = type_votes.argmax() + 1

        # Resize si nécessaire
        if original_size != (self.img_size, self.img_size):
            instance_map = cv2.resize(
                instance_map.astype(np.float32),
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
            nt_mask = cv2.resize(
                nt_mask.astype(np.float32),
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        # Compter (mode sur les valeurs valides, cohérent avec morphometry.py)
        counts = {name: 0 for name in CELL_TYPES}
        for inst_id in range(1, instance_map.max() + 1):
            inst_mask = instance_map == inst_id
            if inst_mask.sum() == 0:
                continue
            # Utiliser le mode des valeurs valides (>= 0) pour robustesse
            types_in_inst = nt_mask[inst_mask]
            types_valid = types_in_inst[types_in_inst >= 0]
            if len(types_valid) > 0:
                inst_type = int(np.bincount(types_valid).argmax())
                # inst_type est dans [1-5] après +1, convertir vers [0-4] pour indexer CELL_TYPES
                if 1 <= inst_type <= 5:
                    counts[CELL_TYPES[inst_type - 1]] += 1

        return {
            'organ': result.organ,
            'family': result.family,
            'family_hv_reliable': result.family_hv_reliable,
            'np_mask': instance_map > 0,
            'instance_map': instance_map,
            'nt_mask': nt_mask,
            'counts': counts,
            'n_cells': instance_map.max(),
            'confidence_level': result.confidence_level,
            'is_ood': result.is_ood,
            'ood_score_global': result.ood_score_global,
            'uncertainty': result.uncertainty,
            'multifamily_result': result,
        }

    def visualize(
        self,
        image: np.ndarray,
        result: Dict,
        show_contours: bool = True,
        show_overlay: bool = True,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Visualisation avec overlay coloré."""
        vis = image.copy()

        if show_overlay:
            overlay = np.zeros_like(image)
            nt_mask = result['nt_mask']
            instance_map = result['instance_map']

            for inst_id in range(1, instance_map.max() + 1):
                inst_mask = instance_map == inst_id
                if not inst_mask.any():
                    continue

                inst_type = nt_mask[inst_mask][0]
                # inst_type est dans [1-5] après +1, convertir vers [0-4] pour indexer CELL_TYPES
                if 1 <= inst_type <= 5:
                    color = CELL_COLORS[CELL_TYPES[inst_type - 1]]
                    overlay[inst_mask] = color

            mask_any = instance_map > 0
            if mask_any.any():
                vis[mask_any] = cv2.addWeighted(
                    vis[mask_any], 1 - alpha,
                    overlay[mask_any], alpha, 0
                )

        if show_contours:
            instance_map = result['instance_map']
            for inst_id in range(1, instance_map.max() + 1):
                inst_mask = (instance_map == inst_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)

        return vis

    def visualize_uncertainty(
        self,
        image: np.ndarray,
        result: Dict,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Carte d'incertitude (rouge=incertain, vert=fiable)."""
        uncertainty = result.get('uncertainty')
        if uncertainty is None or uncertainty.uncertainty_map is None:
            return image.copy()

        uncertainty_map = uncertainty.uncertainty_map
        if uncertainty_map.shape[:2] != image.shape[:2]:
            uncertainty_map = cv2.resize(
                uncertainty_map,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        heatmap = np.zeros_like(image)
        heatmap[..., 0] = (uncertainty_map * 255).astype(np.uint8)
        heatmap[..., 1] = ((1 - uncertainty_map) * 255).astype(np.uint8)

        vis = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return vis

    def generate_report(self, result: Dict) -> str:
        """Génère un rapport textuel complet."""
        organ = result['organ']
        family = result['family']
        family_reliable = result['family_hv_reliable']
        counts = result['counts']
        n_cells = result['n_cells']
        total = sum(counts.values())
        confidence_level = result['confidence_level']
        is_ood = result['is_ood']

        level_emoji = {
            ConfidenceLevel.FIABLE: "✅",
            ConfidenceLevel.A_REVOIR: "⚠️",
            ConfidenceLevel.HORS_DOMAINE: "🚫",
        }
        emoji = level_emoji.get(confidence_level, "❓")
        hv_status = "✅ Fiable" if family_reliable else "⚠️ Vérifier"

        lines = [
            "╔════════════════════════════════════════════════════════════════╗",
            "║         RAPPORT OPTIMUS-GATE (5 Familles)                      ║",
            "╠════════════════════════════════════════════════════════════════╣",
            f"║ {emoji} NIVEAU: {confidence_level.value.upper():46} ║",
            "╠════════════════════════════════════════════════════════════════╣",
            "║ 🔬 ROUTAGE AUTOMATIQUE                                         ║",
            f"║    Organe: {organ.organ_name:20} ({organ.confidence:.1%})          ║",
            f"║    → Famille: {family.upper():16}                              ║",
            f"║    → Séparation HV: {hv_status:14}                            ║",
            "╠════════════════════════════════════════════════════════════════╣",
            "║ 🔎 ANALYSE CELLULAIRE                                          ║",
            f"║    Cellules: {n_cells:4}                                             ║",
        ]

        if total > 0:
            for name in CELL_TYPES:
                count = counts.get(name, 0)
                pct = count / total * 100
                emoji = CELL_EMOJIS.get(name, '•')
                bar_len = int(pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"║    {emoji} {name:12}: {bar} {count:3} ({pct:5.1f}%)  ║")

        lines.extend([
            "╠════════════════════════════════════════════════════════════════╣",
            "║ 🛡️ SÉCURITÉ                                                     ║",
            f"║    OOD: {result['ood_score_global']:.3f} {'🚫 HORS DOMAINE' if is_ood else '✓ OK':18}             ║",
            "╚════════════════════════════════════════════════════════════════╝",
        ])

        return "\n".join(lines)
