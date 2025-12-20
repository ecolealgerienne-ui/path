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
from torchvision import transforms

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

# Normalisation H-optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


def create_hoptimus_transform():
    """
    Crée la transformation EXACTE utilisée pendant l'extraction des features.

    IMPORTANT: Doit être identique à scripts/preprocessing/extract_features.py
    pour garantir la cohérence entre entraînement et inférence.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])

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

        # 1. Charger H-optimus-0 backbone
        import timm
        print("  ⏳ Chargement H-optimus-0 (1.1B params)...")
        self.backbone = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        self.backbone.eval()
        self.backbone.to(device)

        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  ✓ H-optimus-0 chargé")

        # 2. Charger Optimus-Gate Multi-Famille
        self.model = OptimusGateMultiFamily.from_pretrained(
            checkpoint_dir=checkpoint_dir,
            device=device,
        )

        # 3. Créer le transform (DOIT être identique à extract_features.py)
        self.transform = create_hoptimus_transform()

        print("  ✅ Optimus-Gate Multi-Famille prêt!")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Prétraitement de l'image pour H-optimus-0.

        IMPORTANT: Utilise EXACTEMENT le même pipeline torchvision que
        l'extraction des features (scripts/preprocessing/extract_features.py)
        pour garantir la cohérence entre entraînement et inférence.

        Gère automatiquement:
        - uint8 [0, 255] → via ToPILImage + ToTensor + Normalize
        - float [0, 1] → converti en uint8 d'abord
        - float [0, 255] → converti en uint8 d'abord
        """
        # Convertir en uint8 [0, 255] pour ToPILImage
        # (ToPILImage attend uint8 ou float [0,1], pas float [0,255])
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                # float [0, 1] → uint8 [0, 255]
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                # float [0, 255] → uint8 [0, 255]
                image = image.clip(0, 255).astype(np.uint8)

        # Appliquer le transform torchvision (identique à l'entraînement)
        tensor = self.transform(image)

        # Ajouter dimension batch
        return tensor.unsqueeze(0).to(self.device)

    def post_process_hv(
        self,
        np_pred: np.ndarray,
        hv_pred: np.ndarray,
    ) -> np.ndarray:
        """Watershed sur les cartes HV pour séparer les instances."""
        binary_mask = np_pred > self.np_threshold

        if not binary_mask.any():
            return np.zeros_like(np_pred, dtype=np.int32)

        h_grad = np.abs(cv2.Sobel(hv_pred[0], cv2.CV_64F, 1, 0, ksize=3))
        v_grad = np.abs(cv2.Sobel(hv_pred[1], cv2.CV_64F, 0, 1, ksize=3))

        edge = h_grad + v_grad
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

        markers = np_pred.copy()
        markers[edge > 0.3] = 0
        markers = (markers > 0.7).astype(np.uint8)

        dist = ndimage.distance_transform_edt(binary_mask)
        markers = ndimage.label(markers * (dist > 2))[0]

        if markers.max() > 0:
            from skimage.segmentation import watershed
            instance_map = watershed(-dist, markers, mask=binary_mask)
        else:
            instance_map = ndimage.label(binary_mask)[0]

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
        features = self.backbone.forward_features(x)

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
            nt_mask[inst_mask] = type_votes.argmax()

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

        # Compter
        counts = {name: 0 for name in CELL_TYPES}
        for inst_id in range(1, instance_map.max() + 1):
            inst_mask = instance_map == inst_id
            if inst_mask.sum() == 0:
                continue
            inst_type = nt_mask[inst_mask][0]
            if 0 <= inst_type < 5:
                counts[CELL_TYPES[inst_type]] += 1

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
                if 0 <= inst_type < 5:
                    color = CELL_COLORS[CELL_TYPES[inst_type]]
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
