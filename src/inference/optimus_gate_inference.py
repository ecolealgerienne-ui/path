#!/usr/bin/env python3
"""
Wrapper d'inférence pour Optimus-Gate.

Pipeline complet: Image → H-optimus-0 → OptimusGate → Résultats

Combine:
- Classification d'organe (OrganHead)
- Segmentation cellulaire (HoVer-Net)
- Triple sécurité OOD
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional
from scipy import ndimage
from torchvision import transforms

# Imports locaux
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.optimus_gate import OptimusGate, OptimusGateResult
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

# Couleurs pour visualisation (RGB)
CELL_COLORS = {
    'Neoplastic': (255, 0, 0),      # Rouge
    'Inflammatory': (0, 255, 0),     # Vert
    'Connective': (0, 0, 255),       # Bleu
    'Dead': (255, 255, 0),           # Jaune
    'Epithelial': (0, 255, 255),     # Cyan
}

CELL_EMOJIS = {
    'Neoplastic': '🔴',
    'Inflammatory': '🟢',
    'Connective': '🔵',
    'Dead': '🟡',
    'Epithelial': '🩵',
}

TYPE_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']


class OptimusGateInference:
    """
    Wrapper d'inférence pour Optimus-Gate.

    Combine H-optimus-0 (backbone) + OptimusGate (OrganHead + HoVer-Net).

    Usage:
        model = OptimusGateInference(
            hovernet_path="models/checkpoints/hovernet_best.pth",
            organ_head_path="models/checkpoints/organ_head_best.pth"
        )
        result = model.predict(image)
        vis = model.visualize(image, result)
        report = model.generate_report(result)
    """

    def __init__(
        self,
        hovernet_path: str,
        organ_head_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        np_threshold: float = 0.5,
    ):
        self.device = device
        self.img_size = 224
        self.np_threshold = np_threshold

        print(f"🚀 Chargement Optimus-Gate sur {device}...")

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

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  ✓ H-optimus-0 chargé")

        # 2. Charger OptimusGate (OrganHead + HoVer-Net)
        self.model = OptimusGate.from_pretrained(
            hovernet_path=hovernet_path,
            organ_head_path=organ_head_path,
            device=device,
        )

        # 3. Créer le transform (DOIT être identique à extract_features.py)
        self.transform = create_hoptimus_transform()

        print("  ✅ Optimus-Gate prêt!")

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
        """Post-processing HoVer-Net: watershed sur les cartes HV."""
        # Masque binaire
        binary_mask = np_pred > self.np_threshold

        if not binary_mask.any():
            return np.zeros_like(np_pred, dtype=np.int32)

        # Gradient des cartes HV
        h_grad = np.abs(cv2.Sobel(hv_pred[0], cv2.CV_64F, 1, 0, ksize=3))
        v_grad = np.abs(cv2.Sobel(hv_pred[1], cv2.CV_64F, 0, 1, ksize=3))

        # Combiner les gradients
        edge = h_grad + v_grad
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

        # Marqueurs pour watershed
        markers = np_pred.copy()
        markers[edge > 0.3] = 0
        markers = (markers > 0.7).astype(np.uint8)

        # Distance transform pour seeds
        dist = ndimage.distance_transform_edt(binary_mask)
        markers = ndimage.label(markers * (dist > 2))[0]

        # Watershed
        if markers.max() > 0:
            from skimage.segmentation import watershed
            instance_map = watershed(-dist, markers, mask=binary_mask)
        else:
            instance_map = ndimage.label(binary_mask)[0]

        return instance_map

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict:
        """
        Prédit la segmentation cellulaire et l'organe.

        Args:
            image: Image RGB (H, W, 3)

        Returns:
            Dict avec:
                - organ: OrganPrediction
                - np_mask: Masque binaire noyaux
                - instance_map: Labels d'instances
                - nt_mask: Type par instance
                - counts: Comptages par type
                - n_cells: Nombre total
                - confidence_level: ConfidenceLevel
                - is_ood: Booléen OOD
                - optimus_result: OptimusGateResult complet
        """
        original_size = image.shape[:2]

        # Prétraitement
        x = self.preprocess(image)

        # Forward backbone - obtenir les features
        features = self.backbone.forward_features(x)  # (1, 261, 1536)

        # Forward OptimusGate
        result = self.model.predict(features)

        # Instance segmentation via watershed sur HV maps
        hv_pred = result.hv_map  # (2, 224, 224)
        np_prob = result.np_mask.astype(np.float32)  # Binary to prob
        instance_map = self.post_process_hv(np_prob, hv_pred)

        # Type par instance
        nt_mask = np.zeros_like(instance_map, dtype=np.int32) - 1
        type_probs = result.type_probs  # (5, H, W)

        for inst_id in range(1, instance_map.max() + 1):
            inst_mask = instance_map == inst_id
            if inst_mask.sum() == 0:
                continue

            type_votes = np.zeros(5)
            for t in range(5):
                type_votes[t] = type_probs[t][inst_mask].mean()
            nt_mask[inst_mask] = type_votes.argmax()

        # Redimensionner à la taille originale
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

        # Compter les cellules
        counts = {name: 0 for name in TYPE_NAMES}
        for inst_id in range(1, instance_map.max() + 1):
            inst_mask = instance_map == inst_id
            if inst_mask.sum() == 0:
                continue
            inst_type = nt_mask[inst_mask][0]
            if 0 <= inst_type < 5:
                counts[TYPE_NAMES[inst_type]] += 1

        return {
            'organ': result.organ,
            'np_mask': instance_map > 0,
            'instance_map': instance_map,
            'nt_mask': nt_mask,
            'counts': counts,
            'n_cells': instance_map.max(),
            'confidence_level': result.confidence_level,
            'is_ood': result.is_ood,
            'ood_score_global': result.ood_score_global,
            'ood_score_local': result.ood_score_local,
            'ood_score_combined': result.ood_score_combined,
            'uncertainty': result.uncertainty,
            'optimus_result': result,
        }

    def visualize(
        self,
        image: np.ndarray,
        result: Dict,
        show_contours: bool = True,
        show_overlay: bool = True,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Crée une visualisation avec overlay coloré par instance."""
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
                    color = CELL_COLORS[TYPE_NAMES[inst_type]]
                    overlay[inst_mask] = color

            # Blend
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
        """Visualise la carte d'incertitude (rouge=incertain, vert=fiable)."""
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

        # Heatmap rouge-vert
        heatmap = np.zeros_like(image)
        heatmap[..., 0] = (uncertainty_map * 255).astype(np.uint8)  # Rouge
        heatmap[..., 1] = ((1 - uncertainty_map) * 255).astype(np.uint8)  # Vert

        vis = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return vis

    def generate_report(self, result: Dict) -> str:
        """Génère un rapport textuel complet."""
        organ = result['organ']
        counts = result['counts']
        n_cells = result['n_cells']
        total = sum(counts.values())
        confidence_level = result['confidence_level']
        is_ood = result['is_ood']

        # Emoji niveau de confiance
        level_emoji = {
            ConfidenceLevel.FIABLE: "✅",
            ConfidenceLevel.A_REVOIR: "⚠️",
            ConfidenceLevel.HORS_DOMAINE: "🚫",
        }
        emoji = level_emoji.get(confidence_level, "❓")

        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║           RAPPORT OPTIMUS-GATE                               ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ {emoji} NIVEAU: {confidence_level.value.upper():44} ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ 🔬 CONTEXTE TISSULAIRE (Flux Global)                         ║",
            f"║    Organe prédit: {organ.organ_name:20} ({organ.confidence:.1%})    ║",
            f"║    Entropie: {organ.entropy:.3f}                                        ║",
            f"║    OOD Score: {result['ood_score_global']:.3f}                                       ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ 🔎 ANALYSE CELLULAIRE (Flux Local)                           ║",
            f"║    Cellules détectées: {n_cells:4}                                    ║",
        ]

        if total > 0:
            lines.append("║                                                              ║")
            for name in TYPE_NAMES:
                count = counts.get(name, 0)
                pct = count / total * 100
                emoji = CELL_EMOJIS.get(name, '•')
                bar_len = int(pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"║    {emoji} {name:12}: {bar} {count:3} ({pct:5.1f}%) ║")

        lines.extend([
            "╠══════════════════════════════════════════════════════════════╣",
            "║ 🛡️ TRIPLE SÉCURITÉ OOD                                        ║",
            f"║    Score Global:  {result['ood_score_global']:.3f}                                    ║",
            f"║    Score Local:   {result['ood_score_local']:.3f}                                    ║",
            f"║    Score Combiné: {result['ood_score_combined']:.3f}                                    ║",
            f"║    Hors Domaine:  {'OUI 🚫' if is_ood else 'NON ✓':5}                                    ║",
            "╚══════════════════════════════════════════════════════════════╝",
        ])

        return "\n".join(lines)


# Test
if __name__ == "__main__":
    print("Test OptimusGateInference...")

    # Créer une image test
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Charger le modèle
    model = OptimusGateInference(
        hovernet_path="models/checkpoints/hovernet_best.pth",
        organ_head_path="models/checkpoints/organ_head_best.pth",
    )

    # Prédiction
    result = model.predict(test_image)
    print(f"\nOrgane: {result['organ'].organ_name}")
    print(f"Cellules: {result['n_cells']}")
    print(f"Niveau: {result['confidence_level'].value}")

    # Rapport
    print("\n" + model.generate_report(result))
