#!/usr/bin/env python3
"""
Inférence H-optimus-0 + HoVer-Net decoder pour la démo Gradio.

Pipeline: Image → H-optimus-0 (features finales) → HoVer-Net → Segmentation cellulaire

Architecture basée sur la littérature:
- H-optimus-0: Foundation model Bioptimus (1.1B params)
- HoVer-Net decoder: 3 branches (NP, HV, NT) pour instance segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy import ndimage

# Imports des modules centralisés (Phase 1 Refactoring)
from src.preprocessing import preprocess_image, validate_features
from src.models.loader import ModelLoader

# Importer l'estimateur d'incertitude
try:
    from src.uncertainty import UncertaintyEstimator, UncertaintyResult
    UNCERTAINTY_AVAILABLE = True
except ImportError:
    UNCERTAINTY_AVAILABLE = False

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


class HOptimusHoVerNetInference:
    """
    Inférence avec H-optimus-0 + HoVer-Net decoder.

    Usage:
        model = HOptimusHoVerNetInference("models/checkpoints/hovernet_best.pth")
        result = model.predict(image)
        vis = model.visualize(image, result)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        np_threshold: float = 0.5,
        enable_uncertainty: bool = True,
    ):
        self.device = device
        self.img_size = 224
        self.np_threshold = np_threshold
        self.enable_uncertainty = enable_uncertainty and UNCERTAINTY_AVAILABLE

        # Estimateur d'incertitude
        if self.enable_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator()
        else:
            self.uncertainty_estimator = None

        print(f"Chargement H-optimus-0 + HoVer-Net sur {device}...")

        # Charger H-optimus-0 backbone via ModelLoader (centralisé)
        self.backbone = ModelLoader.load_hoptimus0(device=device)

        # Charger le décodeur HoVer-Net via ModelLoader (centralisé)
        self.decoder = ModelLoader.load_hovernet(
            checkpoint_path=Path(checkpoint_path),
            device=device,
            num_classes=6  # BG + 5 types cellulaires
        )

        print(f"✅ Modèles chargés avec succès")

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
        Post-processing HoVer-Net: watershed sur les cartes HV.

        Args:
            np_pred: (H, W) probabilité noyau
            hv_pred: (2, H, W) cartes H et V

        Returns:
            instance_map: (H, W) labels d'instances
        """
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
        markers[edge > 0.3] = 0  # Supprimer les bords
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
        Prédit la segmentation cellulaire.

        Args:
            image: Image RGB (H, W, 3)

        Returns:
            Dict avec:
                - np_mask: Masque binaire noyaux (H, W)
                - instance_map: Labels d'instances (H, W)
                - nt_mask: Type par instance (H, W)
                - counts: Dict des comptages par type
                - n_cells: Nombre total de cellules
        """
        original_size = image.shape[:2]

        # Prétraitement
        x = self.preprocess(image)

        # Forward backbone - obtenir les tokens (avec LayerNorm final)
        features = self.extract_features(x)  # (1, 261, 1536)

        # Forward décodeur HoVer-Net
        np_logits, hv_maps, nt_logits = self.decoder(features)

        # Post-traitement
        np_prob = F.softmax(np_logits, dim=1)[0, 1].cpu().numpy()  # (224, 224)
        hv_pred = hv_maps[0].cpu().numpy()  # (2, 224, 224)
        nt_probs = F.softmax(nt_logits, dim=1)[0].cpu().numpy()  # (5, 224, 224)

        # Instance segmentation via watershed
        instance_map = self.post_process_hv(np_prob, hv_pred)

        # Type par instance (vote majoritaire)
        nt_mask = np.zeros_like(instance_map, dtype=np.int32) - 1  # -1 = background
        n_instances = instance_map.max()

        for inst_id in range(1, n_instances + 1):
            inst_mask = instance_map == inst_id
            if inst_mask.sum() == 0:
                continue

            # Vote majoritaire pour le type
            type_votes = np.zeros(5)
            for t in range(5):
                type_votes[t] = nt_probs[t][inst_mask].mean()

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

        # Compter les cellules par type
        counts = {name: 0 for name in TYPE_NAMES}
        for inst_id in range(1, instance_map.max() + 1):
            inst_mask = instance_map == inst_id
            if inst_mask.sum() == 0:
                continue
            inst_type = nt_mask[inst_mask][0]
            if 0 <= inst_type < 5:
                counts[TYPE_NAMES[inst_type]] += 1

        # Estimation d'incertitude
        uncertainty_result = None
        if self.enable_uncertainty and self.uncertainty_estimator is not None:
            # Convertir en format (H, W, C) pour l'estimateur
            np_probs_hwc = np.stack([1 - np_prob, np_prob], axis=-1)  # (H, W, 2)
            nt_probs_hwc = np.transpose(nt_probs, (1, 2, 0))  # (H, W, 5)

            # Embeddings pour OOD (moyenne des patch tokens)
            embeddings = features[0, 1:257, :].mean(dim=0).cpu().numpy()  # (1536,)

            uncertainty_result = self.uncertainty_estimator.estimate(
                np_probs=np_probs_hwc,
                nt_probs=nt_probs_hwc,
                embeddings=embeddings,
                compute_map=True,
            )

        return {
            'np_mask': instance_map > 0,
            'instance_map': instance_map,
            'nt_mask': nt_mask,
            'counts': counts,
            'n_cells': instance_map.max(),
            'uncertainty': uncertainty_result,
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
                contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)

        return vis

    def generate_report(self, result: Dict) -> str:
        """Génère un rapport textuel."""
        counts = result['counts']
        n_cells = result['n_cells']
        total = sum(counts.values())
        uncertainty = result.get('uncertainty')

        lines = []

        # Section incertitude (en premier si disponible)
        if uncertainty is not None:
            lines.append(self.uncertainty_estimator.generate_report(uncertainty))
            lines.append("")

        # Section analyse cellulaire
        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📊 ANALYSE CELLULAIRE (HoVer-Net)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"Cellules détectées: {n_cells}",
            "",
        ])

        if total > 0:
            for name, count in counts.items():
                pct = (count / total * 100)
                emoji = CELL_EMOJIS.get(name, '⚪')
                bar_len = int(pct / 5)  # 20 chars max
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"{emoji} {name:12}: {bar} {count:3} ({pct:5.1f}%)")
        else:
            lines.append("Aucune cellule détectée")

        lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ])

        return "\n".join(lines)

    def visualize_uncertainty(
        self,
        image: np.ndarray,
        result: Dict,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Visualise la carte d'incertitude.

        Rouge = haute incertitude
        Vert = basse incertitude
        """
        uncertainty = result.get('uncertainty')
        if uncertainty is None or uncertainty.uncertainty_map is None:
            return image.copy()

        # Redimensionner la carte si nécessaire
        uncertainty_map = uncertainty.uncertainty_map
        if uncertainty_map.shape[:2] != image.shape[:2]:
            uncertainty_map = cv2.resize(
                uncertainty_map,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Créer heatmap (rouge = incertain, vert = confiant)
        heatmap = np.zeros_like(image)
        heatmap[..., 0] = (uncertainty_map * 255).astype(np.uint8)  # Rouge
        heatmap[..., 1] = ((1 - uncertainty_map) * 255).astype(np.uint8)  # Vert

        # Blend avec l'image
        vis = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return vis
