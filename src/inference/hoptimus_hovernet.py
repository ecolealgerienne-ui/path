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
from typing import Dict, Tuple
from scipy import ndimage

# Normalisation H-optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

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
    ):
        self.device = device
        self.img_size = 224
        self.np_threshold = np_threshold

        print(f"Chargement H-optimus-0 + HoVer-Net sur {device}...")

        # Charger H-optimus-0 backbone
        import timm
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

        # Charger le décodeur HoVer-Net
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.models.hovernet_decoder import HoVerNetDecoder

        self.decoder = HoVerNetDecoder(embed_dim=1536, n_classes=5)

        # Charger les poids entraînés
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.decoder.load_state_dict(checkpoint['model_state_dict'])
        self.decoder.eval()
        self.decoder.to(device)

        best_dice = checkpoint.get('best_dice', None)
        if best_dice is not None:
            print(f"✅ Modèle chargé (Dice: {best_dice:.4f})")
        else:
            print("✅ Modèle chargé")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Prétraitement de l'image pour H-optimus-0."""
        # Redimensionner à 224x224
        if image.shape[:2] != (self.img_size, self.img_size):
            image = cv2.resize(image, (self.img_size, self.img_size))

        # Normaliser [0, 255] -> [0, 1] -> normalized
        img = image.astype(np.float32) / 255.0

        # Appliquer la normalisation H-optimus-0
        for c in range(3):
            img[:, :, c] = (img[:, :, c] - HOPTIMUS_MEAN[c]) / HOPTIMUS_STD[c]

        # HWC -> CHW -> BCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        return torch.from_numpy(img).to(self.device)

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

        # Forward backbone - obtenir les tokens
        features = self.backbone.forward_features(x)  # (1, 261, 1536)

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

        return {
            'np_mask': instance_map > 0,
            'instance_map': instance_map,
            'nt_mask': nt_mask,
            'counts': counts,
            'n_cells': instance_map.max(),
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

        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📊 ANALYSE CELLULAIRE (HoVer-Net)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"Cellules détectées: {n_cells}",
            "",
        ]

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
