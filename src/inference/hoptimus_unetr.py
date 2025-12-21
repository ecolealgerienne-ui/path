#!/usr/bin/env python3
"""
Inférence H-optimus-0 + UNETR pour la démo Gradio.

Pipeline: Image → H-optimus-0 (features) → UNETR → Segmentation cellulaire
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional
from torchvision import transforms

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


class HOptimusUNETRInference:
    """
    Inférence avec H-optimus-0 + UNETR entraîné.

    Usage:
        model = HOptimusUNETRInference("models/checkpoints/unetr_best.pth")
        result = model.predict(image)
        vis = model.visualize(image, result)
    """

    def __init__(
        self,
        unetr_checkpoint: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.img_size = 224

        print(f"Chargement H-optimus-0 + UNETR sur {device}...")

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

        # Hook pour extraire les features intermédiaires
        self.features = {}
        self.layer_indices = [5, 11, 17, 23]  # Couches 6, 12, 18, 24
        self._register_hooks()

        # Charger le décodeur UNETR
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.models.unetr_decoder import UNETRDecoder

        self.decoder = UNETRDecoder(
            embed_dim=1536,
            patch_size=14,
            img_size=224,
            n_classes=5,
        )

        # Charger les poids entraînés
        checkpoint = torch.load(unetr_checkpoint, map_location=device)
        self.decoder.load_state_dict(checkpoint['model_state_dict'])
        self.decoder.eval()
        self.decoder.to(device)

        best_dice = checkpoint.get('best_dice', None)
        if best_dice is not None:
            print(f"✅ Modèle chargé (Dice: {best_dice:.4f})")
        else:
            print("✅ Modèle chargé")

        # Créer le transform (DOIT être identique à extract_features.py)
        self.transform = create_hoptimus_transform()

    def _register_hooks(self):
        """Enregistre des hooks pour extraire les features."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for idx in self.layer_indices:
            layer = self.backbone.blocks[idx]
            layer.register_forward_hook(get_hook(f'layer_{idx}'))

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

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict:
        """
        Prédit la segmentation cellulaire.

        Args:
            image: Image RGB (H, W, 3)

        Returns:
            Dict avec:
                - np_mask: Masque binaire noyaux (H, W)
                - nt_mask: Masque types cellulaires (H, W) valeurs 0-5
                - np_prob: Probabilité noyau (H, W)
                - nt_probs: Probabilités par type (H, W, 5)
                - counts: Dict des comptages par type
        """
        original_size = image.shape[:2]

        # Prétraitement
        x = self.preprocess(image)

        # Forward backbone
        _ = self.backbone.forward_features(x)

        # Récupérer features
        f6 = self.features['layer_5']
        f12 = self.features['layer_11']
        f18 = self.features['layer_17']
        f24 = self.features['layer_23']

        # Forward décodeur
        np_logits, hv_maps, nt_logits = self.decoder(f6, f12, f18, f24)

        # Post-traitement
        np_prob = F.softmax(np_logits, dim=1)[0, 1]  # Probabilité noyau
        np_mask = (np_prob > 0.5).cpu().numpy()

        nt_probs = F.softmax(nt_logits, dim=1)[0]  # (5, H, W)
        nt_mask = nt_probs.argmax(dim=0).cpu().numpy()  # (H, W)

        # Appliquer le masque NP au typage
        nt_mask_filtered = nt_mask.copy()
        nt_mask_filtered[~np_mask] = -1  # Background

        # Redimensionner à la taille originale
        if original_size != (self.img_size, self.img_size):
            np_mask = cv2.resize(np_mask.astype(np.uint8),
                               (original_size[1], original_size[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            nt_mask_filtered = cv2.resize(nt_mask_filtered.astype(np.int8),
                                         (original_size[1], original_size[0]),
                                         interpolation=cv2.INTER_NEAREST)

        # Compter les cellules par type
        type_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
        counts = {}
        for i, name in enumerate(type_names):
            counts[name] = int(np.sum(nt_mask_filtered == i))

        return {
            'np_mask': np_mask,
            'nt_mask': nt_mask_filtered,
            'np_prob': np_prob.cpu().numpy(),
            'counts': counts,
        }

    def visualize(
        self,
        image: np.ndarray,
        result: Dict,
        show_contours: bool = True,
        show_overlay: bool = True,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Crée une visualisation avec overlay coloré."""
        vis = image.copy()

        if show_overlay:
            overlay = np.zeros_like(image)
            nt_mask = result['nt_mask']

            type_names = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
            for i, name in enumerate(type_names):
                mask = nt_mask == i
                if np.any(mask):
                    color = CELL_COLORS[name]
                    overlay[mask] = color

            # Blend
            mask_any = nt_mask >= 0
            vis[mask_any] = cv2.addWeighted(
                vis[mask_any], 1 - alpha,
                overlay[mask_any], alpha, 0
            )

        if show_contours:
            np_mask = result['np_mask'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(np_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)

        return vis

    def generate_report(self, result: Dict) -> str:
        """Génère un rapport textuel."""
        counts = result['counts']
        total = sum(counts.values())

        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📊 ANALYSE CELLULAIRE",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"Total pixels cellulaires: {total:,}",
            "",
        ]

        for name, count in counts.items():
            pct = (count / total * 100) if total > 0 else 0
            emoji = CELL_EMOJIS.get(name, '⚪')
            bar_len = int(pct / 5)  # 20 chars max
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"{emoji} {name:12}: {bar} {pct:5.1f}%")

        lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ])

        return "\n".join(lines)
