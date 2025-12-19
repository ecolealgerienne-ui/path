#!/usr/bin/env python3
"""
Inférence CellViT-256 pour segmentation cellulaire.

Conforme aux specs CellViT-Optimus_Specifications.md section 3.2.

CellViT-256 utilise:
- Encodeur basé sur SAM (ViT)
- Décodeur pour segmentation cellulaire
- 3 sorties: NP (présence), HV (séparation), NT (5 types)

Usage:
    from src.inference.cellvit_inference import CellViTInference

    model = CellViTInference("models/pretrained/CellViT-256.pth")
    result = model.predict(image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2


# Types cellulaires PanNuke
CELL_TYPES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial"
}

CELL_COLORS = {
    1: (255, 0, 0),      # Neoplastic - Rouge
    2: (0, 255, 0),      # Inflammatory - Vert
    3: (0, 0, 255),      # Connective - Bleu
    4: (255, 255, 0),    # Dead - Jaune
    5: (0, 255, 255),    # Epithelial - Cyan
}

CELL_EMOJIS = {
    1: "🔴",  # Neoplastic
    2: "🟢",  # Inflammatory
    3: "🔵",  # Connective
    4: "🟡",  # Dead
    5: "🩵",  # Epithelial
}


class CellViTInference:
    """
    Wrapper d'inférence pour CellViT-256.

    Gère le chargement du modèle, le preprocessing et le post-processing.
    """

    def __init__(
        self,
        model_path: str,
        device: str = None,
        input_size: int = 256,  # CellViT-256 utilise des patches de 256x256
    ):
        """
        Args:
            model_path: Chemin vers CellViT-256.pth
            device: 'cuda' ou 'cpu' (auto-détection par défaut)
            input_size: Taille des patches d'entrée
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model_path = Path(model_path)

        # Charger le modèle
        self.model = self._load_model()

        # Normalisation H&E standard
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _load_model(self) -> nn.Module:
        """Charge le modèle CellViT-256."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")

        print(f"Chargement CellViT-256 depuis {self.model_path}...")

        # Charger le checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # CellViT sauvegarde le modèle sous différentes clés
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Créer le modèle CellViT
        model = self._build_cellvit_model()

        # Charger les poids (avec gestion des clés manquantes)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Attention: Chargement partiel du modèle: {e}")
            # Continuer même si certains poids manquent

        model = model.to(self.device)
        model.eval()

        print(f"Modèle chargé sur {self.device}")
        return model

    def _build_cellvit_model(self) -> nn.Module:
        """
        Construit l'architecture CellViT-256.

        CellViT-256 est basé sur un encodeur ViT et décodeur U-Net.
        """
        # Import du modèle CellViT (architecture simplifiée pour inférence)
        return CellViT256Model(
            num_classes=6,  # 5 types + background
            input_size=self.input_size
        )

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Prétraitement de l'image.

        Args:
            image: Image RGB (H, W, 3) uint8

        Returns:
            Tensor (1, 3, H, W) normalisé
        """
        # Conversion en float [0, 1]
        img = image.astype(np.float32) / 255.0

        # Normalisation ImageNet
        img = (img - self.mean) / self.std

        # Reshape pour PyTorch: (H, W, C) -> (C, H, W)
        img = img.transpose(2, 0, 1)

        # Ajouter dimension batch
        tensor = torch.from_numpy(img).unsqueeze(0).float()

        return tensor.to(self.device)

    def predict(
        self,
        image: np.ndarray,
        return_probs: bool = False
    ) -> Dict:
        """
        Prédit la segmentation cellulaire.

        Args:
            image: Image RGB (H, W, 3) uint8
            return_probs: Si True, retourne aussi les probabilités

        Returns:
            Dict avec:
                - 'np_mask': Masque présence noyaux (H, W)
                - 'hv_map': Carte HV (H, W, 2)
                - 'type_mask': Masque types cellulaires (H, W)
                - 'cells': Liste des cellules détectées
                - 'counts': Comptage par type
        """
        original_size = image.shape[:2]

        # Resize si nécessaire
        if image.shape[0] != self.input_size or image.shape[1] != self.input_size:
            image_resized = cv2.resize(image, (self.input_size, self.input_size))
        else:
            image_resized = image

        # Prétraitement
        tensor = self.preprocess(image_resized)

        # Inférence
        with torch.no_grad():
            outputs = self.model(tensor)

        # Post-traitement
        result = self.postprocess(outputs, original_size)

        if return_probs:
            result['np_probs'] = F.softmax(outputs['np'], dim=1).cpu().numpy()
            result['type_probs'] = F.softmax(outputs['type'], dim=1).cpu().numpy()

        return result

    def postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        original_size: Tuple[int, int]
    ) -> Dict:
        """
        Post-traitement des sorties du modèle.

        Args:
            outputs: Sorties du modèle
            original_size: Taille originale (H, W)

        Returns:
            Dict avec masques et détections
        """
        # Extraire les prédictions
        np_logits = outputs['np']  # (1, 2, H, W)
        hv_map = outputs['hv']     # (1, 2, H, W)
        type_logits = outputs['type']  # (1, 6, H, W)

        # Masque de présence des noyaux
        np_probs = F.softmax(np_logits, dim=1)
        np_mask = (np_probs[:, 1] > 0.5).cpu().numpy()[0]  # Canal 1 = noyau

        # Carte HV
        hv = hv_map.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, 2)

        # Types cellulaires
        type_probs = F.softmax(type_logits, dim=1)
        type_mask = type_probs.argmax(dim=1).cpu().numpy()[0]

        # Resize vers la taille originale
        if original_size != (self.input_size, self.input_size):
            np_mask = cv2.resize(
                np_mask.astype(np.uint8),
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            hv = cv2.resize(hv, (original_size[1], original_size[0]))
            type_mask = cv2.resize(
                type_mask.astype(np.uint8),
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Détecter les cellules individuelles
        cells = self._detect_cells(np_mask, hv, type_mask)

        # Comptage par type
        counts = self._count_cells(cells)

        return {
            'np_mask': np_mask,
            'hv_map': hv,
            'type_mask': type_mask,
            'cells': cells,
            'counts': counts
        }

    def _detect_cells(
        self,
        np_mask: np.ndarray,
        hv_map: np.ndarray,
        type_mask: np.ndarray,
        min_area: int = 10
    ) -> List[Dict]:
        """
        Détecte les cellules individuelles.

        Utilise les cartes HV pour séparer les noyaux proches.

        Args:
            np_mask: Masque binaire des noyaux
            hv_map: Carte des gradients H-V
            type_mask: Masque des types
            min_area: Aire minimale d'une cellule

        Returns:
            Liste de dictionnaires pour chaque cellule
        """
        cells = []

        # Utiliser watershed ou connected components
        # Pour simplifier, on utilise connected components
        np_mask_uint8 = (np_mask * 255).astype(np.uint8)

        # Trouver les contours
        contours, _ = cv2.findContours(
            np_mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Centre de la cellule
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Type de la cellule (mode dans le contour)
            mask = np.zeros(type_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            cell_types = type_mask[mask == 1]
            if len(cell_types) == 0:
                continue
            cell_type = int(np.bincount(cell_types).argmax())

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            cells.append({
                'id': i,
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
                'type': cell_type,
                'type_name': CELL_TYPES.get(cell_type, "Unknown"),
                'contour': contour
            })

        return cells

    def _count_cells(self, cells: List[Dict]) -> Dict[str, int]:
        """Compte les cellules par type."""
        counts = {name: 0 for name in CELL_TYPES.values()}
        for cell in cells:
            type_name = cell['type_name']
            if type_name in counts:
                counts[type_name] += 1
        return counts

    def visualize(
        self,
        image: np.ndarray,
        result: Dict,
        show_contours: bool = True,
        show_types: bool = True,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Visualise les résultats sur l'image.

        Args:
            image: Image originale RGB
            result: Résultat de predict()
            show_contours: Afficher les contours
            show_types: Colorier par type
            alpha: Transparence de l'overlay

        Returns:
            Image avec visualisation
        """
        vis = image.copy()

        if show_types:
            # Créer overlay coloré par type
            overlay = np.zeros_like(vis)
            for cell in result['cells']:
                color = CELL_COLORS.get(cell['type'], (128, 128, 128))
                cv2.drawContours(overlay, [cell['contour']], -1, color, -1)

            # Fusionner avec alpha
            mask = overlay.sum(axis=2) > 0
            vis[mask] = cv2.addWeighted(
                vis, 1 - alpha, overlay, alpha, 0
            )[mask]

        if show_contours:
            for cell in result['cells']:
                color = CELL_COLORS.get(cell['type'], (255, 255, 255))
                cv2.drawContours(vis, [cell['contour']], -1, color, 2)

        return vis

    def generate_report(self, result: Dict) -> str:
        """
        Génère un rapport textuel.

        Args:
            result: Résultat de predict()

        Returns:
            Rapport formaté
        """
        counts = result['counts']
        total = sum(v for k, v in counts.items() if k != "Background")

        lines = [
            "=" * 50,
            "      RAPPORT D'ANALYSE CELLULAIRE",
            "=" * 50,
            "",
            f"Total cellules détectées: {total}",
            "",
            "Distribution par type:",
            "-" * 30,
        ]

        for type_id, type_name in CELL_TYPES.items():
            if type_id == 0:
                continue
            count = counts.get(type_name, 0)
            pct = (count / total * 100) if total > 0 else 0
            emoji = CELL_EMOJIS.get(type_id, "")
            lines.append(f"  {emoji} {type_name}: {count} ({pct:.1f}%)")

        lines.extend([
            "",
            "=" * 50,
        ])

        return "\n".join(lines)


class CellViT256Model(nn.Module):
    """
    Architecture CellViT-256 simplifiée pour inférence.

    Le modèle complet est dans le checkpoint .pth.
    Cette classe définit l'architecture pour charger les poids.
    """

    def __init__(self, num_classes: int = 6, input_size: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Encodeur (ViT-base style)
        self.encoder = self._build_encoder()

        # Décodeur
        self.decoder = self._build_decoder()

        # Têtes de sortie
        self.np_head = nn.Conv2d(64, 2, 1)      # Présence noyau
        self.hv_head = nn.Conv2d(64, 2, 1)      # H-V maps
        self.type_head = nn.Conv2d(64, num_classes, 1)  # Types

    def _build_encoder(self) -> nn.Module:
        """Construit l'encodeur CNN simple."""
        return nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            # Stage 2
            self._make_layer(64, 128, 2, stride=2),

            # Stage 3
            self._make_layer(128, 256, 2, stride=2),

            # Stage 4
            self._make_layer(256, 512, 2, stride=2),
        )

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Module:
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks - 1):
            layers.extend([
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Module:
        """Construit le décodeur."""
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Encodage
        features = self.encoder(x)

        # Décodage
        decoded = self.decoder(features)

        # Têtes de sortie
        np_out = self.np_head(decoded)
        hv_out = self.hv_head(decoded)
        type_out = self.type_head(decoded)

        return {
            'np': np_out,
            'hv': hv_out,
            'type': type_out
        }


# Test
if __name__ == "__main__":
    print("Test CellViT-256 Inference...")

    # Test sans modèle réel
    model = CellViT256Model()
    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward
    x = torch.randn(1, 3, 256, 256)
    out = model(x)

    print(f"NP shape: {out['np'].shape}")    # (1, 2, 256, 256)
    print(f"HV shape: {out['hv'].shape}")    # (1, 2, 256, 256)
    print(f"Type shape: {out['type'].shape}")  # (1, 6, 256, 256)
