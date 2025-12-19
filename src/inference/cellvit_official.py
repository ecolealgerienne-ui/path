#!/usr/bin/env python3
"""
Wrapper pour CellViT officiel (TIO-IKIM).

Utilise le code original pour charger et exécuter CellViT-256.
POC: Cette approche utilise le dépôt cloné pour garantir la compatibilité.

Usage:
    from src.inference.cellvit_official import CellViTOfficial

    model = CellViTOfficial("models/pretrained/CellViT-256.pth")
    result = model.predict(image)
"""

import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

# Ajouter le dépôt CellViT cloné au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
CELLVIT_REPO = PROJECT_ROOT / "CellViT"

if CELLVIT_REPO.exists():
    sys.path.insert(0, str(CELLVIT_REPO))


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


class CellViTOfficial:
    """
    Wrapper pour le modèle CellViT officiel.

    Charge le modèle depuis le dépôt TIO-IKIM/CellViT cloné.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
    ):
        """
        Args:
            checkpoint_path: Chemin vers CellViT-256.pth
            device: 'cuda' ou 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)

        if not CELLVIT_REPO.exists():
            raise RuntimeError(
                f"CellViT repo non trouvé: {CELLVIT_REPO}\n"
                "Clonez-le avec: git clone https://github.com/TIO-IKIM/CellViT.git"
            )

        self.model = self._load_model()

        # Normalisation ImageNet standard
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _load_model(self):
        """Charge le modèle CellViT depuis le repo officiel."""
        print(f"Chargement CellViT depuis {self.checkpoint_path}...")

        try:
            # Importer depuis le repo cloné
            from cell_segmentation.inference.inference_cellvit_experiment_pannuke import (
                InferenceCellViT
            )

            # Utiliser le loader officiel
            # Note: InferenceCellViT attend un dossier d'expérience
            # On va charger le modèle directement
            raise ImportError("Utiliser chargement direct")

        except ImportError:
            # Fallback: charger le modèle directement
            return self._load_model_direct()

    def _load_model_direct(self):
        """Charge le modèle directement depuis le checkpoint."""
        print("Chargement direct du checkpoint...")

        try:
            # Importer l'architecture depuis le repo
            from cell_segmentation.trainer.trainer_cellvit import CellViT256
        except ImportError:
            # Essayer un autre chemin
            try:
                from cellvit.models.cellvit import CellViT256
            except ImportError:
                print("Import CellViT échoué, utilisation architecture locale")
                return self._load_with_local_architecture()

        # Charger le checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Créer le modèle
        model = CellViT256(
            num_nuclei_classes=6,
            num_tissue_classes=0,  # Pas de classification tissu
        )

        # Charger les poids
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        model.eval()

        print(f"Modèle chargé sur {self.device}")
        return model

    def _load_with_local_architecture(self):
        """
        Charge avec une architecture reconstruite localement.
        Fallback si l'import CellViT échoue.
        """
        # Import absolu pour éviter les erreurs de package
        try:
            from src.inference.cellvit256_model import load_cellvit256_from_checkpoint
            return load_cellvit256_from_checkpoint(
                str(self.checkpoint_path),
                self.device
            )
        except ImportError:
            # Dernier fallback: charger partiellement
            return self._load_partial()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Prétraitement de l'image.

        Args:
            image: Image RGB (H, W, 3) uint8

        Returns:
            Tensor (1, 3, 256, 256) normalisé
        """
        # Resize à 256x256 si nécessaire
        if image.shape[0] != 256 or image.shape[1] != 256:
            image = cv2.resize(image, (256, 256))

        # Normalisation
        img = image.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # (H, W, C) -> (C, H, W)
        img = img.transpose(2, 0, 1)

        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, image: np.ndarray) -> Dict:
        """
        Prédit la segmentation cellulaire.

        Args:
            image: Image RGB (H, W, 3) uint8

        Returns:
            Dict avec masques et détections
        """
        original_size = image.shape[:2]

        # Prétraitement
        tensor = self.preprocess(image)

        # Inférence
        with torch.no_grad():
            outputs = self.model(tensor)

        # Post-traitement
        return self.postprocess(outputs, original_size)

    def postprocess(self, outputs: Dict, original_size: Tuple[int, int]) -> Dict:
        """Post-traitement des sorties."""

        # Extraire les prédictions selon la structure CellViT
        if isinstance(outputs, dict):
            np_logits = outputs.get('nuclei_binary_map', outputs.get('np'))
            hv_map = outputs.get('hv_map', outputs.get('hv'))
            type_logits = outputs.get('nuclei_type_maps', outputs.get('type'))
        else:
            # Si tuple/list
            np_logits, hv_map, type_logits = outputs[:3]

        # Convertir en numpy
        np_probs = torch.softmax(np_logits, dim=1).cpu().numpy()[0]
        np_mask = np_probs[1] > 0.5  # Canal 1 = noyau

        hv = hv_map.cpu().numpy()[0].transpose(1, 2, 0)

        type_probs = torch.softmax(type_logits, dim=1).cpu().numpy()[0]
        type_mask = type_probs.argmax(axis=0)

        # Resize si nécessaire
        if original_size != (256, 256):
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

        # Détecter les cellules
        cells = self._detect_cells(np_mask, hv, type_mask)
        counts = self._count_cells(cells)

        return {
            'np_mask': np_mask,
            'hv_map': hv,
            'type_mask': type_mask,
            'cells': cells,
            'counts': counts,
            'np_probs': np_probs,
            'type_probs': type_probs,
        }

    def _detect_cells(
        self,
        np_mask: np.ndarray,
        hv_map: np.ndarray,
        type_mask: np.ndarray,
        min_area: int = 10
    ) -> List[Dict]:
        """Détecte les cellules individuelles."""
        cells = []

        np_mask_uint8 = (np_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            np_mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Type de la cellule
            mask = np.zeros(type_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            cell_types = type_mask[mask == 1]
            if len(cell_types) == 0:
                continue
            cell_type = int(np.bincount(cell_types).argmax())

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
        """Visualise les résultats."""
        vis = image.copy()

        if show_types:
            overlay = np.zeros_like(vis)
            for cell in result['cells']:
                color = CELL_COLORS.get(cell['type'], (128, 128, 128))
                cv2.drawContours(overlay, [cell['contour']], -1, color, -1)

            mask = overlay.sum(axis=2) > 0
            vis[mask] = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)[mask]

        if show_contours:
            for cell in result['cells']:
                color = CELL_COLORS.get(cell['type'], (255, 255, 255))
                cv2.drawContours(vis, [cell['contour']], -1, color, 2)

        return vis

    def generate_report(self, result: Dict) -> str:
        """Génère un rapport textuel."""
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

        lines.extend(["", "=" * 50])
        return "\n".join(lines)


# Test
if __name__ == "__main__":
    import sys

    checkpoint = PROJECT_ROOT / "models" / "pretrained" / "CellViT-256.pth"

    if not checkpoint.exists():
        print(f"Checkpoint non trouvé: {checkpoint}")
        sys.exit(1)

    print("Test CellViT Official Wrapper...")

    try:
        model = CellViTOfficial(str(checkpoint))

        # Image test
        img = np.random.randint(150, 220, (256, 256, 3), dtype=np.uint8)
        for _ in range(15):
            cx, cy = np.random.randint(20, 236, 2)
            cv2.circle(img, (cx, cy), np.random.randint(5, 12), (80, 40, 100), -1)

        result = model.predict(img)

        print(f"✅ Cellules détectées: {len(result['cells'])}")
        print(model.generate_report(result))

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
