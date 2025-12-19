#!/usr/bin/env python3
"""
Détection de tissu dans les images histopathologiques.

Utilisé pour:
- Filtrer le fond blanc
- Identifier les ROIs (Regions of Interest)
- Optimiser l'extraction de tuiles

Usage:
    python scripts/preprocessing/tissue_detection.py --input image.png --output mask.png
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List
import cv2


class TissueDetector:
    """Détecteur de tissu dans les images H&E."""

    def __init__(
        self,
        threshold_method: str = "otsu",
        min_area: int = 1000,
        morph_size: int = 5,
    ):
        """
        Args:
            threshold_method: Méthode de seuillage ('otsu', 'adaptive', 'fixed')
            min_area: Surface minimale des régions (pixels)
            morph_size: Taille du kernel morphologique
        """
        self.threshold_method = threshold_method
        self.min_area = min_area
        self.morph_size = morph_size

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Détecte les régions de tissu.

        Args:
            image: Image RGB (H, W, 3)

        Returns:
            Masque binaire (H, W) où 255 = tissu
        """
        # Convertir en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Inverser (fond blanc -> noir)
        gray = 255 - gray

        # Seuillage
        if self.threshold_method == "otsu":
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.threshold_method == "adaptive":
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 5
            )
        else:  # fixed
            _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # Opérations morphologiques
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_size, self.morph_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Filtrer les petites régions
        mask = self._filter_small_regions(mask)

        return mask

    def _filter_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Supprime les régions trop petites."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        filtered_mask = np.zeros_like(mask)

        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                filtered_mask[labels == i] = 255

        return filtered_mask

    def get_bounding_boxes(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Retourne les bounding boxes des régions de tissu."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= self.min_area:
                boxes.append((x, y, w, h))

        return boxes

    def get_tissue_ratio(self, mask: np.ndarray) -> float:
        """Calcule le ratio tissu/total."""
        return np.mean(mask > 0)

    def get_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Retourne les contours des régions."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours


def visualize_detection(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """Crée une visualisation de la détection."""
    result = image.copy()

    # Overlay vert sur le tissu
    green_overlay = np.zeros_like(image)
    green_overlay[:, :, 1] = 255

    tissue_mask = mask > 0
    result[tissue_mask] = cv2.addWeighted(
        result[tissue_mask], 1 - alpha,
        green_overlay[tissue_mask], alpha, 0
    )

    # Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Détection de tissu dans les images histopathologiques"
    )
    parser.add_argument("--input", type=str, required=True, help="Image d'entrée")
    parser.add_argument("--output", type=str, help="Masque de sortie")
    parser.add_argument("--visualize", type=str, help="Image de visualisation")
    parser.add_argument("--method", type=str, default="otsu",
                        choices=["otsu", "adaptive", "fixed"])
    parser.add_argument("--min-area", type=int, default=1000, help="Surface minimale")

    args = parser.parse_args()

    detector = TissueDetector(
        threshold_method=args.method,
        min_area=args.min_area,
    )

    # Charger l'image
    img = cv2.imread(args.input)
    if img is None:
        print(f"Erreur lecture: {args.input}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Détecter le tissu
    mask = detector.detect(img_rgb)

    # Statistiques
    ratio = detector.get_tissue_ratio(mask)
    boxes = detector.get_bounding_boxes(mask)

    print(f"Ratio tissu: {ratio*100:.1f}%")
    print(f"Régions détectées: {len(boxes)}")

    # Sauvegarder le masque
    if args.output:
        cv2.imwrite(args.output, mask)
        print(f"Masque sauvegardé: {args.output}")

    # Visualisation
    if args.visualize:
        vis = visualize_detection(img_rgb, mask)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.visualize, vis_bgr)
        print(f"Visualisation sauvegardée: {args.visualize}")


if __name__ == "__main__":
    main()
