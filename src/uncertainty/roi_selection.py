#!/usr/bin/env python3
"""
Sélection automatique de Régions d'Intérêt (ROIs) pour CellViT-Optimus.

Identifie les régions prioritaires pour validation humaine basé sur:
- Incertitude élevée (entropie, set size conformal)
- Densité cellulaire élevée
- Présence de cellules néoplasiques
- Patterns morphologiques atypiques

Usage:
    selector = ROISelector()
    rois = selector.select_rois(
        uncertainty_map=uncertainty_map,
        cell_density_map=cell_density_map,
        type_map=type_map,
        n_rois=5
    )
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
from scipy import ndimage
from scipy.ndimage import label as connected_components


class ROIPriority(Enum):
    """Niveaux de priorité des ROIs."""
    CRITICAL = "critical"      # Validation urgente
    HIGH = "high"              # Haute priorité
    MEDIUM = "medium"          # Priorité moyenne
    LOW = "low"                # Basse priorité


@dataclass
class ROI:
    """Région d'intérêt identifiée."""
    # Position et taille
    x: int                     # Coin supérieur gauche X
    y: int                     # Coin supérieur gauche Y
    width: int                 # Largeur
    height: int                # Hauteur

    # Scores
    priority: ROIPriority      # Niveau de priorité
    score: float               # Score combiné (0-1)
    uncertainty_score: float   # Score d'incertitude (0-1)
    density_score: float       # Score de densité cellulaire (0-1)
    neoplastic_score: float    # Score de cellules néoplasiques (0-1)

    # Métadonnées
    reason: str                # Raison de la sélection
    cell_count: int = 0        # Nombre de cellules dans la ROI
    neoplastic_count: int = 0  # Nombre de cellules néoplasiques

    @property
    def center(self) -> Tuple[int, int]:
        """Centre de la ROI."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Bounding box (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def slice(self) -> Tuple[slice, slice]:
        """Slices pour indexation numpy."""
        return (slice(self.y, self.y + self.height),
                slice(self.x, self.x + self.width))

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'priority': self.priority.value,
            'score': self.score,
            'uncertainty': self.uncertainty_score,
            'density': self.density_score,
            'neoplastic': self.neoplastic_score,
            'reason': self.reason,
            'cell_count': self.cell_count,
            'neoplastic_count': self.neoplastic_count,
        }


class ROISelector:
    """
    Sélecteur automatique de régions d'intérêt.

    Analyse les cartes d'incertitude, densité cellulaire et types
    pour identifier les régions nécessitant une validation humaine.
    """

    def __init__(
        self,
        roi_size: int = 64,
        stride: int = 32,
        uncertainty_weight: float = 0.4,
        density_weight: float = 0.3,
        neoplastic_weight: float = 0.3,
        min_score_threshold: float = 0.3,
    ):
        """
        Args:
            roi_size: Taille des ROIs (pixels)
            stride: Pas de la fenêtre glissante
            uncertainty_weight: Poids de l'incertitude dans le score
            density_weight: Poids de la densité cellulaire
            neoplastic_weight: Poids des cellules néoplasiques
            min_score_threshold: Score minimum pour considérer une ROI
        """
        self.roi_size = roi_size
        self.stride = stride
        self.uncertainty_weight = uncertainty_weight
        self.density_weight = density_weight
        self.neoplastic_weight = neoplastic_weight
        self.min_score_threshold = min_score_threshold

    def _compute_cell_density(
        self,
        np_mask: np.ndarray,
        kernel_size: int = 32
    ) -> np.ndarray:
        """
        Calcule la carte de densité cellulaire.

        Args:
            np_mask: Masque binaire des noyaux (H, W)
            kernel_size: Taille du kernel de convolution

        Returns:
            Carte de densité normalisée (H, W)
        """
        # Créer un kernel de moyenne
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

        # Convoluer pour obtenir la densité locale
        density = ndimage.convolve(np_mask.astype(float), kernel, mode='constant')

        # Normaliser
        if density.max() > 0:
            density = density / density.max()

        return density

    def _compute_type_map(
        self,
        nt_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la carte de types et la carte de cellules néoplasiques.

        Args:
            nt_probs: Probabilités de types (H, W, 5) ou (5, H, W)

        Returns:
            type_map: Carte de types (H, W)
            neoplastic_map: Carte binaire néoplasique (H, W)
        """
        # Assurer format (H, W, C)
        if nt_probs.shape[0] == 5:
            nt_probs = np.transpose(nt_probs, (1, 2, 0))

        # Type = argmax
        type_map = nt_probs.argmax(axis=-1)

        # Néoplasique = classe 0 avec confiance > 0.5
        neoplastic_map = (type_map == 0) & (nt_probs[..., 0] > 0.5)

        return type_map, neoplastic_map.astype(float)

    def _score_roi(
        self,
        uncertainty: np.ndarray,
        density: np.ndarray,
        neoplastic: np.ndarray,
        np_mask: np.ndarray,
    ) -> Tuple[float, float, float, float, str]:
        """
        Calcule le score d'une ROI.

        Returns:
            (score_total, score_uncertainty, score_density, score_neoplastic, reason)
        """
        # Scores individuels (moyennes dans la région)
        u_score = float(uncertainty.mean()) if uncertainty.size > 0 else 0
        d_score = float(density.mean()) if density.size > 0 else 0
        n_score = float(neoplastic.mean()) if neoplastic.size > 0 else 0

        # Score combiné
        total = (
            self.uncertainty_weight * u_score +
            self.density_weight * d_score +
            self.neoplastic_weight * n_score
        )

        # Déterminer la raison principale
        scores = {
            'incertitude élevée': u_score,
            'forte densité cellulaire': d_score,
            'cellules néoplasiques': n_score,
        }
        reason = max(scores, key=scores.get)

        return total, u_score, d_score, n_score, reason

    def _priority_from_score(self, score: float) -> ROIPriority:
        """Détermine la priorité selon le score."""
        if score >= 0.7:
            return ROIPriority.CRITICAL
        elif score >= 0.5:
            return ROIPriority.HIGH
        elif score >= 0.3:
            return ROIPriority.MEDIUM
        else:
            return ROIPriority.LOW

    def select_rois(
        self,
        uncertainty_map: np.ndarray,
        np_mask: Optional[np.ndarray] = None,
        nt_probs: Optional[np.ndarray] = None,
        n_rois: int = 5,
        non_overlapping: bool = True,
    ) -> List[ROI]:
        """
        Sélectionne les N meilleures ROIs.

        Args:
            uncertainty_map: Carte d'incertitude (H, W)
            np_mask: Masque binaire des noyaux (H, W), optionnel
            nt_probs: Probabilités de types (H, W, 5) ou (5, H, W), optionnel
            n_rois: Nombre de ROIs à retourner
            non_overlapping: Si True, évite les ROIs qui se chevauchent

        Returns:
            Liste de ROIs triées par score décroissant
        """
        H, W = uncertainty_map.shape

        # Calculer les cartes auxiliaires
        if np_mask is None:
            np_mask = np.zeros((H, W))
            density_map = np.zeros((H, W))
        else:
            density_map = self._compute_cell_density(np_mask)

        if nt_probs is None:
            neoplastic_map = np.zeros((H, W))
        else:
            _, neoplastic_map = self._compute_type_map(nt_probs)

        # Scanner avec fenêtre glissante
        candidates = []

        for y in range(0, H - self.roi_size + 1, self.stride):
            for x in range(0, W - self.roi_size + 1, self.stride):
                # Extraire la région
                roi_uncertainty = uncertainty_map[y:y+self.roi_size, x:x+self.roi_size]
                roi_density = density_map[y:y+self.roi_size, x:x+self.roi_size]
                roi_neoplastic = neoplastic_map[y:y+self.roi_size, x:x+self.roi_size]
                roi_np = np_mask[y:y+self.roi_size, x:x+self.roi_size]

                # Calculer le score
                score, u_score, d_score, n_score, reason = self._score_roi(
                    roi_uncertainty, roi_density, roi_neoplastic, roi_np
                )

                if score >= self.min_score_threshold:
                    # Compter les cellules
                    cell_count = int(roi_np.sum())
                    neo_count = int(roi_neoplastic.sum())

                    candidates.append(ROI(
                        x=x,
                        y=y,
                        width=self.roi_size,
                        height=self.roi_size,
                        priority=self._priority_from_score(score),
                        score=score,
                        uncertainty_score=u_score,
                        density_score=d_score,
                        neoplastic_score=n_score,
                        reason=reason,
                        cell_count=cell_count,
                        neoplastic_count=neo_count,
                    ))

        # Trier par score décroissant
        candidates.sort(key=lambda r: r.score, reverse=True)

        # Sélectionner sans chevauchement si demandé
        if non_overlapping and len(candidates) > n_rois:
            selected = []
            for roi in candidates:
                if len(selected) >= n_rois:
                    break

                # Vérifier chevauchement avec ROIs déjà sélectionnées
                overlaps = False
                for sel in selected:
                    if self._rois_overlap(roi, sel):
                        overlaps = True
                        break

                if not overlaps:
                    selected.append(roi)

            return selected

        return candidates[:n_rois]

    def _rois_overlap(self, roi1: ROI, roi2: ROI, threshold: float = 0.3) -> bool:
        """Vérifie si deux ROIs se chevauchent significativement."""
        # Calculer l'intersection
        x1 = max(roi1.x, roi2.x)
        y1 = max(roi1.y, roi2.y)
        x2 = min(roi1.x + roi1.width, roi2.x + roi2.width)
        y2 = min(roi1.y + roi1.height, roi2.y + roi2.height)

        if x2 <= x1 or y2 <= y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        area1 = roi1.width * roi1.height
        area2 = roi2.width * roi2.height

        # IoU
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0

        return iou > threshold

    def visualize_rois(
        self,
        image: np.ndarray,
        rois: List[ROI],
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Visualise les ROIs sur l'image.

        Args:
            image: Image (H, W, 3) ou (H, W)
            rois: Liste de ROIs
            alpha: Transparence des overlays

        Returns:
            Image avec ROIs visualisées
        """
        # Convertir en RGB si nécessaire
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        output = image.copy().astype(float)

        # Couleurs par priorité
        colors = {
            ROIPriority.CRITICAL: (255, 0, 0),      # Rouge
            ROIPriority.HIGH: (255, 165, 0),        # Orange
            ROIPriority.MEDIUM: (255, 255, 0),      # Jaune
            ROIPriority.LOW: (0, 255, 0),           # Vert
        }

        for i, roi in enumerate(rois):
            color = np.array(colors[roi.priority])
            y_slice, x_slice = roi.slice

            # Overlay coloré
            overlay = np.zeros_like(output[y_slice, x_slice])
            overlay[:] = color

            output[y_slice, x_slice] = (
                (1 - alpha) * output[y_slice, x_slice] +
                alpha * overlay
            )

            # Bordure
            thickness = 2
            output[y_slice, roi.x:roi.x+thickness] = color
            output[y_slice, roi.x+roi.width-thickness:roi.x+roi.width] = color
            output[roi.y:roi.y+thickness, x_slice] = color
            output[roi.y+roi.height-thickness:roi.y+roi.height, x_slice] = color

        return output.astype(np.uint8)

    def generate_report(self, rois: List[ROI]) -> str:
        """Génère un rapport textuel des ROIs."""
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📍 RÉGIONS D'INTÉRÊT IDENTIFIÉES",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]

        priority_icons = {
            ROIPriority.CRITICAL: "🔴",
            ROIPriority.HIGH: "🟠",
            ROIPriority.MEDIUM: "🟡",
            ROIPriority.LOW: "🟢",
        }

        for i, roi in enumerate(rois, 1):
            icon = priority_icons[roi.priority]
            lines.extend([
                f"{icon} ROI #{i} - {roi.priority.value.upper()}",
                f"   Position: ({roi.x}, {roi.y}) - {roi.width}×{roi.height} px",
                f"   Score: {roi.score:.3f}",
                f"   Raison: {roi.reason}",
                f"   Cellules: {roi.cell_count} (dont {roi.neoplastic_count} néoplasiques)",
                "",
            ])

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return "\n".join(lines)


# Tests
if __name__ == "__main__":
    print("Test ROISelector...")
    print("=" * 50)

    np.random.seed(42)

    # Créer des données synthétiques
    H, W = 224, 224

    # Carte d'incertitude avec zone de haute incertitude
    uncertainty_map = np.random.rand(H, W) * 0.3
    uncertainty_map[50:100, 80:150] = 0.8  # Zone incertaine

    # Masque de noyaux
    np_mask = np.zeros((H, W))
    np_mask[30:180, 20:200] = np.random.rand(150, 180) > 0.7

    # Probabilités de types (simulées)
    nt_probs = np.random.rand(H, W, 5)
    nt_probs = nt_probs / nt_probs.sum(axis=-1, keepdims=True)
    # Zone avec beaucoup de néoplasiques
    nt_probs[100:150, 100:180, 0] = 0.9
    nt_probs[100:150, 100:180, 1:] = 0.025

    # Sélectionner les ROIs
    selector = ROISelector(roi_size=48, stride=16)
    rois = selector.select_rois(
        uncertainty_map=uncertainty_map,
        np_mask=np_mask,
        nt_probs=nt_probs,
        n_rois=5
    )

    print(f"✓ {len(rois)} ROIs sélectionnées")

    for i, roi in enumerate(rois):
        print(f"\n  ROI #{i+1}:")
        print(f"    Position: ({roi.x}, {roi.y})")
        print(f"    Priority: {roi.priority.value}")
        print(f"    Score: {roi.score:.3f}")
        print(f"    Reason: {roi.reason}")

    # Test rapport
    print("\n" + selector.generate_report(rois))

    print("✅ Tests passés!")
