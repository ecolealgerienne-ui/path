#!/usr/bin/env python3
"""
Métriques Morphométriques Cliniques pour l'analyse histopathologique.

Ce module calcule des indicateurs cliniquement pertinents à partir
de la segmentation HoVer-Net, adaptés au langage des pathologistes.

Note: HoVer-Net segmente les NOYAUX, pas les cellules entières.
      Le ratio N/C exact n'est donc pas calculable.
      On utilise des métriques alternatives acceptées en pratique clinique.

Métriques disponibles:
- Aire Nucléaire Moyenne (µm²)
- Anisocaryose (variation taille noyaux)
- Index de Circularité (régularité forme)
- Score d'Hypercellularité (encombrement tissulaire)
- Rapport Immuno-Épithélial (TILs)
- Distance Stroma-Tumeur
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import Voronoi, distance
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2


@dataclass
class NucleusMetrics:
    """Métriques pour un noyau individuel."""
    id: int
    centroid: Tuple[int, int]  # (y, x)
    area_pixels: int
    area_um2: float  # Si calibration disponible
    perimeter: float
    circularity: float  # 4π × area / perimeter²
    type_idx: int
    type_name: str


@dataclass
class MorphometryReport:
    """Rapport morphométrique complet pour un patch."""

    # Statistiques nucléaires
    n_nuclei: int
    mean_area_um2: float
    std_area_um2: float  # Anisocaryose
    mean_circularity: float
    std_circularity: float  # Atypie de forme

    # Hypercellularité
    nuclear_density_percent: float  # Surface noyaux / Surface patch
    nuclei_per_mm2: float

    # Distribution par type
    type_counts: Dict[str, int]
    type_percentages: Dict[str, float]

    # Rapports cliniques
    immuno_epithelial_ratio: float  # Inflammatory / Epithelial
    neoplastic_ratio: float  # Neoplastic / Total
    stroma_tumor_distance_um: float  # Distance moyenne connective-neoplastic

    # Alertes cliniques
    alerts: List[str]

    # Niveau de confiance
    confidence_level: str  # "Haute", "Modérée", "Faible"


# Types cellulaires PanNuke
CELL_TYPES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]


class MorphometryAnalyzer:
    """
    Analyseur morphométrique pour segmentation HoVer-Net.

    Convertit les données techniques (instance_map, nt_mask) en
    métriques cliniquement pertinentes pour les pathologistes.
    """

    def __init__(
        self,
        pixel_size_um: float = 0.5,  # MPP (microns per pixel)
        min_nucleus_area: int = 20,   # Pixels minimum pour un noyau valide
    ):
        """
        Args:
            pixel_size_um: Taille d'un pixel en micromètres (0.5 pour 20x)
            min_nucleus_area: Surface minimale pour considérer un noyau
        """
        self.pixel_size_um = pixel_size_um
        self.min_nucleus_area = min_nucleus_area
        self.pixel_area_um2 = pixel_size_um ** 2

    def analyze(
        self,
        instance_map: np.ndarray,
        type_map: np.ndarray,
        patch_size_um: Optional[float] = None,
    ) -> MorphometryReport:
        """
        Analyse morphométrique complète d'un patch.

        Args:
            instance_map: Carte d'instances (H, W) avec labels 0=fond, 1..N=noyaux
            type_map: Carte de types (H, W) avec 0-4 = types PanNuke
            patch_size_um: Taille du patch en µm (calculé si None)

        Returns:
            MorphometryReport avec toutes les métriques cliniques
        """
        h, w = instance_map.shape

        if patch_size_um is None:
            patch_size_um = h * self.pixel_size_um

        patch_area_um2 = (h * self.pixel_size_um) * (w * self.pixel_size_um)
        patch_area_mm2 = patch_area_um2 / 1e6

        # Extraire les métriques par noyau
        nuclei = self._extract_nucleus_metrics(instance_map, type_map)

        if len(nuclei) == 0:
            return self._empty_report()

        # Statistiques nucléaires
        areas = [n.area_um2 for n in nuclei]
        circularities = [n.circularity for n in nuclei]

        mean_area = np.mean(areas)
        std_area = np.std(areas)
        mean_circ = np.mean(circularities)
        std_circ = np.std(circularities)

        # Hypercellularité
        total_nuclear_area_pixels = sum(n.area_pixels for n in nuclei)
        nuclear_density = (total_nuclear_area_pixels / (h * w)) * 100
        nuclei_per_mm2 = len(nuclei) / patch_area_mm2

        # Distribution par type
        type_counts = {t: 0 for t in CELL_TYPES}
        for n in nuclei:
            type_counts[n.type_name] += 1

        total = len(nuclei)
        type_percentages = {t: (c / total) * 100 for t, c in type_counts.items()}

        # Rapports cliniques
        n_inflammatory = type_counts["Inflammatory"]
        n_epithelial = type_counts["Epithelial"]
        n_neoplastic = type_counts["Neoplastic"]
        n_connective = type_counts["Connective"]

        immuno_epithelial = n_inflammatory / max(n_epithelial, 1)
        neoplastic_ratio = n_neoplastic / max(total, 1)

        # Distance Stroma-Tumeur
        stroma_tumor_dist = self._compute_stroma_tumor_distance(nuclei)

        # Générer les alertes cliniques
        alerts = self._generate_alerts(
            mean_area, std_area, mean_circ, std_circ,
            nuclear_density, neoplastic_ratio, immuno_epithelial
        )

        # Niveau de confiance
        confidence = self._assess_confidence(len(nuclei), nuclear_density)

        return MorphometryReport(
            n_nuclei=len(nuclei),
            mean_area_um2=mean_area,
            std_area_um2=std_area,
            mean_circularity=mean_circ,
            std_circularity=std_circ,
            nuclear_density_percent=nuclear_density,
            nuclei_per_mm2=nuclei_per_mm2,
            type_counts=type_counts,
            type_percentages=type_percentages,
            immuno_epithelial_ratio=immuno_epithelial,
            neoplastic_ratio=neoplastic_ratio,
            stroma_tumor_distance_um=stroma_tumor_dist,
            alerts=alerts,
            confidence_level=confidence,
        )

    def _extract_nucleus_metrics(
        self,
        instance_map: np.ndarray,
        type_map: np.ndarray,
    ) -> List[NucleusMetrics]:
        """Extrait les métriques pour chaque noyau."""
        nuclei = []

        for inst_id in range(1, instance_map.max() + 1):
            mask = instance_map == inst_id
            area_pixels = mask.sum()

            if area_pixels < self.min_nucleus_area:
                continue

            # Centroïde
            coords = np.where(mask)
            cy, cx = int(coords[0].mean()), int(coords[1].mean())

            # Périmètre via contours
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            perimeter = cv2.arcLength(contours[0], True)

            # Circularité: 4π × area / perimeter²
            # = 1 pour cercle parfait, < 1 pour formes irrégulières
            if perimeter > 0:
                circularity = (4 * np.pi * area_pixels) / (perimeter ** 2)
                circularity = min(circularity, 1.0)  # Clamp
            else:
                circularity = 0.0

            # Type cellulaire (mode dans le masque)
            types_in_mask = type_map[mask]
            if len(types_in_mask) > 0:
                type_idx = int(np.bincount(types_in_mask.astype(int)).argmax())
            else:
                type_idx = 0

            nuclei.append(NucleusMetrics(
                id=inst_id,
                centroid=(cy, cx),
                area_pixels=area_pixels,
                area_um2=area_pixels * self.pixel_area_um2,
                perimeter=perimeter * self.pixel_size_um,
                circularity=circularity,
                type_idx=type_idx,
                type_name=CELL_TYPES[type_idx] if type_idx < 5 else "Unknown",
            ))

        return nuclei

    def _compute_stroma_tumor_distance(
        self,
        nuclei: List[NucleusMetrics],
    ) -> float:
        """Calcule la distance moyenne entre cellules stromales et tumorales."""
        neoplastic = [n for n in nuclei if n.type_name == "Neoplastic"]
        connective = [n for n in nuclei if n.type_name == "Connective"]

        if len(neoplastic) == 0 or len(connective) == 0:
            return 0.0

        # Centres
        neo_centers = np.array([n.centroid for n in neoplastic])
        conn_centers = np.array([n.centroid for n in connective])

        # Distance minimale de chaque cellule néoplasique au stroma
        distances = []
        for nc in neo_centers:
            dists = np.sqrt(np.sum((conn_centers - nc) ** 2, axis=1))
            distances.append(dists.min())

        mean_dist_pixels = np.mean(distances)
        return mean_dist_pixels * self.pixel_size_um

    def _generate_alerts(
        self,
        mean_area: float,
        std_area: float,
        mean_circ: float,
        std_circ: float,
        nuclear_density: float,
        neoplastic_ratio: float,
        immuno_epithelial: float,
    ) -> List[str]:
        """Génère des alertes cliniques basées sur les métriques."""
        alerts = []

        # Coefficient de variation de l'aire (Anisocaryose)
        if mean_area > 0:
            cv_area = std_area / mean_area
            if cv_area > 0.5:
                alerts.append(f"⚠️ Anisocaryose marquée (CV={cv_area:.2f})")
            elif cv_area > 0.3:
                alerts.append(f"⚡ Anisocaryose modérée (CV={cv_area:.2f})")

        # Atypie de forme (circularité faible = noyaux irréguliers)
        if mean_circ < 0.6:
            alerts.append(f"⚠️ Atypie nucléaire (Circularité={mean_circ:.2f})")

        # Hypercellularité
        if nuclear_density > 50:
            alerts.append(f"🔴 Hypercellularité sévère ({nuclear_density:.0f}%)")
        elif nuclear_density > 30:
            alerts.append(f"⚠️ Hypercellularité modérée ({nuclear_density:.0f}%)")

        # Proportion néoplasique
        if neoplastic_ratio > 0.5:
            alerts.append(f"🔴 Prédominance néoplasique ({neoplastic_ratio:.0%})")
        elif neoplastic_ratio > 0.2:
            alerts.append(f"⚠️ Composante néoplasique significative ({neoplastic_ratio:.0%})")

        # Infiltration lymphocytaire (TILs)
        if immuno_epithelial > 2.0:
            alerts.append(f"🔵 Infiltration lymphocytaire importante (ratio={immuno_epithelial:.1f})")
        elif immuno_epithelial > 0.5:
            alerts.append(f"🔵 Infiltration lymphocytaire modérée (ratio={immuno_epithelial:.1f})")

        return alerts

    def _assess_confidence(
        self,
        n_nuclei: int,
        nuclear_density: float,
    ) -> str:
        """Évalue le niveau de confiance de l'analyse."""
        if n_nuclei < 10:
            return "Faible"
        elif n_nuclei < 50 or nuclear_density < 5:
            return "Modérée"
        else:
            return "Haute"

    def _empty_report(self) -> MorphometryReport:
        """Retourne un rapport vide."""
        return MorphometryReport(
            n_nuclei=0,
            mean_area_um2=0.0,
            std_area_um2=0.0,
            mean_circularity=0.0,
            std_circularity=0.0,
            nuclear_density_percent=0.0,
            nuclei_per_mm2=0.0,
            type_counts={t: 0 for t in CELL_TYPES},
            type_percentages={t: 0.0 for t in CELL_TYPES},
            immuno_epithelial_ratio=0.0,
            neoplastic_ratio=0.0,
            stroma_tumor_distance_um=0.0,
            alerts=["⚠️ Aucun noyau détecté"],
            confidence_level="Faible",
        )

    def generate_clinical_report(self, report: MorphometryReport, organ: str, family: str) -> str:
        """
        Génère un compte-rendu textuel clinique.

        Format adapté pour être directement copié dans un rapport médical.
        """
        # Déterminer le type tissulaire dominant
        dominant_type = max(report.type_percentages.items(), key=lambda x: x[1])

        # Construire le texte
        lines = [
            f"ANALYSE MORPHOMÉTRIQUE AUTOMATISÉE",
            f"{'=' * 50}",
            f"",
            f"Tissu analysé : {organ.upper()} (Famille {family})",
            f"Noyaux détectés : {report.n_nuclei}",
            f"Densité : {report.nuclei_per_mm2:.0f} noyaux/mm²",
            f"",
            f"POPULATION CELLULAIRE",
            f"-" * 30,
        ]

        for cell_type in CELL_TYPES:
            pct = report.type_percentages[cell_type]
            count = report.type_counts[cell_type]
            if count > 0:
                lines.append(f"  • {cell_type:15}: {count:4} ({pct:5.1f}%)")

        lines.extend([
            f"",
            f"CARACTÉRISTIQUES NUCLÉAIRES",
            f"-" * 30,
            f"  • Aire moyenne     : {report.mean_area_um2:.1f} ± {report.std_area_um2:.1f} µm²",
            f"  • Circularité      : {report.mean_circularity:.2f} ± {report.std_circularity:.2f}",
            f"  • Hypercellularité : {report.nuclear_density_percent:.1f}%",
            f"",
        ])

        if report.alerts:
            lines.append("ALERTES CLINIQUES")
            lines.append("-" * 30)
            for alert in report.alerts:
                lines.append(f"  {alert}")
            lines.append("")

        # Résumé narratif
        lines.extend([
            "SYNTHÈSE",
            "-" * 30,
        ])

        # Construire le texte narratif
        narrative = f"L'analyse automatisée sur tissu {organ.upper()} révèle "
        narrative += f"une population de {report.n_nuclei} noyaux "
        narrative += f"avec prédominance {dominant_type[0].lower()} ({dominant_type[1]:.0f}%). "

        if report.neoplastic_ratio > 0.2:
            narrative += f"Présence significative de cellules néoplasiques ({report.neoplastic_ratio:.0%}). "
        else:
            narrative += "Absence de massif néoplasique significatif. "

        if report.immuno_epithelial_ratio > 0.5:
            narrative += f"Infiltration inflammatoire notable (ratio I/E={report.immuno_epithelial_ratio:.1f}). "

        narrative += f"Confiance du modèle : {report.confidence_level}."

        lines.append(narrative)

        return "\n".join(lines)


def compute_voronoi_territories(
    instance_map: np.ndarray,
    type_map: np.ndarray,
) -> np.ndarray:
    """
    Approximation des territoires cytoplasmiques par tessellation de Voronoi.

    Note: Ce n'est PAS une vraie segmentation cellulaire, mais une estimation
    géométrique du territoire de chaque noyau basée sur les voisins les plus proches.

    Args:
        instance_map: Carte d'instances des noyaux
        type_map: Carte des types cellulaires

    Returns:
        voronoi_map: Carte où chaque pixel appartient au noyau le plus proche
    """
    h, w = instance_map.shape

    # Extraire les centroïdes
    centroids = []
    for inst_id in range(1, instance_map.max() + 1):
        mask = instance_map == inst_id
        if mask.sum() < 10:
            continue
        coords = np.where(mask)
        cy, cx = coords[0].mean(), coords[1].mean()
        centroids.append((inst_id, cy, cx))

    if len(centroids) < 3:
        return instance_map.copy()

    # Créer la carte de Voronoi par distance
    points = np.array([(c[2], c[1]) for c in centroids])  # (x, y)
    ids = [c[0] for c in centroids]

    # Grille de coordonnées
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Distance à chaque centroïde
    distances = distance.cdist(coords, points)

    # Assigner au plus proche
    closest = distances.argmin(axis=1)
    voronoi_map = np.array([ids[i] for i in closest]).reshape(h, w)

    return voronoi_map


# Test
if __name__ == "__main__":
    print("Test MorphometryAnalyzer...")

    # Créer des données de test
    instance_map = np.zeros((224, 224), dtype=np.int32)
    type_map = np.zeros((224, 224), dtype=np.int32)

    # Ajouter quelques noyaux simulés
    np.random.seed(42)
    for i in range(1, 51):
        cy, cx = np.random.randint(20, 204, 2)
        radius = np.random.randint(5, 15)
        yy, xx = np.ogrid[-cy:224-cy, -cx:224-cx]
        mask = xx**2 + yy**2 <= radius**2
        instance_map[mask] = i
        type_map[mask] = np.random.randint(0, 5)

    # Analyser
    analyzer = MorphometryAnalyzer(pixel_size_um=0.5)
    report = analyzer.analyze(instance_map, type_map)

    print(f"\n✓ Noyaux détectés: {report.n_nuclei}")
    print(f"✓ Aire moyenne: {report.mean_area_um2:.1f} µm²")
    print(f"✓ Circularité: {report.mean_circularity:.2f}")
    print(f"✓ Hypercellularité: {report.nuclear_density_percent:.1f}%")
    print(f"✓ Densité: {report.nuclei_per_mm2:.0f} noyaux/mm²")

    print(f"\nDistribution:")
    for t, pct in report.type_percentages.items():
        print(f"  {t}: {pct:.1f}%")

    print(f"\nAlertes: {report.alerts}")

    # Rapport clinique
    clinical = analyzer.generate_clinical_report(report, "Colon", "digestive")
    print(f"\n{clinical}")

    print("\n✅ Test passé!")
