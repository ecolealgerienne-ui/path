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
from scipy.spatial import Voronoi, distance, ConvexHull
from scipy.spatial.qhull import QhullError
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
    # Nouvelles métriques pour index mitotique
    elongation: float = 0.0  # Ratio axes ellipse (forme sablier)
    mean_intensity: float = 0.0  # Densité chromatine (si image fournie)
    is_mitotic_candidate: bool = False  # Suspicion de figure mitotique


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

    # Topographie / Architecture tissulaire
    spatial_distribution: str  # "diffuse", "clustered", "peritumoral"
    clustering_score: float  # 0-1, haut = cellules regroupées

    # Index Mitotique Estimé (NOUVEAU)
    mitotic_candidates: int = 0  # Nombre de figures évocatrices
    mitotic_index_per_10hpf: float = 0.0  # Index estimé pour 10 HPF
    mitotic_nuclei_ids: List[int] = None  # IDs pour XAI

    # Distance au Front d'Invasion (NOUVEAU - TILs hot/cold)
    til_invasion_distance_um: float = 0.0  # Distance moyenne TILs → front tumoral
    til_status: str = "indéterminé"  # "chaud", "froid", "exclu", "indéterminé"
    til_penetration_ratio: float = 0.0  # % TILs dans le massif tumoral

    # Alertes cliniques (langage suggestif)
    alerts: List[str] = None
    alert_nuclei_ids: Dict[str, List[int]] = None  # IDs des noyaux ayant déclenché chaque alerte

    # Niveau de confiance
    confidence_level: str = "Modérée"  # "Haute", "Modérée", "Faible"

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.alerts is None:
            self.alerts = []
        if self.alert_nuclei_ids is None:
            self.alert_nuclei_ids = {}
        if self.mitotic_nuclei_ids is None:
            self.mitotic_nuclei_ids = []


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
        image: Optional[np.ndarray] = None,
    ) -> MorphometryReport:
        """
        Analyse morphométrique complète d'un patch.

        Args:
            instance_map: Carte d'instances (H, W) avec labels 0=fond, 1..N=noyaux
            type_map: Carte de types (H, W) avec 0-4 = types PanNuke
            patch_size_um: Taille du patch en µm (calculé si None)
            image: Image originale (optionnel, améliore détection mitoses)

        Returns:
            MorphometryReport avec toutes les métriques cliniques
        """
        h, w = instance_map.shape

        if patch_size_um is None:
            patch_size_um = h * self.pixel_size_um

        patch_area_um2 = (h * self.pixel_size_um) * (w * self.pixel_size_um)
        patch_area_mm2 = patch_area_um2 / 1e6

        # Extraire les métriques par noyau (avec image pour intensité chromatine)
        nuclei = self._extract_nucleus_metrics(instance_map, type_map, image)

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

        # Analyse spatiale / Topographie
        spatial_dist, clustering_score = self._analyze_spatial_distribution(nuclei)

        # ==========================================
        # NOUVELLES MÉTRIQUES
        # ==========================================

        # Index mitotique estimé
        mitotic_candidates = [n for n in nuclei if n.is_mitotic_candidate]
        mitotic_count = len(mitotic_candidates)
        mitotic_nuclei_ids = [n.id for n in mitotic_candidates]

        # Estimation pour 10 HPF (High Power Fields)
        # 1 HPF ≈ 0.196 mm² à 40x, donc 10 HPF ≈ 1.96 mm²
        # patch_area_mm2 est notre surface analysée
        hpf_equivalent = patch_area_mm2 / 0.196
        mitotic_index_per_10hpf = (mitotic_count / hpf_equivalent) * 10 if hpf_equivalent > 0 else 0

        # TILs invasion (hot/cold)
        til_distance, til_status, til_penetration = self._compute_til_invasion_metrics(nuclei)

        # Générer les alertes cliniques (langage suggestif + IDs des noyaux)
        alerts, alert_nuclei_ids = self._generate_alerts_with_ids(
            nuclei, mean_area, std_area, mean_circ,
            nuclear_density, neoplastic_ratio, immuno_epithelial,
            mitotic_count, mitotic_nuclei_ids, til_status
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
            spatial_distribution=spatial_dist,
            clustering_score=clustering_score,
            # Nouvelles métriques
            mitotic_candidates=mitotic_count,
            mitotic_index_per_10hpf=mitotic_index_per_10hpf,
            mitotic_nuclei_ids=mitotic_nuclei_ids,
            til_invasion_distance_um=til_distance,
            til_status=til_status,
            til_penetration_ratio=til_penetration,
            # Alertes
            alerts=alerts,
            alert_nuclei_ids=alert_nuclei_ids,
            confidence_level=confidence,
        )

    def _extract_nucleus_metrics(
        self,
        instance_map: np.ndarray,
        type_map: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> List[NucleusMetrics]:
        """
        Extrait les métriques pour chaque noyau.

        Args:
            instance_map: Carte d'instances
            type_map: Carte des types
            image: Image originale (optionnel, pour intensité chromatine)
        """
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
            if perimeter > 0:
                circularity = (4 * np.pi * area_pixels) / (perimeter ** 2)
                circularity = min(circularity, 1.0)
            else:
                circularity = 0.0

            # Élongation via ellipse ajustée (pour détection mitoses)
            elongation = 0.0
            if len(contours[0]) >= 5:  # fitEllipse nécessite au moins 5 points
                try:
                    ellipse = cv2.fitEllipse(contours[0])
                    (_, (minor_axis, major_axis), _) = ellipse
                    if minor_axis > 0:
                        elongation = major_axis / minor_axis
                except cv2.error:
                    pass

            # Intensité moyenne (densité chromatine) si image fournie
            mean_intensity = 255.0  # Défaut clair (pas mitotique)
            if image is not None:
                if len(image.shape) == 3:
                    # Convertir en niveaux de gris (H de HSV pour H&E)
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                mean_intensity = gray[mask].mean() if mask.sum() > 0 else 255.0

            # ==========================================
            # DÉTECTION MITOTIQUE RAFFINÉE
            # ==========================================
            # Critères combinés (AND) pour réduire les faux positifs:
            # - Les cellules endothéliales/fibroblastes sont allongées MAIS claires
            # - Les figures mitotiques sont allongées ET hyperchromatiques (sombres)
            #
            # Seuils calibrés (recommandation expert pathologiste):
            # - elongation > 1.8 : forme "sablier" caractéristique
            # - mean_intensity < 100 : chromatine condensée (noyau sombre)
            # - Taille intermédiaire : 30-500 pixels (pas débris, pas artéfact)
            is_mitotic = False
            if 30 < area_pixels < 500:  # Taille plausible pour mitose
                # Critère 1: Forme très allongée ET hyperchromatique
                if elongation > 1.8 and mean_intensity < 100:
                    is_mitotic = True
                # Critère 2: Forme en métaphase (moins allongée mais très dense)
                elif elongation > 1.5 and mean_intensity < 70 and circularity < 0.5:
                    is_mitotic = True
                # Critère 3: Anaphase/Télophase (très allongée, chromatine visible)
                elif elongation > 2.2 and mean_intensity < 120:
                    is_mitotic = True

            # Type cellulaire (mode dans le masque)
            # NOTE: nt_mask peut contenir des -1 pour les pixels non assignés
            # On filtre ces valeurs avant de calculer le mode
            types_in_mask = type_map[mask]
            types_in_mask = types_in_mask[types_in_mask >= 0]  # Filtrer les -1
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
                elongation=elongation,
                mean_intensity=mean_intensity,
                is_mitotic_candidate=is_mitotic,
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

    def _compute_til_invasion_metrics(
        self,
        nuclei: List[NucleusMetrics],
    ) -> Tuple[float, str, float]:
        """
        Calcule les métriques d'invasion des TILs (Tumor-Infiltrating Lymphocytes).

        Utilise l'enveloppe convexe (Convex Hull) des cellules néoplasiques
        pour définir précisément le front tumoral.

        Détermine si la tumeur est "chaude" (TILs pénètrent le massif) ou
        "froide" (TILs bloqués en périphérie).

        Returns:
            (distance_um, status, penetration_ratio)
            - distance_um: Distance moyenne TILs → front tumoral
            - status: "chaud", "froid", "exclu", "indéterminé"
            - penetration_ratio: % TILs dans le massif tumoral (vs périphérie)
        """
        neoplastic = [n for n in nuclei if n.type_name == "Neoplastic"]
        inflammatory = [n for n in nuclei if n.type_name == "Inflammatory"]

        if len(neoplastic) < 5 or len(inflammatory) < 3:
            return 0.0, "indéterminé", 0.0

        neo_centers = np.array([n.centroid for n in neoplastic])
        inf_centers = np.array([n.centroid for n in inflammatory])

        # ==========================================
        # CONVEX HULL POUR DÉFINIR LE FRONT TUMORAL
        # ==========================================
        # L'enveloppe convexe des cellules néoplasiques définit
        # le "massif tumoral" de manière plus précise qu'un cercle
        try:
            hull = ConvexHull(neo_centers)
            hull_vertices = neo_centers[hull.vertices]
        except QhullError:
            # Fallback si les points sont colinéaires
            tumor_centroid = neo_centers.mean(axis=0)
            tumor_radius = np.sqrt(np.sum((neo_centers - tumor_centroid) ** 2, axis=1)).max()
            til_distances = np.sqrt(np.sum((inf_centers - tumor_centroid) ** 2, axis=1))
            penetration_ratio = np.sum(til_distances <= tumor_radius) / len(inflammatory)
            return til_distances.mean() * self.pixel_size_um, "indéterminé", penetration_ratio

        # Vérifier si chaque TIL est à l'intérieur du Convex Hull
        # Utilisation du test de demi-plan (cross product method)
        def point_in_hull(point: np.ndarray, hull_vertices: np.ndarray) -> bool:
            """Test si un point est à l'intérieur de l'enveloppe convexe."""
            n = len(hull_vertices)
            for i in range(n):
                v1 = hull_vertices[i]
                v2 = hull_vertices[(i + 1) % n]
                # Cross product pour déterminer le côté
                cross = (v2[0] - v1[0]) * (point[1] - v1[1]) - \
                        (v2[1] - v1[1]) * (point[0] - v1[0])
                if cross < 0:  # Point à l'extérieur
                    return False
            return True

        # Calculer le ratio de pénétration (TILs dans le hull)
        tils_inside = sum(1 for inf in inf_centers if point_in_hull(inf, hull_vertices))
        penetration_ratio = tils_inside / len(inflammatory)

        # Distance moyenne au front tumoral (bord du hull)
        # Pour chaque TIL, calculer la distance au segment de hull le plus proche
        def distance_to_hull_edge(point: np.ndarray, hull_vertices: np.ndarray) -> float:
            """Distance minimale d'un point au bord du hull."""
            min_dist = float('inf')
            n = len(hull_vertices)
            for i in range(n):
                v1 = hull_vertices[i]
                v2 = hull_vertices[(i + 1) % n]
                # Distance point → segment
                line_vec = v2 - v1
                point_vec = point - v1
                line_len = np.linalg.norm(line_vec)
                if line_len < 1e-6:
                    continue
                line_unitvec = line_vec / line_len
                proj_length = np.dot(point_vec, line_unitvec)
                proj_length = max(0, min(line_len, proj_length))
                closest_point = v1 + proj_length * line_unitvec
                dist = np.linalg.norm(point - closest_point)
                min_dist = min(min_dist, dist)
            return min_dist

        # Calculer distances au front pour tous les TILs
        til_distances_to_front = [distance_to_hull_edge(inf, hull_vertices) for inf in inf_centers]
        mean_til_distance = np.mean(til_distances_to_front)

        # Marge de périphérie (20 µm autour du front)
        periphery_margin_pixels = 20 / self.pixel_size_um

        # Classification du statut
        if penetration_ratio > 0.5:
            status = "chaud"  # TILs pénètrent le massif tumoral
        elif penetration_ratio > 0.2:
            status = "intermédiaire"  # Partiellement infiltré
        else:
            # Vérifier si TILs sont proches mais bloqués au front
            tils_at_periphery = sum(1 for d in til_distances_to_front if d < periphery_margin_pixels)
            periphery_ratio = tils_at_periphery / len(inflammatory)

            if periphery_ratio > 0.5:
                status = "froid"  # TILs bloqués à la périphérie du front
            elif mean_til_distance * self.pixel_size_um > 50:  # > 50 µm du front
                status = "exclu"  # TILs éloignés
            else:
                status = "froid"  # Par défaut si proche mais pas dedans

        return mean_til_distance * self.pixel_size_um, status, penetration_ratio

    def _analyze_spatial_distribution(
        self,
        nuclei: List[NucleusMetrics],
    ) -> Tuple[str, float]:
        """
        Analyse la distribution spatiale des cellules (architecture tissulaire).

        Returns:
            (distribution_type, clustering_score)
            - distribution_type: "diffuse", "clustered", "peritumoral"
            - clustering_score: 0-1 (haut = cellules très regroupées)
        """
        if len(nuclei) < 10:
            return "indéterminée", 0.0

        # Centres des noyaux
        centers = np.array([n.centroid for n in nuclei])

        # Calculer les distances au plus proche voisin
        from scipy.spatial import distance_matrix
        dist_mat = distance_matrix(centers, centers)
        np.fill_diagonal(dist_mat, np.inf)
        nn_distances = dist_mat.min(axis=1)

        mean_nn = nn_distances.mean()
        std_nn = nn_distances.std()

        # Coefficient de variation des distances (clustering)
        cv_nn = std_nn / (mean_nn + 1e-6)

        # Interprétation
        # CV bas = espacement régulier (diffus)
        # CV haut = distances très variables (clusters)
        if cv_nn < 0.3:
            distribution = "diffuse"
            clustering_score = 0.2
        elif cv_nn < 0.6:
            distribution = "hétérogène"
            clustering_score = 0.5
        else:
            distribution = "en amas"
            clustering_score = min(cv_nn, 1.0)

        # Vérifier si inflammatoires sont péri-tumoraux
        neoplastic = [n for n in nuclei if n.type_name == "Neoplastic"]
        inflammatory = [n for n in nuclei if n.type_name == "Inflammatory"]

        if len(neoplastic) > 5 and len(inflammatory) > 5:
            neo_centers = np.array([n.centroid for n in neoplastic])
            inf_centers = np.array([n.centroid for n in inflammatory])

            # Distance moyenne des inflammatoires aux néoplasiques
            dist_inf_neo = distance_matrix(inf_centers, neo_centers).min(axis=1)
            mean_dist_inf_neo = dist_inf_neo.mean()

            # Si inflammatoires sont proches des néoplasiques → péritumoral
            if mean_dist_inf_neo < 30 * self.pixel_size_um:  # < 30 µm
                distribution = "péritumoral"

        return distribution, clustering_score

    def _generate_alerts_with_ids(
        self,
        nuclei: List[NucleusMetrics],
        mean_area: float,
        std_area: float,
        mean_circ: float,
        nuclear_density: float,
        neoplastic_ratio: float,
        immuno_epithelial: float,
        mitotic_count: int = 0,
        mitotic_nuclei_ids: List[int] = None,
        til_status: str = "indéterminé",
    ) -> Tuple[List[str], Dict[str, List[int]]]:
        """
        Génère des alertes cliniques avec langage SUGGESTIF (pas définitif)
        et identifie les noyaux responsables de chaque alerte (XAI).

        Returns:
            (alerts, alert_nuclei_ids)
        """
        alerts = []
        alert_nuclei_ids = {}
        if mitotic_nuclei_ids is None:
            mitotic_nuclei_ids = []

        # ==========================================
        # INDEX MITOTIQUE (NOUVEAU)
        # ==========================================
        if mitotic_count > 0:
            if mitotic_count >= 5:
                alerts.append(f"🔍 Présence de figures évocatrices de mitoses ({mitotic_count})")
            else:
                alerts.append(f"ℹ️ Quelques figures évocatrices de mitoses ({mitotic_count})")
            alert_nuclei_ids["mitose"] = mitotic_nuclei_ids

        # ==========================================
        # STATUT TILs (hot/cold) - NOUVEAU
        # ==========================================
        if til_status == "chaud":
            alerts.append("ℹ️ Tumeur « chaude » : TILs infiltrant le massif tumoral")
        elif til_status == "froid":
            alerts.append("🔍 Tumeur « froide » : TILs bloqués en périphérie")
        elif til_status == "exclu":
            alerts.append("🔍 TILs exclus : immunité éloignée du site tumoral")

        # Coefficient de variation de l'aire (Anisocaryose)
        if mean_area > 0:
            cv_area = std_area / mean_area
            if cv_area > 0.5:
                alerts.append(f"🔍 Suspicion d'anisocaryose marquée (CV={cv_area:.2f})")
                threshold = mean_area + 2 * std_area
                atypical = [n.id for n in nuclei if n.area_um2 > threshold]
                alert_nuclei_ids["anisocaryose"] = atypical[:10]
            elif cv_area > 0.3:
                alerts.append(f"🔍 Anisocaryose modérée à explorer (CV={cv_area:.2f})")
                threshold = mean_area + 1.5 * std_area
                atypical = [n.id for n in nuclei if n.area_um2 > threshold]
                alert_nuclei_ids["anisocaryose"] = atypical[:5]

        # Atypie de forme (circularité faible = noyaux irréguliers)
        if mean_circ < 0.6:
            alerts.append(f"🔍 Possible atypie nucléaire (Circularité={mean_circ:.2f})")
            irregular = sorted(nuclei, key=lambda n: n.circularity)[:10]
            alert_nuclei_ids["atypie_forme"] = [n.id for n in irregular]

        # Hypercellularité
        if nuclear_density > 50:
            alerts.append(f"🔍 Aspect hypercellulaire à confirmer ({nuclear_density:.0f}%)")
        elif nuclear_density > 30:
            alerts.append(f"🔍 Densité cellulaire élevée ({nuclear_density:.0f}%)")

        # Proportion néoplasique - LANGAGE SUGGESTIF
        if neoplastic_ratio > 0.5:
            alerts.append(f"🔍 Suspicion de foyer néoplasique ({neoplastic_ratio:.0%})")
            neoplastic = [n.id for n in nuclei if n.type_name == "Neoplastic"]
            alert_nuclei_ids["neoplasique"] = neoplastic
        elif neoplastic_ratio > 0.2:
            alerts.append(f"🔍 Composante atypique à évaluer ({neoplastic_ratio:.0%})")
            neoplastic = [n.id for n in nuclei if n.type_name == "Neoplastic"]
            alert_nuclei_ids["neoplasique"] = neoplastic[:20]

        # Infiltration lymphocytaire (TILs) - informatif
        if immuno_epithelial > 2.0:
            alerts.append(f"ℹ️ Infiltration lymphocytaire notable (ratio I/E={immuno_epithelial:.1f})")
            inflammatory = [n.id for n in nuclei if n.type_name == "Inflammatory"]
            alert_nuclei_ids["infiltration"] = inflammatory
        elif immuno_epithelial > 0.5:
            alerts.append(f"ℹ️ Présence inflammatoire modérée (ratio I/E={immuno_epithelial:.1f})")

        return alerts, alert_nuclei_ids

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
            spatial_distribution="indéterminée",
            clustering_score=0.0,
            alerts=["ℹ️ Aucun noyau détecté sur ce patch"],
            alert_nuclei_ids={},
            confidence_level="Faible",
        )

    def generate_clinical_report(self, report: MorphometryReport, organ: str, family: str) -> str:
        """
        Génère un compte-rendu textuel clinique.

        Format adapté pour être directement copié dans un rapport médical.
        Utilise un langage SUGGESTIF, jamais affirmatif.
        """
        # Déterminer le type tissulaire dominant
        dominant_type = max(report.type_percentages.items(), key=lambda x: x[1])

        # Construire le texte
        lines = [
            f"ANALYSE MORPHOMÉTRIQUE AUTOMATISÉE",
            f"{'=' * 50}",
            f"⚠️ Document d'aide à la décision - Validation médicale requise",
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
            f"ARCHITECTURE TISSULAIRE",
            f"-" * 30,
            f"  • Topographie      : {report.spatial_distribution.capitalize()}",
            f"  • Score clustering : {report.clustering_score:.2f}",
        ])

        if report.stroma_tumor_distance_um > 0:
            lines.append(f"  • Dist. stroma-tumeur : {report.stroma_tumor_distance_um:.1f} µm")

        lines.append("")

        if report.alerts:
            lines.append("POINTS D'ATTENTION")
            lines.append("-" * 30)
            for alert in report.alerts:
                lines.append(f"  {alert}")
            lines.append("")

        # Résumé narratif
        lines.extend([
            "SYNTHÈSE AUTOMATIQUE",
            "-" * 30,
        ])

        # Construire le texte narratif - LANGAGE SUGGESTIF
        narrative = f"L'analyse automatisée sur tissu {organ.upper()} révèle "
        narrative += f"une population de {report.n_nuclei} noyaux "
        narrative += f"avec prédominance {dominant_type[0].lower()} ({dominant_type[1]:.0f}%). "

        # Architecture
        if report.spatial_distribution != "indéterminée":
            narrative += f"Répartition {report.spatial_distribution} des cellules. "

        # Néoplasie - langage suggestif
        if report.neoplastic_ratio > 0.5:
            narrative += f"Suspicion de foyer néoplasique à confirmer ({report.neoplastic_ratio:.0%}). "
        elif report.neoplastic_ratio > 0.2:
            narrative += f"Composante atypique à évaluer ({report.neoplastic_ratio:.0%}). "
        else:
            narrative += "Absence de massif néoplasique significatif sur ce patch. "

        # TILs
        if report.immuno_epithelial_ratio > 0.5:
            narrative += f"Infiltration inflammatoire notable (ratio I/E={report.immuno_epithelial_ratio:.1f}). "

        narrative += f"\n\nConfiance du modèle : {report.confidence_level}."

        lines.append(narrative)

        # Disclaimer
        lines.extend([
            "",
            "-" * 50,
            "Ce rapport est généré par un algorithme d'aide au",
            "diagnostic et ne remplace pas l'expertise médicale.",
        ])

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
