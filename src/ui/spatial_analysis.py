"""
CellViT-Optimus R&D Cockpit — Module d'Analyse Spatiale (Phase 3).

Ce module implémente les fonctionnalités d'intelligence spatiale:
- Pléomorphisme (score anisocaryose [1-3])
- Analyse chromatine (texture LBP, entropie)
- Topologie Voronoï (graphe adjacence cellules)
- Clustering spatial (hotspots, patterns)
- Détection mitoses avancée (forme + chromatine)

Author: CellViT-Optimus Project
Date: 2025-12-30
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.spatial import Voronoi, Delaunay
from scipy.ndimage import label as scipy_label
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ChromatinFeatures:
    """Caractéristiques de texture chromatinienne pour un noyau."""
    lbp_mean: float = 0.0           # Moyenne LBP (Local Binary Pattern)
    lbp_std: float = 0.0            # Écart-type LBP
    entropy: float = 0.0            # Entropie de Shannon
    contrast: float = 0.0           # Contraste texture
    homogeneity: float = 0.0        # Homogénéité
    is_heterogeneous: bool = False  # Chromatine hétérogène (signe malignité)


@dataclass
class PleomorphismScore:
    """Score de pléomorphisme nucléaire (anisocaryose)."""
    score: int = 1                  # 1 = faible, 2 = modéré, 3 = sévère
    area_cv: float = 0.0            # Coefficient de variation aire
    circularity_cv: float = 0.0     # CV circularité
    size_range_ratio: float = 0.0   # Ratio (max/min) des aires
    shape_heterogeneity: float = 0.0  # Hétérogénéité forme
    description: str = ""           # Description textuelle


@dataclass
class VoronoiCell:
    """Cellule de Voronoï avec info adjacence."""
    nucleus_id: int
    centroid: Tuple[int, int]       # (y, x)
    area: float                     # Aire de la cellule Voronoï
    neighbors: List[int] = field(default_factory=list)  # IDs voisins
    is_boundary: bool = False       # Cellule au bord de l'image


@dataclass
class SpatialCluster:
    """Cluster spatial de noyaux."""
    cluster_id: int
    nucleus_ids: List[int]
    centroid: Tuple[float, float]   # Centre du cluster
    area: float                     # Aire approximative
    density: float                  # Densité locale
    is_hotspot: bool = False        # Zone de haute densité


@dataclass
class SpatialAnalysisResult:
    """Résultat complet de l'analyse spatiale Phase 3."""

    # Pléomorphisme
    pleomorphism: PleomorphismScore = field(default_factory=PleomorphismScore)

    # Chromatine par noyau
    chromatin_features: Dict[int, ChromatinFeatures] = field(default_factory=dict)
    mean_entropy: float = 0.0
    heterogeneous_nuclei_ids: List[int] = field(default_factory=list)

    # Topologie Voronoï
    voronoi_cells: Dict[int, VoronoiCell] = field(default_factory=dict)
    mean_neighbors: float = 0.0
    adjacency_graph: Dict[int, List[int]] = field(default_factory=dict)

    # Clustering
    clusters: List[SpatialCluster] = field(default_factory=list)
    n_hotspots: int = 0
    hotspot_ids: List[int] = field(default_factory=list)  # IDs des noyaux dans hotspots

    # Mitoses avancées
    mitosis_candidates: List[int] = field(default_factory=list)
    mitosis_scores: Dict[int, float] = field(default_factory=dict)


# =============================================================================
# PLÉOMORPHISME
# =============================================================================

def compute_pleomorphism_score(
    areas: List[float],
    circularities: List[float],
    perimeters: Optional[List[float]] = None,
) -> PleomorphismScore:
    """
    Calcule le score de pléomorphisme (anisocaryose).

    Score basé sur:
    - Coefficient de variation (CV) de l'aire
    - CV de la circularité
    - Ratio taille max/min
    - Hétérogénéité des formes

    Args:
        areas: Liste des aires des noyaux (µm² ou pixels)
        circularities: Liste des circularités [0, 1]
        perimeters: Liste optionnelle des périmètres

    Returns:
        PleomorphismScore avec score 1-3
    """
    if len(areas) < 3:
        return PleomorphismScore(
            score=1,
            description="Échantillon insuffisant (< 3 noyaux)"
        )

    areas = np.array(areas)
    circularities = np.array(circularities)

    # Coefficient de variation aire
    area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0

    # CV circularité
    circ_cv = np.std(circularities) / np.mean(circularities) if np.mean(circularities) > 0 else 0

    # Ratio max/min (filtré pour outliers)
    areas_filtered = areas[(areas > np.percentile(areas, 5)) & (areas < np.percentile(areas, 95))]
    if len(areas_filtered) > 2:
        size_range_ratio = np.max(areas_filtered) / np.min(areas_filtered) if np.min(areas_filtered) > 0 else 1
    else:
        size_range_ratio = np.max(areas) / np.min(areas) if np.min(areas) > 0 else 1

    # Hétérogénéité forme (écart-type normalisé de la circularité)
    shape_heterogeneity = circ_cv

    # Score composite
    # Seuils basés sur littérature pathologie:
    # - CV aire < 0.25: faible, 0.25-0.50: modéré, > 0.50: sévère
    # - Size ratio < 3: faible, 3-6: modéré, > 6: sévère

    score_area = 1 if area_cv < 0.25 else (2 if area_cv < 0.50 else 3)
    score_ratio = 1 if size_range_ratio < 3 else (2 if size_range_ratio < 6 else 3)
    score_shape = 1 if shape_heterogeneity < 0.15 else (2 if shape_heterogeneity < 0.30 else 3)

    # Score final = max des composantes (approche conservative)
    final_score = max(score_area, score_ratio, score_shape)

    # Description
    descriptions = {
        1: "Pléomorphisme faible — noyaux relativement uniformes",
        2: "Pléomorphisme modéré — variation notable de taille/forme",
        3: "Pléomorphisme sévère — forte anisocaryose (suspicion malignité)",
    }

    return PleomorphismScore(
        score=final_score,
        area_cv=area_cv,
        circularity_cv=circ_cv,
        size_range_ratio=size_range_ratio,
        shape_heterogeneity=shape_heterogeneity,
        description=descriptions[final_score],
    )


# =============================================================================
# ANALYSE CHROMATINE
# =============================================================================

def compute_chromatin_features(
    image_gray: np.ndarray,
    mask: np.ndarray,
    lbp_radius: int = 3,
    lbp_points: int = 24,
) -> ChromatinFeatures:
    """
    Calcule les caractéristiques de texture chromatinienne pour un noyau.

    Utilise:
    - LBP (Local Binary Pattern) pour la texture
    - Entropie de Shannon pour la complexité
    - Contraste et homogénéité

    Args:
        image_gray: Image en niveaux de gris (H, W)
        mask: Masque binaire du noyau (H, W)
        lbp_radius: Rayon pour LBP
        lbp_points: Nombre de points pour LBP

    Returns:
        ChromatinFeatures avec métriques texture
    """
    if mask.sum() < 10:  # Trop petit
        return ChromatinFeatures()

    # Région d'intérêt
    coords = np.where(mask)
    min_y, max_y = coords[0].min(), coords[0].max() + 1
    min_x, max_x = coords[1].min(), coords[1].max() + 1

    roi = image_gray[min_y:max_y, min_x:max_x].copy()
    roi_mask = mask[min_y:max_y, min_x:max_x]

    if roi.size == 0 or roi_mask.sum() == 0:
        return ChromatinFeatures()

    # LBP (Local Binary Pattern)
    try:
        lbp = local_binary_pattern(roi, lbp_points, lbp_radius, method='uniform')
        lbp_values = lbp[roi_mask > 0]
        lbp_mean = float(np.mean(lbp_values)) if len(lbp_values) > 0 else 0
        lbp_std = float(np.std(lbp_values)) if len(lbp_values) > 0 else 0
    except Exception:
        lbp_mean, lbp_std = 0.0, 0.0

    # Entropie
    try:
        # Normaliser ROI pour entropie
        roi_norm = ((roi - roi.min()) / (roi.max() - roi.min() + 1e-8) * 255).astype(np.uint8)
        entropy_img = rank_entropy(roi_norm, disk(3))
        entropy_values = entropy_img[roi_mask > 0]
        entropy_val = float(np.mean(entropy_values)) if len(entropy_values) > 0 else 0
    except Exception:
        entropy_val = 0.0

    # Contraste et homogénéité (basés sur intensité)
    pixel_values = image_gray[mask > 0]
    if len(pixel_values) > 0:
        contrast = float(np.std(pixel_values))
        # Homogénéité = 1 - (std normalisé)
        homogeneity = 1.0 - (contrast / 128.0) if contrast < 128 else 0.0
    else:
        contrast, homogeneity = 0.0, 1.0

    # Détection chromatine hétérogène (entropie élevée + contraste élevé)
    is_heterogeneous = entropy_val > 4.0 and contrast > 30

    return ChromatinFeatures(
        lbp_mean=lbp_mean,
        lbp_std=lbp_std,
        entropy=entropy_val,
        contrast=contrast,
        homogeneity=homogeneity,
        is_heterogeneous=is_heterogeneous,
    )


def analyze_all_chromatin(
    image_rgb: np.ndarray,
    instance_map: np.ndarray,
) -> Tuple[Dict[int, ChromatinFeatures], float, List[int]]:
    """
    Analyse la chromatine de tous les noyaux.

    Args:
        image_rgb: Image RGB
        instance_map: Carte d'instances

    Returns:
        Tuple (features_dict, mean_entropy, heterogeneous_ids)
    """
    # Convertir en niveaux de gris
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    features_dict = {}
    entropies = []
    heterogeneous_ids = []

    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue

        mask = (instance_map == inst_id).astype(np.uint8)
        features = compute_chromatin_features(image_gray, mask)
        features_dict[inst_id] = features

        if features.entropy > 0:
            entropies.append(features.entropy)

        if features.is_heterogeneous:
            heterogeneous_ids.append(inst_id)

    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    return features_dict, mean_entropy, heterogeneous_ids


# =============================================================================
# TOPOLOGIE VORONOÏ
# =============================================================================

def build_voronoi_topology(
    centroids: List[Tuple[int, int]],
    nucleus_ids: List[int],
    image_shape: Tuple[int, int],
) -> Tuple[Dict[int, VoronoiCell], Dict[int, List[int]]]:
    """
    Construit la tessellation de Voronoï et le graphe d'adjacence.

    Args:
        centroids: Liste de (y, x) pour chaque noyau
        nucleus_ids: IDs correspondants
        image_shape: (H, W) de l'image

    Returns:
        Tuple (voronoi_cells, adjacency_graph)
    """
    if len(centroids) < 4:
        return {}, {}

    h, w = image_shape

    # Convertir en (x, y) pour scipy
    points = np.array([(c[1], c[0]) for c in centroids])

    voronoi_cells = {}
    adjacency_graph = {nid: [] for nid in nucleus_ids}

    try:
        # Triangulation de Delaunay pour adjacence
        tri = Delaunay(points)

        # Construire graphe d'adjacence à partir des triangles
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    id_i = nucleus_ids[simplex[i]]
                    id_j = nucleus_ids[simplex[j]]

                    if id_j not in adjacency_graph[id_i]:
                        adjacency_graph[id_i].append(id_j)
                    if id_i not in adjacency_graph[id_j]:
                        adjacency_graph[id_j].append(id_i)

        # Tessellation de Voronoï
        vor = Voronoi(points)

        for idx, nid in enumerate(nucleus_ids):
            region_idx = vor.point_region[idx]
            region = vor.regions[region_idx]

            # Vérifier si au bord
            is_boundary = -1 in region or any(
                v >= 0 and (vor.vertices[v][0] < 0 or vor.vertices[v][0] >= w or
                           vor.vertices[v][1] < 0 or vor.vertices[v][1] >= h)
                for v in region
            )

            # Calculer aire approximative (pour cellules non-infinies)
            area = 0.0
            if -1 not in region and len(region) >= 3:
                vertices = vor.vertices[region]
                # Shoelace formula
                n = len(vertices)
                area = 0.5 * abs(sum(
                    vertices[i][0] * vertices[(i + 1) % n][1] -
                    vertices[(i + 1) % n][0] * vertices[i][1]
                    for i in range(n)
                ))

            voronoi_cells[nid] = VoronoiCell(
                nucleus_id=nid,
                centroid=centroids[idx],
                area=area,
                neighbors=adjacency_graph[nid],
                is_boundary=is_boundary,
            )

    except Exception as e:
        logger.warning(f"Voronoi computation failed: {e}")

    return voronoi_cells, adjacency_graph


# =============================================================================
# CLUSTERING SPATIAL
# =============================================================================

def find_spatial_clusters(
    centroids: List[Tuple[int, int]],
    nucleus_ids: List[int],
    areas: List[float],
    image_shape: Tuple[int, int],
    density_threshold: float = 1.5,  # Multiplier of mean density
    min_cluster_size: int = 5,
) -> Tuple[List[SpatialCluster], List[int]]:
    """
    Identifie les clusters spatiaux (hotspots de haute densité).

    Utilise une grille de densité + connected components.

    Args:
        centroids: Liste de (y, x)
        nucleus_ids: IDs correspondants
        areas: Aires des noyaux
        image_shape: (H, W)
        density_threshold: Seuil pour hotspot (× densité moyenne)
        min_cluster_size: Taille minimum d'un cluster

    Returns:
        Tuple (clusters, hotspot_nucleus_ids)
    """
    if len(centroids) < min_cluster_size:
        return [], []

    h, w = image_shape

    # Grille de densité (résolution 16x16 pixels)
    grid_size = 16
    grid_h, grid_w = h // grid_size + 1, w // grid_size + 1
    density_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    nucleus_grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]

    for idx, (cy, cx) in enumerate(centroids):
        gy, gx = min(cy // grid_size, grid_h - 1), min(cx // grid_size, grid_w - 1)
        density_grid[gy, gx] += 1
        nucleus_grid[gy][gx].append(nucleus_ids[idx])

    # Seuil pour hotspots
    mean_density = np.mean(density_grid[density_grid > 0]) if np.any(density_grid > 0) else 0
    hotspot_threshold = mean_density * density_threshold

    # Masque binaire des zones denses
    hotspot_mask = (density_grid > hotspot_threshold).astype(np.uint8)

    # Connected components
    labeled, n_clusters = scipy_label(hotspot_mask)

    clusters = []
    all_hotspot_ids = []

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled == cluster_id
        cluster_coords = np.where(cluster_mask)

        # Collecter les noyaux dans ce cluster
        cluster_nucleus_ids = []
        for gy, gx in zip(cluster_coords[0], cluster_coords[1]):
            cluster_nucleus_ids.extend(nucleus_grid[gy][gx])

        if len(cluster_nucleus_ids) < min_cluster_size:
            continue

        # Centroïde du cluster
        cluster_centroids = [centroids[nucleus_ids.index(nid)] for nid in cluster_nucleus_ids]
        cy = float(np.mean([c[0] for c in cluster_centroids]))
        cx = float(np.mean([c[1] for c in cluster_centroids]))

        # Aire approximative
        area = float(cluster_mask.sum() * grid_size * grid_size)

        # Densité locale
        local_density = len(cluster_nucleus_ids) / (area + 1)

        clusters.append(SpatialCluster(
            cluster_id=cluster_id,
            nucleus_ids=cluster_nucleus_ids,
            centroid=(cy, cx),
            area=area,
            density=local_density,
            is_hotspot=True,
        ))

        all_hotspot_ids.extend(cluster_nucleus_ids)

    return clusters, all_hotspot_ids


# =============================================================================
# DÉTECTION MITOSES AVANCÉE
# =============================================================================

def detect_mitosis_advanced(
    instance_map: np.ndarray,
    image_rgb: np.ndarray,
    nucleus_ids: List[int],
    areas: List[float],
    circularities: List[float],
    chromatin_features: Dict[int, ChromatinFeatures],
) -> Tuple[List[int], Dict[int, float]]:
    """
    Détection avancée des mitoses basée sur forme + chromatine.

    Critères:
    - Forme irrégulière (circularité < 0.6)
    - Taille moyenne à grande (pas micro-noyaux)
    - Chromatine condensée (entropie spécifique)
    - Intensité plus foncée

    Args:
        instance_map: Carte d'instances
        image_rgb: Image RGB
        nucleus_ids: IDs des noyaux
        areas: Aires
        circularities: Circularités
        chromatin_features: Features chromatine déjà calculées

    Returns:
        Tuple (mitosis_candidate_ids, mitosis_scores)
    """
    if len(nucleus_ids) < 3:
        return [], {}

    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Statistiques de référence
    mean_area = np.mean(areas)
    mean_circ = np.mean(circularities)

    candidates = []
    scores = {}

    for idx, nid in enumerate(nucleus_ids):
        area = areas[idx]
        circ = circularities[idx]

        mask = (instance_map == nid)
        intensities = image_gray[mask]
        mean_intensity = np.mean(intensities) if len(intensities) > 0 else 128

        # Score basé sur critères
        score = 0.0

        # 1. Forme irrégulière (important pour mitoses)
        if circ < 0.5:
            score += 0.4
        elif circ < 0.65:
            score += 0.2

        # 2. Taille moyenne-grande (mitoses ne sont pas micro)
        if 0.7 * mean_area <= area <= 2.0 * mean_area:
            score += 0.2

        # 3. Intensité plus foncée (chromatine condensée)
        if mean_intensity < 100:  # Foncé
            score += 0.2
        elif mean_intensity < 130:
            score += 0.1

        # 4. Chromatine hétérogène
        if nid in chromatin_features:
            cf = chromatin_features[nid]
            if cf.entropy > 3.5:  # Texture complexe
                score += 0.2
            if cf.contrast > 40:  # Contraste élevé
                score += 0.1

        # Seuil pour candidat mitose
        if score >= 0.5:
            candidates.append(nid)
            scores[nid] = score

    # Trier par score décroissant
    candidates.sort(key=lambda x: scores[x], reverse=True)

    return candidates, scores


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def run_spatial_analysis(
    instance_map: np.ndarray,
    image_rgb: np.ndarray,
    areas: List[float],
    circularities: List[float],
    centroids: List[Tuple[int, int]],
    nucleus_ids: List[int],
) -> SpatialAnalysisResult:
    """
    Exécute l'analyse spatiale complète (Phase 3).

    Args:
        instance_map: Carte d'instances (H, W)
        image_rgb: Image RGB (H, W, 3)
        areas: Aires des noyaux (µm² ou pixels)
        circularities: Circularités [0, 1]
        centroids: Centroïdes (y, x)
        nucleus_ids: IDs des noyaux

    Returns:
        SpatialAnalysisResult complet
    """
    result = SpatialAnalysisResult()

    if len(nucleus_ids) < 3:
        logger.warning("Spatial analysis skipped: < 3 nuclei")
        return result

    # 1. Pléomorphisme
    logger.debug("Computing pleomorphism score...")
    result.pleomorphism = compute_pleomorphism_score(areas, circularities)

    # 2. Analyse chromatine
    logger.debug("Analyzing chromatin features...")
    chromatin_dict, mean_entropy, heterogeneous_ids = analyze_all_chromatin(
        image_rgb, instance_map
    )
    result.chromatin_features = chromatin_dict
    result.mean_entropy = mean_entropy
    result.heterogeneous_nuclei_ids = heterogeneous_ids

    # 3. Topologie Voronoï
    logger.debug("Building Voronoi topology...")
    voronoi_cells, adjacency_graph = build_voronoi_topology(
        centroids, nucleus_ids, instance_map.shape
    )
    result.voronoi_cells = voronoi_cells
    result.adjacency_graph = adjacency_graph
    if voronoi_cells:
        result.mean_neighbors = np.mean([
            len(vc.neighbors) for vc in voronoi_cells.values()
        ])

    # 4. Clustering spatial
    logger.debug("Finding spatial clusters...")
    clusters, hotspot_ids = find_spatial_clusters(
        centroids, nucleus_ids, areas, instance_map.shape
    )
    result.clusters = clusters
    result.n_hotspots = len(clusters)
    result.hotspot_ids = hotspot_ids

    # 5. Détection mitoses avancée
    logger.debug("Detecting mitosis candidates...")
    mitosis_candidates, mitosis_scores = detect_mitosis_advanced(
        instance_map, image_rgb, nucleus_ids, areas, circularities, chromatin_dict
    )
    result.mitosis_candidates = mitosis_candidates
    result.mitosis_scores = mitosis_scores

    logger.info(
        f"Spatial analysis complete: pleomorphism={result.pleomorphism.score}, "
        f"hotspots={result.n_hotspots}, mitosis_candidates={len(result.mitosis_candidates)}"
    )

    return result
