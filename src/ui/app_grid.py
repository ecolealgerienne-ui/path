#!/usr/bin/env python3
"""
CellViT-Optimus — Interface Unifiée (Patches + WSI).

Deux modes de fonctionnement:
1. **Mode Patch** (256×256): Simulation WSI avec images PanNuke
2. **Mode WSI** (réel): Traitement de lames entières (.svs, .ndpi, etc.)

**Mode WSI:**
- Parcourir un dossier de lames
- Afficher le thumbnail de la lame sélectionnée
- Lancer le traitement avec timer
- Afficher les résultats diagnostiques

Usage:
    python -m src.ui.app_grid --organ Lung --wsi_dir data/wsi_test
    python src/ui/app_grid.py --organ Breast --port 7861
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from scipy import ndimage
import sys
import time

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le chemin racine au PYTHONPATH si nécessaire
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports: Logique partagée (core)
from src.ui.core import (
    state,
    preload_backbone_core,
    load_engine_core,
    run_analysis_core,
)

# Imports: Moteur et configuration
from src.ui.inference_engine import ORGAN_CHOICES, AnalysisResult

# Imports: Visualisations
from src.ui.visualizations import (
    create_segmentation_overlay,
)

# Imports: WSI support
from src.wsi.input_router import InputRouter, get_input_metadata, InputType

# ==============================================================================
# CONSTANTES
# ==============================================================================

# WSI Extensions supportées
WSI_EXTENSIONS = {'.svs', '.ndpi', '.mrxs', '.scn', '.tiff', '.tif', '.bif'}

# Dossier WSI par défaut
DEFAULT_WSI_DIR = "data/wsi_test"

PANNUKE_SIZE = 256
PATCH_SIZE = 224
OFFSET = PANNUKE_SIZE - PATCH_SIZE  # 32 pixels

# Positions des 4 patches (grille 2×2 avec chevauchement)
# Format: (x_offset, y_offset) pour image[y:y+224, x:x+224]
PATCH_POSITIONS = [
    (0, 0),           # Patch 0: Top-Left
    (OFFSET, 0),      # Patch 1: Top-Right  (x=32)
    (0, OFFSET),      # Patch 2: Bottom-Left (y=32)
    (OFFSET, OFFSET), # Patch 3: Bottom-Right (x=32, y=32)
]

PATCH_NAMES = ["Haut-Gauche", "Haut-Droite", "Bas-Gauche", "Bas-Droite"]

# ==============================================================================
# ZONES VALIDES (Stitching WSI Standard)
# ==============================================================================
# Diviser l'image 256×256 en 4 quadrants sans chevauchement.
# Chaque patch ne "possède" que les noyaux dans son quadrant assigné.
#
# Image 256×256:
# ┌─────────────┬─────────────┐
# │  Q0 (0,0)   │  Q1 (128,0) │
# │  0:128      │  128:256    │
# ├─────────────┼─────────────┤
# │  Q2 (0,128) │  Q3 (128,128)│
# │  0:128      │  128:256    │
# └─────────────┴─────────────┘

# Zones valides en coordonnées IMAGE (y_min, y_max, x_min, x_max)
VALID_ZONES_IMAGE = [
    (0, 128, 0, 128),      # Patch 0 → Quadrant haut-gauche
    (0, 128, 128, 256),    # Patch 1 → Quadrant haut-droit
    (128, 256, 0, 128),    # Patch 2 → Quadrant bas-gauche
    (128, 256, 128, 256),  # Patch 3 → Quadrant bas-droit
]

# Zones valides en coordonnées PATCH LOCAL (y_min, y_max, x_min, x_max)
# Calculé: zone_image - patch_offset
VALID_ZONES_PATCH = [
    (0, 128, 0, 128),      # Patch 0: offset (0,0) → [0:128, 0:128]
    (0, 128, 96, 224),     # Patch 1: offset (32,0) → [0:128, 128-32:256-32] = [0:128, 96:224]
    (96, 224, 0, 128),     # Patch 2: offset (0,32) → [128-32:256-32, 0:128] = [96:224, 0:128]
    (96, 224, 96, 224),    # Patch 3: offset (32,32) → [96:224, 96:224]
]


# ==============================================================================
# ÉTAT GRILLE (SIMULATION WSI)
# ==============================================================================

@dataclass
class NucleusInfo:
    """Information sur un noyau individuel (pour stitching)."""
    id: int  # ID dans le patch
    centroid_patch: Tuple[int, int]  # (y, x) en coordonnées patch
    centroid_image: Tuple[int, int]  # (y, x) en coordonnées image 256×256
    cell_type: int  # Type from HoVer-Net
    area_pixels: int = 0
    in_valid_zone: bool = False


@dataclass
class PatchInfo:
    """Information sur un patch extrait."""
    index: int
    name: str
    position: Tuple[int, int]  # (x, y) offset
    image: np.ndarray  # Patch 224×224
    result: Optional[AnalysisResult] = None
    overlay: Optional[np.ndarray] = None
    is_analyzed: bool = False
    # Stitching
    nuclei: List[NucleusInfo] = field(default_factory=list)
    valid_nuclei_count: int = 0


@dataclass
class WSIState:
    """État de la simulation WSI (une image source)."""
    source_image: Optional[np.ndarray] = None  # Image originale 256×256
    source_filename: str = ""
    patches: List[PatchInfo] = field(default_factory=list)
    selected_index: int = 0
    # Stitching results
    stitched_instance_map: Optional[np.ndarray] = None  # 256×256
    stitched_type_map: Optional[np.ndarray] = None  # 256×256
    stitched_overlay: Optional[np.ndarray] = None  # 256×256 RGB

    def clear(self):
        """Réinitialise l'état."""
        self.source_image = None
        self.source_filename = ""
        self.patches = []
        self.selected_index = 0
        self.stitched_instance_map = None
        self.stitched_type_map = None
        self.stitched_overlay = None

    def get_selected(self) -> Optional[PatchInfo]:
        """Retourne le patch sélectionné."""
        if 0 <= self.selected_index < len(self.patches):
            return self.patches[self.selected_index]
        return None

    def all_analyzed(self) -> bool:
        """Vérifie si tous les patches sont analysés."""
        return all(p.is_analyzed for p in self.patches)

    def get_aggregated_metrics_stitched(self) -> Dict[str, Any]:
        """Calcule les métriques agrégées sur les noyaux VALIDES uniquement (sans doublons)."""
        if not self.all_analyzed():
            return {"total_patches": len(self.patches), "analyzed": 0}

        # Compter uniquement les noyaux dans les zones valides
        total_nuclei = 0
        type_counts = {}

        for p in self.patches:
            total_nuclei += p.valid_nuclei_count
            for nucleus in p.nuclei:
                if nucleus.in_valid_zone:
                    cell_type = nucleus.cell_type
                    type_counts[cell_type] = type_counts.get(cell_type, 0) + 1

        # Surface = 256×256 pixels (pas de chevauchement dans les métriques)
        # À 0.5 MPP: 256 × 0.5 = 128 µm → surface = 128² = 16384 µm² = 0.016384 mm²
        total_area_mm2 = (PANNUKE_SIZE * 0.5) ** 2 / 1_000_000  # µm² → mm²

        return {
            "total_patches": len(self.patches),
            "analyzed": len([p for p in self.patches if p.is_analyzed]),
            "total_nuclei": total_nuclei,
            "type_counts": type_counts,
            "total_area_mm2": total_area_mm2,
            "density_per_mm2": total_nuclei / total_area_mm2 if total_area_mm2 > 0 else 0,
        }


# Instance globale
wsi_state = WSIState()


# ==============================================================================
# ÉTAT WSI RÉEL (Lames entières)
# ==============================================================================

@dataclass
class TileResult:
    """Résultat d'analyse d'un tile avec score de gravité."""
    index: int
    x: int  # Coordonnée X dans la lame
    y: int  # Coordonnée Y dans la lame
    image: np.ndarray  # Image RGB 224×224
    overlay: np.ndarray  # Overlay segmentation
    total_nuclei: int
    type_counts: Dict[int, int]
    severity_score: float  # Score de gravité [0-1]

    @property
    def neoplastic_ratio(self) -> float:
        """Ratio de cellules néoplasiques."""
        total = sum(self.type_counts.values())
        if total == 0:
            return 0.0
        return self.type_counts.get(1, 0) / total

    @property
    def severity_label(self) -> str:
        """Label de gravité basé sur le score."""
        if self.severity_score >= 0.5:
            return "🔴 Élevé"
        elif self.severity_score >= 0.2:
            return "🟠 Modéré"
        elif self.severity_score >= 0.05:
            return "🟡 Faible"
        else:
            return "🟢 Normal"


@dataclass
class RealWSIState:
    """État pour le traitement de lames WSI réelles."""
    # Dossier et fichiers
    wsi_dir: Path = field(default_factory=lambda: Path(DEFAULT_WSI_DIR))
    available_files: List[str] = field(default_factory=list)
    selected_file: Optional[str] = None

    # Métadonnées de la lame
    slide_dimensions: Tuple[int, int] = (0, 0)  # (width, height)
    slide_mpp: Optional[float] = None
    slide_levels: int = 0
    thumbnail: Optional[np.ndarray] = None

    # Traitement
    is_processing: bool = False
    processing_start_time: float = 0.0
    processing_elapsed: float = 0.0
    tiles_total: int = 0
    tiles_processed: int = 0

    # Résultats
    results: Dict[str, Any] = field(default_factory=dict)
    total_nuclei: int = 0
    type_counts: Dict[int, int] = field(default_factory=dict)

    # Tiles analysés (triés par gravité)
    tile_results: List[TileResult] = field(default_factory=list)
    selected_tile_index: int = -1

    def clear_results(self):
        """Réinitialise les résultats."""
        self.is_processing = False
        self.processing_start_time = 0.0
        self.processing_elapsed = 0.0
        self.tiles_total = 0
        self.tiles_processed = 0
        self.results = {}
        self.total_nuclei = 0
        self.type_counts = {}
        self.tile_results = []
        self.selected_tile_index = -1

    def scan_wsi_folder(self) -> List[str]:
        """Scanne le dossier WSI et retourne la liste des fichiers."""
        self.available_files = []

        if not self.wsi_dir.exists():
            logger.warning(f"Dossier WSI non trouvé: {self.wsi_dir}")
            return []

        for ext in WSI_EXTENSIONS:
            self.available_files.extend([f.name for f in self.wsi_dir.glob(f"*{ext}")])
            self.available_files.extend([f.name for f in self.wsi_dir.glob(f"*{ext.upper()}")])

        self.available_files = sorted(set(self.available_files))
        logger.info(f"Trouvé {len(self.available_files)} fichiers WSI dans {self.wsi_dir}")
        return self.available_files


# Instance globale pour WSI réel
real_wsi_state = RealWSIState()


# ==============================================================================
# FONCTIONS WSI RÉEL
# ==============================================================================

def get_wsi_thumbnail(slide_path: Path, max_size: int = 512) -> Optional[np.ndarray]:
    """
    Extrait le thumbnail d'une lame WSI.

    Args:
        slide_path: Chemin vers la lame
        max_size: Taille maximale du thumbnail (défaut: 512px)

    Returns:
        Image RGB numpy array ou None si erreur
    """
    try:
        import openslide
        slide = openslide.OpenSlide(str(slide_path))

        # Calculer le ratio pour le thumbnail
        w, h = slide.dimensions
        ratio = max_size / max(w, h)
        thumb_size = (int(w * ratio), int(h * ratio))

        # Extraire le thumbnail
        thumbnail = slide.get_thumbnail(thumb_size)
        thumbnail = np.array(thumbnail.convert('RGB'))

        slide.close()
        return thumbnail

    except ImportError:
        logger.error("OpenSlide non installé. pip install openslide-python openslide-bin")
        return None
    except Exception as e:
        logger.error(f"Erreur lecture thumbnail: {e}")
        return None


def get_wsi_metadata(slide_path: Path) -> Dict[str, Any]:
    """Extrait les métadonnées d'une lame WSI."""
    try:
        import openslide
        slide = openslide.OpenSlide(str(slide_path))

        metadata = {
            "dimensions": slide.dimensions,
            "levels": slide.level_count,
            "level_dimensions": slide.level_dimensions,
            "mpp_x": slide.properties.get('openslide.mpp-x'),
            "mpp_y": slide.properties.get('openslide.mpp-y'),
            "vendor": slide.properties.get('openslide.vendor', 'unknown'),
            "objective": slide.properties.get('openslide.objective-power'),
        }

        slide.close()
        return metadata

    except Exception as e:
        return {"error": str(e)}


def calculate_severity_score(type_counts: Dict[int, int]) -> float:
    """
    Calcule le score de gravité basé sur la distribution des types cellulaires.

    Score = 0.7 * ratio_neoplastic + 0.2 * ratio_dead + 0.1 * density_factor

    Returns:
        Score entre 0 et 1
    """
    total = sum(type_counts.values())
    if total == 0:
        return 0.0

    neoplastic = type_counts.get(1, 0)  # Type 1 = Neoplastic
    dead = type_counts.get(4, 0)  # Type 4 = Dead

    ratio_neo = neoplastic / total
    ratio_dead = dead / total

    # Density factor: plus de noyaux = plus suspect (normalisé)
    density_factor = min(total / 100, 1.0)  # Cap à 100 noyaux

    score = 0.7 * ratio_neo + 0.2 * ratio_dead + 0.1 * density_factor
    return min(score, 1.0)


def process_wsi_slide(
    slide_path: Path,
    max_tiles: int = 100,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Traite une lame WSI complète et stocke les résultats par tile.

    Args:
        slide_path: Chemin vers la lame
        max_tiles: Nombre maximum de tiles à traiter
        progress_callback: Fonction de callback pour mise à jour progress

    Returns:
        Dictionnaire avec résultats
    """
    start_time = time.time()

    results = {
        "success": False,
        "tiles_processed": 0,
        "total_nuclei": 0,
        "type_counts": {},
        "elapsed_seconds": 0.0,
        "tiles_per_second": 0.0,
        "error": None,
    }

    # Vérifier que le moteur est chargé
    if state.engine is None:
        results["error"] = "Moteur non chargé - sélectionner un organe d'abord"
        return results

    # Réinitialiser les résultats de tiles
    real_wsi_state.tile_results = []

    try:
        # Initialiser le router
        router = InputRouter(filter_tiles=True)

        # Traiter les tiles
        tiles_processed = 0
        total_nuclei = 0
        type_counts = {}

        for tile in router.process(slide_path, max_tiles=max_tiles):
            # Analyser le tile
            result, preprocessed, error = run_analysis_core(tile.image, use_auto_params=True)

            if error:
                logger.warning(f"Erreur tile ({tile.x}, {tile.y}): {error}")
                continue

            # Compter les noyaux pour ce tile
            instance_map = result.instance_map
            type_map = result.type_map

            unique_ids = np.unique(instance_map)
            unique_ids = unique_ids[unique_ids > 0]

            tile_type_counts = {}
            tile_nuclei = 0

            for nid in unique_ids:
                mask = instance_map == nid
                cell_type = int(np.median(type_map[mask]))
                tile_type_counts[cell_type] = tile_type_counts.get(cell_type, 0) + 1
                type_counts[cell_type] = type_counts.get(cell_type, 0) + 1
                tile_nuclei += 1
                total_nuclei += 1

            # Créer l'overlay pour ce tile
            overlay = create_segmentation_overlay(
                tile.image, instance_map, type_map, alpha=0.4
            )

            # Calculer le score de gravité
            severity = calculate_severity_score(tile_type_counts)

            # Stocker le résultat du tile
            tile_result = TileResult(
                index=tiles_processed,
                x=tile.x,
                y=tile.y,
                image=tile.image.copy(),
                overlay=overlay,
                total_nuclei=tile_nuclei,
                type_counts=tile_type_counts,
                severity_score=severity,
            )
            real_wsi_state.tile_results.append(tile_result)

            tiles_processed += 1

            # Callback de progression
            if progress_callback:
                progress_callback(tiles_processed, max_tiles, time.time() - start_time)

        elapsed = time.time() - start_time

        # Trier les tiles par score de gravité (décroissant)
        real_wsi_state.tile_results.sort(key=lambda t: t.severity_score, reverse=True)

        # Réindexer après tri
        for i, tile in enumerate(real_wsi_state.tile_results):
            tile.index = i

        logger.info(f"Tiles triés par gravité. Top score: {real_wsi_state.tile_results[0].severity_score:.2f}" if real_wsi_state.tile_results else "Aucun tile")

        results.update({
            "success": True,
            "tiles_processed": tiles_processed,
            "total_nuclei": total_nuclei,
            "type_counts": type_counts,
            "elapsed_seconds": elapsed,
            "tiles_per_second": tiles_processed / elapsed if elapsed > 0 else 0,
        })

    except Exception as e:
        results["error"] = str(e)
        logger.exception(f"Erreur traitement WSI: {e}")

    return results


# ==============================================================================
# EXTRACTION DE PATCHES
# ==============================================================================

def extract_patches_2x2(image: np.ndarray) -> List[np.ndarray]:
    """
    Extrait 4 patches 224×224 d'une image 256×256 (grille 2×2 avec chevauchement).
    """
    h, w = image.shape[:2]
    if h != PANNUKE_SIZE or w != PANNUKE_SIZE:
        raise ValueError(f"Image doit être {PANNUKE_SIZE}×{PANNUKE_SIZE}, reçu {w}×{h}")

    patches = []
    for x, y in PATCH_POSITIONS:
        patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
        patches.append(patch)

    return patches


# ==============================================================================
# STITCHING WSI
# ==============================================================================

def extract_nuclei_info(
    instance_map: np.ndarray,
    type_map: np.ndarray,
    patch_index: int,
) -> List[NucleusInfo]:
    """
    Extrait les informations de chaque noyau d'un patch.
    Détermine si le centroïde est dans la zone valide.
    """
    nuclei = []
    patch_offset_x, patch_offset_y = PATCH_POSITIONS[patch_index]
    valid_zone = VALID_ZONES_PATCH[patch_index]  # (y_min, y_max, x_min, x_max)

    # Trouver tous les IDs de noyaux
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]  # Exclure le fond (0)

    for nucleus_id in unique_ids:
        mask = instance_map == nucleus_id

        # Centroïde en coordonnées patch
        coords = np.where(mask)
        if len(coords[0]) == 0:
            continue

        cy_patch = int(np.mean(coords[0]))
        cx_patch = int(np.mean(coords[1]))

        # Centroïde en coordonnées image
        cy_image = cy_patch + patch_offset_y
        cx_image = cx_patch + patch_offset_x

        # Type cellulaire (valeur majoritaire dans le masque)
        cell_type = int(np.median(type_map[mask]))

        # Aire
        area = int(np.sum(mask))

        # Vérifier si dans zone valide (coordonnées patch)
        y_min, y_max, x_min, x_max = valid_zone
        in_valid = (y_min <= cy_patch < y_max) and (x_min <= cx_patch < x_max)

        nuclei.append(NucleusInfo(
            id=int(nucleus_id),
            centroid_patch=(cy_patch, cx_patch),
            centroid_image=(cy_image, cx_image),
            cell_type=cell_type,
            area_pixels=area,
            in_valid_zone=in_valid,
        ))

    return nuclei


def stitch_segmentation_maps() -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruit les cartes de segmentation 256×256 à partir des 4 patches.
    Utilise uniquement les noyaux dans les zones valides.

    Returns:
        (instance_map_256, type_map_256)
    """
    instance_map = np.zeros((PANNUKE_SIZE, PANNUKE_SIZE), dtype=np.int32)
    type_map = np.zeros((PANNUKE_SIZE, PANNUKE_SIZE), dtype=np.int32)

    global_id = 1  # ID global pour les noyaux stitchés

    for patch in wsi_state.patches:
        if not patch.is_analyzed or patch.result is None:
            continue

        patch_offset_x, patch_offset_y = patch.position
        patch_inst = patch.result.instance_map

        # Pour chaque noyau valide
        for nucleus in patch.nuclei:
            if not nucleus.in_valid_zone:
                continue

            # Masque du noyau dans le patch
            mask_patch = patch_inst == nucleus.id

            # Coordonnées dans le patch
            coords = np.where(mask_patch)
            if len(coords[0]) == 0:
                continue

            # Type HoVer-Net
            nucleus_type = nucleus.cell_type

            # Transférer vers l'image 256×256
            for py, px in zip(coords[0], coords[1]):
                iy = py + patch_offset_y
                ix = px + patch_offset_x

                if 0 <= iy < PANNUKE_SIZE and 0 <= ix < PANNUKE_SIZE:
                    instance_map[iy, ix] = global_id
                    type_map[iy, ix] = nucleus_type

            global_id += 1

    return instance_map, type_map


def create_stitched_overlay(
    source_image: np.ndarray,
    instance_map: np.ndarray,
    type_map: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Crée l'overlay de segmentation pour l'image stitchée 256×256."""
    return create_segmentation_overlay(source_image, instance_map, type_map, alpha)


# ==============================================================================
# FONCTIONS D'ANALYSE
# ==============================================================================

def process_uploaded_image(image: np.ndarray) -> Tuple[
    List[Tuple[np.ndarray, str]],  # gallery items
    np.ndarray,  # selected patch
    np.ndarray,  # patch overlay
    str,  # patch metrics
    np.ndarray,  # stitched overlay (WSI)
    str,  # wsi metrics
    str,  # status
]:
    """
    Traite une image uploadée: extraction + analyse + stitching.

    Returns:
        (gallery, selected_patch, patch_overlay, patch_metrics, stitched_overlay, wsi_metrics, status)
    """
    empty_patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    empty_wsi = np.zeros((PANNUKE_SIZE, PANNUKE_SIZE, 3), dtype=np.uint8)

    # Vérifier que le moteur est chargé
    if state.engine is None:
        return [], empty_patch, empty_patch, "", empty_wsi, "", "❌ Moteur non chargé — Sélectionner un organe d'abord"

    # Vérifier la taille de l'image
    if image is None:
        return [], empty_patch, empty_patch, "", empty_wsi, "", "❌ Aucune image"

    h, w = image.shape[:2]
    if h != PANNUKE_SIZE or w != PANNUKE_SIZE:
        return [], empty_patch, empty_patch, "", empty_wsi, "", f"❌ Taille invalide: {w}×{h} (attendu: {PANNUKE_SIZE}×{PANNUKE_SIZE})"

    # Réinitialiser l'état
    wsi_state.clear()
    wsi_state.source_image = image
    wsi_state.source_filename = "uploaded_image.png"

    # Extraire les 4 patches
    logger.info(f"Extraction de 4 patches {PATCH_SIZE}×{PATCH_SIZE} (grille 2×2)")
    patch_images = extract_patches_2x2(image)

    # Créer les PatchInfo
    for i, (patch_img, pos, name) in enumerate(zip(patch_images, PATCH_POSITIONS, PATCH_NAMES)):
        patch = PatchInfo(
            index=i,
            name=name,
            position=pos,
            image=patch_img,
        )
        wsi_state.patches.append(patch)

    # Analyser tous les patches automatiquement
    logger.info("Analyse automatique des 4 patches...")
    n_success = 0

    for patch in wsi_state.patches:
        result, preprocessed, error = run_analysis_core(patch.image, use_auto_params=True)

        if error:
            logger.warning(f"Erreur patch {patch.name}: {error}")
            continue

        patch.result = result
        patch.is_analyzed = True

        # Créer overlay du patch
        overlay = create_segmentation_overlay(
            result.image_rgb,
            result.instance_map,
            result.type_map,
            alpha=0.4,
        )
        patch.overlay = overlay

        # Extraire info noyaux pour stitching
        patch.nuclei = extract_nuclei_info(
            result.instance_map,
            result.type_map,
            patch.index,
        )
        patch.valid_nuclei_count = sum(1 for n in patch.nuclei if n.in_valid_zone)

        logger.info(f"  {patch.name}: {len(patch.nuclei)} noyaux, {patch.valid_nuclei_count} dans zone valide")
        n_success += 1

    # === STITCHING ===
    logger.info("Stitching des segmentations...")
    stitched_inst, stitched_type = stitch_segmentation_maps()
    wsi_state.stitched_instance_map = stitched_inst
    wsi_state.stitched_type_map = stitched_type

    # Overlay stitché
    wsi_state.stitched_overlay = create_stitched_overlay(
        image, stitched_inst, stitched_type, alpha=0.4
    )

    total_stitched = len(np.unique(stitched_inst)) - 1  # -1 pour le fond
    logger.info(f"Stitching terminé: {total_stitched} noyaux uniques")

    # Construire la galerie
    gallery_items = []
    for p in wsi_state.patches:
        thumb = cv2.resize(p.image, (112, 112))
        label = f"{p.name}"
        if p.is_analyzed:
            label = f"✅ {p.valid_nuclei_count}n"
        gallery_items.append((thumb, label))

    # Sélectionner le premier patch
    wsi_state.selected_index = 0
    selected = wsi_state.get_selected()

    # Métriques
    patch_md = format_patch_metrics(selected) if selected else ""
    wsi_md = format_wsi_metrics_stitched()

    status = f"✅ {n_success}/4 patches | {total_stitched} noyaux"

    return (
        gallery_items,
        selected.image if selected else empty_patch,
        selected.overlay if selected and selected.overlay is not None else empty_patch,
        patch_md,
        wsi_state.stitched_overlay if wsi_state.stitched_overlay is not None else empty_wsi,
        wsi_md,
        status,
    )


def on_patch_select(evt: gr.SelectData) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Gère le clic sur un patch dans la galerie.
    """
    empty = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)

    index = evt.index
    if index < 0 or index >= len(wsi_state.patches):
        return empty, empty, "❌ Index invalide"

    wsi_state.selected_index = index
    patch = wsi_state.patches[index]

    if patch.is_analyzed and patch.overlay is not None:
        return patch.image, patch.overlay, format_patch_metrics(patch)

    return patch.image, empty, format_patch_metrics(patch)


# ==============================================================================
# FORMATAGE
# ==============================================================================

# Mapping type index → nom (PanNuke)
TYPE_NAMES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


def format_patch_metrics(patch: Optional[PatchInfo]) -> str:
    """Formate les métriques d'un patch."""
    if patch is None:
        return "*Aucun patch sélectionné*"

    if not patch.is_analyzed or patch.result is None:
        return f"### {patch.name}\n\n*Non analysé*"

    lines = [
        f"### Patch: {patch.name}",
        f"*Position: ({patch.position[0]}, {patch.position[1]})*",
        "",
        f"**Noyaux totaux:** {len(patch.nuclei)}",
        f"**Dans zone valide:** {patch.valid_nuclei_count} ✓",
        "",
    ]

    # Distribution par type (zone valide uniquement)
    type_counts = {}
    for n in patch.nuclei:
        if n.in_valid_zone:
            t = TYPE_NAMES.get(n.cell_type, f"Type{n.cell_type}")
            type_counts[t] = type_counts.get(t, 0) + 1

    if type_counts:
        lines.append("**Distribution (zone valide):**")
        total = sum(type_counts.values())
        for cell_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total if total > 0 else 0
            lines.append(f"- {cell_type}: {count} ({pct:.1f}%)")

    return "\n".join(lines)


def format_wsi_metrics_stitched() -> str:
    """Formate les métriques WSI stitchées (sans doublons)."""
    agg = wsi_state.get_aggregated_metrics_stitched()

    if agg["analyzed"] == 0:
        return "**Patches:** 0 analysés\n\n*En attente d'analyse...*"

    lines = [
        "## 🧩 Métriques WSI (Stitched)",
        "",
        f"**Patches:** {agg['analyzed']} / {agg['total_patches']}",
        f"**Surface:** {agg['total_area_mm2']*1e6:.0f} µm² ({agg['total_area_mm2']:.4f} mm²)",
        "",
        f"### Total Noyaux: {agg['total_nuclei']}",
        f"**Densité:** {agg['density_per_mm2']:.0f} /mm²",
        "",
        "### Distribution Globale",
    ]

    type_counts = agg.get("type_counts", {})
    total = sum(type_counts.values()) if type_counts else 0

    # Convertir indices en noms
    named_counts = {}
    for type_idx, count in type_counts.items():
        name = TYPE_NAMES.get(type_idx, f"Type{type_idx}")
        named_counts[name] = count

    for cell_type, count in sorted(named_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"- **{cell_type}:** {count} ({pct:.1f}%)")

    return "\n".join(lines)


# ==============================================================================
# CHARGEMENT MOTEUR
# ==============================================================================

def load_engine_for_grid(organ: str) -> str:
    """Charge le moteur pour l'organe spécifié."""
    result = load_engine_core(organ, device="cuda")
    if result["success"]:
        return f"✅ Moteur chargé: {organ} ({result['model_type']})"
    return f"❌ Erreur: {result['error']}"


# ==============================================================================
# FONCTIONS GRADIO - MODE WSI RÉEL
# ==============================================================================

def refresh_wsi_list(wsi_dir: str) -> Tuple[gr.Dropdown, str]:
    """Rafraîchit la liste des fichiers WSI."""
    real_wsi_state.wsi_dir = Path(wsi_dir)
    files = real_wsi_state.scan_wsi_folder()

    if not files:
        return gr.Dropdown(choices=[], value=None), f"❌ Aucun fichier WSI trouvé dans {wsi_dir}"

    return gr.Dropdown(choices=files, value=files[0]), f"✅ {len(files)} fichiers WSI trouvés"


def on_wsi_selected(filename: str) -> Tuple[np.ndarray, str]:
    """Appelé quand un fichier WSI est sélectionné."""
    empty_thumb = np.zeros((512, 512, 3), dtype=np.uint8)

    if not filename:
        return empty_thumb, "*Sélectionnez un fichier*"

    slide_path = real_wsi_state.wsi_dir / filename
    real_wsi_state.selected_file = filename

    # Récupérer le thumbnail
    thumbnail = get_wsi_thumbnail(slide_path, max_size=512)
    if thumbnail is None:
        return empty_thumb, f"❌ Erreur lecture thumbnail: {filename}"

    real_wsi_state.thumbnail = thumbnail

    # Récupérer les métadonnées
    metadata = get_wsi_metadata(slide_path)
    if "error" in metadata:
        return thumbnail, f"❌ Erreur métadonnées: {metadata['error']}"

    real_wsi_state.slide_dimensions = metadata["dimensions"]
    real_wsi_state.slide_mpp = float(metadata["mpp_x"]) if metadata["mpp_x"] else None
    real_wsi_state.slide_levels = metadata["levels"]

    # Formatage des infos
    w, h = metadata["dimensions"]
    mpp = f"{float(metadata['mpp_x']):.3f}" if metadata["mpp_x"] else "N/A"

    info_lines = [
        f"### {filename}",
        "",
        f"**Dimensions:** {w:,} × {h:,} pixels",
        f"**MPP:** {mpp} µm/px",
        f"**Niveaux:** {metadata['levels']}",
        f"**Vendor:** {metadata['vendor']}",
        f"**Objectif:** {metadata['objective']}x" if metadata['objective'] else "",
        "",
        f"**Tiles estimés:** ~{int(w/224 * h/224 * 0.3):,} (avec ~30% tissu)",
    ]

    return thumbnail, "\n".join(info_lines)


def build_tile_gallery() -> List[Tuple[np.ndarray, str]]:
    """Construit la galerie de tiles triés par gravité."""
    gallery_items = []

    for tile in real_wsi_state.tile_results:
        # Créer le label avec score et indicateur
        label = f"{tile.severity_label}\n{tile.severity_score:.0%} | {tile.total_nuclei}n"
        gallery_items.append((tile.overlay, label))

    return gallery_items


def on_tile_select(evt: gr.SelectData) -> Tuple[np.ndarray, np.ndarray, str]:
    """Gère le clic sur un tile dans la galerie."""
    empty = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)

    index = evt.index
    if index < 0 or index >= len(real_wsi_state.tile_results):
        return empty, empty, "❌ Index invalide"

    real_wsi_state.selected_tile_index = index
    tile = real_wsi_state.tile_results[index]

    # Formater les métriques du tile
    metrics_lines = [
        f"### Tile #{tile.index + 1}",
        f"**Position:** ({tile.x:,}, {tile.y:,})",
        f"**Score gravité:** {tile.severity_score:.1%} {tile.severity_label}",
        "",
        f"**Noyaux:** {tile.total_nuclei}",
        "",
        "**Distribution:**",
    ]

    total = sum(tile.type_counts.values())
    for type_idx, count in sorted(tile.type_counts.items(), key=lambda x: -x[1]):
        name = TYPE_NAMES.get(type_idx, f"Type{type_idx}")
        pct = 100 * count / total if total > 0 else 0
        metrics_lines.append(f"- {name}: {count} ({pct:.1f}%)")

    return tile.image, tile.overlay, "\n".join(metrics_lines)


def run_wsi_analysis(filename: str, max_tiles: int):
    """
    Lance l'analyse WSI et retourne les résultats formatés.

    Args:
        filename: Nom du fichier WSI sélectionné
        max_tiles: Nombre maximum de tiles à traiter

    Returns:
        (status, timer_display, results_markdown, gallery_items, first_tile_image, first_tile_overlay, first_tile_metrics)
    """
    empty_tile = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    empty_gallery = []
    empty_metrics = "*Aucun tile analysé*"

    if not filename:
        return ("❌ Aucun fichier sélectionné", "00:00", "*Sélectionnez un fichier WSI*",
                empty_gallery, empty_tile, empty_tile, empty_metrics)

    if state.engine is None:
        return ("❌ Moteur non chargé", "00:00", "*Chargez un modèle d'abord*",
                empty_gallery, empty_tile, empty_tile, empty_metrics)

    slide_path = real_wsi_state.wsi_dir / filename

    if not slide_path.exists():
        return (f"❌ Fichier non trouvé: {slide_path}", "00:00", "*Fichier introuvable*",
                empty_gallery, empty_tile, empty_tile, empty_metrics)

    # Lancer le traitement
    real_wsi_state.clear_results()
    real_wsi_state.is_processing = True
    real_wsi_state.processing_start_time = time.time()

    results = process_wsi_slide(slide_path, max_tiles=int(max_tiles))

    real_wsi_state.is_processing = False
    real_wsi_state.processing_elapsed = results.get("elapsed_seconds", 0)
    real_wsi_state.results = results

    # Formatter le temps
    elapsed = results.get("elapsed_seconds", 0)
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    timer_str = f"{minutes:02d}:{seconds:02d}"

    if not results.get("success"):
        return (f"❌ Erreur: {results.get('error')}", timer_str, "*Erreur de traitement*",
                empty_gallery, empty_tile, empty_tile, empty_metrics)

    # Formatter les résultats
    results_md = format_wsi_results(results)
    status = f"✅ Terminé: {results['tiles_processed']} tiles, {results['total_nuclei']} noyaux"

    # Construire la galerie triée
    gallery_items = build_tile_gallery()

    # Sélectionner le premier tile (plus haute gravité)
    if real_wsi_state.tile_results:
        first_tile = real_wsi_state.tile_results[0]
        real_wsi_state.selected_tile_index = 0

        first_metrics = [
            f"### Tile #1 (Top Gravité)",
            f"**Position:** ({first_tile.x:,}, {first_tile.y:,})",
            f"**Score:** {first_tile.severity_score:.1%} {first_tile.severity_label}",
            f"**Noyaux:** {first_tile.total_nuclei}",
        ]
        first_tile_md = "\n".join(first_metrics)

        return (status, timer_str, results_md,
                gallery_items, first_tile.image, first_tile.overlay, first_tile_md)

    return (status, timer_str, results_md,
            empty_gallery, empty_tile, empty_tile, empty_metrics)


def format_wsi_results(results: Dict[str, Any]) -> str:
    """Formate les résultats WSI pour affichage."""
    if not results.get("success"):
        return f"**Erreur:** {results.get('error', 'Inconnue')}"

    lines = [
        "## 📊 Résultats Diagnostic",
        "",
        f"**Tiles analysés:** {results['tiles_processed']}",
        f"**Temps total:** {results['elapsed_seconds']:.1f}s",
        f"**Vitesse:** {results['tiles_per_second']:.1f} tiles/s",
        "",
        f"### 🔬 Total Noyaux: {results['total_nuclei']:,}",
        "",
        "### Distribution par Type",
    ]

    type_counts = results.get("type_counts", {})
    total = sum(type_counts.values()) if type_counts else 0

    # Trier par fréquence
    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])

    for type_idx, count in sorted_types:
        name = TYPE_NAMES.get(type_idx, f"Type{type_idx}")
        pct = 100 * count / total if total > 0 else 0

        # Indicateur visuel pour les types importants
        indicator = ""
        if name == "Neoplastic" and pct > 10:
            indicator = " ⚠️"
        elif name == "Neoplastic" and pct > 30:
            indicator = " 🔴"

        lines.append(f"- **{name}:** {count:,} ({pct:.1f}%){indicator}")

    # Ratio diagnostic simple
    neoplastic = type_counts.get(1, 0)
    if total > 0:
        neo_ratio = neoplastic / total
        lines.extend([
            "",
            "---",
            "### 🎯 Indicateurs",
            f"**Ratio Néoplasique:** {neo_ratio*100:.1f}%",
        ])
        if neo_ratio > 0.3:
            lines.append("**⚠️ Attention:** Ratio néoplasique élevé")
        elif neo_ratio < 0.05:
            lines.append("**✅ Normal:** Ratio néoplasique bas")

    return "\n".join(lines)


# ==============================================================================
# INTERFACE GRADIO
# ==============================================================================

def create_grid_ui(wsi_dir: str = DEFAULT_WSI_DIR):
    """Crée l'interface unifiée avec onglets Patch et WSI."""

    # Initialiser le dossier WSI
    real_wsi_state.wsi_dir = Path(wsi_dir)
    real_wsi_state.scan_wsi_folder()

    with gr.Blocks(
        title="CellViT-Optimus — Diagnostic WSI",
        theme=gr.themes.Soft(),
        css="""
        .timer-display {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 8px;
        }
        """
    ) as app:

        gr.Markdown("""
        # 🔬 CellViT-Optimus — Diagnostic Histopathologique

        Analyse de noyaux cellulaires par segmentation HoVer-Net.
        """)

        # === CONFIGURATION COMMUNE ===
        with gr.Row():
            with gr.Column(scale=2):
                organ_dropdown = gr.Dropdown(
                    choices=ORGAN_CHOICES,
                    value="Lung",
                    label="🏥 Organe",
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("🚀 Charger Modèle", variant="primary")
            with gr.Column(scale=2):
                model_status = gr.Textbox(
                    label="Status Moteur",
                    interactive=False,
                    value="⏳ Aucun modèle chargé",
                )

        # === ONGLETS ===
        with gr.Tabs():

            # =================================================================
            # ONGLET 1: MODE WSI RÉEL
            # =================================================================
            with gr.TabItem("🔬 Lames WSI", id="wsi_tab"):
                gr.Markdown(f"""
                ### Traitement de Lames Entières
                Dossier: `{wsi_dir}`
                """)

                with gr.Row():
                    # --- Colonne gauche: Sélection fichier ---
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Sélection Lame")

                        wsi_dir_input = gr.Textbox(
                            label="Dossier WSI",
                            value=str(wsi_dir),
                            interactive=True,
                        )
                        refresh_btn = gr.Button("🔄 Rafraîchir", size="sm")
                        wsi_file_dropdown = gr.Dropdown(
                            choices=real_wsi_state.available_files,
                            value=real_wsi_state.available_files[0] if real_wsi_state.available_files else None,
                            label="Fichier WSI",
                            interactive=True,
                        )
                        wsi_scan_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value=f"✅ {len(real_wsi_state.available_files)} fichiers" if real_wsi_state.available_files else "❌ Aucun fichier",
                        )

                        gr.Markdown("#### 2. Paramètres")
                        max_tiles_slider = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=50,
                            step=10,
                            label="Nombre max de tiles",
                            info="Plus de tiles = plus précis mais plus lent",
                        )

                        run_wsi_btn = gr.Button(
                            "▶️ Lancer Analyse",
                            variant="primary",
                            size="lg",
                        )

                        gr.Markdown("#### ⏱️ Timer")
                        timer_display = gr.Textbox(
                            value="00:00",
                            label="Temps écoulé",
                            interactive=False,
                            elem_classes=["timer-display"],
                        )
                        wsi_analysis_status = gr.Textbox(
                            label="Status Analyse",
                            interactive=False,
                        )

                    # --- Colonne centrale: Thumbnail + Résultats ---
                    with gr.Column(scale=1):
                        gr.Markdown("#### Aperçu Lame")
                        wsi_thumbnail = gr.Image(
                            label="Thumbnail",
                            height=250,
                        )
                        wsi_info = gr.Markdown(
                            value="*Sélectionnez un fichier WSI*",
                        )

                        gr.Markdown("#### 📊 Résultats Diagnostic")
                        wsi_results = gr.Markdown(
                            value="*Lancez une analyse pour voir les résultats*",
                        )

                    # --- Colonne droite: Galerie Tiles triés par gravité (VERTICAL) ---
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🔥 Tiles par Gravité")
                        tile_gallery = gr.Gallery(
                            label="Cliquer pour voir le détail",
                            columns=1,  # VERTICAL: 1 colonne
                            rows=5,
                            height=500,
                            object_fit="contain",
                            allow_preview=False,
                        )

                # === LIGNE 2: Détail du tile sélectionné ===
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🔍 Tile Sélectionné")
                    with gr.Column(scale=1):
                        selected_tile_image = gr.Image(
                            label="Image Originale",
                            height=224,
                        )
                    with gr.Column(scale=1):
                        selected_tile_overlay = gr.Image(
                            label="Segmentation",
                            height=224,
                        )
                    with gr.Column(scale=1):
                        selected_tile_metrics = gr.Markdown(
                            value="*Cliquez sur un tile dans la galerie*",
                        )

                # === ÉVÉNEMENTS WSI ===
                refresh_btn.click(
                    fn=refresh_wsi_list,
                    inputs=[wsi_dir_input],
                    outputs=[wsi_file_dropdown, wsi_scan_status],
                )

                wsi_file_dropdown.change(
                    fn=on_wsi_selected,
                    inputs=[wsi_file_dropdown],
                    outputs=[wsi_thumbnail, wsi_info],
                )

                run_wsi_btn.click(
                    fn=run_wsi_analysis,
                    inputs=[wsi_file_dropdown, max_tiles_slider],
                    outputs=[
                        wsi_analysis_status,
                        timer_display,
                        wsi_results,
                        tile_gallery,
                        selected_tile_image,
                        selected_tile_overlay,
                        selected_tile_metrics,
                    ],
                )

                # Clic sur tile dans galerie
                tile_gallery.select(
                    fn=on_tile_select,
                    outputs=[selected_tile_image, selected_tile_overlay, selected_tile_metrics],
                )

                # Charger le thumbnail au démarrage si un fichier est sélectionné
                app.load(
                    fn=on_wsi_selected,
                    inputs=[wsi_file_dropdown],
                    outputs=[wsi_thumbnail, wsi_info],
                )

            # =================================================================
            # ONGLET 2: MODE PATCH (Simulation)
            # =================================================================
            with gr.TabItem("📋 Patches 256×256", id="patch_tab"):
                gr.Markdown("""
                ### Mode Simulation (Images PanNuke)
                Upload d'images 256×256 avec extraction de 4 patches et stitching.
                """)

                with gr.Row():
                    # --- Colonne gauche: Upload + Grille ---
                    with gr.Column(scale=1):
                        gr.Markdown("#### Image Source")

                        input_image = gr.Image(
                            label="Upload PanNuke 256×256",
                            type="numpy",
                            height=180,
                        )
                        analysis_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("#### Grille Patches (2×2)")
                        gallery = gr.Gallery(
                            label="Cliquer pour sélectionner",
                            columns=2,
                            rows=2,
                            height=240,
                            object_fit="contain",
                            allow_preview=False,
                        )

                    # --- Colonne centrale: Patch sélectionné ---
                    with gr.Column(scale=1):
                        gr.Markdown("#### Patch Sélectionné")

                        selected_image = gr.Image(
                            label="Original",
                            height=224,
                        )
                        patch_overlay = gr.Image(
                            label="Segmentation",
                            height=224,
                        )
                        patch_metrics = gr.Markdown(
                            value="*Sélectionnez un patch*",
                        )

                    # --- Colonne droite: WSI Stitched ---
                    with gr.Column(scale=1):
                        gr.Markdown("#### WSI Reconstituée")

                        stitched_overlay = gr.Image(
                            label="Segmentation Stitchée",
                            height=280,
                        )
                        wsi_metrics = gr.Markdown(
                            value="*Uploader une image*",
                        )

                # === ÉVÉNEMENTS PATCH ===
                input_image.upload(
                    fn=process_uploaded_image,
                    inputs=[input_image],
                    outputs=[
                        gallery,
                        selected_image,
                        patch_overlay,
                        patch_metrics,
                        stitched_overlay,
                        wsi_metrics,
                        analysis_status,
                    ],
                )

                gallery.select(
                    fn=on_patch_select,
                    outputs=[selected_image, patch_overlay, patch_metrics],
                )

        # === ÉVÉNEMENT COMMUN: Chargement modèle ===
        load_btn.click(
            fn=load_engine_for_grid,
            inputs=[organ_dropdown],
            outputs=[model_status],
        )

    return app


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CellViT-Optimus Diagnostic WSI")
    parser.add_argument("--organ", type=str, default=None,
                        help="Organe à précharger (ex: Lung, Breast)")
    parser.add_argument("--port", type=int, default=7861,
                        help="Port Gradio (défaut: 7861)")
    parser.add_argument("--share", action="store_true",
                        help="Créer un lien public Gradio")
    parser.add_argument("--preload", action="store_true",
                        help="Précharger le backbone au démarrage")
    parser.add_argument("--wsi_dir", type=str, default=DEFAULT_WSI_DIR,
                        help=f"Dossier contenant les lames WSI (défaut: {DEFAULT_WSI_DIR})")
    args = parser.parse_args()

    # Préchargement optionnel
    if args.preload:
        logger.info("Préchargement du backbone...")
        preload_backbone_core(device="cuda")

    # Chargement organe si spécifié
    if args.organ:
        logger.info(f"Chargement du modèle pour {args.organ}...")
        load_engine_core(args.organ, device="cuda")

    # Lancer l'interface
    logger.info(f"Dossier WSI: {args.wsi_dir}")
    app = create_grid_ui(wsi_dir=args.wsi_dir)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
