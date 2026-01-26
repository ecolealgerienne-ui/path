"""
CellViT-Optimus UI Core — Opérations moteur partagées.

Ce module contient la logique métier partagée pour:
- Chargement du moteur
- Analyse d'image
- Gestion des clics
- Changement d'organe

Les fonctions retournent des données brutes (pas de formatage).
Le formatage est géré par src/ui/formatters/.

Note: Intégration InputRouter (2026-01-26)
    - Supporte automatiquement les images 256×256 (PanNuke) → center crop 224×224
    - Les images 224×224 passent directement (aucune transformation)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from src.ui.inference_engine import CellVitEngine, AnalysisResult
from src.ui.organ_config import get_model_for_organ
from src.wsi import transform_pannuke_to_224, TARGET_SIZE, PANNUKE_SIZE
from src.ui.visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    create_type_distribution_chart,
    create_debug_panel_enhanced,
    create_anomaly_overlay,
    create_phase3_combined_overlay,
    create_spatial_debug_panel,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# ÉTAT GLOBAL PARTAGÉ
# ==============================================================================

@dataclass
class UIState:
    """État global partagé entre les interfaces."""
    engine: Optional[CellVitEngine] = None
    current_result: Optional[AnalysisResult] = None
    is_loading: bool = False


# Instance unique de l'état (singleton)
state = UIState()


# ==============================================================================
# OPÉRATIONS MOTEUR
# ==============================================================================


def preload_backbone_core(device: str = "cuda") -> Dict[str, Any]:
    """
    Précharge le backbone H-optimus-0 et OrganHead au démarrage.

    Cette fonction charge les modèles partagés AVANT qu'un organe soit sélectionné.
    Avantage: Réduit le temps d'attente lors du premier choix d'organe.

    Returns:
        Dict avec:
            - success: bool
            - backbone_loaded: bool
            - organ_head_loaded: bool
            - error: Optional[str]
    """
    try:
        state.is_loading = True
        logger.info("Preloading backbone models (H-optimus-0 + OrganHead)...")

        # Créer le moteur sans organe (backbone + OrganHead seulement)
        state.engine = CellVitEngine(
            device=device,
            organ=None,  # Pas d'organe → pas de HoVer-Net
            load_backbone=True,
            load_organ_head=True,
        )

        state.is_loading = False
        logger.info("Backbone preload complete")

        return {
            "success": True,
            "backbone_loaded": state.engine.backbone is not None,
            "organ_head_loaded": state.engine.organ_head is not None,
            "error": None,
        }

    except Exception as e:
        state.is_loading = False
        logger.error(f"Error preloading backbone: {e}")
        return {
            "success": False,
            "backbone_loaded": False,
            "organ_head_loaded": False,
            "error": str(e),
        }


def load_engine_core(organ: str, device: str = "cuda") -> Dict[str, Any]:
    """
    Charge le moteur d'inférence.

    Si le backbone est déjà préchargé (via preload_backbone_core),
    seul le HoVer-Net est chargé (rapide).

    Returns:
        Dict avec:
            - success: bool
            - organ: str
            - model_type: str ("dédié" ou "famille X")
            - device: str
            - error: Optional[str]
    """
    try:
        state.is_loading = True
        organ_info = get_model_for_organ(organ)

        # Si le moteur existe déjà avec backbone préchargé
        if state.engine is not None and state.engine.backbone is not None:
            logger.info(f"Backbone already loaded, switching to organ '{organ}'...")
            state.engine.change_organ(organ)
        else:
            # Chargement complet (backbone + OrganHead + HoVer-Net)
            logger.info(f"Loading full engine for organ '{organ}' on {device}...")
            state.engine = CellVitEngine(
                device=device,
                organ=organ,
                load_backbone=True,
                load_organ_head=True,
            )

        state.is_loading = False
        model_type = "dédié" if organ_info["is_dedicated"] else f"famille {organ_info['family']}"

        return {
            "success": True,
            "organ": organ,
            "model_type": model_type,
            "device": device,
            "error": None,
        }

    except Exception as e:
        state.is_loading = False
        logger.error(f"Error loading engine: {e}")
        return {
            "success": False,
            "organ": organ,
            "model_type": None,
            "device": device,
            "error": str(e),
        }


def change_organ_core(organ: str) -> Dict[str, Any]:
    """
    Change l'organe actif (recharge le modèle HoVer-Net si nécessaire).

    Returns:
        Dict avec:
            - success: bool
            - organ: str
            - model_type: str
            - watershed_params: Optional[dict]
            - error: Optional[str]
    """
    if state.engine is None:
        return {
            "success": False,
            "organ": organ,
            "model_type": None,
            "watershed_params": None,
            "error": "Moteur non chargé",
        }

    try:
        state.engine.change_organ(organ)
        organ_info = get_model_for_organ(organ)
        model_type = "dédié" if organ_info["is_dedicated"] else f"famille {organ_info['family']}"

        return {
            "success": True,
            "organ": organ,
            "model_type": model_type,
            "watershed_params": state.engine.watershed_params,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error changing organ: {e}")
        return {
            "success": False,
            "organ": organ,
            "model_type": None,
            "watershed_params": None,
            "error": str(e),
        }


@dataclass
class AnalysisOutput:
    """Résultat d'analyse avec visualisations."""
    success: bool
    result: Optional[AnalysisResult] = None
    preprocessed_image: Optional[np.ndarray] = None  # Image après preprocessing (224×224)
    overlay: Optional[np.ndarray] = None
    contours: Optional[np.ndarray] = None
    chart: Optional[np.ndarray] = None
    debug: Optional[np.ndarray] = None
    anomaly_overlay: Optional[np.ndarray] = None
    phase3_overlay: Optional[np.ndarray] = None
    phase3_debug: Optional[np.ndarray] = None
    error: Optional[str] = None


# ==============================================================================
# FONCTION D'ANALYSE PARTAGÉE (sans visualisations)
# ==============================================================================

def run_analysis_core(
    image: np.ndarray,
    use_auto_params: bool = True,
    watershed_params: Optional[Dict] = None,
) -> Tuple[Optional[AnalysisResult], Optional[np.ndarray], Optional[str]]:
    """
    Exécute l'analyse d'image et retourne le résultat brut.

    Fonction partagée entre les deux UIs (R&D et Pathologiste).
    Ne génère PAS de visualisations - chaque UI crée les siennes.

    Tailles acceptées:
        - 224×224: Direct (aucune transformation)
        - 256×256: PanNuke → Center crop automatique vers 224×224

    Args:
        image: Image RGB (H, W, 3)
        use_auto_params: Si True, utilise les params de organ_config.py
        watershed_params: Params manuels (ignorés si use_auto_params=True)

    Returns:
        (AnalysisResult, preprocessed_image, None) si succès
        (None, None, error_message) si erreur
    """
    if state.engine is None:
        return None, None, "Moteur non chargé"

    if image is None:
        return None, None, "Aucune image"

    # === PREPROCESSING ADAPTATIF (InputRouter Integration) ===
    h, w = image.shape[:2]

    if h == PANNUKE_SIZE and w == PANNUKE_SIZE:
        # PanNuke 256×256 → Center crop 224×224
        logger.info(f"InputRouter: PanNuke {PANNUKE_SIZE}×{PANNUKE_SIZE} → center crop {TARGET_SIZE}×{TARGET_SIZE}")
        image = transform_pannuke_to_224(image, method="center_crop")
        h, w = image.shape[:2]

    elif h == TARGET_SIZE and w == TARGET_SIZE:
        # Déjà 224×224 → Direct
        pass

    else:
        # Taille non supportée
        return None, None, f"Image {w}×{h} — Tailles acceptées: 224×224 ou 256×256 (PanNuke)"

    try:
        # Paramètres watershed
        params = None if use_auto_params else watershed_params

        # Analyse
        result = state.engine.analyze(
            image,
            watershed_params=params,
            compute_morphometry=True,
            compute_uncertainty=True,
        )

        state.current_result = result
        return result, image, None

    except Exception as e:
        logger.error(f"Erreur analyse: {e}")
        return None, None, f"Erreur: {e}"


def analyze_image_core(
    image: np.ndarray,
    np_threshold: float,
    min_size: int,
    beta: float,
    min_distance: int,
    use_auto_params: bool = True,
) -> AnalysisOutput:
    """
    Analyse une image et retourne le résultat + visualisations R&D.

    Utilise run_analysis_core() pour l'analyse (logique partagée).
    Ajoute les visualisations R&D spécifiques (debug panels, anomalies, etc.).

    Args:
        image: Image RGB (H, W, 3)
        np_threshold: Seuil NP (utilisé si use_auto_params=False)
        min_size: Taille min (utilisé si use_auto_params=False)
        beta: Beta (utilisé si use_auto_params=False)
        min_distance: Distance min (utilisé si use_auto_params=False)
        use_auto_params: Si True, utilise les params optimisés de organ_config.py
                         Si False, utilise les valeurs des sliders

    Returns:
        AnalysisOutput avec toutes les données et visualisations.
    """
    empty = np.zeros((224, 224, 3), dtype=np.uint8)
    empty_debug = np.zeros((100, 400, 3), dtype=np.uint8)
    empty_phase3_debug = np.zeros((80, 400, 3), dtype=np.uint8)

    # Préparer les params manuels si nécessaire
    watershed_params = None
    if not use_auto_params:
        watershed_params = {
            "np_threshold": np_threshold,
            "min_size": int(min_size),
            "beta": beta,
            "min_distance": int(min_distance),
        }

    # Analyse via fonction partagée
    result, preprocessed_image, error = run_analysis_core(image, use_auto_params, watershed_params)

    if error:
        return AnalysisOutput(
            success=False,
            error=error,
            preprocessed_image=None,
            overlay=empty,
            contours=empty,
            chart=empty,
            debug=empty_debug,
            anomaly_overlay=empty,
            phase3_overlay=empty,
            phase3_debug=empty_phase3_debug,
        )

    try:
        # Visualisations R&D
        overlay = create_segmentation_overlay(
            result.image_rgb,
            result.instance_map,
            result.type_map,
            alpha=0.4,
        )

        contours = create_contour_overlay(
            result.image_rgb,
            result.instance_map,
            result.type_map,
            thickness=1,
        )

        # Chart distribution
        if result.morphometry:
            chart = create_type_distribution_chart(result.morphometry.type_counts)
        else:
            chart = np.zeros((200, 300, 3), dtype=np.uint8)

        # Debug panel amélioré (Phase 2)
        debug = create_debug_panel_enhanced(
            result.np_pred,
            result.hv_pred,
            result.instance_map,
            n_fusions=result.n_fusions,
            n_over_seg=result.n_over_seg,
        )

        # Overlay anomalies (Phase 2)
        anomaly_overlay = create_anomaly_overlay(
            result.image_rgb,
            result.instance_map,
            result.fusion_ids,
            result.over_seg_ids,
        )

        # Phase 3: Overlay combiné
        phase3_overlay = empty.copy()
        phase3_debug = empty_phase3_debug.copy()

        if result.spatial_analysis:
            sa = result.spatial_analysis
            phase3_overlay = create_phase3_combined_overlay(
                result.image_rgb,
                result.instance_map,
                hotspot_ids=result.hotspot_ids,
                mitosis_ids=result.mitosis_candidate_ids,
                mitosis_scores=sa.mitosis_scores,
                heterogeneous_ids=sa.heterogeneous_nuclei_ids,
            )

            phase3_debug = create_spatial_debug_panel(
                pleomorphism_score=result.pleomorphism_score,
                pleomorphism_description=result.pleomorphism_description,
                n_hotspots=result.n_hotspots,
                n_mitosis_candidates=result.n_mitosis_candidates,
                n_heterogeneous=result.n_heterogeneous_nuclei,
                mean_neighbors=result.mean_neighbors,
                mean_entropy=result.mean_chromatin_entropy,
            )

        return AnalysisOutput(
            success=True,
            result=result,
            preprocessed_image=preprocessed_image,
            overlay=overlay,
            contours=contours,
            chart=chart,
            debug=debug,
            anomaly_overlay=anomaly_overlay,
            phase3_overlay=phase3_overlay,
            phase3_debug=phase3_debug,
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return AnalysisOutput(
            success=False,
            preprocessed_image=None,
            error=str(e),
            overlay=empty,
            contours=empty,
            chart=empty,
            debug=empty_debug,
            anomaly_overlay=empty,
            phase3_overlay=empty,
            phase3_debug=empty_phase3_debug,
        )


def on_image_click_core(x: int, y: int) -> Dict[str, Any]:
    """
    Gère le clic sur l'image pour identifier un noyau.

    Returns:
        Dict avec:
            - found: bool
            - nucleus_id: Optional[int]
            - cell_type: Optional[str]
            - area_um2: Optional[float]
            - circularity: Optional[float]
            - position: Optional[Tuple[int, int]]
    """
    if state.current_result is None:
        return {"found": False, "error": "Aucune analyse"}

    result = state.current_result

    # Vérifier les limites
    if x < 0 or y < 0 or x >= 224 or y >= 224:
        return {"found": False, "error": "Clic hors limites"}

    # Obtenir l'ID du noyau
    nucleus_id = result.instance_map[y, x]

    if nucleus_id == 0:
        return {"found": False, "clicked_background": True}

    # Récupérer les infos du noyau
    if result.nuclei_data and nucleus_id in result.nuclei_data:
        nuc = result.nuclei_data[nucleus_id]
        return {
            "found": True,
            "nucleus_id": nucleus_id,
            "cell_type": nuc.get("type_name", "Unknown"),
            "area_um2": nuc.get("area_um2", 0),
            "circularity": nuc.get("circularity", 0),
            "position": (x, y),
        }

    # Fallback: juste l'ID
    return {
        "found": True,
        "nucleus_id": nucleus_id,
        "cell_type": "Unknown",
        "area_um2": None,
        "circularity": None,
        "position": (x, y),
    }
