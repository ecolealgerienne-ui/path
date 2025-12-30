"""
CellViT-Optimus UI Core — Opérations moteur partagées.

Ce module contient la logique métier partagée pour:
- Chargement du moteur
- Analyse d'image
- Gestion des clics
- Changement d'organe

Les fonctions retournent des données brutes (pas de formatage).
Le formatage est géré par src/ui/formatters/.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from src.ui.inference_engine import CellVitEngine, AnalysisResult
from src.ui.organ_config import get_model_for_organ
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

def load_engine_core(organ: str, device: str = "cuda") -> Dict[str, Any]:
    """
    Charge le moteur d'inférence.

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
        logger.info(f"Loading engine for organ '{organ}' on {device}...")

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
    overlay: Optional[np.ndarray] = None
    contours: Optional[np.ndarray] = None
    chart: Optional[np.ndarray] = None
    debug: Optional[np.ndarray] = None
    anomaly_overlay: Optional[np.ndarray] = None
    phase3_overlay: Optional[np.ndarray] = None
    phase3_debug: Optional[np.ndarray] = None
    error: Optional[str] = None


def analyze_image_core(
    image: np.ndarray,
    np_threshold: float,
    min_size: int,
    beta: float,
    min_distance: int,
) -> AnalysisOutput:
    """
    Analyse une image et retourne le résultat + visualisations.

    Returns:
        AnalysisOutput avec toutes les données et visualisations.
    """
    empty = np.zeros((224, 224, 3), dtype=np.uint8)
    empty_debug = np.zeros((100, 400, 3), dtype=np.uint8)
    empty_phase3_debug = np.zeros((80, 400, 3), dtype=np.uint8)

    if state.engine is None:
        return AnalysisOutput(
            success=False,
            error="Moteur non chargé",
            overlay=empty,
            contours=empty,
            chart=empty,
            debug=empty_debug,
            anomaly_overlay=empty,
            phase3_overlay=empty,
            phase3_debug=empty_phase3_debug,
        )

    if image is None:
        return AnalysisOutput(
            success=False,
            error="Aucune image",
            overlay=empty,
            contours=empty,
            chart=empty,
            debug=empty_debug,
            anomaly_overlay=empty,
            phase3_overlay=empty,
            phase3_debug=empty_phase3_debug,
        )

    # Vérification taille 224×224
    h, w = image.shape[:2]
    if h != 224 or w != 224:
        return AnalysisOutput(
            success=False,
            error=f"Image {w}×{h} pixels — Veuillez charger une image 224×224",
            overlay=empty,
            contours=empty,
            chart=empty,
            debug=empty_debug,
            anomaly_overlay=empty,
            phase3_overlay=empty,
            phase3_debug=empty_phase3_debug,
        )

    try:
        # Paramètres watershed personnalisés
        params = {
            "np_threshold": np_threshold,
            "min_size": int(min_size),
            "beta": beta,
            "min_distance": int(min_distance),
        }

        # Analyse
        result = state.engine.analyze(
            image,
            watershed_params=params,
            compute_morphometry=True,
            compute_uncertainty=True,
        )

        state.current_result = result

        # Visualisations
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
