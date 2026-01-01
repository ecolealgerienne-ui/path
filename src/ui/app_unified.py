"""
CellViT-Optimus — Interface Unifiée Professionnelle V2.

Design "Clinical-Grade" avec:
- Header sombre avec status LED
- Visual Stage (images en majesté)
- Barre d'actions unifiée
- Ruban Diagnostic dynamique
- Deep Dive en 3 colonnes

Usage:
    python -m src.ui.app_unified --organ Lung --port 7860
"""

import argparse
import logging
import numpy as np
import gradio as gr
from typing import Optional, Tuple, List

from src.ui.core import (
    state,
    preload_backbone_core,
    load_engine_core,
    analyze_image_core,
    change_organ_core,
    export_pdf_core,
    export_nuclei_csv_core,
    export_summary_csv_core,
    export_json_core,
)

from src.ui.inference_engine import ORGAN_CHOICES
from src.ui.organ_config import ORGANS, get_model_for_organ

from src.ui.formatters import (
    # R&D formatters
    format_metrics_rnd,
    format_alerts_rnd,
    format_organ_change_rnd,
    # Clinical formatters
    format_identification_clinical,
    format_metrics_clinical,
    format_alerts_clinical,
    format_organ_change_clinical,
    format_confidence_badge,
)

from src.ui.visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    create_type_distribution_chart,
    create_debug_panel_enhanced,
    create_anomaly_overlay,
    create_phase3_combined_overlay,
    create_spatial_debug_panel,
    create_hotspot_overlay,
    create_mitosis_overlay,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# PROFILS
# ==============================================================================

PROFILES = {
    "🔬 R&D Cockpit": {
        "description": "Vue technique complète avec debug et paramètres",
        "show_debug": True,
        "show_params": True,
        "clinical_language": False,
    },
    "🩺 Pathologiste": {
        "description": "Vue clinique simplifiée pour l'aide au diagnostic",
        "show_debug": False,
        "show_params": False,
        "clinical_language": True,
    },
}

# ==============================================================================
# ÉTAT GLOBAL POUR UI
# ==============================================================================

# Track si une image est chargée
_image_loaded = False
_model_loaded = False


def get_organ_info():
    """Retourne les infos de l'organe actuel."""
    if state.engine is None:
        return None, None, False
    return state.engine.organ, state.engine.family, state.engine.is_dedicated_model


def check_analyze_ready() -> Tuple[bool, str]:
    """
    Vérifie si l'analyse peut être lancée.

    Returns:
        (is_ready, status_message)
    """
    global _model_loaded, _image_loaded

    if not _model_loaded:
        return False, "⏳ Sélectionnez un organe"
    if not _image_loaded:
        return False, "⏳ Chargez une image"
    return True, "✅ Prêt"


def on_image_change(image):
    """Callback quand l'image change."""
    global _image_loaded
    _image_loaded = image is not None

    is_ready, _ = check_analyze_ready()
    return gr.update(interactive=is_ready, variant="primary" if is_ready else "secondary")


def load_engine(organ: str):
    """
    Charge ou change l'organe du moteur.

    Returns:
        Tuple[str, gr.update, gr.update]: (status_message, status_led, analyze_button)
    """
    global _model_loaded

    # Si aucun organe sélectionné
    if not organ:
        _model_loaded = False
        return (
            "⏳ Sélectionnez un organe",
            "<span style='color: #ffc107; font-size: 1.5em;'>●</span>",  # LED jaune
            gr.update(interactive=False, variant="secondary")
        )

    # Si même organe déjà chargé
    if state.engine is not None and state.engine.organ == organ and state.engine.hovernet is not None:
        _model_loaded = True
        is_ready, _ = check_analyze_ready()
        return (
            f"✅ {organ}",
            "<span style='color: #28a745; font-size: 1.5em;'>●</span>",  # LED verte
            gr.update(interactive=is_ready, variant="primary" if is_ready else "secondary")
        )

    # Charger le modèle
    result = load_engine_core(organ)
    if result["success"]:
        _model_loaded = True
        is_ready, _ = check_analyze_ready()
        return (
            f"✅ {result['organ']} ({result['model_type']})",
            "<span style='color: #28a745; font-size: 1.5em;'>●</span>",  # LED verte
            gr.update(interactive=is_ready, variant="primary" if is_ready else "secondary")
        )
    else:
        _model_loaded = False
        return (
            f"❌ {result.get('error', 'Erreur')}",
            "<span style='color: #dc3545; font-size: 1.5em;'>●</span>",  # LED rouge
            gr.update(interactive=False, variant="secondary")
        )


# ==============================================================================
# RUBAN DIAGNOSTIC
# ==============================================================================


def create_diagnostic_ribbon(result, organ_name: str, organ_confidence: float, confidence_level: str) -> str:
    """
    Crée le ruban diagnostic HTML avec couleur selon la confiance.
    """
    # Couleur selon confiance
    if confidence_level == "Élevée":
        bg_color = "#28a745"  # Vert
        icon = "✓"
    elif confidence_level == "Modérée":
        bg_color = "#ffc107"  # Orange
        icon = "⚠"
    else:
        bg_color = "#dc3545"  # Rouge
        icon = "✕"

    # Statistiques
    n_nuclei = result.n_nuclei if result else 0
    n_mitosis = result.n_mitosis_candidates if result else 0

    html = f"""
    <div style="
        background: linear-gradient(90deg, {bg_color} 0%, {bg_color}dd 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 10px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">
        <span style="font-size: 1.1em;">
            {icon} {organ_name} ({int(organ_confidence * 100)}%)
        </span>
        <span>
            Confiance: <b>{confidence_level}</b>
        </span>
        <span>
            🧬 {n_nuclei} noyaux
        </span>
        <span>
            🔴 {n_mitosis} mitoses suspectes
        </span>
    </div>
    """
    return html


# ==============================================================================
# ANALYSE
# ==============================================================================


def analyze_image(
    image: np.ndarray,
    profile: str,
    np_threshold: float,
    min_size: int,
    beta: float,
    min_distance: int,
    use_auto_params: bool,
):
    """
    Analyse l'image et retourne les outputs adaptés au profil.
    """
    empty = np.zeros((224, 224, 3), dtype=np.uint8)
    empty_debug = np.zeros((100, 400, 3), dtype=np.uint8)
    empty_phase3_debug = np.zeros((80, 400, 3), dtype=np.uint8)
    empty_ribbon = ""

    is_clinical = PROFILES.get(profile, {}).get("clinical_language", False)

    # Appel core
    output = analyze_image_core(image, np_threshold, min_size, beta, min_distance, use_auto_params)

    organ, family, is_dedicated = get_organ_info()

    if not output.success:
        error_msg = output.error or "Erreur inconnue"
        return (
            output.overlay,
            error_msg,
            "",
            empty,
            output.debug,
            output.anomaly_overlay,
            output.phase3_overlay,
            output.phase3_debug,
            "",  # confidence badge
            empty_ribbon,  # ribbon
        )

    result = output.result

    # Overlay adapté au profil
    if is_clinical:
        overlay = create_segmentation_overlay(
            result.image_rgb, result.instance_map, result.type_map, alpha=0.4
        )
        overlay = create_contour_overlay(
            overlay, result.instance_map, result.type_map, thickness=1
        )
        if result.hotspot_ids:
            overlay = create_hotspot_overlay(overlay, result.instance_map, result.hotspot_ids)
        if result.mitosis_candidate_ids and result.spatial_analysis:
            overlay = create_mitosis_overlay(
                overlay, result.instance_map,
                result.mitosis_candidate_ids,
                result.spatial_analysis.mitosis_scores
            )
    else:
        overlay = output.overlay

    # Chart distribution
    if result.morphometry:
        chart = create_type_distribution_chart(result.morphometry.type_counts)
    else:
        chart = empty

    # Formatage selon profil
    if is_clinical:
        identification = format_identification_clinical(result, organ, family, is_dedicated)
        metrics = format_metrics_clinical(result, organ, family, is_dedicated)
        metrics_text = f"{identification}\n\n---\n\n{metrics}"
        alerts_text = format_alerts_clinical(result)
        confidence = format_confidence_badge(result)

        # Déterminer le niveau de confiance pour le ruban
        confidence_level = "Modérée"  # Défaut
        if "Élevée" in confidence:
            confidence_level = "Élevée"
        elif "Faible" in confidence:
            confidence_level = "Faible"

        ribbon = create_diagnostic_ribbon(
            result,
            result.organ_name,
            result.organ_confidence,
            confidence_level
        )
    else:
        metrics_text = format_metrics_rnd(result, organ, family, is_dedicated)
        alerts_text = format_alerts_rnd(result)
        confidence = ""
        ribbon = ""

    return (
        overlay,
        metrics_text,
        alerts_text,
        chart,
        output.debug,
        output.anomaly_overlay,
        output.phase3_overlay,
        output.phase3_debug,
        confidence,
        ribbon,
    )


# ==============================================================================
# GESTION DU CLIC SUR NOYAU
# ==============================================================================


def create_zoom_crop(x: int, y: int, zoom: int = 3) -> Optional[np.ndarray]:
    """
    Crée un crop zoomé ×3 centré sur le point cliqué.
    Affiche l'image originale ET la segmentation côte à côte.
    """
    if state.current_result is None or state.current_result.image_rgb is None:
        return None

    import cv2

    result = state.current_result
    original = result.image_rgb
    h, w = original.shape[:2]

    # Créer l'overlay de segmentation
    overlay = create_segmentation_overlay(
        original, result.instance_map, result.type_map, alpha=0.4
    )
    overlay = create_contour_overlay(
        overlay, result.instance_map, result.type_map, thickness=1
    )

    # Taille du crop source
    crop_size = 50  # 50px × 3 = 150px affiché

    half = crop_size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)

    crop_original = original[y1:y2, x1:x2].copy()
    crop_overlay = overlay[y1:y2, x1:x2].copy()

    if crop_original.size == 0:
        return None

    zoomed_original = cv2.resize(crop_original, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    zoomed_overlay = cv2.resize(crop_overlay, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

    zh, zw = zoomed_original.shape[:2]
    cx, cy = zw // 2, zh // 2

    # Crosshair
    zoomed_original[cy-1:cy+2, :] = [255, 0, 0]
    zoomed_original[:, cx-1:cx+2] = [255, 0, 0]
    zoomed_original[cy-1:cy+2, cx-1:cx+2] = [255, 255, 255]

    zoomed_overlay[cy-1:cy+2, :] = [255, 0, 0]
    zoomed_overlay[:, cx-1:cx+2] = [255, 0, 0]
    zoomed_overlay[cy-1:cy+2, cx-1:cx+2] = [255, 255, 255]

    # Séparateur
    separator_width = 4
    separator = np.ones((zh, separator_width, 3), dtype=np.uint8) * 180

    combined = np.concatenate([zoomed_original, separator, zoomed_overlay], axis=1)
    return combined


def on_image_click(evt: gr.SelectData, profile: str) -> Tuple[str, Optional[np.ndarray]]:
    """Gère le clic sur l'image pour afficher les infos du noyau."""
    if state.current_result is None:
        return "⚠️ Aucune analyse active", None

    is_clinical = PROFILES.get(profile, {}).get("clinical_language", False)

    try:
        x, y = evt.index
        zoom_image = create_zoom_crop(int(x), int(y))
        nucleus = state.current_result.get_nucleus_at(y, x)

        if nucleus is None:
            if state.current_result.instance_map is not None:
                y_int, x_int = int(y), int(x)
                if 0 <= y_int < state.current_result.instance_map.shape[0] and \
                   0 <= x_int < state.current_result.instance_map.shape[1]:
                    raw_id = state.current_result.instance_map[y_int, x_int]
                    if raw_id > 0:
                        if is_clinical:
                            return f"⚠️ Noyau ID:{raw_id} — exclu (filtrage qualité)", zoom_image
                        else:
                            return f"⚠️ Noyau ID:{raw_id} exclu (filtrage Phase 2/3)", zoom_image
            return "Clic sur le fond", zoom_image

        is_small = nucleus.circularity == 0 and nucleus.perimeter_um == 0

        if is_clinical:
            lines = [
                f"### Noyau #{nucleus.id}",
                f"**Type:** {nucleus.cell_type}",
                f"**Aire:** {nucleus.area_um2:.1f} µm²",
            ]
            if nucleus.circularity > 0:
                shape = "Régulière" if nucleus.circularity > 0.7 else "Irrégulière"
                lines.append(f"**Forme:** {shape}")
            if nucleus.is_mitosis_candidate:
                lines.append("🔴 **Mitose suspecte**")
            if nucleus.is_in_hotspot:
                lines.append("🟠 **Zone hypercellulaire**")
        else:
            lines = [
                f"### Noyau #{nucleus.id}",
                f"**Type:** {nucleus.cell_type}",
                f"**Position:** ({nucleus.centroid[1]}, {nucleus.centroid[0]})",
                f"**Aire:** {nucleus.area_um2:.1f} µm²",
            ]
            if is_small:
                lines.append(f"**Confiance:** {nucleus.confidence:.1%} *(réduite)*")
            else:
                lines.append(f"**Périmètre:** {nucleus.perimeter_um:.1f} µm")
                lines.append(f"**Circularité:** {nucleus.circularity:.2f}")
                lines.append(f"**Confiance:** {nucleus.confidence:.1%}")

            if nucleus.is_potential_fusion:
                lines.append("⚠️ **FUSION POTENTIELLE**")
            if nucleus.is_potential_over_seg:
                lines.append("⚠️ **SUR-SEGMENTATION**")

        return "\n".join(lines), zoom_image

    except Exception as e:
        return f"Erreur: {e}", None


# ==============================================================================
# CHANGEMENT DE PROFIL
# ==============================================================================


def on_profile_change(profile: str):
    """Change la visibilité des éléments selon le profil."""
    is_rnd = profile == "🔬 R&D Cockpit"

    return (
        gr.update(visible=is_rnd),  # params_accordion
        gr.update(visible=is_rnd),  # debug_accordion_phase2
        gr.update(visible=is_rnd),  # debug_accordion_phase3
        gr.update(visible=is_rnd),  # exports_rnd
        gr.update(visible=not is_rnd),  # ribbon (pathologiste only)
        gr.update(visible=not is_rnd),  # confidence badge
    )


# ==============================================================================
# EXPORTS
# ==============================================================================


def export_pdf_handler() -> Optional[str]:
    return export_pdf_core()


def export_json_handler() -> str:
    return export_json_core()


def export_nuclei_csv_handler() -> Optional[str]:
    return export_nuclei_csv_core()


def export_summary_csv_handler() -> Optional[str]:
    return export_summary_csv_core()


# ==============================================================================
# INTERFACE GRADIO
# ==============================================================================


def create_ui():
    """Crée l'interface Gradio unifiée - Design Clinical-Grade."""

    custom_css = """
    /* ============================================
       LAYOUT PLEINE LARGEUR
       ============================================ */
    .gradio-container {
        max-width: 100% !important;
        padding: 0 30px !important;
    }

    /* Conteneur principal */
    .main {
        max-width: 100% !important;
    }

    /* ============================================
       TYPOGRAPHIE GRANDE (ÉCRAN LARGE)
       ============================================ */
    body, .gradio-container {
        font-size: 18px !important;
    }

    /* Titres */
    h1 { font-size: 2.2em !important; }
    h2 { font-size: 1.8em !important; }
    h3 { font-size: 1.5em !important; }

    /* Labels et textes */
    label, .label-wrap, span {
        font-size: 1.1em !important;
    }

    /* Markdown */
    .markdown-text, .prose {
        font-size: 1.1em !important;
        line-height: 1.6 !important;
    }

    .prose p, .prose li {
        font-size: 1.1em !important;
    }

    /* Boutons */
    button {
        font-size: 1.2em !important;
        padding: 14px 28px !important;
    }

    /* Dropdowns et inputs */
    select, input, textarea {
        font-size: 1.1em !important;
        padding: 10px !important;
    }

    /* ============================================
       HEADER SOMBRE
       ============================================ */
    .header-dark {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px 35px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    .header-dark h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 2em !important;
    }
    .header-dark .version {
        color: #888;
        font-size: 1em;
    }

    /* ============================================
       BARRE D'ACTIONS
       ============================================ */
    .action-bar {
        background: #f8f9fa;
        padding: 18px 24px;
        border-radius: 8px;
        margin: 18px 0;
        border: 1px solid #dee2e6;
    }

    .action-bar button {
        min-width: 200px;
        font-weight: 600 !important;
    }

    /* ============================================
       DEEP DIVE - 3 COLONNES
       ============================================ */
    .deep-dive {
        margin-top: 25px;
        gap: 25px;
    }

    .deep-dive h3 {
        border-bottom: 2px solid #007bff;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }

    /* ============================================
       DISCLAIMER
       ============================================ */
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 14px 24px;
        border-radius: 5px;
        margin-bottom: 18px;
        text-align: center;
        font-size: 1.1em;
    }

    /* ============================================
       IMAGES
       ============================================ */
    .image-container img {
        max-height: none !important;
    }

    /* ============================================
       CACHER LE MENU SUR LA LOUPE
       ============================================ */
    #loupe-container .image-container > div:first-child,
    #loupe-container .icon-buttons,
    #loupe-container button[aria-label],
    #loupe-container .download-button,
    #loupe-container .share-button,
    #loupe-container .fullscreen-button,
    .loupe-image .image-container > div:first-child {
        display: none !important;
    }

    /* Cacher tous les boutons d'action sur les images de la loupe */
    #loupe-container .svelte-1pijsyv,
    #loupe-container [class*="icon"],
    #loupe-container .absolute {
        display: none !important;
    }
    """

    with gr.Blocks(
        title="CellViT-Optimus",
        css=custom_css,
        theme=gr.themes.Soft(),
    ) as app:

        # ==================================================================
        # 1. HEADER SOMBRE
        # ==================================================================
        with gr.Row(elem_classes="header-dark"):
            with gr.Column(scale=3):
                gr.HTML("<h1>CellViT-Optimus <span class='version'>V13 Hybrid</span></h1>")

            with gr.Column(scale=2):
                with gr.Row():
                    organ_select = gr.Dropdown(
                        choices=ORGAN_CHOICES,
                        value=None,
                        label="Organe",
                        interactive=True,
                        scale=2,
                    )
                    status_led = gr.HTML(
                        "<span style='color: #ffc107; font-size: 1.5em;'>●</span>",
                        label="",
                    )

            with gr.Column(scale=2):
                status_text = gr.Textbox(
                    value="⏳ Sélectionnez un organe",
                    interactive=False,
                    show_label=False,
                )

            with gr.Column(scale=1):
                profile_select = gr.Dropdown(
                    choices=list(PROFILES.keys()),
                    value="🩺 Pathologiste",
                    label="Profil",
                    interactive=True,
                )

        # Disclaimer
        gr.HTML("""
        <div class="disclaimer">
            <b>Document d'aide à la décision — Validation médicale requise</b>
        </div>
        """)

        # ==================================================================
        # 2. VISUAL STAGE (Images en majesté)
        # ==================================================================
        with gr.Row():
            input_image = gr.Image(
                label="Image H&E (224×224)",
                type="numpy",
                height=400,
            )
            output_image = gr.Image(
                label="Segmentation",
                type="numpy",
                height=400,
                interactive=True,
            )

        # ==================================================================
        # 3. BARRE D'ACTIONS UNIFIÉE
        # ==================================================================
        with gr.Row(elem_classes="action-bar"):
            gr.Column(scale=1)  # Spacer gauche
            analyze_btn = gr.Button(
                "🚀 ANALYSER",
                variant="secondary",
                size="lg",
                interactive=False,
                scale=2,
            )
            export_pdf_btn = gr.Button(
                "📄 RAPPORT PDF",
                variant="secondary",
                size="lg",
                scale=1,
            )
            settings_btn = gr.Button("⚙️", scale=0, size="sm", visible=False)
            gr.Column(scale=1)  # Spacer droit

        # Fichier téléchargeable (toujours visible)
        export_file_download = gr.File(label="📥 Télécharger le rapport", visible=True)

        # Paramètres Watershed (R&D seulement, caché par défaut)
        params_accordion = gr.Accordion("⚙️ Paramètres Watershed", open=False, visible=False)
        with params_accordion:
            with gr.Row():
                use_auto_params = gr.Checkbox(value=True, label="Auto (organ_config)")
                np_threshold = gr.Slider(0.2, 0.8, 0.40, step=0.05, label="Seuil NP")
                min_size = gr.Slider(10, 100, 30, step=5, label="Taille min")
                beta = gr.Slider(0.1, 2.0, 0.50, step=0.1, label="Beta")
                min_distance = gr.Slider(1, 10, 5, step=1, label="Dist min")

        # ==================================================================
        # 4. RUBAN DIAGNOSTIC (Pathologiste seulement)
        # ==================================================================
        diagnostic_ribbon = gr.HTML("", visible=True, elem_id="diagnostic-ribbon")

        # ==================================================================
        # 5. DEEP DIVE - 3 COLONNES ÉGALES
        # ==================================================================
        with gr.Row(elem_classes="deep-dive"):
            # Colonne 1: Alertes
            with gr.Column(scale=1):
                gr.Markdown("### 🚨 Alertes")
                alerts_md = gr.Markdown("*Aucune alerte*")

                # Confidence badge (pathologiste)
                confidence_html = gr.HTML("", visible=True)

            # Colonne 2: Métriques
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Métriques")
                metrics_md = gr.Markdown("*En attente d'analyse*")
                chart_image = gr.Image(label="Distribution", height=180, show_label=False)

            # Colonne 3: Loupe Interactive
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Loupe ×3")
                zoom_display = gr.Image(height=180, show_label=False)
                nucleus_info = gr.Markdown("*Cliquer sur un noyau*")

        # ==================================================================
        # 6. DEBUG PANELS (R&D seulement)
        # ==================================================================
        debug_accordion_phase2 = gr.Accordion("🔬 Debug IA (Phase 2)", open=False, visible=False)
        with debug_accordion_phase2:
            with gr.Row():
                debug_panel = gr.Image(label="Pipeline NP/HV/Instances", height=180)
                anomaly_image = gr.Image(label="Anomalies", height=180)

        debug_accordion_phase3 = gr.Accordion("🧬 Intelligence Spatiale (Phase 3)", open=False, visible=False)
        with debug_accordion_phase3:
            with gr.Row():
                phase3_debug = gr.Image(label="Pléomorphisme/Clustering", height=120)
                phase3_overlay = gr.Image(label="Phase 3 Overlay", height=180)

        # ==================================================================
        # 7. EXPORTS R&D (tous formats)
        # ==================================================================
        exports_rnd_accordion = gr.Accordion("📤 Exports avancés", open=False, visible=False)
        with exports_rnd_accordion:
            with gr.Row():
                export_json_btn = gr.Button("📋 JSON", size="sm")
                export_nuclei_btn = gr.Button("📊 CSV Noyaux", size="sm")
                export_summary_btn = gr.Button("📊 CSV Résumé", size="sm")
            export_file_rnd = gr.File(label="Fichier généré")

        # ==================================================================
        # CONNEXIONS
        # ==================================================================

        # Image chargée → met à jour état du bouton
        input_image.change(
            fn=on_image_change,
            inputs=[input_image],
            outputs=[analyze_btn],
        )

        # Changement d'organe → charge modèle
        organ_select.change(
            fn=load_engine,
            inputs=[organ_select],
            outputs=[status_text, status_led, analyze_btn],
        )

        # Changement de profil → visibilité
        profile_select.change(
            fn=on_profile_change,
            inputs=[profile_select],
            outputs=[
                params_accordion,
                debug_accordion_phase2,
                debug_accordion_phase3,
                exports_rnd_accordion,
                diagnostic_ribbon,
                confidence_html,
            ],
        )

        # Analyse
        analyze_btn.click(
            fn=analyze_image,
            inputs=[
                input_image,
                profile_select,
                np_threshold, min_size, beta, min_distance,
                use_auto_params,
            ],
            outputs=[
                output_image,
                metrics_md,
                alerts_md,
                chart_image,
                debug_panel,
                anomaly_image,
                phase3_overlay,
                phase3_debug,
                confidence_html,
                diagnostic_ribbon,
            ],
        )

        # Clic sur image → infos noyau
        output_image.select(
            fn=on_image_click,
            inputs=[profile_select],
            outputs=[nucleus_info, zoom_display],
        )

        # Exports
        export_pdf_btn.click(fn=export_pdf_handler, outputs=[export_file_download])
        export_json_btn.click(fn=export_json_handler, outputs=[export_file_download])
        export_nuclei_btn.click(fn=export_nuclei_csv_handler, outputs=[export_file_download])
        export_summary_btn.click(fn=export_summary_csv_handler, outputs=[export_file_download])

    return app


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="CellViT-Optimus Interface Unifiée")
    parser.add_argument("--organ", default=None, help="Organe à précharger (optionnel)")
    parser.add_argument("--port", type=int, default=7860, help="Port Gradio")
    parser.add_argument("--no-preload", action="store_true", help="Ne pas précharger le backbone")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Préchargement backbone
    if not args.no_preload:
        logger.info("Préchargement du backbone H-optimus-0 + OrganHead...")
        result = preload_backbone_core()
        if result["success"]:
            logger.info("✅ Backbone préchargé avec succès")
        else:
            logger.warning(f"⚠️ Erreur préchargement: {result['error']}")

    # Précharger organe si spécifié
    if args.organ:
        logger.info(f"Chargement du modèle HoVer-Net pour {args.organ}...")
        load_engine(args.organ)

    app = create_ui()
    app.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
