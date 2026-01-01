"""
CellViT-Optimus — Interface Unifiée Professionnelle.

Interface unique avec sélecteur de profil:
- Profil "R&D": Vue technique détaillée (debug, paramètres, métriques complètes)
- Profil "Pathologiste": Vue clinique simplifiée (langage médical, alertes)

Le moteur est partagé entre les profils → cohérence garantie.

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
# FONCTIONS UTILITAIRES
# ==============================================================================


def get_organ_info():
    """Retourne les infos de l'organe actuel."""
    if state.engine is None:
        return None, None, False
    return state.engine.organ, state.engine.family, state.engine.is_dedicated_model


def load_engine(organ: str) -> str:
    """
    Charge ou change l'organe du moteur.

    - Premier appel: charge H-optimus-0 + OrganHead + HoVer-Net
    - Appels suivants: ne recharge que HoVer-Net (changement rapide)
    """
    # Si le moteur n'existe pas encore, chargement complet
    if state.engine is None:
        result = load_engine_core(organ)
        if result["success"]:
            return f"✅ {result['organ']} ({result['model_type']}) chargé"
        else:
            return f"❌ Erreur: {result.get('error', 'Inconnue')}"

    # Si même organe, rien à faire
    if state.engine.organ == organ:
        return f"✅ {organ} déjà chargé"

    # Changement d'organe: ne recharge que HoVer-Net
    result = change_organ_core(organ)
    if result["success"]:
        return f"✅ {result['organ']} ({result['model_type']}) — modèle changé"
    else:
        return f"❌ Erreur: {result.get('error', 'Inconnue')}"


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
        )

    result = output.result

    # Overlay adapté au profil
    if is_clinical:
        # Vue clinique: overlay simplifié avec hotspots/mitoses
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
        # Vue R&D: overlay complet
        overlay = output.overlay

    # Chart distribution
    if result.morphometry:
        chart = create_type_distribution_chart(result.morphometry.type_counts)
    else:
        chart = empty

    # Formatage selon profil
    if is_clinical:
        # Identification + Métriques cliniques combinés
        identification = format_identification_clinical(result, organ, family, is_dedicated)
        metrics = format_metrics_clinical(result, organ, family, is_dedicated)
        metrics_text = f"{identification}\n\n---\n\n{metrics}"
        alerts_text = format_alerts_clinical(result)
        confidence = format_confidence_badge(result)
    else:
        metrics_text = format_metrics_rnd(result, organ, family, is_dedicated)
        alerts_text = format_alerts_rnd(result)
        confidence = ""

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
    )


# ==============================================================================
# GESTION DU CLIC SUR NOYAU
# ==============================================================================


def on_image_click(evt: gr.SelectData, profile: str) -> str:
    """Gère le clic sur l'image pour afficher les infos du noyau."""
    if state.current_result is None:
        return "⚠️ Aucune analyse active"

    is_clinical = PROFILES.get(profile, {}).get("clinical_language", False)

    try:
        x, y = evt.index
        nucleus = state.current_result.get_nucleus_at(y, x)

        if nucleus is None:
            # Vérification de sécurité
            if state.current_result.instance_map is not None:
                y_int, x_int = int(y), int(x)
                if 0 <= y_int < state.current_result.instance_map.shape[0] and \
                   0 <= x_int < state.current_result.instance_map.shape[1]:
                    raw_id = state.current_result.instance_map[y_int, x_int]
                    if raw_id > 0:
                        if is_clinical:
                            return f"⚠️ Noyau ID:{raw_id} — exclu du rapport (filtrage qualité)"
                        else:
                            return f"⚠️ Noyau identifié (ID:{raw_id}) mais exclu des métriques (filtrage Phase 2/3)"
            return "Clic sur le fond (pas de noyau)"

        # Format selon profil
        is_small = nucleus.circularity == 0 and nucleus.perimeter_um == 0

        if is_clinical:
            # Format clinique simplifié
            lines = [
                f"### Noyau #{nucleus.id}",
                "",
                f"**Type:** {nucleus.cell_type}",
                f"**Aire:** {nucleus.area_um2:.1f} µm²",
            ]

            if nucleus.circularity > 0:
                shape = "Régulière" if nucleus.circularity > 0.7 else "Irrégulière"
                lines.append(f"**Forme:** {shape}")
            else:
                lines.append("**Forme:** *Non évaluée (petit noyau)*")

            if nucleus.is_mitosis_candidate:
                lines.append("")
                lines.append("🔴 **Mitose suspecte**")

            if nucleus.is_in_hotspot:
                lines.append("🟠 **Zone hypercellulaire**")

        else:
            # Format R&D détaillé
            lines = [
                f"### Noyau #{nucleus.id}",
                "",
                f"**Type:** {nucleus.cell_type}",
                f"**Position:** ({nucleus.centroid[1]}, {nucleus.centroid[0]})",
                f"**Aire:** {nucleus.area_um2:.1f} µm²",
            ]

            if is_small:
                lines.append("**Périmètre:** *N/A (petit noyau)*")
                lines.append("**Circularité:** *N/A*")
                lines.append(f"**Confiance:** {nucleus.confidence:.1%} *(réduite)*")
            else:
                lines.append(f"**Périmètre:** {nucleus.perimeter_um:.1f} µm")
                lines.append(f"**Circularité:** {nucleus.circularity:.2f}")
                lines.append(f"**Confiance:** {nucleus.confidence:.1%}")

            if nucleus.is_uncertain:
                lines.append("**Status:** Incertain")
            if nucleus.is_mitotic:
                lines.append("**Status:** Mitose suspecte")

            if nucleus.is_potential_fusion:
                lines.append("")
                lines.append("⚠️ **FUSION POTENTIELLE**")
                lines.append(f"   {nucleus.anomaly_reason}")
            if nucleus.is_potential_over_seg:
                lines.append("")
                lines.append("⚠️ **SUR-SEGMENTATION**")
                lines.append(f"   {nucleus.anomaly_reason}")

        return "\n".join(lines)

    except Exception as e:
        return f"Erreur: {e}"


# ==============================================================================
# CHANGEMENT DE PROFIL
# ==============================================================================


def on_profile_change(profile: str):
    """
    Change la visibilité des éléments selon le profil.

    Returns:
        Tuple of gr.update() pour les composants conditionnels
    """
    is_rnd = profile == "🔬 R&D Cockpit"

    return (
        # Paramètres watershed (visible seulement en R&D)
        gr.update(visible=is_rnd),
        # Debug panel Phase 2
        gr.update(visible=is_rnd),
        # Debug panel Phase 3
        gr.update(visible=is_rnd),
        # Overlays checkboxes R&D
        gr.update(visible=is_rnd),
        # Exports R&D (tous les exports)
        gr.update(visible=is_rnd),
        # Exports Pathologiste (PDF seulement)
        gr.update(visible=not is_rnd),
        # Confidence badge (pathologiste seulement)
        gr.update(visible=not is_rnd),
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
    """Crée l'interface Gradio unifiée avec sélecteur de profil."""

    # CSS
    custom_css = """
    .profile-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .profile-selector {
        font-size: 1.1em;
    }
    """

    # JavaScript loupe
    loupe_script = """
    <script>
    (function() {
        let lens = null;
        const zoomFactor = 2.5;
        const lensSize = 150;

        function init() {
            if (!lens) {
                lens = document.createElement('div');
                lens.style.cssText = "position:fixed; border:2px solid #007bff; border-radius:50%; width:150px; height:150px; pointer-events:none; display:none; z-index:10000; box-shadow:0 0 10px rgba(0,0,0,0.5); background-repeat:no-repeat; background-color:black;";
                document.body.appendChild(lens);
            }

            document.addEventListener('mousemove', function(e) {
                const container = e.target.closest('.gradio-image, .image-container, [class*="image"]');
                const img = container ? container.querySelector('img') : null;

                if (img && img.src && img.naturalWidth > 0 && !img.src.includes('data:image/svg')) {
                    const rect = img.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;

                    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
                        lens.style.display = 'none';
                        return;
                    }

                    const ratioX = img.naturalWidth / rect.width;
                    const ratioY = img.naturalHeight / rect.height;

                    lens.style.display = 'block';
                    lens.style.left = (e.clientX - lensSize / 2) + 'px';
                    lens.style.top = (e.clientY - lensSize / 2) + 'px';
                    lens.style.backgroundImage = 'url(' + img.src + ')';
                    lens.style.backgroundSize = (img.naturalWidth * zoomFactor) + 'px ' + (img.naturalHeight * zoomFactor) + 'px';
                    lens.style.backgroundPosition = '-' + (x * ratioX * zoomFactor - lensSize / 2) + 'px -' + (y * ratioY * zoomFactor - lensSize / 2) + 'px';
                } else {
                    lens.style.display = 'none';
                }
            });
        }
        setTimeout(init, 1000);
    })();
    </script>
    """

    with gr.Blocks(
        title="CellViT-Optimus",
        css=custom_css,
        theme=gr.themes.Soft(),
    ) as app:

        gr.HTML(loupe_script)

        # ======================================================================
        # HEADER AVEC SÉLECTEUR DE PROFIL
        # ======================================================================
        with gr.Row(elem_classes="profile-header"):
            gr.Markdown("# CellViT-Optimus")

            profile_select = gr.Dropdown(
                choices=list(PROFILES.keys()),
                value="🩺 Pathologiste",
                label="Profil",
                interactive=True,
                scale=1,
                elem_classes="profile-selector",
            )

        # Disclaimer
        gr.HTML("""
        <div style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 5px; margin-bottom: 15px; text-align: center;">
            <b>Document d'aide à la décision — Validation médicale requise</b>
        </div>
        """)

        # ======================================================================
        # CONTRÔLES PRINCIPAUX
        # ======================================================================
        with gr.Row():
            with gr.Column(scale=2):
                # Sélection organe
                with gr.Row():
                    organ_select = gr.Dropdown(
                        choices=ORGAN_CHOICES,
                        value="Lung",
                        label="Organe (★ = modèle dédié)",
                        interactive=True,
                    )
                    load_btn = gr.Button("Charger", variant="primary")
                    status_text = gr.Textbox(label="Status", interactive=False, scale=2)

                # Images
                with gr.Row():
                    input_image = gr.Image(
                        label="Image H&E (224×224)",
                        type="numpy",
                        height=300,
                    )
                    output_image = gr.Image(
                        label="Segmentation",
                        type="numpy",
                        height=300,
                        interactive=True,
                    )

                # Overlays R&D (conditionnels)
                overlays_row = gr.Row(visible=False)
                with overlays_row:
                    show_seg = gr.Checkbox(label="Segmentation", value=True)
                    show_contours = gr.Checkbox(label="Contours", value=True)
                    show_uncertainty = gr.Checkbox(label="Incertitude", value=False)
                    show_anomalies = gr.Checkbox(label="Anomalies", value=False)
                    show_hotspots = gr.Checkbox(label="Hotspots", value=False)
                    show_mitoses = gr.Checkbox(label="Mitoses", value=False)

                # Paramètres Watershed (R&D seulement)
                params_accordion = gr.Accordion("Paramètres Watershed", open=False, visible=False)
                with params_accordion:
                    use_auto_params = gr.Checkbox(value=True, label="Params Auto (organ_config.py)")
                    np_threshold = gr.Slider(0.2, 0.8, 0.40, step=0.05, label="Seuil NP")
                    min_size = gr.Slider(10, 100, 30, step=5, label="Taille min")
                    beta = gr.Slider(0.1, 2.0, 0.50, step=0.1, label="Beta (HV)")
                    min_distance = gr.Slider(1, 10, 5, step=1, label="Distance min")

                analyze_btn = gr.Button("🔬 Analyser", variant="primary", size="lg")

            # Colonne droite: Métriques
            with gr.Column(scale=1):
                metrics_md = gr.Markdown("### En attente\n*Charger une image et lancer l'analyse*")

                with gr.Accordion("Alertes", open=True):
                    alerts_md = gr.Markdown("*Aucune alerte*")

                with gr.Accordion("Noyau sélectionné", open=True):
                    nucleus_info = gr.Markdown("*Cliquer sur un noyau*")

                # Confidence badge (pathologiste)
                confidence_html = gr.HTML("", visible=True)

                chart_image = gr.Image(label="Distribution", height=180)

        # ======================================================================
        # DEBUG PANELS (R&D seulement)
        # ======================================================================
        debug_accordion_phase2 = gr.Accordion("Debug IA (Phase 2)", open=False, visible=False)
        with debug_accordion_phase2:
            with gr.Row():
                debug_panel = gr.Image(label="Pipeline NP/HV/Instances", height=180)
                anomaly_image = gr.Image(label="Anomalies", height=180)

        debug_accordion_phase3 = gr.Accordion("Intelligence Spatiale (Phase 3)", open=False, visible=False)
        with debug_accordion_phase3:
            with gr.Row():
                phase3_debug = gr.Image(label="Pléomorphisme/Clustering", height=120)
                phase3_overlay = gr.Image(label="Phase 3 Overlay", height=180)

        # ======================================================================
        # EXPORTS R&D (tous les exports)
        # ======================================================================
        exports_rnd_accordion = gr.Accordion("📤 Exports", open=False, visible=False)
        with exports_rnd_accordion:
            with gr.Row():
                export_pdf_rnd_btn = gr.Button("📄 Rapport PDF", variant="primary")
                export_json_btn = gr.Button("📋 JSON")
                export_nuclei_btn = gr.Button("📊 CSV Noyaux")
                export_summary_btn = gr.Button("📊 CSV Résumé")

            export_file_rnd = gr.File(label="Fichier généré")

        # ======================================================================
        # EXPORTS PATHOLOGISTE (PDF seulement)
        # ======================================================================
        exports_patho_accordion = gr.Accordion("📤 Export Rapport", open=False, visible=True)
        with exports_patho_accordion:
            export_pdf_patho_btn = gr.Button("📄 Générer Rapport PDF", variant="primary", size="lg")
            export_file_patho = gr.File(label="Rapport PDF")

        # ======================================================================
        # CONNEXIONS
        # ======================================================================

        # Changement de profil → met à jour la visibilité
        profile_select.change(
            fn=on_profile_change,
            inputs=[profile_select],
            outputs=[
                params_accordion,
                debug_accordion_phase2,
                debug_accordion_phase3,
                overlays_row,
                exports_rnd_accordion,
                exports_patho_accordion,
                confidence_html,
            ],
        )

        # Chargement moteur (bouton ou changement dropdown)
        load_btn.click(
            fn=load_engine,
            inputs=[organ_select],
            outputs=[status_text],
        )

        # Auto-chargement quand on change l'organe
        organ_select.change(
            fn=load_engine,
            inputs=[organ_select],
            outputs=[status_text],
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
            ],
        )

        # Clic sur image
        output_image.select(
            fn=on_image_click,
            inputs=[profile_select],
            outputs=[nucleus_info],
        )

        # Exports R&D
        export_pdf_rnd_btn.click(fn=export_pdf_handler, outputs=[export_file_rnd])
        export_json_btn.click(fn=export_json_handler, outputs=[export_file_rnd])
        export_nuclei_btn.click(fn=export_nuclei_csv_handler, outputs=[export_file_rnd])
        export_summary_btn.click(fn=export_summary_csv_handler, outputs=[export_file_rnd])

        # Export Pathologiste (PDF seulement)
        export_pdf_patho_btn.click(fn=export_pdf_handler, outputs=[export_file_patho])

    return app


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="CellViT-Optimus Interface Unifiée")
    parser.add_argument("--organ", default="Lung", help="Organe initial")
    parser.add_argument("--port", type=int, default=7860, help="Port Gradio")
    parser.add_argument("--preload", action="store_true", help="Précharger le modèle")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.preload:
        logger.info(f"Préchargement du moteur pour {args.organ}...")
        load_engine(args.organ)

    app = create_ui()
    app.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
