#!/usr/bin/env python3
"""
CellViT-Optimus R&D Cockpit — Interface Gradio.

Application web interactive pour l'exploration et la validation
du moteur IA de segmentation cellulaire.

Note: Ceci est un outil R&D, pas une IHM clinique.

Architecture: Utilise src.ui.core pour la logique métier partagée
et src.ui.formatters pour l'affichage R&D (technique).

Usage:
    python -m src.ui.app
    # ou
    python src/ui/app.py

    Puis ouvrir http://localhost:7860
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Optional, Tuple
import sys

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
    load_engine_core,
    analyze_image_core,
    change_organ_core,
    on_image_click_core,
    export_pdf_core,
    export_nuclei_csv_core,
    export_summary_csv_core,
    export_json_core,
)

# Imports: Formatage R&D (technique)
from src.ui.formatters import (
    format_metrics_rnd,
    format_alerts_rnd,
    format_nucleus_info_rnd,
    format_load_status_rnd,
    format_organ_change_rnd,
)

# Imports: Moteur et configuration
from src.ui.inference_engine import ORGAN_CHOICES
from src.ui.organ_config import get_model_for_organ

# Imports: Visualisations
from src.ui.visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    create_uncertainty_overlay,
    create_density_heatmap,
    create_type_distribution_chart,
    create_anomaly_overlay,
    create_voronoi_overlay,
    create_hotspot_overlay,
    create_mitosis_overlay,
    create_chromatin_overlay,
)


# ==============================================================================
# WRAPPERS UI (utilisent core + formatters)
# ==============================================================================

def load_engine(organ: str, device: str = "cuda") -> str:
    """Charge le moteur d'inférence (wrapper UI)."""
    result = load_engine_core(organ, device)
    return format_load_status_rnd(result)


def analyze_image(
    image: np.ndarray,
    np_threshold: float,
    min_size: int,
    beta: float,
    min_distance: int,
    use_auto_params: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyse une image et retourne les visualisations (wrapper UI).

    Args:
        use_auto_params: Si True, utilise les params optimisés de organ_config.py
                         Si False, utilise les valeurs des sliders

    Returns:
        (overlay, contours, metrics_text, alerts_text, chart, debug, anomaly_overlay, phase3_overlay, phase3_debug)
    """
    empty = np.zeros((224, 224, 3), dtype=np.uint8)
    empty_debug = np.zeros((100, 400, 3), dtype=np.uint8)
    empty_phase3_debug = np.zeros((80, 400, 3), dtype=np.uint8)

    # Appel core
    output = analyze_image_core(image, np_threshold, min_size, beta, min_distance, use_auto_params)

    if not output.success:
        error_msg = output.error or "Erreur inconnue"
        return (
            output.overlay, output.contours, error_msg, "",
            output.chart, output.debug, output.anomaly_overlay,
            output.phase3_overlay, output.phase3_debug
        )

    # Formatage R&D
    organ = state.engine.organ if state.engine else None
    family = state.engine.family if state.engine else None
    is_dedicated = state.engine.is_dedicated_model if state.engine else False

    metrics_text = format_metrics_rnd(output.result, organ, family, is_dedicated)
    alerts_text = format_alerts_rnd(output.result)

    return (
        output.overlay, output.contours, metrics_text, alerts_text,
        output.chart, output.debug, output.anomaly_overlay,
        output.phase3_overlay, output.phase3_debug
    )


def export_json() -> str:
    """Exporte les résultats en JSON (wrapper UI)."""
    return export_json_core()


def export_pdf_handler() -> Optional[str]:
    """Génère et retourne le chemin du rapport PDF (wrapper UI)."""
    return export_pdf_core()


def export_nuclei_csv_handler() -> Optional[str]:
    """Génère et retourne le chemin du CSV des noyaux (wrapper UI)."""
    return export_nuclei_csv_core()


def export_summary_csv_handler() -> Optional[str]:
    """Génère et retourne le chemin du CSV résumé (wrapper UI)."""
    return export_summary_csv_core()


def on_image_click(evt: gr.SelectData) -> str:
    """Gère le clic sur l'image pour afficher les infos du noyau."""
    if state.current_result is None:
        return "⚠️ Aucune analyse active"

    try:
        x, y = evt.index
        nucleus = state.current_result.get_nucleus_at(y, x)

        if nucleus is None:
            # Vérification de sécurité: récupérer l'ID brut sur la map
            if state.current_result.instance_map is not None:
                y_int, x_int = int(y), int(x)
                if 0 <= y_int < state.current_result.instance_map.shape[0] and \
                   0 <= x_int < state.current_result.instance_map.shape[1]:
                    raw_id = state.current_result.instance_map[y_int, x_int]
                    if raw_id > 0:
                        return f"⚠️ Noyau identifié (ID:{raw_id}) mais exclu des métriques (filtrage Phase 2/3)"
            return "Clic sur le fond (pas de noyau)"

        # Détection petit noyau (pas de métriques complètes)
        is_small = nucleus.circularity == 0 and nucleus.perimeter_um == 0

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

        # Status spéciaux
        if nucleus.is_uncertain:
            lines.append("**Status:** Incertain")
        if nucleus.is_mitotic:
            lines.append("**Status:** Mitose suspecte")

        # Phase 2: Anomalies
        if nucleus.is_potential_fusion:
            lines.append("")
            lines.append("⚠️ **FUSION POTENTIELLE**")
            lines.append(f"   {nucleus.anomaly_reason}")
        if nucleus.is_potential_over_seg:
            lines.append("")
            lines.append("⚠️ **SUR-SEGMENTATION**")
            lines.append(f"   {nucleus.anomaly_reason}")

        # Info petit noyau
        if is_small:
            lines.append("")
            lines.append("⚠️ **PETIT NOYAU** (< 10 pixels)")
            lines.append("   Métriques morphologiques non calculées")

        # Phase 3: Intelligence spatiale (seulement si pas petit noyau)
        if not is_small:
            lines.append("")
            lines.append("---")
            lines.append("### Phase 3")
            lines.append(f"- Entropie chromatine: **{nucleus.chromatin_entropy:.2f}**")
            lines.append(f"- Voisins Voronoï: **{nucleus.n_neighbors}**")

            if nucleus.chromatin_heterogeneous:
                lines.append("- 🟣 **Chromatine hétérogène**")
            if nucleus.is_mitosis_candidate:
                lines.append(f"- 🔴 **Candidat mitose** (score: {nucleus.mitosis_score:.2f})")
            if nucleus.is_in_hotspot:
                lines.append("- 🟠 **Dans hotspot** (zone haute densité)")

        return "\n".join(lines)

    except Exception as e:
        return f"Erreur: {e}"


def update_overlay(
    show_segmentation: bool,
    show_contours: bool,
    show_uncertainty: bool,
    show_density: bool,
    show_anomalies: bool = False,  # Phase 2
    show_voronoi: bool = False,     # Phase 3
    show_hotspots: bool = False,    # Phase 3
    show_mitoses: bool = False,     # Phase 3
    show_chromatin: bool = False,   # Phase 3
) -> np.ndarray:
    """Met à jour l'overlay selon les options."""
    if state.current_result is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    result = state.current_result
    image = result.image_rgb.copy()

    if show_density:
        density = create_density_heatmap(result.instance_map)
        image = cv2.addWeighted(image, 0.6, density, 0.4, 0)

    if show_uncertainty and result.uncertainty_map is not None:
        image = create_uncertainty_overlay(image, result.uncertainty_map, threshold=0.5)

    if show_segmentation:
        image = create_segmentation_overlay(
            image, result.instance_map, result.type_map, alpha=0.3
        )

    if show_contours:
        image = create_contour_overlay(
            image, result.instance_map, result.type_map, thickness=1
        )

    # Phase 2: Overlay anomalies
    if show_anomalies and (result.fusion_ids or result.over_seg_ids):
        image = create_anomaly_overlay(
            image, result.instance_map, result.fusion_ids, result.over_seg_ids
        )

    # Phase 3: Voronoï
    if show_voronoi and result.nucleus_info:
        centroids = [n.centroid for n in result.nucleus_info]
        image = create_voronoi_overlay(image, centroids)

    # Phase 3: Hotspots
    if show_hotspots and result.hotspot_ids:
        image = create_hotspot_overlay(image, result.instance_map, result.hotspot_ids)

    # Phase 3: Mitoses
    if show_mitoses and result.mitosis_candidate_ids and result.spatial_analysis:
        image = create_mitosis_overlay(
            image, result.instance_map,
            result.mitosis_candidate_ids,
            result.spatial_analysis.mitosis_scores
        )

    # Phase 3: Chromatine hétérogène
    if show_chromatin and result.spatial_analysis:
        image = create_chromatin_overlay(
            image, result.instance_map,
            result.spatial_analysis.heterogeneous_nuclei_ids
        )

    return image


def change_organ(organ: str) -> str:
    """Change l'organe du modèle (wrapper UI)."""
    result = change_organ_core(organ)
    return format_organ_change_rnd(result)


# ==============================================================================
# INTERFACE GRADIO
# ==============================================================================

def create_ui():
    """Crée l'interface Gradio."""

    # CSS pour l'effet loupe
    custom_css = """
    /* Loupe (lentille de zoom) - Compatible Gradio 4.x */
    .loupe-lens {
        position: fixed;
        border: 3px solid #007bff;
        border-radius: 50%;
        width: 180px;
        height: 180px;
        pointer-events: none;
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.6);
        background-repeat: no-repeat;
        display: none;
        z-index: 9999;
        background-color: white;
    }

    .loupe-lens.active {
        display: block;
    }

    /* Style pour les images zoomables */
    .gradio-image img {
        cursor: crosshair;
    }
    """

    # JavaScript pour l'effet loupe (compatible Gradio 4.x avec overlay)
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
                // Utiliser closest() pour contourner les overlays Gradio
                const container = e.target.closest('.gradio-image, .image-container, [class*="image"]');
                const img = container ? container.querySelector('img') : null;

                if (img && img.src && img.naturalWidth > 0 && !img.src.includes('data:image/svg')) {
                    const rect = img.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;

                    // Vérifier qu'on est dans l'image
                    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
                        lens.style.display = 'none';
                        return;
                    }

                    // Ratio pour images redimensionnées
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

        // Délai pour laisser Gradio charger complètement
        setTimeout(init, 1000);
    })();
    </script>
    """

    with gr.Blocks(
        title="CellViT-Optimus R&D Cockpit",
        css=custom_css,
    ) as app:

        # Injection du script loupe
        gr.HTML(loupe_script)

        # Header
        gr.Markdown("# CellViT-Optimus — R&D Cockpit")
        gr.HTML("""
        <div style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <b>Document d'aide à la décision — Validation médicale requise</b><br>
            Ceci est un outil R&D pour l'exploration et la validation du moteur IA.
        </div>
        """)

        with gr.Row():
            # ================================================================
            # COLONNE GAUCHE: IMAGE & CONTRÔLES
            # ================================================================
            with gr.Column(scale=2):

                # Status du moteur
                with gr.Row():
                    organ_select = gr.Dropdown(
                        choices=ORGAN_CHOICES,
                        value="Lung",
                        label="Organe (★ = modèle dédié)",
                        interactive=True,
                    )
                    load_btn = gr.Button("Charger le moteur", variant="primary")
                    status_text = gr.Textbox(label="Status", interactive=False)

                # Image input/output
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

                # Overlays controls - Base
                with gr.Row():
                    show_seg = gr.Checkbox(label="Segmentation", value=True)
                    show_contours = gr.Checkbox(label="Contours", value=True)
                    show_uncertainty = gr.Checkbox(label="Incertitude", value=False)
                    show_density = gr.Checkbox(label="Densité", value=False)
                    show_anomalies = gr.Checkbox(label="Anomalies", value=False)  # Phase 2

                # Overlays controls - Phase 3
                with gr.Row():
                    show_voronoi = gr.Checkbox(label="Voronoï", value=False)
                    show_hotspots = gr.Checkbox(label="Hotspots", value=False)
                    show_mitoses = gr.Checkbox(label="Mitoses", value=False)
                    show_chromatin = gr.Checkbox(label="Chromatine", value=False)

                # Paramètres Watershed
                use_auto_params = gr.Checkbox(
                    value=True,
                    label="Params Auto (organ_config.py)",
                    info="Décocher pour utiliser les sliders manuels"
                )
                with gr.Accordion("Paramètres Watershed (Manuel)", open=False):
                    gr.Markdown("*⚠️ Ces sliders ne sont actifs que si 'Params Auto' est décoché*")
                    np_threshold = gr.Slider(
                        minimum=0.2, maximum=0.8, value=0.40, step=0.05,
                        label="Seuil NP"
                    )
                    min_size = gr.Slider(
                        minimum=10, maximum=100, value=30, step=5,
                        label="Taille min (pixels)"
                    )
                    beta = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.50, step=0.1,
                        label="Beta (HV)"
                    )
                    min_distance = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Distance min (peaks)"
                    )

                # Bouton analyse
                analyze_btn = gr.Button("Analyser", variant="primary", size="lg")

            # ================================================================
            # COLONNE DROITE: MÉTRIQUES & ALERTES
            # ================================================================
            with gr.Column(scale=1):

                # Métriques
                metrics_md = gr.Markdown("### Métriques\n*Charger une image*")

                # Alertes
                with gr.Accordion("Points d'attention", open=True):
                    alerts_md = gr.Markdown("*Aucune alerte*")

                # Info noyau au clic
                with gr.Accordion("Noyau sélectionné", open=True):
                    nucleus_info = gr.Markdown("*Cliquer sur un noyau*")

                # Chart distribution
                type_chart = gr.Image(label="Distribution", height=200)

        # Debug panel (accordéon fermé) - Phase 2 amélioré
        with gr.Accordion("Debug IA (Phase 2)", open=False):
            debug_panel = gr.Image(label="Pipeline NP/HV/Instances + Alertes", height=200)

            with gr.Row():
                anomaly_image = gr.Image(label="Vue Anomalies (F=fusion, S=sur-seg)", height=200)

            gr.Markdown("""
            **Légende Pipeline:**
            - NP Probability: Rouge = haute probabilité nucléaire
            - HV Horizontal/Vertical: Gradients [-1, 1] (bleu = négatif, rouge = positif)
            - Instances: Couleurs aléatoires par instance
            - Alertes: Fusions (>2× aire moy.) et Sur-segmentations (<0.5× aire moy.)

            **Vue Anomalies:**
            - **Magenta (F)**: Fusion potentielle - noyaux anormalement grands
            - **Cyan (S)**: Sur-segmentation - fragments trop petits
            """)

        # Debug panel Phase 3 - Intelligence Spatiale
        with gr.Accordion("Intelligence Spatiale (Phase 3)", open=False):
            phase3_debug_panel = gr.Image(label="Pléomorphisme / Clustering / Biomarqueurs", height=150)

            with gr.Row():
                phase3_overlay_image = gr.Image(label="Vue Phase 3 (Hotspots + Mitoses + Chromatine)", height=200)

            gr.Markdown("""
            **Légende Phase 3:**
            - **Pléomorphisme**: Score 1-3 basé sur variation taille/forme (anisocaryose)
            - **Hotspots** 🟠: Zones de haute densité cellulaire
            - **Mitoses** 🔴: Candidats mitose détectés par forme + chromatine
            - **Chromatine** 🟣: Noyaux à chromatine hétérogène (texture LBP + entropie)
            - **Voronoï**: Tessellation pour analyse topologique des voisinages
            """)

        # ================================================================
        # PANNEAU ZOOM (images agrandies)
        # ================================================================
        with gr.Accordion("🔍 Zoom (vue agrandie)", open=False):
            gr.Markdown("*Images affichées en taille réelle (224×224 → 448×448 pixels)*")
            with gr.Row():
                zoom_input = gr.Image(
                    label="Image Source (zoom)",
                    type="numpy",
                    height=450,
                    interactive=False,
                )
                zoom_output = gr.Image(
                    label="Segmentation (zoom)",
                    type="numpy",
                    height=450,
                    interactive=False,
                )

        # Export Phase 4
        with gr.Accordion("Export Résultats (Phase 4)", open=False):
            gr.Markdown("""
            **Formats d'export disponibles:**
            - **PDF**: Rapport clinique complet avec visualisations et métriques
            - **CSV Noyaux**: Données détaillées de chaque noyau détecté
            - **CSV Résumé**: Métriques globales et paramètres
            - **JSON**: Données complètes structurées
            """)

            with gr.Row():
                export_pdf_btn = gr.Button("Télécharger PDF", variant="primary")
                export_nuclei_csv_btn = gr.Button("CSV Noyaux", variant="secondary")
                export_summary_csv_btn = gr.Button("CSV Résumé", variant="secondary")
                export_json_btn = gr.Button("JSON", variant="secondary")

            with gr.Row():
                pdf_download = gr.File(label="Rapport PDF", visible=True)
                nuclei_csv_download = gr.File(label="CSV Noyaux", visible=True)
                summary_csv_download = gr.File(label="CSV Résumé", visible=True)

            json_output = gr.Textbox(
                label="JSON (prévisualisation)",
                lines=8,
                max_lines=15,
                interactive=False,
            )

            gr.Markdown("""
            **Note:** Chaque export inclut un identifiant unique d'analyse pour la traçabilité.
            """)

        # ================================================================
        # EVENTS
        # ================================================================

        # Charger le moteur
        load_btn.click(
            fn=lambda o: load_engine(o, "cuda"),
            inputs=[organ_select],
            outputs=[status_text],
        )

        # Changer l'organe
        organ_select.change(
            fn=change_organ,
            inputs=[organ_select],
            outputs=[status_text],
        )

        # Analyser l'image (9 outputs: overlay, contours, metrics, alerts, chart, debug, anomaly, phase3_overlay, phase3_debug)
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, np_threshold, min_size, beta, min_distance, use_auto_params],
            outputs=[output_image, output_image, metrics_md, alerts_md, type_chart, debug_panel, anomaly_image, phase3_overlay_image, phase3_debug_panel],
        )

        # Auto-analyse quand image uploadée
        input_image.change(
            fn=analyze_image,
            inputs=[input_image, np_threshold, min_size, beta, min_distance, use_auto_params],
            outputs=[output_image, output_image, metrics_md, alerts_md, type_chart, debug_panel, anomaly_image, phase3_overlay_image, phase3_debug_panel],
        )

        # Export Phase 4
        export_pdf_btn.click(
            fn=export_pdf_handler,
            outputs=[pdf_download],
        )
        export_nuclei_csv_btn.click(
            fn=export_nuclei_csv_handler,
            outputs=[nuclei_csv_download],
        )
        export_summary_csv_btn.click(
            fn=export_summary_csv_handler,
            outputs=[summary_csv_download],
        )
        export_json_btn.click(
            fn=export_json,
            outputs=[json_output],
        )

        # Clic sur l'image
        output_image.select(
            fn=on_image_click,
            outputs=[nucleus_info],
        )

        # Synchroniser le zoom avec les images principales
        input_image.change(
            fn=lambda img: img,
            inputs=[input_image],
            outputs=[zoom_input],
        )
        output_image.change(
            fn=lambda img: img,
            inputs=[output_image],
            outputs=[zoom_output],
        )

        # Update overlays (Phase 2 + Phase 3)
        all_checkboxes = [
            show_seg, show_contours, show_uncertainty, show_density, show_anomalies,
            show_voronoi, show_hotspots, show_mitoses, show_chromatin
        ]
        for checkbox in all_checkboxes:
            checkbox.change(
                fn=update_overlay,
                inputs=all_checkboxes,
                outputs=[output_image],
            )

    return app


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="CellViT-Optimus R&D Cockpit")
    parser.add_argument("--port", type=int, default=7860, help="Port Gradio")
    parser.add_argument("--share", action="store_true", help="Créer lien public")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--organ", default="Lung", help="Organe initial (ex: Lung, Breast, Colon)")
    parser.add_argument("--preload", action="store_true", help="Précharger le moteur")
    args = parser.parse_args()

    # Précharger le moteur si demandé
    if args.preload:
        logger.info("Preloading engine...")
        load_engine(args.organ, args.device)
        logger.info("Engine preloaded")

    # Créer et lancer l'interface
    app = create_ui()
    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
