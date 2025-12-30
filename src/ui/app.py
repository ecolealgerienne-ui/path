#!/usr/bin/env python3
"""
CellViT-Optimus R&D Cockpit — Interface Gradio.

Application web interactive pour l'exploration et la validation
du moteur IA de segmentation cellulaire.

Note: Ceci est un outil R&D, pas une IHM clinique.

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

# Imports locaux
from src.ui.inference_engine import CellVitEngine, AnalysisResult, WATERSHED_PARAMS
from src.ui.visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    create_uncertainty_overlay,
    create_uncertainty_map,
    create_density_heatmap,
    create_type_distribution_chart,
    create_morphometry_summary,
    create_debug_panel,
    highlight_nuclei,
    CELL_COLORS,
    TYPE_NAMES,
)
from src.constants import FAMILIES


# ==============================================================================
# ÉTAT GLOBAL
# ==============================================================================

class AppState:
    """État global de l'application."""
    engine: Optional[CellVitEngine] = None
    current_result: Optional[AnalysisResult] = None
    is_loading: bool = False


state = AppState()


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def load_engine(family: str, device: str = "cuda") -> str:
    """Charge le moteur d'inférence."""
    try:
        state.is_loading = True
        logger.info(f"Loading engine for family '{family}' on {device}...")

        state.engine = CellVitEngine(
            device=device,
            family=family,
            load_backbone=True,
            load_organ_head=True,
        )

        state.is_loading = False
        return f"Moteur chargé : {family} sur {device}"

    except Exception as e:
        state.is_loading = False
        logger.error(f"Error loading engine: {e}")
        return f"Erreur : {e}"


def analyze_image(
    image: np.ndarray,
    np_threshold: float,
    min_size: int,
    beta: float,
    min_distance: int,
) -> Tuple[np.ndarray, np.ndarray, str, str, np.ndarray, np.ndarray]:
    """
    Analyse une image et retourne les visualisations.

    Returns:
        (overlay, contours, metrics_text, alerts_text, chart, debug)
    """
    if state.engine is None:
        empty = np.zeros((224, 224, 3), dtype=np.uint8)
        return empty, empty, "Moteur non chargé", "", empty, empty

    if image is None:
        empty = np.zeros((224, 224, 3), dtype=np.uint8)
        return empty, empty, "Aucune image", "", empty, empty

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

        # Métriques texte
        metrics_text = format_metrics(result)

        # Alertes texte
        alerts_text = format_alerts(result)

        # Chart distribution
        if result.morphometry:
            chart = create_type_distribution_chart(result.morphometry.type_counts)
        else:
            chart = np.zeros((200, 300, 3), dtype=np.uint8)

        # Debug panel
        debug = create_debug_panel(
            result.np_pred,
            result.hv_pred,
            result.instance_map,
        )

        return overlay, contours, metrics_text, alerts_text, chart, debug

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        empty = np.zeros((224, 224, 3), dtype=np.uint8)
        return empty, empty, f"Erreur : {e}", "", empty, empty


def format_metrics(result: AnalysisResult) -> str:
    """Formate les métriques en texte."""
    lines = [
        f"### Organe détecté: {result.organ_name} ({result.organ_confidence:.1%})",
        f"### Famille: {result.family}",
        "",
        f"**Noyaux détectés:** {result.n_nuclei}",
        f"**Temps d'inférence:** {result.inference_time_ms:.0f} ms",
        "",
    ]

    if result.morphometry:
        m = result.morphometry
        lines.extend([
            "---",
            "### Morphométrie",
            f"- Densité: **{m.nuclei_per_mm2:.0f}** noyaux/mm²",
            f"- Aire moyenne: **{m.mean_area_um2:.1f}** ± {m.std_area_um2:.1f} µm²",
            f"- Circularité: **{m.mean_circularity:.2f}** ± {m.std_circularity:.2f}",
            f"- Hypercellularité: **{m.nuclear_density_percent:.1f}%**",
            "",
            "### Index & Ratios",
            f"- Index mitotique: **{m.mitotic_index_per_10hpf:.1f}**/10 HPF",
            f"- Ratio néoplasique: **{m.neoplastic_ratio:.1%}**",
            f"- Ratio I/E: **{m.immuno_epithelial_ratio:.2f}**",
            f"- TILs status: **{m.til_status}**",
            "",
            "### Distribution",
        ])

        for t in TYPE_NAMES:
            count = m.type_counts.get(t, 0)
            pct = m.type_percentages.get(t, 0)
            lines.append(f"- {t}: {count} ({pct:.1f}%)")

        lines.extend([
            "",
            f"**Confiance:** {m.confidence_level}",
        ])

    return "\n".join(lines)


def format_alerts(result: AnalysisResult) -> str:
    """Formate les alertes en texte."""
    if not result.morphometry or not result.morphometry.alerts:
        return "Aucune alerte"

    lines = ["### Points d'attention", ""]
    for alert in result.morphometry.alerts:
        lines.append(f"- {alert}")

    return "\n".join(lines)


def on_image_click(evt: gr.SelectData) -> str:
    """Gère le clic sur l'image pour afficher les infos du noyau."""
    if state.current_result is None:
        return "Aucune analyse"

    try:
        x, y = evt.index
        nucleus = state.current_result.get_nucleus_at(y, x)

        if nucleus is None:
            return "Clic sur le fond (pas de noyau)"

        lines = [
            f"### Noyau #{nucleus.id}",
            "",
            f"**Type:** {nucleus.cell_type}",
            f"**Position:** ({nucleus.centroid[1]}, {nucleus.centroid[0]})",
            f"**Aire:** {nucleus.area_um2:.1f} µm²",
            f"**Périmètre:** {nucleus.perimeter_um:.1f} µm",
            f"**Circularité:** {nucleus.circularity:.2f}",
            f"**Confiance:** {nucleus.confidence:.1%}",
        ]

        if nucleus.is_uncertain:
            lines.append("**Status:** Incertain")
        if nucleus.is_mitotic:
            lines.append("**Status:** Mitose suspecte")

        return "\n".join(lines)

    except Exception as e:
        return f"Erreur: {e}"


def update_overlay(
    show_segmentation: bool,
    show_contours: bool,
    show_uncertainty: bool,
    show_density: bool,
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

    return image


def change_family(family: str) -> str:
    """Change la famille du modèle."""
    if state.engine is None:
        return "Moteur non chargé"

    try:
        state.engine.change_family(family)
        params = state.engine.watershed_params
        return f"Famille changée: {family}\nParams: {params}"
    except Exception as e:
        return f"Erreur: {e}"


# ==============================================================================
# INTERFACE GRADIO
# ==============================================================================

def create_ui():
    """Crée l'interface Gradio."""

    with gr.Blocks(
        title="CellViT-Optimus R&D Cockpit",
        theme=gr.themes.Soft(),
        css="""
        .disclaimer {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        """
    ) as app:

        # Header
        gr.Markdown("# CellViT-Optimus — R&D Cockpit")
        gr.HTML("""
        <div class="disclaimer">
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
                    family_select = gr.Dropdown(
                        choices=FAMILIES,
                        value="respiratory",
                        label="Famille d'organes",
                        interactive=True,
                    )
                    load_btn = gr.Button("Charger le moteur", variant="primary")
                    status_text = gr.Textbox(label="Status", interactive=False)

                # Image input/output
                with gr.Row():
                    input_image = gr.Image(
                        label="Image H&E (256×256)",
                        type="numpy",
                        height=300,
                    )
                    output_image = gr.Image(
                        label="Segmentation",
                        type="numpy",
                        height=300,
                        interactive=True,
                    )

                # Overlays controls
                with gr.Row():
                    show_seg = gr.Checkbox(label="Segmentation", value=True)
                    show_contours = gr.Checkbox(label="Contours", value=True)
                    show_uncertainty = gr.Checkbox(label="Incertitude", value=False)
                    show_density = gr.Checkbox(label="Densité", value=False)

                # Paramètres Watershed
                with gr.Accordion("Paramètres Watershed", open=False):
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

        # Debug panel (accordéon fermé)
        with gr.Accordion("Debug IA", open=False):
            debug_panel = gr.Image(label="Pipeline NP/HV/Instances", height=200)
            gr.Markdown("""
            **Légende:**
            - NP Probability: Rouge = haute probabilité nucléaire
            - HV Horizontal/Vertical: Gradients [-1, 1] (bleu = négatif, rouge = positif)
            - Instances: Couleurs aléatoires par instance
            """)

        # ================================================================
        # EVENTS
        # ================================================================

        # Charger le moteur
        load_btn.click(
            fn=lambda f: load_engine(f, "cuda"),
            inputs=[family_select],
            outputs=[status_text],
        )

        # Changer la famille
        family_select.change(
            fn=change_family,
            inputs=[family_select],
            outputs=[status_text],
        )

        # Analyser l'image
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, np_threshold, min_size, beta, min_distance],
            outputs=[output_image, output_image, metrics_md, alerts_md, type_chart, debug_panel],
        )

        # Auto-analyse quand image uploadée
        input_image.change(
            fn=analyze_image,
            inputs=[input_image, np_threshold, min_size, beta, min_distance],
            outputs=[output_image, output_image, metrics_md, alerts_md, type_chart, debug_panel],
        )

        # Clic sur l'image
        output_image.select(
            fn=on_image_click,
            outputs=[nucleus_info],
        )

        # Update overlays
        for checkbox in [show_seg, show_contours, show_uncertainty, show_density]:
            checkbox.change(
                fn=update_overlay,
                inputs=[show_seg, show_contours, show_uncertainty, show_density],
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
    parser.add_argument("--family", default="respiratory", help="Famille initiale")
    parser.add_argument("--preload", action="store_true", help="Précharger le moteur")
    args = parser.parse_args()

    # Précharger le moteur si demandé
    if args.preload:
        logger.info("Preloading engine...")
        load_engine(args.family, args.device)
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
