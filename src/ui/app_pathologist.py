#!/usr/bin/env python3
"""
CellViT-Optimus — Interface Pathologiste.

Interface clinique simplifiée pour l'analyse histopathologique.
Masque les détails techniques et présente des métriques interprétées.

Note: Document d'aide à la décision — Validation médicale requise.

Architecture: Utilise src.ui.core pour la logique métier partagée
et src.ui.formatters pour l'affichage clinique (simplifié).

Usage:
    python -m src.ui.app_pathologist
    # ou
    python src/ui/app_pathologist.py

    Puis ouvrir http://localhost:7861
"""

import gradio as gr
import numpy as np
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
    change_organ_core,
    export_pdf_core,
)

# Imports: Formatage clinique (simplifié)
from src.ui.formatters import (
    format_metrics_clinical,
    format_alerts_clinical,
    format_nucleus_info_clinical,
    format_load_status_clinical,
    format_organ_change_clinical,
    format_identification_clinical,
    format_confidence_badge,
)

# Imports: Moteur et configuration
from src.ui.inference_engine import ORGAN_CHOICES
from src.ui.organ_config import ORGANS, get_model_for_organ

# Imports: Visualisations
from src.ui.visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    create_type_distribution_chart,
    create_hotspot_overlay,
    create_mitosis_overlay,
)


# ==============================================================================
# WRAPPERS UI (utilisent core + formatters cliniques)
# ==============================================================================

def load_engine(organ: str, device: str = "cuda") -> str:
    """Charge le moteur d'inférence (wrapper UI)."""
    result = load_engine_core(organ, device)
    return format_load_status_clinical(result)


def analyze_image(
    image: np.ndarray,
) -> Tuple[np.ndarray, str, str, str, np.ndarray, str]:
    """
    Analyse une image et retourne les visualisations cliniques (wrapper UI).

    Returns:
        (overlay, identification, metrics, alerts, chart, confidence_html)
    """
    empty = np.zeros((224, 224, 3), dtype=np.uint8)

    if state.engine is None:
        return empty, "Moteur non chargé", "", "", empty, ""

    if image is None:
        return empty, "Aucune image", "", "", empty, ""

    # Vérification taille 224×224
    h, w = image.shape[:2]
    if h != 224 or w != 224:
        error_msg = f"Image {w}×{h} — Requis: 224×224"
        return empty, error_msg, "", "", empty, ""

    try:
        # Mode Auto: laisser le moteur utiliser les params optimisés
        # pour l'organe prédit (organ_config.py)
        # Note: watershed_params=None déclenche le mode Auto dans inference_engine

        # Analyse via le moteur
        result = state.engine.analyze(
            image,
            watershed_params=None,  # Auto: utilise organ_config.py
            compute_morphometry=True,
            compute_uncertainty=True,
        )

        state.current_result = result

        # Overlay simplifié (types + contours)
        overlay = create_segmentation_overlay(
            result.image_rgb,
            result.instance_map,
            result.type_map,
            alpha=0.4,
        )
        overlay = create_contour_overlay(
            overlay,
            result.instance_map,
            result.type_map,
            thickness=1,
        )

        # Ajouter hotspots si présents
        if result.hotspot_ids:
            overlay = create_hotspot_overlay(overlay, result.instance_map, result.hotspot_ids)

        # Ajouter mitoses si présentes
        if result.mitosis_candidate_ids and result.spatial_analysis:
            overlay = create_mitosis_overlay(
                overlay, result.instance_map,
                result.mitosis_candidate_ids,
                result.spatial_analysis.mitosis_scores
            )

        # Formatage clinique
        organ = state.engine.organ
        family = state.engine.family
        is_dedicated = state.engine.is_dedicated_model

        identification = format_identification_clinical(result, organ, family, is_dedicated)
        metrics = format_metrics_clinical(result, organ, family, is_dedicated)
        alerts = format_alerts_clinical(result)

        # Chart distribution
        if result.morphometry:
            chart = create_type_distribution_chart(result.morphometry.type_counts)
        else:
            chart = np.zeros((200, 300, 3), dtype=np.uint8)

        # Indicateur confiance
        confidence_html = format_confidence_badge(result)

        return overlay, identification, metrics, alerts, chart, confidence_html

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return empty, f"Erreur : {e}", "", "", empty, ""


def on_image_click(evt: gr.SelectData) -> str:
    """Gère le clic sur l'image pour afficher les infos du noyau (simplifié)."""
    if state.current_result is None:
        return "Aucune analyse"

    try:
        x, y = evt.index
        nucleus = state.current_result.get_nucleus_at(y, x)

        if nucleus is None:
            return "*Cliquer sur un noyau pour voir ses détails*"

        lines = [
            f"### Noyau #{nucleus.id}",
            "",
            f"**Type:** {nucleus.cell_type}",
            f"**Aire:** {nucleus.area_um2:.1f} µm²",
            f"**Forme:** {'Régulière' if nucleus.circularity > 0.7 else 'Irrégulière'}",
        ]

        # Alertes simplifiées
        if nucleus.is_mitosis_candidate:
            lines.append("")
            lines.append("🔴 **Mitose suspecte**")

        if nucleus.is_in_hotspot:
            lines.append("🟠 **Zone hypercellulaire**")

        return "\n".join(lines)

    except Exception as e:
        return f"Erreur: {e}"


def update_overlay(
    show_types: bool,
    show_contours: bool,
    show_hotspots: bool,
    show_mitoses: bool,
) -> np.ndarray:
    """Met à jour l'overlay selon les options cliniques."""
    if state.current_result is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    result = state.current_result
    image = result.image_rgb.copy()

    if show_types:
        image = create_segmentation_overlay(
            image, result.instance_map, result.type_map, alpha=0.3
        )

    if show_contours:
        image = create_contour_overlay(
            image, result.instance_map, result.type_map, thickness=1
        )

    if show_hotspots and result.hotspot_ids:
        image = create_hotspot_overlay(image, result.instance_map, result.hotspot_ids)

    if show_mitoses and result.mitosis_candidate_ids and result.spatial_analysis:
        image = create_mitosis_overlay(
            image, result.instance_map,
            result.mitosis_candidate_ids,
            result.spatial_analysis.mitosis_scores
        )

    return image


def export_pdf_handler() -> Optional[str]:
    """Génère et retourne le chemin du rapport PDF (wrapper UI)."""
    return export_pdf_core()


def change_organ(organ: str) -> str:
    """Change l'organe du modèle (wrapper UI)."""
    result = change_organ_core(organ)
    return format_organ_change_clinical(result)


# ==============================================================================
# INTERFACE GRADIO — PATHOLOGISTE
# ==============================================================================

def create_ui():
    """Crée l'interface Gradio pour pathologistes."""

    with gr.Blocks(
        title="CellViT-Optimus — Analyse Histopathologique",
    ) as app:

        # Header
        gr.Markdown("# CellViT-Optimus — Analyse Histopathologique")
        gr.HTML("""
        <div style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 12px; border-radius: 8px; margin-bottom: 15px; text-align: center;">
            <b>Document d'aide à la décision — Validation médicale requise</b><br>
            Les résultats présentés sont des suggestions algorithmiques et doivent être validés par un pathologiste.
        </div>
        """)

        with gr.Row():
            # ================================================================
            # COLONNE GAUCHE: IMAGE
            # ================================================================
            with gr.Column(scale=2):

                # Sélection organe
                with gr.Row():
                    # Créer les choix avec labels clairs pour les pathologistes
                    pathologist_choices = []
                    for organ_name in ORGAN_CHOICES:
                        organ_info = ORGANS[organ_name]
                        if organ_info.has_dedicated_model:
                            pathologist_choices.append(f"{organ_name} ★")
                        else:
                            pathologist_choices.append(f"{organ_name} ({organ_info.family})")

                    organ_select = gr.Dropdown(
                        choices=ORGAN_CHOICES,
                        value="Lung",
                        label="Organe (★ = modèle dédié)",
                        interactive=True,
                    )
                    load_btn = gr.Button("Charger", variant="primary")
                    status_text = gr.Textbox(label="Status", interactive=False, scale=2)

                # Image
                with gr.Row():
                    input_image = gr.Image(
                        label="Image H&E (224×224)",
                        type="numpy",
                        height=320,
                    )
                    output_image = gr.Image(
                        label="Analyse",
                        type="numpy",
                        height=320,
                        interactive=True,
                    )

                # Overlays cliniques (4 seulement)
                with gr.Row():
                    show_types = gr.Checkbox(label="Types cellulaires", value=True)
                    show_contours = gr.Checkbox(label="Contours", value=True)
                    show_hotspots = gr.Checkbox(label="Zones denses", value=True)
                    show_mitoses = gr.Checkbox(label="Mitoses", value=True)

                # Bouton analyse
                analyze_btn = gr.Button("Analyser", variant="primary", size="lg")

            # ================================================================
            # COLONNE DROITE: RÉSULTATS CLINIQUES
            # ================================================================
            with gr.Column(scale=1):

                # Badge confiance
                confidence_badge = gr.HTML("")

                # Identification
                identification_md = gr.Markdown("### En attente d'analyse...")

                # Métriques cliniques
                gr.Markdown("---")
                metrics_md = gr.Markdown("*Charger une image*")

                # Distribution
                type_chart = gr.Image(label="Distribution cellulaire", height=180)

                # Alertes (encadré)
                gr.Markdown("---")
                gr.Markdown("### Points d'attention")
                alerts_md = gr.Markdown("*Aucune alerte*")

        # Info noyau au clic
        with gr.Accordion("Détails du noyau sélectionné", open=False):
            nucleus_info = gr.Markdown("*Cliquer sur un noyau dans l'image*")

        # Détails avancés (optionnel, masqué par défaut)
        with gr.Accordion("Informations complémentaires", open=False):
            gr.Markdown("""
            **Pour experts uniquement** — Ces informations techniques peuvent aider à comprendre
            l'analyse mais ne sont pas nécessaires pour l'interprétation clinique.
            """)

            with gr.Row():
                advanced_info = gr.Markdown("*Analyser une image pour voir les détails*")

        # Export
        with gr.Row():
            export_pdf_btn = gr.Button("Télécharger le rapport PDF", variant="primary", size="lg")
            pdf_download = gr.File(label="Rapport", visible=True)

        # Footer
        gr.Markdown("""
        ---
        <center>
        <small>CellViT-Optimus v4.0 — Ce système est un outil d'aide à la décision et ne remplace pas le diagnostic médical professionnel.</small>
        </center>
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

        # Analyser l'image
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[output_image, identification_md, metrics_md, alerts_md, type_chart, confidence_badge],
        )

        # Auto-analyse quand image uploadée
        input_image.change(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[output_image, identification_md, metrics_md, alerts_md, type_chart, confidence_badge],
        )

        # Export PDF
        export_pdf_btn.click(
            fn=export_pdf_handler,
            outputs=[pdf_download],
        )

        # Clic sur l'image
        output_image.select(
            fn=on_image_click,
            outputs=[nucleus_info],
        )

        # Update overlays
        overlay_checkboxes = [show_types, show_contours, show_hotspots, show_mitoses]
        for checkbox in overlay_checkboxes:
            checkbox.change(
                fn=update_overlay,
                inputs=overlay_checkboxes,
                outputs=[output_image],
            )

    return app


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="CellViT-Optimus — Interface Pathologiste")
    parser.add_argument("--port", type=int, default=7861, help="Port Gradio")
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
