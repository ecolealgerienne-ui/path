#!/usr/bin/env python3
"""
CellViT-Optimus — Interface Pathologiste.

Interface clinique simplifiée pour l'analyse histopathologique.
Masque les détails techniques et présente des métriques interprétées.

Note: Document d'aide à la décision — Validation médicale requise.

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

# Imports locaux
from src.ui.inference_engine import CellVitEngine, AnalysisResult, WATERSHED_PARAMS
from src.ui.visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    create_type_distribution_chart,
    create_hotspot_overlay,
    create_mitosis_overlay,
    TYPE_NAMES,
)
from src.ui.export import (
    create_audit_metadata,
    create_report_pdf,
)
from src.constants import FAMILIES
import tempfile
import os


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
# FONCTIONS CLINIQUES
# ==============================================================================

def compute_confidence_level(result: AnalysisResult) -> Tuple[str, str]:
    """
    Calcule le niveau de confiance global de l'IA.

    Returns:
        (niveau, couleur) - ex: ("Élevée", "green")
    """
    if result.uncertainty_map is None:
        return "Non disponible", "gray"

    # Moyenne d'incertitude
    mean_uncertainty = result.uncertainty_map.mean()

    # Confiance organe
    organ_conf = result.organ_confidence

    # Score combiné
    if mean_uncertainty < 0.3 and organ_conf > 0.9:
        return "Élevée", "green"
    elif mean_uncertainty < 0.5 and organ_conf > 0.7:
        return "Modérée", "orange"
    else:
        return "Faible", "red"


def interpret_density(density: float) -> str:
    """Interprète la densité en langage clinique."""
    if density < 1000:
        return "Faible"
    elif density < 2000:
        return "Normale"
    elif density < 3500:
        return "Élevée"
    else:
        return "Très élevée"


def interpret_pleomorphism(score: int) -> str:
    """Interprète le score de pléomorphisme."""
    interpretations = {
        1: "Faible (compatible grade I)",
        2: "Modéré (compatible grade II)",
        3: "Sévère (compatible grade III)",
    }
    return interpretations.get(score, "Non évalué")


def interpret_mitotic_index(index: float) -> str:
    """Interprète l'index mitotique."""
    if index < 3:
        return f"{index:.0f}/10 HPF (Faible)"
    elif index < 8:
        return f"{index:.0f}/10 HPF (Modéré)"
    else:
        return f"{index:.0f}/10 HPF (Élevé)"


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
        return f"Moteur chargé : {family}"

    except Exception as e:
        state.is_loading = False
        logger.error(f"Error loading engine: {e}")
        return f"Erreur : {e}"


def analyze_image(
    image: np.ndarray,
) -> Tuple[np.ndarray, str, str, str, np.ndarray, str]:
    """
    Analyse une image et retourne les visualisations cliniques.

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
        # Paramètres watershed automatiques (pas de sliders exposés)
        params = state.engine.watershed_params

        # Analyse
        result = state.engine.analyze(
            image,
            watershed_params=params,
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

        # Identification
        identification = format_identification(result)

        # Métriques cliniques
        metrics = format_metrics_clinical(result)

        # Alertes
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


def format_identification(result: AnalysisResult) -> str:
    """Formate l'identification de l'organe."""
    return f"""### {result.organ_name}
**Confiance:** {result.organ_confidence:.0%}
**Famille:** {result.family.capitalize()}"""


def format_metrics_clinical(result: AnalysisResult) -> str:
    """Formate les métriques en langage clinique (pas de valeurs brutes techniques)."""
    lines = [
        f"**Noyaux détectés:** {result.n_nuclei}",
        "",
    ]

    if result.morphometry:
        m = result.morphometry

        # Densité interprétée
        density_label = interpret_density(m.nuclei_per_mm2)
        lines.append(f"**Densité cellulaire:** {density_label} ({m.nuclei_per_mm2:.0f}/mm²)")

        # Index mitotique interprété
        mitotic_label = interpret_mitotic_index(m.mitotic_index_per_10hpf)
        lines.append(f"**Index mitotique:** {mitotic_label}")

        # Ratio néoplasique
        if m.neoplastic_ratio > 0.5:
            lines.append(f"**Ratio néoplasique:** Élevé ({m.neoplastic_ratio:.0%})")
        elif m.neoplastic_ratio > 0.2:
            lines.append(f"**Ratio néoplasique:** Modéré ({m.neoplastic_ratio:.0%})")
        else:
            lines.append(f"**Ratio néoplasique:** Faible ({m.neoplastic_ratio:.0%})")

        # TILs
        lines.append(f"**TILs:** {m.til_status}")

    # Phase 3: Pléomorphisme (interprété)
    if result.spatial_analysis:
        pleo_label = interpret_pleomorphism(result.pleomorphism_score)
        lines.append("")
        lines.append(f"**Pléomorphisme:** {pleo_label}")

    return "\n".join(lines)


def format_alerts_clinical(result: AnalysisResult) -> str:
    """Formate les alertes en langage clinique."""
    alerts = []

    # Phase 3: Alertes spatiales
    if result.spatial_analysis:
        if result.pleomorphism_score >= 3:
            alerts.append("🔴 **Anisocaryose sévère** — forte variation taille/forme nucléaire")
        elif result.pleomorphism_score == 2:
            alerts.append("🟡 **Anisocaryose modérée** — variation notable")

        if result.n_mitosis_candidates > 3:
            alerts.append(f"🔴 **Activité mitotique élevée** — {result.n_mitosis_candidates} figures suspectes")
        elif result.n_mitosis_candidates > 0:
            alerts.append(f"🟡 **Mitoses présentes** — {result.n_mitosis_candidates} figure(s)")

        if result.n_hotspots > 0:
            alerts.append(f"🟠 **Zones hypercellulaires** — {result.n_hotspots} cluster(s) identifié(s)")

    # Morphométrie
    if result.morphometry:
        m = result.morphometry
        if m.neoplastic_ratio > 0.7:
            alerts.append("🔴 **Prédominance néoplasique** — ratio > 70%")

        if m.mitotic_index_per_10hpf > 10:
            alerts.append("🔴 **Index mitotique très élevé**")

    if not alerts:
        return "✅ Aucune alerte particulière"

    return "\n\n".join(alerts)


def format_confidence_badge(result: AnalysisResult) -> str:
    """Crée le badge de confiance HTML."""
    level, color = compute_confidence_level(result)

    color_map = {
        "green": "#28a745",
        "orange": "#fd7e14",
        "red": "#dc3545",
        "gray": "#6c757d",
    }

    bg_color = color_map.get(color, "#6c757d")

    return f"""
    <div style="
        display: inline-block;
        background-color: {bg_color};
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    ">
        Confiance IA : {level}
    </div>
    """


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
    """Génère et retourne le chemin du rapport PDF."""
    if state.current_result is None:
        return None

    try:
        result = state.current_result

        # Créer l'overlay pour le PDF
        overlay = create_segmentation_overlay(
            result.image_rgb,
            result.instance_map,
            result.type_map,
            alpha=0.4,
        )

        # Créer les métadonnées d'audit
        audit = create_audit_metadata(result)

        # Générer le PDF en mémoire
        pdf_content = create_report_pdf(result, overlay, audit)

        # Sauvegarder dans un fichier temporaire
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, f"rapport_analyse_{audit.analysis_id}.pdf")

        with open(pdf_path, 'wb') as f:
            f.write(pdf_content)

        logger.info(f"PDF exported to {pdf_path}")
        return pdf_path

    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return None


def change_family(family: str) -> str:
    """Change la famille du modèle."""
    if state.engine is None:
        return "Moteur non chargé"

    try:
        state.engine.change_family(family)
        return f"Famille: {family}"
    except Exception as e:
        return f"Erreur: {e}"


# ==============================================================================
# INTERFACE GRADIO — PATHOLOGISTE
# ==============================================================================

def create_ui():
    """Crée l'interface Gradio pour pathologistes."""

    with gr.Blocks(
        title="CellViT-Optimus — Analyse Histopathologique",
        theme=gr.themes.Soft(),
        css="""
        .disclaimer {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
        }
        .clinical-alert {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 10px;
            border-radius: 5px;
        }
        """
    ) as app:

        # Header
        gr.Markdown("# CellViT-Optimus — Analyse Histopathologique")
        gr.HTML("""
        <div class="disclaimer">
            <b>Document d'aide à la décision — Validation médicale requise</b><br>
            Les résultats présentés sont des suggestions algorithmiques et doivent être validés par un pathologiste.
        </div>
        """)

        with gr.Row():
            # ================================================================
            # COLONNE GAUCHE: IMAGE
            # ================================================================
            with gr.Column(scale=2):

                # Sélection famille (simplifié)
                with gr.Row():
                    family_select = gr.Dropdown(
                        choices=FAMILIES,
                        value="respiratory",
                        label="Type de tissu",
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
