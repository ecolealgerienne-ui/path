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
from src.ui.inference_engine import (
    CellVitEngine,
    AnalysisResult,
    WATERSHED_PARAMS,
    MODEL_CHOICES,
    ORGAN_SPECIFIC_MODELS,
)
from src.ui.visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    create_uncertainty_overlay,
    create_uncertainty_map,
    create_density_heatmap,
    create_type_distribution_chart,
    create_morphometry_summary,
    create_debug_panel,
    create_debug_panel_enhanced,  # Phase 2
    create_anomaly_overlay,        # Phase 2
    highlight_nuclei,
    create_voronoi_overlay,
    # Phase 3
    create_hotspot_overlay,
    create_mitosis_overlay,
    create_chromatin_overlay,
    create_spatial_debug_panel,
    create_phase3_combined_overlay,
    CELL_COLORS,
    TYPE_NAMES,
)
# Phase 4
from src.ui.export import (
    create_audit_metadata,
    export_nuclei_csv,
    export_summary_csv,
    create_report_pdf,
)
# FAMILIES imported via inference_engine.MODEL_CHOICES
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
) -> Tuple[np.ndarray, np.ndarray, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyse une image et retourne les visualisations.

    Returns:
        (overlay, contours, metrics_text, alerts_text, chart, debug, anomaly_overlay, phase3_overlay, phase3_debug)
    """
    empty = np.zeros((224, 224, 3), dtype=np.uint8)
    empty_debug = np.zeros((100, 400, 3), dtype=np.uint8)
    empty_phase3_debug = np.zeros((80, 400, 3), dtype=np.uint8)

    if state.engine is None:
        return empty, empty, "Moteur non chargé", "", empty, empty_debug, empty, empty, empty_phase3_debug

    if image is None:
        return empty, empty, "Aucune image", "", empty, empty_debug, empty, empty, empty_phase3_debug

    # Vérification taille 224×224
    h, w = image.shape[:2]
    if h != 224 or w != 224:
        error_msg = f"**Erreur : Image {w}×{h} pixels**\n\nVeuillez charger une image de **224×224 pixels**."
        return empty, empty, error_msg, "", empty, empty_debug, empty, empty, empty_phase3_debug

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

        # Alertes texte (incluant anomalies Phase 2 et Phase 3)
        alerts_text = format_alerts(result)

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

        # Phase 3: Overlay combiné (hotspots + mitoses + chromatine)
        phase3_overlay = empty.copy()
        phase3_debug = empty_phase3_debug.copy()

        if result.spatial_analysis:
            sa = result.spatial_analysis
            # Overlay combiné Phase 3
            phase3_overlay = create_phase3_combined_overlay(
                result.image_rgb,
                result.instance_map,
                hotspot_ids=result.hotspot_ids,
                mitosis_ids=result.mitosis_candidate_ids,
                mitosis_scores=sa.mitosis_scores,
                heterogeneous_ids=sa.heterogeneous_nuclei_ids,
            )

            # Debug panel Phase 3
            phase3_debug = create_spatial_debug_panel(
                pleomorphism_score=result.pleomorphism_score,
                pleomorphism_description=result.pleomorphism_description,
                n_hotspots=result.n_hotspots,
                n_mitosis_candidates=result.n_mitosis_candidates,
                n_heterogeneous=result.n_heterogeneous_nuclei,
                mean_neighbors=result.mean_neighbors,
                mean_entropy=result.mean_chromatin_entropy,
            )

        return overlay, contours, metrics_text, alerts_text, chart, debug, anomaly_overlay, phase3_overlay, phase3_debug

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return empty, empty, f"Erreur : {e}", "", empty, empty_debug, empty, empty, empty_phase3_debug


def format_metrics(result: AnalysisResult) -> str:
    """Formate les métriques en texte."""
    # Vérifier si c'est un modèle organe spécifique
    if result.family in ORGAN_SPECIFIC_MODELS:
        model_info = f"### Modèle: **{result.family}** (dédié)"
        family_info = f"*Famille parent: {ORGAN_SPECIFIC_MODELS[result.family]['family']}*"
    else:
        model_info = f"### Famille: {result.family}"
        family_info = ""

    lines = [
        f"### Organe détecté: {result.organ_name} ({result.organ_confidence:.1%})",
        model_info,
    ]
    if family_info:
        lines.append(family_info)

    lines.extend([
        "",
        f"**Noyaux détectés:** {result.n_nuclei}",
        f"**Temps d'inférence:** {result.inference_time_ms:.0f} ms",
        "",
    ])

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

    # Phase 3: Intelligence Spatiale
    if result.spatial_analysis:
        score_labels = {1: "Faible", 2: "Modéré", 3: "Sévère"}
        score_emoji = {1: "🟢", 2: "🟡", 3: "🔴"}

        lines.extend([
            "",
            "---",
            "### Phase 3 — Intelligence Spatiale",
            f"- Pléomorphisme: **{result.pleomorphism_score}/3** {score_emoji.get(result.pleomorphism_score, '')} ({score_labels.get(result.pleomorphism_score, '')})",
            f"- Hotspots: **{result.n_hotspots}** zones haute densité",
            f"- Mitoses candidates: **{result.n_mitosis_candidates}**",
            f"- Chromatine hétérogène: **{result.n_heterogeneous_nuclei}** noyaux",
            f"- Voisins moyens (Voronoï): **{result.mean_neighbors:.1f}**",
            f"- Entropie chromatine: **{result.mean_chromatin_entropy:.2f}**",
        ])

    return "\n".join(lines)


def format_alerts(result: AnalysisResult) -> str:
    """Formate les alertes en texte (incluant anomalies Phase 2 et Phase 3)."""
    lines = ["### Points d'attention", ""]

    # Alertes morphométriques
    if result.morphometry and result.morphometry.alerts:
        for alert in result.morphometry.alerts:
            lines.append(f"- {alert}")

    # Phase 2: Alertes anomalies
    if result.n_fusions > 0:
        lines.append(f"- **{result.n_fusions} fusion(s) potentielle(s)** (aire > 2× moyenne)")
    if result.n_over_seg > 0:
        lines.append(f"- **{result.n_over_seg} sur-segmentation(s)** (aire < 0.5× moyenne)")

    # Phase 3: Alertes intelligence spatiale
    if result.spatial_analysis:
        if result.pleomorphism_score >= 3:
            lines.append("- 🔴 **Pléomorphisme sévère** — anisocaryose marquée")
        elif result.pleomorphism_score == 2:
            lines.append("- 🟡 **Pléomorphisme modéré** — variation notable")

        if result.n_mitosis_candidates > 3:
            lines.append(f"- 🔴 **{result.n_mitosis_candidates} mitoses suspectes** — activité proliférative")
        elif result.n_mitosis_candidates > 0:
            lines.append(f"- 🟡 **{result.n_mitosis_candidates} mitose(s) candidate(s)**")

        if result.n_hotspots > 0:
            lines.append(f"- 🟠 **{result.n_hotspots} hotspot(s)** — zones haute densité")

        if result.n_heterogeneous_nuclei > 5:
            lines.append(f"- 🟣 **{result.n_heterogeneous_nuclei} noyaux chromatine hétérogène**")

    if len(lines) == 2:  # Seulement le titre
        return "Aucune alerte"

    return "\n".join(lines)


def export_json() -> str:
    """Exporte les résultats en JSON."""
    if state.current_result is None:
        return '{"error": "Aucune analyse disponible"}'

    return state.current_result.to_json(indent=2)


# ==============================================================================
# FONCTIONS EXPORT PHASE 4
# ==============================================================================

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
        pdf_path = os.path.join(temp_dir, f"cellvit_report_{audit.analysis_id}.pdf")

        with open(pdf_path, 'wb') as f:
            f.write(pdf_content)

        logger.info(f"PDF exported to {pdf_path}")
        return pdf_path

    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return None


def export_nuclei_csv_handler() -> Optional[str]:
    """Génère et retourne le chemin du CSV des noyaux."""
    if state.current_result is None:
        return None

    try:
        result = state.current_result
        audit = create_audit_metadata(result)

        # Générer le CSV
        csv_content = export_nuclei_csv(result)

        # Sauvegarder dans un fichier temporaire
        temp_dir = tempfile.gettempdir()
        csv_path = os.path.join(temp_dir, f"cellvit_nuclei_{audit.analysis_id}.csv")

        with open(csv_path, 'w') as f:
            f.write(csv_content)

        logger.info(f"Nuclei CSV exported to {csv_path}")
        return csv_path

    except Exception as e:
        logger.error(f"Nuclei CSV export error: {e}")
        return None


def export_summary_csv_handler() -> Optional[str]:
    """Génère et retourne le chemin du CSV résumé."""
    if state.current_result is None:
        return None

    try:
        result = state.current_result
        audit = create_audit_metadata(result)

        # Générer le CSV
        csv_content = export_summary_csv(result, audit)

        # Sauvegarder dans un fichier temporaire
        temp_dir = tempfile.gettempdir()
        csv_path = os.path.join(temp_dir, f"cellvit_summary_{audit.analysis_id}.csv")

        with open(csv_path, 'w') as f:
            f.write(csv_content)

        logger.info(f"Summary CSV exported to {csv_path}")
        return csv_path

    except Exception as e:
        logger.error(f"Summary CSV export error: {e}")
        return None


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

        # Phase 3: Intelligence spatiale
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
                        choices=MODEL_CHOICES,
                        value="respiratory",
                        label="Modèle (Famille ou Organe)",
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

        # Analyser l'image (9 outputs: overlay, contours, metrics, alerts, chart, debug, anomaly, phase3_overlay, phase3_debug)
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, np_threshold, min_size, beta, min_distance],
            outputs=[output_image, output_image, metrics_md, alerts_md, type_chart, debug_panel, anomaly_image, phase3_overlay_image, phase3_debug_panel],
        )

        # Auto-analyse quand image uploadée
        input_image.change(
            fn=analyze_image,
            inputs=[input_image, np_threshold, min_size, beta, min_distance],
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
