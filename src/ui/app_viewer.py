#!/usr/bin/env python3
"""
CellViT-Optimus — Interface Pathologiste Simplifiée.

Design "Pathologist-First":
- Vue lame dominante (80% de l'écran)
- Score unique de suspicion avec indicateur coloré
- Focus of Interest sur zone critique
- Miniatures cliquables des zones suspectes
- Détails à la demande

Inspiré de: Paige FullFocus, PathAI AISight, QuPath

Usage:
    python -m src.ui.app_viewer
    python src/ui/app_viewer.py --wsi_dir data/wsi_test --port 7862
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import time

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports projet
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.core import (
    state,
    preload_backbone_core,
    load_engine_core,
    run_analysis_core,
)
from src.ui.inference_engine import ORGAN_CHOICES
from src.models.organ_head import OrganPrediction
from src.preprocessing import preprocess_image
from src.ui.visualizations import create_segmentation_overlay
from src.wsi.input_router import InputRouter

# ==============================================================================
# CONSTANTES
# ==============================================================================

WSI_EXTENSIONS = {'.svs', '.ndpi', '.mrxs', '.scn', '.tiff', '.tif', '.bif'}
DEFAULT_WSI_DIR = "data/wsi_test"
TILE_SIZE = 224

# Seuils de suspicion
THRESHOLD_HIGH = 0.30      # >= 30% néoplasique = Rouge
THRESHOLD_MODERATE = 0.10  # >= 10% = Orange
THRESHOLD_LOW = 0.05       # >= 5% = Jaune

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SuspiciousZone:
    """Zone suspecte identifiée dans la lame."""
    index: int
    x: int
    y: int
    image: np.ndarray      # 224x224 original
    overlay: np.ndarray    # 224x224 avec segmentation
    score: float           # Score de suspicion [0-1]
    nuclei_count: int
    neoplastic_count: int
    type_distribution: Dict[int, int]

    @property
    def label(self) -> str:
        """Label court pour affichage."""
        return f"{self.score:.0%}"

    @property
    def color_class(self) -> str:
        """Classe de couleur basée sur le score."""
        if self.score >= THRESHOLD_HIGH:
            return "high"
        elif self.score >= THRESHOLD_MODERATE:
            return "moderate"
        elif self.score >= THRESHOLD_LOW:
            return "low"
        return "normal"


@dataclass
class AnalysisState:
    """État de l'analyse en cours."""
    wsi_dir: Path = field(default_factory=lambda: Path(DEFAULT_WSI_DIR))
    available_files: List[str] = field(default_factory=list)
    selected_file: Optional[str] = None
    thumbnail: Optional[np.ndarray] = None

    # Résultats
    is_analyzed: bool = False
    global_score: float = 0.0
    total_nuclei: int = 0
    total_neoplastic: int = 0
    zones: List[SuspiciousZone] = field(default_factory=list)
    selected_zone_index: int = -1

    # Timing
    analysis_time: float = 0.0

    def clear(self):
        """Réinitialise l'état."""
        self.is_analyzed = False
        self.global_score = 0.0
        self.total_nuclei = 0
        self.total_neoplastic = 0
        self.zones = []
        self.selected_zone_index = -1
        self.analysis_time = 0.0

    def scan_folder(self) -> List[str]:
        """Scanne le dossier WSI."""
        self.available_files = []
        if self.wsi_dir.exists():
            for ext in WSI_EXTENSIONS:
                self.available_files.extend([f.name for f in self.wsi_dir.glob(f"*{ext}")])
                self.available_files.extend([f.name for f in self.wsi_dir.glob(f"*{ext.upper()}")])
        self.available_files = sorted(set(self.available_files))
        return self.available_files


# État global
analysis_state = AnalysisState()

# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def get_score_indicator(score: float) -> Tuple[str, str, str]:
    """
    Retourne l'indicateur visuel pour un score.

    Returns:
        (emoji, label, css_class)
    """
    if score >= THRESHOLD_HIGH:
        return "🔴", "SUSPECT", "score-high"
    elif score >= THRESHOLD_MODERATE:
        return "🟠", "À SURVEILLER", "score-moderate"
    elif score >= THRESHOLD_LOW:
        return "🟡", "FAIBLE RISQUE", "score-low"
    return "🟢", "NORMAL", "score-normal"


def create_opacity_overlay(
    base_image: np.ndarray,
    overlay: np.ndarray,
    opacity: float = 0.25
) -> np.ndarray:
    """
    Crée un overlay avec opacité contrôlée (style Paige).

    Args:
        base_image: Image originale RGB
        overlay: Image overlay RGB
        opacity: Opacité de l'overlay [0-1]

    Returns:
        Image fusionnée
    """
    return cv2.addWeighted(base_image, 1 - opacity, overlay, opacity, 0)


def get_wsi_thumbnail(slide_path: Path, max_size: int = 600) -> Optional[np.ndarray]:
    """Extrait le thumbnail d'une lame WSI."""
    try:
        import openslide
        slide = openslide.OpenSlide(str(slide_path))
        w, h = slide.dimensions
        ratio = max_size / max(w, h)
        thumb_size = (int(w * ratio), int(h * ratio))
        thumbnail = slide.get_thumbnail(thumb_size)
        thumbnail = np.array(thumbnail.convert('RGB'))
        slide.close()
        return thumbnail
    except Exception as e:
        logger.error(f"Erreur thumbnail: {e}")
        return None


def detect_organ(slide_path: Path, n_tiles: int = 3) -> Tuple[str, float]:
    """Détecte l'organe via OrganHead."""
    import torch

    if state.engine is None:
        preload_backbone_core(device="cuda")

    if state.engine is None or state.engine.organ_head is None:
        return "Lung", 0.0

    try:
        router = InputRouter(filter_tiles=True)
        predictions = []
        confidences = []

        for i, tile in enumerate(router.process(slide_path, max_tiles=n_tiles)):
            tensor = preprocess_image(tile.image, device="cuda")
            with torch.no_grad():
                features = state.engine.backbone.forward_features(tensor)
            cls_token = features[:, 0, :]
            organ_pred: OrganPrediction = state.engine.organ_head.predict_with_ood(cls_token)
            predictions.append(organ_pred.organ_name)
            confidences.append(organ_pred.confidence_calibrated)
            if i + 1 >= n_tiles:
                break

        if predictions:
            from collections import Counter
            predicted_organ = Counter(predictions).most_common(1)[0][0]
            avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == predicted_organ])
            return predicted_organ, avg_conf

    except Exception as e:
        logger.error(f"Erreur détection organe: {e}")

    return "Lung", 0.0


# ==============================================================================
# FONCTIONS D'ANALYSE
# ==============================================================================

def analyze_wsi(filename: str, max_tiles: int = 50) -> Tuple[Any, ...]:
    """
    Analyse une lame WSI et retourne les résultats simplifiés.

    Returns:
        (thumbnail_with_indicator, score_html, zones_gallery, zone_detail_image,
         zone_detail_overlay, zone_detail_text, status_text)
    """
    empty_image = np.zeros((400, 600, 3), dtype=np.uint8)
    empty_tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)

    if not filename:
        return (empty_image, "", [], empty_tile, empty_tile, "", "Sélectionnez une lame")

    analysis_state.clear()
    slide_path = analysis_state.wsi_dir / filename

    if not slide_path.exists():
        return (empty_image, "", [], empty_tile, empty_tile, "", f"❌ Fichier non trouvé")

    start_time = time.time()

    # 1. Thumbnail
    thumbnail = get_wsi_thumbnail(slide_path)
    if thumbnail is None:
        thumbnail = empty_image.copy()
    analysis_state.thumbnail = thumbnail

    # 2. Détection d'organe et chargement modèle
    if state.engine is None or state.engine.hovernet is None:
        organ, confidence = detect_organ(slide_path)
        logger.info(f"Organe détecté: {organ} ({confidence:.1%})")
        load_engine_core(organ, device="cuda")

    if state.engine is None or state.engine.hovernet is None:
        return (thumbnail, "", [], empty_tile, empty_tile, "", "❌ Erreur chargement modèle")

    # 3. Analyse des tiles
    router = InputRouter(filter_tiles=True)
    zones = []
    total_nuclei = 0
    total_neoplastic = 0

    for i, tile in enumerate(router.process(slide_path, max_tiles=max_tiles)):
        result, _, error = run_analysis_core(tile.image, use_auto_params=True)

        if error or result is None:
            continue

        # Compter les noyaux
        instance_map = result.instance_map
        type_map = result.type_map

        unique_ids = np.unique(instance_map)
        unique_ids = unique_ids[unique_ids > 0]

        type_counts = {}
        for nid in unique_ids:
            mask = instance_map == nid
            cell_type = int(np.median(type_map[mask]))
            type_counts[cell_type] = type_counts.get(cell_type, 0) + 1

        tile_nuclei = sum(type_counts.values())
        tile_neoplastic = type_counts.get(1, 0)  # Type 1 = Neoplastic

        total_nuclei += tile_nuclei
        total_neoplastic += tile_neoplastic

        # Score de suspicion
        score = tile_neoplastic / tile_nuclei if tile_nuclei > 0 else 0.0

        # Créer overlay
        overlay = create_segmentation_overlay(
            tile.image, instance_map, type_map, alpha=0.35
        )

        zone = SuspiciousZone(
            index=i,
            x=tile.x,
            y=tile.y,
            image=tile.image.copy(),
            overlay=overlay,
            score=score,
            nuclei_count=tile_nuclei,
            neoplastic_count=tile_neoplastic,
            type_distribution=type_counts,
        )
        zones.append(zone)

    # Trier par score décroissant
    zones.sort(key=lambda z: z.score, reverse=True)
    analysis_state.zones = zones
    analysis_state.total_nuclei = total_nuclei
    analysis_state.total_neoplastic = total_neoplastic
    analysis_state.global_score = total_neoplastic / total_nuclei if total_nuclei > 0 else 0.0
    analysis_state.is_analyzed = True
    analysis_state.analysis_time = time.time() - start_time

    # 4. Construire les outputs

    # Score global HTML
    emoji, label, css_class = get_score_indicator(analysis_state.global_score)
    score_html = f"""
    <div class="score-container {css_class}">
        <div class="score-emoji">{emoji}</div>
        <div class="score-value">{analysis_state.global_score:.0%}</div>
        <div class="score-label">{label}</div>
        <div class="score-details">{total_neoplastic:,} / {total_nuclei:,} noyaux</div>
    </div>
    """

    # Galerie des zones (top 5)
    gallery_items = []
    for zone in zones[:5]:
        emoji, _, _ = get_score_indicator(zone.score)
        label = f"{emoji} {zone.score:.0%}"
        gallery_items.append((zone.overlay, label))

    # Détails première zone
    if zones:
        first_zone = zones[0]
        analysis_state.selected_zone_index = 0
        detail_text = format_zone_details(first_zone)
        return (thumbnail, score_html, gallery_items, first_zone.image,
                first_zone.overlay, detail_text, f"✅ Analyse terminée en {analysis_state.analysis_time:.1f}s")

    return (thumbnail, score_html, gallery_items, empty_tile, empty_tile,
            "Aucune zone analysée", f"✅ Analyse terminée ({analysis_state.analysis_time:.1f}s)")


def format_zone_details(zone: SuspiciousZone) -> str:
    """Formate les détails d'une zone."""
    emoji, label, _ = get_score_indicator(zone.score)

    TYPE_NAMES = {
        0: "Background",
        1: "Néoplasique",
        2: "Inflammatoire",
        3: "Connectif",
        4: "Nécrotique",
        5: "Épithélial",
    }

    lines = [
        f"### Zone #{zone.index + 1} — {emoji} {label}",
        "",
        f"**Score:** {zone.score:.1%}",
        f"**Position:** ({zone.x:,}, {zone.y:,})",
        f"**Noyaux:** {zone.nuclei_count}",
        "",
        "**Distribution:**",
    ]

    total = sum(zone.type_distribution.values())
    for type_idx, count in sorted(zone.type_distribution.items(), key=lambda x: -x[1]):
        if type_idx == 0:
            continue
        name = TYPE_NAMES.get(type_idx, f"Type {type_idx}")
        pct = 100 * count / total if total > 0 else 0
        bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
        lines.append(f"{bar} {name}: {count} ({pct:.0f}%)")

    return "\n".join(lines)


def on_zone_select(evt: gr.SelectData) -> Tuple[np.ndarray, np.ndarray, str]:
    """Gère le clic sur une zone dans la galerie."""
    empty = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)

    if evt.index < 0 or evt.index >= len(analysis_state.zones):
        return empty, empty, "Zone non trouvée"

    zone = analysis_state.zones[evt.index]
    analysis_state.selected_zone_index = evt.index

    return zone.image, zone.overlay, format_zone_details(zone)


def on_file_select(filename: str) -> Tuple[np.ndarray, str]:
    """Gère la sélection d'un fichier."""
    empty = np.zeros((400, 600, 3), dtype=np.uint8)

    if not filename:
        return empty, "*Sélectionnez une lame*"

    slide_path = analysis_state.wsi_dir / filename
    analysis_state.selected_file = filename
    analysis_state.clear()

    thumbnail = get_wsi_thumbnail(slide_path)
    if thumbnail is None:
        return empty, f"❌ Erreur lecture: {filename}"

    analysis_state.thumbnail = thumbnail

    # Métadonnées
    try:
        import openslide
        slide = openslide.OpenSlide(str(slide_path))
        w, h = slide.dimensions
        mpp = slide.properties.get('openslide.mpp-x', 'N/A')
        vendor = slide.properties.get('openslide.vendor', 'N/A')
        slide.close()

        info = f"""### {filename}

**Dimensions:** {w:,} × {h:,} px
**MPP:** {mpp}
**Scanner:** {vendor}"""

    except Exception as e:
        info = f"### {filename}\n\n*Métadonnées non disponibles*"

    return thumbnail, info


# ==============================================================================
# INTERFACE GRADIO
# ==============================================================================

def create_viewer_ui(wsi_dir: str = DEFAULT_WSI_DIR):
    """Crée l'interface simplifiée Pathologist-First."""

    analysis_state.wsi_dir = Path(wsi_dir)
    analysis_state.scan_folder()

    # CSS personnalisé pour les scores
    custom_css = """
    .score-container {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .score-high { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
    .score-moderate { background: linear-gradient(135deg, #ffaa00, #ff8800); color: white; }
    .score-low { background: linear-gradient(135deg, #ffdd00, #ffcc00); color: #333; }
    .score-normal { background: linear-gradient(135deg, #44dd44, #22aa22); color: white; }

    .score-emoji { font-size: 48px; margin-bottom: 8px; }
    .score-value { font-size: 36px; font-weight: bold; }
    .score-label { font-size: 18px; margin-top: 4px; text-transform: uppercase; letter-spacing: 2px; }
    .score-details { font-size: 14px; margin-top: 8px; opacity: 0.9; }

    .main-viewer { min-height: 500px; }
    .zone-gallery img { border-radius: 8px; }
    """

    with gr.Blocks(
        title="CellViT-Optimus Viewer",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:

        # === HEADER ===
        gr.Markdown("""
        # 🔬 CellViT-Optimus — Diagnostic Histopathologique
        *Interface simplifiée pour pathologistes*
        """)

        with gr.Row():
            # === COLONNE GAUCHE: Contrôles (20%) ===
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Lame")

                file_dropdown = gr.Dropdown(
                    choices=analysis_state.available_files,
                    value=analysis_state.available_files[0] if analysis_state.available_files else None,
                    label="Fichier",
                    interactive=True,
                )

                analyze_btn = gr.Button(
                    "🔍 Analyser",
                    variant="primary",
                    size="lg",
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Prêt",
                )

                gr.Markdown("---")

                # Score global
                score_display = gr.HTML(
                    value="""
                    <div class="score-container score-normal">
                        <div class="score-emoji">⏳</div>
                        <div class="score-value">—</div>
                        <div class="score-label">En attente</div>
                    </div>
                    """,
                )

                gr.Markdown("---")

                # Info lame
                slide_info = gr.Markdown(
                    value="*Sélectionnez une lame*",
                )

            # === COLONNE CENTRALE: Vue principale (60%) ===
            with gr.Column(scale=3):
                gr.Markdown("### 🖼️ Vue Lame")

                main_thumbnail = gr.Image(
                    label=None,
                    height=500,
                    show_label=False,
                    elem_classes=["main-viewer"],
                )

                gr.Markdown("### 🔥 Zones Suspectes (Top 5)")

                zones_gallery = gr.Gallery(
                    label=None,
                    columns=5,
                    rows=1,
                    height=180,
                    object_fit="contain",
                    allow_preview=False,
                    elem_classes=["zone-gallery"],
                )

            # === COLONNE DROITE: Détails zone (20%) ===
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Zone Sélectionnée")

                zone_image = gr.Image(
                    label="Original",
                    height=180,
                )

                zone_overlay = gr.Image(
                    label="Segmentation",
                    height=180,
                )

                zone_details = gr.Markdown(
                    value="*Cliquez sur une zone*",
                )

        # === ÉVÉNEMENTS ===

        file_dropdown.change(
            fn=on_file_select,
            inputs=[file_dropdown],
            outputs=[main_thumbnail, slide_info],
        )

        analyze_btn.click(
            fn=analyze_wsi,
            inputs=[file_dropdown],
            outputs=[
                main_thumbnail,
                score_display,
                zones_gallery,
                zone_image,
                zone_overlay,
                zone_details,
                status_text,
            ],
        )

        zones_gallery.select(
            fn=on_zone_select,
            outputs=[zone_image, zone_overlay, zone_details],
        )

        # Charger le thumbnail au démarrage
        app.load(
            fn=on_file_select,
            inputs=[file_dropdown],
            outputs=[main_thumbnail, slide_info],
        )

    return app


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CellViT-Optimus Viewer (Pathologist Interface)")
    parser.add_argument("--wsi_dir", type=str, default=DEFAULT_WSI_DIR,
                        help=f"Dossier WSI (défaut: {DEFAULT_WSI_DIR})")
    parser.add_argument("--port", type=int, default=7862,
                        help="Port (défaut: 7862)")
    parser.add_argument("--share", action="store_true",
                        help="Créer un lien public")
    args = parser.parse_args()

    logger.info(f"Démarrage CellViT-Optimus Viewer sur port {args.port}")
    logger.info(f"Dossier WSI: {args.wsi_dir}")

    app = create_viewer_ui(wsi_dir=args.wsi_dir)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
