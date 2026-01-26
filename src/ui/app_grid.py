#!/usr/bin/env python3
"""
CellViT-Optimus — Navigation Grille Multi-Patches (Simulation WSI).

Interface de simulation WSI utilisant des images PanNuke (256×256).
Extraction automatique de 4 patches 224×224 (grille 2×2 avec chevauchement).

Workflow:
1. Upload d'une image 256×256
2. Extraction automatique → 4 patches 224×224
3. Analyse automatique des 4 patches
4. Affichage grille 4 miniatures cliquables
5. Clic → Détails du patch sélectionné
6. Métriques WSI agrégées toujours visibles

Usage:
    python -m src.ui.app_grid --organ Lung
    python src/ui/app_grid.py --organ Breast --port 7861
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
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
    preload_backbone_core,
    load_engine_core,
    run_analysis_core,
)

# Imports: Moteur et configuration
from src.ui.inference_engine import ORGAN_CHOICES, AnalysisResult

# Imports: Visualisations
from src.ui.visualizations import (
    create_segmentation_overlay,
)

# ==============================================================================
# CONSTANTES
# ==============================================================================

PANNUKE_SIZE = 256
PATCH_SIZE = 224
OFFSET = PANNUKE_SIZE - PATCH_SIZE  # 32 pixels

# Positions des 4 patches (grille 2×2 avec chevauchement)
PATCH_POSITIONS = [
    (0, 0),           # Top-Left
    (OFFSET, 0),      # Top-Right
    (0, OFFSET),      # Bottom-Left
    (OFFSET, OFFSET), # Bottom-Right
]

PATCH_NAMES = ["Haut-Gauche", "Haut-Droite", "Bas-Gauche", "Bas-Droite"]


# ==============================================================================
# ÉTAT GRILLE (SIMULATION WSI)
# ==============================================================================

@dataclass
class PatchInfo:
    """Information sur un patch extrait."""
    index: int
    name: str
    position: Tuple[int, int]  # (x, y) offset
    image: np.ndarray  # Patch 224×224
    result: Optional[AnalysisResult] = None
    overlay: Optional[np.ndarray] = None
    is_analyzed: bool = False


@dataclass
class WSIState:
    """État de la simulation WSI (une image source)."""
    source_image: Optional[np.ndarray] = None  # Image originale 256×256
    source_filename: str = ""
    patches: List[PatchInfo] = field(default_factory=list)
    selected_index: int = 0

    def clear(self):
        """Réinitialise l'état."""
        self.source_image = None
        self.source_filename = ""
        self.patches = []
        self.selected_index = 0

    def get_selected(self) -> Optional[PatchInfo]:
        """Retourne le patch sélectionné."""
        if 0 <= self.selected_index < len(self.patches):
            return self.patches[self.selected_index]
        return None

    def all_analyzed(self) -> bool:
        """Vérifie si tous les patches sont analysés."""
        return all(p.is_analyzed for p in self.patches)

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques agrégées sur tous les patches."""
        analyzed = [p for p in self.patches if p.is_analyzed and p.result]

        if not analyzed:
            return {"total_patches": len(self.patches), "analyzed": 0}

        # Agrégation
        total_nuclei = 0
        type_counts = {}
        total_area = 0.0

        for p in analyzed:
            r = p.result
            if r.morphometry:
                total_nuclei += r.morphometry.n_nuclei
                for cell_type, count in r.morphometry.type_counts.items():
                    type_counts[cell_type] = type_counts.get(cell_type, 0) + count
                total_area += r.morphometry.patch_area_mm2 * 1_000_000  # mm² → µm²

        return {
            "total_patches": len(self.patches),
            "analyzed": len(analyzed),
            "total_nuclei": total_nuclei,
            "type_counts": type_counts,
            "total_area_um2": total_area,
            "avg_nuclei_per_patch": total_nuclei / len(analyzed) if analyzed else 0,
        }


# Instance globale
wsi_state = WSIState()


# ==============================================================================
# EXTRACTION DE PATCHES
# ==============================================================================

def extract_patches_2x2(image: np.ndarray) -> List[np.ndarray]:
    """
    Extrait 4 patches 224×224 d'une image 256×256 (grille 2×2 avec chevauchement).

    Args:
        image: Image source 256×256 RGB

    Returns:
        Liste de 4 patches 224×224
    """
    h, w = image.shape[:2]
    if h != PANNUKE_SIZE or w != PANNUKE_SIZE:
        raise ValueError(f"Image doit être {PANNUKE_SIZE}×{PANNUKE_SIZE}, reçu {w}×{h}")

    patches = []
    for x, y in PATCH_POSITIONS:
        patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
        patches.append(patch)

    return patches


# ==============================================================================
# FONCTIONS D'ANALYSE
# ==============================================================================

def process_uploaded_image(image: np.ndarray) -> Tuple[
    List[Tuple[np.ndarray, str]],  # gallery items
    np.ndarray,  # selected patch
    np.ndarray,  # overlay
    str,  # patch metrics
    str,  # wsi metrics
    str,  # status
]:
    """
    Traite une image uploadée: extraction + analyse automatique des 4 patches.

    Returns:
        (gallery, selected_patch, overlay, patch_metrics, wsi_metrics, status)
    """
    empty = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)

    # Vérifier que le moteur est chargé
    if state.engine is None:
        return [], empty, empty, "", "", "❌ Moteur non chargé — Sélectionner un organe d'abord"

    # Vérifier la taille de l'image
    if image is None:
        return [], empty, empty, "", "", "❌ Aucune image"

    h, w = image.shape[:2]
    if h != PANNUKE_SIZE or w != PANNUKE_SIZE:
        return [], empty, empty, "", "", f"❌ Taille invalide: {w}×{h} (attendu: {PANNUKE_SIZE}×{PANNUKE_SIZE})"

    # Réinitialiser l'état
    wsi_state.clear()
    wsi_state.source_image = image
    wsi_state.source_filename = "uploaded_image.png"

    # Extraire les 4 patches
    logger.info(f"Extraction de 4 patches {PATCH_SIZE}×{PATCH_SIZE} (grille 2×2)")
    patch_images = extract_patches_2x2(image)

    # Créer les PatchInfo
    for i, (patch_img, pos, name) in enumerate(zip(patch_images, PATCH_POSITIONS, PATCH_NAMES)):
        patch = PatchInfo(
            index=i,
            name=name,
            position=pos,
            image=patch_img,
        )
        wsi_state.patches.append(patch)

    # Analyser tous les patches automatiquement
    logger.info("Analyse automatique des 4 patches...")
    n_success = 0

    for patch in wsi_state.patches:
        # run_analysis_core attend du 224×224 (InputRouter va le passer directement)
        result, preprocessed, error = run_analysis_core(patch.image, use_auto_params=True)

        if error:
            logger.warning(f"Erreur patch {patch.name}: {error}")
            continue

        patch.result = result
        patch.is_analyzed = True

        # Créer overlay
        overlay = create_segmentation_overlay(
            result.image_rgb,
            result.instance_map,
            result.type_map,
            alpha=0.4,
        )
        patch.overlay = overlay
        n_success += 1

    # Construire la galerie
    gallery_items = []
    for p in wsi_state.patches:
        # Miniature avec indicateur de statut
        thumb = cv2.resize(p.image, (112, 112))
        label = f"{p.name}"
        if p.is_analyzed:
            label = f"✅ {label}"
        gallery_items.append((thumb, label))

    # Sélectionner le premier patch
    wsi_state.selected_index = 0
    selected = wsi_state.get_selected()

    # Métriques
    patch_md = format_patch_metrics(selected) if selected else ""
    wsi_md = format_wsi_metrics()

    status = f"✅ {n_success}/4 patches analysés"

    return (
        gallery_items,
        selected.image if selected else empty,
        selected.overlay if selected and selected.overlay is not None else empty,
        patch_md,
        wsi_md,
        status,
    )


def on_patch_select(evt: gr.SelectData) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Gère le clic sur un patch dans la galerie.

    Returns:
        (patch_image, overlay, patch_metrics)
    """
    empty = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)

    index = evt.index
    if index < 0 or index >= len(wsi_state.patches):
        return empty, empty, "❌ Index invalide"

    wsi_state.selected_index = index
    patch = wsi_state.patches[index]

    if patch.is_analyzed and patch.overlay is not None:
        return patch.image, patch.overlay, format_patch_metrics(patch)

    return patch.image, empty, format_patch_metrics(patch)


# ==============================================================================
# FORMATAGE
# ==============================================================================

def format_patch_metrics(patch: Optional[PatchInfo]) -> str:
    """Formate les métriques d'un patch."""
    if patch is None:
        return "*Aucun patch sélectionné*"

    if not patch.is_analyzed or patch.result is None:
        return f"### {patch.name}\n\n*Non analysé*"

    r = patch.result
    lines = [
        f"### Patch: {patch.name}",
        f"*Position: ({patch.position[0]}, {patch.position[1]})*",
        "",
    ]

    if r.morphometry:
        m = r.morphometry
        lines.extend([
            f"**Noyaux:** {m.n_nuclei}",
            f"**Densité:** {m.nuclei_per_mm2:.0f} /mm²",
            "",
            "**Distribution:**",
        ])
        for cell_type, count in sorted(m.type_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / m.n_nuclei if m.n_nuclei > 0 else 0
            lines.append(f"- {cell_type}: {count} ({pct:.1f}%)")

    return "\n".join(lines)


def format_wsi_metrics() -> str:
    """Formate les métriques WSI agrégées."""
    agg = wsi_state.get_aggregated_metrics()

    if agg["analyzed"] == 0:
        return "**Patches:** 0 analysés\n\n*En attente d'analyse...*"

    lines = [
        "## Métriques WSI Globales",
        "",
        f"**Source:** {wsi_state.source_filename}",
        f"**Patches:** {agg['analyzed']} / {agg['total_patches']} analysés",
        "",
        f"### Total Noyaux: {agg['total_nuclei']}",
        f"**Moyenne/patch:** {agg['avg_nuclei_per_patch']:.1f}",
        f"**Surface totale:** {agg['total_area_um2']:.0f} µm²",
        "",
        "### Distribution Globale",
    ]

    type_counts = agg.get("type_counts", {})
    total = sum(type_counts.values()) if type_counts else 0

    for cell_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"- **{cell_type}:** {count} ({pct:.1f}%)")

    return "\n".join(lines)


# ==============================================================================
# CHARGEMENT MOTEUR
# ==============================================================================

def load_engine_for_grid(organ: str) -> str:
    """Charge le moteur pour l'organe spécifié."""
    result = load_engine_core(organ, device="cuda")
    if result["success"]:
        return f"✅ Moteur chargé: {organ} ({result['model_type']})"
    return f"❌ Erreur: {result['error']}"


# ==============================================================================
# INTERFACE GRADIO
# ==============================================================================

def create_grid_ui():
    """Crée l'interface de navigation grille WSI."""

    with gr.Blocks(
        title="CellViT-Optimus — Simulation WSI",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown("""
        # 🔬 CellViT-Optimus — Simulation WSI (PanNuke 256×256)

        **Workflow automatique:**
        1. Sélectionner un **organe** → Charger le modèle
        2. **Uploader** une image PanNuke (256×256)
        3. Extraction automatique de **4 patches** 224×224 (grille 2×2)
        4. **Analyse automatique** des 4 patches
        5. **Cliquer** sur un patch pour voir les détails
        """)

        with gr.Row():
            # === COLONNE GAUCHE: Config + Grille ===
            with gr.Column(scale=1):
                gr.Markdown("### 1. Configuration")

                organ_dropdown = gr.Dropdown(
                    choices=ORGAN_CHOICES,
                    value="Lung",
                    label="Organe",
                )
                load_btn = gr.Button("🚀 Charger Modèle", variant="primary")
                model_status = gr.Textbox(label="Status Modèle", interactive=False)

                gr.Markdown("### 2. Image Source (256×256)")

                input_image = gr.Image(
                    label="Upload PanNuke 256×256",
                    type="numpy",
                    height=200,
                )

                analysis_status = gr.Textbox(label="Status Analyse", interactive=False)

                gr.Markdown("### 3. Grille Patches (2×2)")

                gallery = gr.Gallery(
                    label="4 Patches 224×224 — Cliquer pour sélectionner",
                    columns=2,
                    rows=2,
                    height=280,
                    object_fit="contain",
                    allow_preview=False,
                )

            # === COLONNE DROITE: Visualisation ===
            with gr.Column(scale=2):
                gr.Markdown("### 4. Patch Sélectionné")

                with gr.Row():
                    selected_image = gr.Image(
                        label="Patch Original (224×224)",
                        height=280,
                        width=280,
                    )
                    overlay_image = gr.Image(
                        label="Segmentation",
                        height=280,
                        width=280,
                    )

                patch_metrics = gr.Markdown(
                    value="*Sélectionnez un patch dans la grille*",
                )

                gr.Markdown("---")
                gr.Markdown("### 5. Métriques WSI Globales")

                wsi_metrics = gr.Markdown(
                    value="*Uploader une image pour commencer*",
                )

        # === ÉVÉNEMENTS ===

        # Chargement modèle
        load_btn.click(
            fn=load_engine_for_grid,
            inputs=[organ_dropdown],
            outputs=[model_status],
        )

        # Upload image → Extraction + Analyse automatique
        input_image.upload(
            fn=process_uploaded_image,
            inputs=[input_image],
            outputs=[gallery, selected_image, overlay_image, patch_metrics, wsi_metrics, analysis_status],
        )

        # Clic sur patch dans galerie
        gallery.select(
            fn=on_patch_select,
            outputs=[selected_image, overlay_image, patch_metrics],
        )

    return app


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CellViT-Optimus Simulation WSI")
    parser.add_argument("--organ", type=str, default=None,
                        help="Organe à précharger (ex: Lung, Breast)")
    parser.add_argument("--port", type=int, default=7861,
                        help="Port Gradio (défaut: 7861)")
    parser.add_argument("--share", action="store_true",
                        help="Créer un lien public Gradio")
    parser.add_argument("--preload", action="store_true",
                        help="Précharger le backbone au démarrage")
    args = parser.parse_args()

    # Préchargement optionnel
    if args.preload:
        logger.info("Préchargement du backbone...")
        preload_backbone_core(device="cuda")

    # Chargement organe si spécifié
    if args.organ:
        logger.info(f"Chargement du modèle pour {args.organ}...")
        load_engine_core(args.organ, device="cuda")

    # Lancer l'interface
    app = create_grid_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
