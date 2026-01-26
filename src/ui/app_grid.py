#!/usr/bin/env python3
"""
CellViT-Optimus — Navigation Grille Multi-Patches.

Interface de simulation WSI utilisant des patches PanNuke (256×256 ou 224×224)
pour préparer la navigation sur lames entières.

Fonctionnalités:
- Chargement multiple d'images (dossier ou fichiers)
- Vue grille avec miniatures
- Clic pour sélectionner et analyser un patch
- Métriques agrégées multi-patches

Usage:
    python -m src.ui.app_grid --organ Lung
    # ou
    python src/ui/app_grid.py --organ Breast --port 7861

Note: Préparation pour la navigation WSI réelle.
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
from src.ui.organ_config import get_model_for_organ

# Imports: Visualisations
from src.ui.visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    create_type_distribution_chart,
)

# ==============================================================================
# ÉTAT GRILLE (SIMULATION WSI)
# ==============================================================================

@dataclass
class PatchInfo:
    """Information sur un patch chargé."""
    index: int
    filename: str
    image: np.ndarray  # Image originale (256×256 ou 224×224)
    preprocessed: Optional[np.ndarray] = None  # Image après preprocessing (224×224)
    result: Optional[AnalysisResult] = None
    overlay: Optional[np.ndarray] = None
    is_analyzed: bool = False


@dataclass
class GridState:
    """État global de la grille multi-patches."""
    patches: List[PatchInfo] = field(default_factory=list)
    selected_index: int = -1
    page: int = 0
    patches_per_page: int = 16  # Grille 4×4

    def clear(self):
        """Réinitialise l'état."""
        self.patches = []
        self.selected_index = -1
        self.page = 0

    def total_pages(self) -> int:
        """Nombre total de pages."""
        if not self.patches:
            return 1
        return (len(self.patches) - 1) // self.patches_per_page + 1

    def current_page_patches(self) -> List[PatchInfo]:
        """Retourne les patches de la page courante."""
        start = self.page * self.patches_per_page
        end = start + self.patches_per_page
        return self.patches[start:end]

    def get_selected(self) -> Optional[PatchInfo]:
        """Retourne le patch sélectionné."""
        if 0 <= self.selected_index < len(self.patches):
            return self.patches[self.selected_index]
        return None

    def analyzed_count(self) -> int:
        """Nombre de patches analysés."""
        return sum(1 for p in self.patches if p.is_analyzed)

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques agrégées sur tous les patches analysés."""
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
                total_nuclei += r.morphometry.total_nuclei
                for cell_type, count in r.morphometry.type_counts.items():
                    type_counts[cell_type] = type_counts.get(cell_type, 0) + count
                total_area += r.morphometry.total_area_um2

        return {
            "total_patches": len(self.patches),
            "analyzed": len(analyzed),
            "total_nuclei": total_nuclei,
            "type_counts": type_counts,
            "total_area_um2": total_area,
            "avg_nuclei_per_patch": total_nuclei / len(analyzed) if analyzed else 0,
        }


# Instance globale
grid_state = GridState()


# ==============================================================================
# FONCTIONS DE CHARGEMENT
# ==============================================================================

def load_images_from_folder(folder_path: str) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """
    Charge les images PNG/JPG d'un dossier.

    Returns:
        (list of (thumbnail, label), status_message)
    """
    grid_state.clear()

    if not folder_path:
        return [], "❌ Aucun dossier spécifié"

    folder = Path(folder_path)
    if not folder.exists():
        return [], f"❌ Dossier non trouvé: {folder_path}"

    # Extensions supportées
    extensions = {".png", ".jpg", ".jpeg"}
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        return [], f"❌ Aucune image trouvée dans {folder_path}"

    # Charger les images
    gallery_items = []
    for i, img_path in enumerate(sorted(image_files)):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Créer PatchInfo
        patch = PatchInfo(
            index=i,
            filename=img_path.name,
            image=img_rgb,
        )
        grid_state.patches.append(patch)

        # Créer miniature pour galerie (64×64)
        thumb = cv2.resize(img_rgb, (64, 64))
        gallery_items.append((thumb, img_path.stem))

    n_loaded = len(grid_state.patches)
    return gallery_items, f"✅ {n_loaded} images chargées depuis {folder.name}/"


def load_images_from_files(files: List[Any]) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """
    Charge les images depuis une liste de fichiers uploadés.

    Returns:
        (list of (thumbnail, label), status_message)
    """
    grid_state.clear()

    if not files:
        return [], "❌ Aucun fichier sélectionné"

    gallery_items = []
    for i, file in enumerate(files):
        # Gradio file upload: file.name contient le chemin temporaire
        if hasattr(file, 'name'):
            img_path = file.name
        else:
            img_path = str(file)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Créer PatchInfo
        filename = Path(img_path).name
        patch = PatchInfo(
            index=i,
            filename=filename,
            image=img_rgb,
        )
        grid_state.patches.append(patch)

        # Créer miniature pour galerie
        thumb = cv2.resize(img_rgb, (64, 64))
        gallery_items.append((thumb, filename[:20]))  # Tronquer nom long

    n_loaded = len(grid_state.patches)
    return gallery_items, f"✅ {n_loaded} images chargées"


# ==============================================================================
# FONCTIONS D'ANALYSE
# ==============================================================================

def analyze_selected_patch() -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Analyse le patch sélectionné.

    Returns:
        (image_preprocessed, overlay, metrics_text, status)
    """
    empty = np.zeros((224, 224, 3), dtype=np.uint8)

    patch = grid_state.get_selected()
    if patch is None:
        return empty, empty, "", "❌ Aucun patch sélectionné"

    if state.engine is None:
        return empty, empty, "", "❌ Moteur non chargé (sélectionner un organe)"

    # Analyse via core
    result, preprocessed, error = run_analysis_core(patch.image, use_auto_params=True)

    if error:
        return empty, empty, "", f"❌ {error}"

    # Stocker résultats
    patch.result = result
    patch.preprocessed = preprocessed
    patch.is_analyzed = True

    # Créer overlay
    overlay = create_segmentation_overlay(
        result.image_rgb,
        result.instance_map,
        result.type_map,
        alpha=0.4,
    )
    patch.overlay = overlay

    # Formater métriques
    metrics = format_patch_metrics(patch)

    return preprocessed, overlay, metrics, f"✅ Patch analysé: {patch.filename}"


def analyze_all_patches() -> Tuple[str, str]:
    """
    Analyse tous les patches non analysés.

    Returns:
        (aggregated_metrics, status)
    """
    if state.engine is None:
        return "", "❌ Moteur non chargé"

    if not grid_state.patches:
        return "", "❌ Aucun patch chargé"

    # Analyser les patches non analysés
    n_analyzed = 0
    n_errors = 0

    for patch in grid_state.patches:
        if patch.is_analyzed:
            continue

        result, preprocessed, error = run_analysis_core(patch.image, use_auto_params=True)

        if error:
            n_errors += 1
            continue

        patch.result = result
        patch.preprocessed = preprocessed
        patch.is_analyzed = True

        # Créer overlay
        overlay = create_segmentation_overlay(
            result.image_rgb,
            result.instance_map,
            result.type_map,
            alpha=0.4,
        )
        patch.overlay = overlay
        n_analyzed += 1

    # Métriques agrégées
    agg_metrics = format_aggregated_metrics()

    status = f"✅ {n_analyzed} patches analysés"
    if n_errors > 0:
        status += f" ({n_errors} erreurs)"

    return agg_metrics, status


def on_gallery_select(evt: gr.SelectData) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Gère la sélection d'un patch dans la galerie.

    Returns:
        (selected_image, overlay_or_empty, patch_info, status)
    """
    empty = np.zeros((224, 224, 3), dtype=np.uint8)

    index = evt.index
    if index < 0 or index >= len(grid_state.patches):
        return empty, empty, "", "❌ Index invalide"

    grid_state.selected_index = index
    patch = grid_state.patches[index]

    # Si déjà analysé, retourner les résultats
    if patch.is_analyzed and patch.preprocessed is not None:
        metrics = format_patch_metrics(patch)
        return patch.preprocessed, patch.overlay, metrics, f"✅ {patch.filename} (analysé)"

    # Sinon retourner l'image originale
    # Resize pour affichage si nécessaire
    h, w = patch.image.shape[:2]
    if h != 224 or w != 224:
        display_img = cv2.resize(patch.image, (224, 224))
    else:
        display_img = patch.image

    return display_img, empty, f"**{patch.filename}**\nTaille: {w}×{h}\n\n*Non analysé*", f"Sélectionné: {patch.filename}"


# ==============================================================================
# FORMATAGE
# ==============================================================================

def format_patch_metrics(patch: PatchInfo) -> str:
    """Formate les métriques d'un patch."""
    if not patch.is_analyzed or patch.result is None:
        return f"**{patch.filename}**\n\n*Non analysé*"

    r = patch.result
    lines = [
        f"### {patch.filename}",
        "",
    ]

    if r.morphometry:
        m = r.morphometry
        lines.extend([
            f"**Noyaux détectés:** {m.total_nuclei}",
            f"**Densité:** {m.density_per_mm2:.0f} /mm²",
            "",
            "**Distribution:**",
        ])
        for cell_type, count in sorted(m.type_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / m.total_nuclei if m.total_nuclei > 0 else 0
            lines.append(f"- {cell_type}: {count} ({pct:.1f}%)")

    return "\n".join(lines)


def format_aggregated_metrics() -> str:
    """Formate les métriques agrégées."""
    agg = grid_state.get_aggregated_metrics()

    if agg["analyzed"] == 0:
        return f"**Patches chargés:** {agg['total_patches']}\n\n*Aucun patch analysé*"

    lines = [
        "## Métriques Agrégées (Simulation WSI)",
        "",
        f"**Patches:** {agg['analyzed']} / {agg['total_patches']} analysés",
        f"**Noyaux totaux:** {agg['total_nuclei']}",
        f"**Moyenne par patch:** {agg['avg_nuclei_per_patch']:.1f}",
        f"**Surface totale:** {agg['total_area_um2']:.0f} µm²",
        "",
        "### Distribution globale",
    ]

    type_counts = agg.get("type_counts", {})
    total = sum(type_counts.values()) if type_counts else 0

    for cell_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"- **{cell_type}:** {count} ({pct:.1f}%)")

    return "\n".join(lines)


# ==============================================================================
# NAVIGATION PAGES
# ==============================================================================

def prev_page() -> List[Tuple[np.ndarray, str]]:
    """Page précédente."""
    if grid_state.page > 0:
        grid_state.page -= 1
    return get_gallery_items()


def next_page() -> List[Tuple[np.ndarray, str]]:
    """Page suivante."""
    if grid_state.page < grid_state.total_pages() - 1:
        grid_state.page += 1
    return get_gallery_items()


def get_gallery_items() -> List[Tuple[np.ndarray, str]]:
    """Retourne les miniatures de la page courante."""
    patches = grid_state.current_page_patches()
    items = []
    for p in patches:
        thumb = cv2.resize(p.image, (64, 64))
        # Ajouter indicateur si analysé
        label = p.filename[:15]
        if p.is_analyzed:
            label = f"✅ {label}"
        items.append((thumb, label))
    return items


def get_page_info() -> str:
    """Retourne info de pagination."""
    return f"Page {grid_state.page + 1} / {grid_state.total_pages()}"


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
    """Crée l'interface de navigation grille."""

    with gr.Blocks(
        title="CellViT-Optimus — Navigation Grille",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown("""
        # 🔬 CellViT-Optimus — Navigation Grille Multi-Patches

        **Simulation WSI** — Testez la navigation sur plusieurs patches avant le déploiement sur lames entières.

        1. **Sélectionner un organe** pour charger le modèle
        2. **Charger des images** (dossier ou fichiers)
        3. **Cliquer sur un patch** pour l'analyser
        4. Ou **Analyser tout** pour traitement batch
        """)

        with gr.Row():
            # === COLONNE GAUCHE: Contrôles et Grille ===
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                organ_dropdown = gr.Dropdown(
                    choices=ORGAN_CHOICES,
                    value="Lung",
                    label="Organe",
                )
                load_btn = gr.Button("🚀 Charger Modèle", variant="primary")
                model_status = gr.Textbox(label="Status Modèle", interactive=False)

                gr.Markdown("### Chargement Images")

                folder_input = gr.Textbox(
                    label="Chemin dossier",
                    placeholder="data/samples/",
                )
                load_folder_btn = gr.Button("📁 Charger Dossier")

                file_input = gr.File(
                    label="Ou uploader des fichiers",
                    file_count="multiple",
                    file_types=["image"],
                )

                load_status = gr.Textbox(label="Status Chargement", interactive=False)

                gr.Markdown("### Grille Patches")

                gallery = gr.Gallery(
                    label="Patches",
                    columns=4,
                    rows=4,
                    height=400,
                    object_fit="contain",
                    allow_preview=False,
                )

                with gr.Row():
                    prev_btn = gr.Button("◀ Précédent")
                    page_info = gr.Textbox(value="Page 1 / 1", interactive=False, scale=2)
                    next_btn = gr.Button("Suivant ▶")

                analyze_all_btn = gr.Button("⚡ Analyser Tous", variant="secondary")

            # === COLONNE DROITE: Visualisation ===
            with gr.Column(scale=2):
                gr.Markdown("### Patch Sélectionné")

                with gr.Row():
                    selected_image = gr.Image(
                        label="Image (224×224)",
                        height=300,
                        width=300,
                    )
                    overlay_image = gr.Image(
                        label="Segmentation",
                        height=300,
                        width=300,
                    )

                with gr.Row():
                    analyze_btn = gr.Button("🔬 Analyser ce Patch", variant="primary")
                    analysis_status = gr.Textbox(label="Status", interactive=False)

                patch_metrics = gr.Markdown(
                    value="*Sélectionnez un patch dans la grille*",
                    label="Métriques Patch",
                )

                gr.Markdown("---")
                gr.Markdown("### Métriques Agrégées (Multi-Patches)")

                aggregated_metrics = gr.Markdown(
                    value="*Aucun patch analysé*",
                )

        # === ÉVÉNEMENTS ===

        # Chargement modèle
        load_btn.click(
            fn=load_engine_for_grid,
            inputs=[organ_dropdown],
            outputs=[model_status],
        )

        # Chargement images depuis dossier
        load_folder_btn.click(
            fn=load_images_from_folder,
            inputs=[folder_input],
            outputs=[gallery, load_status],
        )

        # Chargement images depuis fichiers uploadés
        file_input.change(
            fn=load_images_from_files,
            inputs=[file_input],
            outputs=[gallery, load_status],
        )

        # Sélection dans galerie
        gallery.select(
            fn=on_gallery_select,
            outputs=[selected_image, overlay_image, patch_metrics, analysis_status],
        )

        # Analyse patch sélectionné
        analyze_btn.click(
            fn=analyze_selected_patch,
            outputs=[selected_image, overlay_image, patch_metrics, analysis_status],
        )

        # Analyse tous les patches
        analyze_all_btn.click(
            fn=analyze_all_patches,
            outputs=[aggregated_metrics, analysis_status],
        ).then(
            fn=get_gallery_items,
            outputs=[gallery],
        )

        # Navigation pages
        prev_btn.click(fn=prev_page, outputs=[gallery]).then(
            fn=get_page_info, outputs=[page_info]
        )
        next_btn.click(fn=next_page, outputs=[gallery]).then(
            fn=get_page_info, outputs=[page_info]
        )

    return app


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CellViT-Optimus Navigation Grille")
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
