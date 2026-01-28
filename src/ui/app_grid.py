#!/usr/bin/env python3
"""
CellViT-Optimus — Navigation Grille Multi-Patches (Simulation WSI).

Interface de simulation WSI utilisant des images PanNuke (256×256).
Extraction automatique de 4 patches 224×224 (grille 2×2 avec chevauchement).

**STITCHING WSI STANDARD:**
- Chaque patch a une zone valide (sans chevauchement)
- Seuls les noyaux dont le centroïde est dans la zone valide sont comptés
- Reconstruction de la segmentation 256×256 sans doublons

Workflow:
1. Upload d'une image 256×256
2. Extraction automatique → 4 patches 224×224
3. Analyse automatique des 4 patches
4. Stitching: filtrage par zone valide + reconstruction
5. Affichage: grille cliquable + vue stitchée WSI

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
from scipy import ndimage
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
# Format: (x_offset, y_offset) pour image[y:y+224, x:x+224]
PATCH_POSITIONS = [
    (0, 0),           # Patch 0: Top-Left
    (OFFSET, 0),      # Patch 1: Top-Right  (x=32)
    (0, OFFSET),      # Patch 2: Bottom-Left (y=32)
    (OFFSET, OFFSET), # Patch 3: Bottom-Right (x=32, y=32)
]

PATCH_NAMES = ["Haut-Gauche", "Haut-Droite", "Bas-Gauche", "Bas-Droite"]

# ==============================================================================
# ZONES VALIDES (Stitching WSI Standard)
# ==============================================================================
# Diviser l'image 256×256 en 4 quadrants sans chevauchement.
# Chaque patch ne "possède" que les noyaux dans son quadrant assigné.
#
# Image 256×256:
# ┌─────────────┬─────────────┐
# │  Q0 (0,0)   │  Q1 (128,0) │
# │  0:128      │  128:256    │
# ├─────────────┼─────────────┤
# │  Q2 (0,128) │  Q3 (128,128)│
# │  0:128      │  128:256    │
# └─────────────┴─────────────┘

# Zones valides en coordonnées IMAGE (y_min, y_max, x_min, x_max)
VALID_ZONES_IMAGE = [
    (0, 128, 0, 128),      # Patch 0 → Quadrant haut-gauche
    (0, 128, 128, 256),    # Patch 1 → Quadrant haut-droit
    (128, 256, 0, 128),    # Patch 2 → Quadrant bas-gauche
    (128, 256, 128, 256),  # Patch 3 → Quadrant bas-droit
]

# Zones valides en coordonnées PATCH LOCAL (y_min, y_max, x_min, x_max)
# Calculé: zone_image - patch_offset
VALID_ZONES_PATCH = [
    (0, 128, 0, 128),      # Patch 0: offset (0,0) → [0:128, 0:128]
    (0, 128, 96, 224),     # Patch 1: offset (32,0) → [0:128, 128-32:256-32] = [0:128, 96:224]
    (96, 224, 0, 128),     # Patch 2: offset (0,32) → [128-32:256-32, 0:128] = [96:224, 0:128]
    (96, 224, 96, 224),    # Patch 3: offset (32,32) → [96:224, 96:224]
]


# ==============================================================================
# ÉTAT GRILLE (SIMULATION WSI)
# ==============================================================================

@dataclass
class NucleusInfo:
    """Information sur un noyau individuel (pour stitching)."""
    id: int  # ID dans le patch
    centroid_patch: Tuple[int, int]  # (y, x) en coordonnées patch
    centroid_image: Tuple[int, int]  # (y, x) en coordonnées image 256×256
    cell_type: int  # Type from HoVer-Net
    area_pixels: int = 0
    in_valid_zone: bool = False


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
    # Stitching
    nuclei: List[NucleusInfo] = field(default_factory=list)
    valid_nuclei_count: int = 0


@dataclass
class WSIState:
    """État de la simulation WSI (une image source)."""
    source_image: Optional[np.ndarray] = None  # Image originale 256×256
    source_filename: str = ""
    patches: List[PatchInfo] = field(default_factory=list)
    selected_index: int = 0
    # Stitching results
    stitched_instance_map: Optional[np.ndarray] = None  # 256×256
    stitched_type_map: Optional[np.ndarray] = None  # 256×256
    stitched_overlay: Optional[np.ndarray] = None  # 256×256 RGB

    def clear(self):
        """Réinitialise l'état."""
        self.source_image = None
        self.source_filename = ""
        self.patches = []
        self.selected_index = 0
        self.stitched_instance_map = None
        self.stitched_type_map = None
        self.stitched_overlay = None

    def get_selected(self) -> Optional[PatchInfo]:
        """Retourne le patch sélectionné."""
        if 0 <= self.selected_index < len(self.patches):
            return self.patches[self.selected_index]
        return None

    def all_analyzed(self) -> bool:
        """Vérifie si tous les patches sont analysés."""
        return all(p.is_analyzed for p in self.patches)

    def get_aggregated_metrics_stitched(self) -> Dict[str, Any]:
        """Calcule les métriques agrégées sur les noyaux VALIDES uniquement (sans doublons)."""
        if not self.all_analyzed():
            return {"total_patches": len(self.patches), "analyzed": 0}

        # Compter uniquement les noyaux dans les zones valides
        total_nuclei = 0
        type_counts = {}

        for p in self.patches:
            total_nuclei += p.valid_nuclei_count
            for nucleus in p.nuclei:
                if nucleus.in_valid_zone:
                    cell_type = nucleus.cell_type
                    type_counts[cell_type] = type_counts.get(cell_type, 0) + 1

        # Surface = 256×256 pixels (pas de chevauchement dans les métriques)
        # À 0.5 MPP: 256 × 0.5 = 128 µm → surface = 128² = 16384 µm² = 0.016384 mm²
        total_area_mm2 = (PANNUKE_SIZE * 0.5) ** 2 / 1_000_000  # µm² → mm²

        return {
            "total_patches": len(self.patches),
            "analyzed": len([p for p in self.patches if p.is_analyzed]),
            "total_nuclei": total_nuclei,
            "type_counts": type_counts,
            "total_area_mm2": total_area_mm2,
            "density_per_mm2": total_nuclei / total_area_mm2 if total_area_mm2 > 0 else 0,
        }


# Instance globale
wsi_state = WSIState()


# ==============================================================================
# EXTRACTION DE PATCHES
# ==============================================================================

def extract_patches_2x2(image: np.ndarray) -> List[np.ndarray]:
    """
    Extrait 4 patches 224×224 d'une image 256×256 (grille 2×2 avec chevauchement).
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
# STITCHING WSI
# ==============================================================================

def extract_nuclei_info(
    instance_map: np.ndarray,
    type_map: np.ndarray,
    patch_index: int,
) -> List[NucleusInfo]:
    """
    Extrait les informations de chaque noyau d'un patch.
    Détermine si le centroïde est dans la zone valide.
    """
    nuclei = []
    patch_offset_x, patch_offset_y = PATCH_POSITIONS[patch_index]
    valid_zone = VALID_ZONES_PATCH[patch_index]  # (y_min, y_max, x_min, x_max)

    # Trouver tous les IDs de noyaux
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]  # Exclure le fond (0)

    for nucleus_id in unique_ids:
        mask = instance_map == nucleus_id

        # Centroïde en coordonnées patch
        coords = np.where(mask)
        if len(coords[0]) == 0:
            continue

        cy_patch = int(np.mean(coords[0]))
        cx_patch = int(np.mean(coords[1]))

        # Centroïde en coordonnées image
        cy_image = cy_patch + patch_offset_y
        cx_image = cx_patch + patch_offset_x

        # Type cellulaire (valeur majoritaire dans le masque)
        cell_type = int(np.median(type_map[mask]))

        # Aire
        area = int(np.sum(mask))

        # Vérifier si dans zone valide (coordonnées patch)
        y_min, y_max, x_min, x_max = valid_zone
        in_valid = (y_min <= cy_patch < y_max) and (x_min <= cx_patch < x_max)

        nuclei.append(NucleusInfo(
            id=int(nucleus_id),
            centroid_patch=(cy_patch, cx_patch),
            centroid_image=(cy_image, cx_image),
            cell_type=cell_type,
            area_pixels=area,
            in_valid_zone=in_valid,
        ))

    return nuclei


def stitch_segmentation_maps() -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruit les cartes de segmentation 256×256 à partir des 4 patches.
    Utilise uniquement les noyaux dans les zones valides.

    Returns:
        (instance_map_256, type_map_256)
    """
    instance_map = np.zeros((PANNUKE_SIZE, PANNUKE_SIZE), dtype=np.int32)
    type_map = np.zeros((PANNUKE_SIZE, PANNUKE_SIZE), dtype=np.int32)

    global_id = 1  # ID global pour les noyaux stitchés

    for patch in wsi_state.patches:
        if not patch.is_analyzed or patch.result is None:
            continue

        patch_offset_x, patch_offset_y = patch.position
        patch_inst = patch.result.instance_map

        # Pour chaque noyau valide
        for nucleus in patch.nuclei:
            if not nucleus.in_valid_zone:
                continue

            # Masque du noyau dans le patch
            mask_patch = patch_inst == nucleus.id

            # Coordonnées dans le patch
            coords = np.where(mask_patch)
            if len(coords[0]) == 0:
                continue

            # Type HoVer-Net
            nucleus_type = nucleus.cell_type

            # Transférer vers l'image 256×256
            for py, px in zip(coords[0], coords[1]):
                iy = py + patch_offset_y
                ix = px + patch_offset_x

                if 0 <= iy < PANNUKE_SIZE and 0 <= ix < PANNUKE_SIZE:
                    instance_map[iy, ix] = global_id
                    type_map[iy, ix] = nucleus_type

            global_id += 1

    return instance_map, type_map


def create_stitched_overlay(
    source_image: np.ndarray,
    instance_map: np.ndarray,
    type_map: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Crée l'overlay de segmentation pour l'image stitchée 256×256."""
    return create_segmentation_overlay(source_image, instance_map, type_map, alpha)


# ==============================================================================
# FONCTIONS D'ANALYSE
# ==============================================================================

def process_uploaded_image(image: np.ndarray) -> Tuple[
    List[Tuple[np.ndarray, str]],  # gallery items
    np.ndarray,  # selected patch
    np.ndarray,  # patch overlay
    str,  # patch metrics
    np.ndarray,  # stitched overlay (WSI)
    str,  # wsi metrics
    str,  # status
]:
    """
    Traite une image uploadée: extraction + analyse + stitching.

    Returns:
        (gallery, selected_patch, patch_overlay, patch_metrics, stitched_overlay, wsi_metrics, status)
    """
    empty_patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    empty_wsi = np.zeros((PANNUKE_SIZE, PANNUKE_SIZE, 3), dtype=np.uint8)

    # Vérifier que le moteur est chargé
    if state.engine is None:
        return [], empty_patch, empty_patch, "", empty_wsi, "", "❌ Moteur non chargé — Sélectionner un organe d'abord"

    # Vérifier la taille de l'image
    if image is None:
        return [], empty_patch, empty_patch, "", empty_wsi, "", "❌ Aucune image"

    h, w = image.shape[:2]
    if h != PANNUKE_SIZE or w != PANNUKE_SIZE:
        return [], empty_patch, empty_patch, "", empty_wsi, "", f"❌ Taille invalide: {w}×{h} (attendu: {PANNUKE_SIZE}×{PANNUKE_SIZE})"

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
        result, preprocessed, error = run_analysis_core(patch.image, use_auto_params=True)

        if error:
            logger.warning(f"Erreur patch {patch.name}: {error}")
            continue

        patch.result = result
        patch.is_analyzed = True

        # Créer overlay du patch
        overlay = create_segmentation_overlay(
            result.image_rgb,
            result.instance_map,
            result.type_map,
            alpha=0.4,
        )
        patch.overlay = overlay

        # Extraire info noyaux pour stitching
        patch.nuclei = extract_nuclei_info(
            result.instance_map,
            result.type_map,
            patch.index,
        )
        patch.valid_nuclei_count = sum(1 for n in patch.nuclei if n.in_valid_zone)

        logger.info(f"  {patch.name}: {len(patch.nuclei)} noyaux, {patch.valid_nuclei_count} dans zone valide")
        n_success += 1

    # === STITCHING ===
    logger.info("Stitching des segmentations...")
    stitched_inst, stitched_type = stitch_segmentation_maps()
    wsi_state.stitched_instance_map = stitched_inst
    wsi_state.stitched_type_map = stitched_type

    # Overlay stitché
    wsi_state.stitched_overlay = create_stitched_overlay(
        image, stitched_inst, stitched_type, alpha=0.4
    )

    total_stitched = len(np.unique(stitched_inst)) - 1  # -1 pour le fond
    logger.info(f"Stitching terminé: {total_stitched} noyaux uniques")

    # Construire la galerie
    gallery_items = []
    for p in wsi_state.patches:
        thumb = cv2.resize(p.image, (112, 112))
        label = f"{p.name}"
        if p.is_analyzed:
            label = f"✅ {p.valid_nuclei_count}n"
        gallery_items.append((thumb, label))

    # Sélectionner le premier patch
    wsi_state.selected_index = 0
    selected = wsi_state.get_selected()

    # Métriques
    patch_md = format_patch_metrics(selected) if selected else ""
    wsi_md = format_wsi_metrics_stitched()

    status = f"✅ {n_success}/4 patches | {total_stitched} noyaux"

    return (
        gallery_items,
        selected.image if selected else empty_patch,
        selected.overlay if selected and selected.overlay is not None else empty_patch,
        patch_md,
        wsi_state.stitched_overlay if wsi_state.stitched_overlay is not None else empty_wsi,
        wsi_md,
        status,
    )


def on_patch_select(evt: gr.SelectData) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Gère le clic sur un patch dans la galerie.
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

# Mapping type index → nom (PanNuke)
TYPE_NAMES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


def format_patch_metrics(patch: Optional[PatchInfo]) -> str:
    """Formate les métriques d'un patch."""
    if patch is None:
        return "*Aucun patch sélectionné*"

    if not patch.is_analyzed or patch.result is None:
        return f"### {patch.name}\n\n*Non analysé*"

    lines = [
        f"### Patch: {patch.name}",
        f"*Position: ({patch.position[0]}, {patch.position[1]})*",
        "",
        f"**Noyaux totaux:** {len(patch.nuclei)}",
        f"**Dans zone valide:** {patch.valid_nuclei_count} ✓",
        "",
    ]

    # Distribution par type (zone valide uniquement)
    type_counts = {}
    for n in patch.nuclei:
        if n.in_valid_zone:
            t = TYPE_NAMES.get(n.cell_type, f"Type{n.cell_type}")
            type_counts[t] = type_counts.get(t, 0) + 1

    if type_counts:
        lines.append("**Distribution (zone valide):**")
        total = sum(type_counts.values())
        for cell_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total if total > 0 else 0
            lines.append(f"- {cell_type}: {count} ({pct:.1f}%)")

    return "\n".join(lines)


def format_wsi_metrics_stitched() -> str:
    """Formate les métriques WSI stitchées (sans doublons)."""
    agg = wsi_state.get_aggregated_metrics_stitched()

    if agg["analyzed"] == 0:
        return "**Patches:** 0 analysés\n\n*En attente d'analyse...*"

    lines = [
        "## 🧩 Métriques WSI (Stitched)",
        "",
        f"**Patches:** {agg['analyzed']} / {agg['total_patches']}",
        f"**Surface:** {agg['total_area_mm2']*1e6:.0f} µm² ({agg['total_area_mm2']:.4f} mm²)",
        "",
        f"### Total Noyaux: {agg['total_nuclei']}",
        f"**Densité:** {agg['density_per_mm2']:.0f} /mm²",
        "",
        "### Distribution Globale",
    ]

    type_counts = agg.get("type_counts", {})
    total = sum(type_counts.values()) if type_counts else 0

    # Convertir indices en noms
    named_counts = {}
    for type_idx, count in type_counts.items():
        name = TYPE_NAMES.get(type_idx, f"Type{type_idx}")
        named_counts[name] = count

    for cell_type, count in sorted(named_counts.items(), key=lambda x: -x[1]):
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
    """Crée l'interface de navigation grille WSI avec stitching."""

    with gr.Blocks(
        title="CellViT-Optimus — Simulation WSI",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown("""
        # 🔬 CellViT-Optimus — Simulation WSI (Stitching)

        **Workflow automatique avec stitching industriel:**
        1. Sélectionner un **organe** → Charger le modèle
        2. **Uploader** une image PanNuke (256×256)
        3. Extraction de **4 patches** 224×224 (grille 2×2 avec overlap)
        4. Analyse + **Stitching** (zones valides, pas de doublon)
        5. Visualisation: **patches individuels** + **WSI reconstituée**
        """)

        with gr.Row():
            # === COLONNE GAUCHE: Config + Grille + WSI Stitched ===
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
                    height=180,
                )

                analysis_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### 3. Grille Patches (2×2)")

                gallery = gr.Gallery(
                    label="Cliquer pour sélectionner",
                    columns=2,
                    rows=2,
                    height=240,
                    object_fit="contain",
                    allow_preview=False,
                )

            # === COLONNE CENTRALE: Patch sélectionné ===
            with gr.Column(scale=1):
                gr.Markdown("### 4. Patch Sélectionné (224×224)")

                selected_image = gr.Image(
                    label="Original",
                    height=224,
                )
                patch_overlay = gr.Image(
                    label="Segmentation Patch",
                    height=224,
                )

                patch_metrics = gr.Markdown(
                    value="*Sélectionnez un patch*",
                )

            # === COLONNE DROITE: WSI Stitched ===
            with gr.Column(scale=1):
                gr.Markdown("### 5. WSI Reconstituée (256×256)")

                stitched_overlay = gr.Image(
                    label="Segmentation Stitchée (sans doublons)",
                    height=280,
                )

                wsi_metrics = gr.Markdown(
                    value="*Uploader une image*",
                )

        # === ÉVÉNEMENTS ===

        # Chargement modèle
        load_btn.click(
            fn=load_engine_for_grid,
            inputs=[organ_dropdown],
            outputs=[model_status],
        )

        # Upload image → Extraction + Analyse + Stitching
        input_image.upload(
            fn=process_uploaded_image,
            inputs=[input_image],
            outputs=[
                gallery,
                selected_image,
                patch_overlay,
                patch_metrics,
                stitched_overlay,
                wsi_metrics,
                analysis_status,
            ],
        )

        # Clic sur patch dans galerie
        gallery.select(
            fn=on_patch_select,
            outputs=[selected_image, patch_overlay, patch_metrics],
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
