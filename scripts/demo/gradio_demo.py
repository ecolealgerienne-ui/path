#!/usr/bin/env python3
"""
Interface Gradio pour la démonstration CellViT-Optimus.

Permet de visualiser interactivement les segmentations cellulaires
et d'explorer les différents types de tissus.
"""

import gradio as gr
import numpy as np
from pathlib import Path
import cv2
import sys

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.demo.visualize_cells import (
    overlay_mask,
    draw_contours,
    count_cells,
    generate_report,
    CELL_COLORS,
    CELL_TYPE_INDICES
)
from scripts.demo.synthetic_cells import generate_synthetic_tissue, TISSUE_CONFIGS


class CellVitDemo:
    """Interface de démonstration CellViT-Optimus."""

    def __init__(self, data_dir: str = "data/demo"):
        self.data_dir = Path(data_dir)
        self.images = None
        self.masks = None
        self.types = None
        self.current_idx = 0

        self.load_data()

    def load_data(self):
        """Charge les données de démonstration."""
        if self.data_dir.exists():
            try:
                self.images = np.load(self.data_dir / "images.npy")
                self.masks = np.load(self.data_dir / "masks.npy")
                self.types = np.load(self.data_dir / "types.npy")
                print(f"✓ Chargé {len(self.images)} images depuis {self.data_dir}")
            except Exception as e:
                print(f"Erreur chargement: {e}")
                self.generate_default_data()
        else:
            print("Données non trouvées, génération...")
            self.generate_default_data()

    def generate_default_data(self):
        """Génère des données par défaut."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        images = []
        masks = []
        types = []

        for i, tissue in enumerate(list(TISSUE_CONFIGS.keys())[:8]):
            img, mask, _ = generate_synthetic_tissue(tissue, seed=42 + i)
            images.append(img)
            masks.append(mask)
            types.append(tissue)

        self.images = np.array(images)
        self.masks = np.array(masks)
        self.types = np.array(types)

        np.save(self.data_dir / "images.npy", self.images)
        np.save(self.data_dir / "masks.npy", self.masks)
        np.save(self.data_dir / "types.npy", self.types)

    def process_image(
        self,
        idx: int,
        show_overlay: bool,
        show_contours: bool,
        alpha: float
    ):
        """Traite une image du dataset."""
        if self.images is None or idx >= len(self.images):
            return None, "Erreur: données non disponibles"

        idx = int(idx)
        image = self.images[idx].copy()
        mask = self.masks[idx]
        tissue_type = str(self.types[idx])

        # Appliquer les visualisations
        if show_overlay and show_contours:
            result = overlay_mask(image, mask, alpha)
            result = draw_contours(result, mask, thickness=2)
        elif show_overlay:
            result = overlay_mask(image, mask, alpha)
        elif show_contours:
            result = draw_contours(image, mask, thickness=2)
        else:
            result = image

        # Générer le rapport
        report = generate_report(mask, tissue_type)

        return result, report

    def generate_new_tissue(self, tissue_type: str, n_cells: int, seed: int):
        """Génère un nouveau tissu à la demande."""
        image, mask, info = generate_synthetic_tissue(
            tissue_type=tissue_type,
            n_cells=int(n_cells),
            seed=int(seed)
        )

        report = generate_report(mask, tissue_type)

        # Créer la visualisation par défaut
        result = overlay_mask(image, mask, 0.4)
        result = draw_contours(result, mask, thickness=2)

        return image, result, report

    def get_cell_stats(self, idx: int):
        """Retourne les statistiques cellulaires."""
        if self.masks is None or idx >= len(self.masks):
            return {}

        counts = count_cells(self.masks[int(idx)])
        total = sum(counts.values())

        stats = []
        for cell_type, count in counts.items():
            pct = (count / total * 100) if total > 0 else 0
            stats.append(f"{cell_type}: {count} ({pct:.1f}%)")

        return "\n".join(stats)


def create_demo_interface():
    """Crée l'interface Gradio complète."""
    demo = CellVitDemo()

    with gr.Blocks(title="CellViT-Optimus Demo") as interface:

        gr.Markdown("""
        # 🔬 CellViT-Optimus — Démonstration

        **Système d'assistance au triage histopathologique**

        Ce démo permet de visualiser la segmentation et classification des cellules
        dans des images de tissus histopathologiques.

        ---
        """)

        with gr.Tabs():
            # Tab 1: Explorer le dataset
            with gr.TabItem("📊 Explorer le Dataset"):
                with gr.Row():
                    with gr.Column(scale=1):
                        idx_slider = gr.Slider(
                            minimum=0,
                            maximum=len(demo.images) - 1 if demo.images is not None else 0,
                            step=1,
                            value=0,
                            label="Image #"
                        )

                        show_overlay = gr.Checkbox(
                            value=True,
                            label="Afficher l'overlay coloré"
                        )
                        show_contours = gr.Checkbox(
                            value=True,
                            label="Afficher les contours"
                        )
                        alpha_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.4,
                            label="Transparence overlay"
                        )

                        process_btn = gr.Button(
                            "🔄 Mettre à jour",
                            variant="primary"
                        )

                    with gr.Column(scale=2):
                        output_image = gr.Image(
                            label="Visualisation",
                            type="numpy"
                        )

                with gr.Row():
                    report_output = gr.Textbox(
                        label="Rapport d'analyse",
                        lines=15,
                        max_lines=20
                    )

                process_btn.click(
                    fn=demo.process_image,
                    inputs=[idx_slider, show_overlay, show_contours, alpha_slider],
                    outputs=[output_image, report_output]
                )

                # Auto-update on slider change
                idx_slider.change(
                    fn=demo.process_image,
                    inputs=[idx_slider, show_overlay, show_contours, alpha_slider],
                    outputs=[output_image, report_output]
                )

            # Tab 2: Générer de nouveaux tissus
            with gr.TabItem("🧬 Générateur de Tissus"):
                gr.Markdown("""
                ### Génération de tissus synthétiques

                Créez des images de tissus simulées avec différentes
                compositions cellulaires pour tester le système.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        tissue_dropdown = gr.Dropdown(
                            choices=list(TISSUE_CONFIGS.keys()),
                            value="Breast",
                            label="Type de tissu"
                        )
                        n_cells_slider = gr.Slider(
                            minimum=20,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Nombre de cellules"
                        )
                        seed_slider = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=42,
                            step=1,
                            label="Seed (reproductibilité)"
                        )

                        generate_btn = gr.Button(
                            "🧬 Générer",
                            variant="primary"
                        )

                    with gr.Column(scale=2):
                        with gr.Row():
                            gen_original = gr.Image(
                                label="Image originale",
                                type="numpy"
                            )
                            gen_segmented = gr.Image(
                                label="Segmentation",
                                type="numpy"
                            )

                gen_report = gr.Textbox(
                    label="Analyse",
                    lines=10
                )

                generate_btn.click(
                    fn=demo.generate_new_tissue,
                    inputs=[tissue_dropdown, n_cells_slider, seed_slider],
                    outputs=[gen_original, gen_segmented, gen_report]
                )

            # Tab 3: À propos
            with gr.TabItem("ℹ️ À propos"):
                gr.Markdown("""
                ## CellViT-Optimus

                ### Architecture

                ```
                ┌─────────────────────────────────────────┐
                │           Image H&E (WSI)               │
                └─────────────────────────────────────────┘
                                    │
                                    ▼
                ┌─────────────────────────────────────────┐
                │      H-OPTIMUS-0 (Backbone gelé)        │
                │      ViT-Giant/14, 1.1B params          │
                │      → Embeddings 1536-dim              │
                └─────────────────────────────────────────┘
                                    │
                                    ▼
                ┌─────────────────────────────────────────┐
                │         Décodeur UNETR                  │
                │      Segmentation cellulaire            │
                └─────────────────────────────────────────┘
                                    │
                                    ▼
                ┌─────────────────────────────────────────┐
                │       Classification (5 types)          │
                │  • Neoplastic (tumeur)                  │
                │  • Inflammatory                         │
                │  • Connective                           │
                │  • Dead (nécrose)                       │
                │  • Epithelial                           │
                └─────────────────────────────────────────┘
                ```

                ### Types de Cellules

                | Type | Couleur | Description |
                |------|---------|-------------|
                | Neoplastic | 🔴 Rouge | Cellules tumorales |
                | Inflammatory | 🟢 Vert | Cellules immunitaires |
                | Connective | 🔵 Bleu | Tissu de soutien |
                | Dead | 🟡 Jaune | Cellules nécrotiques |
                | Epithelial | 🔷 Cyan | Cellules épithéliales |

                ### Références

                - **H-optimus-0**: [Bioptimus](https://huggingface.co/bioptimus/H-optimus-0)
                - **CellViT**: [TIO-IKIM](https://github.com/TIO-IKIM/CellViT)
                - **PanNuke**: [Warwick TIA](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/)

                ---

                *Ce démo utilise des données synthétiques pour illustration.*
                *En production, le système utilise de vraies images histopathologiques.*
                """)

        # Charger la première image au démarrage
        interface.load(
            fn=demo.process_image,
            inputs=[
                gr.Number(value=0, visible=False),
                gr.Checkbox(value=True, visible=False),
                gr.Checkbox(value=True, visible=False),
                gr.Number(value=0.4, visible=False)
            ],
            outputs=[output_image, report_output]
        )

    return interface


if __name__ == "__main__":
    print("🚀 Lancement de CellViT-Optimus Demo...")
    print("   URL: http://localhost:7860")
    print("   Ctrl+C pour arrêter\n")

    interface = create_demo_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
