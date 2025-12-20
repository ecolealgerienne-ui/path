#!/usr/bin/env python3
"""
Interface Gradio pour la démonstration CellViT-Optimus.

Permet de visualiser interactivement les segmentations cellulaires
et d'explorer les différents types de tissus.

Architecture cible: H-optimus-0 (backbone gelé) + HoVer-Net decoder
Basé sur la littérature: CellViT, HoVer-Net
"""

import gradio as gr
import numpy as np
from pathlib import Path
import cv2
import sys

# Ajouter le chemin du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo.visualize_cells import (
    overlay_mask,
    draw_contours,
    count_cells,
    generate_report,
    CELL_COLORS,
    CELL_TYPE_INDICES
)
from scripts.demo.synthetic_cells import generate_synthetic_tissue, TISSUE_CONFIGS

# Configuration des modèles
HOVERNET_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "hovernet_best.pth"
UNETR_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "unetr_best.pth"

# Tenter de charger les modèles (priorité: HoVer-Net > UNETR > simulation)
MODEL_AVAILABLE = False
inference_model = None
MODEL_NAME = "Simulation"

# 1. Essayer HoVer-Net (meilleure architecture)
try:
    from src.inference.hoptimus_hovernet import HOptimusHoVerNetInference

    if HOVERNET_CHECKPOINT.exists():
        print(f"Chargement H-optimus-0 + HoVer-Net depuis {HOVERNET_CHECKPOINT}...")
        inference_model = HOptimusHoVerNetInference(str(HOVERNET_CHECKPOINT))
        MODEL_AVAILABLE = True
        MODEL_NAME = "H-optimus-0 + HoVer-Net"
        print(f"✅ {MODEL_NAME} chargé avec succès!")
except Exception as e:
    print(f"HoVer-Net non disponible: {e}")

# 2. Sinon essayer UNETR (fallback)
if not MODEL_AVAILABLE:
    try:
        from src.inference.hoptimus_unetr import HOptimusUNETRInference

        if UNETR_CHECKPOINT.exists():
            print(f"Chargement H-optimus-0 + UNETR depuis {UNETR_CHECKPOINT}...")
            inference_model = HOptimusUNETRInference(str(UNETR_CHECKPOINT))
            MODEL_AVAILABLE = True
            MODEL_NAME = "H-optimus-0 + UNETR"
            print(f"✅ {MODEL_NAME} chargé avec succès!")
    except Exception as e:
        print(f"UNETR non disponible: {e}")

if not MODEL_AVAILABLE:
    print("⚠️ Aucun modèle disponible - Mode simulation activé")


def detect_nuclei_simple(image: np.ndarray) -> np.ndarray:
    """
    Détection simple de noyaux par seuillage.
    Utilisé comme placeholder en attendant le vrai modèle.
    """
    # Convertir en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Inverser (noyaux sombres -> blancs)
    gray = 255 - gray

    # Seuillage adaptatif
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 5
    )

    # Opérations morphologiques pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Trouver les composantes connexes
    n_labels, labels = cv2.connectedComponents(binary)

    return labels, n_labels - 1  # -1 pour exclure le fond


def create_mask_from_labels(
    labels: np.ndarray,
    n_cells: int,
    tissue_type: str = "Breast"
) -> np.ndarray:
    """
    Crée un masque 6-canaux à partir des labels détectés.
    Assigne des types cellulaires selon la distribution du tissu.
    """
    h, w = labels.shape
    mask = np.zeros((h, w, 6), dtype=np.uint8)
    mask[:, :, 0] = 255  # Background par défaut

    if n_cells == 0:
        return mask

    # Distribution selon le type de tissu
    config = TISSUE_CONFIGS.get(tissue_type, TISSUE_CONFIGS["Breast"])
    type_names = list(config.keys())
    type_probs = [config[t] for t in type_names]

    # Mapping des noms vers les indices
    name_to_idx = {
        "Neoplastic": 1,
        "Inflammatory": 2,
        "Connective": 3,
        "Dead": 4,
        "Epithelial": 5
    }

    # Assigner un type à chaque cellule
    np.random.seed(42)
    cell_types = np.random.choice(type_names, size=n_cells, p=type_probs)

    for cell_id in range(1, n_cells + 1):
        cell_mask = labels == cell_id
        if not np.any(cell_mask):
            continue

        cell_type = cell_types[cell_id - 1] if cell_id <= len(cell_types) else "Epithelial"
        type_idx = name_to_idx.get(cell_type, 5)

        mask[:, :, 0][cell_mask] = 0  # Retirer du background
        mask[:, :, type_idx][cell_mask] = 255

    return mask


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

    def analyze_uploaded_image(self, image, tissue_type: str):
        """Analyse une image uploadée par l'utilisateur."""
        if image is None:
            return None, None, None, "⚠️ Veuillez uploader une image"

        # Redimensionner si nécessaire
        h, w = image.shape[:2]
        max_size = 512
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

        # Utiliser H-optimus-0 + HoVer-Net si disponible
        if MODEL_AVAILABLE and inference_model is not None:
            try:
                # Inférence avec le modèle cible
                result_data = inference_model.predict(image)

                # Visualisation segmentation
                result = inference_model.visualize(
                    image, result_data,
                    show_contours=True,
                    show_overlay=True,
                    alpha=0.4
                )

                # Visualisation incertitude (si disponible)
                uncertainty_vis = None
                if hasattr(inference_model, 'visualize_uncertainty'):
                    uncertainty_vis = inference_model.visualize_uncertainty(
                        image, result_data, alpha=0.4
                    )

                # Rapport
                report = inference_model.generate_report(result_data)
                report = f"""
✅ MODÈLE {MODEL_NAME} ACTIF
Architecture: H-optimus-0 (1.1B params) + Décodeur HoVer-Net
Couche 3: Estimation d'incertitude active

{report}
"""
                return image, result, uncertainty_vis, report

            except Exception as e:
                print(f"Erreur {MODEL_NAME}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback vers simulation
                pass

        # Fallback: détection simulée
        labels, n_cells = detect_nuclei_simple(image)
        mask = create_mask_from_labels(labels, n_cells, tissue_type)

        # Visualisation
        overlay = overlay_mask(image, mask, 0.4)
        result = draw_contours(overlay, mask, thickness=2)

        # Rapport
        report = f"""
⚠️ MODE DÉMONSTRATION
La classification des cellules est simulée.
{MODEL_NAME} non disponible ou erreur.

Pour activer le modèle:
1. Entraîner HoVer-Net: python scripts/training/train_hovernet.py
2. Checkpoint attendu: models/checkpoints/hovernet_best.pth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        report += generate_report(mask, tissue_type)

        return image, result, None, report


def create_demo_interface():
    """Crée l'interface Gradio complète."""
    demo = CellVitDemo()

    with gr.Blocks(title="CellViT-Optimus Demo") as interface:

        # Statut du modèle
        if MODEL_AVAILABLE:
            model_status = f"✅ {MODEL_NAME} actif"
        else:
            model_status = "⚠️ Mode simulation"

        gr.Markdown(f"""
        # 🔬 CellViT-Optimus — Démonstration

        **Système d'assistance au triage histopathologique**

        Ce démo permet de visualiser la segmentation et classification des cellules
        dans des images de tissus histopathologiques.

        **Statut modèle:** {model_status}

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

            # Tab 2: Analyser une image uploadée
            with gr.TabItem("📤 Analyser votre Image"):
                if MODEL_AVAILABLE:
                    gr.Markdown(f"""
                    ### Analysez votre propre image histopathologique

                    Uploadez une image de tissu coloré H&E pour obtenir une analyse cellulaire.

                    **✅ {MODEL_NAME} est actif** — L'analyse utilise l'architecture cible:
                    - **Backbone**: H-optimus-0 (1.1B paramètres, gelé)
                    - **Décodeur**: HoVer-Net (3 branches: NP, HV, NT)
                    - **Couche 3**: Estimation d'incertitude (entropie + Mahalanobis)
                    - **Sortie**: {{Fiable | À revoir | Hors domaine}}
                    """)
                else:
                    gr.Markdown(f"""
                    ### Analysez votre propre image histopathologique

                    Uploadez une image de tissu coloré H&E pour obtenir une analyse cellulaire.

                    **⚠️ Mode simulation** — {MODEL_NAME} non disponible.

                    Pour activer le modèle:
                    1. Entraîner: `python scripts/training/train_hovernet.py`
                    2. Checkpoint attendu: `models/checkpoints/hovernet_best.pth`
                    """)

                with gr.Row():
                    with gr.Column(scale=1):
                        upload_image = gr.Image(
                            label="Uploader une image",
                            type="numpy",
                            sources=["upload", "clipboard"]
                        )
                        upload_tissue = gr.Dropdown(
                            choices=list(TISSUE_CONFIGS.keys()),
                            value="Breast",
                            label="Type de tissu (fallback simulation)"
                        )
                        analyze_btn = gr.Button(
                            "🔬 Analyser",
                            variant="primary"
                        )

                    with gr.Column(scale=2):
                        with gr.Row():
                            upload_original = gr.Image(
                                label="Image originale",
                                type="numpy"
                            )
                            upload_result = gr.Image(
                                label="Segmentation cellulaire",
                                type="numpy"
                            )
                        with gr.Row():
                            upload_uncertainty = gr.Image(
                                label="Carte d'incertitude (vert=fiable, rouge=incertain)",
                                type="numpy"
                            )

                upload_report = gr.Textbox(
                    label="Rapport d'analyse (inclut niveau de confiance)",
                    lines=20
                )

                analyze_btn.click(
                    fn=demo.analyze_uploaded_image,
                    inputs=[upload_image, upload_tissue],
                    outputs=[upload_original, upload_result, upload_uncertainty, upload_report]
                )

            # Tab 3: Générer de nouveaux tissus
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

            # Tab 4: À propos
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
