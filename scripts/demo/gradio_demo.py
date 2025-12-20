#!/usr/bin/env python3
"""
Interface Gradio pour la démonstration CellViT-Optimus.

Permet de visualiser interactivement les segmentations cellulaires
et d'explorer les différents types de tissus.

Architecture cible: Optimus-Gate (H-optimus-0 + OrganHead + HoVer-Net)

Fonctionnalités IHM Clinique:
- Panneau morphométrique (métriques pathologiques)
- Gestion des calques (RAW/SEG/HEAT)
- XAI: Cliquer sur une alerte pour voir les noyaux responsables
"""

import gradio as gr
import numpy as np
from pathlib import Path
import cv2
import sys
from typing import Dict, List, Optional, Tuple

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

# Import du module morphométrie
try:
    from src.metrics.morphometry import (
        MorphometryAnalyzer,
        MorphometryReport,
        CELL_TYPES as MORPHO_CELL_TYPES,
    )
    MORPHOMETRY_AVAILABLE = True
except ImportError:
    MORPHOMETRY_AVAILABLE = False
    print("⚠️ Module morphométrie non disponible")

# Import du module feedback Active Learning
try:
    from src.feedback.active_learning import (
        FeedbackCollector,
        FeedbackType,
        get_feedback_collector,
    )
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    print("⚠️ Module feedback non disponible")

# Liste des 19 organes PanNuke pour comparaison
PANNUKE_ORGANS = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix",
    "Colon", "Esophagus", "HeadNeck", "Kidney", "Liver",
    "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
    "Stomach", "Testis", "Thyroid", "Uterus"
]

# Configuration des modèles
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
HOVERNET_CHECKPOINT = CHECKPOINT_DIR / "hovernet_best.pth"
ORGAN_HEAD_CHECKPOINT = CHECKPOINT_DIR / "organ_head_best.pth"
UNETR_CHECKPOINT = CHECKPOINT_DIR / "unetr_best.pth"

# Mapping familles
FAMILY_CHECKPOINTS = {
    "glandular": CHECKPOINT_DIR / "hovernet_glandular_best.pth",
    "digestive": CHECKPOINT_DIR / "hovernet_digestive_best.pth",
    "urologic": CHECKPOINT_DIR / "hovernet_urologic_best.pth",
    "respiratory": CHECKPOINT_DIR / "hovernet_respiratory_best.pth",
    "epidermal": CHECKPOINT_DIR / "hovernet_epidermal_best.pth",
}

# Tenter de charger les modèles (priorité: Multi-Famille > OptimusGate > HoVer-Net > simulation)
MODEL_AVAILABLE = False
inference_model = None
MODEL_NAME = "Simulation"
IS_OPTIMUS_GATE = False
IS_MULTI_FAMILY = False

# 1. Essayer Optimus-Gate Multi-Famille (5 décodeurs spécialisés)
try:
    from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily

    # Vérifier si au moins 1 famille est disponible
    n_families = sum(1 for p in FAMILY_CHECKPOINTS.values() if p.exists())
    if ORGAN_HEAD_CHECKPOINT.exists() and n_families > 0:
        print(f"Chargement Optimus-Gate Multi-Famille ({n_families}/5 familles)...")
        inference_model = OptimusGateInferenceMultiFamily(
            checkpoint_dir=str(CHECKPOINT_DIR),
        )
        MODEL_AVAILABLE = True
        MODEL_NAME = f"Optimus-Gate Multi-Famille ({n_families}/5)"
        IS_OPTIMUS_GATE = True
        IS_MULTI_FAMILY = True
        print(f"✅ {MODEL_NAME} chargé avec succès!")
except Exception as e:
    print(f"Multi-Famille non disponible: {e}")
    import traceback
    traceback.print_exc()

# 2. Sinon essayer OptimusGate simple (1 HoVer-Net global)
if not MODEL_AVAILABLE:
    try:
        from src.inference.optimus_gate_inference import OptimusGateInference

        if HOVERNET_CHECKPOINT.exists() and ORGAN_HEAD_CHECKPOINT.exists():
            print(f"Chargement Optimus-Gate...")
            inference_model = OptimusGateInference(
                hovernet_path=str(HOVERNET_CHECKPOINT),
                organ_head_path=str(ORGAN_HEAD_CHECKPOINT),
            )
            MODEL_AVAILABLE = True
            MODEL_NAME = "Optimus-Gate"
            IS_OPTIMUS_GATE = True
            print(f"✅ {MODEL_NAME} chargé avec succès!")
    except Exception as e:
        print(f"OptimusGate non disponible: {e}")

# 3. Sinon essayer HoVer-Net seul
if not MODEL_AVAILABLE:
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

# 4. Sinon essayer UNETR (fallback)
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


# =============================================================================
# XAI - Fonctions de visualisation explicable
# =============================================================================

def highlight_nuclei_by_ids(
    image: np.ndarray,
    instance_map: np.ndarray,
    nuclei_ids: List[int],
    color: Tuple[int, int, int] = (255, 0, 255),  # Magenta
    thickness: int = 3,
    pulse: bool = True,
) -> np.ndarray:
    """
    Met en surbrillance les noyaux spécifiés par leurs IDs.

    Utilisé pour le XAI: quand l'utilisateur clique sur une alerte,
    on montre quels noyaux ont déclenché cette alerte.

    Args:
        image: Image de base
        instance_map: Carte d'instances (H, W) avec labels 1..N
        nuclei_ids: Liste des IDs de noyaux à surbriller
        color: Couleur de surbrillance
        thickness: Épaisseur des contours
        pulse: Si True, ajoute un effet visuel (contour double)

    Returns:
        Image avec noyaux surbrillés
    """
    result = image.copy()

    for nid in nuclei_ids:
        if nid <= 0 or nid > instance_map.max():
            continue

        mask = (instance_map == nid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Contour principal
            cv2.drawContours(result, contours, -1, color, thickness)

            if pulse:
                # Contour externe (effet "pulse")
                cv2.drawContours(result, contours, -1, (255, 255, 255), thickness + 2)
                cv2.drawContours(result, contours, -1, color, thickness)

    return result


def create_layer_view(
    image: np.ndarray,
    result_data: Dict,
    layer_mode: str = "SEG",
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Crée une vue selon le mode de calque sélectionné.

    Args:
        image: Image originale
        result_data: Résultats de l'inférence
        layer_mode: Mode de calque ("RAW", "SEG", "HEAT", "BOTH")
        alpha: Transparence pour les overlays

    Returns:
        Image avec le calque approprié
    """
    if layer_mode == "RAW":
        return image.copy()

    elif layer_mode == "SEG":
        # Segmentation avec overlay coloré
        if MODEL_AVAILABLE and inference_model is not None:
            return inference_model.visualize(
                image, result_data,
                show_contours=True,
                show_overlay=True,
                alpha=alpha
            )
        return image.copy()

    elif layer_mode == "HEAT":
        # Carte d'incertitude uniquement
        if MODEL_AVAILABLE and hasattr(inference_model, 'visualize_uncertainty'):
            return inference_model.visualize_uncertainty(image, result_data, alpha=0.6)
        return image.copy()

    elif layer_mode == "BOTH":
        # Superposition segmentation + incertitude
        seg = create_layer_view(image, result_data, "SEG", alpha * 0.7)
        if MODEL_AVAILABLE and hasattr(inference_model, 'visualize_uncertainty'):
            heat = inference_model.visualize_uncertainty(image, result_data, alpha=0.3)
            # Blend the two
            return cv2.addWeighted(seg, 0.7, heat, 0.3, 0)
        return seg

    return image.copy()


def generate_morphometry_panel(
    morpho_report: 'MorphometryReport',
    organ: str = "Unknown",
    family: str = "unknown",
) -> str:
    """
    Génère un panneau morphométrique formaté pour l'IHM.

    Présente les métriques cliniques de façon structurée et lisible.
    """
    if morpho_report is None:
        return "❌ Analyse morphométrique non disponible"

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║            PANNEAU MORPHOMÉTRIQUE CLINIQUE               ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ Tissu: {organ.upper():20} | Famille: {family.upper():15} ║",
        "╠══════════════════════════════════════════════════════════╣",
        "║ 📊 MÉTRIQUES NUCLÉAIRES                                   ║",
        f"║   Noyaux détectés    : {morpho_report.n_nuclei:>8}                      ║",
        f"║   Densité            : {morpho_report.nuclei_per_mm2:>8.0f} noyaux/mm²           ║",
        f"║   Hypercellularité   : {morpho_report.nuclear_density_percent:>8.1f}%                      ║",
        "╠══════════════════════════════════════════════════════════╣",
        "║ 📐 MORPHOLOGIE                                            ║",
        f"║   Aire moyenne       : {morpho_report.mean_area_um2:>6.1f} ± {morpho_report.std_area_um2:.1f} µm²           ║",
        f"║   Circularité        : {morpho_report.mean_circularity:>6.2f} ± {morpho_report.std_circularity:.2f}               ║",
    ]

    # Anisocaryose (CV de l'aire)
    if morpho_report.mean_area_um2 > 0:
        cv_area = morpho_report.std_area_um2 / morpho_report.mean_area_um2
        aniso_status = "⚠️" if cv_area > 0.3 else "✓"
        lines.append(f"║   Anisocaryose (CV)  : {cv_area:>6.2f} {aniso_status}                       ║")

    # Index mitotique estimé (NOUVEAU)
    if hasattr(morpho_report, 'mitotic_candidates') and morpho_report.mitotic_candidates > 0:
        lines.extend([
            "╠══════════════════════════════════════════════════════════╣",
            "║ ⚡ INDEX MITOTIQUE ESTIMÉ                                ║",
            f"║   Figures évocatrices: {morpho_report.mitotic_candidates:>8}                      ║",
            f"║   Index /10 HPF      : {morpho_report.mitotic_index_per_10hpf:>8.1f}                      ║",
        ])

    lines.extend([
        "╠══════════════════════════════════════════════════════════╣",
        "║ 🏗️ ARCHITECTURE TISSULAIRE                               ║",
        f"║   Topographie        : {morpho_report.spatial_distribution:>20}         ║",
        f"║   Score clustering   : {morpho_report.clustering_score:>6.2f}                         ║",
    ])

    if morpho_report.stroma_tumor_distance_um > 0:
        lines.append(f"║   Dist. stroma-tumeur: {morpho_report.stroma_tumor_distance_um:>6.1f} µm                      ║")

    # Statut TILs (hot/cold) - NOUVEAU
    if hasattr(morpho_report, 'til_status') and morpho_report.til_status != "indéterminé":
        til_emoji = {"chaud": "🔥", "froid": "❄️", "exclu": "🚫", "intermédiaire": "〰️"}.get(morpho_report.til_status, "❓")
        lines.append(f"║   Statut TILs        : {til_emoji} {morpho_report.til_status.upper():17}         ║")
        if morpho_report.til_penetration_ratio > 0:
            lines.append(f"║   Pénétration TILs   : {morpho_report.til_penetration_ratio:>6.0%}                        ║")

    lines.extend([
        "╠══════════════════════════════════════════════════════════╣",
        "║ 🔬 POPULATION CELLULAIRE                                 ║",
    ])

    # Types avec barres de progression
    emoji_map = {
        "Neoplastic": "🔴",
        "Inflammatory": "🟢",
        "Connective": "🔵",
        "Dead": "🟡",
        "Epithelial": "🩵",
    }

    for cell_type in MORPHO_CELL_TYPES if MORPHOMETRY_AVAILABLE else []:
        pct = morpho_report.type_percentages.get(cell_type, 0)
        count = morpho_report.type_counts.get(cell_type, 0)
        emoji = emoji_map.get(cell_type, "•")
        bar_len = int(pct / 5)  # 20 chars max
        bar = "█" * bar_len + "░" * (15 - bar_len)
        lines.append(f"║   {emoji} {cell_type:12}: {bar} {count:>3} ({pct:>4.1f}%) ║")

    lines.extend([
        "╠══════════════════════════════════════════════════════════╣",
        "║ 📊 RAPPORTS CLINIQUES                                    ║",
        f"║   Ratio I/E (TILs)   : {morpho_report.immuno_epithelial_ratio:>6.2f}                         ║",
        f"║   Ratio néoplasique  : {morpho_report.neoplastic_ratio:>6.1%}                        ║",
        "╠══════════════════════════════════════════════════════════╣",
    ])

    # Alertes avec possibilité de clic (XAI)
    if morpho_report.alerts:
        lines.append("║ ⚠️ POINTS D'ATTENTION (cliquez pour localiser)          ║")
        for i, alert in enumerate(morpho_report.alerts):
            # Troncature si trop long
            alert_short = alert[:50] + "..." if len(alert) > 50 else alert
            lines.append(f"║   [{i+1}] {alert_short:<48} ║")
    else:
        lines.append("║ ✅ Aucune alerte particulière                            ║")

    lines.extend([
        "╠══════════════════════════════════════════════════════════╣",
        f"║ 🎯 Confiance analyse : {morpho_report.confidence_level:>15}                 ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        "⚠️ Document d'aide à la décision - Validation médicale requise",
    ])

    return "\n".join(lines)


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
    """
    Interface de démonstration CellViT-Optimus.

    Fonctionnalités IHM Clinique:
    - Analyse morphométrique complète
    - Gestion des calques (RAW/SEG/HEAT)
    - XAI: Cliquer sur une alerte pour voir les noyaux responsables
    """

    def __init__(self, data_dir: str = "data/demo"):
        self.data_dir = Path(data_dir)
        self.images = None
        self.masks = None
        self.types = None
        self.current_idx = 0

        # État pour XAI et interactions
        self.current_image = None
        self.current_result_data = None
        self.current_morpho_report = None
        self.current_organ = None
        self.current_family = None

        # Analyseur morphométrique
        if MORPHOMETRY_AVAILABLE:
            self.morpho_analyzer = MorphometryAnalyzer(pixel_size_um=0.5)
        else:
            self.morpho_analyzer = None

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
        """
        Analyse une image uploadée avec morphométrie clinique.

        Returns:
            Tuple: (original, segmentation, uncertainty, morpho_panel, ml_report)
        """
        if image is None:
            return None, None, None, "⚠️ Veuillez uploader une image", ""

        # Redimensionner si nécessaire
        h, w = image.shape[:2]
        max_size = 512
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

        # Stocker l'image courante pour XAI
        self.current_image = image.copy()

        # Utiliser le modèle si disponible
        if MODEL_AVAILABLE and inference_model is not None:
            try:
                # Inférence avec le modèle
                result_data = inference_model.predict(image)

                # Stocker pour XAI
                self.current_result_data = result_data

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

                # Extraire organe et famille
                organ_name = "Unknown"
                family = "unknown"
                organ_conf = 0.0

                if IS_OPTIMUS_GATE:
                    organ_info = result_data.get('organ')
                    if organ_info:
                        organ_name = organ_info.organ_name
                        organ_conf = organ_info.confidence
                    family = result_data.get('family', 'unknown')

                self.current_organ = organ_name
                self.current_family = family

                # ==========================================
                # ANALYSE MORPHOMÉTRIQUE
                # ==========================================
                morpho_panel = "❌ Morphométrie non disponible"

                if self.morpho_analyzer is not None:
                    instance_map = result_data.get('instance_map')
                    nt_mask = result_data.get('nt_mask')

                    if instance_map is not None and nt_mask is not None:
                        # Analyse morphométrique
                        self.current_morpho_report = self.morpho_analyzer.analyze(
                            instance_map, nt_mask
                        )
                        morpho_panel = generate_morphometry_panel(
                            self.current_morpho_report,
                            organ=organ_name,
                            family=family
                        )

                # Rapport ML (technique)
                ml_report = inference_model.generate_report(result_data)

                # Header selon le modèle
                if IS_OPTIMUS_GATE:
                    # Comparaison avec l'organe attendu
                    expected = tissue_type
                    predicted = organ_name
                    match = expected.lower().replace("_", "").replace("-", "") == \
                            predicted.lower().replace("_", "").replace("-", "")

                    if match:
                        comparison = f"✅ CORRECT — Prédit: {predicted} = Attendu: {expected}"
                    else:
                        comparison = f"❌ DIFFÉRENT — Prédit: {predicted} ≠ Attendu: {expected}"

                    header = f"""
✅ OPTIMUS-GATE ACTIF ({MODEL_NAME})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 Architecture:
   • Backbone: H-optimus-0 (1.1B params)
   • Flux Global: OrganHead (classification)
   • Flux Local: HoVer-Net[{family}] (segmentation)
   • Sécurité: Triple OOD

🏥 Organe détecté: {organ_name} ({organ_conf:.1%})
🎯 {comparison}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
                else:
                    header = f"""
✅ MODÈLE {MODEL_NAME} ACTIF
Architecture: H-optimus-0 + HoVer-Net
Couche 3: Estimation d'incertitude active

"""
                ml_report = header + ml_report
                return image, result, uncertainty_vis, morpho_panel, ml_report

            except Exception as e:
                print(f"Erreur {MODEL_NAME}: {e}")
                import traceback
                traceback.print_exc()
                pass

        # Fallback: détection simulée
        labels, n_cells = detect_nuclei_simple(image)
        mask = create_mask_from_labels(labels, n_cells, tissue_type)

        self.current_result_data = None
        self.current_morpho_report = None

        # Visualisation
        overlay = overlay_mask(image, mask, 0.4)
        result = draw_contours(overlay, mask, thickness=2)

        # Rapport
        ml_report = f"""
⚠️ MODE DÉMONSTRATION
La classification des cellules est simulée.
{MODEL_NAME} non disponible ou erreur.

Pour activer Optimus-Gate:
1. python scripts/training/train_hovernet.py
2. python scripts/training/train_organ_head.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        ml_report += generate_report(mask, tissue_type)

        return image, result, None, "❌ Morphométrie non disponible (mode simulation)", ml_report

    def switch_layer(self, layer_mode: str, alpha: float = 0.4):
        """
        Change le mode de calque affiché.

        Args:
            layer_mode: "RAW", "SEG", "HEAT", "BOTH"

        Returns:
            Image avec le calque sélectionné
        """
        if self.current_image is None or self.current_result_data is None:
            return None

        return create_layer_view(
            self.current_image,
            self.current_result_data,
            layer_mode,
            alpha
        )

    def highlight_alert(self, alert_idx: int):
        """
        Met en surbrillance les noyaux ayant déclenché une alerte spécifique.

        Args:
            alert_idx: Index de l'alerte (0-based)

        Returns:
            Image avec noyaux surbrillés
        """
        if self.current_image is None or self.current_result_data is None:
            return None
        if self.current_morpho_report is None:
            return self.current_image.copy()

        # Récupérer la clé de l'alerte
        alert_keys = list(self.current_morpho_report.alert_nuclei_ids.keys())

        if alert_idx < 0 or alert_idx >= len(alert_keys):
            # Pas d'alerte valide, retourner segmentation normale
            if MODEL_AVAILABLE and inference_model is not None:
                return inference_model.visualize(
                    self.current_image, self.current_result_data,
                    show_contours=True, show_overlay=True, alpha=0.4
                )
            return self.current_image.copy()

        alert_key = alert_keys[alert_idx]
        nuclei_ids = self.current_morpho_report.alert_nuclei_ids.get(alert_key, [])

        if not nuclei_ids:
            return self.current_image.copy()

        # Créer la base (segmentation)
        if MODEL_AVAILABLE and inference_model is not None:
            base = inference_model.visualize(
                self.current_image, self.current_result_data,
                show_contours=True, show_overlay=True, alpha=0.4
            )
        else:
            base = self.current_image.copy()

        # Surbrillance des noyaux
        instance_map = self.current_result_data.get('instance_map')
        if instance_map is not None:
            # Couleur selon le type d'alerte
            color_map = {
                "anisocaryose": (255, 0, 255),    # Magenta
                "atypie_forme": (255, 165, 0),    # Orange
                "neoplasique": (255, 0, 0),       # Rouge
                "infiltration": (0, 255, 0),      # Vert
                "mitose": (255, 255, 0),          # Jaune (figures mitotiques)
            }
            color = color_map.get(alert_key, (255, 255, 0))

            result = highlight_nuclei_by_ids(base, instance_map, nuclei_ids, color)
            return result

        return base


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

            # Tab 2: Analyser une image uploadée — IHM CLINIQUE
            with gr.TabItem("🏥 Analyse Clinique"):
                if IS_OPTIMUS_GATE:
                    gr.Markdown(f"""
                    ### 🔬 Interface d'Analyse Morphométrique Clinique

                    Uploadez une image H&E pour une analyse pathologique complète.

                    **✅ {MODEL_NAME}** | 📊 Morphométrie | 🔍 XAI (cliquez sur les alertes)
                    """)
                elif MODEL_AVAILABLE:
                    gr.Markdown(f"""
                    ### 🔬 Analyse Histopathologique

                    **✅ {MODEL_NAME}** actif | 📊 Morphométrie clinique disponible
                    """)
                else:
                    gr.Markdown("""
                    ### 🔬 Analyse Histopathologique

                    **⚠️ Mode simulation** — Modèle non disponible
                    """)

                with gr.Row():
                    # Colonne gauche: Upload et contrôles
                    with gr.Column(scale=1):
                        upload_image = gr.Image(
                            label="📤 Uploader une image H&E",
                            type="numpy",
                            sources=["upload", "clipboard"]
                        )
                        upload_tissue = gr.Dropdown(
                            choices=PANNUKE_ORGANS,
                            value="Prostate",
                            label="🎯 Organe attendu (comparaison)"
                        )
                        analyze_btn = gr.Button(
                            "🔬 Analyser",
                            variant="primary"
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 🎨 Gestion des Calques")

                        layer_mode = gr.Radio(
                            choices=["RAW", "SEG", "HEAT", "BOTH"],
                            value="SEG",
                            label="Mode d'affichage",
                            info="RAW=Image brute | SEG=Segmentation | HEAT=Incertitude | BOTH=Combiné"
                        )
                        layer_alpha = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.4,
                            label="Transparence overlay"
                        )
                        layer_btn = gr.Button("🔄 Appliquer calque")

                        gr.Markdown("---")
                        gr.Markdown("### 🔍 XAI — Expliquer les alertes")

                        alert_selector = gr.Dropdown(
                            choices=[],
                            label="Sélectionner une alerte",
                            info="Les noyaux concernés seront surbrillés",
                            interactive=True
                        )
                        highlight_btn = gr.Button("✨ Localiser les noyaux")

                    # Colonne droite: Visualisations
                    with gr.Column(scale=2):
                        with gr.Row():
                            upload_original = gr.Image(
                                label="📷 Image originale",
                                type="numpy"
                            )
                            upload_result = gr.Image(
                                label="🔬 Segmentation / XAI",
                                type="numpy"
                            )
                        with gr.Row():
                            upload_uncertainty = gr.Image(
                                label="🌡️ Carte d'incertitude (vert=fiable, rouge=incertain)",
                                type="numpy"
                            )

                # Panneaux de rapport en bas
                with gr.Row():
                    with gr.Column(scale=1):
                        morpho_panel = gr.Textbox(
                            label="📊 Panneau Morphométrique Clinique",
                            lines=25,
                            max_lines=35,
                            show_copy_button=True
                        )
                    with gr.Column(scale=1):
                        ml_report = gr.Textbox(
                            label="🤖 Rapport Technique (ML)",
                            lines=25,
                            max_lines=35,
                            show_copy_button=True
                        )

                # Fonction pour mettre à jour les alertes disponibles
                def update_alert_choices(morpho_text):
                    """Extrait les alertes du panneau morphométrique."""
                    if morpho_text is None or "POINTS D'ATTENTION" not in morpho_text:
                        return gr.update(choices=[], value=None)

                    alerts = []
                    lines = morpho_text.split("\n")
                    for line in lines:
                        if line.strip().startswith("[") and "]" in line:
                            # Format: [1] Alerte texte...
                            try:
                                idx = line.split("]")[0].replace("[", "").replace("║", "").strip()
                                alert_text = line.split("]")[1].strip()
                                alerts.append(f"[{idx}] {alert_text[:40]}...")
                            except:
                                pass

                    return gr.update(choices=alerts, value=alerts[0] if alerts else None)

                # Handler pour highlight
                def handle_highlight(alert_choice):
                    if alert_choice is None:
                        return demo.switch_layer("SEG", 0.4)
                    try:
                        # Extraire l'index de "[1] ..."
                        idx = int(alert_choice.split("]")[0].replace("[", "").strip()) - 1
                        return demo.highlight_alert(idx)
                    except:
                        return demo.switch_layer("SEG", 0.4)

                # Connexions
                analyze_btn.click(
                    fn=demo.analyze_uploaded_image,
                    inputs=[upload_image, upload_tissue],
                    outputs=[upload_original, upload_result, upload_uncertainty, morpho_panel, ml_report]
                ).then(
                    fn=update_alert_choices,
                    inputs=[morpho_panel],
                    outputs=[alert_selector]
                )

                layer_btn.click(
                    fn=demo.switch_layer,
                    inputs=[layer_mode, layer_alpha],
                    outputs=[upload_result]
                )

                highlight_btn.click(
                    fn=handle_highlight,
                    inputs=[alert_selector],
                    outputs=[upload_result]
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
                ## CellViT-Optimus — Architecture Optimus-Gate

                ### Architecture Double Flux

                ```
                ┌─────────────────────────────────────────────────────────┐
                │                   Image H&E (WSI)                       │
                └─────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                ┌─────────────────────────────────────────────────────────┐
                │            H-OPTIMUS-0 (Backbone gelé)                  │
                │            ViT-Giant/14, 1.1B params                    │
                │                                                         │
                │      Sortie: CLS token (1×1536) + Patches (256×1536)   │
                └─────────────────────────────────────────────────────────┘
                                           │
                          ┌────────────────┴────────────────┐
                          ▼                                 ▼
                ┌─────────────────────────┐   ┌─────────────────────────┐
                │   FLUX GLOBAL           │   │   FLUX LOCAL            │
                │   OrganHead             │   │   HoVer-Net Decoder     │
                │                         │   │                         │
                │   • CLS token → MLP     │   │   • Patches → Upsampling│
                │   • 19 organes PanNuke  │   │   • NP: noyaux binaire  │
                │   • OOD Mahalanobis     │   │   • HV: séparation      │
                │                         │   │   • NT: 5 types cell.   │
                │   ✅ Accuracy: 96%      │   │   ✅ Dice: 0.96         │
                └─────────────────────────┘   └─────────────────────────┘
                          │                                 │
                          └────────────────┬────────────────┘
                                           ▼
                ┌─────────────────────────────────────────────────────────┐
                │              TRIPLE SÉCURITÉ OOD                        │
                │                                                         │
                │   1. Entropie softmax (incertitude classification)     │
                │   2. Mahalanobis global (distance CLS token)           │
                │   3. Mahalanobis local (distance patches)              │
                │                                                         │
                │   Sortie: {Fiable ✅ | À revoir ⚠️ | Hors domaine 🚫}  │
                └─────────────────────────────────────────────────────────┘
                ```

                ### Types de Cellules (Flux Local)

                | Type | Couleur | Description |
                |------|---------|-------------|
                | Neoplastic | 🔴 Rouge | Cellules tumorales |
                | Inflammatory | 🟢 Vert | Cellules immunitaires |
                | Connective | 🔵 Bleu | Tissu de soutien |
                | Dead | 🟡 Jaune | Cellules nécrotiques |
                | Epithelial | 🩵 Cyan | Cellules épithéliales |

                ### Organes Supportés (Flux Global)

                Les 19 organes du dataset PanNuke:
                - Adrenal gland, Bile duct, Bladder, Breast, Cervix
                - Colon, Esophagus, HeadNeck, Kidney, Liver
                - Lung, Ovarian, Pancreatic, Prostate, Skin
                - Stomach, Testis, Thyroid, Uterus

                ### Références

                - **H-optimus-0**: [Bioptimus](https://huggingface.co/bioptimus/H-optimus-0)
                - **CellViT**: [TIO-IKIM](https://github.com/TIO-IKIM/CellViT)
                - **HoVer-Net**: [Warwick TIA](https://github.com/vqdang/hover_net)
                - **PanNuke**: [Warwick TIA](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/)

                ---

                *Système d'assistance au triage histopathologique.*
                *Ne remplace pas le pathologiste - aide à prioriser et sécuriser.*
                """)

            # Tab 5: Feedback Expert (Active Learning)
            with gr.TabItem("📝 Feedback Expert"):
                if FEEDBACK_AVAILABLE:
                    gr.Markdown("""
                    ### Mode "Seconde Lecture" - Arbitrage Expert

                    Ce panneau permet aux pathologistes de signaler les désaccords
                    avec les prédictions du modèle. Les corrections sont collectées
                    pour améliorer le système de manière continue.

                    **Types de corrections possibles:**
                    - 🔴 **Type cellulaire incorrect** — Le modèle a mal classifié une cellule
                    - 🟡 **Fausse mitose** — Une figure mitotique était en fait autre chose
                    - 🟢 **Mitose manquée** — Le modèle n'a pas détecté une vraie mitose
                    - 🔵 **Statut TILs incorrect** — Chaud/Froid mal évalué
                    - ⚫ **Organe incorrect** — Mauvais organe détecté

                    ---
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            fb_type = gr.Dropdown(
                                choices=[
                                    "Type cellulaire incorrect",
                                    "Fausse mitose (faux positif)",
                                    "Mitose manquée (faux négatif)",
                                    "Statut TILs incorrect",
                                    "Organe incorrect",
                                    "Alerte non justifiée",
                                    "Autre",
                                ],
                                value="Type cellulaire incorrect",
                                label="Type de correction"
                            )

                            fb_predicted = gr.Textbox(
                                label="Prédiction du modèle",
                                placeholder="ex: Neoplastic"
                            )

                            fb_corrected = gr.Textbox(
                                label="Correction experte",
                                placeholder="ex: Inflammatory"
                            )

                            fb_severity = gr.Radio(
                                choices=["low", "medium", "high", "critical"],
                                value="medium",
                                label="Sévérité de l'erreur"
                            )

                            fb_comment = gr.Textbox(
                                label="Commentaire (optionnel)",
                                placeholder="Détails supplémentaires...",
                                lines=3
                            )

                            fb_submit = gr.Button(
                                "📝 Enregistrer la correction",
                                variant="primary"
                            )

                        with gr.Column(scale=1):
                            fb_status = gr.Textbox(
                                label="Statut",
                                lines=5,
                                interactive=False
                            )

                            fb_stats = gr.Textbox(
                                label="Statistiques de la session",
                                lines=10,
                                interactive=False
                            )

                            fb_refresh = gr.Button("🔄 Rafraîchir les statistiques")

                    # Handlers
                    def submit_feedback(fb_type, predicted, corrected, severity, comment):
                        collector = get_feedback_collector()

                        type_mapping = {
                            "Type cellulaire incorrect": FeedbackType.CELL_TYPE_WRONG,
                            "Fausse mitose (faux positif)": FeedbackType.MITOSIS_FALSE_POSITIVE,
                            "Mitose manquée (faux négatif)": FeedbackType.MITOSIS_MISSED,
                            "Statut TILs incorrect": FeedbackType.TILS_STATUS_WRONG,
                            "Organe incorrect": FeedbackType.ORGAN_WRONG,
                            "Alerte non justifiée": FeedbackType.FALSE_ALARM,
                            "Autre": FeedbackType.OTHER,
                        }

                        entry = collector.add_feedback(
                            feedback_type=type_mapping.get(fb_type, FeedbackType.OTHER),
                            predicted_class=predicted,
                            corrected_class=corrected,
                            severity=severity,
                            expert_comment=comment,
                        )

                        # Sauvegarder immédiatement
                        path = collector.save_session()

                        summary = collector.get_session_summary()

                        return (
                            f"✅ Correction enregistrée!\n\n"
                            f"ID: {entry.id}\n"
                            f"Type: {fb_type}\n"
                            f"Sévérité: {severity}\n\n"
                            f"Fichier: {path}",
                            summary
                        )

                    def refresh_stats():
                        collector = get_feedback_collector()
                        stats = collector.get_statistics()

                        if stats.get("total", 0) == 0:
                            return "Aucun feedback collecté dans cette session."

                        lines = [
                            f"📊 STATISTIQUES GLOBALES",
                            f"========================",
                            f"Total corrections: {stats['total']}",
                            "",
                            "Par type:",
                        ]
                        for t, count in stats.get("by_type", {}).items():
                            lines.append(f"  - {t}: {count}")

                        lines.extend(["", "Par sévérité:"])
                        for s, count in stats.get("by_severity", {}).items():
                            if count > 0:
                                lines.append(f"  - {s}: {count}")

                        if stats.get("common_corrections"):
                            lines.extend(["", "Corrections fréquentes:"])
                            for corr, count in stats["common_corrections"][:5]:
                                lines.append(f"  - {corr}: {count}x")

                        return "\n".join(lines)

                    fb_submit.click(
                        fn=submit_feedback,
                        inputs=[fb_type, fb_predicted, fb_corrected, fb_severity, fb_comment],
                        outputs=[fb_status, fb_stats]
                    )

                    fb_refresh.click(
                        fn=refresh_stats,
                        inputs=[],
                        outputs=[fb_stats]
                    )

                else:
                    gr.Markdown("""
                    ### ⚠️ Module Feedback non disponible

                    Le module `src.feedback.active_learning` n'est pas chargé.
                    Vérifiez l'installation du projet.
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
