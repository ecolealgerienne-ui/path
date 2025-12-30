"""
CellViT-Optimus R&D Cockpit — Module de Visualisation.

Fonctions pour créer des overlays, cartes de couleurs et visualisations
à partir des résultats d'analyse.

Réutilise les conventions de couleurs de scripts/demo/visualize_cells.py.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io


# Palette de couleurs pour les types cellulaires (cohérent avec visualize_cells.py)
CELL_COLORS = {
    "Background": (0, 0, 0),
    "Neoplastic": (255, 50, 50),       # Rouge vif - tumeur
    "Inflammatory": (50, 255, 50),      # Vert vif - inflammation
    "Connective": (50, 50, 255),        # Bleu vif - conjonctif
    "Dead": (255, 255, 50),             # Jaune - nécrose
    "Epithelial": (50, 255, 255),       # Cyan - épithélium
}

CELL_COLORS_RGB = {k: v for k, v in CELL_COLORS.items()}
CELL_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CELL_COLORS.items()}

# Index vers type
TYPE_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]


# =============================================================================
# CONFIGURATION OVERLAYS (Palette R&D standardisée)
# =============================================================================

OVERLAY_CONFIG = {
    # Transparence
    "segmentation_alpha": 0.4,
    "contour_thickness": 1,
    "anomaly_alpha": 0.5,

    # Couleurs Phase 1 (RGB)
    "uncertainty_color": (255, 191, 0),     # Ambre
    "density_cmap": "YlOrRd",               # Jaune-Orange-Rouge

    # Couleurs Phase 2 (RGB)
    "fusion_color": (255, 0, 255),          # Magenta
    "over_seg_color": (0, 255, 255),        # Cyan

    # Couleurs Phase 3 (RGB)
    "hotspot_color": (255, 165, 0),         # Orange
    "mitosis_high_color": (255, 0, 0),      # Rouge (score élevé)
    "mitosis_low_color": (255, 255, 0),     # Jaune (score bas)
    "chromatin_color": (148, 0, 211),       # Violet
    "voronoi_color": (100, 100, 100),       # Gris
}

# Ordre de superposition des overlays (z-index croissant)
OVERLAY_ORDER = [
    "density",          # Fond - heatmap densité
    "segmentation",     # Couleurs par type
    "contours",         # Bordures des noyaux
    "voronoi",          # Tessellation spatiale
    "uncertainty",      # Zones incertaines
    "hotspots",         # Clusters haute densité
    "chromatin",        # Noyaux chromatine hétérogène
    "mitoses",          # Candidats mitose
    "anomalies",        # Fusions / sur-segmentations (dernier = plus visible)
]


def create_segmentation_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    type_map: Optional[np.ndarray] = None,
    alpha: float = 0.4,
    show_ids: bool = False,
) -> np.ndarray:
    """
    Crée un overlay coloré des instances segmentées.

    Args:
        image: Image RGB originale (H, W, 3)
        instance_map: Carte d'instances (H, W) avec IDs
        type_map: Carte des types (H, W) optionnelle
        alpha: Transparence de l'overlay
        show_ids: Afficher les IDs des instances

    Returns:
        Image avec overlay (H, W, 3) RGB
    """
    result = image.copy()
    h, w = instance_map.shape

    # Générer des couleurs aléatoires pour les instances
    np.random.seed(42)  # Reproductibilité
    n_instances = instance_map.max() + 1
    colors = np.random.randint(100, 255, size=(n_instances, 3))
    colors[0] = [0, 0, 0]  # Background

    # Si type_map disponible, utiliser les couleurs de type
    if type_map is not None:
        for inst_id in range(1, n_instances):
            mask = instance_map == inst_id
            if mask.sum() == 0:
                continue

            # Type dominant
            types_in_mask = type_map[mask]
            if len(types_in_mask) > 0:
                type_idx = int(np.bincount(types_in_mask.astype(int)).argmax())
                if type_idx < len(TYPE_NAMES):
                    colors[inst_id] = CELL_COLORS_RGB[TYPE_NAMES[type_idx]]

    # Créer l'overlay
    overlay = np.zeros_like(image)
    for inst_id in range(1, n_instances):
        mask = instance_map == inst_id
        overlay[mask] = colors[inst_id]

    # Blend avec numpy (plus robuste que cv2.addWeighted sur masques)
    mask_any = instance_map > 0
    if mask_any.any():
        blended = (
            image[mask_any].astype(np.float32) * (1 - alpha) +
            overlay[mask_any].astype(np.float32) * alpha
        )
        result[mask_any] = np.clip(blended, 0, 255).astype(np.uint8)

    # Afficher les IDs
    if show_ids:
        for inst_id in range(1, n_instances):
            mask = instance_map == inst_id
            if mask.sum() < 20:
                continue
            coords = np.where(mask)
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            cv2.putText(
                result, str(inst_id), (cx - 5, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
            )

    return result


def create_contour_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    type_map: Optional[np.ndarray] = None,
    thickness: int = 1,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Dessine les contours des instances sur l'image.

    Args:
        image: Image RGB originale (H, W, 3)
        instance_map: Carte d'instances (H, W)
        type_map: Carte des types pour couleur par type
        thickness: Épaisseur des contours
        color: Couleur fixe (sinon utilise type)

    Returns:
        Image avec contours (H, W, 3) RGB
    """
    result = image.copy()

    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue

        mask = (instance_map == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Couleur
        if color is not None:
            c = color
        elif type_map is not None:
            types_in_mask = type_map[instance_map == inst_id]
            if len(types_in_mask) > 0:
                type_idx = int(np.bincount(types_in_mask.astype(int)).argmax())
                if type_idx < len(TYPE_NAMES):
                    c = CELL_COLORS_RGB[TYPE_NAMES[type_idx]]
                else:
                    c = (200, 200, 200)
            else:
                c = (200, 200, 200)
        else:
            c = (0, 255, 0)  # Vert par défaut

        cv2.drawContours(result, contours, -1, c, thickness)

    return result


def create_uncertainty_overlay(
    image: np.ndarray,
    uncertainty_map: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Crée un overlay des zones d'incertitude.

    Args:
        image: Image RGB originale
        uncertainty_map: Carte d'incertitude (H, W) [0, 1]
        threshold: Seuil pour affichage
        alpha: Transparence

    Returns:
        Image avec zones incertaines en ambre
    """
    result = image.copy()

    # Couleur ambre pour incertitude
    amber = np.array([255, 191, 0], dtype=np.uint8)

    # Créer overlay
    uncertain_mask = uncertainty_map > threshold
    overlay = np.zeros_like(image)
    overlay[uncertain_mask] = amber

    # Blend
    if uncertain_mask.any():
        intensity = uncertainty_map[uncertain_mask] * alpha
        for c in range(3):
            result[uncertain_mask, c] = (
                image[uncertain_mask, c] * (1 - intensity) +
                overlay[uncertain_mask, c] * intensity
            ).astype(np.uint8)

    return result


def create_uncertainty_map(
    uncertainty_map: np.ndarray,
    colormap: str = "YlOrRd"
) -> np.ndarray:
    """
    Crée une visualisation colorée de la carte d'incertitude.

    Args:
        uncertainty_map: Carte (H, W) [0, 1]
        colormap: Colormap matplotlib

    Returns:
        Image RGB (H, W, 3)
    """
    # Normaliser
    normalized = np.clip(uncertainty_map, 0, 1)

    # Appliquer colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized)[:, :, :3]  # Drop alpha

    return (colored * 255).astype(np.uint8)


def create_density_heatmap(
    instance_map: np.ndarray,
    kernel_size: int = 21,
) -> np.ndarray:
    """
    Crée une carte de densité nucléaire.

    Args:
        instance_map: Carte d'instances (H, W)
        kernel_size: Taille du noyau gaussien

    Returns:
        Heatmap RGB (H, W, 3)
    """
    # Créer carte binaire des noyaux
    binary = (instance_map > 0).astype(np.float32)

    # Kernel density estimation via blur gaussien
    density = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)

    # Normaliser
    if density.max() > 0:
        density = density / density.max()

    # Colormap "hot"
    cmap = plt.get_cmap("hot")
    colored = cmap(density)[:, :, :3]

    return (colored * 255).astype(np.uint8)


def create_type_distribution_chart(
    type_counts: Dict[str, int],
    figsize: Tuple[int, int] = (6, 4),
) -> np.ndarray:
    """
    Crée un bar chart de la distribution des types cellulaires.

    Args:
        type_counts: Dictionnaire {type: count}
        figsize: Taille de la figure

    Returns:
        Image RGB du graphique
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    types = [t for t in TYPE_NAMES if t in type_counts]
    counts = [type_counts.get(t, 0) for t in types]
    colors = [tuple(c/255 for c in CELL_COLORS_RGB[t]) for t in types]

    bars = ax.barh(types, counts, color=colors, edgecolor='black', linewidth=0.5)

    # Style
    ax.set_xlabel("Nombre de noyaux")
    ax.set_title("Distribution des types cellulaires")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotations
    for bar, count in zip(bars, counts):
        ax.annotate(
            f'{count}',
            xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
            xytext=(5, 0),
            textcoords='offset points',
            va='center',
            fontsize=9
        )

    plt.tight_layout()

    # Convertir en image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    plt.close(fig)
    buf.seek(0)

    img = plt.imread(buf)
    buf.close()

    return (img[:, :, :3] * 255).astype(np.uint8)


def create_morphometry_summary(
    report,  # MorphometryReport
    figsize: Tuple[int, int] = (8, 6),
) -> np.ndarray:
    """
    Crée une visualisation résumée des métriques morphométriques.

    Args:
        report: MorphometryReport
        figsize: Taille de la figure

    Returns:
        Image RGB
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='white')

    # 1. Distribution des types
    ax1 = axes[0, 0]
    types = [t for t in TYPE_NAMES if t in report.type_counts]
    counts = [report.type_counts.get(t, 0) for t in types]
    colors = [tuple(c/255 for c in CELL_COLORS_RGB[t]) for t in types]
    ax1.pie(counts, labels=types, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Types cellulaires")

    # 2. Métriques clés
    ax2 = axes[0, 1]
    ax2.axis('off')
    metrics_text = f"""
    Noyaux détectés: {report.n_nuclei}
    Densité: {report.nuclei_per_mm2:.0f}/mm²

    Aire moyenne: {report.mean_area_um2:.1f} ± {report.std_area_um2:.1f} µm²
    Circularité: {report.mean_circularity:.2f} ± {report.std_circularity:.2f}

    Index mitotique: {report.mitotic_index_per_10hpf:.1f}/10 HPF
    TILs status: {report.til_status}
    """
    ax2.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', transform=ax2.transAxes)
    ax2.set_title("Métriques clés")

    # 3. Alertes
    ax3 = axes[1, 0]
    ax3.axis('off')
    if report.alerts:
        alerts_text = "\n".join(report.alerts[:5])
    else:
        alerts_text = "Aucune alerte"
    ax3.text(0.1, 0.5, alerts_text, fontsize=9, verticalalignment='center',
             wrap=True, transform=ax3.transAxes)
    ax3.set_title("Points d'attention")

    # 4. Confiance
    ax4 = axes[1, 1]
    ax4.axis('off')
    confidence_colors = {"Haute": "green", "Modérée": "orange", "Faible": "red"}
    ax4.text(0.5, 0.5, report.confidence_level, fontsize=24,
             horizontalalignment='center', verticalalignment='center',
             color=confidence_colors.get(report.confidence_level, "gray"),
             transform=ax4.transAxes)
    ax4.set_title("Niveau de confiance")

    plt.tight_layout()

    # Convertir en image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    plt.close(fig)
    buf.seek(0)

    img = plt.imread(buf)
    buf.close()

    return (img[:, :, :3] * 255).astype(np.uint8)


def highlight_nuclei(
    image: np.ndarray,
    instance_map: np.ndarray,
    nucleus_ids: List[int],
    color: Tuple[int, int, int] = (255, 0, 255),  # Magenta
    thickness: int = 2,
) -> np.ndarray:
    """
    Met en évidence des noyaux spécifiques (pour alertes).

    Args:
        image: Image RGB
        instance_map: Carte d'instances
        nucleus_ids: IDs des noyaux à surligner
        color: Couleur de surlignage
        thickness: Épaisseur du contour

    Returns:
        Image avec noyaux surlignés
    """
    result = image.copy()

    for nid in nucleus_ids:
        mask = (instance_map == nid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, thickness)

    return result


def create_debug_panel(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    instance_map: np.ndarray,
    figsize: Tuple[int, int] = (12, 4),
) -> np.ndarray:
    """
    Crée un panneau de debug montrant les étapes du pipeline.

    Args:
        np_pred: Prédiction NP (H, W) [0, 1]
        hv_pred: Prédiction HV (2, H, W) [-1, 1]
        instance_map: Carte d'instances finale

    Returns:
        Image RGB du panneau
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize, facecolor='white')

    # 1. NP probability
    axes[0].imshow(np_pred, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title("NP Probability")
    axes[0].axis('off')

    # 2. HV horizontal
    axes[1].imshow(hv_pred[0], cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title("HV Horizontal")
    axes[1].axis('off')

    # 3. HV vertical
    axes[2].imshow(hv_pred[1], cmap='RdBu', vmin=-1, vmax=1)
    axes[2].set_title("HV Vertical")
    axes[2].axis('off')

    # 4. Instances
    axes[3].imshow(instance_map, cmap='nipy_spectral')
    axes[3].set_title(f"Instances ({instance_map.max()})")
    axes[3].axis('off')

    plt.tight_layout()

    # Convertir en image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    plt.close(fig)
    buf.seek(0)

    img = plt.imread(buf)
    buf.close()

    return (img[:, :, :3] * 255).astype(np.uint8)


def create_anomaly_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    fusion_ids: List[int],
    over_seg_ids: List[int],
    thickness: int = 2,
) -> np.ndarray:
    """
    Crée un overlay mettant en évidence les anomalies détectées.

    Phase 2: Surligne les fusions potentielles en magenta et
    les sur-segmentations en cyan.

    Args:
        image: Image RGB originale
        instance_map: Carte d'instances
        fusion_ids: IDs des noyaux potentiellement fusionnés
        over_seg_ids: IDs des sur-segmentations
        thickness: Épaisseur des contours

    Returns:
        Image avec anomalies surlignées
    """
    result = image.copy()

    # Fusions en magenta
    for nid in fusion_ids:
        mask = (instance_map == nid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 0, 255), thickness)  # Magenta

        # Ajouter indicateur "F"
        coords = np.where(instance_map == nid)
        if len(coords[0]) > 0:
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            cv2.putText(result, "F", (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 0, 255), 1)

    # Sur-segmentations en cyan
    for nid in over_seg_ids:
        mask = (instance_map == nid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 255), thickness)  # Cyan

        # Ajouter indicateur "S"
        coords = np.where(instance_map == nid)
        if len(coords[0]) > 0:
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            cv2.putText(result, "S", (cx - 3, cy + 3), cv2.FONT_HERSHEY_SIMPLEX,
                       0.3, (0, 255, 255), 1)

    return result


def create_debug_panel_enhanced(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    instance_map: np.ndarray,
    n_fusions: int = 0,
    n_over_seg: int = 0,
    figsize: Tuple[int, int] = (14, 4),
) -> np.ndarray:
    """
    Crée un panneau de debug amélioré avec alertes anomalies.

    Phase 2: Ajoute un panneau d'alertes à droite.

    Args:
        np_pred: Prédiction NP (H, W) [0, 1]
        hv_pred: Prédiction HV (2, H, W) [-1, 1]
        instance_map: Carte d'instances finale
        n_fusions: Nombre de fusions détectées
        n_over_seg: Nombre de sur-segmentations

    Returns:
        Image RGB du panneau
    """
    fig, axes = plt.subplots(1, 5, figsize=figsize, facecolor='white',
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.6]})

    # 1. NP probability
    axes[0].imshow(np_pred, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title("NP Probability")
    axes[0].axis('off')

    # 2. HV horizontal
    axes[1].imshow(hv_pred[0], cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title("HV Horizontal")
    axes[1].axis('off')

    # 3. HV vertical
    axes[2].imshow(hv_pred[1], cmap='RdBu', vmin=-1, vmax=1)
    axes[2].set_title("HV Vertical")
    axes[2].axis('off')

    # 4. Instances
    axes[3].imshow(instance_map, cmap='nipy_spectral')
    axes[3].set_title(f"Instances ({instance_map.max()})")
    axes[3].axis('off')

    # 5. Alertes anomalies
    axes[4].axis('off')
    axes[4].set_xlim(0, 1)
    axes[4].set_ylim(0, 1)

    alert_text = "ALERTES\n" + "─" * 12 + "\n\n"

    if n_fusions > 0:
        alert_text += f"⚠ {n_fusions} fusion(s)\n"
        alert_text += "  (aire > 2× moy.)\n\n"
    else:
        alert_text += "✓ 0 fusion\n\n"

    if n_over_seg > 0:
        alert_text += f"⚠ {n_over_seg} sur-seg.\n"
        alert_text += "  (aire < 0.5× moy.)\n"
    else:
        alert_text += "✓ 0 sur-seg.\n"

    # Couleur du texte selon anomalies
    text_color = 'red' if (n_fusions > 0 or n_over_seg > 0) else 'green'

    axes[4].text(0.1, 0.9, alert_text, transform=axes[4].transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 color=text_color,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Convertir en image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    plt.close(fig)
    buf.seek(0)

    img = plt.imread(buf)
    buf.close()

    return (img[:, :, :3] * 255).astype(np.uint8)


def create_voronoi_overlay(
    image: np.ndarray,
    centroids: List[Tuple[int, int]],
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Crée un overlay avec tessellation de Voronoï.

    Args:
        image: Image RGB
        centroids: Liste de (y, x) pour chaque noyau
        alpha: Transparence

    Returns:
        Image avec Voronoï
    """
    from scipy.spatial import Voronoi

    if len(centroids) < 4:
        return image.copy()

    result = image.copy()
    h, w = image.shape[:2]

    # Convertir en (x, y) pour scipy
    points = np.array([(c[1], c[0]) for c in centroids])

    try:
        vor = Voronoi(points)

        # Dessiner les arêtes
        for ridge in vor.ridge_vertices:
            if -1 not in ridge:
                v0 = vor.vertices[ridge[0]]
                v1 = vor.vertices[ridge[1]]

                # Vérifier les limites
                if (0 <= v0[0] < w and 0 <= v0[1] < h and
                    0 <= v1[0] < w and 0 <= v1[1] < h):
                    pt1 = (int(v0[0]), int(v0[1]))
                    pt2 = (int(v1[0]), int(v1[1]))
                    cv2.line(result, pt1, pt2, (100, 100, 255), 1)

    except Exception:
        pass  # Voronoï peut échouer sur certaines configurations

    return result


# =============================================================================
# PHASE 3: VISUALISATIONS INTELLIGENCE SPATIALE
# =============================================================================

def create_hotspot_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    hotspot_ids: List[int],
    color: Tuple[int, int, int] = (255, 165, 0),  # Orange
    thickness: int = 2,
    fill_alpha: float = 0.2,
) -> np.ndarray:
    """
    Crée un overlay mettant en évidence les hotspots (clusters haute densité).

    Phase 3: Les noyaux dans des zones de haute densité sont surlignés en orange.

    Args:
        image: Image RGB originale
        instance_map: Carte d'instances
        hotspot_ids: IDs des noyaux dans les hotspots
        color: Couleur de surlignage (défaut: orange)
        thickness: Épaisseur des contours
        fill_alpha: Transparence du remplissage

    Returns:
        Image avec hotspots surlignés
    """
    result = image.copy()

    if not hotspot_ids:
        return result

    # Créer masque combiné des hotspots
    hotspot_mask = np.zeros(instance_map.shape, dtype=np.uint8)
    for nid in hotspot_ids:
        hotspot_mask[instance_map == nid] = 255

    # Remplissage semi-transparent
    overlay = result.copy()
    overlay[hotspot_mask > 0] = color
    result = cv2.addWeighted(result, 1 - fill_alpha, overlay, fill_alpha, 0)

    # Contours individuels
    for nid in hotspot_ids:
        mask = (instance_map == nid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, thickness)

    return result


def create_mitosis_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    mitosis_ids: List[int],
    mitosis_scores: Dict[int, float],
    thickness: int = 2,
) -> np.ndarray:
    """
    Crée un overlay mettant en évidence les candidats mitoses.

    Phase 3: Les mitoses candidates sont surlignées avec couleur
    proportionnelle au score (jaune = probable, rouge = très probable).

    Args:
        image: Image RGB originale
        instance_map: Carte d'instances
        mitosis_ids: IDs des candidats mitoses
        mitosis_scores: Scores de mitose par ID
        thickness: Épaisseur des contours

    Returns:
        Image avec mitoses surlignées
    """
    result = image.copy()

    if not mitosis_ids:
        return result

    for nid in mitosis_ids:
        mask = (instance_map == nid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Couleur basée sur le score (jaune → rouge)
        score = mitosis_scores.get(nid, 0.5)
        r = int(255)
        g = int(255 * (1 - score))  # Plus le score est haut, moins de vert
        b = 0
        color = (r, g, b)

        cv2.drawContours(result, contours, -1, color, thickness)

        # Ajouter indicateur "M"
        coords = np.where(instance_map == nid)
        if len(coords[0]) > 0:
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            cv2.putText(result, "M", (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1)

    return result


def create_chromatin_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    heterogeneous_ids: List[int],
    color: Tuple[int, int, int] = (148, 0, 211),  # Violet
    thickness: int = 2,
) -> np.ndarray:
    """
    Crée un overlay mettant en évidence les noyaux à chromatine hétérogène.

    Phase 3: Chromatine hétérogène = signe potentiel de malignité.

    Args:
        image: Image RGB originale
        instance_map: Carte d'instances
        heterogeneous_ids: IDs des noyaux à chromatine hétérogène
        color: Couleur de surlignage (défaut: violet)
        thickness: Épaisseur des contours

    Returns:
        Image avec noyaux hétérogènes surlignés
    """
    result = image.copy()

    for nid in heterogeneous_ids:
        mask = (instance_map == nid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, thickness)

        # Ajouter indicateur "C" (Chromatin)
        coords = np.where(instance_map == nid)
        if len(coords[0]) > 0:
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            cv2.putText(result, "C", (cx - 4, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                       0.3, color, 1)

    return result


def create_pleomorphism_badge(
    score: int,
    description: str,
    size: Tuple[int, int] = (200, 80),
) -> np.ndarray:
    """
    Crée un badge visuel pour le score de pléomorphisme.

    Args:
        score: Score 1-3
        description: Description textuelle
        size: Taille du badge (w, h)

    Returns:
        Image RGB du badge
    """
    w, h = size

    # Couleurs selon score
    colors = {
        1: (144, 238, 144),   # Vert clair
        2: (255, 215, 0),     # Or
        3: (255, 99, 71),     # Rouge tomate
    }
    bg_color = colors.get(score, (200, 200, 200))

    # Créer image
    badge = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Score au centre
    score_text = f"{score}/3"
    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    tx = (w - text_size[0]) // 2
    ty = h // 2 + 10
    cv2.putText(badge, score_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Label en haut
    cv2.putText(badge, "Pleomorphisme", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return badge


def create_spatial_debug_panel(
    pleomorphism_score: int,
    pleomorphism_description: str,
    n_hotspots: int,
    n_mitosis_candidates: int,
    n_heterogeneous: int,
    mean_neighbors: float,
    mean_entropy: float,
    figsize: Tuple[int, int] = (12, 3),
) -> np.ndarray:
    """
    Crée un panneau de debug pour l'analyse spatiale Phase 3.

    Args:
        pleomorphism_score: Score 1-3
        pleomorphism_description: Description
        n_hotspots: Nombre de hotspots
        n_mitosis_candidates: Nombre de candidats mitoses
        n_heterogeneous: Nombre de noyaux à chromatine hétérogène
        mean_neighbors: Moyenne voisins Voronoï
        mean_entropy: Entropie moyenne chromatine

    Returns:
        Image RGB du panneau
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize, facecolor='white',
                             gridspec_kw={'width_ratios': [1, 1, 1, 1.2]})

    # 1. Score Pléomorphisme (gauge)
    ax1 = axes[0]
    ax1.axis('off')
    colors = {1: 'green', 2: 'orange', 3: 'red'}
    labels = {1: 'Faible', 2: 'Modéré', 3: 'Sévère'}

    ax1.text(0.5, 0.8, "PLÉOMORPHISME", ha='center', fontsize=10, fontweight='bold',
             transform=ax1.transAxes)
    ax1.text(0.5, 0.5, f"{pleomorphism_score}/3", ha='center', fontsize=28,
             color=colors.get(pleomorphism_score, 'gray'), fontweight='bold',
             transform=ax1.transAxes)
    ax1.text(0.5, 0.2, labels.get(pleomorphism_score, ''), ha='center', fontsize=10,
             color=colors.get(pleomorphism_score, 'gray'),
             transform=ax1.transAxes)

    # 2. Hotspots & Clustering
    ax2 = axes[1]
    ax2.axis('off')
    ax2.text(0.5, 0.85, "CLUSTERING", ha='center', fontsize=10, fontweight='bold',
             transform=ax2.transAxes)

    hotspot_text = f"Hotspots: {n_hotspots}"
    hotspot_color = 'orange' if n_hotspots > 0 else 'green'
    ax2.text(0.5, 0.55, hotspot_text, ha='center', fontsize=12,
             color=hotspot_color, transform=ax2.transAxes)

    neighbors_text = f"Voisins moy.: {mean_neighbors:.1f}"
    ax2.text(0.5, 0.25, neighbors_text, ha='center', fontsize=10,
             color='gray', transform=ax2.transAxes)

    # 3. Mitoses & Chromatine
    ax3 = axes[2]
    ax3.axis('off')
    ax3.text(0.5, 0.85, "BIOMARQUEURS", ha='center', fontsize=10, fontweight='bold',
             transform=ax3.transAxes)

    mitosis_color = 'red' if n_mitosis_candidates > 3 else ('orange' if n_mitosis_candidates > 0 else 'green')
    ax3.text(0.5, 0.55, f"Mitoses susp.: {n_mitosis_candidates}", ha='center', fontsize=11,
             color=mitosis_color, transform=ax3.transAxes)

    chromatin_color = 'purple' if n_heterogeneous > 5 else 'gray'
    ax3.text(0.5, 0.25, f"Chrom. hétér.: {n_heterogeneous}", ha='center', fontsize=10,
             color=chromatin_color, transform=ax3.transAxes)

    # 4. Légende Phase 3
    ax4 = axes[3]
    ax4.axis('off')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    legend_text = """PHASE 3 - Intelligence Spatiale
--------------------------------
[H] Hotspots = zones haute densité
[M] Mitoses = forme + chromatine
[C] Chromatine hétérogène
[P] Pléomorphisme = anisocaryose
"""
    ax4.text(0.05, 0.9, legend_text, transform=ax4.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

    plt.tight_layout()

    # Convertir en image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    plt.close(fig)
    buf.seek(0)

    img = plt.imread(buf)
    buf.close()

    return (img[:, :, :3] * 255).astype(np.uint8)


def create_phase3_combined_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    hotspot_ids: List[int],
    mitosis_ids: List[int],
    mitosis_scores: Dict[int, float],
    heterogeneous_ids: List[int],
    show_hotspots: bool = True,
    show_mitoses: bool = True,
    show_chromatin: bool = True,
) -> np.ndarray:
    """
    Crée un overlay combiné de toutes les annotations Phase 3.

    Args:
        image: Image RGB originale
        instance_map: Carte d'instances
        hotspot_ids: IDs des noyaux dans hotspots
        mitosis_ids: IDs des candidats mitoses
        mitosis_scores: Scores mitose
        heterogeneous_ids: IDs chromatine hétérogène
        show_hotspots: Afficher hotspots
        show_mitoses: Afficher mitoses
        show_chromatin: Afficher chromatine

    Returns:
        Image avec toutes les annotations Phase 3
    """
    result = image.copy()

    if show_hotspots and hotspot_ids:
        result = create_hotspot_overlay(result, instance_map, hotspot_ids)

    if show_chromatin and heterogeneous_ids:
        result = create_chromatin_overlay(result, instance_map, heterogeneous_ids)

    if show_mitoses and mitosis_ids:
        result = create_mitosis_overlay(result, instance_map, mitosis_ids, mitosis_scores)

    return result
