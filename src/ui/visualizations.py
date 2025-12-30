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

    # Blend
    mask_any = instance_map > 0
    result[mask_any] = cv2.addWeighted(
        image[mask_any],
        1 - alpha,
        overlay[mask_any],
        alpha,
        0
    )

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
