#!/usr/bin/env python3
"""
Module de visualisation des segmentations cellulaires.

Fonctions pour superposer les masques colorés sur les images
et créer des visualisations informatives.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional


# Palette de couleurs pour les types cellulaires
CELL_COLORS = {
    "Background": (0, 0, 0),
    "Neoplastic": (255, 50, 50),      # Rouge vif - tumeur
    "Inflammatory": (50, 255, 50),     # Vert vif - inflammation
    "Connective": (50, 50, 255),       # Bleu vif - conjonctif
    "Dead": (255, 255, 50),            # Jaune - nécrose
    "Epithelial": (50, 255, 255),      # Cyan - épithélium
}

CELL_TYPE_INDICES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convertit un masque 6-canaux en image RGB colorée.

    Args:
        mask: Masque (H, W, 6) avec les canaux par type cellulaire

    Returns:
        Image RGB (H, W, 3) avec les cellules colorées
    """
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for idx in range(1, 6):  # Skip background
        cell_type = CELL_TYPE_INDICES[idx]
        color = CELL_COLORS[cell_type]
        cell_mask = mask[:, :, idx] > 0
        rgb[cell_mask] = color

    return rgb


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Superpose le masque coloré sur l'image originale.

    Args:
        image: Image originale (H, W, 3)
        mask: Masque 6-canaux (H, W, 6)
        alpha: Transparence de l'overlay (0-1)

    Returns:
        Image avec overlay
    """
    mask_rgb = mask_to_rgb(mask)

    # Créer l'overlay uniquement où il y a des cellules
    has_cells = np.any(mask[:, :, 1:] > 0, axis=2)

    result = image.copy()
    result[has_cells] = cv2.addWeighted(
        image[has_cells],
        1 - alpha,
        mask_rgb[has_cells],
        alpha,
        0
    )

    return result


def draw_contours(
    image: np.ndarray,
    mask: np.ndarray,
    thickness: int = 1
) -> np.ndarray:
    """
    Dessine les contours des cellules sur l'image.

    Args:
        image: Image originale
        mask: Masque 6-canaux
        thickness: Épaisseur des contours

    Returns:
        Image avec contours
    """
    result = image.copy()

    for idx in range(1, 6):
        cell_type = CELL_TYPE_INDICES[idx]
        color = CELL_COLORS[cell_type]
        cell_mask = mask[:, :, idx].astype(np.uint8)

        contours, _ = cv2.findContours(
            cell_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(result, contours, -1, color, thickness)

    return result


def count_cells(mask: np.ndarray) -> Dict[str, int]:
    """
    Compte le nombre de cellules de chaque type.

    Args:
        mask: Masque 6-canaux

    Returns:
        Dictionnaire avec les comptages
    """
    counts = {}

    for idx in range(1, 6):
        cell_type = CELL_TYPE_INDICES[idx]
        cell_mask = mask[:, :, idx].astype(np.uint8)

        # Trouver les composantes connexes
        n_labels, _ = cv2.connectedComponents(cell_mask)
        counts[cell_type] = n_labels - 1  # -1 pour exclure le fond

    return counts


def create_legend(
    height: int = 300,
    width: int = 200,
    counts: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Crée une image de légende avec les couleurs et comptages.

    Args:
        height: Hauteur de l'image
        width: Largeur de l'image
        counts: Comptages optionnels

    Returns:
        Image RGB de la légende
    """
    legend = np.full((height, width, 3), 255, dtype=np.uint8)

    y_offset = 30
    for idx in range(1, 6):
        cell_type = CELL_TYPE_INDICES[idx]
        color = CELL_COLORS[cell_type]

        # Carré de couleur
        cv2.rectangle(
            legend,
            (20, y_offset - 15),
            (40, y_offset + 5),
            color[::-1],  # BGR pour OpenCV
            -1
        )

        # Texte
        text = cell_type
        if counts and cell_type in counts:
            text += f": {counts[cell_type]}"

        cv2.putText(
            legend,
            text,
            (50, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

        y_offset += 45

    return legend


def create_comparison_figure(
    image: np.ndarray,
    mask: np.ndarray,
    tissue_type: str = "Unknown"
) -> np.ndarray:
    """
    Crée une figure comparative: original | overlay | contours.

    Args:
        image: Image originale
        mask: Masque 6-canaux
        tissue_type: Type de tissu

    Returns:
        Image composite
    """
    h, w = image.shape[:2]

    # Créer les trois vues
    overlay = overlay_mask(image, mask, alpha=0.4)
    contours = draw_contours(image.copy(), mask, thickness=2)

    # Comptages
    counts = count_cells(mask)
    total = sum(counts.values())

    # Assembler horizontalement
    combined = np.hstack([image, overlay, contours])

    # Ajouter barre de titre
    title_bar = np.full((60, combined.shape[1], 3), 40, dtype=np.uint8)

    # Titre
    title = f"Tissue: {tissue_type} | Total Cells: {total}"
    cv2.putText(
        title_bar,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Labels sous les images
    labels = ["Original", "Overlay", "Contours"]
    for i, label in enumerate(labels):
        x = i * w + w // 2 - 30
        cv2.putText(
            title_bar,
            label,
            (x, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )

    result = np.vstack([title_bar, combined])

    return result


def generate_report(
    mask: np.ndarray,
    tissue_type: str = "Unknown"
) -> str:
    """
    Génère un rapport textuel sur la composition cellulaire.

    Args:
        mask: Masque 6-canaux
        tissue_type: Type de tissu

    Returns:
        Rapport formaté
    """
    counts = count_cells(mask)
    total = sum(counts.values())

    report = f"""
╔══════════════════════════════════════════╗
║      RAPPORT D'ANALYSE CELLULAIRE        ║
╠══════════════════════════════════════════╣
║  Tissu: {tissue_type:<31} ║
╠══════════════════════════════════════════╣
"""

    for cell_type, count in counts.items():
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        report += f"║  {cell_type:<12}: {count:>4} ({pct:>5.1f}%) {bar} ║\n"

    report += f"""╠══════════════════════════════════════════╣
║  TOTAL: {total:<32} ║
╚══════════════════════════════════════════╝
"""

    # Analyse automatique
    if total > 0:
        neo_pct = counts.get("Neoplastic", 0) / total * 100
        inf_pct = counts.get("Inflammatory", 0) / total * 100
        dead_pct = counts.get("Dead", 0) / total * 100

        report += "\n📋 OBSERVATIONS:\n"

        if neo_pct > 25:
            report += "  ⚠️  Proportion élevée de cellules néoplasiques\n"
        if inf_pct > 30:
            report += "  ⚠️  Infiltration inflammatoire importante\n"
        if dead_pct > 10:
            report += "  ⚠️  Présence notable de nécrose\n"

        if neo_pct <= 25 and inf_pct <= 30 and dead_pct <= 10:
            report += "  ✅ Composition cellulaire dans les normes\n"

    return report


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Charger les données de démo
    demo_dir = Path("data/demo")
    if demo_dir.exists():
        images = np.load(demo_dir / "images.npy")
        masks = np.load(demo_dir / "masks.npy")
        types = np.load(demo_dir / "types.npy")

        print(f"Chargé {len(images)} images")

        # Visualiser la première image
        fig = create_comparison_figure(images[0], masks[0], types[0])
        plt.figure(figsize=(15, 6))
        plt.imshow(fig)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("data/demo/visualization_example.png", dpi=150)
        print("✓ Exemple sauvegardé: data/demo/visualization_example.png")

        # Rapport
        print(generate_report(masks[0], types[0]))
    else:
        print("Dataset de démo non trouvé. Exécuter d'abord synthetic_cells.py")
