#!/usr/bin/env python3
"""
Générateur de données cellulaires synthétiques pour démonstration.

Crée des images et masques simulant des tissus histopathologiques
avec différents types de cellules colorées.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
import cv2


# Types de cellules PanNuke
CELL_TYPES = {
    0: ("Background", (0, 0, 0)),
    1: ("Neoplastic", (255, 0, 0)),      # Rouge - cellules tumorales
    2: ("Inflammatory", (0, 255, 0)),     # Vert - cellules inflammatoires
    3: ("Connective", (0, 0, 255)),       # Bleu - tissu conjonctif
    4: ("Dead", (255, 255, 0)),           # Jaune - cellules mortes
    5: ("Epithelial", (0, 255, 255)),     # Cyan - cellules épithéliales
}

# Configurations de tissus (proportions de chaque type)
TISSUE_CONFIGS = {
    "Breast": {"Neoplastic": 0.3, "Epithelial": 0.25, "Connective": 0.25, "Inflammatory": 0.15, "Dead": 0.05},
    "Colon": {"Epithelial": 0.35, "Connective": 0.25, "Inflammatory": 0.2, "Neoplastic": 0.15, "Dead": 0.05},
    "Lung": {"Neoplastic": 0.25, "Inflammatory": 0.3, "Epithelial": 0.2, "Connective": 0.2, "Dead": 0.05},
    "Kidney": {"Epithelial": 0.4, "Connective": 0.3, "Inflammatory": 0.15, "Neoplastic": 0.1, "Dead": 0.05},
    "Liver": {"Epithelial": 0.35, "Connective": 0.2, "Inflammatory": 0.25, "Neoplastic": 0.15, "Dead": 0.05},
    "Prostate": {"Epithelial": 0.3, "Neoplastic": 0.25, "Connective": 0.25, "Inflammatory": 0.15, "Dead": 0.05},
    "Skin": {"Epithelial": 0.4, "Connective": 0.2, "Inflammatory": 0.2, "Neoplastic": 0.15, "Dead": 0.05},
    "Stomach": {"Epithelial": 0.35, "Inflammatory": 0.25, "Connective": 0.2, "Neoplastic": 0.15, "Dead": 0.05},
}


def generate_cell_centers(
    img_size: int = 256,
    n_cells: int = 50,
    min_dist: int = 15
) -> np.ndarray:
    """Génère des centres de cellules avec espacement minimum."""
    centers = []
    margin = 20
    max_attempts = 1000

    for _ in range(n_cells):
        for attempt in range(max_attempts):
            x = np.random.randint(margin, img_size - margin)
            y = np.random.randint(margin, img_size - margin)

            # Vérifier la distance avec les autres centres
            valid = True
            for cx, cy in centers:
                if np.sqrt((x - cx)**2 + (y - cy)**2) < min_dist:
                    valid = False
                    break

            if valid:
                centers.append((x, y))
                break

    return np.array(centers)


def generate_cell_mask(
    center: Tuple[int, int],
    img_size: int = 256,
    base_radius: int = 8
) -> np.ndarray:
    """Génère un masque pour une cellule avec forme irrégulière."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Rayon variable
    radius = base_radius + np.random.randint(-2, 4)

    # Forme elliptique avec légère rotation
    axes = (radius, int(radius * (0.7 + 0.6 * np.random.random())))
    angle = np.random.randint(0, 180)

    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

    return mask


def generate_synthetic_tissue(
    tissue_type: str = "Breast",
    img_size: int = 256,
    n_cells: int = 50,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Génère une image et un masque de tissu synthétique.

    Returns:
        image: Image RGB simulant une coloration H&E
        mask: Masque avec 6 canaux (background + 5 types)
        info: Dictionnaire avec les statistiques
    """
    if seed is not None:
        np.random.seed(seed)

    config = TISSUE_CONFIGS.get(tissue_type, TISSUE_CONFIGS["Breast"])

    # Génération du fond (couleur H&E)
    # Rose pâle avec variations
    bg_r = np.random.randint(200, 240)
    bg_g = np.random.randint(180, 210)
    bg_b = np.random.randint(200, 230)

    image = np.full((img_size, img_size, 3), [bg_r, bg_g, bg_b], dtype=np.uint8)

    # Ajouter texture de fond
    noise = np.random.normal(0, 5, (img_size, img_size, 3))
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    # Masque multi-canal (6 canaux: bg + 5 types)
    mask = np.zeros((img_size, img_size, 6), dtype=np.uint8)
    mask[:, :, 0] = 255  # Background par défaut

    # Générer les centres de cellules
    centers = generate_cell_centers(img_size, n_cells)

    # Assigner les types selon la configuration
    type_names = list(config.keys())
    type_probs = [config[t] for t in type_names]

    cell_stats = {name: 0 for name in CELL_TYPES.values()}
    cell_stats = {"Neoplastic": 0, "Inflammatory": 0, "Connective": 0, "Dead": 0, "Epithelial": 0}

    for cx, cy in centers:
        # Choisir le type
        cell_type_name = np.random.choice(type_names, p=type_probs)
        type_idx = list(CELL_TYPES.keys())[list(n for n, _ in CELL_TYPES.values()).index(cell_type_name)]

        # Générer le masque de la cellule
        cell_mask = generate_cell_mask((int(cx), int(cy)), img_size)

        # Mettre à jour le masque multi-canal
        mask[:, :, 0][cell_mask > 0] = 0  # Retirer du background
        mask[:, :, type_idx][cell_mask > 0] = 255

        # Colorer l'image
        _, color = CELL_TYPES[type_idx]

        # Couleur H&E simulée selon le type
        if cell_type_name == "Neoplastic":
            # Noyaux foncés, violets
            cell_color = (120 + np.random.randint(-20, 20),
                         80 + np.random.randint(-20, 20),
                         140 + np.random.randint(-20, 20))
        elif cell_type_name == "Inflammatory":
            # Petits, ronds, bleu-violet foncé
            cell_color = (100 + np.random.randint(-15, 15),
                         90 + np.random.randint(-15, 15),
                         150 + np.random.randint(-15, 15))
        elif cell_type_name == "Connective":
            # Allongés, rose pâle
            cell_color = (180 + np.random.randint(-20, 20),
                         140 + np.random.randint(-20, 20),
                         170 + np.random.randint(-20, 20))
        elif cell_type_name == "Dead":
            # Fragmentés, grisâtres
            cell_color = (160 + np.random.randint(-20, 20),
                         155 + np.random.randint(-20, 20),
                         160 + np.random.randint(-20, 20))
        else:  # Epithelial
            # Organisés, violet moyen
            cell_color = (140 + np.random.randint(-20, 20),
                         100 + np.random.randint(-20, 20),
                         160 + np.random.randint(-20, 20))

        image[cell_mask > 0] = cell_color
        cell_stats[cell_type_name] += 1

    # Léger flou pour réalisme
    image = cv2.GaussianBlur(image, (3, 3), 0)

    info = {
        "tissue_type": tissue_type,
        "n_cells": len(centers),
        "cell_counts": cell_stats
    }

    return image, mask, info


def create_demo_dataset(
    output_dir: str = "data/demo",
    n_images: int = 20,
    seed: int = 42
) -> None:
    """Crée un dataset de démonstration."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    images = []
    masks = []
    types = []

    tissue_types = list(TISSUE_CONFIGS.keys())

    for i in range(n_images):
        tissue = tissue_types[i % len(tissue_types)]
        image, mask, info = generate_synthetic_tissue(
            tissue_type=tissue,
            n_cells=40 + np.random.randint(0, 30),
            seed=seed + i
        )

        images.append(image)
        masks.append(mask)
        types.append(tissue)

        print(f"  Image {i+1}/{n_images}: {tissue} - {info['n_cells']} cellules")

    # Sauvegarder
    images = np.array(images)
    masks = np.array(masks)
    types = np.array(types)

    np.save(output_path / "images.npy", images)
    np.save(output_path / "masks.npy", masks)
    np.save(output_path / "types.npy", types)

    print(f"\n✓ Dataset sauvegardé dans {output_path}")
    print(f"  - images.npy: {images.shape}")
    print(f"  - masks.npy: {masks.shape}")
    print(f"  - types.npy: {types.shape}")


if __name__ == "__main__":
    print("Génération du dataset de démonstration...")
    create_demo_dataset(n_images=24)
