#!/usr/bin/env python3
"""
DIAGNOSTIC VISUEL: Vérifier que les vecteurs HV pointent vers les centroïdes

Objectif:
- Charger une image PanNuke
- Générer HV maps avec compute_hv_maps() v7
- Pour chaque instance, vérifier visuellement que les vecteurs HV pointent vers le centroïde

Si les vecteurs NE POINTENT PAS vers le centroïde → Bug dans compute_hv_maps()
Si les vecteurs POINTENT vers le centroïde → Bug ailleurs (matching, indexation, etc.)

Usage:
    python scripts/validation/diagnose_hv_vectors_visual.py \
        --family epidermal \
        --sample_idx 0
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE


# =============================================================================
# EXTRACT PANNUKE INSTANCES (NATIVE IDS)
# =============================================================================

def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """
    Extrait les vraies instances de PanNuke avec IDs séparés.

    Args:
        mask: Mask PanNuke (H, W, 6)

    Returns:
        inst_map: Instance map (H, W) avec IDs séparés (0 = background)
    """
    inst_map = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: IDs d'instances natifs
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5: Epithelial (binaire)
    epithelial_mask = mask[:, :, 5]
    if epithelial_mask.max() > 0:
        _, epithelial_labels = cv2.connectedComponents(epithelial_mask.astype(np.uint8))
        epithelial_ids = np.unique(epithelial_labels)
        epithelial_ids = epithelial_ids[epithelial_ids > 0]

        for epi_id in epithelial_ids:
            epi_mask = epithelial_labels == epi_id
            inst_map[epi_mask] = instance_counter
            instance_counter += 1

    return inst_map


# =============================================================================
# COMPUTE HV MAPS v7 (RADIAL NORMALIZATION)
# =============================================================================

def compute_hv_maps_v7(inst_map: np.ndarray) -> np.ndarray:
    """
    Calcule les cartes HV avec NORMALISATION RADIALE (v7).

    IDENTIQUE à prepare_family_data_FIXED_v7.py
    """
    h, w = inst_map.shape
    hv_map = np.zeros((2, h, w), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) == 0:
            continue

        # Calculer le centroïde
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)

        # Distance depuis centroïde
        y_dist = y_coords - center_y
        x_dist = x_coords - center_x

        # Normalisation RADIALE (v7)
        dist_max = np.max(np.sqrt(y_dist**2 + x_dist**2)) + 1e-7
        v_dist = y_dist / dist_max
        h_dist = x_dist / dist_max

        # Clip
        v_dist = np.clip(v_dist, -1.0, 1.0)
        h_dist = np.clip(h_dist, -1.0, 1.0)

        # Convention: hv_map[0] = V, hv_map[1] = H
        hv_map[0, y_coords, x_coords] = v_dist
        hv_map[1, y_coords, x_coords] = h_dist

    # Gaussian smoothing
    hv_map[0] = gaussian_filter(hv_map[0], sigma=0.5)
    hv_map[1] = gaussian_filter(hv_map[1], sigma=0.5)

    return hv_map


# =============================================================================
# VISUAL DIAGNOSTIC
# =============================================================================

def diagnose_hv_vectors(
    image: np.ndarray,
    inst_map: np.ndarray,
    hv_map: np.ndarray,
    max_instances: int = 5
):
    """
    Diagnostic visuel des vecteurs HV.

    Pour chaque instance:
    1. Calculer centroïde réel
    2. Échantillonner pixels au bord
    3. Tracer vecteurs HV depuis ces pixels
    4. Vérifier visuellement s'ils pointent vers le centroïde

    Args:
        image: Image RGB (H, W, 3)
        inst_map: Instance map (H, W)
        hv_map: HV maps (2, H, W)
        max_instances: Nombre max d'instances à afficher
    """
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0][:max_instances]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Subplot 1: Image + Centroïdes GT
    axes[0].imshow(image)
    axes[0].set_title("Image + Centroïdes GT (RÉELS)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Subplot 2: Image + Vecteurs HV
    axes[1].imshow(image)
    axes[1].set_title("Image + Vecteurs HV (v7 radial)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) < 5:
            continue

        # Centroïde RÉEL
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)

        # Marquer centroïde sur subplot 1
        axes[0].scatter([center_x], [center_y], c='red', s=100, marker='x', linewidths=3)
        axes[0].text(center_x + 5, center_y - 5, f'#{inst_id}', color='red', fontsize=10, fontweight='bold')

        # Marquer centroïde sur subplot 2
        axes[1].scatter([center_x], [center_y], c='red', s=100, marker='x', linewidths=3)

        # Échantillonner pixels au BORD de l'instance
        # Stratégie: prendre 8 pixels répartis uniformément
        n_samples = 8
        sample_indices = np.linspace(0, len(y_coords) - 1, n_samples, dtype=int)

        for idx in sample_indices:
            py, px = y_coords[idx], x_coords[idx]

            # Vecteur HV à ce pixel
            v_val = hv_map[0, py, px]  # Vertical
            h_val = hv_map[1, py, px]  # Horizontal

            # Tracer vecteur depuis (px, py)
            # Le vecteur HV (h_val, v_val) devrait pointer vers le centroïde
            # Magnitude arbitraire pour visualisation (scaling factor)
            scale = 30.0

            dx = h_val * scale  # Horizontal displacement
            dy = v_val * scale  # Vertical displacement

            # Arrow: (px, py) → (px + dx, py + dy)
            axes[1].arrow(
                px, py,
                dx, dy,
                color='lime',
                width=1.5,
                head_width=5,
                head_length=5,
                alpha=0.8
            )

            # Marquer point de départ
            axes[1].scatter([px], [py], c='yellow', s=30, marker='o', alpha=0.7)

    plt.tight_layout()
    plt.savefig('results/hv_vectors_diagnostic.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Diagnostic sauvegardé: results/hv_vectors_diagnostic.png")
    plt.close()


# =============================================================================
# QUANTITATIVE CHECK
# =============================================================================

def quantitative_check(inst_map: np.ndarray, hv_map: np.ndarray):
    """
    Vérification QUANTITATIVE: Les vecteurs HV pointent-ils vers le centroïde?

    Pour chaque instance:
    1. Calculer centroïde réel
    2. Pour chaque pixel de l'instance:
       - Calculer vecteur attendu: (center - pixel) normalisé
       - Comparer avec vecteur HV stocké
       - Calculer erreur angulaire

    Si erreur angulaire moyenne > 10° → Bug dans compute_hv_maps()
    """
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    all_errors = []

    print("\n" + "=" * 80)
    print("VÉRIFICATION QUANTITATIVE: DIRECTION DES VECTEURS HV")
    print("=" * 80)

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) < 5:
            continue

        # Centroïde RÉEL
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)

        errors_inst = []

        for py, px in zip(y_coords, x_coords):
            # Vecteur ATTENDU: (center - pixel) normalisé
            expected_dy = center_y - py
            expected_dx = center_x - px
            expected_mag = np.sqrt(expected_dy**2 + expected_dx**2)

            if expected_mag < 1e-6:
                continue  # Pixel au centroïde

            expected_dy /= expected_mag
            expected_dx /= expected_mag

            # Vecteur HV STOCKÉ
            v_val = hv_map[0, py, px]
            h_val = hv_map[1, py, px]
            stored_mag = np.sqrt(v_val**2 + h_val**2)

            if stored_mag < 1e-6:
                continue  # Vecteur nul

            # Normaliser
            v_norm = v_val / stored_mag
            h_norm = h_val / stored_mag

            # Produit scalaire: dot = cos(angle)
            dot = expected_dy * v_norm + expected_dx * h_norm
            dot = np.clip(dot, -1.0, 1.0)

            # Erreur angulaire (degrés)
            angle_error = np.rad2deg(np.arccos(dot))
            errors_inst.append(angle_error)

        if len(errors_inst) > 0:
            mean_error = np.mean(errors_inst)
            all_errors.extend(errors_inst)

            status = "✅ OK" if mean_error < 10 else "❌ BUG"
            print(f"  Instance #{inst_id:3d}: Erreur angulaire = {mean_error:6.2f}° {status}")

    if len(all_errors) > 0:
        global_mean = np.mean(all_errors)
        global_std = np.std(all_errors)

        print()
        print("=" * 80)
        print(f"ERREUR ANGULAIRE GLOBALE: {global_mean:.2f}° ± {global_std:.2f}°")

        if global_mean < 5:
            print("✅ EXCELLENT - Vecteurs HV alignés avec centroïdes")
        elif global_mean < 10:
            print("🟡 ACCEPTABLE - Légère dérive angulaire")
        elif global_mean < 45:
            print("⚠️  PROBLÉMATIQUE - Dérive angulaire significative")
        else:
            print("❌ BUG CRITIQUE - Vecteurs HV pointent dans MAUVAISE direction!")

        print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnostic visuel des vecteurs HV")
    parser.add_argument("--family", type=str, required=True, help="Famille à diagnostiquer")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index échantillon à visualiser")
    parser.add_argument(
        "--pannuke_dir",
        type=str,
        default="/home/amar/data/PanNuke",
        help="Répertoire PanNuke source"
    )

    args = parser.parse_args()

    # Load NPZ
    data_file = Path(f"data/family_FIXED/{args.family}_data_FIXED.npz")
    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return 1

    data = np.load(data_file)
    fold_id = data['fold_ids'][args.sample_idx]
    image_id = data['image_ids'][args.sample_idx]

    print(f"\nÉchantillon: {args.sample_idx} (fold{fold_id}, img{image_id})")

    # Load REAL PanNuke data
    pannuke_dir = Path(args.pannuke_dir)
    fold_dir = pannuke_dir / f"fold{fold_id}"

    images_file = fold_dir / "images.npy"
    masks_file = fold_dir / "masks.npy"

    if not images_file.exists():
        print(f"❌ Fichier non trouvé: {images_file}")
        return 1

    images = np.load(images_file, mmap_mode='r')
    masks = np.load(masks_file, mmap_mode='r')

    image = images[image_id]
    mask = masks[image_id]

    # Generate inst_map and HV maps with v7
    inst_map = extract_pannuke_instances(mask)
    hv_map = compute_hv_maps_v7(inst_map)

    print(f"Instances trouvées: {len(np.unique(inst_map)) - 1}")
    print(f"HV map range: [{hv_map.min():.4f}, {hv_map.max():.4f}]")

    # Visual diagnostic
    Path("results").mkdir(exist_ok=True)
    diagnose_hv_vectors(image, inst_map, hv_map, max_instances=10)

    # Quantitative check
    quantitative_check(inst_map, hv_map)

    return 0


if __name__ == "__main__":
    exit(main())
