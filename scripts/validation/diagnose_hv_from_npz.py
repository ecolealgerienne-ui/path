#!/usr/bin/env python3
"""
DIAGNOSTIC HV DEPUIS NPZ - Test les données RÉELLEMENT sauvegardées

CRITIQUE: Ce script charge les HV maps depuis le NPZ au lieu de les recalculer.
Cela permet de vérifier que les données v8 ont été correctement générées.

Usage:
    python scripts/validation/diagnose_hv_from_npz.py \
        --family epidermal \
        --sample_idx 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE


# =============================================================================
# VISUAL DIAGNOSTIC
# =============================================================================

def diagnose_hv_vectors(
    image: np.ndarray,
    hv_map: np.ndarray,
    np_target: np.ndarray,
    max_instances: int = 10
):
    """
    Diagnostic visuel des vecteurs HV DEPUIS NPZ.

    Args:
        image: Image RGB (H, W, 3)
        hv_map: HV maps (2, H, W) - CHARGÉ DEPUIS NPZ
        np_target: NP target (H, W) - Pour identifier instances
        max_instances: Nombre max d'instances à afficher
    """
    import cv2
    from scipy.ndimage import label

    # Créer instance map depuis np_target
    labeled_mask, n_instances = label(np_target > 0)
    inst_ids = np.unique(labeled_mask)
    inst_ids = inst_ids[inst_ids > 0][:max_instances]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Subplot 1: Image + Centroïdes GT
    axes[0].imshow(image)
    axes[0].set_title("Image + Centroïdes GT (RÉELS)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Subplot 2: Image + Vecteurs HV
    axes[1].imshow(image)
    axes[1].set_title("Image + Vecteurs HV (DEPUIS NPZ v8)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    for inst_id in inst_ids:
        inst_mask = labeled_mask == inst_id
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

        # Échantillonner pixels au BORD
        n_samples = 8
        sample_indices = np.linspace(0, len(y_coords) - 1, n_samples, dtype=int)

        for idx in sample_indices:
            py, px = y_coords[idx], x_coords[idx]

            # Vecteur HV à ce pixel (CHARGÉ DEPUIS NPZ)
            v_val = hv_map[0, py, px]  # Vertical
            h_val = hv_map[1, py, px]  # Horizontal

            # Tracer vecteur depuis (px, py)
            scale = 30.0

            dx = h_val * scale
            dy = v_val * scale

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
    plt.savefig('results/hv_vectors_from_npz.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Diagnostic sauvegardé: results/hv_vectors_from_npz.png")
    plt.close()


# =============================================================================
# QUANTITATIVE CHECK
# =============================================================================

def quantitative_check(hv_map: np.ndarray, np_target: np.ndarray):
    """
    Vérification QUANTITATIVE: Les vecteurs HV pointent-ils vers le centroïde?

    Args:
        hv_map: HV maps (2, H, W) - CHARGÉ DEPUIS NPZ
        np_target: NP target (H, W)
    """
    from scipy.ndimage import label

    # Créer instance map depuis np_target
    labeled_mask, n_instances = label(np_target > 0)
    inst_ids = np.unique(labeled_mask)
    inst_ids = inst_ids[inst_ids > 0]

    all_errors = []

    print("\n" + "=" * 80)
    print("VÉRIFICATION QUANTITATIVE: HV DEPUIS NPZ v8")
    print("=" * 80)

    for inst_id in inst_ids:
        inst_mask = labeled_mask == inst_id
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
                continue

            expected_dy /= expected_mag
            expected_dx /= expected_mag

            # Vecteur HV STOCKÉ (DEPUIS NPZ)
            v_val = hv_map[0, py, px]
            h_val = hv_map[1, py, px]
            stored_mag = np.sqrt(v_val**2 + h_val**2)

            if stored_mag < 1e-6:
                continue

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
        print(f"ERREUR ANGULAIRE GLOBALE (NPZ v8): {global_mean:.2f}° ± {global_std:.2f}°")

        if global_mean < 5:
            print("✅ EXCELLENT - Vecteurs HV alignés avec centroïdes (NPZ v8 CORRECT)")
        elif global_mean < 10:
            print("🟡 ACCEPTABLE - Légère dérive angulaire")
        elif global_mean < 45:
            print("⚠️  PROBLÉMATIQUE - Dérive angulaire significative")
        else:
            print("❌ BUG CRITIQUE - NPZ contient encore données v7 (centrifuges)!")
            print("   → Vérifier que prepare_family_data_FIXED_v8.py a bien été exécuté")

        print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnostic HV depuis NPZ v8")
    parser.add_argument("--family", type=str, required=True, help="Famille à diagnostiquer")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index échantillon")

    args = parser.parse_args()

    # Load NPZ
    data_file = Path(f"data/family_FIXED/{args.family}_data_FIXED.npz")
    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return 1

    data = np.load(data_file)

    image = data['images'][args.sample_idx]
    hv_target = data['hv_targets'][args.sample_idx]  # ✅ CHARGÉ DEPUIS NPZ
    np_target = data['np_targets'][args.sample_idx]
    fold_id = data['fold_ids'][args.sample_idx]
    image_id = data['image_ids'][args.sample_idx]

    print(f"\nÉchantillon: {args.sample_idx} (fold{fold_id}, img{image_id})")
    print(f"HV map range: [{hv_target.min():.4f}, {hv_target.max():.4f}]")

    # Vérifier timestamp NPZ
    import datetime
    timestamp = datetime.datetime.fromtimestamp(data_file.stat().st_mtime)
    print(f"NPZ créé: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Visual diagnostic
    Path("results").mkdir(exist_ok=True)
    diagnose_hv_vectors(image, hv_target, np_target, max_instances=10)

    # Quantitative check
    quantitative_check(hv_target, np_target)

    return 0


if __name__ == "__main__":
    exit(main())
