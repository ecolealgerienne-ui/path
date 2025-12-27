#!/usr/bin/env python3
"""
Validation du recalcul des centres HV locaux pour crops 224×224.

Ce script vérifie que la fonction extract_crop() recalcule correctement
les HV targets pour pointer vers les centres LOCAUX (dans le crop) au lieu
des centres GLOBAUX (dans l'image source 256×256).

Usage:
    # Validation rapide (sans visualisations)
    python scripts/validation/validate_local_hv_centers.py \
        --pannuke_dir /home/amar/data/PanNuke \
        --fold 0 \
        --n_samples 5

    # Avec visualisations comparatives
    python scripts/validation/validate_local_hv_centers.py \
        --pannuke_dir /home/amar/data/PanNuke \
        --fold 0 \
        --n_samples 5 \
        --visualize \
        --output_dir results/hv_validation
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE


def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """Calcule cartes HV (Horizontal/Vertical) centripètes."""
    h, w = inst_map.shape
    hv_map = np.zeros((2, h, w), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) == 0:
            continue

        # Centroïde
        cy = y_coords.mean()
        cx = x_coords.mean()

        # Max distances pour normalisation
        max_dist_y = max(abs(y_coords - cy).max(), 1e-6)
        max_dist_x = max(abs(x_coords - cx).max(), 1e-6)

        # Vecteurs centripètes normalisés [-1, 1]
        h_map = (cx - x_coords) / max_dist_x
        v_map = (cy - y_coords) / max_dist_y

        hv_map[0, y_coords, x_coords] = h_map
        hv_map[1, y_coords, x_coords] = v_map

    return hv_map


def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """Extrait instance map depuis masque PanNuke."""
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)

    h, w = mask.shape[:2]
    inst_map = np.zeros((h, w), dtype=np.int32)
    instance_counter = 1

    # Channel 0
    channel_0 = mask[:, :, 0]
    if channel_0.max() > 0:
        inst_ids_0 = np.unique(channel_0)
        inst_ids_0 = inst_ids_0[inst_ids_0 > 0]
        for inst_id in inst_ids_0:
            inst_mask = channel_0 == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canaux 1-4
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        if channel_mask.max() > 0:
            inst_ids = np.unique(channel_mask)
            inst_ids = inst_ids[inst_ids > 0]
            for inst_id in inst_ids:
                inst_mask = channel_mask == inst_id
                inst_mask_new = inst_mask & (inst_map == 0)
                if inst_mask_new.sum() > 0:
                    inst_map[inst_mask_new] = instance_counter
                    instance_counter += 1

    return inst_map


def extract_crop_OLD_METHOD(
    hv_target_global: np.ndarray,
    x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """Ancienne méthode (BUG): Simple slicing des HV globaux."""
    return hv_target_global[:, y1:y2, x1:x2]


def extract_crop_NEW_METHOD(
    np_target: np.ndarray,
    x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """Nouvelle méthode (FIX): Recalcul centres HV locaux."""
    crop_np = np_target[y1:y2, x1:x2]

    # Relabel instances locales
    binary_mask = (crop_np > 0.5).astype(np.uint8)
    local_inst_map, _ = ndimage.label(binary_mask, structure=np.ones((3, 3)))

    # Recalculer HV avec centres locaux
    crop_hv = compute_hv_maps(local_inst_map)

    return crop_hv


def compute_hv_divergence(hv_map: np.ndarray, np_mask: np.ndarray) -> float:
    """
    Calcule divergence moyenne des vecteurs HV.

    Divergence < 0 = vecteurs pointent VERS le centre (correct)
    Divergence > 0 = vecteurs pointent VERS l'extérieur (incorrect)
    """
    h_map = hv_map[0]  # Horizontal
    v_map = hv_map[1]  # Vertical

    # Gradients (approximation différence finie)
    dh_dx = np.gradient(h_map, axis=1)
    dv_dy = np.gradient(v_map, axis=0)

    # Divergence = ∂H/∂x + ∂V/∂y
    divergence = dh_dx + dv_dy

    # Moyenner uniquement sur pixels de noyaux
    mask_bool = np_mask > 0.5
    if mask_bool.sum() == 0:
        return 0.0

    mean_div = divergence[mask_bool].mean()
    return mean_div


def find_hv_centers(hv_map: np.ndarray, np_mask: np.ndarray) -> list:
    """
    Trouve les centres (local maxima) des vecteurs HV.

    Un centre est un pixel où les vecteurs convergent (magnitude faible).
    """
    h_map = hv_map[0]
    v_map = hv_map[1]

    # Magnitude HV (distance au centre)
    magnitude = np.sqrt(h_map**2 + v_map**2)

    # Masquer background
    magnitude_masked = magnitude.copy()
    magnitude_masked[np_mask <= 0.5] = 1e6

    # Trouver local minima (centres)
    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(magnitude_masked, size=3)
    centers_mask = (magnitude_masked == local_min) & (magnitude_masked < 0.1)

    centers = np.argwhere(centers_mask)
    return centers.tolist()


def visualize_hv_comparison(
    crop_image: np.ndarray,
    crop_hv_old: np.ndarray,
    crop_hv_new: np.ndarray,
    crop_np: np.ndarray,
    crop_name: str,
    output_path: Path
):
    """
    Génère visualisation comparative OLD vs NEW HV maps.

    Créé une figure 2×3:
    Row 1: RGB | HV magnitude OLD | HV magnitude NEW
    Row 2: Overlay centres OLD | Overlay centres NEW | Divergence map
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Validation HV Centres Locaux - {crop_name}", fontsize=16, fontweight='bold')

    # Calculer magnitudes
    mag_old = np.sqrt(crop_hv_old[0]**2 + crop_hv_old[1]**2)
    mag_new = np.sqrt(crop_hv_new[0]**2 + crop_hv_new[1]**2)

    # Masquer background
    mag_old_masked = mag_old.copy()
    mag_new_masked = mag_new.copy()
    mag_old_masked[crop_np <= 0.5] = 0
    mag_new_masked[crop_np <= 0.5] = 0

    # Trouver centres
    centers_old = find_hv_centers(crop_hv_old, crop_np)
    centers_new = find_hv_centers(crop_hv_new, crop_np)

    # Row 1, Col 1: RGB crop
    axes[0, 0].imshow(crop_image)
    axes[0, 0].set_title('Image RGB Crop', fontweight='bold')
    axes[0, 0].axis('off')

    # Row 1, Col 2: HV magnitude OLD
    im1 = axes[0, 1].imshow(mag_old_masked, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title(f'HV Magnitude OLD\n({len(centers_old)} centres globaux)', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Row 1, Col 3: HV magnitude NEW
    im2 = axes[0, 2].imshow(mag_new_masked, cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title(f'HV Magnitude NEW\n({len(centers_new)} centres locaux)', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Row 2, Col 1: Overlay centres OLD
    axes[1, 0].imshow(crop_image)
    for cy, cx in centers_old:
        circle = mpatches.Circle((cx, cy), radius=5, color='red', fill=False, linewidth=2)
        axes[1, 0].add_patch(circle)
        # Croix pour marquer centre
        axes[1, 0].plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
    axes[1, 0].set_title('Centres OLD (globaux)\n❌ Peuvent être hors crop', fontweight='bold', color='red')
    axes[1, 0].axis('off')

    # Row 2, Col 2: Overlay centres NEW
    axes[1, 1].imshow(crop_image)
    for cy, cx in centers_new:
        circle = mpatches.Circle((cx, cy), radius=5, color='lime', fill=False, linewidth=2)
        axes[1, 1].add_patch(circle)
        axes[1, 1].plot(cx, cy, 'g+', markersize=10, markeredgewidth=2)
    axes[1, 1].set_title('Centres NEW (locaux)\n✅ Tous dans crop', fontweight='bold', color='green')
    axes[1, 1].axis('off')

    # Row 2, Col 3: Divergence map
    # Calculer divergence
    h_old = crop_hv_old[0]
    v_old = crop_hv_old[1]
    dh_dx_old = np.gradient(h_old, axis=1)
    dv_dy_old = np.gradient(v_old, axis=0)
    div_old = dh_dx_old + dv_dy_old

    h_new = crop_hv_new[0]
    v_new = crop_hv_new[1]
    dh_dx_new = np.gradient(h_new, axis=1)
    dv_dy_new = np.gradient(v_new, axis=0)
    div_new = dh_dx_new + dv_dy_new

    # Masquer background
    div_new_masked = div_new.copy()
    div_new_masked[crop_np <= 0.5] = 0

    im3 = axes[1, 2].imshow(div_new_masked, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title('Divergence NEW\n(Bleu=centripète, Rouge=sortant)', fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    # Statistiques
    div_mean_old = div_old[crop_np > 0.5].mean()
    div_mean_new = div_new[crop_np > 0.5].mean()

    stats_text = (
        f"Statistiques:\n"
        f"  Divergence OLD: {div_mean_old:+.6f}\n"
        f"  Divergence NEW: {div_mean_new:+.6f}\n"
        f"  Centres OLD: {len(centers_old)}\n"
        f"  Centres NEW: {len(centers_new)}\n"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"      💾 Visualisation sauvegardée: {output_path.name}")


def validate_crop(
    crop_hv_old: np.ndarray,
    crop_hv_new: np.ndarray,
    crop_np: np.ndarray,
    crop_name: str
) -> dict:
    """Valide un crop en comparant ancienne vs nouvelle méthode."""
    results = {
        'crop_name': crop_name,
        'valid': True,
        'errors': []
    }

    # 1. Vérifier range [-1, 1]
    if crop_hv_new.min() < -1.0 or crop_hv_new.max() > 1.0:
        results['valid'] = False
        results['errors'].append(
            f"HV range invalid: [{crop_hv_new.min():.3f}, {crop_hv_new.max():.3f}]"
        )

    # 2. Vérifier divergence < 0 (centripète)
    div_old = compute_hv_divergence(crop_hv_old, crop_np)
    div_new = compute_hv_divergence(crop_hv_new, crop_np)

    results['divergence_old'] = div_old
    results['divergence_new'] = div_new

    if div_new > 0:
        results['valid'] = False
        results['errors'].append(
            f"Divergence positive (vecteurs sortants): {div_new:.6f}"
        )

    # 3. Trouver centres HV
    centers_old = find_hv_centers(crop_hv_old, crop_np)
    centers_new = find_hv_centers(crop_hv_new, crop_np)

    results['n_centers_old'] = len(centers_old)
    results['n_centers_new'] = len(centers_new)

    # 4. Vérifier que centres NEW sont DANS le crop [0, 224)
    crop_size = 224
    centers_out_of_bounds = 0
    for cy, cx in centers_new:
        if not (0 <= cy < crop_size and 0 <= cx < crop_size):
            centers_out_of_bounds += 1

    if centers_out_of_bounds > 0:
        results['valid'] = False
        results['errors'].append(
            f"{centers_out_of_bounds} centres hors du crop [0, {crop_size})"
        )

    results['centers_out_of_bounds'] = centers_out_of_bounds

    return results


def main():
    parser = argparse.ArgumentParser(description="Valide recalcul centres HV locaux")
    parser.add_argument('--pannuke_dir', type=Path, required=True,
                        help="Répertoire PanNuke")
    parser.add_argument('--fold', type=int, default=0,
                        help="Fold PanNuke (0, 1, 2)")
    parser.add_argument('--n_samples', type=int, default=5,
                        help="Nombre d'échantillons à tester")
    parser.add_argument('--visualize', action='store_true',
                        help="Générer visualisations comparatives (sauvées dans results/)")
    parser.add_argument('--output_dir', type=Path, default=Path('results/hv_validation'),
                        help="Répertoire de sortie pour visualisations")

    args = parser.parse_args()

    # Créer répertoire de sortie si visualisation activée
    if args.visualize:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Visualisations seront sauvegardées dans: {args.output_dir}\n")

    # Charger données PanNuke
    fold_dir = args.pannuke_dir / f"fold{args.fold}"
    images_path = fold_dir / "images.npy"
    masks_path = fold_dir / "masks.npy"

    if not images_path.exists():
        print(f"❌ Fichiers manquants: {fold_dir}")
        return 1

    print(f"\n{'='*70}")
    print(f"VALIDATION CENTRES HV LOCAUX - Fold {args.fold}")
    print(f"{'='*70}\n")

    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')

    # Positions de crop (centre uniquement pour test rapide)
    crop_positions = {
        'center': (16, 16, 240, 240),
        'top_left': (0, 0, 224, 224),
        'bottom_right': (32, 32, 256, 256),
    }

    all_results = []
    n_valid = 0
    n_total = 0

    for i in range(min(args.n_samples, len(images))):
        print(f"\n📸 Image {i+1}/{args.n_samples}")

        image = images[i]
        mask = masks[i]

        # Générer targets globaux 256×256
        inst_map_global = extract_pannuke_instances(mask)
        hv_global = compute_hv_maps(inst_map_global)
        np_global = mask[:, :, :5].sum(axis=-1) > 0
        np_global = np_global.astype(np.float32)

        # Tester chaque position de crop
        for crop_name, (x1, y1, x2, y2) in crop_positions.items():
            # OLD method: Simple slicing
            crop_hv_old = extract_crop_OLD_METHOD(hv_global, x1, y1, x2, y2)

            # NEW method: Recalcul local
            crop_hv_new = extract_crop_NEW_METHOD(np_global, x1, y1, x2, y2)

            # NP crop (pour validation)
            crop_np = np_global[y1:y2, x1:x2]

            # Valider
            result = validate_crop(crop_hv_old, crop_hv_new, crop_np, crop_name)
            all_results.append(result)
            n_total += 1

            if result['valid']:
                n_valid += 1
                status = "✅"
            else:
                status = "❌"

            print(f"  {status} {crop_name:15} | "
                  f"Div: {result['divergence_new']:+.6f} | "
                  f"Centers: {result['n_centers_new']} | "
                  f"Errors: {len(result['errors'])}")

            if result['errors']:
                for error in result['errors']:
                    print(f"      ⚠️  {error}")

            # Générer visualisation (seulement premiers échantillons)
            if args.visualize and i < 2:  # Limiter à 2 images pour ne pas saturer
                crop_image = image[y1:y2, x1:x2]
                output_path = args.output_dir / f"validation_image{i+1}_{crop_name}.png"
                visualize_hv_comparison(
                    crop_image, crop_hv_old, crop_hv_new, crop_np,
                    f"Image {i+1} - {crop_name}", output_path
                )

    # Résumé
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ")
    print(f"{'='*70}")
    print(f"Total crops testés: {n_total}")
    print(f"Valides: {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"Invalides: {n_total - n_valid}")

    # Statistiques divergence
    divs_new = [r['divergence_new'] for r in all_results]
    div_mean = np.mean(divs_new)
    div_negative_pct = sum(1 for d in divs_new if d < 0) / len(divs_new) * 100

    print(f"\nDivergence moyenne: {div_mean:+.6f}")
    print(f"Divergence négative: {div_negative_pct:.1f}% (cible: ~100%)")

    if n_valid == n_total and div_negative_pct > 95:
        print(f"\n🎉 TOUS LES TESTS PASSENT - Fix validé!")
        return 0
    else:
        print(f"\n⚠️  ÉCHEC - Fix nécessite correction")
        return 1


if __name__ == '__main__':
    sys.exit(main())
