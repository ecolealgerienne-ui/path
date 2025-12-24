#!/usr/bin/env python3
"""
Script de Vérification Pixel-Perfect de l'Alignement Spatial

Ce script vérifie que les HV targets sont PARFAITEMENT alignés avec les vraies
positions des noyaux dans les images. C'est un test GO/NO-GO CRITIQUE avant
le re-training.

Problème ciblé (Bug #4):
- Si features training sont corrompues (Bug #1/#2), les HV targets peuvent être
  décalés spatialement par rapport aux vraies positions des noyaux.
- Conséquence: Le modèle apprend un décalage systématique → AJI catastrophique.

Test:
1. Charger image + HV targets
2. Calculer gradients HV (magnitude + direction)
3. Superposer sur l'image
4. Vérifier que les vecteurs pointent VERS les centres des noyaux
5. Verdict: GO (alignement OK) ou NO-GO (décalage détecté)

Usage:
    python scripts/validation/verify_spatial_alignment.py \
        --family glandular \
        --n_samples 5 \
        --output_dir results/spatial_alignment

Critère GO/NO-GO:
    - GO: Vecteurs HV pointent vers centres ± 2 pixels
    - NO-GO: Décalage > 2 pixels → NE PAS LANCER LE TRAINING
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE

def load_family_data(family):
    """
    Charge les données d'une famille.

    Returns:
        dict: {'images': array, 'hv_targets': array, 'np_targets': array}
    """
    # Chercher dans data/cache/family_data/ ou family_data_FIXED/
    possible_paths = [
        Path(f"data/cache/family_data/{family}_data_FIXED.npz"),
        Path(f"data/cache/family_data_FIXED/{family}_data_FIXED.npz"),
        Path(f"data/cache/family_data/{family}_data.npz"),
    ]

    for path in possible_paths:
        if path.exists():
            print(f"✅ Chargement depuis: {path}")
            data = np.load(path, allow_pickle=True)
            return {
                'images': data['images'],
                'hv_targets': data['hv_targets'],
                'np_targets': data['np_targets'],
                'path': path
            }

    raise FileNotFoundError(
        f"❌ Aucun fichier de données trouvé pour famille '{family}'.\n"
        f"   Cherché dans: {[str(p) for p in possible_paths]}"
    )

def compute_hv_gradient_magnitude(hv):
    """
    Calcule la magnitude du gradient HV.

    Args:
        hv: (2, H, W) - Horizontal et Vertical maps

    Returns:
        magnitude: (H, W) - sqrt(H^2 + V^2)
    """
    h_map = hv[0]  # (H, W)
    v_map = hv[1]  # (H, W)

    # Calculer gradients par différences finies
    grad_h_x = np.diff(h_map, axis=1, prepend=h_map[:, :1])  # Gradient de H en x
    grad_h_y = np.diff(h_map, axis=0, prepend=h_map[:1, :])  # Gradient de H en y

    grad_v_x = np.diff(v_map, axis=1, prepend=v_map[:, :1])
    grad_v_y = np.diff(v_map, axis=0, prepend=v_map[:1, :])

    # Magnitude du gradient (norme L2)
    magnitude = np.sqrt(grad_h_x**2 + grad_h_y**2 + grad_v_x**2 + grad_v_y**2)

    return magnitude

def visualize_alignment(image, hv_target, np_target, sample_idx, output_path):
    """
    Visualise l'alignement spatial entre image et HV targets.

    Args:
        image: (H, W, 3) - Image RGB uint8
        hv_target: (2, H, W) - HV maps float32 [-1, 1]
        np_target: (H, W) - Nuclear Presence mask
        sample_idx: int - Index de l'échantillon
        output_path: Path - Chemin de sortie
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Image + NP mask + HV magnitude
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Image Originale (Sample {sample_idx})", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np_target, cmap='gray')
    axes[0, 1].set_title("NP Target (Noyaux)", fontsize=12)
    axes[0, 1].axis('off')

    # Magnitude du gradient HV
    hv_magnitude = compute_hv_gradient_magnitude(hv_target)
    axes[0, 2].imshow(hv_magnitude, cmap='hot')
    axes[0, 2].set_title("HV Gradient Magnitude\n(Fort = frontières)", fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: Superpositions
    # 2.1: Image + NP contours
    axes[1, 0].imshow(image)
    # Contours des noyaux
    from scipy import ndimage
    edges = ndimage.sobel(np_target.astype(float))
    axes[1, 0].contour(edges, colors='lime', linewidths=2, levels=[0.5])
    axes[1, 0].set_title("Image + Contours NP", fontsize=12)
    axes[1, 0].axis('off')

    # 2.2: Image + Champ de vecteurs HV (quiver)
    axes[1, 1].imshow(image)

    # Sous-échantillonner pour lisibilité (tous les N pixels)
    step = 16
    H, W = hv_target.shape[1:]
    Y, X = np.mgrid[0:H:step, 0:W:step]

    # Vecteurs HV
    U = hv_target[0, ::step, ::step]  # Horizontal
    V = hv_target[1, ::step, ::step]  # Vertical

    # Masquer le background
    mask = np_target[::step, ::step] > 0

    # Quiver plot (flèches pointant VERS le centre)
    # Note: Les HV maps encodent la distance AU centre, donc on inverse
    axes[1, 1].quiver(
        X[mask], Y[mask],
        -U[mask], -V[mask],  # Inverser pour pointer vers centre
        color='yellow',
        scale=20,
        width=0.003,
        headwidth=4,
        headlength=5,
        alpha=0.8
    )
    axes[1, 1].set_title("Image + Vecteurs HV\n(Flèches → centres noyaux)", fontsize=12)
    axes[1, 1].axis('off')

    # 2.3: HV magnitude + Contours
    axes[1, 2].imshow(hv_magnitude, cmap='hot')
    axes[1, 2].contour(edges, colors='cyan', linewidths=2, levels=[0.5])
    axes[1, 2].set_title("HV Magnitude + Contours\n(Doivent coïncider)", fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  💾 Visualisation sauvée: {output_path}")

def compute_alignment_score(hv_target, np_target):
    """
    Calcule un score d'alignement spatial.

    Méthode:
    1. Détecter les pics de gradient HV (frontières attendues)
    2. Comparer avec les contours NP réels
    3. Calculer la distance moyenne entre pics HV et contours NP

    Returns:
        float: Distance moyenne en pixels (0 = parfait, >2 = problématique)
    """
    from scipy import ndimage

    # Gradients HV
    hv_magnitude = compute_hv_gradient_magnitude(hv_target)

    # Contours NP réels
    np_edges = ndimage.sobel(np_target.astype(float))

    # Seuiller pour avoir des binaires
    hv_peaks = hv_magnitude > np.percentile(hv_magnitude[np_target > 0], 75)
    np_edges_bin = np_edges > 0.5

    # Distance entre pics HV et contours NP
    from scipy.ndimage import distance_transform_edt

    # Distance de chaque pixel de hv_peaks au contour NP le plus proche
    dist_map = distance_transform_edt(~np_edges_bin)

    # Distance moyenne des pics HV aux contours NP
    if hv_peaks.sum() > 0:
        mean_dist = dist_map[hv_peaks].mean()
    else:
        mean_dist = float('inf')

    return mean_dist

def verify_spatial_alignment(family, n_samples=5, output_dir="results/spatial_alignment"):
    """
    Vérifie l'alignement spatial sur N échantillons.

    Returns:
        dict: {
            'verdict': 'GO' ou 'NO-GO',
            'mean_distance': float,
            'max_distance': float,
            'samples_checked': int
        }
    """
    print("="*80)
    print("VÉRIFICATION PIXEL-PERFECT DE L'ALIGNEMENT SPATIAL")
    print("="*80)
    print(f"Famille: {family}")
    print(f"Échantillons: {n_samples}")
    print()

    # Charger données
    data = load_family_data(family)
    images = data['images']
    hv_targets = data['hv_targets']
    np_targets = data['np_targets']

    print(f"✅ Données chargées depuis: {data['path']}")
    print(f"   Images: {images.shape}")
    print(f"   HV targets: {hv_targets.shape}, dtype={hv_targets.dtype}, range=[{hv_targets.min():.3f}, {hv_targets.max():.3f}]")
    print(f"   NP targets: {np_targets.shape}, dtype={np_targets.dtype}")
    print()

    # Vérifications préliminaires
    if hv_targets.dtype != np.float32:
        print(f"⚠️ WARNING: HV dtype est {hv_targets.dtype} au lieu de float32")

    if not (-1.0 <= hv_targets.min() <= hv_targets.max() <= 1.0):
        print(f"⚠️ WARNING: HV range [{hv_targets.min():.3f}, {hv_targets.max():.3f}] hors de [-1, 1]")

    # Créer répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Tester N échantillons aléatoires
    np.random.seed(42)
    n_total = len(images)
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    distances = []

    print("Vérification échantillons:")
    print("-" * 40)

    for i, idx in enumerate(indices):
        image = images[idx]
        hv_target = hv_targets[idx]
        np_target = np_targets[idx]

        # Calculer score d'alignement
        dist = compute_alignment_score(hv_target, np_target)
        distances.append(dist)

        print(f"  [{i+1}/{len(indices)}] Sample {idx}: distance={dist:.2f} px", end="")

        if dist <= 2.0:
            print(" ✅")
        elif dist <= 5.0:
            print(" ⚠️")
        else:
            print(" ❌ PROBLÉMATIQUE")

        # Visualiser
        vis_path = output_path / f"alignment_sample_{idx:04d}.png"
        visualize_alignment(image, hv_target, np_target, idx, vis_path)

    print()

    # Statistiques
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    min_dist = np.min(distances)

    print("="*80)
    print("RÉSULTATS")
    print("="*80)
    print(f"Distance moyenne: {mean_dist:.2f} pixels")
    print(f"Distance min:     {min_dist:.2f} pixels")
    print(f"Distance max:     {max_dist:.2f} pixels")
    print()

    # Verdict GO/NO-GO
    if mean_dist <= 2.0 and max_dist <= 5.0:
        verdict = "GO"
        icon = "✅"
        message = "Alignement EXCELLENT - Training peut être lancé"
    elif mean_dist <= 5.0:
        verdict = "CAUTION"
        icon = "⚠️"
        message = "Alignement ACCEPTABLE - Vérifier visuellement les plots"
    else:
        verdict = "NO-GO"
        icon = "❌"
        message = "Alignement PROBLÉMATIQUE - NE PAS LANCER LE TRAINING"

    print(f"{icon} VERDICT: {verdict}")
    print(f"   {message}")
    print()

    if verdict == "NO-GO":
        print("🚨 ACTION REQUISE:")
        print("   1. Vérifier que les features ont été ré-générées (Bug #4)")
        print("   2. Vérifier que prepare_family_data_FIXED.py a été utilisé")
        print("   3. Consulter les visualisations dans:", output_dir)
        print()

    print("="*80)
    print("Visualisations sauvées dans:", output_dir)
    print("="*80)

    return {
        'verdict': verdict,
        'mean_distance': mean_dist,
        'max_distance': max_dist,
        'min_distance': min_dist,
        'samples_checked': len(indices)
    }

def main():
    parser = argparse.ArgumentParser(description="Vérification pixel-perfect de l'alignement spatial")
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
                        help="Famille à vérifier")
    parser.add_argument('--n_samples', type=int, default=5,
                        help="Nombre d'échantillons à vérifier (défaut: 5)")
    parser.add_argument('--output_dir', type=str, default="results/spatial_alignment",
                        help="Répertoire de sortie pour les visualisations")

    args = parser.parse_args()

    try:
        result = verify_spatial_alignment(
            family=args.family,
            n_samples=args.n_samples,
            output_dir=args.output_dir
        )

        # Exit code basé sur le verdict
        if result['verdict'] == 'GO':
            sys.exit(0)
        elif result['verdict'] == 'CAUTION':
            sys.exit(1)
        else:  # NO-GO
            sys.exit(2)

    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()
