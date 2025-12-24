#!/usr/bin/env python3
"""
VÉRIFICATION SPATIAL ALIGNMENT FIXED (Expert-Guided 2025-12-23)

FIX DÉTECTION MULTI-PICS (v2 → FIXED - Expert Diagnosis):
    ❌ v2: peak_local_max sur magnitude des VALEURS HV
           → Détecte plusieurs pics par noyau (72 pred vs 43 GT)
           → Precision 27%, Distance 31px
    ✅ FIXED: Label instances + trouve pixel de gradient HV MINIMAL
           → Force 1 centre par instance (prédictions = GT)
           → Precision >95%, Distance <2px (attendu)

FIX BUG FANTÔME FOLD OFFSET (v1 → v2):
    ❌ v1: Comparait HV targets vs NP targets (tous deux depuis NPZ)
    ✅ v2+: Charge le VRAI masque GT depuis PanNuke/fold{fold_id}/masks.npy[image_id]

Usage:
    python scripts/validation/verify_spatial_alignment_FIXED.py --family epidermal
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_gt_centroids(mask: np.ndarray) -> list:
    """
    Extrait les centroides des instances depuis masque GT PanNuke.

    Args:
        mask: (256, 256, 6) - Masque PanNuke (canaux 1-5 = instances)

    Returns:
        list of (cy, cx): Centroides des instances
    """
    centroids = []

    # Canaux 1-4: instances annotées
    for c in range(1, 5):
        class_instances = mask[:, :, c]
        inst_ids = np.unique(class_instances)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = class_instances == inst_id
            y_coords, x_coords = np.where(inst_mask)

            if len(y_coords) > 0:
                cy = np.mean(y_coords)
                cx = np.mean(x_coords)
                centroids.append((cy, cx))

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
    epithelial_binary = mask[:, :, 5] > 0
    if epithelial_binary.any():
        _, labels = cv2.connectedComponents(epithelial_binary.astype(np.uint8))
        inst_ids = np.unique(labels)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = labels == inst_id
            y_coords, x_coords = np.where(inst_mask)

            if len(y_coords) > 0:
                cy = np.mean(y_coords)
                cx = np.mean(x_coords)
                centroids.append((cy, cx))

    return centroids


def predict_centroids_from_hv(hv_map: np.ndarray, np_mask: np.ndarray) -> list:
    """
    Prédit les centroides depuis HV maps (VERSION FIXÉE - Expert 2025-12-23).

    Méthode HoVer-Net CORRECTE:
    1. Labelliser le masque binaire pour identifier chaque instance
    2. Pour chaque instance, calculer magnitude du GRADIENT HV
    3. Trouver le pixel de magnitude MINIMALE (centre théorique)

    FIX v2 → FIXED:
        ❌ v2: peak_local_max sur magnitude des VALEURS HV
               → Détecte plusieurs pics par noyau (Precision 27%)
        ✅ FIXED: Labelliser instances + trouver gradient minimal
               → Force 1 centre par instance (Precision >95%)

    Args:
        hv_map: (2, H, W) - HV maps [V, H]
        np_mask: (H, W) - Masque binaire noyaux

    Returns:
        list of (cy, cx): Centroides prédits (UN par instance)
    """
    from scipy.ndimage import label

    # 1. Labelliser les instances dans le masque binaire
    labeled_mask, n_instances = label(np_mask > 0)

    if n_instances == 0:
        return []

    # 2. Calculer magnitude du GRADIENT HV (pas des valeurs!)
    # np.gradient retourne [grad_y, grad_x] pour un array 2D
    grad_v = np.gradient(hv_map[0])  # [dV/dy, dV/dx]
    grad_h = np.gradient(hv_map[1])  # [dH/dy, dH/dx]

    # Magnitude combinée (gradient spatial total)
    mag = np.sqrt(grad_v[0]**2 + grad_v[1]**2 + grad_h[0]**2 + grad_h[1]**2)

    # 3. Pour chaque instance, trouver pixel de magnitude MINIMALE
    centroids = []
    for inst_id in range(1, n_instances + 1):
        inst_mask = (labeled_mask == inst_id)

        # Coordonnées de tous les pixels de cette instance
        coords = np.argwhere(inst_mask)

        if len(coords) == 0:
            continue

        # Magnitudes pour ces pixels
        mags_in_inst = mag[inst_mask]

        # Pixel avec gradient MINIMAL = centre théorique
        min_idx = np.argmin(mags_in_inst)
        center_y, center_x = coords[min_idx]

        centroids.append((center_y, center_x))

    return centroids


def compute_centroid_matching_distance(pred_centroids: list, gt_centroids: list) -> dict:
    """
    Calcule la distance moyenne entre centroides prédits et GT (avec matching).

    Utilise Hungarian algorithm (linear_sum_assignment) pour apparier
    chaque centroide prédit au GT le plus proche.

    Returns:
        dict: {
            'mean_distance': float,
            'matched_pairs': int,
            'tp': int (True Positives - matched),
            'fp': int (False Positives - pred sans GT),
            'fn': int (False Negatives - GT sans pred)
        }
    """
    if len(pred_centroids) == 0 and len(gt_centroids) == 0:
        return {
            'mean_distance': 0.0,
            'matched_pairs': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0
        }

    if len(pred_centroids) == 0:
        return {
            'mean_distance': float('inf'),
            'matched_pairs': 0,
            'tp': 0,
            'fp': 0,
            'fn': len(gt_centroids)
        }

    if len(gt_centroids) == 0:
        return {
            'mean_distance': float('inf'),
            'matched_pairs': 0,
            'tp': 0,
            'fp': len(pred_centroids),
            'fn': 0
        }

    # Matrice de distances (pred x gt)
    pred_coords = np.array(pred_centroids)
    gt_coords = np.array(gt_centroids)

    dist_matrix = cdist(pred_coords, gt_coords, metric='euclidean')

    # Hungarian matching
    pred_indices, gt_indices = linear_sum_assignment(dist_matrix)

    # Calculer distances des paires matchées
    matched_distances = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        dist = dist_matrix[pred_idx, gt_idx]
        matched_distances.append(dist)

    mean_dist = np.mean(matched_distances) if matched_distances else float('inf')

    # Compter TP, FP, FN (avec seuil de distance)
    threshold = 10.0  # pixels (tolérance pour considérer un match correct)

    tp = sum(1 for d in matched_distances if d <= threshold)
    fp = len(pred_centroids) - len(pred_indices)  # Prédictions non matchées
    fn = len(gt_centroids) - len(gt_indices)      # GT non matchés

    return {
        'mean_distance': mean_dist,
        'matched_pairs': len(matched_distances),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def visualize_centroid_comparison(
    image: np.ndarray,
    pred_centroids: list,
    gt_centroids: list,
    sample_idx: int,
    fold_id: int,
    image_id: int,
    distance: float,
    output_path: Path
):
    """Visualise comparaison centroides prédits vs GT."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Image + GT centroids
    axes[0].imshow(image)
    axes[0].set_title(f"GT Centroids (fold{fold_id}, img{image_id})\n{len(gt_centroids)} instances", fontsize=12)
    axes[0].axis('off')

    for cy, cx in gt_centroids:
        axes[0].plot(cx, cy, 'g*', markersize=10, markeredgecolor='white', markeredgewidth=1)

    # Image + Predicted centroids
    axes[1].imshow(image)
    axes[1].set_title(f"Predicted Centroids (from HV maps)\n{len(pred_centroids)} instances\nDist: {distance:.2f}px", fontsize=12)
    axes[1].axis('off')

    for cy, cx in pred_centroids:
        axes[1].plot(cx, cy, 'r*', markersize=10, markeredgecolor='white', markeredgewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Vérification Spatial Alignment FIXED (Expert-Guided)")
    parser.add_argument("--family", type=str, required=True,
                        choices=["glandular", "digestive", "urologic", "respiratory", "epidermal"])
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/spatial_alignment_FIXED")
    args = parser.parse_args()

    print("="*80)
    print("VÉRIFICATION SPATIAL ALIGNMENT FIXED (EXPERT-GUIDED)")
    print("="*80)
    print(f"Famille: {args.family}")
    print(f"Échantillons: {args.n_samples}")
    print()

    # 1. Charger données NPZ
    data_file = Path(f"data/family_FIXED/{args.family}_data_FIXED.npz")
    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return 1

    data = np.load(data_file)

    images = data['images']
    hv_targets = data['hv_targets']
    np_targets = data['np_targets']
    fold_ids = data['fold_ids']
    image_ids = data['image_ids']

    print(f"✅ Données chargées: {len(images)} échantillons")
    print(f"   Fold IDs: {np.unique(fold_ids)}")
    print()

    # 2. Vérifier échantillons aléatoires
    np.random.seed(42)
    n_total = len(images)
    indices = np.random.choice(n_total, min(args.n_samples, n_total), replace=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_distances = []
    all_tps = []
    all_fps = []
    all_fns = []

    print("Vérification échantillons:")
    print("-"*80)

    pannuke_dir = Path("/home/amar/data/PanNuke")

    for i, idx in enumerate(indices):
        image = images[idx]
        hv_target = hv_targets[idx]
        np_target = np_targets[idx]
        fold_id = int(fold_ids[idx])
        image_id = int(image_ids[idx])

        # ✅ CRITIQUE: Charger le VRAI masque GT avec fold_id correct
        gt_mask_file = pannuke_dir / f"fold{fold_id}" / "masks.npy"

        if not gt_mask_file.exists():
            print(f"  ❌ Masque GT non trouvé: {gt_mask_file}")
            continue

        masks = np.load(gt_mask_file, mmap_mode='r')
        gt_mask = masks[image_id]

        # Extraire centroides GT
        gt_centroids = extract_gt_centroids(gt_mask)

        # Prédire centroides depuis HV maps
        pred_centroids = predict_centroids_from_hv(hv_target, np_target)

        # Comparer
        result = compute_centroid_matching_distance(pred_centroids, gt_centroids)

        all_distances.append(result['mean_distance'])
        all_tps.append(result['tp'])
        all_fps.append(result['fp'])
        all_fns.append(result['fn'])

        status = "✅" if result['mean_distance'] <= 2.0 else "❌"

        print(f"  [{i+1}/{len(indices)}] Sample {idx} (fold{fold_id}, img{image_id}): "
              f"dist={result['mean_distance']:.2f}px, "
              f"GT={len(gt_centroids)}, Pred={len(pred_centroids)}, "
              f"TP={result['tp']}, FP={result['fp']}, FN={result['fn']} {status}")

        # Visualiser
        vis_path = output_dir / f"alignment_sample_{idx:04d}_fold{fold_id}_img{image_id}.png"
        visualize_centroid_comparison(
            image, pred_centroids, gt_centroids,
            idx, fold_id, image_id,
            result['mean_distance'], vis_path
        )

    print()

    # 3. Statistiques
    mean_dist = np.mean(all_distances)
    min_dist = np.min(all_distances)
    max_dist = np.max(all_distances)

    total_tp = sum(all_tps)
    total_fp = sum(all_fps)
    total_fn = sum(all_fns)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    print("="*80)
    print("RÉSULTATS")
    print("="*80)
    print(f"Distance moyenne: {mean_dist:.2f} pixels")
    print(f"Distance min:     {min_dist:.2f} pixels")
    print(f"Distance max:     {max_dist:.2f} pixels")
    print()
    print(f"Detection (seuil 10px):")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print()

    # Verdict
    if mean_dist <= 2.0 and max_dist <= 5.0:
        verdict = "✅ GO"
        message = "Alignement EXCELLENT - Training peut être lancé"
    elif mean_dist <= 5.0:
        verdict = "⚠️ CAUTION"
        message = "Alignement ACCEPTABLE - Vérifier visuellement"
    else:
        verdict = "❌ NO-GO"
        message = "Alignement PROBLÉMATIQUE - NE PAS LANCER LE TRAINING"

    print(f"{verdict}")
    print(f"   {message}")
    print()

    print("="*80)
    print(f"Visualisations: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    exit(main())
