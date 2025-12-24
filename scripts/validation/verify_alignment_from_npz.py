#!/usr/bin/env python3
"""
VÉRIFICATION SPATIAL ALIGNMENT - DEPUIS NPZ (PAS DE RECALCUL)

CRITIQUE: Ce script charge les HV targets DEPUIS LE NPZ au lieu de les recalculer.
Cela permet de vérifier l'alignement des données RÉELLEMENT sauvegardées.

Différence avec verify_spatial_alignment_FIXED.py:
- ❌ FIXED: Recalcule hv_target = compute_hv_maps(inst_map)  (formule v7)
- ✅ CETTE VERSION: Charge hv_target depuis NPZ (données v8)

Usage:
    python scripts/validation/verify_alignment_from_npz.py --family epidermal
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# CENTROID EXTRACTION FROM HV MAPS (STORED IN NPZ)
# =============================================================================

def extract_centers_from_hv(hv_map: np.ndarray, inst_map: np.ndarray) -> np.ndarray:
    """
    Extrait les centroïdes depuis HV maps en cherchant minimum de gradient.

    Args:
        hv_map: HV maps (2, H, W) - CHARGÉ DEPUIS NPZ
        inst_map: Instance map (H, W) - Pour délimiter instances

    Returns:
        centers: Array (N, 2) de centroïdes (y, x)
    """
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    centers = []

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) == 0:
            continue

        # Calculer gradient de HV
        grad_v = np.gradient(hv_map[0])  # [grad_y, grad_x]
        grad_h = np.gradient(hv_map[1])  # [grad_y, grad_x]

        # Magnitude du gradient
        mag = np.sqrt(grad_v[0]**2 + grad_v[1]**2 + grad_h[0]**2 + grad_h[1]**2)

        # Trouver minimum dans l'instance
        mags_in_inst = mag[y_coords, x_coords]
        min_idx = np.argmin(mags_in_inst)

        center_y = y_coords[min_idx]
        center_x = x_coords[min_idx]

        centers.append([center_y, center_x])

    return np.array(centers) if len(centers) > 0 else np.zeros((0, 2))


# =============================================================================
# GT CENTROID EXTRACTION
# =============================================================================

def extract_gt_centers(inst_map: np.ndarray) -> np.ndarray:
    """
    Extrait les centroïdes GT depuis instance map.

    Args:
        inst_map: Instance map (H, W)

    Returns:
        centers: Array (N, 2) de centroïdes (y, x)
    """
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    centers = []

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) > 0:
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            centers.append([center_y, center_x])

    return np.array(centers) if len(centers) > 0 else np.zeros((0, 2))


# =============================================================================
# MATCHING AND METRICS
# =============================================================================

def match_and_measure(gt_centers: np.ndarray, pred_centers: np.ndarray) -> dict:
    """
    Match centroïdes GT et prédits avec Hungarian algorithm.

    Args:
        gt_centers: Centroïdes GT (N, 2)
        pred_centers: Centroïdes prédits (M, 2)

    Returns:
        dict avec distances et métriques
    """
    if len(gt_centers) == 0 or len(pred_centers) == 0:
        return {
            "mean_distance": np.inf,
            "min_distance": np.inf,
            "max_distance": np.inf,
            "tp": 0,
            "fp": len(pred_centers),
            "fn": len(gt_centers),
        }

    # Matrice de coûts (distances euclidiennes)
    from scipy.spatial.distance import cdist
    cost_matrix = cdist(gt_centers, pred_centers)

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Distances des paires matchées
    distances = cost_matrix[row_ind, col_ind]

    # Métriques
    threshold = 10.0  # pixels
    tp = np.sum(distances < threshold)
    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp

    return {
        "mean_distance": np.mean(distances),
        "min_distance": np.min(distances),
        "max_distance": np.max(distances),
        "distances": distances,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_pairs": list(zip(row_ind, col_ind)),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_alignment(
    image: np.ndarray,
    gt_centers: np.ndarray,
    pred_centers: np.ndarray,
    result: dict,
    output_path: Path,
    sample_name: str
):
    """
    Visualise l'alignement GT vs Prédictions.

    Args:
        image: Image RGB (H, W, 3)
        gt_centers: Centroïdes GT (N, 2)
        pred_centers: Centroïdes prédits (M, 2)
        result: Résultat de match_and_measure()
        output_path: Répertoire de sortie
        sample_name: Nom de l'échantillon
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.imshow(image)

    # Centroïdes GT (rouge)
    if len(gt_centers) > 0:
        ax.scatter(gt_centers[:, 1], gt_centers[:, 0], c='red', s=100, marker='x',
                   linewidths=3, label=f'GT ({len(gt_centers)})', zorder=3)

    # Centroïdes prédits (vert)
    if len(pred_centers) > 0:
        ax.scatter(pred_centers[:, 1], pred_centers[:, 0], c='lime', s=80, marker='o',
                   alpha=0.7, label=f'Pred ({len(pred_centers)})', zorder=2)

    # Lignes de matching
    if "matched_pairs" in result:
        for gt_idx, pred_idx in result["matched_pairs"]:
            gt_y, gt_x = gt_centers[gt_idx]
            pred_y, pred_x = pred_centers[pred_idx]
            ax.plot([gt_x, pred_x], [gt_y, pred_y], 'yellow', linewidth=1, alpha=0.5, zorder=1)

    ax.set_title(f"{sample_name} | Dist: {result['mean_distance']:.2f}px | "
                 f"TP:{result['tp']} FP:{result['fp']} FN:{result['fn']}", fontsize=12)
    ax.legend(loc='upper right')
    ax.axis('off')

    plt.tight_layout()
    output_file = output_path / f"{sample_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN VERIFICATION
# =============================================================================

def verify_alignment(family: str, n_samples: int = 5):
    """
    Vérifie l'alignement spatial en chargeant HV depuis NPZ.

    Args:
        family: Nom de la famille
        n_samples: Nombre d'échantillons à tester
    """
    print("=" * 80)
    print("VÉRIFICATION ALIGNMENT DEPUIS NPZ v8 (PAS DE RECALCUL)")
    print("=" * 80)
    print(f"Famille: {family}")
    print(f"Échantillons: {n_samples}")
    print()

    # Charger NPZ
    data_file = Path(f"data/family_FIXED/{family}_data_FIXED.npz")
    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return 1

    data = np.load(data_file)

    print(f"✅ Données chargées: {len(data['images'])} échantillons")
    print(f"   Fold IDs: {np.unique(data['fold_ids'])}")

    # Vérifier timestamp NPZ
    import datetime
    timestamp = datetime.datetime.fromtimestamp(data_file.stat().st_mtime)
    print(f"   NPZ créé: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Sélectionner échantillons aléatoires
    total_samples = len(data['images'])
    n_samples = min(n_samples, total_samples)
    sample_indices = np.random.choice(total_samples, n_samples, replace=False)

    # Créer répertoire de sortie
    output_dir = Path("results/alignment_from_npz")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_distances = []
    all_tp, all_fp, all_fn = 0, 0, 0

    print("Vérification échantillons:")
    print("-" * 80)

    for i, idx in enumerate(sample_indices):
        image = data['images'][idx]
        hv_target = data['hv_targets'][idx]  # ✅ CHARGÉ DEPUIS NPZ (v8)
        np_target = data['np_targets'][idx]
        fold_id = data['fold_ids'][idx]
        image_id = data['image_ids'][idx]

        # Créer instance map depuis np_target
        labeled_mask, n_instances = label(np_target > 0)

        # Extraire centroïdes GT
        gt_centers = extract_gt_centers(labeled_mask)

        # Extraire centroïdes prédits depuis HV (DEPUIS NPZ, PAS RECALCULÉ)
        pred_centers = extract_centers_from_hv(hv_target, labeled_mask)

        # Matching et métriques
        result = match_and_measure(gt_centers, pred_centers)

        # Visualiser
        sample_name = f"sample_{idx:04d}_fold{fold_id}_img{image_id}"
        visualize_alignment(image, gt_centers, pred_centers, result, output_dir, sample_name)

        # Afficher résultat
        status = "✅" if result['mean_distance'] < 2.0 else "❌"
        print(f"  [{i+1}/{n_samples}] Sample {idx} (fold{fold_id}, img{image_id}): "
              f"dist={result['mean_distance']:.2f}px, GT={len(gt_centers)}, "
              f"Pred={len(pred_centers)}, TP={result['tp']}, FP={result['fp']}, FN={result['fn']} {status}")

        all_distances.append(result['mean_distance'])
        all_tp += result['tp']
        all_fp += result['fp']
        all_fn += result['fn']

    # Résultats globaux
    print()
    print("=" * 80)
    print("RÉSULTATS")
    print("=" * 80)
    print(f"Distance moyenne: {np.mean(all_distances):.2f} pixels")
    print(f"Distance min:     {np.min(all_distances):.2f} pixels")
    print(f"Distance max:     {np.max(all_distances):.2f} pixels")
    print()
    print("Detection (seuil 10px):")
    print(f"  TP: {all_tp}, FP: {all_fp}, FN: {all_fn}")

    precision = all_tp / (all_tp + all_fp) * 100 if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) * 100 if (all_tp + all_fn) > 0 else 0

    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print()

    mean_dist = np.mean(all_distances)
    if mean_dist < 2.0:
        print("✅ GO - Alignement PARFAIT (NPZ v8 CORRECT)")
        print("   → Régénérer les 5 familles avec v8")
    elif mean_dist < 10.0:
        print("🟡 ACCEPTABLE - Légère dérive")
    else:
        print("❌ NO-GO")
        print("   Alignement PROBLÉMATIQUE - NPZ contient encore données v7?")
        print(f"   Vérifier timestamp NPZ: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    print()
    print("=" * 80)
    print(f"Visualisations: {output_dir}")
    print("=" * 80)

    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vérification alignment depuis NPZ")
    parser.add_argument("--family", type=str, required=True, help="Famille à vérifier")
    parser.add_argument("--n_samples", type=int, default=5, help="Nombre d'échantillons")

    args = parser.parse_args()

    return verify_alignment(args.family, args.n_samples)


if __name__ == "__main__":
    exit(main())
