#!/usr/bin/env python3
"""
DIAGNOSTIC HV SIGN INVERSION (Expert Analysis 2025-12-24)

Teste 4 inversions de signe pour identifier la direction correcte des gradients:
1. Original (as-is)
2. InvertV (vertical sign inverted: -V, +H)
3. InvertH (horizontal sign inverted: +V, -H)
4. InvertBoth (both signs inverted: -V, -H)

Hypothèse: Les gradients HV pointent AWAY FROM center au lieu de TOWARDS center

Usage:
    python scripts/validation/diagnose_hv_sign.py --family epidermal
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def normalize_mask_format(mask: np.ndarray) -> np.ndarray:
    """Normalise le format du mask vers HWC (256, 256, 6)."""
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D with shape {mask.shape}")

    if mask.shape == (256, 256, 6):
        return mask
    elif mask.shape == (6, 256, 256):
        mask_hwc = np.transpose(mask, (1, 2, 0))
        mask_hwc = np.ascontiguousarray(mask_hwc)
        return mask_hwc
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")


def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """Extrait les vraies instances de PanNuke avec IDs séparés."""
    mask = normalize_mask_format(mask)

    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

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


def extract_gt_centroids(mask: np.ndarray) -> list:
    """Extrait les centroides des instances depuis masque GT PanNuke."""
    mask = normalize_mask_format(mask)

    centroids = []

    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            y_coords, x_coords = np.where(inst_mask)

            if len(y_coords) > 0:
                cy = np.mean(y_coords)
                cx = np.mean(x_coords)
                centroids.append((cy, cx))

    epithelial_mask = mask[:, :, 5]
    if epithelial_mask.max() > 0:
        _, labels = cv2.connectedComponents(epithelial_mask.astype(np.uint8))
        for inst_id in range(1, labels.max() + 1):
            inst_mask = labels == inst_id
            y_coords, x_coords = np.where(inst_mask)

            if len(y_coords) > 0:
                cy = np.mean(y_coords)
                cx = np.mean(x_coords)
                centroids.append((cy, cx))

    return centroids


def predict_centroids_from_hv(hv_map: np.ndarray, inst_map: np.ndarray) -> list:
    """Prédit les centroides depuis HV maps."""
    n_instances = int(inst_map.max())

    if n_instances == 0:
        return []

    grad_v = np.gradient(hv_map[0])
    grad_h = np.gradient(hv_map[1])

    mag = np.sqrt(grad_v[0]**2 + grad_v[1]**2 + grad_h[0]**2 + grad_h[1]**2)

    centroids = []
    for inst_id in range(1, n_instances + 1):
        inst_mask = (inst_map == inst_id)

        if not np.any(inst_mask):
            continue

        coords = np.argwhere(inst_mask)
        mags_in_inst = mag[inst_mask]

        min_idx = np.argmin(mags_in_inst)
        center_y, center_x = coords[min_idx]

        centroids.append((center_y, center_x))

    return centroids


def compute_mean_distance(pred_centroids: list, gt_centroids: list) -> float:
    """Calcule la distance moyenne avec Hungarian matching."""
    if len(pred_centroids) == 0 or len(gt_centroids) == 0:
        return float('inf')

    pred_coords = np.array(pred_centroids)
    gt_coords = np.array(gt_centroids)

    dist_matrix = cdist(pred_coords, gt_coords, metric='euclidean')
    pred_indices, gt_indices = linear_sum_assignment(dist_matrix)

    matched_distances = [dist_matrix[pred_idx, gt_idx] for pred_idx, gt_idx in zip(pred_indices, gt_indices)]

    return np.mean(matched_distances)


def test_configuration(hv_map: np.ndarray, inst_map: np.ndarray, gt_centroids: list, config_name: str) -> dict:
    """Teste une configuration HV et retourne les métriques."""
    pred_centroids = predict_centroids_from_hv(hv_map, inst_map)
    distance = compute_mean_distance(pred_centroids, gt_centroids)

    return {
        'name': config_name,
        'distance': distance,
        'n_pred': len(pred_centroids),
        'n_gt': len(gt_centroids)
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnostic HV Sign Inversion")
    parser.add_argument("--family", type=str, required=True,
                        choices=["glandular", "digestive", "urologic", "respiratory", "epidermal"])
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    print("="*80)
    print("DIAGNOSTIC HV SIGN INVERSION")
    print("="*80)
    print(f"Famille: {args.family}")
    print(f"Échantillons: {args.n_samples}")
    print()

    # Charger données
    data_file = Path(f"data/family_FIXED/{args.family}_data_FIXED.npz")

    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return 1

    data = np.load(data_file)
    images = data['images']
    hv_targets = data['hv_targets']
    fold_ids = data['fold_ids']
    image_ids = data['image_ids']

    print(f"✅ Données chargées: {len(images)} échantillons")
    print()

    np.random.seed(42)
    n_total = len(images)
    indices = np.random.choice(n_total, min(args.n_samples, n_total), replace=False)

    pannuke_dir = Path("/home/amar/data/PanNuke")

    results_by_config = {
        'original': [],
        'invertV': [],
        'invertH': [],
        'invertBoth': []
    }

    print("Test des 4 inversions de signe:")
    print("-"*80)

    for i, idx in enumerate(indices):
        hv_target = hv_targets[idx]
        fold_id = int(fold_ids[idx])
        image_id = int(image_ids[idx])

        gt_mask_file = pannuke_dir / f"fold{fold_id}" / "masks.npy"

        if not gt_mask_file.exists():
            print(f"  ❌ Masque GT non trouvé: {gt_mask_file}")
            continue

        masks = np.load(gt_mask_file, mmap_mode='r')
        gt_mask = masks[image_id]

        inst_map = extract_pannuke_instances(gt_mask)
        gt_centroids = extract_gt_centroids(gt_mask)

        # Tester 4 inversions de signe
        configs = {
            'original': hv_target,                                      # [+V, +H]
            'invertV': np.stack([-hv_target[0], hv_target[1]]),        # [-V, +H]
            'invertH': np.stack([hv_target[0], -hv_target[1]]),        # [+V, -H]
            'invertBoth': -hv_target                                    # [-V, -H]
        }

        print(f"  Sample {idx} (fold{fold_id}, img{image_id}):")

        for config_name, hv_test in configs.items():
            result = test_configuration(hv_test, inst_map, gt_centroids, config_name)
            results_by_config[config_name].append(result['distance'])

            status = "✅" if result['distance'] <= 2.0 else "❌"
            print(f"    {config_name:12s}: {result['distance']:6.2f}px {status}")

        print()

    # Résumé
    print("="*80)
    print("RÉSUMÉ PAR CONFIGURATION")
    print("="*80)
    print()

    best_config = None
    best_distance = float('inf')

    for config_name in ['original', 'invertV', 'invertH', 'invertBoth']:
        distances = results_by_config[config_name]

        if len(distances) == 0:
            continue

        mean_dist = np.mean(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        verdict = "✅ GO" if mean_dist <= 2.0 else "❌ NO-GO"

        print(f"{config_name.upper():12s}:")
        print(f"  Distance moyenne: {mean_dist:6.2f}px")
        print(f"  Distance min:     {min_dist:6.2f}px")
        print(f"  Distance max:     {max_dist:6.2f}px")
        print(f"  Verdict: {verdict}")
        print()

        if mean_dist < best_distance:
            best_distance = mean_dist
            best_config = config_name

    # Recommandation
    print("="*80)
    print("RECOMMANDATION")
    print("="*80)
    print()

    if best_distance <= 2.0:
        print(f"✅ Configuration GAGNANTE: {best_config.upper()}")
        print(f"   Distance moyenne: {best_distance:.2f}px")
        print()
        print("🔧 ACTIONS:")

        if best_config == 'original':
            print("   → Aucune modification nécessaire!")
        elif best_config == 'invertV':
            print("   → INVERSER LE SIGNE DE V dans compute_hv_maps():")
            print("   → Changer: v_dist = (y_coords - center_y) / bbox_h")
            print("   → En:      v_dist = (center_y - y_coords) / bbox_h")
        elif best_config == 'invertH':
            print("   → INVERSER LE SIGNE DE H dans compute_hv_maps():")
            print("   → Changer: h_dist = (x_coords - center_x) / bbox_w")
            print("   → En:      h_dist = (center_x - x_coords) / bbox_w")
        elif best_config == 'invertBoth':
            print("   → INVERSER LES DEUX SIGNES dans compute_hv_maps():")
            print("   → Changer: v_dist = (y_coords - center_y) / bbox_h")
            print("   →          h_dist = (x_coords - center_x) / bbox_w")
            print("   → En:      v_dist = (center_y - y_coords) / bbox_h")
            print("   →          h_dist = (center_x - x_coords) / bbox_w")

        print()
        print("   Puis régénérer les 5 familles")

    else:
        print(f"❌ AUCUNE CONFIGURATION < 2px")
        print(f"   Meilleure: {best_config.upper()} ({best_distance:.2f}px)")
        print()
        print("⚠️  PROBLÈME PLUS PROFOND - Vérifier:")
        print("   1. Formule de normalisation (bbox_h, bbox_w)")
        print("   2. Convention (y_coords - center_y) vs (center_y - y_coords)")
        print("   3. Gaussian smoothing (sigma=0.5 trop agressif?)")

    return 0 if best_distance <= 2.0 else 1


if __name__ == "__main__":
    exit(main())
