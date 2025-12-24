#!/usr/bin/env python3
"""
DIAGNOSTIC HV FLIP/MIRROR (Expert-Guided 2025-12-24)

Teste 4 configurations pour identifier la transformation correcte:
1. Original (aucune modification)
2. FlipUD (flip vertical - Y inversé)
3. FlipLR (flip horizontal - X inversé)
4. Rot180 (rotation 180°)

Objectif: Trouver quelle config donne distance <2px

Usage:
    python scripts/validation/diagnose_hv_flip.py --family epidermal
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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

    # Canaux 1-4: IDs d'instances natifs PanNuke
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
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

    # Canaux 1-4: instances annotées
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

    # Canal 5 (Epithelial)
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

    # Calculer magnitude du GRADIENT HV
    grad_v = np.gradient(hv_map[0])  # [dV/dy, dV/dx]
    grad_h = np.gradient(hv_map[1])  # [dH/dy, dH/dx]

    # Magnitude combinée
    mag = np.sqrt(grad_v[0]**2 + grad_v[1]**2 + grad_h[0]**2 + grad_h[1]**2)

    # Pour chaque instance, trouver pixel de magnitude MINIMALE
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
    parser = argparse.ArgumentParser(description="Diagnostic HV Flip/Mirror")
    parser.add_argument("--family", type=str, required=True,
                        choices=["glandular", "digestive", "urologic", "respiratory", "epidermal"])
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Nombre d'échantillons à tester")
    args = parser.parse_args()

    print("="*80)
    print("DIAGNOSTIC HV FLIP/MIRROR")
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
    fold_ids = data['fold_ids']
    image_ids = data['image_ids']

    print(f"✅ Données chargées: {len(images)} échantillons")
    print()

    # 2. Sélectionner échantillons aléatoires
    np.random.seed(42)
    n_total = len(images)
    indices = np.random.choice(n_total, min(args.n_samples, n_total), replace=False)

    pannuke_dir = Path("/home/amar/data/PanNuke")

    # Accumuler résultats par configuration
    results_by_config = {
        'original': [],
        'flipud': [],
        'fliplr': [],
        'rot180': []
    }

    print("Test des 4 configurations:")
    print("-"*80)

    for i, idx in enumerate(indices):
        hv_target = hv_targets[idx]
        fold_id = int(fold_ids[idx])
        image_id = int(image_ids[idx])

        # Charger GT mask
        gt_mask_file = pannuke_dir / f"fold{fold_id}" / "masks.npy"

        if not gt_mask_file.exists():
            print(f"  ❌ Masque GT non trouvé: {gt_mask_file}")
            continue

        masks = np.load(gt_mask_file, mmap_mode='r')
        gt_mask = masks[image_id]

        # Extraire inst_map et centroides GT
        inst_map = extract_pannuke_instances(gt_mask)
        gt_centroids = extract_gt_centroids(gt_mask)

        # Tester 4 configurations
        configs = {
            'original': hv_target,
            'flipud': np.flip(hv_target, axis=1),      # Flip vertical (axe Y)
            'fliplr': np.flip(hv_target, axis=2),      # Flip horizontal (axe X)
            'rot180': np.rot90(hv_target, k=2, axes=(1, 2))  # Rotation 180°
        }

        print(f"  Sample {idx} (fold{fold_id}, img{image_id}):")

        for config_name, hv_test in configs.items():
            result = test_configuration(hv_test, inst_map, gt_centroids, config_name)
            results_by_config[config_name].append(result['distance'])

            status = "✅" if result['distance'] <= 2.0 else "❌"
            print(f"    {config_name:10s}: {result['distance']:6.2f}px {status}")

        print()

    # 3. Résumé final
    print("="*80)
    print("RÉSUMÉ PAR CONFIGURATION")
    print("="*80)
    print()

    best_config = None
    best_distance = float('inf')

    for config_name in ['original', 'flipud', 'fliplr', 'rot180']:
        distances = results_by_config[config_name]

        if len(distances) == 0:
            continue

        mean_dist = np.mean(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        verdict = "✅ GO" if mean_dist <= 2.0 else "❌ NO-GO"

        print(f"{config_name.upper():10s}:")
        print(f"  Distance moyenne: {mean_dist:6.2f}px")
        print(f"  Distance min:     {min_dist:6.2f}px")
        print(f"  Distance max:     {max_dist:6.2f}px")
        print(f"  Verdict: {verdict}")
        print()

        if mean_dist < best_distance:
            best_distance = mean_dist
            best_config = config_name

    # 4. Recommandation
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
            print("   → Les données sont DÉJÀ correctes")
        elif best_config == 'flipud':
            print("   → Appliquer FLIP VERTICAL dans prepare_family_data_FIXED_v6.py")
            print("   → Ajouter: hv_map = np.flip(hv_map, axis=1)")
        elif best_config == 'fliplr':
            print("   → Appliquer FLIP HORIZONTAL dans prepare_family_data_FIXED_v6.py")
            print("   → Ajouter: hv_map = np.flip(hv_map, axis=2)")
        elif best_config == 'rot180':
            print("   → Appliquer ROTATION 180° dans prepare_family_data_FIXED_v6.py")
            print("   → Ajouter: hv_map = np.rot90(hv_map, k=2, axes=(1,2))")

        print()
        print("   Puis régénérer les 5 familles avec le fix")

    else:
        print(f"❌ AUCUNE CONFIGURATION < 2px")
        print(f"   Meilleure: {best_config.upper()} ({best_distance:.2f}px)")
        print()
        print("⚠️  PROBLÈME PLUS COMPLEXE:")
        print("   - Possible rotation autre que 90°/180°/270°")
        print("   - Possible scaling/distortion")
        print("   - Vérifier compute_hv_maps() dans prepare_family_data_FIXED_v6.py")

    return 0 if best_distance <= 2.0 else 1


if __name__ == "__main__":
    exit(main())
