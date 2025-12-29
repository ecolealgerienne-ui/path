#!/usr/bin/env python3
"""
Vérification des données V13 Smart Crops.

Ce script vérifie que les données ont été générées avec les fixes critiques:
1. LOCAL relabeling (pas de collision d'IDs)
2. HV rotation mathematics correcte (vecteurs pointent vers l'intérieur)
3. HV targets en float32 [-1, 1] (pas int8)
4. inst_maps présents et valides

Usage:
    python scripts/validation/verify_v13_smart_crops_data.py --family epidermal
    python scripts/validation/verify_v13_smart_crops_data.py --data_file path/to/data.npz
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import sobel
import sys


def compute_hv_divergence(hv_map: np.ndarray, np_mask: np.ndarray) -> float:
    """
    Calcule la divergence des vecteurs HV sur les pixels de noyaux.

    Si les vecteurs HV pointent vers l'INTÉRIEUR des noyaux (correct),
    la divergence doit être NÉGATIVE.

    Si la divergence est positive, les vecteurs pointent vers l'extérieur
    (bug de rotation HV).

    Args:
        hv_map: (2, H, W) H et V components
        np_mask: (H, W) masque binaire des noyaux

    Returns:
        Divergence moyenne sur les pixels de noyaux
    """
    h_map = hv_map[0]  # Horizontal component
    v_map = hv_map[1]  # Vertical component

    # Gradient de H selon x, gradient de V selon y
    dh_dx = sobel(h_map, axis=1, mode='constant')
    dv_dy = sobel(v_map, axis=0, mode='constant')

    # Divergence = dH/dx + dV/dy
    divergence = dh_dx + dv_dy

    # Moyenne sur les pixels de noyaux uniquement
    mask = np_mask > 0.5
    if mask.sum() == 0:
        return 0.0

    return divergence[mask].mean()


def check_inst_map_ids(inst_map: np.ndarray) -> dict:
    """
    Vérifie les IDs dans inst_map pour détecter les collisions.

    LOCAL relabeling correct: IDs séquentiels [0, 1, 2, 3, ...]
    Bug collision: IDs non séquentiels ou gaps

    Returns:
        {
            'n_instances': int,
            'unique_ids': list,
            'is_sequential': bool,
            'has_gaps': bool,
            'max_id': int
        }
    """
    unique_ids = np.unique(inst_map)
    unique_ids = unique_ids[unique_ids > 0]  # Exclure background (0)

    n_instances = len(unique_ids)

    if n_instances == 0:
        return {
            'n_instances': 0,
            'unique_ids': [],
            'is_sequential': True,
            'has_gaps': False,
            'max_id': 0
        }

    expected_ids = set(range(1, n_instances + 1))
    actual_ids = set(unique_ids.tolist())

    is_sequential = (actual_ids == expected_ids)
    has_gaps = (max(unique_ids) > n_instances)

    return {
        'n_instances': n_instances,
        'unique_ids': sorted(unique_ids.tolist()),
        'is_sequential': is_sequential,
        'has_gaps': has_gaps,
        'max_id': int(max(unique_ids))
    }


def verify_hv_targets(hv_targets: np.ndarray) -> dict:
    """
    Vérifie le format des HV targets.

    Correct: float32, range [-1, 1]
    Bug #3: int8, range [-127, 127]

    Returns:
        {
            'dtype': str,
            'min': float,
            'max': float,
            'is_float32': bool,
            'is_correct_range': bool,
            'is_bug3': bool
        }
    """
    dtype_str = str(hv_targets.dtype)
    min_val = float(hv_targets.min())
    max_val = float(hv_targets.max())

    is_float32 = (hv_targets.dtype == np.float32)
    is_correct_range = (-1.5 <= min_val <= 0) and (0 <= max_val <= 1.5)
    is_bug3 = (hv_targets.dtype == np.int8) or (abs(min_val) > 10 or abs(max_val) > 10)

    return {
        'dtype': dtype_str,
        'min': min_val,
        'max': max_val,
        'is_float32': is_float32,
        'is_correct_range': is_correct_range,
        'is_bug3': is_bug3
    }


def verify_data_file(data_path: Path, n_samples: int = 10, verbose: bool = True) -> dict:
    """
    Vérifie un fichier de données V13 Smart Crops.

    Args:
        data_path: Chemin vers le fichier .npz
        n_samples: Nombre d'échantillons à vérifier
        verbose: Afficher les détails

    Returns:
        Dictionnaire avec résultats de vérification
    """
    if not data_path.exists():
        return {'error': f"Fichier non trouvé: {data_path}"}

    print(f"\n{'='*70}")
    print(f"VÉRIFICATION: {data_path.name}")
    print(f"{'='*70}\n")

    # Charger les données
    data = np.load(data_path, allow_pickle=True)

    available_keys = list(data.keys())
    print(f"Clés disponibles: {available_keys}")

    results = {
        'file': str(data_path),
        'keys': available_keys,
        'checks': {}
    }

    # 1. Vérifier HV targets
    print(f"\n{'─'*50}")
    print("1. VÉRIFICATION HV TARGETS")
    print(f"{'─'*50}")

    if 'hv_targets' in data:
        hv_targets = data['hv_targets']
        print(f"   Shape: {hv_targets.shape}")

        hv_check = verify_hv_targets(hv_targets)
        results['checks']['hv_targets'] = hv_check

        if hv_check['is_float32'] and hv_check['is_correct_range']:
            print(f"   ✅ Dtype: {hv_check['dtype']} (correct)")
            print(f"   ✅ Range: [{hv_check['min']:.4f}, {hv_check['max']:.4f}] (correct)")
        elif hv_check['is_bug3']:
            print(f"   ❌ BUG #3 DÉTECTÉ!")
            print(f"   ❌ Dtype: {hv_check['dtype']} (devrait être float32)")
            print(f"   ❌ Range: [{hv_check['min']:.4f}, {hv_check['max']:.4f}] (devrait être [-1, 1])")
        else:
            print(f"   ⚠️ Dtype: {hv_check['dtype']}")
            print(f"   ⚠️ Range: [{hv_check['min']:.4f}, {hv_check['max']:.4f}]")
    else:
        print("   ⚠️ Clé 'hv_targets' non trouvée")
        results['checks']['hv_targets'] = {'error': 'key not found'}

    # 2. Vérifier inst_maps (LOCAL relabeling)
    print(f"\n{'─'*50}")
    print("2. VÉRIFICATION INST_MAPS (LOCAL relabeling)")
    print(f"{'─'*50}")

    if 'inst_maps' in data:
        inst_maps = data['inst_maps']
        print(f"   Shape: {inst_maps.shape}")

        # Vérifier quelques échantillons
        n_to_check = min(n_samples, len(inst_maps))
        sequential_count = 0
        collision_samples = []

        for i in range(n_to_check):
            inst_check = check_inst_map_ids(inst_maps[i])
            if inst_check['is_sequential']:
                sequential_count += 1
            else:
                collision_samples.append({
                    'index': i,
                    'n_instances': inst_check['n_instances'],
                    'max_id': inst_check['max_id'],
                    'unique_ids': inst_check['unique_ids'][:10]  # Premiers 10
                })

        results['checks']['inst_maps'] = {
            'sequential_count': sequential_count,
            'total_checked': n_to_check,
            'collision_samples': collision_samples
        }

        if sequential_count == n_to_check:
            print(f"   ✅ {sequential_count}/{n_to_check} échantillons avec IDs séquentiels")
            print(f"   ✅ LOCAL relabeling correctement appliqué")
        else:
            print(f"   ❌ {n_to_check - sequential_count}/{n_to_check} échantillons avec COLLISIONS")
            print(f"   ❌ LOCAL relabeling NON appliqué ou buggé")
            if collision_samples:
                print(f"   ❌ Exemple collision (sample {collision_samples[0]['index']}):")
                print(f"      n_instances={collision_samples[0]['n_instances']}, max_id={collision_samples[0]['max_id']}")
    else:
        print("   ⚠️ Clé 'inst_maps' non trouvée")
        print("   ⚠️ Les inst_maps sont REQUIS pour évaluation AJI correcte!")
        results['checks']['inst_maps'] = {'error': 'key not found'}

    # 3. Vérifier divergence HV (rotation math)
    print(f"\n{'─'*50}")
    print("3. VÉRIFICATION DIVERGENCE HV (rotation math)")
    print(f"{'─'*50}")

    if 'hv_targets' in data and 'np_targets' in data:
        hv_targets = data['hv_targets']
        np_targets = data['np_targets']

        n_to_check = min(n_samples, len(hv_targets))
        divergences = []
        negative_count = 0

        for i in range(n_to_check):
            div = compute_hv_divergence(hv_targets[i], np_targets[i])
            divergences.append(div)
            if div < 0:
                negative_count += 1

        mean_div = np.mean(divergences)

        results['checks']['hv_divergence'] = {
            'mean_divergence': float(mean_div),
            'negative_count': negative_count,
            'total_checked': n_to_check,
            'divergences': divergences
        }

        if negative_count == n_to_check and mean_div < 0:
            print(f"   ✅ Divergence moyenne: {mean_div:.4f} (négative = correct)")
            print(f"   ✅ {negative_count}/{n_to_check} échantillons avec divergence négative")
            print(f"   ✅ Rotation HV math correcte (vecteurs pointent vers l'intérieur)")
        elif mean_div > 0:
            print(f"   ❌ Divergence moyenne: {mean_div:.4f} (POSITIVE = BUG!)")
            print(f"   ❌ Seulement {negative_count}/{n_to_check} échantillons corrects")
            print(f"   ❌ BUG ROTATION HV: Vecteurs pointent vers l'EXTÉRIEUR!")
        else:
            print(f"   ⚠️ Divergence moyenne: {mean_div:.4f}")
            print(f"   ⚠️ {negative_count}/{n_to_check} échantillons avec divergence négative")
    else:
        print("   ⚠️ Clés 'hv_targets' ou 'np_targets' non trouvées")
        results['checks']['hv_divergence'] = {'error': 'keys not found'}

    # 4. Vérifier cohérence shapes
    print(f"\n{'─'*50}")
    print("4. VÉRIFICATION COHÉRENCE SHAPES")
    print(f"{'─'*50}")

    shapes = {}
    for key in ['images', 'np_targets', 'hv_targets', 'nt_targets', 'inst_maps']:
        if key in data:
            shapes[key] = data[key].shape
            print(f"   {key}: {data[key].shape}")

    results['shapes'] = shapes

    # Vérifier cohérence
    n_samples_list = [s[0] for s in shapes.values()]
    if len(set(n_samples_list)) == 1:
        print(f"   ✅ Toutes les arrays ont {n_samples_list[0]} échantillons")
    else:
        print(f"   ❌ INCOHÉRENCE: Nombres d'échantillons différents!")

    # 5. Résumé
    print(f"\n{'='*70}")
    print("RÉSUMÉ")
    print(f"{'='*70}")

    all_ok = True

    # HV targets check
    hv_ok = results['checks'].get('hv_targets', {}).get('is_float32', False) and \
            results['checks'].get('hv_targets', {}).get('is_correct_range', False)
    if hv_ok:
        print("✅ HV targets: float32 [-1, 1]")
    else:
        print("❌ HV targets: PROBLÈME DÉTECTÉ")
        all_ok = False

    # inst_maps check
    inst_ok = 'inst_maps' in data and \
              results['checks'].get('inst_maps', {}).get('sequential_count', 0) == \
              results['checks'].get('inst_maps', {}).get('total_checked', 0)
    if inst_ok:
        print("✅ inst_maps: LOCAL relabeling OK")
    elif 'inst_maps' not in data:
        print("❌ inst_maps: MANQUANTS (requis pour AJI)")
        all_ok = False
    else:
        print("❌ inst_maps: COLLISIONS DÉTECTÉES")
        all_ok = False

    # HV divergence check
    div_ok = results['checks'].get('hv_divergence', {}).get('mean_divergence', 1) < 0
    if div_ok:
        print("✅ HV divergence: Rotation math OK")
    else:
        print("❌ HV divergence: BUG ROTATION")
        all_ok = False

    print(f"\n{'='*70}")
    if all_ok:
        print("🎉 VERDICT: Données V13 Smart Crops VALIDES")
        print("   Prêtes pour entraînement avec les fixes critiques.")
    else:
        print("⚠️ VERDICT: Données V13 Smart Crops INVALIDES")
        print("   Régénérer avec: python scripts/preprocessing/prepare_v13_smart_crops.py")
    print(f"{'='*70}\n")

    results['all_ok'] = all_ok
    return results


def main():
    parser = argparse.ArgumentParser(description="Vérifier les données V13 Smart Crops")
    parser.add_argument("--family", type=str, default="epidermal",
                        help="Famille à vérifier (epidermal, glandular, etc.)")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Chemin direct vers le fichier .npz")
    parser.add_argument("--data_dir", type=str,
                        default="data/family_data_v13_smart_crops",
                        help="Répertoire des données V13")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Nombre d'échantillons à vérifier")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "all"],
                        help="Split à vérifier (train, val, ou all pour les deux)")

    args = parser.parse_args()

    # Déterminer les splits à vérifier
    if args.split == "all":
        splits_to_check = ["train", "val"]
    else:
        splits_to_check = [args.split]

    all_results = {}
    all_ok = True

    for split in splits_to_check:
        if args.data_file:
            data_path = Path(args.data_file)
        else:
            data_path = Path(args.data_dir) / f"{args.family}_{split}_v13_smart_crops.npz"

        results = verify_data_file(data_path, n_samples=args.n_samples)
        all_results[split] = results

        if 'error' in results:
            print(f"❌ ERREUR ({split}): {results['error']}")
            all_ok = False
        elif not results.get('all_ok', False):
            all_ok = False

    # Résumé final si plusieurs splits
    if len(splits_to_check) > 1:
        print(f"\n{'='*70}")
        print("RÉSUMÉ GLOBAL (TRAIN + VAL)")
        print(f"{'='*70}")
        for split, results in all_results.items():
            if 'error' in results:
                print(f"  {split.upper()}: ❌ ERREUR - {results['error']}")
            elif results.get('all_ok', False):
                print(f"  {split.upper()}: ✅ VALIDE")
            else:
                print(f"  {split.upper()}: ❌ INVALIDE")
        print(f"{'='*70}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
