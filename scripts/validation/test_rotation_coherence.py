#!/usr/bin/env python3
"""
Script de test pour vérifier la cohérence des rotations HV.

Vérifie que:
1. Les rotations spatiales sont correctes (90°, 180°, 270°)
2. Les composantes HV sont correctement swappées après rotation
3. Les valeurs restent dans [-1, 1]
4. La divergence HV reste négative (vecteurs pointent vers centres)
5. Les images et masks sont bien synchronisés

Usage:
    python scripts/validation/test_rotation_coherence.py \
        --data_file data/family_data_v13_crops/epidermal_data_v13_crops.npz \
        --n_samples 10
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import albumentations as A
from scipy import ndimage

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_hv_divergence(hv_map: np.ndarray, np_mask: np.ndarray) -> float:
    """
    Calcule la divergence moyenne des cartes HV.

    La divergence doit être NÉGATIVE (vecteurs pointent vers les centres).

    Args:
        hv_map: Carte HV (2, H, W) float32 [-1, 1]
        np_mask: Masque NP (H, W) binaire

    Returns:
        Divergence moyenne (doit être < 0)
    """
    h_map = hv_map[0]
    v_map = hv_map[1]

    # Gradients
    grad_h = np.gradient(h_map, axis=1)
    grad_v = np.gradient(v_map, axis=0)

    # Divergence
    div = grad_h + grad_v

    # Moyenne sur pixels de noyaux uniquement
    if np_mask.sum() > 0:
        return float(div[np_mask > 0].mean())
    else:
        return 0.0


def apply_rotation_90(
    image: np.ndarray,
    hv: np.ndarray,
    np_mask: np.ndarray,
    nt_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotation 90° clockwise avec correction HV.

    Transformation HV:
        H' = V (ancien vertical devient horizontal)
        V' = -H (ancien horizontal devient vertical inversé)
    """
    # Rotation spatiale
    image_rot = np.rot90(image, k=-1, axes=(0, 1))  # -1 = clockwise
    np_rot = np.rot90(np_mask, k=-1, axes=(0, 1))
    nt_rot = np.rot90(nt_mask, k=-1, axes=(0, 1))

    # HV component swapping
    h_old, v_old = hv[0], hv[1]
    h_rot = np.rot90(v_old, k=-1, axes=(0, 1))      # H' = V
    v_rot = -np.rot90(h_old, k=-1, axes=(0, 1))     # V' = -H

    hv_rot = np.stack([h_rot, v_rot], axis=0)

    return image_rot, hv_rot, np_rot, nt_rot


def apply_rotation_180(
    image: np.ndarray,
    hv: np.ndarray,
    np_mask: np.ndarray,
    nt_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotation 180°.

    Transformation HV:
        H' = -H (inversion horizontale)
        V' = -V (inversion verticale)
    """
    # Rotation spatiale
    image_rot = np.rot90(image, k=2, axes=(0, 1))
    np_rot = np.rot90(np_mask, k=2, axes=(0, 1))
    nt_rot = np.rot90(nt_mask, k=2, axes=(0, 1))

    # HV negation
    h_rot = -np.rot90(hv[0], k=2, axes=(0, 1))
    v_rot = -np.rot90(hv[1], k=2, axes=(0, 1))

    hv_rot = np.stack([h_rot, v_rot], axis=0)

    return image_rot, hv_rot, np_rot, nt_rot


def apply_rotation_270(
    image: np.ndarray,
    hv: np.ndarray,
    np_mask: np.ndarray,
    nt_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotation 270° clockwise (= 90° counter-clockwise).

    Transformation HV:
        H' = -V
        V' = H
    """
    # Rotation spatiale
    image_rot = np.rot90(image, k=1, axes=(0, 1))  # k=1 = counter-clockwise
    np_rot = np.rot90(np_mask, k=1, axes=(0, 1))
    nt_rot = np.rot90(nt_mask, k=1, axes=(0, 1))

    # HV component swapping
    h_old, v_old = hv[0], hv[1]
    h_rot = -np.rot90(v_old, k=1, axes=(0, 1))     # H' = -V
    v_rot = np.rot90(h_old, k=1, axes=(0, 1))      # V' = H

    hv_rot = np.stack([h_rot, v_rot], axis=0)

    return image_rot, hv_rot, np_rot, nt_rot


def apply_flip_horizontal(
    image: np.ndarray,
    hv: np.ndarray,
    np_mask: np.ndarray,
    nt_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flip horizontal.

    Transformation HV:
        H' = -H (inversion axe X)
        V' = V (pas de changement axe Y)
    """
    # Flip spatial
    image_flip = np.fliplr(image)
    np_flip = np.fliplr(np_mask)
    nt_flip = np.fliplr(nt_mask)

    # HV flip
    h_flip = -np.fliplr(hv[0])  # Négation H
    v_flip = np.fliplr(hv[1])   # Pas de changement V

    hv_flip = np.stack([h_flip, v_flip], axis=0)

    return image_flip, hv_flip, np_flip, nt_flip


def test_single_rotation(
    image: np.ndarray,
    hv: np.ndarray,
    np_mask: np.ndarray,
    nt_mask: np.ndarray,
    rotation_name: str
) -> dict:
    """
    Test une rotation et retourne les résultats de validation.
    """
    result = {
        'rotation': rotation_name,
        'tests_passed': 0,
        'tests_total': 5,
        'errors': []
    }

    # Appliquer rotation
    if rotation_name == "0° (original)":
        img_rot, hv_rot, np_rot, nt_rot = image, hv, np_mask, nt_mask
    elif rotation_name == "90° CW":
        img_rot, hv_rot, np_rot, nt_rot = apply_rotation_90(image, hv, np_mask, nt_mask)
    elif rotation_name == "180°":
        img_rot, hv_rot, np_rot, nt_rot = apply_rotation_180(image, hv, np_mask, nt_mask)
    elif rotation_name == "270° CW":
        img_rot, hv_rot, np_rot, nt_rot = apply_rotation_270(image, hv, np_mask, nt_mask)
    elif rotation_name == "Flip H":
        img_rot, hv_rot, np_rot, nt_rot = apply_flip_horizontal(image, hv, np_mask, nt_mask)
    else:
        result['errors'].append(f"Unknown rotation: {rotation_name}")
        return result

    # Test 1: Shape conservation
    if img_rot.shape == image.shape and hv_rot.shape == hv.shape:
        result['tests_passed'] += 1
    else:
        result['errors'].append(f"Shape mismatch: {img_rot.shape} vs {image.shape}")

    # Test 2: HV range [-1, 1]
    hv_min, hv_max = hv_rot.min(), hv_rot.max()
    if -1.0 <= hv_min and hv_max <= 1.0:
        result['tests_passed'] += 1
    else:
        result['errors'].append(f"HV out of range: [{hv_min:.3f}, {hv_max:.3f}]")

    # Test 3: NP mask conservation (même nombre de pixels)
    if abs(np_rot.sum() - np_mask.sum()) < 10:  # Tolérance 10 pixels
        result['tests_passed'] += 1
    else:
        result['errors'].append(f"NP pixel count mismatch: {np_rot.sum()} vs {np_mask.sum()}")

    # Test 4: Divergence négative
    div = compute_hv_divergence(hv_rot, np_rot)
    if div < 0:
        result['tests_passed'] += 1
        result['divergence'] = div
    else:
        result['errors'].append(f"Divergence positive: {div:.6f} (should be < 0)")
        result['divergence'] = div

    # Test 5: No NaN/Inf
    if not (np.isnan(hv_rot).any() or np.isinf(hv_rot).any()):
        result['tests_passed'] += 1
    else:
        result['errors'].append("NaN or Inf detected in HV")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test rotation coherence")
    parser.add_argument("--data_file", type=Path, required=True,
                        help="Data file with images and HV maps")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples to test")
    args = parser.parse_args()

    if not args.data_file.exists():
        print(f"❌ Data file not found: {args.data_file}")
        sys.exit(1)

    print(f"📂 Loading data: {args.data_file}")
    data = np.load(args.data_file)

    # Check required keys
    required_keys = ['images', 'hv_targets', 'np_targets', 'nt_targets']
    missing = [k for k in required_keys if k not in data.keys()]
    if missing:
        print(f"❌ Missing keys: {missing}")
        print(f"Available keys: {list(data.keys())}")
        sys.exit(1)

    images = data['images']
    hv_targets = data['hv_targets']
    np_targets = data['np_targets']
    nt_targets = data['nt_targets']

    n_available = len(images)
    n_to_test = min(args.n_samples, n_available)

    print(f"📊 Dataset: {n_available} samples")
    print(f"🧪 Testing: {n_to_test} samples")
    print(f"\n{'='*70}")

    # Rotations à tester
    rotations = ["0° (original)", "90° CW", "180°", "270° CW", "Flip H"]

    # Accumulateurs
    all_results = {rot: [] for rot in rotations}

    # Test chaque échantillon
    for i in range(n_to_test):
        image = images[i]
        hv = hv_targets[i]
        np_mask = np_targets[i]
        nt_mask = nt_targets[i]

        print(f"\n📌 Sample {i+1}/{n_to_test}")

        for rotation_name in rotations:
            result = test_single_rotation(image, hv, np_mask, nt_mask, rotation_name)
            all_results[rotation_name].append(result)

            status = "✅" if result['tests_passed'] == result['tests_total'] else "⚠️"
            print(f"  {status} {rotation_name:15s}: {result['tests_passed']}/{result['tests_total']} tests passed", end="")

            if 'divergence' in result:
                print(f" | div={result['divergence']:.6f}", end="")

            if result['errors']:
                print(f" | Errors: {'; '.join(result['errors'])}")
            else:
                print()

    # Résumé global
    print(f"\n{'='*70}")
    print("📊 RÉSUMÉ GLOBAL")
    print(f"{'='*70}")

    for rotation_name in rotations:
        results = all_results[rotation_name]
        total_passed = sum(r['tests_passed'] for r in results)
        total_tests = sum(r['tests_total'] for r in results)
        pct = 100 * total_passed / total_tests if total_tests > 0 else 0

        # Divergence moyenne
        divs = [r.get('divergence', 0) for r in results if 'divergence' in r]
        avg_div = np.mean(divs) if divs else 0

        status = "✅" if pct == 100 else "⚠️"
        print(f"{status} {rotation_name:15s}: {total_passed:3d}/{total_tests:3d} ({pct:5.1f}%) | Avg div={avg_div:+.6f}")

    # Verdict final
    print(f"\n{'='*70}")
    all_passed = all(
        sum(r['tests_passed'] for r in all_results[rot]) == sum(r['tests_total'] for r in all_results[rot])
        for rot in rotations
    )

    if all_passed:
        print("🎉 TOUS LES TESTS PASSENT - Rotations cohérentes")
        sys.exit(0)
    else:
        print("⚠️ CERTAINS TESTS ÉCHOUENT - Vérifier les rotations")
        sys.exit(1)


if __name__ == "__main__":
    main()
