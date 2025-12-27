#!/usr/bin/env python3
"""
Test de validation des rotations HV AVANT génération complète des données.

Ce script teste la logique de rotation HV sur UNE seule image PanNuke
pour éviter de perdre du temps si la logique est incorrecte.

Usage:
    python scripts/validation/test_hv_rotation_BEFORE_generation.py

Critères de validation:
    1. Ordre des canaux correct (H à index 0, V à index 1)
    2. Divergence correcte selon direction vecteurs
    3. Range HV dans [-1, 1]
    4. Les 5 rotations cohérentes

Référence: Graham et al. 2019 (HoVer-Net)
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from prepare script
from scripts.preprocessing.prepare_v13_smart_crops import (
    extract_pannuke_instances,
    compute_hv_maps,
    apply_rotation
)


def compute_divergence(hv_map: np.ndarray, mask: np.ndarray) -> float:
    """
    Calcule la divergence d'un champ de vecteurs HV.

    Convention HoVer-Net standard:
        - hv_map[0] = H (composante horizontale/X)
        - hv_map[1] = V (composante verticale/Y)

    Divergence = ∂H/∂x + ∂V/∂y

    Args:
        hv_map: (2, H, W) avec [H-channel, V-channel]
        mask: (H, W) masque binaire des noyaux

    Returns:
        Divergence moyenne sur les pixels masqués
    """
    h_map = hv_map[0]  # Horizontal (X)
    v_map = hv_map[1]  # Vertical (Y)

    # Gradients
    grad_h_x = np.gradient(h_map, axis=1)  # ∂H/∂x
    grad_v_y = np.gradient(v_map, axis=0)  # ∂V/∂y

    divergence = grad_h_x + grad_v_y

    if mask.sum() == 0:
        return 0.0

    return float(divergence[mask > 0].mean())


def test_single_rotation(
    image: np.ndarray,
    np_target: np.ndarray,
    hv_target: np.ndarray,
    nt_target: np.ndarray,
    rotation: str
) -> dict:
    """
    Teste une rotation spécifique.

    Returns:
        dict avec clés: 'passed', 'errors', 'divergence', 'hv_range'
    """
    errors = []

    # Appliquer rotation
    img_rot, np_rot, hv_rot, nt_rot = apply_rotation(
        image, np_target, hv_target, nt_target, rotation
    )

    # Test 1: Shape conservation
    if hv_rot.shape != hv_target.shape:
        errors.append(f"Shape mismatch: {hv_rot.shape} != {hv_target.shape}")

    # Test 2: HV range [-1, 1]
    hv_min, hv_max = hv_rot.min(), hv_rot.max()
    if not (-1.0 <= hv_min <= hv_max <= 1.0):
        errors.append(f"HV range invalid: [{hv_min:.3f}, {hv_max:.3f}] (expected [-1, 1])")

    # Test 3: NP mask pixels conservation (environ)
    np_orig_pixels = np_target.sum()
    np_rot_pixels = np_rot.sum()
    pixel_diff_ratio = abs(np_rot_pixels - np_orig_pixels) / max(np_orig_pixels, 1)
    if pixel_diff_ratio > 0.05:  # Tolérance 5%
        errors.append(f"NP pixels changed: {np_orig_pixels} -> {np_rot_pixels} ({pixel_diff_ratio*100:.1f}%)")

    # Test 4: Divergence
    div = compute_divergence(hv_rot, np_rot > 0)

    # Test 5: Pas de NaN/Inf
    if np.isnan(hv_rot).any() or np.isinf(hv_rot).any():
        errors.append("NaN or Inf detected in HV")

    return {
        'passed': len(errors) == 0,
        'errors': errors,
        'divergence': div,
        'hv_range': (hv_min, hv_max)
    }


def load_test_image(pannuke_dir: Path = Path("/home/amar/data/PanNuke")):
    """
    Charge UNE image de test depuis PanNuke fold 0.

    Returns:
        image (256, 256, 3), mask (256, 256, 6), organ_name
    """
    fold_dir = pannuke_dir / "fold0"
    images_path = fold_dir / "images.npy"
    masks_path = fold_dir / "masks.npy"
    types_path = fold_dir / "types.npy"

    if not images_path.exists():
        raise FileNotFoundError(f"PanNuke fold0 not found at {fold_dir}")

    print(f"📂 Loading test image from {fold_dir}")

    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')
    types = np.load(types_path)

    # Prendre la première image avec des noyaux
    for i in range(min(100, len(images))):
        mask = np.array(masks[i])
        if mask[:, :, 1:5].sum() > 1000:  # Au moins 1000 pixels de noyaux
            image = np.array(images[i], dtype=np.uint8)
            organ = types[i].decode('utf-8') if isinstance(types[i], bytes) else types[i]
            print(f"✅ Loaded image {i}: {organ}")
            return image, mask, organ

    raise ValueError("No suitable test image found with enough nuclei")


def main():
    """Test principal."""
    print("=" * 70)
    print("TEST ROTATION HV - AVANT GÉNÉRATION COMPLÈTE")
    print("=" * 70)
    print()

    # 1. Charger une image de test
    try:
        image, mask, organ = load_test_image()
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return 1

    # 2. Extraire crop central 224×224
    print(f"\n📐 Extraction crop central 224×224...")
    x1, y1 = 16, 16
    x2, y2 = 240, 240

    image_crop = image[y1:y2, x1:x2, :]
    mask_crop = mask[y1:y2, x1:x2, :]

    # 3. Générer targets (comme dans prepare_v13_smart_crops.py)
    print(f"🔧 Génération targets HV...")

    # Extract instances
    inst_map = extract_pannuke_instances(mask_crop)

    # Compute HV maps
    hv_target = compute_hv_maps(inst_map)

    # NP target
    np_target = (mask_crop[:, :, 1:5].sum(axis=-1) > 0).astype(np.float32)

    # NT target
    nt_target = np.zeros((224, 224), dtype=np.int64)

    print(f"✅ Instances détectées: {len(np.unique(inst_map)) - 1}")
    print(f"✅ HV shape: {hv_target.shape}")
    print(f"✅ HV range: [{hv_target.min():.3f}, {hv_target.max():.3f}]")

    # 4. Vérifier l'ordre des canaux (H vs V)
    print(f"\n🔍 Vérification ordre des canaux (convention HoVer-Net)...")
    print(f"   hv_target[0] = ? (attendu: H - Horizontal)")
    print(f"   hv_target[1] = ? (attendu: V - Vertical)")

    # 5. Tester les 5 rotations
    print(f"\n🔄 Test des 5 rotations...")
    print()

    rotations = ['0', '90', '180', '270', 'flip_h']
    results = {}

    for rotation in rotations:
        result = test_single_rotation(
            image_crop, np_target, hv_target, nt_target, rotation
        )
        results[rotation] = result

        status = "✅" if result['passed'] else "❌"
        div_sign = "+" if result['divergence'] >= 0 else ""

        print(f"{status} {rotation:12s}: div={div_sign}{result['divergence']:.6f}, "
              f"range=[{result['hv_range'][0]:.3f}, {result['hv_range'][1]:.3f}]")

        if not result['passed']:
            for error in result['errors']:
                print(f"     ⚠️  {error}")

    # 6. Analyse de la divergence
    print()
    print("=" * 70)
    print("📊 ANALYSE DIVERGENCE")
    print("=" * 70)

    divergences = [r['divergence'] for r in results.values()]
    div_mean = np.mean(divergences)
    div_std = np.std(divergences)

    print(f"Moyenne: {div_mean:.6f}")
    print(f"Std:     {div_std:.6f}")
    print()

    # Déterminer direction des vecteurs
    if abs(div_mean) > 0.001:
        if div_mean > 0:
            print("📌 Vecteurs pointent vers l'EXTÉRIEUR (x - cx)")
            print("   → Divergence POSITIVE attendue")
        else:
            print("📌 Vecteurs pointent vers l'INTÉRIEUR (cx - x)")
            print("   → Divergence NÉGATIVE attendue")
    else:
        print("⚠️  Divergence proche de zéro - vérifier calcul HV maps")

    # 7. Vérifier cohérence rotations
    print()
    print("=" * 70)
    print("🔍 COHÉRENCE ROTATIONS")
    print("=" * 70)

    # Les rotations 0° et 180° devraient avoir divergences de signes opposés
    # Les rotations 90° et 270° devraient avoir divergences de signes opposés
    div_0 = results['0']['divergence']
    div_90 = results['90']['divergence']
    div_180 = results['180']['divergence']
    div_270 = results['270']['divergence']
    div_flip = results['flip_h']['divergence']

    checks = []

    # Check 1: 0° et 180° opposés
    if np.sign(div_0) == -np.sign(div_180):
        checks.append("✅ 0° et 180° ont divergences opposées")
    else:
        checks.append("❌ 0° et 180° devraient avoir divergences opposées")

    # Check 2: 90° et 270° opposés
    if np.sign(div_90) == -np.sign(div_270):
        checks.append("✅ 90° et 270° ont divergences opposées")
    else:
        checks.append("❌ 90° et 270° devraient avoir divergences opposées")

    # Check 3: Toutes rotations même signe de divergence (idéalement)
    all_same_sign = len(set(np.sign(d) for d in divergences if abs(d) > 0.001)) == 1
    if all_same_sign:
        checks.append("✅ Toutes rotations ont même signe de divergence")
    else:
        checks.append("❌ Rotations ont signes de divergence incohérents")

    for check in checks:
        print(check)

    # 8. Verdict final
    print()
    print("=" * 70)
    print("🎯 VERDICT FINAL")
    print("=" * 70)

    all_passed = all(r['passed'] for r in results.values())
    coherent = all_same_sign

    if all_passed and coherent:
        print("✅ TOUS LES TESTS PASSENT - Logique HV rotation CORRECTE")
        print("   Vous pouvez lancer la génération complète des données.")
        return 0
    else:
        print("❌ ÉCHECS DÉTECTÉS - NE PAS lancer la génération complète")
        print()
        if not all_passed:
            print("   Problèmes:")
            for rot, res in results.items():
                if not res['passed']:
                    print(f"     - {rot}: {', '.join(res['errors'])}")
        if not coherent:
            print("   Incohérence dans les signes de divergence")
        print()
        print("   Vérifier:")
        print("   1. Ordre des canaux HV (H à index 0, V à index 1)")
        print("   2. Formules de rotation (Graham et al. 2019)")
        print("   3. Direction des vecteurs HV (centripètes vs centrifuges)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
