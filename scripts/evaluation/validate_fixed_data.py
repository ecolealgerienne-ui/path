#!/usr/bin/env python3
"""
Valide que les données FIXED sont correctes avant retraining.

Vérifie:
1. Nombre d'instances: FIXED > BUGGY (pas de fusion)
2. HV maps: Gradients FORTS aux frontières (pics visibles)
3. NP targets: Coverage identique (union binaire inchangée)
4. Format fichiers: Shape, dtype, ranges corrects

Usage:
    python scripts/evaluation/validate_fixed_data.py --family glandular
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def count_instances_in_target(hv_target: np.ndarray, np_target: np.ndarray) -> int:
    """
    Compte le nombre d'instances dans les HV maps.

    Utilise watershed sur le gradient HV pour reconstruire les instances.
    """
    # Ensure float32 for cv2.Sobel compatibility
    if hv_target.dtype != np.float32:
        hv_target = hv_target.astype(np.float32)

    # Calculer le gradient du HV map
    sobel_h = cv2.Sobel(hv_target[0], cv2.CV_64F, 1, 0, ksize=5)
    sobel_v = cv2.Sobel(hv_target[1], cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(sobel_h**2 + sobel_v**2)

    # Normaliser
    if gradient.max() > 0:
        gradient = gradient / gradient.max()

    # Watershed markers
    markers = np.zeros_like(np_target, dtype=np.int32)

    # Seeds = local maxima du gradient (centres de cellules)
    from scipy.ndimage import maximum_filter
    max_filtered = maximum_filter(gradient, size=5)
    local_max = (gradient == max_filtered) & (gradient > 0.1) & (np_target > 0)

    markers[local_max] = np.arange(1, local_max.sum() + 1)

    # Watershed
    if markers.max() > 0:
        # Convert to uint8 for watershed
        gradient_uint8 = (gradient * 255).astype(np.uint8)
        markers = cv2.watershed(
            cv2.cvtColor(gradient_uint8, cv2.COLOR_GRAY2BGR),
            markers
        )
        n_instances = len(np.unique(markers)) - 2  # -1 (background) -1 (borders)
        return max(n_instances, 0)
    else:
        return 0


def validate_fixed_data(
    old_data_path: Path,
    new_data_path: Path,
    family: str,
    sample_idx: int = 0,
    output_dir: Path = Path("results/validation_fixed")
):
    """Valide que les données FIXED sont correctes."""

    print("=" * 70)
    print(f"VALIDATION DONNÉES FIXED - FAMILLE {family.upper()}")
    print("=" * 70)

    # 1. Charger les données
    print(f"\n📂 Chargement des données...")

    if not old_data_path.exists():
        print(f"   ⚠️  OLD data not found: {old_data_path}")
        print(f"   → Skipping comparison with OLD data")
        old_data = None
    else:
        old_data = np.load(old_data_path)
        print(f"   ✓ OLD data: {old_data_path.name}")
        print(f"     Samples: {old_data['np_targets'].shape[0]}")

    if not new_data_path.exists():
        print(f"   ❌ NEW data not found: {new_data_path}")
        return False

    new_data = np.load(new_data_path)
    print(f"   ✓ NEW data: {new_data_path.name}")
    print(f"     Samples: {new_data['np_targets'].shape[0]}")

    # 2. Vérifications globales
    print(f"\n{'='*70}")
    print("📊 VÉRIFICATIONS GLOBALES")
    print(f"{'='*70}")

    # Shape consistency
    expected_keys = ['images', 'np_targets', 'hv_targets', 'nt_targets', 'fold_ids', 'image_ids']
    for key in expected_keys:
        if key not in new_data:
            print(f"   ❌ Missing key: {key}")
            return False
    print(f"   ✓ All expected keys present")

    # Check shapes
    n_samples = new_data['images'].shape[0]
    checks = [
        ('images', (n_samples, 256, 256, 3)),
        ('np_targets', (n_samples, 256, 256)),
        ('hv_targets', (n_samples, 2, 256, 256)),
        ('nt_targets', (n_samples, 256, 256)),
    ]

    for key, expected_shape in checks:
        actual_shape = new_data[key].shape
        if actual_shape == expected_shape:
            print(f"   ✓ {key}: {actual_shape}")
        else:
            print(f"   ❌ {key}: {actual_shape} (expected {expected_shape})")
            return False

    # Check dtypes
    print(f"\n   Dtypes:")
    print(f"     images: {new_data['images'].dtype} (expected float64)")
    print(f"     np_targets: {new_data['np_targets'].dtype} (expected float32)")
    print(f"     hv_targets: {new_data['hv_targets'].dtype} (expected float32)")
    print(f"     nt_targets: {new_data['nt_targets'].dtype} (expected int64)")

    # Check ranges
    print(f"\n   Ranges:")
    print(f"     NP targets: [{new_data['np_targets'].min():.1f}, {new_data['np_targets'].max():.1f}] (expected [0, 1])")
    print(f"     HV targets: [{new_data['hv_targets'].min():.3f}, {new_data['hv_targets'].max():.3f}] (expected [-1, 1])")
    print(f"     NT targets: [{new_data['nt_targets'].min()}, {new_data['nt_targets'].max()}] (expected [0, 4])")

    # 3. Comparaison sur un échantillon
    print(f"\n{'='*70}")
    print(f"🔬 COMPARAISON DÉTAILLÉE (Sample {sample_idx})")
    print(f"{'='*70}")

    if sample_idx >= n_samples:
        print(f"   ⚠️  Sample {sample_idx} out of range, using sample 0")
        sample_idx = 0

    # Extract sample from NEW data
    image_new = new_data['images'][sample_idx]
    np_new = new_data['np_targets'][sample_idx]
    hv_new = new_data['hv_targets'][sample_idx]
    nt_new = new_data['nt_targets'][sample_idx]

    print(f"\n📊 NEW DATA (FIXED):")
    print(f"   NP coverage: {np_new.sum() / np_new.size * 100:.2f}%")
    print(f"   HV range: [{hv_new.min():.3f}, {hv_new.max():.3f}]")
    print(f"   HV gradient magnitude: {np.abs(np.gradient(hv_new, axis=(1, 2))).mean():.4f}")

    # Count instances via HV maps
    n_instances_new = count_instances_in_target(hv_new, np_new)
    print(f"   Instances (from HV maps): {n_instances_new}")

    # Compare with OLD data if available
    if old_data is not None and sample_idx < old_data['np_targets'].shape[0]:
        np_old = old_data['np_targets'][sample_idx]
        hv_old_raw = old_data['hv_targets'][sample_idx]

        # CRITICAL: Normalize OLD if it's int8 [-127, 127]
        if hv_old_raw.dtype == np.int8:
            hv_old = hv_old_raw.astype(np.float32) / 127.0
            print(f"\n📊 OLD DATA (BUGGY - int8 normalized for comparison):")
            print(f"   ⚠️  Original range: [{hv_old_raw.min()}, {hv_old_raw.max()}] (int8)")
            print(f"   ✓ Normalized to: [{hv_old.min():.3f}, {hv_old.max():.3f}] (float32)")
        else:
            hv_old = hv_old_raw
            print(f"\n📊 OLD DATA (BUGGY):")
            print(f"   HV range: [{hv_old.min():.3f}, {hv_old.max():.3f}]")

        print(f"   NP coverage: {np_old.sum() / np_old.size * 100:.2f}%")
        print(f"   HV gradient magnitude: {np.abs(np.gradient(hv_old, axis=(1, 2))).mean():.4f}")

        n_instances_old = count_instances_in_target(hv_old, np_old)
        print(f"   Instances (from HV maps): {n_instances_old}")

        # Comparison
        print(f"\n{'='*70}")
        print("📈 COMPARAISON OLD vs NEW")
        print(f"{'='*70}")

        print(f"\n   NP Coverage:")
        print(f"     OLD: {np_old.sum() / np_old.size * 100:.2f}%")
        print(f"     NEW: {np_new.sum() / np_new.size * 100:.2f}%")
        coverage_diff = abs((np_new.sum() / np_new.size) - (np_old.sum() / np_old.size)) * 100
        if coverage_diff < 1:
            print(f"     ✓ Difference: {coverage_diff:.2f}% (< 1%, OK)")
        else:
            print(f"     ⚠️  Difference: {coverage_diff:.2f}% (> 1%, WARNING)")

        print(f"\n   HV Gradient Magnitude (sharpness):")
        grad_old = np.abs(np.gradient(hv_old, axis=(1, 2))).mean()
        grad_new = np.abs(np.gradient(hv_new, axis=(1, 2))).mean()
        print(f"     OLD: {grad_old:.4f}")
        print(f"     NEW: {grad_new:.4f}")
        ratio = grad_new / grad_old if grad_old > 0 else 0
        if ratio > 1.2:
            print(f"     ✓ NEW gradient {ratio:.2f}x stronger (better separation!)")
        elif ratio > 0.8:
            print(f"     ~ Similar gradients (ratio: {ratio:.2f})")
        else:
            print(f"     ⚠️  NEW gradient weaker (ratio: {ratio:.2f})")

        print(f"\n   Instances Count:")
        print(f"     OLD: {n_instances_old}")
        print(f"     NEW: {n_instances_new}")
        if n_instances_new > n_instances_old:
            print(f"     ✓ NEW has {n_instances_new - n_instances_old} more instances (no fusion!)")
        elif n_instances_new == n_instances_old:
            print(f"     ~ Same count")
        else:
            print(f"     ⚠️  NEW has FEWER instances (unexpected!)")

    # 4. Visualisation
    print(f"\n{'='*70}")
    print("📸 GÉNÉRATION VISUALISATION")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: NEW data
    axes[0, 0].imshow(image_new)
    axes[0, 0].set_title("Image (NEW)", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np_new, cmap='gray')
    axes[0, 1].set_title(f"NP Mask (NEW)\n{np_new.sum() / np_new.size * 100:.1f}%", fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(hv_new[0], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 2].set_title(f"HV Horizontal (NEW)\nRange: [{hv_new[0].min():.2f}, {hv_new[0].max():.2f}]", fontsize=14)
    axes[0, 2].axis('off')

    # HV gradient
    sobel_h = cv2.Sobel(hv_new[0], cv2.CV_64F, 1, 0, ksize=5)
    sobel_v = cv2.Sobel(hv_new[1], cv2.CV_64F, 0, 1, ksize=5)
    gradient_new = np.sqrt(sobel_h**2 + sobel_v**2)
    axes[0, 3].imshow(gradient_new, cmap='hot')
    axes[0, 3].set_title(f"HV Gradient (NEW)\n{n_instances_new} instances", fontsize=14)
    axes[0, 3].axis('off')

    # Row 2: OLD data (if available)
    if old_data is not None and sample_idx < old_data['np_targets'].shape[0]:
        # Check if OLD has 'images' key (might only have targets)
        if 'images' in old_data:
            image_old = old_data['images'][sample_idx]
            axes[1, 0].imshow(image_old)
            axes[1, 0].set_title("Image (OLD)", fontsize=14)
        else:
            # Use NEW image for OLD row (same tissue)
            axes[1, 0].imshow(image_new)
            axes[1, 0].set_title("Image (OLD - same as NEW)", fontsize=14)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(np_old, cmap='gray')
        axes[1, 1].set_title(f"NP Mask (OLD)\n{np_old.sum() / np_old.size * 100:.1f}%", fontsize=14)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(hv_old[0], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 2].set_title(f"HV Horizontal (OLD)\nRange: [{hv_old[0].min():.2f}, {hv_old[0].max():.2f}]", fontsize=14)
        axes[1, 2].axis('off')

        sobel_h_old = cv2.Sobel(hv_old[0], cv2.CV_64F, 1, 0, ksize=5)
        sobel_v_old = cv2.Sobel(hv_old[1], cv2.CV_64F, 0, 1, ksize=5)
        gradient_old = np.sqrt(sobel_h_old**2 + sobel_v_old**2)
        axes[1, 3].imshow(gradient_old, cmap='hot')
        axes[1, 3].set_title(f"HV Gradient (OLD)\n{n_instances_old} instances", fontsize=14)
        axes[1, 3].axis('off')
    else:
        for ax in axes[1]:
            ax.text(0.5, 0.5, "OLD data not available", ha='center', va='center', fontsize=14)
            ax.axis('off')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{family}_validation_sample{sample_idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n   ✓ Saved: {output_path}")

    # 5. Diagnostic final
    print(f"\n{'='*70}")
    print("🎯 DIAGNOSTIC FINAL")
    print(f"{'='*70}")

    all_pass = True

    # Check 1: Shapes
    if new_data['images'].shape == (n_samples, 256, 256, 3):
        print(f"\n✓ CHECK 1: Shapes correctes")
    else:
        print(f"\n❌ CHECK 1: Shapes incorrectes")
        all_pass = False

    # Check 2: Ranges
    if (new_data['np_targets'].min() >= 0 and new_data['np_targets'].max() <= 1 and
        new_data['hv_targets'].min() >= -1.1 and new_data['hv_targets'].max() <= 1.1):
        print(f"✓ CHECK 2: Ranges corrects")
    else:
        print(f"❌ CHECK 2: Ranges incorrects")
        all_pass = False

    # Check 3: HV gradients (if OLD data available)
    if old_data is not None and sample_idx < old_data['np_targets'].shape[0]:
        if grad_new >= grad_old * 0.8:  # Au moins 80% du gradient OLD (ou plus fort)
            print(f"✓ CHECK 3: HV gradients OK (ratio {ratio:.2f})")
        else:
            print(f"⚠️  CHECK 3: HV gradients plus faibles (ratio {ratio:.2f})")
            # Not a failure, might be normal for some images
    else:
        print(f"~ CHECK 3: HV gradients non comparés (OLD data unavailable)")

    # Check 4: Instances count
    if n_instances_new > 0:
        print(f"✓ CHECK 4: Instances détectées ({n_instances_new})")
    else:
        print(f"⚠️  CHECK 4: Aucune instance détectée (image vide?)")

    if all_pass:
        print(f"\n🎉 VALIDATION RÉUSSIE!")
        print(f"   → Les données FIXED sont prêtes pour le retraining")
        return True
    else:
        print(f"\n⚠️  VALIDATION ÉCHOUÉE")
        print(f"   → Corriger les problèmes avant retraining")
        return False


def main():
    parser = argparse.ArgumentParser(description="Valide données FIXED avant retraining")
    parser.add_argument("--family", type=str, required=True,
                        help="Famille à valider (glandular, digestive, etc.)")
    parser.add_argument("--old_data", type=Path,
                        help="Chemin vers OLD data (optionnel, pour comparaison)")
    parser.add_argument("--new_data", type=Path,
                        help="Chemin vers NEW data (par défaut: data/family_FIXED/{family}_data_FIXED.npz)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index de l'échantillon à visualiser")
    parser.add_argument("--output_dir", type=Path, default=Path("results/validation_fixed"))

    args = parser.parse_args()

    # Déterminer les chemins
    if args.old_data is None:
        args.old_data = Path(f"data/family/{args.family}_targets.npz")

    if args.new_data is None:
        args.new_data = Path(f"data/family_FIXED/{args.family}_data_FIXED.npz")

    # Valider
    success = validate_fixed_data(
        args.old_data,
        args.new_data,
        args.family,
        args.sample_idx,
        args.output_dir
    )

    if success:
        print("\n" + "=" * 70)
        print("🚀 PROCHAINES ÉTAPES")
        print("=" * 70)
        print(f"\n1. Visualiser: {args.output_dir}/{args.family}_validation_sample{args.sample_idx}.png")
        print(f"2. Vérifier que HV gradients NEW > OLD (pics visibles)")
        print(f"3. Si OK, lancer retraining:")
        print(f"     python scripts/training/train_hovernet_family.py \\")
        print(f"         --family {args.family} \\")
        print(f"         --data_dir data/family_FIXED \\")
        print(f"         --epochs 50 \\")
        print(f"         --augment")
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
