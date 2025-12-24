#!/usr/bin/env python3
"""
Vérification CRITIQUE des HV targets dans les fichiers .npz.

Suite à l'analyse expert: magnitude HV 0.022 (50× trop faible) indique soit:
1. Mismatch normalisation (targets pas dans [-1, 1])
2. Absence Tanh (déjà vérifié - présent)

Ce script vérifie la cause #1.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DEFAULT_FAMILY_DATA_DIR


def verify_hv_targets(family: str = "epidermal"):
    """
    Vérifie que les HV targets sont bien normalisés [-1, 1].

    Selon littérature HoVer-Net (Graham et al., 2019):
    - HV maps DOIVENT être dans [-1, 1]
    - Sinon le modèle apprend à prédire dans une échelle compressée
    """

    print("\n" + "="*80)
    print(f"VÉRIFICATION CRITIQUE: HV TARGETS - {family.upper()}")
    print("="*80)
    print("\nRéférence: HoVer-Net (Graham et al., 2019)")
    print("Attendu: HV targets normalisés dans [-1.0, 1.0]")
    print("\n" + "─"*80)

    # Charger données
    data_dir = Path(DEFAULT_FAMILY_DATA_DIR)
    targets_path = data_dir / f"{family}_targets.npz"

    if not targets_path.exists():
        print(f"\n❌ ERREUR: Fichier introuvable: {targets_path}")
        return 1

    print(f"\n📁 Fichier: {targets_path}")
    print(f"   Taille: {targets_path.stat().st_size / 1024**2:.1f} MB")

    # Charger avec mmap pour économiser RAM
    data = np.load(targets_path, mmap_mode='r')

    print(f"\n📊 Contenu .npz:")
    for key in data.keys():
        arr = data[key]
        print(f"   • {key:20s}: shape={arr.shape}, dtype={arr.dtype}")

    # Extraire HV targets
    if 'hv_targets' not in data:
        print(f"\n❌ ERREUR: Clé 'hv_targets' introuvable!")
        print(f"   Clés disponibles: {list(data.keys())}")
        return 1

    hv_targets = data['hv_targets']

    # Statistiques COMPLÈTES
    print(f"\n" + "="*80)
    print("STATISTIQUES HV TARGETS")
    print("="*80)

    print(f"\n1️⃣ FORMAT")
    print(f"   Shape:  {hv_targets.shape}")
    print(f"   Dtype:  {hv_targets.dtype}")
    print(f"   Memory: {hv_targets.nbytes / 1024**2:.1f} MB")

    print(f"\n2️⃣ RANGE (Vérifie normalisation)")
    hv_min = float(hv_targets.min())
    hv_max = float(hv_targets.max())
    hv_mean = float(hv_targets.mean())
    hv_std = float(hv_targets.std())

    print(f"   Min:    {hv_min:+.6f}")
    print(f"   Max:    {hv_max:+.6f}")
    print(f"   Mean:   {hv_mean:+.6f}")
    print(f"   Std:    {hv_std:+.6f}")

    # Vérification ranges par canal
    print(f"\n3️⃣ PAR CANAL (H=Horizontal, V=Vertical)")
    for c, name in enumerate(['Vertical (Y)', 'Horizontal (X)']):
        channel = hv_targets[:, c, :, :]
        print(f"   Canal {c} ({name}):")
        print(f"      Range: [{channel.min():+.6f}, {channel.max():+.6f}]")
        print(f"      Mean:  {channel.mean():+.6f}")
        print(f"      Std:   {channel.std():+.6f}")

    # Distribution par bins
    print(f"\n4️⃣ DISTRIBUTION (Vérifie symétrie)")
    bins = [
        (-np.inf, -0.5, "Forte négative (<-0.5)"),
        (-0.5, -0.1, "Négative modérée"),
        (-0.1, 0.1, "Proche de zéro"),
        (0.1, 0.5, "Positive modérée"),
        (0.5, np.inf, "Forte positive (>0.5)"),
    ]

    total_pixels = hv_targets.size
    for low, high, label in bins:
        count = np.sum((hv_targets >= low) & (hv_targets < high))
        pct = count / total_pixels * 100
        print(f"   {label:30s}: {pct:6.2f}%")

    # DIAGNOSTIC
    print(f"\n" + "="*80)
    print("DIAGNOSTIC")
    print("="*80)

    # Check 1: Dtype
    if hv_targets.dtype != np.float32:
        print(f"\n❌ ERREUR DTYPE:")
        print(f"   Dtype actuel: {hv_targets.dtype}")
        print(f"   Dtype attendu: float32")
        print(f"   → Les targets ne sont PAS en float! Conversion requise.")
        return 1
    else:
        print(f"\n✅ Dtype: float32 (correct)")

    # Check 2: Range
    if hv_min < -1.1 or hv_max > 1.1:
        print(f"\n❌ ERREUR RANGE:")
        print(f"   Range actuel: [{hv_min:.3f}, {hv_max:.3f}]")
        print(f"   Range attendu: [-1.0, 1.0]")
        print(f"   → Les targets sont MAL NORMALISÉS!")

        # Diagnostic du facteur
        if abs(hv_min) > 100 or abs(hv_max) > 100:
            print(f"\n   💡 HYPOTHÈSE: Targets en PIXELS bruts (non normalisés)")
            print(f"      Solution: Diviser par rayon maximal")
        elif abs(hv_min) > 10 or abs(hv_max) > 10:
            print(f"\n   💡 HYPOTHÈSE: Targets mal scalés (facteur ~10-100)")
            print(f"      Solution: Vérifier compute_hv_maps()")

        return 1
    elif hv_min < -1.0 or hv_max > 1.0:
        print(f"\n⚠️ WARNING: Légère sur-normalisation")
        print(f"   Range: [{hv_min:.6f}, {hv_max:.6f}]")
        print(f"   Dépassement: {max(abs(hv_min + 1.0), abs(hv_max - 1.0)):.6f}")
        print(f"   → Acceptable (tolérance float), mais vérifier Gaussian smoothing")
    else:
        print(f"\n✅ Range: [{hv_min:.3f}, {hv_max:.3f}] (correct)")

    # Check 3: Symétrie
    if abs(hv_mean) > 0.05:
        print(f"\n⚠️ WARNING: Asymétrie détectée")
        print(f"   Mean: {hv_mean:.6f} (attendu: ~0.0)")
        print(f"   → Les gradients HV ne sont pas centrés!")
        print(f"   → Vérifier que compute_hv_maps() centre bien sur centroïde")
    else:
        print(f"\n✅ Symétrie: Mean={hv_mean:.6f} (centré)")

    # Check 4: Variance
    if hv_std < 0.3:
        print(f"\n⚠️ WARNING: Variance trop faible!")
        print(f"   Std: {hv_std:.6f} (attendu: >0.4)")
        print(f"   → Les gradients HV sont TROP COMPRESSÉS!")
        print(f"   → Cause possible: Gaussian smoothing trop agressif (sigma trop grand)")
        print(f"   → Ou: Normalization radiale trop conservative")
    elif hv_std > 0.7:
        print(f"\n⚠️ WARNING: Variance trop élevée!")
        print(f"   Std: {hv_std:.6f} (attendu: <0.6)")
        print(f"   → Pas de smoothing? Ou normalization incorrecte?")
    else:
        print(f"\n✅ Variance: Std={hv_std:.6f} (bonne dynamique)")

    # Échantillonnage visuel
    print(f"\n5️⃣ ÉCHANTILLONS (Premiers 3)")
    for i in range(min(3, hv_targets.shape[0])):
        sample = hv_targets[i]
        print(f"   Sample {i}:")
        print(f"      Range: [{sample.min():+.4f}, {sample.max():+.4f}]")
        print(f"      Non-zero: {np.count_nonzero(sample)} / {sample.size} pixels")

    # VERDICT FINAL
    print(f"\n" + "="*80)
    print("VERDICT FINAL")
    print("="*80)

    issues = []

    if hv_targets.dtype != np.float32:
        issues.append("Dtype incorrect (pas float32)")

    if hv_min < -1.1 or hv_max > 1.1:
        issues.append(f"Range hors [-1, 1]: [{hv_min:.3f}, {hv_max:.3f}]")

    if abs(hv_mean) > 0.05:
        issues.append(f"Non centré (mean={hv_mean:.4f})")

    if hv_std < 0.3:
        issues.append(f"Variance trop faible (std={hv_std:.4f})")

    if issues:
        print(f"\n❌ PROBLÈMES DÉTECTÉS ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print(f"\n🔧 ACTIONS REQUISES:")
        print(f"   1. Vérifier script de génération: prepare_family_data_FIXED_v8.py")
        print(f"   2. Vérifier fonction: compute_hv_maps()")
        print(f"   3. Régénérer données si nécessaire")

        return 1
    else:
        print(f"\n✅ TARGETS HV CORRECTS!")
        print(f"   • Dtype: float32")
        print(f"   • Range: [{hv_min:.3f}, {hv_max:.3f}]")
        print(f"   • Centré: mean={hv_mean:.4f}")
        print(f"   • Dynamique: std={hv_std:.4f}")

        print(f"\n💡 CONCLUSION:")
        print(f"   Les targets HV sont bien normalisés.")
        print(f"   Le problème de magnitude faible (0.022) vient donc:")
        print(f"   → Soit du MODÈLE (poids mal entraînés)")
        print(f"   → Soit des FEATURES (mismatch normalisation H-optimus-0)")

        print(f"\n🔍 PROCHAINE ÉTAPE:")
        print(f"   Vérifier les features H-optimus-0 utilisées pour le training")
        print(f"   (CLS std doit être dans [0.70, 0.90])")

        return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vérifier HV targets dans .npz")
    parser.add_argument('--family', type=str, default='epidermal',
                       choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
                       help='Famille à vérifier')

    args = parser.parse_args()

    sys.exit(verify_hv_targets(args.family))
