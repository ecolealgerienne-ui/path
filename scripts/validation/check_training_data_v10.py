#!/usr/bin/env python3
"""
Diagnostic: Vérifie les données d'entraînement réellement utilisées.

Charge features.npz et targets.npz pour vérifier la cohérence.
"""

import numpy as np
from pathlib import Path

print("=" * 80)
print("🔍 DIAGNOSTIC DONNÉES ENTRAÎNEMENT v10")
print("=" * 80)

# Paths
features_file = Path("data/cache/family_data/epidermal_features.npz")
targets_file = Path("data/cache/family_data/epidermal_targets.npz")

if not features_file.exists():
    print(f"\n❌ Features non trouvées: {features_file}")
    print("   Lancer d'abord: python scripts/preprocessing/extract_features_from_v10.py --family epidermal")
    exit(1)

if not targets_file.exists():
    print(f"\n❌ Targets non trouvés: {targets_file}")
    print("   Problème avec extract_features_from_v10.py")
    exit(1)

# Load
print(f"\n📦 Chargement données entraînement...")
features_data = np.load(features_file)
targets_data = np.load(targets_file)

features = features_data['features']
np_targets = targets_data['np_targets']
hv_targets = targets_data['hv_targets']
nt_targets = targets_data['nt_targets']

print(f"  Features: {features.shape}")
print(f"  NP targets: {np_targets.shape}")
print(f"  HV targets: {hv_targets.shape}")
print(f"  NT targets: {nt_targets.shape}")

# Vérification tailles
n_samples_feat = len(features)
n_samples_targ = len(np_targets)

print(f"\n✅ Nombre d'échantillons:")
print(f"  Features: {n_samples_feat}")
print(f"  Targets:  {n_samples_targ}")

if n_samples_feat != n_samples_targ:
    print(f"\n❌ MISMATCH: Features ({n_samples_feat}) != Targets ({n_samples_targ})")
    exit(1)
else:
    print(f"  ✅ Nombre cohérent: {n_samples_feat} échantillons")

# Distribution NT
print("\n" + "=" * 80)
print("📊 DISTRIBUTION NT (TARGETS ENTRAÎNEMENT):")
print("=" * 80)

nt_unique, nt_counts = np.unique(nt_targets, return_counts=True)
total = nt_targets.size

for cls, cnt in zip(nt_unique, nt_counts):
    pct = cnt / total * 100
    class_names = {0: "Background", 1: "Neoplastic", 2: "Inflammatory",
                   3: "Connective", 4: "Dead/Epithelial", 5: "ERROR"}
    name = class_names.get(int(cls), f"Unknown({cls})")
    print(f"  Classe {cls} ({name}): {cnt:>12} pixels ({pct:>6.2f}%)")

# NP coverage
np_coverage = np_targets.mean() * 100
print(f"\n📊 NP COVERAGE: {np_coverage:.2f}%")

# HV targets
hv_min, hv_max = hv_targets.min(), hv_targets.max()
hv_mean, hv_std = hv_targets.mean(), hv_targets.std()
print(f"\n📊 HV TARGETS:")
print(f"  Min:  {hv_min:.4f}")
print(f"  Max:  {hv_max:.4f}")
print(f"  Mean: {hv_mean:.4f}")
print(f"  Std:  {hv_std:.4f}")

# Vérification cohérence NP/NT
nt_nuclei = nt_counts[nt_unique > 0].sum() if len(nt_unique) > 1 else 0
nt_nuclei_pct = nt_nuclei / total * 100

print(f"\n⚠️  VÉRIFICATION COHÉRENCE NP/NT:")
print(f"  NP coverage:     {np_coverage:.2f}%")
print(f"  NT nuclei (1-4): {nt_nuclei_pct:.2f}%")
print(f"  Différence:      {abs(np_coverage - nt_nuclei_pct):.2f}%")

if abs(np_coverage - nt_nuclei_pct) > 2:
    print(f"\n❌ MISMATCH CRITIQUE NP/NT!")
    print(f"   → Training Dice sera catastrophique (<0.50)")
    print(f"   → Le modèle apprend des targets incohérents")
else:
    print(f"\n✅ Cohérence NP/NT OK")

# Diagnostic training catastrophique
print("\n" + "=" * 80)
print("🎯 DIAGNOSTIC TRAINING CATASTROPHIQUE (Dice 0.42):")
print("=" * 80)

issues = []

# Check 1: NT max
if nt_targets.max() > 4:
    issues.append("❌ NT contient classe 5 (pas remappée!)")

# Check 2: NT majoritairement background
bg_pct = nt_counts[nt_unique == 0][0] / total * 100 if 0 in nt_unique else 0
if bg_pct > 90:
    issues.append(f"❌ NT = {bg_pct:.1f}% background (noyaux pas typés!)")

# Check 3: NP coverage trop faible
if np_coverage < 10:
    issues.append(f"❌ NP coverage = {np_coverage:.1f}% (trop faible, attendu ~15%)")

# Check 4: HV range anormal
if hv_max > 1.5 or hv_min < -1.5:
    issues.append(f"❌ HV range [{hv_min:.2f}, {hv_max:.2f}] (attendu [-1, 1])")

# Check 5: Mismatch NP/NT
if abs(np_coverage - nt_nuclei_pct) > 2:
    issues.append(f"❌ Mismatch NP ({np_coverage:.1f}%) vs NT ({nt_nuclei_pct:.1f}%)")

if issues:
    print("\n🚨 PROBLÈMES DÉTECTÉS:")
    for issue in issues:
        print(f"  {issue}")

    print("\n📋 ACTIONS RECOMMANDÉES:")
    if "classe 5" in str(issues):
        print("  1. Vérifier compute_nt_target() remappe bien classe 5 → 4")
    if "background" in str(issues):
        print("  2. Vérifier compute_nt_target() utilise bien Channel 0 comme masque")
    if "NP coverage" in str(issues):
        print("  3. Vérifier compute_np_target() utilise bien Channel 0")
    if "Mismatch" in str(issues):
        print("  4. Vérifier que NP et NT utilisent la même source (Channel 0)")
else:
    print("\n✅ Aucun problème évident détecté dans les targets")
    print("\n⚠️  Mais Training Dice = 0.42 suggère un autre problème:")
    print("    - Désalignement image/mask dans le fichier source?")
    print("    - Corruption des features H-optimus-0?")
    print("\n📋 ACTIONS:")
    print("    1. Lancer: python scripts/validation/check_alignment_v10.py")
    print("    2. Vérifier visuellement results/alignment_check_v10.png")

print("\n" + "=" * 80)
