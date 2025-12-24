#!/usr/bin/env python3
"""
Diagnostic: Vérifie la distribution des classes NT dans les données v9.
"""

import numpy as np

# Load v9 data
data = np.load('data/family_FIXED/epidermal_data_FIXED_v9_NUCLEI_ONLY.npz')

nt_targets = data['nt_targets']
np_targets = data['np_targets']

print("=" * 80)
print("🔍 DIAGNOSTIC DISTRIBUTION NT")
print("=" * 80)

# Distribution NT
unique, counts = np.unique(nt_targets, return_counts=True)
total_pixels = nt_targets.size

print(f"\n📊 Distribution NT:")
for cls, count in zip(unique, counts):
    pct = count / total_pixels * 100
    class_names = {0: "Background", 1: "Neoplastic", 2: "Inflammatory",
                   3: "Connective", 4: "Dead", 5: "Epithelial"}
    name = class_names.get(cls, f"Unknown({cls})")
    print(f"  Classe {cls} ({name}): {count:>10} pixels ({pct:>5.2f}%)")

# NP coverage
np_coverage = np_targets.mean() * 100

print(f"\n📊 NP Coverage: {np_coverage:.2f}%")

# Comparison
nt_nuclei_pixels = counts[unique > 0].sum() if len(unique) > 1 else 0
nt_nuclei_pct = nt_nuclei_pixels / total_pixels * 100

print(f"\n⚠️  COMPARAISON:")
print(f"  NP dit: {np_coverage:.2f}% de noyaux")
print(f"  NT dit: {nt_nuclei_pct:.2f}% de noyaux (classes 1-4)")

if abs(np_coverage - nt_nuclei_pct) > 10:
    print(f"\n❌ MISMATCH CRITIQUE: Différence de {abs(np_coverage - nt_nuclei_pct):.1f}%!")
    print(f"   NP détecte des noyaux que NT classe comme background!")
else:
    print(f"\n✅ Cohérence NP/NT OK")

print("\n" + "=" * 80)
