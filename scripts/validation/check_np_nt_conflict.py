#!/usr/bin/env python3
"""
Diagnostic CRITIQUE: Détecte le conflit NP vs NT (Expert 2025-12-24).

Vérifie si NT contient des 0 (background) là où NP contient des 1 (noyau).
Ce conflit empêche le modèle de converger (Dice bloqué à 0.40).

Usage:
    python scripts/validation/check_np_nt_conflict.py [--data_file PATH]
"""

import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Diagnostic conflit NP vs NT")
parser.add_argument(
    '--data_file',
    type=Path,
    default=Path('data/family_FIXED/epidermal_data_FIXED_v11_FORCE_NT1.npz'),
    help='Fichier NPZ à analyser (défaut: v11)'
)
args = parser.parse_args()

print("=" * 80)
print("🚨 DIAGNOSTIC CONFLIT NP vs NT (Expert 2025-12-24)")
print("=" * 80)

# Load data
print(f"\n📂 Chargement: {args.data_file}")
data = np.load(args.data_file)

np_targets = data['np_targets']
nt_targets = data['nt_targets']

print(f"\n📦 Données chargées:")
print(f"  NP targets: {np_targets.shape}")
print(f"  NT targets: {nt_targets.shape}")

# VÉRIFICATION CRITIQUE: Pixels où NP=1 mais NT=0
np_positive = np_targets > 0  # Pixels détectés comme noyaux par NP
nt_background = nt_targets == 0  # Pixels classés comme background par NT

conflict_mask = np_positive & nt_background  # CONFLIT!

n_np_positive = np_positive.sum()
n_conflict = conflict_mask.sum()
conflict_pct = (n_conflict / n_np_positive * 100) if n_np_positive > 0 else 0

print("\n" + "=" * 80)
print("🎯 RÉSULTAT CRITIQUE:")
print("=" * 80)

print(f"\nPixels NP=1 (noyaux détectés):     {n_np_positive:>12}")
print(f"Pixels NP=1 MAIS NT=0 (CONFLIT):   {n_conflict:>12} ({conflict_pct:.2f}%)")

if conflict_pct > 5:
    print(f"\n❌ CONFLIT CRITIQUE DÉTECTÉ!")
    print(f"   {conflict_pct:.1f}% des noyaux (NP=1) sont classés comme background (NT=0)")
    print(f"\n📋 EXPLICATION (Expert):")
    print(f"   Le modèle reçoit des ordres contradictoires:")
    print(f"     - NP branche: 'Prédit 1 ici (c'est un noyau)'")
    print(f"     - NT branche: 'Prédit 0 ici (c'est du background)'")
    print(f"   → Le modèle NE PEUT PAS GAGNER → Dice bloqué à 0.40")
    print(f"\n🛠️  SOLUTION:")
    print(f"   Forcer NT=1 pour TOUS les pixels où NP=1")
    print(f"   (Simplifier: 'noyau' vs 'pas noyau', pas de classification fine)")

elif conflict_pct > 1:
    print(f"\n⚠️  Conflit mineur détecté ({conflict_pct:.2f}%)")
    print(f"   Peut causer instabilité training mais pas bloquant")

else:
    print(f"\n✅ PAS DE CONFLIT MAJEUR")
    print(f"   Seulement {conflict_pct:.2f}% de pixels en conflit")

# Distribution NT pour pixels NP=1
print("\n" + "=" * 80)
print("📊 DISTRIBUTION NT POUR PIXELS NP=1:")
print("=" * 80)

nt_for_nuclei = nt_targets[np_positive]
unique, counts = np.unique(nt_for_nuclei, return_counts=True)

class_names = {0: "Background", 1: "Neoplastic", 2: "Inflammatory",
               3: "Connective", 4: "Dead/Epithelial"}

print(f"\nPour les {n_np_positive} pixels NP=1:")
for cls, cnt in zip(unique, counts):
    pct = cnt / n_np_positive * 100
    name = class_names.get(int(cls), f"Unknown({cls})")
    marker = "❌ CONFLIT!" if cls == 0 else "✅"
    print(f"  Classe {cls} ({name}): {cnt:>10} ({pct:>5.2f}%) {marker}")

# Recommandation
print("\n" + "=" * 80)
print("🎯 RECOMMANDATION:")
print("=" * 80)

if conflict_pct > 5:
    print(f"\n✅ APPLIQUER SOLUTION EXPERT:")
    print(f"   Modifier compute_nt_target() pour forcer NT=1 partout où NP=1")
    print(f"   → Élimine conflit → Dice 0.40 → 0.80+ en 10 epochs")
    print(f"\n   Script prêt: prepare_family_data_FIXED_v11_FORCE_NT1.py")

else:
    print(f"\n⚠️  Conflit faible mais training catastrophique (Dice 0.42)")
    print(f"   Cause probable: AUTRE problème (alignement image/mask?)")
    print(f"   Lancer: python scripts/validation/check_alignment_v10.py")

print("\n" + "=" * 80)
