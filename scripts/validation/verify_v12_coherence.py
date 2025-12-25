#!/usr/bin/env python3
"""
Vérifie la cohérence NP/NT dans les données v12.

Usage:
    python scripts/validation/verify_v12_coherence.py [--data_file PATH]

Résultat attendu:
    ✅ Conflit NP/NT: 0.00% (cohérence parfaite)
"""

import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Vérifie cohérence v12")
parser.add_argument(
    '--data_file',
    type=Path,
    default=Path('data/family_FIXED/epidermal_data_FIXED_v12_COHERENT.npz'),
    help='Fichier NPZ à analyser (défaut: v12)'
)
args = parser.parse_args()

print("=" * 80)
print("🔍 VÉRIFICATION COHÉRENCE v12")
print("=" * 80)

if not args.data_file.exists():
    print(f"\n❌ Fichier non trouvé: {args.data_file}")
    print(f"\n📋 Créez d'abord les données v12:")
    print(f"   python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py --family epidermal")
    exit(1)

# Charger
print(f"\n📂 Chargement: {args.data_file}")
data = np.load(args.data_file)

np_targets = data['np_targets']
nt_targets = data['nt_targets']

print(f"\n📦 Données chargées:")
print(f"  NP shape: {np_targets.shape}")
print(f"  NT shape: {nt_targets.shape}")

# Vérification conflit
np_positive = np_targets > 0
nt_background = nt_targets == 0
conflict_mask = np_positive & nt_background

n_np_positive = np_positive.sum()
n_conflict = conflict_mask.sum()
conflict_pct = (n_conflict / n_np_positive * 100) if n_np_positive > 0 else 0

# Statistiques
np_pct = np_targets.mean() * 100
nt_pct = (nt_targets == 1).sum() / nt_targets.size * 100

print("\n" + "=" * 80)
print("📊 RÉSULTATS:")
print("=" * 80)

print(f"\n  Pixels NP=1 (noyaux):     {n_np_positive:>12}")
print(f"  Pixels NT=1 (noyaux):     {(nt_targets == 1).sum():>12}")
print(f"  Pixels NP=1 & NT=0:       {n_conflict:>12}")
print(f"\n  NP coverage:              {np_pct:.4f}%")
print(f"  NT nuclei:                {nt_pct:.4f}%")
print(f"  Différence:               {abs(np_pct - nt_pct):.6f}%")

print("\n" + "=" * 80)
print("🎯 CONFLIT NP/NT:")
print("=" * 80)

if conflict_pct < 0.01:
    print(f"\n  ✅ CONFLIT: {conflict_pct:.4f}% — COHÉRENCE PARFAITE!")
    print(f"\n  Le script v12 a correctement aligné NP et NT.")
    print(f"  Vous pouvez procéder à l'extraction des features et au training.")
else:
    print(f"\n  ❌ CONFLIT: {conflict_pct:.2f}% — PROBLÈME DÉTECTÉ!")
    print(f"\n  Le conflit devrait être 0.00%. Vérifiez le script v12.")

print("\n" + "=" * 80)
print("📋 PROCHAINES ÉTAPES:")
print("=" * 80)

if conflict_pct < 0.01:
    print("""
1. Extraire les features H-optimus-0:
   python scripts/preprocessing/extract_features_from_v9.py \\
       --input_file data/family_FIXED/epidermal_data_FIXED_v12_COHERENT.npz \\
       --output_dir data/cache/family_data \\
       --family epidermal

2. Ré-entraîner HoVer-Net:
   python scripts/training/train_hovernet_family.py \\
       --family epidermal --epochs 50 --augment

3. Tester AJI final:
   python scripts/evaluation/test_epidermal_aji_FINAL.py \\
       --checkpoint models/checkpoints/hovernet_epidermal_best.pth \\
       --n_samples 50
""")
else:
    print("\n  Corrigez d'abord le conflit avant de continuer.")

print("=" * 80)
