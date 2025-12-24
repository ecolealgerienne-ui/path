#!/usr/bin/env python3
"""
Diagnostic: Inspecter les targets pré-calculés pour identifier le problème
"""

import numpy as np
from pathlib import Path
import sys

def diagnose_targets(family: str, data_dir: str = "data/family_FIXED"):  # ⚠️ FIX: Source de vérité unique
    """Inspecte les targets pré-calculés."""

    data_dir = Path(data_dir)

    # Support both naming conventions (FIXED and old)
    targets_path = data_dir / f"{family}_data_FIXED.npz"
    if not targets_path.exists():
        targets_path = data_dir / f"{family}_targets.npz"  # Fallback to old naming

    if not targets_path.exists():
        print(f"❌ ERREUR: {targets_path} introuvable")
        print(f"\nCherché dans:")
        print(f"  - {data_dir / f'{family}_data_FIXED.npz'}")
        print(f"  - {data_dir / f'{family}_targets.npz'}")
        return

    print("=" * 80)
    print(f"DIAGNOSTIC TARGETS: {family}")
    print("=" * 80)
    print("")

    print(f"Chargement: {targets_path}")
    data = np.load(targets_path)

    np_targets = data['np_targets']
    hv_targets = data['hv_targets']
    nt_targets = data['nt_targets']

    print(f"Fichier: {targets_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("")

    # NP targets
    print("=" * 80)
    print("NP TARGETS (Nuclear Presence)")
    print("=" * 80)
    print(f"Shape:  {np_targets.shape}")
    print(f"Dtype:  {np_targets.dtype}")
    print(f"Min:    {np_targets.min():.3f}")
    print(f"Max:    {np_targets.max():.3f}")
    print(f"Mean:   {np_targets.mean():.3f}")
    print(f"Unique: {np.unique(np_targets)[:10]}")  # Premiers 10
    print("")

    if np_targets.dtype not in [np.float32, np.float64]:
        print("⚠️  WARNING: NP targets ne sont PAS en float !")
        print(f"   Attendu: float32 [0, 1]")
        print(f"   Trouvé:  {np_targets.dtype}")
        print("")

    # HV targets
    print("=" * 80)
    print("HV TARGETS (Horizontal-Vertical Maps)")
    print("=" * 80)
    print(f"Shape:  {hv_targets.shape}")
    print(f"Dtype:  {hv_targets.dtype}")
    print(f"Min:    {hv_targets.min():.3f}")
    print(f"Max:    {hv_targets.max():.3f}")
    print(f"Mean:   {hv_targets.mean():.3f}")
    print(f"Std:    {hv_targets.std():.3f}")
    print("")

    print("HV Targets par canal:")
    print(f"  Canal H: min={hv_targets[:, 0, :, :].min():.3f}, max={hv_targets[:, 0, :, :].max():.3f}")
    print(f"  Canal V: min={hv_targets[:, 1, :, :].min():.3f}, max={hv_targets[:, 1, :, :].max():.3f}")
    print("")

    # Diagnostic HV
    if hv_targets.dtype == np.int8:
        print("❌ ERREUR CRITIQUE: HV targets en int8 [-127, 127] !")
        print("   Attendu: float32 [-1, 1]")
        print("")
        print("   IMPACT:")
        print("   • Modèle prédit en float32 [-1, 1]")
        print("   • Targets en int8 [-127, 127]")
        print(f"   • MSE ≈ (0.5 - 100)² ≈ 10000 ← Explique HV MSE = 4681 !")
        print("")
        print("   SOLUTION:")
        print("   Ré-générer targets avec prepare_family_data_FIXED.py")
        print("")
    elif hv_targets.min() < -10 or hv_targets.max() > 10:
        print("⚠️  WARNING: HV values hors plage [-1, 1] !")
        print(f"   Range trouvé: [{hv_targets.min():.1f}, {hv_targets.max():.1f}]")
        print(f"   Attendu: [-1, 1]")
        print("")

        # Si les valeurs sont dans [-127, 127], c'est probablement int8 converti en float
        if -130 < hv_targets.min() < -50 and 50 < hv_targets.max() < 130:
            print("   → Semble être int8 [-127, 127] converti en float sans normalisation !")
            print("")
    else:
        print("✅ HV targets semblent corrects (range [-1, 1])")
        print("")

    # NT targets
    print("=" * 80)
    print("NT TARGETS (Nuclei Type)")
    print("=" * 80)
    print(f"Shape:  {nt_targets.shape}")
    print(f"Dtype:  {nt_targets.dtype}")
    print(f"Min:    {nt_targets.min()}")
    print(f"Max:    {nt_targets.max()}")
    print(f"Unique: {np.unique(nt_targets)}")
    print("")

    if nt_targets.dtype not in [np.int32, np.int64]:
        print("⚠️  WARNING: NT targets ne sont PAS en int !")
        print(f"   Attendu: int64 [0-4]")
        print(f"   Trouvé:  {nt_targets.dtype}")
        print("")

    if nt_targets.max() > 4:
        print("❌ ERREUR: NT targets ont des valeurs > 4 !")
        print(f"   Attendu: [0-4] (5 classes)")
        print(f"   Trouvé: max = {nt_targets.max()}")
        print("")
    else:
        print("✅ NT targets semblent corrects (range [0-4])")
        print("")

    # Distribution NT
    print("Distribution NT:")
    for i in range(5):
        count = (nt_targets == i).sum()
        pct = count / nt_targets.size * 100
        print(f"  Classe {i}: {count:8d} pixels ({pct:5.1f}%)")
    print("")

    # Verdict global
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print("")

    issues = []

    if np_targets.dtype not in [np.float32, np.float64]:
        issues.append("NP targets dtype incorrect")

    if hv_targets.dtype == np.int8:
        issues.append("HV targets en int8 au lieu de float32")
    elif hv_targets.min() < -10 or hv_targets.max() > 10:
        issues.append("HV targets hors plage [-1, 1]")

    if nt_targets.dtype not in [np.int32, np.int64]:
        issues.append("NT targets dtype incorrect")

    if issues:
        print("❌ PROBLÈMES DÉTECTÉS:")
        for issue in issues:
            print(f"   • {issue}")
        print("")
        print("   → Ré-générer les targets avec le script FIXED")
        print("")
    else:
        print("✅ Tous les targets semblent corrects")
        print("")
        print("   → Le problème vient probablement d'ailleurs:")
        print("     - Chargement du checkpoint")
        print("     - Features d'entraînement")
        print("     - Logique de metrics")
        print("")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnostic des targets pré-calculés")
    parser.add_argument("--family", type=str, required=True, choices=["glandular", "digestive", "urologic", "respiratory", "epidermal"])
    parser.add_argument("--data_dir", type=str, default="data/family_FIXED",
                        help="Répertoire des données (source de vérité unique)")

    args = parser.parse_args()

    diagnose_targets(args.family, args.data_dir)
