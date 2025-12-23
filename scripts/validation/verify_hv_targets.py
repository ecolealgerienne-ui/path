#!/usr/bin/env python3
"""
Script de vérification rapide des HV targets.

Vérifie que les targets dans les .npz sont bien:
- dtype: float32
- range: [-1, 1]
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import validate_targets


def verify_hv_targets(targets_path: Path) -> dict:
    """
    Vérifie les HV targets d'un fichier .npz.

    Returns:
        {
            'valid': bool,
            'dtype': str,
            'range': [min, max],
            'mean': float,
            'std': float,
            'errors': List[str]
        }
    """
    print(f"\n🔍 Vérification: {targets_path.name}")
    print("─" * 60)

    # Charger le fichier
    data = np.load(targets_path)

    np_targets = data['np_targets']
    hv_targets = data['hv_targets']
    nt_targets = data['nt_targets']

    # Statistiques HV
    hv_dtype = str(hv_targets.dtype)
    hv_min = float(hv_targets.min())
    hv_max = float(hv_targets.max())
    hv_mean = float(hv_targets.mean())
    hv_std = float(hv_targets.std())

    print(f"HV Targets:")
    print(f"  Dtype:  {hv_dtype}")
    print(f"  Range:  [{hv_min:.4f}, {hv_max:.4f}]")
    print(f"  Mean:   {hv_mean:.4f}")
    print(f"  Std:    {hv_std:.4f}")

    # Validation avec module centralisé
    try:
        validation = validate_targets(
            np_targets[0],
            hv_targets[0],
            nt_targets[0],
            strict=False
        )

        valid = validation["valid"]
        errors = validation["errors"]

        if valid:
            print(f"\n✅ VALIDATION OK")
        else:
            print(f"\n❌ VALIDATION ÉCHOUÉE:")
            for error in errors:
                print(f"   • {error}")

        return {
            'valid': valid,
            'dtype': hv_dtype,
            'range': [hv_min, hv_max],
            'mean': hv_mean,
            'std': hv_std,
            'errors': errors
        }

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        return {
            'valid': False,
            'dtype': hv_dtype,
            'range': [hv_min, hv_max],
            'mean': hv_mean,
            'std': hv_std,
            'errors': [str(e)]
        }


def main():
    """Vérifie tous les fichiers *_targets.npz."""

    # Répertoire des données
    cache_dir = PROJECT_ROOT / "data" / "cache" / "family_data_FIXED"

    if not cache_dir.exists():
        print(f"❌ Répertoire introuvable: {cache_dir}")
        sys.exit(1)

    # Trouver tous les fichiers *_targets.npz
    target_files = sorted(cache_dir.glob("*_targets.npz"))

    if not target_files:
        print(f"❌ Aucun fichier *_targets.npz trouvé dans {cache_dir}")
        sys.exit(1)

    print(f"\n📁 Répertoire: {cache_dir}")
    print(f"📊 Fichiers trouvés: {len(target_files)}")

    results = {}
    for targets_path in target_files:
        family_name = targets_path.stem.replace("_targets", "")
        results[family_name] = verify_hv_targets(targets_path)

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)

    all_valid = True
    for family_name, result in results.items():
        status = "✅" if result['valid'] else "❌"
        print(f"{status} {family_name:15s} - dtype: {result['dtype']:10s} - range: [{result['range'][0]:.4f}, {result['range'][1]:.4f}]")
        if not result['valid']:
            all_valid = False

    print("\n" + "=" * 60)
    if all_valid:
        print("🎉 TOUS LES FICHIERS SONT VALIDES")
        print("\nLes HV targets sont bien normalisés [-1, 1] en float32.")
        print("Le problème de AJI/PQ vient donc bien de la gradient_loss faible.")
        return 0
    else:
        print("⚠️ CERTAINS FICHIERS SONT INVALIDES")
        print("\nVérifiez les erreurs ci-dessus et régénérez les données si nécessaire.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
