#!/usr/bin/env python3
"""
Diagnostic pour identifier quel chemin de données a été utilisé durant l'entraînement.

Vérifie:
1. Quelle version de train_hovernet_family.py a été utilisée (argument default)
2. Quels fichiers existent dans data/family_data/ vs data/family_FIXED/
3. Les features dans chaque répertoire (shape, CLS std)
4. Recommandation: re-train nécessaire ou pas

Usage:
    python scripts/validation/diagnose_training_data_mismatch.py --family epidermal
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import DEFAULT_FAMILY_DATA_DIR, DEFAULT_FAMILY_FIXED_DIR

def check_directory(dir_path: Path, family: str) -> dict:
    """Vérifie un répertoire et analyse les fichiers."""
    result = {
        "path": str(dir_path),
        "exists": dir_path.exists(),
        "files": {}
    }

    if not dir_path.exists():
        return result

    # Chercher fichiers features et targets
    features_file = dir_path / f"{family}_features.npz"
    targets_file = dir_path / f"{family}_targets.npz"
    fixed_file = dir_path / f"{family}_data_FIXED.npz"

    for file in [features_file, targets_file, fixed_file]:
        if file.exists():
            try:
                data = np.load(file)
                file_info = {
                    "exists": True,
                    "size_mb": file.stat().st_size / 1e6,
                    "keys": list(data.keys()),
                }

                # Analyser features si présentes
                if 'features' in data:
                    features = data['features']
                    cls_tokens = features[:, 0, :]
                    file_info["shape"] = features.shape
                    file_info["dtype"] = str(features.dtype)
                    file_info["cls_std"] = float(cls_tokens.std())
                    file_info["cls_mean"] = float(cls_tokens.mean())

                result["files"][file.name] = file_info
            except Exception as e:
                result["files"][file.name] = {"exists": True, "error": str(e)}
        else:
            result["files"][file.name] = {"exists": False}

    return result

def print_diagnostic(family: str):
    """Affiche diagnostic complet."""
    print("=" * 80)
    print(f"DIAGNOSTIC: TRAINING DATA MISMATCH - {family.upper()}")
    print("=" * 80)
    print()

    # Vérifier les deux chemins possibles
    data_dir = Path(DEFAULT_FAMILY_DATA_DIR)
    fixed_dir = Path(DEFAULT_FAMILY_FIXED_DIR)

    print("1. RÉPERTOIRE DATA (DEFAULT_FAMILY_DATA_DIR):")
    print(f"   Path: {data_dir}")
    data_info = check_directory(data_dir, family)

    if not data_info["exists"]:
        print("   ❌ N'EXISTE PAS")
    else:
        print("   ✅ EXISTE")
        for filename, info in data_info["files"].items():
            if info["exists"]:
                print(f"\n   📄 {filename}:")
                print(f"      Size: {info.get('size_mb', 0):.1f} MB")
                print(f"      Keys: {info.get('keys', [])}")
                if 'cls_std' in info:
                    print(f"      Features shape: {info['shape']}")
                    print(f"      CLS std: {info['cls_std']:.3f}")
            else:
                print(f"\n   ❌ {filename}: N'EXISTE PAS")

    print("\n" + "=" * 80)
    print("2. RÉPERTOIRE FIXED (DEFAULT_FAMILY_FIXED_DIR):")
    print(f"   Path: {fixed_dir}")
    fixed_info = check_directory(fixed_dir, family)

    if not fixed_info["exists"]:
        print("   ❌ N'EXISTE PAS")
    else:
        print("   ✅ EXISTE")
        for filename, info in fixed_info["files"].items():
            if info["exists"]:
                print(f"\n   📄 {filename}:")
                print(f"      Size: {info.get('size_mb', 0):.1f} MB")
                print(f"      Keys: {info.get('keys', [])}")
                if 'cls_std' in info:
                    print(f"      Features shape: {info['shape']}")
                    print(f"      CLS std: {info['cls_std']:.3f}")
            else:
                print(f"\n   ❌ {filename}: N'EXISTE PAS")

    print("\n" + "=" * 80)
    print("3. DIAGNOSTIC ET RECOMMANDATION:")
    print("=" * 80)

    # Analyser la situation
    data_has_features = data_info["exists"] and data_info["files"].get(f"{family}_features.npz", {}).get("exists", False)
    fixed_has_features = fixed_info["exists"] and fixed_info["files"].get(f"{family}_features.npz", {}).get("exists", False)
    fixed_has_raw = fixed_info["exists"] and fixed_info["files"].get(f"{family}_data_FIXED.npz", {}).get("exists", False)

    print()
    if not data_has_features and not fixed_has_features:
        print("❌ PROBLÈME CRITIQUE: Aucun fichier features trouvé!")
        print()
        print("CAUSE:")
        print("  - Les features n'ont jamais été extraites")
        print("  - Ou les chemins sont incorrects")
        print()
        print("ACTION REQUISE:")
        print("  1. Exécuter: python scripts/preprocessing/extract_features_from_fixed.py --family", family)
        print("  2. Vérifier que les fichiers sont créés dans", DEFAULT_FAMILY_DATA_DIR)
        print("  3. Re-train le modèle")

    elif data_has_features and not fixed_has_features:
        print("✅ CONFIGURATION CORRECTE:")
        print(f"  - Features trouvées dans {DEFAULT_FAMILY_DATA_DIR}")
        print(f"  - CLS std: {data_info['files'][f'{family}_features.npz']['cls_std']:.3f}")
        print()
        print("MAIS le test montre NP Dice 0.0000!")
        print()
        print("CAUSE POSSIBLE:")
        print("  - Le modèle a été entraîné AVANT que l'argument default soit fixé")
        print(f"  - Il a peut-être chargé depuis {DEFAULT_FAMILY_FIXED_DIR} (maintenant absent)")
        print("  - Features d'entraînement ≠ features de test")
        print()
        print("ACTION REQUISE:")
        print("  1. RE-TRAIN le modèle avec le chemin corrigé:")
        print(f"     python scripts/training/train_hovernet_family.py --family {family} --epochs 50 --augment")
        print("  2. Le script utilisera maintenant DEFAULT_FAMILY_DATA_DIR par défaut")
        print("  3. Test attendu: NP Dice ~0.95")

    elif fixed_has_features and not data_has_features:
        print("⚠️ CONFIGURATION INVERSÉE:")
        print(f"  - Features trouvées dans {DEFAULT_FAMILY_FIXED_DIR}")
        print(f"  - Mais DEFAULT_FAMILY_DATA_DIR est vide!")
        print()
        print("ACTION REQUISE:")
        print(f"  1. Copier ou déplacer les features vers {DEFAULT_FAMILY_DATA_DIR}:")
        print(f"     mkdir -p {DEFAULT_FAMILY_DATA_DIR}")
        print(f"     cp {fixed_dir}/{family}_features.npz {data_dir}/")
        print(f"     cp {fixed_dir}/{family}_targets.npz {data_dir}/")
        print("  2. Ou mettre à jour DEFAULT_FAMILY_DATA_DIR dans src/constants.py")

    elif fixed_has_raw and not fixed_has_features:
        print("⚠️ DONNÉES NON EXTRAITES:")
        print(f"  - Fichier FIXED.npz trouvé dans {DEFAULT_FAMILY_FIXED_DIR}")
        print("  - Mais pas de features extraites!")
        print()
        print("ACTION REQUISE:")
        print(f"  1. Extraire features depuis FIXED.npz:")
        print(f"     python scripts/preprocessing/extract_features_from_fixed.py --family {family}")
        print(f"  2. Re-train le modèle")

    print()

def main():
    parser = argparse.ArgumentParser(description="Diagnostic training data mismatch")
    parser.add_argument("--family", required=True,
                       choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"],
                       help="Famille à diagnostiquer")
    args = parser.parse_args()

    print_diagnostic(args.family)

    return 0

if __name__ == "__main__":
    sys.exit(main())
