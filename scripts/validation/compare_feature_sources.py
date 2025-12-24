#!/usr/bin/env python3
"""
Compare features from different locations to identify which ones the model was trained on.

Checks:
1. data/family_data/{family}_features.npz (test script default)
2. data/cache/family_data/{family}_features.npz (training script default)
3. Compares shapes, dtypes, CLS std, file timestamps
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def analyze_features(features_path: Path) -> dict:
    """Analyze features file and return diagnostics."""
    if not features_path.exists():
        return {"exists": False}

    data = np.load(features_path)

    # Support both 'features' and 'layer_24' keys
    if 'features' in data:
        features = data['features']
    elif 'layer_24' in data:
        features = data['layer_24']
    else:
        return {"exists": True, "error": f"No features found. Keys: {list(data.keys())}"}

    # Calculate CLS std
    cls_tokens = features[:, 0, :]  # (N, 1536)
    cls_std = cls_tokens.std()

    # Get file info
    file_stat = features_path.stat()
    file_size_gb = file_stat.st_size / 1e9
    file_mtime = datetime.fromtimestamp(file_stat.st_mtime)

    return {
        "exists": True,
        "path": str(features_path),
        "shape": features.shape,
        "dtype": str(features.dtype),
        "cls_std": float(cls_std),
        "size_gb": file_size_gb,
        "modified": file_mtime.strftime("%Y-%m-%d %H:%M:%S"),
        "mean": float(features.mean()),
        "std": float(features.std()),
        "min": float(features.min()),
        "max": float(features.max()),
    }

def main():
    parser = argparse.ArgumentParser(description="Compare feature sources")
    parser.add_argument("--family", required=True)
    args = parser.parse_args()

    print("=" * 80)
    print(f"COMPARAISON SOURCES FEATURES: {args.family}")
    print("=" * 80)
    print("")

    # Check test script location
    test_path = Path("data/family_data") / f"{args.family}_features.npz"
    print(f"1. TEST SCRIPT DEFAULT:")
    print(f"   Path: {test_path}")
    test_info = analyze_features(test_path)
    if test_info["exists"]:
        print(f"   ✅ EXISTS")
        print(f"   Shape: {test_info['shape']}")
        print(f"   CLS std: {test_info['cls_std']:.3f}")
        print(f"   Modified: {test_info['modified']}")
        print(f"   Size: {test_info['size_gb']:.2f} GB")
        print(f"   Range: [{test_info['min']:.3f}, {test_info['max']:.3f}]")
    else:
        print(f"   ❌ NOT FOUND")
    print("")

    # Check training script location
    train_path = Path("data/cache/family_data") / f"{args.family}_features.npz"
    print(f"2. TRAINING SCRIPT DEFAULT:")
    print(f"   Path: {train_path}")
    train_info = analyze_features(train_path)
    if train_info["exists"]:
        print(f"   ✅ EXISTS")
        print(f"   Shape: {train_info['shape']}")
        print(f"   CLS std: {train_info['cls_std']:.3f}")
        print(f"   Modified: {train_info['modified']}")
        print(f"   Size: {train_info['size_gb']:.2f} GB")
        print(f"   Range: [{train_info['min']:.3f}, {train_info['max']:.3f}]")
    else:
        print(f"   ❌ NOT FOUND")
    print("")

    # Compare
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("")

    if test_info["exists"] and train_info["exists"]:
        # Check if they're the same
        same_shape = test_info["shape"] == train_info["shape"]
        same_cls_std = abs(test_info["cls_std"] - train_info["cls_std"]) < 0.01
        same_modified = test_info["modified"] == train_info["modified"]

        if same_shape and same_cls_std and same_modified:
            print("✅ IDENTIQUES: Les deux fichiers sont probablement les mêmes")
        else:
            print("❌ DIFFÉRENTS: Les fichiers sont DIFFÉRENTS!")
            if not same_shape:
                print(f"   - Shape mismatch: {test_info['shape']} vs {train_info['shape']}")
            if not same_cls_std:
                print(f"   - CLS std mismatch: {test_info['cls_std']:.3f} vs {train_info['cls_std']:.3f}")
                print(f"     Δ = {abs(test_info['cls_std'] - train_info['cls_std']):.3f}")
            if not same_modified:
                print(f"   - Modified time different:")
                print(f"     Test: {test_info['modified']}")
                print(f"     Train: {train_info['modified']}")
        print("")

        # Recommend which to use
        print("RECOMMANDATION:")
        if train_info["cls_std"] > 0.70 and train_info["cls_std"] < 0.90:
            print(f"  → Utiliser TRAINING location: {train_path}")
            print(f"    CLS std = {train_info['cls_std']:.3f} ✅ (dans plage 0.70-0.90)")
        elif test_info["cls_std"] > 0.70 and test_info["cls_std"] < 0.90:
            print(f"  → Utiliser TEST location: {test_path}")
            print(f"    CLS std = {test_info['cls_std']:.3f} ✅ (dans plage 0.70-0.90)")
        else:
            print(f"  ⚠️  AUCUNE des deux locations n'a CLS std correct!")
            print(f"     Ré-extraire les features avec extract_features_from_fixed.py")

    elif train_info["exists"]:
        print(f"✅ UTILISER: {train_path}")
        print(f"   (location utilisée pour training)")

    elif test_info["exists"]:
        print(f"⚠️  ATTENTION: Test path existe mais pas training path")
        print(f"   Cela explique pourquoi le modèle ne fonctionne pas!")
        print(f"   Le modèle a été entraîné sur des features DIFFÉRENTES")

    else:
        print(f"❌ ERREUR: Aucune des deux locations n'existe!")
        print(f"   Exécutez: python scripts/preprocessing/extract_features_from_fixed.py --family {args.family}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
