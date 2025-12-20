#!/usr/bin/env python3
"""
Vérification complète du pipeline avant entraînement.

Ce script vérifie:
1. Intégrité des données PanNuke
2. Cohérence du préprocessing (ToPILImage avec uint8)
3. Extraction de features correcte
4. Scripts d'entraînement

Usage:
    python scripts/validation/verify_pipeline.py --data_dir /home/amar/data/PanNuke
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ajouter le chemin racine
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_section(title: str):
    """Affiche un titre de section."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def verify_pannuke_data(data_dir: Path) -> dict:
    """Vérifie l'intégrité des données PanNuke."""
    check_section("1. VÉRIFICATION DES DONNÉES PANNUKE")

    results = {"valid": True, "folds": {}}

    for fold in [0, 1, 2]:
        fold_dir = data_dir / f"fold{fold}"
        fold_result = {"exists": False, "images": None, "masks": None, "types": None}

        if not fold_dir.exists():
            print(f"❌ fold{fold}: Répertoire manquant")
            results["valid"] = False
            results["folds"][fold] = fold_result
            continue

        fold_result["exists"] = True

        # Vérifier images.npy
        images_path = fold_dir / "images.npy"
        if images_path.exists():
            images = np.load(images_path)
            fold_result["images"] = {
                "shape": images.shape,
                "dtype": str(images.dtype),
                "min": float(images.min()),
                "max": float(images.max()),
            }

            # Vérification critique: format des images
            if images.dtype == np.float64 and images.max() > 1.0:
                print(f"⚠️  fold{fold}/images.npy: float64 [0, 255] - ATTENTION")
                print(f"   → Le script extract_features.py convertira en uint8")
            elif images.dtype == np.uint8:
                print(f"✅ fold{fold}/images.npy: uint8 - Format idéal")
            else:
                print(f"ℹ️  fold{fold}/images.npy: {images.dtype} [{images.min():.1f}, {images.max():.1f}]")
        else:
            print(f"❌ fold{fold}/images.npy: Manquant")
            results["valid"] = False

        # Vérifier masks.npy
        masks_path = fold_dir / "masks.npy"
        if masks_path.exists():
            masks = np.load(masks_path)
            fold_result["masks"] = {
                "shape": masks.shape,
                "dtype": str(masks.dtype),
            }
            print(f"✅ fold{fold}/masks.npy: {masks.shape}")
        else:
            print(f"❌ fold{fold}/masks.npy: Manquant")
            results["valid"] = False

        # Vérifier types.npy
        types_path = fold_dir / "types.npy"
        if types_path.exists():
            types = np.load(types_path)
            unique_types = np.unique(types)
            fold_result["types"] = {
                "count": len(types),
                "unique": len(unique_types),
                "organs": list(unique_types),
            }
            print(f"✅ fold{fold}/types.npy: {len(types)} samples, {len(unique_types)} organes")
        else:
            print(f"❌ fold{fold}/types.npy: Manquant")
            results["valid"] = False

        results["folds"][fold] = fold_result

    return results


def verify_preprocessing_consistency() -> dict:
    """Vérifie que le préprocessing est cohérent entre extraction et inférence."""
    check_section("2. VÉRIFICATION DU PRÉPROCESSING")

    results = {"valid": True, "tests": []}

    try:
        from torchvision import transforms
        import torch

        # Constantes H-optimus-0
        HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
        HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

        # Créer le transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
        ])

        # Test 1: Image uint8 [0, 255]
        print("\nTest 1: Image uint8 [0, 255]")
        img_uint8 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        tensor1 = transform(img_uint8)
        print(f"  Input: uint8 [{img_uint8.min()}, {img_uint8.max()}]")
        print(f"  Output: tensor [{tensor1.min():.3f}, {tensor1.max():.3f}]")
        results["tests"].append(("uint8", True))
        print("  ✅ OK")

        # Test 2: Image float64 [0, 255] SANS conversion (problème original)
        print("\nTest 2: Image float64 [0, 255] SANS conversion (bug)")
        img_float = img_uint8.astype(np.float64)
        tensor2_bad = transform(img_float)
        diff_bad = (tensor1 - tensor2_bad).abs().max().item()
        print(f"  Input: float64 [{img_float.min():.1f}, {img_float.max():.1f}]")
        print(f"  Différence avec uint8: {diff_bad:.3f}")
        if diff_bad > 1.0:
            print(f"  ⚠️  CORRUPTION DÉTECTÉE - C'était le bug!")
            results["tests"].append(("float64_raw", False))
        else:
            print(f"  ✅ Pas de corruption (inattendu)")
            results["tests"].append(("float64_raw", True))

        # Test 3: Image float64 [0, 255] AVEC conversion (fix)
        print("\nTest 3: Image float64 [0, 255] AVEC conversion uint8 (fix)")
        img_float_fixed = img_float.clip(0, 255).astype(np.uint8)
        tensor3_fixed = transform(img_float_fixed)
        diff_fixed = (tensor1 - tensor3_fixed).abs().max().item()
        print(f"  Après conversion: uint8 [{img_float_fixed.min()}, {img_float_fixed.max()}]")
        print(f"  Différence avec uint8 original: {diff_fixed:.6f}")
        if diff_fixed < 0.001:
            print(f"  ✅ IDENTIQUE - Le fix fonctionne!")
            results["tests"].append(("float64_fixed", True))
        else:
            print(f"  ❌ Différence inattendue")
            results["tests"].append(("float64_fixed", False))
            results["valid"] = False

    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["valid"] = False
        results["error"] = str(e)

    return results


def verify_extract_features_script() -> dict:
    """Vérifie le script extract_features.py."""
    check_section("3. VÉRIFICATION DE extract_features.py")

    results = {"valid": True, "checks": []}

    script_path = Path(__file__).parent.parent / "preprocessing" / "extract_features.py"

    if not script_path.exists():
        print(f"❌ Script non trouvé: {script_path}")
        results["valid"] = False
        return results

    # Lire le contenu
    content = script_path.read_text()

    # Check 1: Conversion uint8 présente
    if "astype(np.uint8)" in content and "clip(0, 255)" in content:
        print("✅ Conversion uint8 avec clip présente")
        results["checks"].append(("uint8_conversion", True))
    else:
        print("❌ Conversion uint8 manquante!")
        results["checks"].append(("uint8_conversion", False))
        results["valid"] = False

    # Check 2: ToPILImage utilisé
    if "ToPILImage" in content:
        print("✅ ToPILImage utilisé")
        results["checks"].append(("ToPILImage", True))
    else:
        print("⚠️  ToPILImage non trouvé")
        results["checks"].append(("ToPILImage", False))

    # Check 3: Normalisation H-optimus-0
    if "0.707223" in content and "0.211883" in content:
        print("✅ Normalisation H-optimus-0 correcte")
        results["checks"].append(("normalization", True))
    else:
        print("❌ Normalisation H-optimus-0 incorrecte")
        results["checks"].append(("normalization", False))
        results["valid"] = False

    return results


def verify_inference_scripts() -> dict:
    """Vérifie les scripts d'inférence."""
    check_section("4. VÉRIFICATION DES SCRIPTS D'INFÉRENCE")

    results = {"valid": True, "files": {}}

    inference_files = [
        "src/inference/optimus_gate_inference_multifamily.py",
        "src/inference/optimus_gate_inference.py",
        "src/inference/hoptimus_hovernet.py",
    ]

    root = Path(__file__).parent.parent.parent

    for rel_path in inference_files:
        file_path = root / rel_path
        file_result = {"exists": False, "checks": []}

        if not file_path.exists():
            print(f"❌ {rel_path}: Fichier manquant")
            results["valid"] = False
            results["files"][rel_path] = file_result
            continue

        file_result["exists"] = True
        content = file_path.read_text()

        # Check: utilise create_hoptimus_transform()
        if "create_hoptimus_transform" in content:
            file_result["checks"].append(("transform_function", True))
        else:
            file_result["checks"].append(("transform_function", False))
            print(f"⚠️  {rel_path}: create_hoptimus_transform() non utilisé")

        # Check: conversion uint8
        if "astype(np.uint8)" in content:
            file_result["checks"].append(("uint8_conversion", True))
        else:
            file_result["checks"].append(("uint8_conversion", False))
            print(f"⚠️  {rel_path}: Conversion uint8 non trouvée")

        # Résultat
        if all(c[1] for c in file_result["checks"]):
            print(f"✅ {rel_path}: OK")

        results["files"][rel_path] = file_result

    return results


def verify_training_scripts() -> dict:
    """Vérifie les scripts d'entraînement."""
    check_section("5. VÉRIFICATION DES SCRIPTS D'ENTRAÎNEMENT")

    results = {"valid": True, "files": {}}

    training_files = [
        "scripts/training/train_organ_head.py",
        "scripts/training/train_hovernet.py",
        "scripts/training/train_hovernet_family.py",
    ]

    root = Path(__file__).parent.parent.parent

    for rel_path in training_files:
        file_path = root / rel_path
        file_result = {"exists": False, "issues": []}

        if not file_path.exists():
            print(f"⚠️  {rel_path}: Fichier non trouvé (optionnel)")
            results["files"][rel_path] = file_result
            continue

        file_result["exists"] = True
        content = file_path.read_text()

        # Les scripts d'entraînement utilisent des features pré-extraites
        # Donc ils ne devraient PAS avoir de préprocessing d'image

        # Check: pas de ToPILImage (car features déjà extraites)
        if "ToPILImage" in content:
            file_result["issues"].append("ToPILImage trouvé - devrait utiliser features pré-extraites")
            print(f"⚠️  {rel_path}: Contient ToPILImage (vérifier si attendu)")

        # Check: charge des features .npz
        if ".npz" in content or "features" in content.lower():
            print(f"✅ {rel_path}: Utilise des features pré-extraites")
        else:
            print(f"ℹ️  {rel_path}: Vérifier la source des données")

        results["files"][rel_path] = file_result

    return results


def run_quick_extraction_test(data_dir: Path) -> dict:
    """Test rapide d'extraction sur quelques images."""
    check_section("6. TEST D'EXTRACTION RAPIDE")

    results = {"valid": True}

    try:
        import torch
        from torchvision import transforms

        # Charger quelques images du fold0
        images_path = data_dir / "fold0" / "images.npy"
        if not images_path.exists():
            print("⚠️  fold0/images.npy non trouvé, skip du test")
            return results

        images = np.load(images_path)[:5]  # 5 premières images
        print(f"Test sur {len(images)} images...")

        # Transform
        HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
        HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
        ])

        # Tester chaque image
        for i, img in enumerate(images):
            # Convertir en uint8 (le fix)
            if img.dtype != np.uint8:
                img = img.clip(0, 255).astype(np.uint8)

            tensor = transform(img)

            # Vérifier les valeurs
            if tensor.min() < -5 or tensor.max() > 5:
                print(f"  ⚠️  Image {i}: Valeurs extrêmes [{tensor.min():.2f}, {tensor.max():.2f}]")
            else:
                print(f"  ✅ Image {i}: [{tensor.min():.2f}, {tensor.max():.2f}] - OK")

        print("\n✅ Test d'extraction réussi!")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["valid"] = False
        results["error"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(description="Vérification du pipeline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Chemin vers PanNuke")
    parser.add_argument("--skip_extraction_test", action="store_true",
                        help="Skip le test d'extraction")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("\n" + "="*60)
    print("  VÉRIFICATION COMPLÈTE DU PIPELINE")
    print("="*60)
    print(f"Data dir: {data_dir}")

    all_valid = True

    # 1. Données PanNuke
    result1 = verify_pannuke_data(data_dir)
    all_valid &= result1["valid"]

    # 2. Préprocessing
    result2 = verify_preprocessing_consistency()
    all_valid &= result2["valid"]

    # 3. Script extract_features.py
    result3 = verify_extract_features_script()
    all_valid &= result3["valid"]

    # 4. Scripts d'inférence
    result4 = verify_inference_scripts()
    all_valid &= result4["valid"]

    # 5. Scripts d'entraînement
    result5 = verify_training_scripts()
    all_valid &= result5["valid"]

    # 6. Test d'extraction
    if not args.skip_extraction_test:
        result6 = run_quick_extraction_test(data_dir)
        all_valid &= result6["valid"]

    # Résumé final
    check_section("RÉSUMÉ FINAL")

    if all_valid:
        print("🎉 TOUTES LES VÉRIFICATIONS PASSENT!")
        print("\nVous pouvez lancer l'extraction:")
        print(f"  python scripts/preprocessing/extract_features.py \\")
        print(f"      --data_dir {data_dir} --fold 0 --all_layers")
    else:
        print("⚠️  CERTAINES VÉRIFICATIONS ONT ÉCHOUÉ")
        print("Corrigez les problèmes avant de continuer.")
        sys.exit(1)


if __name__ == "__main__":
    main()
