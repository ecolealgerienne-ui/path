#!/usr/bin/env python3
"""
Script de vérification des features H-optimus-0 extraites.

OBJECTIF:
=========
Vérifier que les features extraites sont cohérentes avec la méthode
forward_features() qui inclut le LayerNorm final.

CRITÈRES DE VALIDATION:
=======================
- CLS token std: 0.70 - 0.90 (avec LayerNorm)
- Si std ~0.28, les features sont corrompues (sans LayerNorm)

Usage:
    # Vérifier les features d'un fold
    python scripts/validation/verify_features.py --features_dir data/cache/pannuke_features

    # Vérifier avec comparaison fresh (plus lent, charge H-optimus-0)
    python scripts/validation/verify_features.py --features_dir data/cache/pannuke_features \
        --data_dir /home/amar/data/PanNuke --verify_fresh
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Constantes de validation
EXPECTED_CLS_STD_MIN = 0.70
EXPECTED_CLS_STD_MAX = 0.90
CORRUPTED_CLS_STD_MAX = 0.40  # Features sans LayerNorm ont std ~0.28


def verify_features_file(features_path: Path, verbose: bool = True) -> dict:
    """
    Vérifie un fichier de features.

    Returns:
        dict avec: valid (bool), cls_std, cls_mean, shape, issues (list)
    """
    result = {
        "path": str(features_path),
        "valid": False,
        "issues": []
    }

    if not features_path.exists():
        result["issues"].append(f"Fichier non trouvé: {features_path}")
        return result

    try:
        data = np.load(features_path)
    except Exception as e:
        result["issues"].append(f"Erreur de chargement: {e}")
        return result

    # Déterminer la clé des features
    if 'features' in data.files:
        features = data['features']
        result["key"] = "features"
    elif 'layer_24' in data.files:
        features = data['layer_24']
        result["key"] = "layer_24"
    else:
        result["issues"].append(f"Clé de features non trouvée. Clés disponibles: {data.files}")
        return result

    result["shape"] = features.shape
    result["dtype"] = str(features.dtype)

    # Vérifier la shape
    if len(features.shape) != 3:
        result["issues"].append(f"Shape invalide: {features.shape}, attendu (N, 261, 1536)")
        return result

    n_images, n_tokens, embed_dim = features.shape

    if n_tokens != 261:
        result["issues"].append(f"Nombre de tokens invalide: {n_tokens}, attendu 261")

    if embed_dim != 1536:
        result["issues"].append(f"Dimension d'embedding invalide: {embed_dim}, attendu 1536")

    # Extraire les CLS tokens
    cls_tokens = features[:, 0, :]  # (N, 1536)

    result["cls_std"] = float(cls_tokens.std())
    result["cls_mean"] = float(cls_tokens.mean())
    result["cls_min"] = float(cls_tokens.min())
    result["cls_max"] = float(cls_tokens.max())
    result["n_images"] = n_images

    # Vérifier le std
    if result["cls_std"] < CORRUPTED_CLS_STD_MAX:
        result["issues"].append(
            f"CLS std={result['cls_std']:.4f} < {CORRUPTED_CLS_STD_MAX} "
            f"→ Features CORROMPUES (LayerNorm manquant)!"
        )
    elif result["cls_std"] < EXPECTED_CLS_STD_MIN:
        result["issues"].append(
            f"CLS std={result['cls_std']:.4f} < {EXPECTED_CLS_STD_MIN} "
            f"→ Features suspectes"
        )
    elif result["cls_std"] > EXPECTED_CLS_STD_MAX:
        result["issues"].append(
            f"CLS std={result['cls_std']:.4f} > {EXPECTED_CLS_STD_MAX} "
            f"→ Features anormalement élevées"
        )

    # Vérifier les NaN/Inf
    if np.isnan(features).any():
        result["issues"].append("Features contiennent des NaN!")
    if np.isinf(features).any():
        result["issues"].append("Features contiennent des Inf!")

    result["valid"] = len(result["issues"]) == 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"📁 {features_path.name}")
        print(f"{'='*60}")
        print(f"  Shape: {result['shape']}")
        print(f"  Images: {n_images}")
        print(f"  CLS token stats:")
        print(f"    std:  {result['cls_std']:.4f} (attendu: {EXPECTED_CLS_STD_MIN}-{EXPECTED_CLS_STD_MAX})")
        print(f"    mean: {result['cls_mean']:.4f}")
        print(f"    range: [{result['cls_min']:.4f}, {result['cls_max']:.4f}]")

        if result["valid"]:
            print(f"\n  ✅ VALIDE")
        else:
            print(f"\n  ❌ INVALIDE:")
            for issue in result["issues"]:
                print(f"    → {issue}")

    return result


def verify_fresh_extraction(
    features_path: Path,
    data_dir: Path,
    n_samples: int = 10
) -> dict:
    """
    Compare les features cachées avec une extraction fraîche.

    Permet de détecter les différences de preprocessing.
    """
    import torch
    from torchvision import transforms

    try:
        import timm
    except ImportError:
        return {"valid": False, "issues": ["timm non installé"]}

    result = {"valid": False, "issues": []}

    # Charger les features cachées
    data = np.load(features_path)
    if 'features' in data.files:
        cached_features = data['features']
    elif 'layer_24' in data.files:
        cached_features = data['layer_24']
    else:
        result["issues"].append("Clé de features non trouvée")
        return result

    # Déterminer le fold depuis le nom du fichier
    fold = int(features_path.stem.replace("fold", "").replace("_features", ""))

    # Charger les images
    images_path = data_dir / f"fold{fold}" / "images.npy"
    if not images_path.exists():
        result["issues"].append(f"Images non trouvées: {images_path}")
        return result

    images = np.load(images_path, mmap_mode='r')

    # Sélectionner des échantillons aléatoires
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)

    print(f"\n🔬 Vérification fresh extraction ({n_samples} échantillons)...")

    # Charger H-optimus-0
    print("  Chargement H-optimus-0...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False
    )
    model.eval().to(device)

    # Transform
    HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
    HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])

    # Extraire les features fraîches
    differences = []

    with torch.no_grad():
        for idx in indices:
            img = images[idx]
            if img.dtype != np.uint8:
                img = img.clip(0, 255).astype(np.uint8)

            tensor = transform(img).unsqueeze(0).to(device)
            fresh_features = model.forward_features(tensor).cpu().numpy()

            cached = cached_features[idx:idx+1]

            # Calculer la différence
            diff = np.abs(fresh_features - cached).mean()
            differences.append(diff)

            print(f"    Image {idx}: diff={diff:.6f}")

    mean_diff = np.mean(differences)
    max_diff = np.max(differences)

    print(f"\n  Différence moyenne: {mean_diff:.6f}")
    print(f"  Différence max: {max_diff:.6f}")

    # Les features doivent être identiques (ou très proches à cause de la précision float)
    if mean_diff > 0.001:
        result["issues"].append(
            f"Différence significative entre features cachées et fresh: {mean_diff:.6f}"
        )
    else:
        print(f"  ✅ Features cohérentes avec extraction fraîche")

    result["mean_diff"] = mean_diff
    result["max_diff"] = max_diff
    result["valid"] = len(result["issues"]) == 0

    return result


def main():
    parser = argparse.ArgumentParser(description="Vérification features H-optimus-0")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Répertoire des features (ex: data/cache/pannuke_features)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Répertoire PanNuke pour vérification fresh")
    parser.add_argument("--verify_fresh", action="store_true",
                        help="Vérifier avec extraction fraîche (lent)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Nombre d'échantillons pour vérification fresh")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)

    if not features_dir.exists():
        print(f"❌ Répertoire non trouvé: {features_dir}")
        sys.exit(1)

    # Trouver tous les fichiers de features
    feature_files = sorted(features_dir.glob("fold*_features.npz"))

    if not feature_files:
        print(f"❌ Aucun fichier de features trouvé dans {features_dir}")
        sys.exit(1)

    print("=" * 60)
    print("VÉRIFICATION DES FEATURES H-OPTIMUS-0")
    print("=" * 60)
    print(f"Répertoire: {features_dir}")
    print(f"Fichiers trouvés: {len(feature_files)}")
    print(f"Critères de validation:")
    print(f"  - CLS std attendu: {EXPECTED_CLS_STD_MIN} - {EXPECTED_CLS_STD_MAX}")
    print(f"  - CLS std corrompu: < {CORRUPTED_CLS_STD_MAX}")

    # Vérifier chaque fichier
    all_results = []
    for features_path in feature_files:
        result = verify_features_file(features_path)
        all_results.append(result)

        # Vérification fresh si demandée
        if args.verify_fresh and args.data_dir and result["valid"]:
            fresh_result = verify_fresh_extraction(
                features_path,
                Path(args.data_dir),
                n_samples=args.n_samples
            )
            result["fresh_verification"] = fresh_result
            if not fresh_result["valid"]:
                result["valid"] = False
                result["issues"].extend(fresh_result["issues"])

    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)

    valid_count = sum(1 for r in all_results if r["valid"])
    total_count = len(all_results)

    for result in all_results:
        status = "✅" if result["valid"] else "❌"
        print(f"{status} {Path(result['path']).name}: CLS std={result.get('cls_std', 'N/A'):.4f}")
        if not result["valid"]:
            for issue in result["issues"]:
                print(f"   → {issue}")

    print(f"\nTotal: {valid_count}/{total_count} valides")

    if valid_count == total_count:
        print("\n🎉 TOUTES LES FEATURES SONT VALIDES!")
        print("   → Prêt pour l'entraînement OrganHead/HoVerNet")
        sys.exit(0)
    else:
        print("\n❌ FEATURES INVALIDES DÉTECTÉES!")
        print("   → Ré-extraire les features avec le script corrigé:")
        print("     python scripts/preprocessing/extract_features.py --data_dir /path/to/PanNuke --fold 0")
        sys.exit(1)


if __name__ == "__main__":
    main()
