#!/usr/bin/env python3
"""
Script de diagnostic pour identifier la cause des scores OOD élevés.

Vérifie les 3 suspects:
1. Normalisation (mismatch train/inference)
2. Mapping labels PanNuke
3. Préprocessing images

Usage:
    python scripts/validation/diagnose_ood_issue.py --data_dir /path/to/PanNuke --fold 0
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Normalisation attendue pour H-optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

# Normalisation ImageNet (erreur courante)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def check_image_format(images: np.ndarray):
    """
    SUSPECT #1: Vérification du format des images.
    """
    print("\n" + "=" * 60)
    print("🔍 SUSPECT #1: FORMAT DES IMAGES")
    print("=" * 60)

    print(f"\nShape: {images.shape}")
    print(f"Dtype: {images.dtype}")
    print(f"Min: {images.min()}, Max: {images.max()}")

    # Vérifier si les images sont dans le bon format
    if images.dtype == np.uint8:
        print("✅ Dtype correct (uint8)")
        if images.max() <= 255 and images.min() >= 0:
            print("✅ Range correct [0, 255]")
        else:
            print(f"⚠️ Range anormal: [{images.min()}, {images.max()}]")
    elif images.dtype == np.float32 or images.dtype == np.float64:
        print("⚠️ Images déjà en float!")
        if images.max() <= 1.0:
            print("   → Probablement normalisées [0, 1]")
            print("   → RISQUE: Double normalisation!")
        elif images.max() <= 255:
            print("   → Range [0, 255] en float")
        else:
            print(f"   → Range anormal: [{images.min():.2f}, {images.max():.2f}]")

    # Statistiques par canal (RGB)
    print("\nStatistiques par canal (sur 100 images):")
    sample = images[:100]
    for c, name in enumerate(['R', 'G', 'B']):
        channel = sample[..., c].astype(np.float32) / 255.0
        mean = channel.mean()
        std = channel.std()
        print(f"  {name}: mean={mean:.4f}, std={std:.4f}")

    # Comparer avec les statistiques H-optimus-0 attendues
    print("\nComparaison avec H-optimus-0 attendu:")
    print(f"  Expected Mean: {HOPTIMUS_MEAN}")
    print(f"  Expected Std:  {HOPTIMUS_STD}")


def check_label_mapping(types: np.ndarray):
    """
    SUSPECT #2: Vérification du mapping des labels.
    """
    print("\n" + "=" * 60)
    print("🔍 SUSPECT #2: MAPPING DES LABELS")
    print("=" * 60)

    from src.models.organ_head import PANNUKE_ORGANS

    print(f"\nShape types: {types.shape}")
    print(f"Dtype: {types.dtype}")

    # Extraire les noms uniques
    unique_types = np.unique(types)
    print(f"\nOrganes uniques dans le dataset ({len(unique_types)}):")

    # Créer le mapping attendu
    organ_to_idx = {organ: i for i, organ in enumerate(PANNUKE_ORGANS)}

    mismatches = []
    for t in unique_types:
        t_str = str(t).strip()
        if t_str in organ_to_idx:
            print(f"  ✅ '{t_str}' → index {organ_to_idx[t_str]}")
        else:
            # Chercher correspondance partielle
            found = False
            for organ in PANNUKE_ORGANS:
                if organ.lower() in t_str.lower() or t_str.lower() in organ.lower():
                    print(f"  ⚠️ '{t_str}' → match partiel avec '{organ}'")
                    found = True
                    break
            if not found:
                print(f"  ❌ '{t_str}' → NON TROUVÉ!")
                mismatches.append(t_str)

    if mismatches:
        print(f"\n⚠️ {len(mismatches)} organes non mappés: {mismatches}")
    else:
        print("\n✅ Tous les organes sont mappés correctement")

    # Distribution des types
    print("\nDistribution des types:")
    type_counts = Counter(types)
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {str(t):20}: {count:5} ({count/len(types)*100:.1f}%)")


def check_preprocessing_consistency(images: np.ndarray, data_dir: Path, fold: int):
    """
    SUSPECT #3: Vérification de la cohérence du préprocessing.
    """
    print("\n" + "=" * 60)
    print("🔍 SUSPECT #3: COHÉRENCE DU PRÉPROCESSING")
    print("=" * 60)

    from torchvision import transforms

    # Créer les deux méthodes de préprocessing
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])

    # Méthode d'inférence (manuelle)
    def preprocess_inference(image):
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        img = image.astype(np.float32) / 255.0
        for c in range(3):
            img[:, :, c] = (img[:, :, c] - HOPTIMUS_MEAN[c]) / HOPTIMUS_STD[c]
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float()

    # Comparer sur 10 images
    print("\nComparaison Train vs Inference preprocessing:")
    print(f"  (sur 10 images aléatoires)")

    differences = []
    for i in range(min(10, len(images))):
        img = images[i]

        # Méthode train
        train_tensor = transform_train(img)

        # Méthode inference
        inf_tensor = preprocess_inference(img)

        # Différence
        diff = (train_tensor - inf_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        differences.append(max_diff)

        if i < 3:
            print(f"  Image {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    avg_max_diff = np.mean(differences)
    if avg_max_diff < 1e-5:
        print(f"\n✅ Préprocessing cohérent (diff moyenne: {avg_max_diff:.2e})")
    elif avg_max_diff < 1e-3:
        print(f"\n⚠️ Légères différences de préprocessing (diff: {avg_max_diff:.2e})")
    else:
        print(f"\n❌ DIFFÉRENCES SIGNIFICATIVES! (diff: {avg_max_diff:.4f})")

    # Vérifier les features extraites vs temps réel
    print("\n" + "-" * 40)
    print("Vérification des features pré-extraites...")

    features_path = PROJECT_ROOT / "data" / "cache" / "pannuke_features" / f"fold{fold}_features.npz"
    if features_path.exists():
        data = np.load(features_path)
        if 'layer_24' in data:
            features = data['layer_24']
            print(f"  Shape: {features.shape}")
            print(f"  Dtype: {features.dtype}")
            print(f"  Range: [{features.min():.4f}, {features.max():.4f}]")
            print(f"  Mean: {features.mean():.4f}, Std: {features.std():.4f}")

            # CLS token stats
            cls_tokens = features[:, 0, :]
            print(f"\n  CLS tokens stats:")
            print(f"    Mean: {cls_tokens.mean():.4f}")
            print(f"    Std: {cls_tokens.std():.4f}")
            print(f"    Norm moyenne: {np.linalg.norm(cls_tokens, axis=1).mean():.4f}")

            # Vérifier si les valeurs sont normales
            if np.isnan(features).any():
                print("  ❌ ERREUR: NaN détectés dans les features!")
            elif np.isinf(features).any():
                print("  ❌ ERREUR: Inf détectés dans les features!")
            elif features.std() < 0.1:
                print("  ⚠️ Variance très faible - features peut-être mal extraites")
            else:
                print("  ✅ Features semblent valides")
    else:
        print(f"  ⚠️ Features non trouvées: {features_path}")


def run_live_inference_test(images: np.ndarray, types: np.ndarray):
    """
    Test d'inférence en temps réel pour comparer.
    """
    print("\n" + "=" * 60)
    print("🔍 TEST D'INFÉRENCE EN TEMPS RÉEL")
    print("=" * 60)

    try:
        from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily
        from src.models.organ_head import PANNUKE_ORGANS

        print("\nChargement du modèle...")
        model = OptimusGateInferenceMultiFamily(
            checkpoint_dir=str(PROJECT_ROOT / "models" / "checkpoints"),
        )

        # Tester sur 20 images aléatoires
        n_test = min(20, len(images))
        indices = np.random.choice(len(images), n_test, replace=False)

        correct = 0
        ood_count = 0
        results = []

        print(f"\nTest sur {n_test} images:")
        for i, idx in enumerate(indices):
            img = images[idx]
            expected = str(types[idx]).strip()

            # Prédiction
            result = model.predict(img)

            # Accès correct à l'organe via l'objet OrganPrediction
            organ = result.get('organ')
            predicted = organ.organ_name if organ else 'Unknown'
            ood_score = result.get('ood_score_global', 0)
            is_ood = result.get('is_ood', False)

            # Vérifier le match
            match = expected.lower().replace("_", "").replace("-", "") == \
                    predicted.lower().replace("_", "").replace("-", "")

            if match:
                correct += 1
            if is_ood:
                ood_count += 1

            status = "✅" if match else "❌"
            ood_flag = "🚫" if is_ood else ""
            results.append((expected, predicted, ood_score, match))

            if i < 10:  # Afficher les 10 premiers
                print(f"  {status} Attendu: {expected:15} | Prédit: {predicted:15} | OOD: {ood_score:.3f} {ood_flag}")

        print(f"\n" + "-" * 40)
        print(f"Accuracy: {correct}/{n_test} ({correct/n_test*100:.1f}%)")
        print(f"OOD détectés: {ood_count}/{n_test} ({ood_count/n_test*100:.1f}%)")

        # Statistiques OOD
        ood_scores = [r[2] for r in results]
        print(f"\nStatistiques OOD:")
        print(f"  Min: {min(ood_scores):.3f}")
        print(f"  Max: {max(ood_scores):.3f}")
        print(f"  Mean: {np.mean(ood_scores):.3f}")
        print(f"  Median: {np.median(ood_scores):.3f}")

        if np.mean(ood_scores) > 0.7:
            print("\n⚠️ SCORE OOD MOYEN ÉLEVÉ - Probable problème de normalisation!")
        elif ood_count > n_test * 0.5:
            print("\n⚠️ TROP D'OOD DÉTECTÉS - Vérifier la calibration du seuil")

    except Exception as e:
        print(f"❌ Erreur lors du test d'inférence: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Diagnostic OOD")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Chemin vers PanNuke")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold à tester")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Sauter le test d'inférence")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fold = args.fold

    print("=" * 60)
    print("🔬 DIAGNOSTIC OOD - CellViT-Optimus")
    print("=" * 60)
    print(f"\nData dir: {data_dir}")
    print(f"Fold: {fold}")

    # Charger les données
    fold_dir = data_dir / f"fold{fold}"
    images_path = fold_dir / "images.npy"
    types_path = fold_dir / "types.npy"

    if not images_path.exists():
        print(f"\n❌ Images non trouvées: {images_path}")
        return

    print("\nChargement des données...")
    images = np.load(images_path)
    types = np.load(types_path) if types_path.exists() else None

    print(f"  Images: {images.shape}")
    if types is not None:
        print(f"  Types: {types.shape}")

    # Exécuter les vérifications
    check_image_format(images)

    if types is not None:
        check_label_mapping(types)
    else:
        print("\n⚠️ Fichier types.npy non trouvé - skip label mapping")

    check_preprocessing_consistency(images, data_dir, fold)

    if not args.skip_inference:
        run_live_inference_test(images, types)

    print("\n" + "=" * 60)
    print("🏁 DIAGNOSTIC TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()
