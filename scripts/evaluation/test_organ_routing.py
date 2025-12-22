#!/usr/bin/env python3
"""
Teste le routage OrganHead → Famille pour chaque échantillon de test.

Ce script vérifie que:
1. OrganHead prédit le bon organe
2. L'organe prédit correspond bien à la famille attendue
3. Le routage famille → modèle HoVer-Net fonctionne correctement

Permet de détecter:
- Erreurs de prédiction d'organe (OrganHead mal calibré)
- Erreurs de mapping organe → famille (ORGAN_TO_FAMILY incorrect)
- Problèmes d'intégration multi-famille

Usage:
    python scripts/evaluation/test_organ_routing.py \
        --test_samples_dir data/test_samples_by_family \
        --checkpoint_dir models/checkpoints \
        --output_dir results/routing_validation
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.organ_families import (
    FAMILIES,
    ORGAN_TO_FAMILY,
    get_family
)
from src.models.loader import ModelLoader
from src.preprocessing import preprocess_image, validate_features


# Mapping organes PanNuke (ordre des classes OrganHead)
PANNUKE_ORGANS = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix",
    "Colon", "Esophagus", "HeadNeck", "Kidney", "Liver",
    "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
    "Stomach", "Testis", "Thyroid", "Uterus"
]


def test_routing(
    test_samples_dir: Path,
    checkpoint_dir: Path,
    device: str = "cuda"
):
    """
    Teste le routage organe pour tous les échantillons.

    Returns:
        {
            "total_samples": int,
            "correct_organ": int,
            "correct_family": int,
            "errors": List[Dict]
        }
    """
    print("=" * 70)
    print("TEST DU ROUTAGE ORGANE → FAMILLE")
    print("=" * 70)
    print()

    # Charger OrganHead
    organ_head_path = checkpoint_dir / "organ_head_best.pth"

    if not organ_head_path.exists():
        raise FileNotFoundError(f"OrganHead manquant: {organ_head_path}")

    print("Chargement OrganHead...")
    organ_head = ModelLoader.load_organ_head(
        checkpoint_path=organ_head_path,
        device=device
    )
    print("✅ OrganHead chargé")

    # Charger backbone
    print("\nChargement H-optimus-0...")
    backbone = ModelLoader.load_hoptimus0(device=device)
    print("✅ Backbone chargé")

    # Statistiques
    total_samples = 0
    correct_organ = 0
    correct_family = 0
    errors = []

    # Pour chaque famille
    for family in FAMILIES:
        family_dir = test_samples_dir / family
        samples_file = family_dir / "test_samples.npz"

        if not samples_file.exists():
            print(f"⚠️  {family}: Aucun échantillon")
            continue

        # Charger échantillons
        data = np.load(samples_file)
        images = data["images"]
        organs_gt = data["organs"]

        n_samples = len(images)
        print(f"\n{family}: {n_samples} échantillons")

        # Tester chaque échantillon
        for i in tqdm(range(n_samples)):
            image = images[i]
            organ_gt = str(organs_gt[i]).strip()
            family_gt = get_family(organ_gt)

            # Preprocessing
            tensor = preprocess_image(image, device=device)

            # Extraction features
            with torch.no_grad():
                features = backbone.forward_features(tensor)

                # Validation
                validation = validate_features(features)
                if not validation["valid"]:
                    print(f"\n⚠️  {family}/{i}: {validation['message']}")
                    continue

                # CLS token
                cls_token = features[:, 0, :]

                # Prédiction organe
                logits = organ_head(cls_token)
                probs = torch.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()

                organ_pred = PANNUKE_ORGANS[pred_idx]

            # Vérifier prédiction
            total_samples += 1

            organ_correct = (organ_pred == organ_gt)
            if organ_correct:
                correct_organ += 1

            # Vérifier famille
            try:
                family_pred = get_family(organ_pred)
                family_correct = (family_pred == family_gt)

                if family_correct:
                    correct_family += 1
            except ValueError:
                family_pred = "UNKNOWN"
                family_correct = False

            # Enregistrer erreurs
            if not organ_correct or not family_correct:
                error = {
                    "family_gt": family_gt,
                    "organ_gt": organ_gt,
                    "organ_pred": organ_pred,
                    "family_pred": family_pred,
                    "confidence": float(confidence),
                    "organ_correct": organ_correct,
                    "family_correct": family_correct,
                    "sample_idx": i
                }
                errors.append(error)

    # Rapport
    print("\n" + "=" * 70)
    print("RAPPORT DU ROUTAGE")
    print("=" * 70)
    print()
    print(f"Total échantillons: {total_samples}")
    print(f"Organe correct:     {correct_organ} / {total_samples} ({correct_organ/total_samples*100:.1f}%)")
    print(f"Famille correcte:   {correct_family} / {total_samples} ({correct_family/total_samples*100:.1f}%)")
    print()

    # Erreurs détaillées
    if errors:
        print(f"Erreurs détectées: {len(errors)}")
        print()

        # Grouper par type
        organ_errors = [e for e in errors if not e["organ_correct"]]
        family_errors = [e for e in errors if e["organ_correct"] and not e["family_correct"]]

        if organ_errors:
            print(f"❌ Erreurs de prédiction d'organe: {len(organ_errors)}")
            for e in organ_errors[:10]:  # Afficher max 10
                print(f"   {e['organ_gt']:15s} → {e['organ_pred']:15s} ({e['confidence']:.2%})")

        if family_errors:
            print(f"\n⚠️  Erreurs de routage famille: {len(family_errors)}")
            for e in family_errors[:10]:
                print(f"   {e['organ_pred']:15s} → {e['family_gt']} attendu, {e['family_pred']} prédit")
    else:
        print("✅ Aucune erreur détectée - Routage parfait!")

    return {
        "total_samples": total_samples,
        "correct_organ": correct_organ,
        "correct_family": correct_family,
        "organ_accuracy": correct_organ / total_samples if total_samples > 0 else 0,
        "family_accuracy": correct_family / total_samples if total_samples > 0 else 0,
        "errors": errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="Teste le routage OrganHead → Famille"
    )
    parser.add_argument(
        "--test_samples_dir",
        type=Path,
        required=True,
        help="Répertoire avec échantillons de test par famille"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Répertoire avec les checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/routing_validation"),
        help="Répertoire de sortie"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device PyTorch"
    )

    args = parser.parse_args()

    # Créer output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = test_routing(
            test_samples_dir=args.test_samples_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device
        )

        # Sauvegarder résultats
        with open(args.output_dir / "routing_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Résultats sauvegardés: {args.output_dir / 'routing_results.json'}")

        # Recommandations
        print("\n" + "=" * 70)
        print("RECOMMANDATIONS")
        print("=" * 70)
        print()

        if results["organ_accuracy"] < 0.95:
            print("⚠️  Accuracy organe < 95%")
            print("   → Ré-calibrer OrganHead (Temperature Scaling)")
            print("   → Vérifier normalisation H-optimus-0")
            print()

        if results["family_accuracy"] < 0.99:
            print("⚠️  Accuracy famille < 99%")
            print("   → Vérifier ORGAN_TO_FAMILY mapping")
            print()

        if results["organ_accuracy"] >= 0.95 and results["family_accuracy"] >= 0.99:
            print("✅ Routage fonctionne correctement")
            print("   → Si problèmes persistent, chercher ailleurs:")
            print("     - Modèles HoVer-Net individuels")
            print("     - Post-processing watershed")
            print("     - Instance mismatch train/eval")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
