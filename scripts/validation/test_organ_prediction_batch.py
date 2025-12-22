#!/usr/bin/env python3
"""
Test batch de prédiction d'organe sur les images de sample.

Supporte Temperature Scaling pour comparer les confiances calibrées.

Usage:
    python scripts/validation/test_organ_prediction_batch.py
    python scripts/validation/test_organ_prediction_batch.py --samples_dir data/samples
    python scripts/validation/test_organ_prediction_batch.py --temperature 0.5
    python scripts/validation/test_organ_prediction_batch.py --compare_temps
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Imports des modules centralisés (Phase 1 Refactoring)
from src.preprocessing import create_hoptimus_transform
from src.models.loader import ModelLoader

# Mapping nom fichier → organe attendu
def extract_expected_organ(filename: str) -> str:
    """Extrait l'organe attendu du nom de fichier."""
    # breast_01.png → Breast
    # colon_02.png → Colon
    # prostate_03.png → Prostate
    name = Path(filename).stem  # breast_01
    organ = name.rsplit('_', 1)[0]  # breast
    return organ.capitalize()  # Breast


def test_with_temperature(
    backbone, organ_head, image_files, transform, device,
    temperature: float, PANNUKE_ORGANS, verbose: bool = True
) -> dict:
    """
    Teste les images avec une température donnée.

    Args:
        temperature: T pour softmax(logits / T). T=1.0 = pas de calibration.

    Returns:
        Dict avec résultats et statistiques
    """
    results = []

    with torch.no_grad():
        for img_path in image_files:
            expected = extract_expected_organ(img_path.name)

            # Charger l'image
            img = np.array(Image.open(img_path).convert('RGB'))
            if img.dtype != np.uint8:
                img = img.clip(0, 255).astype(np.uint8)

            # Transform + features
            tensor = transform(img).unsqueeze(0).to(device)
            features = backbone.forward_features(tensor).float()
            cls_token = features[:, 0, :]

            # Prédiction avec température
            logits = organ_head(cls_token)
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
            predicted = PANNUKE_ORGANS[pred_idx]

            match = predicted.lower() == expected.lower()

            results.append({
                'file': img_path.name,
                'expected': expected,
                'predicted': predicted,
                'confidence': confidence,
                'match': match,
            })

    n_correct = sum(1 for r in results if r['match'])
    n_total = len(results)
    mean_conf = np.mean([r['confidence'] for r in results])

    return {
        'results': results,
        'n_correct': n_correct,
        'n_total': n_total,
        'accuracy': n_correct / n_total * 100,
        'mean_confidence': mean_conf,
    }


def main():
    parser = argparse.ArgumentParser(description="Test batch organ prediction")
    parser.add_argument("--samples_dir", type=str, default="data/samples",
                        help="Répertoire des images de test")
    parser.add_argument("--checkpoint", type=str,
                        default="models/checkpoints/organ_head_best.pth",
                        help="Checkpoint OrganHead")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Température pour calibration (T=1.0 = pas de calibration)")
    parser.add_argument("--compare_temps", action="store_true",
                        help="Comparer plusieurs températures (1.0, 0.5, 0.1)")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"❌ Répertoire non trouvé: {samples_dir}")
        sys.exit(1)

    # Trouver toutes les images
    image_files = sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
    if not image_files:
        print(f"❌ Aucune image trouvée dans {samples_dir}")
        sys.exit(1)

    print("=" * 70)
    print("TEST BATCH PRÉDICTION D'ORGANE")
    print("=" * 70)
    print(f"Images: {len(image_files)}")
    print(f"Checkpoint: {args.checkpoint}")

    # Charger H-optimus-0 via le chargeur centralisé (Phase 1 Refactoring)
    print("\n⏳ Chargement H-optimus-0...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = ModelLoader.load_hoptimus0(device=device)
    print("  ✓ H-optimus-0 chargé")

    # Charger OrganHead
    print("⏳ Chargement OrganHead...")
    from src.models.organ_head import OrganHead, PANNUKE_ORGANS

    organ_head = OrganHead(embed_dim=1536, n_organs=len(PANNUKE_ORGANS))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # strict=False pour ignorer les clés OOD (cls_mean, cls_cov_inv)
    organ_head.load_state_dict(checkpoint['model_state_dict'], strict=False)
    organ_head.eval().to(device)

    ood_threshold = checkpoint.get('ood_threshold', 50.0)
    calibrated_temp = checkpoint.get('temperature', None)
    print(f"  ✓ OrganHead chargé (OOD threshold: {ood_threshold:.2f})")
    if calibrated_temp:
        print(f"  ✓ Température calibrée trouvée: T={calibrated_temp:.4f}")

    # Transform centralisé (Phase 1 Refactoring)
    transform = create_hoptimus_transform()

    # Mode comparaison multi-températures
    if args.compare_temps:
        temperatures = [1.0, 0.5, 0.25, 0.1]
        if calibrated_temp and calibrated_temp not in temperatures:
            temperatures.append(calibrated_temp)
            temperatures = sorted(temperatures, reverse=True)

        print("\n" + "=" * 70)
        print("COMPARAISON MULTI-TEMPÉRATURES")
        print("=" * 70)
        print(f"\nTempératures testées: {temperatures}")
        print("\nRappel: T > 1 = confiance réduite, T < 1 = confiance augmentée")

        all_results = {}
        for temp in temperatures:
            all_results[temp] = test_with_temperature(
                backbone, organ_head, image_files, transform, device,
                temp, PANNUKE_ORGANS, verbose=False
            )

        # Tableau comparatif par image
        print("\n" + "=" * 70)
        print("CONFIANCE PAR IMAGE ET TEMPÉRATURE")
        print("=" * 70)

        # Header
        header = f"{'Image':20} | {'Attendu':10} |"
        for temp in temperatures:
            header += f" T={temp:4} |"
        print(header)
        print("-" * len(header))

        # Chaque image
        for i, img_path in enumerate(image_files):
            expected = extract_expected_organ(img_path.name)
            row = f"{img_path.name:20} | {expected:10} |"
            for temp in temperatures:
                result = all_results[temp]['results'][i]
                conf = result['confidence']
                status = "✓" if result['match'] else "✗"
                row += f" {conf:5.1%}{status} |"
            print(row)

        # Résumé statistique
        print("\n" + "=" * 70)
        print("RÉSUMÉ STATISTIQUE")
        print("=" * 70)
        print(f"\n{'Température':12} | {'Accuracy':10} | {'Conf. Moy.':12} | {'Conf. Min':10} | {'Conf. Max':10}")
        print("-" * 65)
        for temp in temperatures:
            stats = all_results[temp]
            confs = [r['confidence'] for r in stats['results']]
            print(f"T = {temp:<8.4f} | {stats['accuracy']:8.1f}%  | {stats['mean_confidence']:10.1%}   | {min(confs):8.1%}   | {max(confs):8.1%}")

        print("\n" + "=" * 70)
        print("INTERPRÉTATION")
        print("=" * 70)
        print("""
• T = 1.0 : Pas de calibration (confiances "brutes")
• T < 1.0 : Augmente les confiances (logits amplifiés)
• T > 1.0 : Réduit les confiances (logits atténués)

⚠️ Une température trop basse (T < 0.2) rend TOUTES les prédictions
   très confiantes (~100%), même les erreurs potentielles.

✓ Une bonne calibration devrait:
  - Garder l'accuracy inchangée (la prédiction ne change pas)
  - Avoir confiance moyenne ≈ accuracy
  - Avoir confiance basse pour les cas difficiles
""")

    else:
        # Mode simple avec une seule température
        temperature = args.temperature
        print(f"\nTempérature utilisée: T = {temperature}")

        print("\n" + "=" * 70)
        print("RÉSULTATS")
        print("=" * 70)

        stats = test_with_temperature(
            backbone, organ_head, image_files, transform, device,
            temperature, PANNUKE_ORGANS
        )

        for r in stats['results']:
            status = "✅" if r['match'] else "❌"
            print(f"{status} {r['file']:20} | Attendu: {r['expected']:12} | Prédit: {r['predicted']:12} ({r['confidence']:.1%})")

        # Résumé
        print("\n" + "=" * 70)
        print("RÉSUMÉ")
        print("=" * 70)
        print(f"Total: {stats['n_correct']}/{stats['n_total']} corrects ({stats['accuracy']:.1f}%)")
        print(f"Confiance moyenne: {stats['mean_confidence']:.1%}")

        # Détails des erreurs
        errors = [r for r in stats['results'] if not r['match']]
        if errors:
            print(f"\n❌ Erreurs ({len(errors)}):")
            for e in errors:
                print(f"   {e['file']}: {e['expected']} → {e['predicted']} ({e['confidence']:.1%})")
        else:
            print("\n🎉 TOUTES LES PRÉDICTIONS SONT CORRECTES!")

        # Code de sortie
        sys.exit(0 if stats['n_correct'] == stats['n_total'] else 1)


if __name__ == "__main__":
    main()
