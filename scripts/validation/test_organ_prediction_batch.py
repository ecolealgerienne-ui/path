#!/usr/bin/env python3
"""
Test batch de prédiction d'organe sur les images de sample.

Usage:
    python scripts/validation/test_organ_prediction_batch.py
    python scripts/validation/test_organ_prediction_batch.py --samples_dir data/samples
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Normalisation H-optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

# Mapping nom fichier → organe attendu
def extract_expected_organ(filename: str) -> str:
    """Extrait l'organe attendu du nom de fichier."""
    # breast_01.png → Breast
    # colon_02.png → Colon
    # prostate_03.png → Prostate
    name = Path(filename).stem  # breast_01
    organ = name.rsplit('_', 1)[0]  # breast
    return organ.capitalize()  # Breast


def create_transform():
    """Transform identique à extract_features.py"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])


def main():
    parser = argparse.ArgumentParser(description="Test batch organ prediction")
    parser.add_argument("--samples_dir", type=str, default="data/samples",
                        help="Répertoire des images de test")
    parser.add_argument("--checkpoint", type=str,
                        default="models/checkpoints/organ_head_best.pth",
                        help="Checkpoint OrganHead")
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

    # Charger H-optimus-0
    print("\n⏳ Chargement H-optimus-0...")
    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    backbone.eval().to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    print("  ✓ H-optimus-0 chargé")

    # Charger OrganHead
    print("⏳ Chargement OrganHead...")
    from src.models.organ_head import OrganHead, PANNUKE_ORGANS

    organ_head = OrganHead(embed_dim=1536, n_organs=len(PANNUKE_ORGANS))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    organ_head.load_state_dict(checkpoint['model_state_dict'])
    organ_head.eval().to(device)

    ood_threshold = checkpoint.get('ood_threshold', 50.0)
    print(f"  ✓ OrganHead chargé (OOD threshold: {ood_threshold:.2f})")

    # Transform
    transform = create_transform()

    # Tester chaque image
    print("\n" + "=" * 70)
    print("RÉSULTATS")
    print("=" * 70)

    results = []

    with torch.no_grad():
        for img_path in image_files:
            expected = extract_expected_organ(img_path.name)

            # Charger l'image
            img = np.array(Image.open(img_path).convert('RGB'))

            # Convertir en uint8 si nécessaire
            if img.dtype != np.uint8:
                img = img.clip(0, 255).astype(np.uint8)

            # Transform
            tensor = transform(img).unsqueeze(0).to(device)

            # Extraction features via forward_features() (avec LayerNorm)
            features = backbone.forward_features(tensor).float()
            cls_token = features[:, 0, :]  # (1, 1536)

            # Prédiction
            logits = organ_head(cls_token)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
            predicted = PANNUKE_ORGANS[pred_idx]

            # Vérification
            match = predicted.lower() == expected.lower()
            status = "✅" if match else "❌"

            results.append({
                'file': img_path.name,
                'expected': expected,
                'predicted': predicted,
                'confidence': confidence,
                'match': match,
            })

            print(f"{status} {img_path.name:20} | Attendu: {expected:12} | Prédit: {predicted:12} ({confidence:.1%})")

    # Résumé
    n_correct = sum(1 for r in results if r['match'])
    n_total = len(results)
    accuracy = n_correct / n_total * 100

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Total: {n_correct}/{n_total} corrects ({accuracy:.1f}%)")

    # Détails des erreurs
    errors = [r for r in results if not r['match']]
    if errors:
        print(f"\n❌ Erreurs ({len(errors)}):")
        for e in errors:
            print(f"   {e['file']}: {e['expected']} → {e['predicted']} ({e['confidence']:.1%})")
    else:
        print("\n🎉 TOUTES LES PRÉDICTIONS SONT CORRECTES!")

    # Code de sortie
    sys.exit(0 if n_correct == n_total else 1)


if __name__ == "__main__":
    main()
