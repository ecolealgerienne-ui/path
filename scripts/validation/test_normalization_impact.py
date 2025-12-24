#!/usr/bin/env python3
"""
Test l'impact de la normalisation (H-optimus-0 vs ImageNet) sur CLS std.

L'expert suspecte que:
- Training features: std=0.82 (avec une certaine normalisation)
- Inference features: std=0.66 (avec une autre normalisation)

Ce script teste les DEUX normalisations pour identifier laquelle était utilisée au training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from torchvision import transforms
from src.models.loader import ModelLoader

# Normalisations à tester
NORMALIZATIONS = {
    "H-optimus-0": {
        "mean": (0.707223, 0.578729, 0.703617),
        "std": (0.211883, 0.230117, 0.177517),
    },
    "ImageNet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


def create_transform(norm_name: str):
    """Crée transform avec normalisation spécifiée."""
    norm = NORMALIZATIONS[norm_name]
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])


def test_normalization(image: np.ndarray, norm_name: str, backbone):
    """Teste une normalisation et retourne CLS std."""
    transform = create_transform(norm_name)

    # Prétraitement
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)

    tensor = transform(image).unsqueeze(0).to("cuda")

    # Extraction features
    with torch.no_grad():
        features = backbone.forward_features(tensor)

    # CLS token
    cls_token = features[:, 0, :].cpu().numpy()
    cls_std = cls_token.std()

    return cls_std, features


def main():
    print("=" * 70)
    print("TEST NORMALISATION: H-optimus-0 vs ImageNet")
    print("=" * 70)

    # Charger modèle
    print("\n🔧 Chargement H-optimus-0...")
    backbone = ModelLoader.load_hoptimus0(device="cuda")

    # Charger une image de test
    print("📸 Chargement image test...")
    # Utiliser une des images de fold2 samples si disponible
    try:
        sample_path = Path("data/temp_fold2_samples/sample_00000.npz")
        if sample_path.exists():
            data = np.load(sample_path)
            image = data['image']
            print(f"   ✓ Image chargée: {image.shape}, dtype={image.dtype}")
        else:
            # Créer une image synthétique si pas de sample
            print("   ⚠️ Pas de sample trouvé, création image synthétique")
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    except Exception as e:
        print(f"   ⚠️ Erreur chargement: {e}")
        print("   Création image synthétique")
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Tester les deux normalisations
    print("\n" + "=" * 70)
    print("RÉSULTATS")
    print("=" * 70)

    results = {}
    for norm_name in ["H-optimus-0", "ImageNet"]:
        print(f"\n📊 Test avec normalisation {norm_name}")
        print(f"   Mean: {NORMALIZATIONS[norm_name]['mean']}")
        print(f"   Std:  {NORMALIZATIONS[norm_name]['std']}")

        cls_std, features = test_normalization(image, norm_name, backbone)
        results[norm_name] = cls_std

        print(f"\n   ➜ CLS token std: {cls_std:.4f}")

        # Interpréter
        if 0.70 <= cls_std <= 0.90:
            print(f"   ✅ DANS PLAGE ATTENDUE [0.70, 0.90]")
        elif cls_std < 0.40:
            print(f"   ❌ TROP BAS (< 0.40) - LayerNorm manquant?")
        else:
            print(f"   ⚠️ HORS PLAGE [0.70, 0.90]")

        # Comparer avec valeurs connues
        if abs(cls_std - 0.82) < 0.05:
            print(f"   🎯 PROCHE de 0.82 (valeur training supposée)")
        if abs(cls_std - 0.66) < 0.05:
            print(f"   🎯 PROCHE de 0.66 (valeur inference mesurée)")

    # Recommandation
    print("\n" + "=" * 70)
    print("RECOMMANDATION")
    print("=" * 70)

    if abs(results["H-optimus-0"] - 0.82) < abs(results["ImageNet"] - 0.82):
        print("\n✅ H-optimus-0 normalisation semble correcte pour training")
        print(f"   CLS std: {results['H-optimus-0']:.4f} (proche de 0.82)")
    else:
        print("\n⚠️ ImageNet normalisation plus proche de training (std=0.82)")
        print(f"   CLS std: {results['ImageNet']:.4f}")
        print("\n   ⚠️ ATTENTION: H-optimus-0 est censé utiliser ses propres constantes!")
        print("   Cela suggère que les features de training ont été générées")
        print("   avec une normalisation incorrecte.")

    if abs(results["H-optimus-0"] - 0.66) < 0.05:
        print(f"\n⚠️ Inference actuelle donne std={results['H-optimus-0']:.4f} (proche de 0.66)")
        print("   Cela correspond à l'alerte 'Features SUSPECTES' observée.")

    print("\n" + "=" * 70)
    print(f"H-optimus-0: {results['H-optimus-0']:.4f}")
    print(f"ImageNet:    {results['ImageNet']:.4f}")
    print(f"Diff:        {abs(results['H-optimus-0'] - results['ImageNet']):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
