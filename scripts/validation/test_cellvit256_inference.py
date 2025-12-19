#!/usr/bin/env python3
"""
Test d'inférence CellViT-256 sur une image.

Valide l'étape 1.5 du plan POC:
- Chargement du checkpoint
- Inférence sur une image test
- Vérification des sorties

Usage:
    python scripts/validation/test_cellvit256_inference.py
"""

import sys
from pathlib import Path

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import cv2


def test_checkpoint_loading():
    """Test 1: Chargement du checkpoint."""
    print("\n" + "=" * 60)
    print("TEST 1: Chargement du checkpoint CellViT-256")
    print("=" * 60)

    checkpoint_path = PROJECT_ROOT / "models" / "pretrained" / "CellViT-256.pth"

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint non trouvé: {checkpoint_path}")
        print("   Téléchargez CellViT-256.pth et placez-le dans models/pretrained/")
        return None

    print(f"📦 Chargement: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"✅ Architecture: {checkpoint.get('arch', 'unknown')}")
    print(f"✅ Epoch: {checkpoint.get('epoch', 'unknown')}")

    state_dict = checkpoint.get('model_state_dict', {})
    print(f"✅ Paramètres: {len(state_dict)} clés")

    return checkpoint


def test_model_architecture():
    """Test 2: Vérification de l'architecture du modèle."""
    print("\n" + "=" * 60)
    print("TEST 2: Architecture du modèle")
    print("=" * 60)

    try:
        from src.inference.cellvit256_model import CellViT256

        model = CellViT256()
        print(f"✅ Modèle créé: {sum(p.numel() for p in model.parameters()):,} params")

        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)

        print(f"✅ nuclei_binary_map: {out['nuclei_binary_map'].shape}")
        print(f"✅ hv_map: {out['hv_map'].shape}")
        print(f"✅ nuclei_type_maps: {out['nuclei_type_maps'].shape}")

        return model

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_weight_compatibility(checkpoint, model):
    """Test 3: Compatibilité des poids."""
    print("\n" + "=" * 60)
    print("TEST 3: Compatibilité des poids")
    print("=" * 60)

    if checkpoint is None or model is None:
        print("⏭️  Skipped (dépendances manquantes)")
        return False

    state_dict = checkpoint.get('model_state_dict', {})
    model_state = model.state_dict()

    # Comparer les clés
    checkpoint_keys = set(state_dict.keys())
    model_keys = set(model_state.keys())

    common = checkpoint_keys & model_keys
    only_checkpoint = checkpoint_keys - model_keys
    only_model = model_keys - checkpoint_keys

    print(f"📊 Clés checkpoint: {len(checkpoint_keys)}")
    print(f"📊 Clés modèle: {len(model_keys)}")
    print(f"📊 Clés communes: {len(common)}")

    if only_checkpoint:
        print(f"\n⚠️  Clés uniquement dans checkpoint ({len(only_checkpoint)}):")
        for k in list(only_checkpoint)[:10]:
            print(f"   - {k}")
        if len(only_checkpoint) > 10:
            print(f"   ... et {len(only_checkpoint) - 10} autres")

    if only_model:
        print(f"\n⚠️  Clés uniquement dans modèle ({len(only_model)}):")
        for k in list(only_model)[:10]:
            print(f"   - {k}")

    # Vérifier les shapes
    shape_mismatches = []
    for key in common:
        if state_dict[key].shape != model_state[key].shape:
            shape_mismatches.append((key, state_dict[key].shape, model_state[key].shape))

    if shape_mismatches:
        print(f"\n❌ Shapes incompatibles ({len(shape_mismatches)}):")
        for k, s1, s2 in shape_mismatches[:5]:
            print(f"   {k}: checkpoint {list(s1)} vs modèle {list(s2)}")
        return False
    else:
        print(f"\n✅ Toutes les shapes communes sont compatibles")

    return len(common) > 0


def test_inference_synthetic():
    """Test 4: Inférence sur image synthétique."""
    print("\n" + "=" * 60)
    print("TEST 4: Inférence sur image synthétique")
    print("=" * 60)

    checkpoint_path = PROJECT_ROOT / "models" / "pretrained" / "CellViT-256.pth"

    if not checkpoint_path.exists():
        print("⏭️  Skipped (checkpoint manquant)")
        return False

    try:
        from src.inference.cellvit256_model import load_cellvit256_from_checkpoint

        # Créer une image synthétique (simule tissu H&E)
        img = np.random.randint(150, 220, (256, 256, 3), dtype=np.uint8)
        # Ajouter des "noyaux" sombres
        for _ in range(20):
            cx, cy = np.random.randint(20, 236, 2)
            cv2.circle(img, (cx, cy), np.random.randint(5, 15), (80, 40, 100), -1)

        print(f"📷 Image synthétique: {img.shape}")

        # Préprocessing
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - 0.5) / 0.5  # Normalisation simple

        # Charger et inférer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Device: {device}")

        model = load_cellvit256_from_checkpoint(str(checkpoint_path), device)

        x = x.to(device)
        with torch.no_grad():
            out = model(x)

        print(f"✅ Inférence réussie!")
        print(f"   nuclei_binary_map: {out['nuclei_binary_map'].shape}")
        print(f"   hv_map: {out['hv_map'].shape}")
        print(f"   nuclei_type_maps: {out['nuclei_type_maps'].shape}")

        # Vérifier les valeurs
        np_probs = torch.softmax(out['nuclei_binary_map'], dim=1)
        print(f"   NP probs range: [{np_probs.min():.3f}, {np_probs.max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Exécute tous les tests."""
    print("\n" + "=" * 60)
    print("  VALIDATION CELLVIT-256 - Étape 1.5 POC")
    print("=" * 60)

    results = {}

    # Test 1
    checkpoint = test_checkpoint_loading()
    results['checkpoint'] = checkpoint is not None

    # Test 2
    model = test_model_architecture()
    results['architecture'] = model is not None

    # Test 3
    results['compatibility'] = test_weight_compatibility(checkpoint, model)

    # Test 4
    results['inference'] = test_inference_synthetic()

    # Résumé
    print("\n" + "=" * 60)
    print("  RÉSUMÉ")
    print("=" * 60)

    all_passed = all(results.values())

    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test}: {status}")

    print()
    if all_passed:
        print("🎉 TOUS LES TESTS PASSENT - Étape 1.5 validée!")
    else:
        print("⚠️  Certains tests ont échoué - voir détails ci-dessus")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
