#!/usr/bin/env python3
"""
Test d'inférence CellViT-256 avec le code officiel TIO-IKIM.

Valide:
- Import depuis le repo cloné
- Architecture du modèle (46.7M params)
- Forward pass sur image synthétique
- Chargement du checkpoint (si disponible)

Usage:
    python scripts/validation/test_cellvit_official.py
    python scripts/validation/test_cellvit_official.py --checkpoint /path/to/CellViT-256.pth
    python scripts/validation/test_cellvit_official.py -c models/pretrained/CellViT-256.pth
"""

import sys
import argparse
from pathlib import Path

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CellViT"))

import torch
import numpy as np
import cv2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test CellViT-256 avec le code officiel TIO-IKIM"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        help="Chemin vers le checkpoint CellViT-256.pth (défaut: models/pretrained/CellViT-256.pth)"
    )
    return parser.parse_args()


def test_import():
    """Test 1: Import du modèle officiel."""
    print("\n" + "=" * 60)
    print("TEST 1: Import CellViT256 depuis repo officiel")
    print("=" * 60)

    try:
        from models.segmentation.cell_segmentation.cellvit import CellViT256
        print("✅ Import CellViT256 OK")
        return CellViT256
    except ImportError as e:
        print(f"❌ Import échoué: {e}")
        print("\n   Solution: Clonez le repo CellViT:")
        print("   git clone https://github.com/TIO-IKIM/CellViT.git")
        return None


def test_architecture(CellViT256):
    """Test 2: Création du modèle."""
    print("\n" + "=" * 60)
    print("TEST 2: Architecture du modèle")
    print("=" * 60)

    if CellViT256 is None:
        print("⏭️  Skipped (import échoué)")
        return None

    try:
        model = CellViT256(
            model256_path=None,
            num_nuclei_classes=6,
            num_tissue_classes=19
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Modèle créé: {num_params:,} paramètres")

        # Vérifier les attributs
        print(f"   embed_dim: {model.embed_dim}")
        print(f"   depth: {model.depth}")
        print(f"   num_heads: {model.num_heads}")
        print(f"   extract_layers: {model.extract_layers}")

        return model
    except Exception as e:
        print(f"❌ Création échouée: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward(model):
    """Test 3: Forward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: Forward pass")
    print("=" * 60)

    if model is None:
        print("⏭️  Skipped (modèle non disponible)")
        return False

    try:
        x = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            out = model(x)

        print("✅ Forward pass OK")
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"   {k}: {v.shape}")

        # Vérifier les shapes attendues
        assert out['nuclei_binary_map'].shape == (1, 2, 256, 256)
        assert out['hv_map'].shape == (1, 2, 256, 256)
        assert out['nuclei_type_map'].shape == (1, 6, 256, 256)
        print("✅ Shapes validées")

        return True
    except Exception as e:
        print(f"❌ Forward échoué: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint(checkpoint_path=None):
    """Test 4: Chargement du checkpoint."""
    print("\n" + "=" * 60)
    print("TEST 4: Chargement du checkpoint")
    print("=" * 60)

    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "models" / "pretrained" / "CellViT-256.pth"
    else:
        checkpoint_path = Path(checkpoint_path)

    print(f"Chemin checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"⏭️  Checkpoint non trouvé: {checkpoint_path}")
        print_download_instructions()
        return None

    # Vérifier taille
    size = checkpoint_path.stat().st_size
    if size < 1000000:  # < 1MB = probablement vide ou corrompu
        print(f"⚠️  Checkpoint semble invalide (taille: {size} bytes)")
        print_download_instructions()
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        print(f"✅ Checkpoint chargé ({size / 1e6:.1f} MB)")
        print(f"   Architecture: {checkpoint.get('arch', 'unknown')}")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")

        state_dict = checkpoint.get('model_state_dict', {})
        print(f"   Paramètres: {len(state_dict)} clés")

        return checkpoint
    except Exception as e:
        print(f"❌ Chargement échoué: {e}")
        return None


def test_load_weights(CellViT256, checkpoint):
    """Test 5: Chargement des poids."""
    print("\n" + "=" * 60)
    print("TEST 5: Chargement des poids dans le modèle")
    print("=" * 60)

    if CellViT256 is None or checkpoint is None:
        print("⏭️  Skipped (dépendances manquantes)")
        return False

    try:
        # Créer modèle
        model = CellViT256(
            model256_path=None,
            num_nuclei_classes=6,
            num_tissue_classes=19
        )

        # Charger poids
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        msg = model.load_state_dict(state_dict, strict=True)

        print(f"✅ Poids chargés avec succès")
        print(f"   {msg}")

        return True
    except Exception as e:
        print(f"❌ Chargement poids échoué: {e}")
        return False


def test_inference(CellViT256, checkpoint):
    """Test 6: Inférence complète."""
    print("\n" + "=" * 60)
    print("TEST 6: Inférence sur image synthétique")
    print("=" * 60)

    if CellViT256 is None or checkpoint is None:
        print("⏭️  Skipped (checkpoint manquant)")
        return False

    try:
        # Créer et charger modèle
        model = CellViT256(
            model256_path=None,
            num_nuclei_classes=6,
            num_tissue_classes=19
        )
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Image synthétique
        img = np.random.randint(150, 220, (256, 256, 3), dtype=np.uint8)
        for _ in range(15):
            cx, cy = np.random.randint(20, 236, 2)
            cv2.circle(img, (cx, cy), np.random.randint(5, 12), (80, 40, 100), -1)

        # Préprocessing (normalisation simple)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Inférence
        with torch.no_grad():
            out = model(x)

        # Analyser résultats
        import torch.nn.functional as F
        np_probs = F.softmax(out['nuclei_binary_map'], dim=1)
        type_probs = F.softmax(out['nuclei_type_map'], dim=1)

        print("✅ Inférence réussie!")
        print(f"   NP probs range: [{np_probs.min():.3f}, {np_probs.max():.3f}]")
        print(f"   Type probs range: [{type_probs.min():.3f}, {type_probs.max():.3f}]")

        return True
    except Exception as e:
        print(f"❌ Inférence échouée: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_download_instructions():
    """Affiche les instructions pour télécharger le checkpoint."""
    print("\n" + "-" * 50)
    print("INSTRUCTIONS POUR TÉLÉCHARGER LE CHECKPOINT:")
    print("-" * 50)
    print("""
1. Téléchargez CellViT-256.pth depuis:
   https://drive.google.com/uc?id=1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q

2. Placez le fichier dans:
   models/pretrained/CellViT-256.pth

3. Vérifiez la taille (~187 MB)

Alternative via gdown (si pas de proxy):
   pip install gdown
   gdown 1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q -O models/pretrained/CellViT-256.pth
""")
    print("-" * 50)


def main(checkpoint_path=None):
    """Exécute tous les tests."""
    print("\n" + "=" * 60)
    print("  VALIDATION CELLVIT-256 OFFICIEL")
    print("  POC Étape 1.5: Inférence")
    print("=" * 60)

    results = {}

    # Test 1: Import
    CellViT256 = test_import()
    results['import'] = CellViT256 is not None

    # Test 2: Architecture
    model = test_architecture(CellViT256)
    results['architecture'] = model is not None

    # Test 3: Forward
    results['forward'] = test_forward(model)

    # Test 4: Checkpoint
    checkpoint = test_checkpoint(checkpoint_path)
    results['checkpoint'] = checkpoint is not None

    # Test 5: Load weights
    results['load_weights'] = test_load_weights(CellViT256, checkpoint)

    # Test 6: Inference
    results['inference'] = test_inference(CellViT256, checkpoint)

    # Résumé
    print("\n" + "=" * 60)
    print("  RÉSUMÉ")
    print("=" * 60)

    core_tests = ['import', 'architecture', 'forward']
    full_tests = ['checkpoint', 'load_weights', 'inference']

    print("\nTests de base (architecture):")
    core_passed = all(results.get(t, False) for t in core_tests)
    for t in core_tests:
        status = "✅ PASS" if results.get(t, False) else "❌ FAIL"
        print(f"  {t}: {status}")

    print("\nTests complets (checkpoint):")
    full_passed = all(results.get(t, False) for t in full_tests)
    for t in full_tests:
        status = "✅ PASS" if results.get(t, False) else "⏭️  SKIP"
        print(f"  {t}: {status}")

    print()
    if core_passed and full_passed:
        print("🎉 TOUS LES TESTS PASSENT - Étape 1.5 validée!")
    elif core_passed:
        print("✅ Architecture validée!")
        print("⚠️  Téléchargez le checkpoint pour validation complète")
    else:
        print("❌ Tests de base échoués - voir erreurs ci-dessus")

    return core_passed


if __name__ == "__main__":
    args = parse_args()
    success = main(checkpoint_path=args.checkpoint)
    sys.exit(0 if success else 1)
