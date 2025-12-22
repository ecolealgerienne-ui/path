#!/usr/bin/env python3
"""
Script de validation du module centralisé de preprocessing.

Vérifie que toutes les fonctions du module src.data.preprocessing
fonctionnent correctement et détectent les bugs connus.
"""

import sys
import numpy as np
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data import (
    TargetFormat,
    validate_targets,
    resize_targets,
    load_targets,
    prepare_batch_for_training,
)


def test_target_format():
    """Test 1: Vérifier que TargetFormat documente les formats attendus."""
    print("\n" + "=" * 80)
    print("TEST 1: TargetFormat Dataclass")
    print("=" * 80)

    fmt = TargetFormat()

    assert fmt.np_dtype == np.float32, "NP dtype devrait être float32"
    assert fmt.np_min == 0.0 and fmt.np_max == 1.0, "NP range [0, 1]"

    assert fmt.hv_dtype == np.float32, "HV dtype devrait être float32"
    assert fmt.hv_min == -1.0 and fmt.hv_max == 1.0, "HV range [-1, 1]"

    assert fmt.nt_dtype == np.int64, "NT dtype devrait être int64"
    assert fmt.nt_min == 0 and fmt.nt_max == 4, "NT range [0, 4]"

    print("✅ TargetFormat: Tous les champs sont corrects")
    print(f"   NP: {fmt.np_dtype} [{fmt.np_min}, {fmt.np_max}]")
    print(f"   HV: {fmt.hv_dtype} [{fmt.hv_min}, {fmt.hv_max}]")
    print(f"   NT: {fmt.nt_dtype} [{fmt.nt_min}, {fmt.nt_max}]")


def test_validate_targets_correct():
    """Test 2: Validation de targets corrects."""
    print("\n" + "=" * 80)
    print("TEST 2: Validation Targets Corrects")
    print("=" * 80)

    # Créer des targets corrects
    np_target = np.random.rand(256, 256).astype(np.float32)
    hv_target = (np.random.rand(2, 256, 256) * 2 - 1).astype(np.float32)
    nt_target = np.random.randint(0, 5, (256, 256), dtype=np.int64)

    result = validate_targets(np_target, hv_target, nt_target, strict=False)

    assert result["valid"] == True, "Targets corrects devraient être valides"
    assert len(result["errors"]) == 0, "Pas d'erreurs attendues"

    print("✅ Validation: Targets corrects acceptés")


def test_validate_targets_int8_bug():
    """Test 3: Détection du Bug #3 (HV int8)."""
    print("\n" + "=" * 80)
    print("TEST 3: Détection Bug #3 (HV int8)")
    print("=" * 80)

    # Créer des targets avec HV en int8 (BUG)
    np_target = np.random.rand(256, 256).astype(np.float32)
    hv_target = np.random.randint(-127, 128, (2, 256, 256), dtype=np.int8)
    nt_target = np.random.randint(0, 5, (256, 256), dtype=np.int64)

    result = validate_targets(np_target, hv_target, nt_target, strict=False)

    assert result["valid"] == False, "HV int8 devrait être détecté comme invalide"
    assert len(result["errors"]) > 0, "Devrait contenir des erreurs"

    # Vérifier que le message d'erreur mentionne int8 et MSE
    error_text = " ".join(result["errors"])
    assert "int8" in error_text, "Message devrait mentionner int8"
    assert "MSE" in error_text or "4681" in error_text, "Message devrait mentionner l'impact MSE"

    print("✅ Détection Bug #3: HV int8 correctement détecté")
    print("   Erreurs détectées:")
    for error in result["errors"]:
        print(f"     • {error}")


def test_resize_targets():
    """Test 4: Resize des targets."""
    print("\n" + "=" * 80)
    print("TEST 4: Resize Targets 256 → 224")
    print("=" * 80)

    # Créer des targets 256×256
    np_target = np.random.rand(256, 256).astype(np.float32)
    hv_target = (np.random.rand(2, 256, 256) * 2 - 1).astype(np.float32)
    nt_target = np.random.randint(0, 5, (256, 256), dtype=np.int64)

    # Resize vers 224×224
    np_resized, hv_resized, nt_resized = resize_targets(
        np_target, hv_target, nt_target, target_size=224
    )

    # Vérifier les shapes
    assert np_resized.shape == (224, 224), f"NP shape invalide: {np_resized.shape}"
    assert hv_resized.shape == (2, 224, 224), f"HV shape invalide: {hv_resized.shape}"
    assert nt_resized.shape == (224, 224), f"NT shape invalide: {nt_resized.shape}"

    # Vérifier les dtypes
    assert np_resized.dtype == np.float32, "NP dtype devrait être float32"
    assert hv_resized.dtype == np.float32, "HV dtype devrait être float32"
    assert nt_resized.dtype == np.int64, "NT dtype devrait être int64"

    # Vérifier les ranges
    assert np_resized.min() >= 0.0 and np_resized.max() <= 1.0, "NP range invalide"
    assert hv_resized.min() >= -1.1 and hv_resized.max() <= 1.1, "HV range invalide (tolérance 0.1)"
    assert nt_resized.min() >= 0 and nt_resized.max() <= 4, "NT range invalide"

    print("✅ Resize: Toutes les targets correctement redimensionnées")
    print(f"   NP: {np_resized.shape}, dtype={np_resized.dtype}")
    print(f"   HV: {hv_resized.shape}, dtype={hv_resized.dtype}")
    print(f"   NT: {nt_resized.shape}, dtype={nt_resized.dtype}")


def test_prepare_batch():
    """Test 5: Préparation d'un batch pour entraînement."""
    print("\n" + "=" * 80)
    print("TEST 5: Préparation Batch (DataLoader)")
    print("=" * 80)

    # Créer des targets simulés (N=10, 256×256)
    N = 10
    np_targets = np.random.rand(N, 256, 256).astype(np.float32)
    hv_targets = (np.random.rand(N, 2, 256, 256) * 2 - 1).astype(np.float32)
    nt_targets = np.random.randint(0, 5, (N, 256, 256), dtype=np.int64)

    # Sélectionner un batch de 4 échantillons
    indices = np.array([0, 3, 5, 8])

    np_batch, hv_batch, nt_batch = prepare_batch_for_training(
        np_targets, hv_targets, nt_targets, indices
    )

    # Vérifier les shapes du batch
    assert np_batch.shape == (4, 224, 224), f"NP batch shape invalide: {np_batch.shape}"
    assert hv_batch.shape == (4, 2, 224, 224), f"HV batch shape invalide: {hv_batch.shape}"
    assert nt_batch.shape == (4, 224, 224), f"NT batch shape invalide: {nt_batch.shape}"

    print("✅ Batch Preparation: Batch correctement formé")
    print(f"   Indices: {indices}")
    print(f"   NP batch: {np_batch.shape}")
    print(f"   HV batch: {hv_batch.shape}")
    print(f"   NT batch: {nt_batch.shape}")


def run_all_tests():
    """Exécute tous les tests de validation."""
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║         VALIDATION DU MODULE CENTRALISÉ DE PREPROCESSING                ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    try:
        test_target_format()
        test_validate_targets_correct()
        test_validate_targets_int8_bug()
        test_resize_targets()
        test_prepare_batch()

        print("\n" + "=" * 80)
        print("✅ TOUS LES TESTS PASSENT")
        print("=" * 80)
        print("\nLe module src.data.preprocessing est prêt à être utilisé.")
        print("\nPROCHAINES ÉTAPES:")
        print("  1. Régénérer les données avec les targets float32 corrects")
        print("  2. Ré-entraîner les 5 familles HoVer-Net")
        print("  3. Valider les performances (Dice ~0.96, HV MSE ~0.01)")

        return True

    except AssertionError as e:
        print(f"\n❌ ÉCHEC: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
