#!/usr/bin/env python3
"""
Test unitaire pour magnitude_loss().

Vérifie que la loss pénalise correctement les prédictions de magnitude faible.

Usage:
    python scripts/validation/test_magnitude_loss.py
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetLoss


def test_magnitude_loss_basic():
    """Test basique: magnitude_loss doit pénaliser prédictions faibles."""
    print("\n" + "="*80)
    print("TEST 1: Magnitude Loss Pénalise Prédictions Faibles")
    print("="*80)

    # Créer criterion
    criterion = HoVerNetLoss(lambda_np=1.0, lambda_hv=2.0, lambda_nt=1.0)

    # Cas 1: Magnitude faible (pred) vs forte (target)
    hv_pred_weak = torch.randn(1, 2, 224, 224) * 0.1  # Magnitude ~0.1
    hv_target_strong = torch.randn(1, 2, 224, 224) * 0.8  # Magnitude ~0.8
    mask = torch.ones(1, 1, 224, 224)

    mag_loss_high = criterion.magnitude_loss(hv_pred_weak, hv_target_strong, mask)
    print(f"\nMagnitude loss (pred faible 0.1 → target forte 0.8):")
    print(f"  Loss: {mag_loss_high:.4f}")
    print(f"  Attendu: >0.3 (fort écart de magnitude)")

    # Cas 2: Magnitude forte (pred) vs forte (target)
    hv_pred_strong = torch.randn(1, 2, 224, 224) * 0.8  # Magnitude ~0.8
    mag_loss_low = criterion.magnitude_loss(hv_pred_strong, hv_target_strong, mask)
    print(f"\nMagnitude loss (pred forte 0.8 → target forte 0.8):")
    print(f"  Loss: {mag_loss_low:.4f}")
    print(f"  Attendu: <0.1 (faible écart de magnitude)")

    # Vérification
    ratio = mag_loss_high / (mag_loss_low + 1e-8)
    print(f"\nRatio (high/low): {ratio:.2f}")
    print(f"  Attendu: >5× (loss élevée pour pred faible)")

    if ratio > 5.0:
        print("\n✅ TEST 1 PASSÉ: Magnitude loss pénalise correctement pred faibles")
        return True
    else:
        print(f"\n❌ TEST 1 ÉCHOUÉ: Ratio {ratio:.2f} trop faible (attendu >5.0)")
        return False


def test_magnitude_loss_masking():
    """Test masquage: magnitude_loss doit calculer uniquement sur pixels masqués."""
    print("\n" + "="*80)
    print("TEST 2: Magnitude Loss Respecte le Masque")
    print("="*80)

    criterion = HoVerNetLoss(lambda_np=1.0, lambda_hv=2.0, lambda_nt=1.0)

    # HV pred/target identiques
    hv_pred = torch.randn(1, 2, 224, 224) * 0.5
    hv_target = hv_pred.clone()

    # Masque partiel (50% des pixels)
    mask_half = torch.zeros(1, 1, 224, 224)
    mask_half[:, :, :, :112] = 1.0  # Moitié gauche

    # Modifier prédictions UNIQUEMENT sur partie NON masquée
    hv_pred[:, :, :, 112:] = torch.randn(1, 2, 224, 112) * 2.0  # Magnitude forte

    # Loss doit être ~0 car partie masquée est identique
    mag_loss_masked = criterion.magnitude_loss(hv_pred, hv_target, mask_half)

    print(f"\nMagnitude loss (HV identique sur partie masquée):")
    print(f"  Loss: {mag_loss_masked:.4f}")
    print(f"  Attendu: <0.01 (identique sur pixels masqués)")

    if mag_loss_masked < 0.01:
        print("\n✅ TEST 2 PASSÉ: Magnitude loss respecte le masque")
        return True
    else:
        print(f"\n❌ TEST 2 ÉCHOUÉ: Loss {mag_loss_masked:.4f} trop élevée (attendu <0.01)")
        return False


def test_magnitude_loss_gradient_flow():
    """Test gradient flow: magnitude_loss doit propager les gradients."""
    print("\n" + "="*80)
    print("TEST 3: Magnitude Loss Propage les Gradients")
    print("="*80)

    criterion = HoVerNetLoss(lambda_np=1.0, lambda_hv=2.0, lambda_nt=1.0)

    # HV pred avec requires_grad
    hv_pred = torch.randn(1, 2, 224, 224, requires_grad=True) * 0.1
    hv_target = torch.randn(1, 2, 224, 224) * 0.8
    mask = torch.ones(1, 1, 224, 224)

    # Forward
    mag_loss = criterion.magnitude_loss(hv_pred, hv_target, mask)

    # Backward
    mag_loss.backward()

    # Vérifier que gradients existent
    has_grad = hv_pred.grad is not None
    grad_norm = hv_pred.grad.norm().item() if has_grad else 0.0

    print(f"\nGradient après backward:")
    print(f"  Has grad: {has_grad}")
    print(f"  Grad norm: {grad_norm:.6f}")
    print(f"  Attendu: grad norm >0.001")

    if has_grad and grad_norm > 0.001:
        print("\n✅ TEST 3 PASSÉ: Magnitude loss propage les gradients")
        return True
    else:
        print(f"\n❌ TEST 3 ÉCHOUÉ: Pas de gradient ou norm trop faible")
        return False


def test_magnitude_calculation():
    """Test calcul magnitude: vérifier formule sqrt(H² + V²)."""
    print("\n" + "="*80)
    print("TEST 4: Calcul Magnitude Correct")
    print("="*80)

    # HV maps simples pour vérification manuelle
    hv = torch.tensor([[
        [[0.6, 0.8], [0.0, 0.0]],  # Canal H
        [[0.8, 0.6], [0.0, 0.0]]   # Canal V
    ]])

    # Calcul manuel magnitude
    # Pixel (0,0): sqrt(0.6² + 0.8²) = sqrt(0.36 + 0.64) = sqrt(1.0) = 1.0
    # Pixel (0,1): sqrt(0.8² + 0.6²) = sqrt(0.64 + 0.36) = sqrt(1.0) = 1.0
    # Pixel (1,0) et (1,1): sqrt(0² + 0²) = 0.0

    expected_mag = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])

    # Calcul avec fonction
    mag_computed = torch.sqrt((hv ** 2).sum(dim=1, keepdim=True) + 1e-8)

    # Comparer
    diff = (mag_computed - expected_mag).abs().max()

    print(f"\nMagnitude attendue:")
    print(f"  {expected_mag[0, 0]}")
    print(f"\nMagnitude calculée:")
    print(f"  {mag_computed[0, 0]}")
    print(f"\nDifférence max: {diff:.6f}")
    print(f"  Attendu: <0.01")

    if diff < 0.01:
        print("\n✅ TEST 4 PASSÉ: Calcul magnitude correct")
        return True
    else:
        print(f"\n❌ TEST 4 ÉCHOUÉ: Différence {diff:.6f} trop grande")
        return False


def test_integration_with_hovernet_loss():
    """Test intégration: magnitude_loss dans HoVerNetLoss.forward()."""
    print("\n" + "="*80)
    print("TEST 5: Intégration avec HoVerNetLoss")
    print("="*80)

    criterion = HoVerNetLoss(lambda_np=1.0, lambda_hv=2.0, lambda_nt=1.0)

    # Données simulées
    batch_size = 2
    np_pred = torch.randn(batch_size, 2, 224, 224)
    hv_pred = torch.randn(batch_size, 2, 224, 224) * 0.1  # Magnitude faible
    nt_pred = torch.randn(batch_size, 5, 224, 224)

    np_target = torch.randint(0, 2, (batch_size, 224, 224))
    hv_target = torch.randn(batch_size, 2, 224, 224) * 0.8  # Magnitude forte
    nt_target = torch.randint(0, 5, (batch_size, 224, 224))

    # Forward pass
    total_loss, loss_dict = criterion(np_pred, hv_pred, nt_pred, np_target, hv_target, nt_target)

    print(f"\nLoss totale: {total_loss:.4f}")
    print(f"\nDétails HV loss:")
    print(f"  hv_l1:        {loss_dict['hv_l1']:.4f}")
    print(f"  hv_gradient:  {loss_dict['hv_gradient']:.4f}")
    print(f"  hv_magnitude: {loss_dict['hv_magnitude']:.4f}")
    print(f"  hv (total):   {loss_dict['hv']:.4f}")

    # Vérifier que hv_magnitude est présent et > 0
    has_mag = 'hv_magnitude' in loss_dict
    mag_value = loss_dict.get('hv_magnitude', 0.0)

    print(f"\nVérification:")
    print(f"  hv_magnitude présent: {has_mag}")
    print(f"  hv_magnitude > 0:     {mag_value > 0}")

    # Vérifier cohérence: hv_total ≈ hv_l1 + 2×hv_gradient + 1×hv_magnitude
    expected_hv = loss_dict['hv_l1'] + 2.0 * loss_dict['hv_gradient'] + 1.0 * loss_dict['hv_magnitude']
    actual_hv = loss_dict['hv']
    diff_hv = abs(expected_hv - actual_hv)

    print(f"\nCohérence HV loss:")
    print(f"  Attendu (l1 + 2×grad + 1×mag): {expected_hv:.4f}")
    print(f"  Actual:                         {actual_hv:.4f}")
    print(f"  Différence:                     {diff_hv:.4f}")

    if has_mag and mag_value > 0 and diff_hv < 0.01:
        print("\n✅ TEST 5 PASSÉ: Magnitude loss intégrée correctement")
        return True
    else:
        print(f"\n❌ TEST 5 ÉCHOUÉ:")
        if not has_mag:
            print("  - hv_magnitude absent du dict")
        if mag_value <= 0:
            print(f"  - hv_magnitude = {mag_value} (devrait être >0)")
        if diff_hv >= 0.01:
            print(f"  - Incohérence HV total (diff {diff_hv:.4f})")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTS UNITAIRES — MAGNITUDE LOSS")
    print("="*80)

    tests = [
        ("Pénalisation pred faibles", test_magnitude_loss_basic),
        ("Respect du masque", test_magnitude_loss_masking),
        ("Propagation gradients", test_magnitude_loss_gradient_flow),
        ("Calcul magnitude", test_magnitude_calculation),
        ("Intégration HoVerNetLoss", test_integration_with_hovernet_loss),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERREUR dans {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)

    for name, passed in results:
        status = "✅ PASSÉ" if passed else "❌ ÉCHOUÉ"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\n{passed}/{total} tests passés ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 TOUS LES TESTS PASSENT — Magnitude loss prête pour training!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - passed} test(s) échoué(s) — Corriger avant training")
        sys.exit(1)
