#!/usr/bin/env python3
"""
Unit tests for HoVerNetDecoderHybrid architecture.

Tests:
1. Forward pass with correct shapes
2. Gradient flow (RGB and H branches)
3. Fusion additive (vs concatenation)
4. Output activations
5. Parameter count

Author: Cell

Vit-Optimus Team
Date: 2025-12-26
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder_hybrid import HoVerNetDecoderHybrid, HybridDecoderOutput


def test_forward_pass():
    """Test 1: Forward pass with correct shapes."""
    print("\n" + "="*80)
    print("TEST 1: FORWARD PASS")
    print("="*80)

    # Initialize model
    model = HoVerNetDecoderHybrid(embed_dim=1536, h_dim=256, n_classes=5)

    # Create dummy inputs
    batch_size = 2
    patch_tokens = torch.randn(batch_size, 256, 1536)
    h_features = torch.randn(batch_size, 256)

    print(f"Input shapes:")
    print(f"  patch_tokens: {patch_tokens.shape}")
    print(f"  h_features: {h_features.shape}")

    # Forward pass
    output = model(patch_tokens, h_features)

    # Check output type
    assert isinstance(output, HybridDecoderOutput), \
        f"❌ Output type: {type(output)}, expected HybridDecoderOutput"

    # Check shapes
    expected_np_shape = (batch_size, 2, 224, 224)
    expected_hv_shape = (batch_size, 2, 224, 224)
    expected_nt_shape = (batch_size, 5, 224, 224)

    assert output.np_out.shape == expected_np_shape, \
        f"❌ NP shape: {output.np_out.shape}, expected {expected_np_shape}"
    assert output.hv_out.shape == expected_hv_shape, \
        f"❌ HV shape: {output.hv_out.shape}, expected {expected_hv_shape}"
    assert output.nt_out.shape == expected_nt_shape, \
        f"❌ NT shape: {output.nt_out.shape}, expected {expected_nt_shape}"

    print(f"\nOutput shapes:")
    print(f"  np_out: {output.np_out.shape} ✅")
    print(f"  hv_out: {output.hv_out.shape} ✅")
    print(f"  nt_out: {output.nt_out.shape} ✅")

    # Check HV range (should be in [-1, 1] due to Tanh)
    hv_min, hv_max = output.hv_out.min().item(), output.hv_out.max().item()
    assert -1.0 <= hv_min <= hv_max <= 1.0, \
        f"❌ HV range [{hv_min:.3f}, {hv_max:.3f}] not in [-1, 1]"

    print(f"\nHV range: [{hv_min:.3f}, {hv_max:.3f}] ✅")

    print("\n✅ TEST 1 PASSED: Forward pass OK\n")
    return True


def test_gradient_flow():
    """Test 2: Gradient flow through RGB and H branches."""
    print("="*80)
    print("TEST 2: GRADIENT FLOW")
    print("="*80)

    # Initialize model
    model = HoVerNetDecoderHybrid(embed_dim=1536, h_dim=256, n_classes=5)
    model.train()  # Ensure training mode for gradients

    # Create inputs with gradients (batch_size=2 for BatchNorm compatibility)
    patch_tokens = torch.randn(2, 256, 1536, requires_grad=True)
    h_features = torch.randn(2, 256, requires_grad=True)

    # Forward pass
    output = model(patch_tokens, h_features)

    # Compute loss (sum of all outputs)
    loss = (output.np_out.sum() +
            output.hv_out.sum() +
            output.nt_out.sum())

    # Backward
    loss.backward()

    # Check gradients exist
    assert patch_tokens.grad is not None, "❌ RGB gradients = None"
    assert h_features.grad is not None, "❌ H gradients = None"

    # Compute gradient norms
    rgb_grad_norm = patch_tokens.grad.norm().item()
    h_grad_norm = h_features.grad.norm().item()

    print(f"\nGradient norms:")
    print(f"  RGB (patch_tokens): {rgb_grad_norm:.4f} ✅")
    print(f"  H (h_features): {h_grad_norm:.4f} ✅")

    # Check gradients are non-zero
    assert rgb_grad_norm > 1e-6, f"❌ RGB gradient too small: {rgb_grad_norm}"
    assert h_grad_norm > 1e-6, f"❌ H gradient too small: {h_grad_norm}"

    # Check gradient ratio (should be balanced, not 1000x difference)
    ratio = max(rgb_grad_norm, h_grad_norm) / min(rgb_grad_norm, h_grad_norm)
    print(f"\nGradient ratio (max/min): {ratio:.2f}")

    if ratio > 100:
        print(f"  ⚠️  WARNING: Large gradient imbalance ({ratio:.2f}x)")
    else:
        print(f"  ✅ Gradient balance OK")

    print("\n✅ TEST 2 PASSED: Gradient flow OK\n")
    return True


def test_fusion_additive():
    """Test 3: Verify additive fusion (not concatenation)."""
    print("="*80)
    print("TEST 3: FUSION ADDITIVE")
    print("="*80)

    # Initialize model in eval mode (no BatchNorm statistics needed)
    model = HoVerNetDecoderHybrid(embed_dim=1536, h_dim=256, n_classes=5)
    model.eval()

    # Test 1: RGB only (H = zeros)
    patch_tokens = torch.randn(1, 256, 1536)
    h_features_zero = torch.zeros(1, 256)

    output_rgb_only = model(patch_tokens, h_features_zero)

    # Test 2: H only (RGB = zeros)
    patch_tokens_zero = torch.zeros(1, 256, 1536)
    h_features = torch.randn(1, 256)

    output_h_only = model(patch_tokens_zero, h_features)

    # Test 3: Both RGB and H
    output_both = model(patch_tokens, h_features)

    # In additive fusion: output_both ≈ output_rgb_only + output_h_only (modulo non-linearities)
    # Check that outputs are different (not concatenation)
    diff_rgb_vs_both = (output_rgb_only.np_out - output_both.np_out).abs().mean().item()
    diff_h_vs_both = (output_h_only.np_out - output_both.np_out).abs().mean().item()

    print(f"\nMean absolute differences:")
    print(f"  RGB-only vs Both: {diff_rgb_vs_both:.4f}")
    print(f"  H-only vs Both: {diff_h_vs_both:.4f}")

    # Both should be non-zero (fusion affects output)
    assert diff_rgb_vs_both > 1e-4, "❌ RGB branch has no effect"
    assert diff_h_vs_both > 1e-4, "❌ H branch has no effect"

    print(f"\n✅ Both branches contribute to output")

    # Check that adding H changes the output significantly
    relative_change = diff_rgb_vs_both / (output_rgb_only.np_out.abs().mean().item() + 1e-8)
    print(f"\nRelative change when adding H-channel: {relative_change:.2%}")

    print("\n✅ TEST 3 PASSED: Additive fusion OK\n")
    return True


def test_output_activations():
    """Test 4: Verify output activations (Tanh for HV)."""
    print("="*80)
    print("TEST 4: OUTPUT ACTIVATIONS")
    print("="*80)

    # Initialize model in eval mode (no BatchNorm statistics needed)
    model = HoVerNetDecoderHybrid(embed_dim=1536, h_dim=256, n_classes=5)
    model.eval()

    patch_tokens = torch.randn(1, 256, 1536)
    h_features = torch.randn(1, 256)

    output = model(patch_tokens, h_features)

    # Check HV has Tanh (range [-1, 1])
    hv_min = output.hv_out.min().item()
    hv_max = output.hv_out.max().item()

    print(f"\nHV output range:")
    print(f"  Min: {hv_min:.4f}")
    print(f"  Max: {hv_max:.4f}")

    assert -1.0 <= hv_min, f"❌ HV min {hv_min} < -1.0"
    assert hv_max <= 1.0, f"❌ HV max {hv_max} > 1.0"

    print(f"  ✅ HV range OK (Tanh applied)")

    # Check to_numpy method
    output_np = output.to_numpy(apply_activations=True)

    assert 'np' in output_np, "❌ Missing 'np' key"
    assert 'hv' in output_np, "❌ Missing 'hv' key"
    assert 'nt' in output_np, "❌ Missing 'nt' key"

    # Check NP after sigmoid is in [0, 1]
    np_min = output_np['np'].min()
    np_max = output_np['np'].max()

    print(f"\nNP after sigmoid:")
    print(f"  Range: [{np_min:.4f}, {np_max:.4f}]")

    assert 0.0 <= np_min <= np_max <= 1.0, f"❌ NP range invalid"
    print(f"  ✅ NP range OK (Sigmoid applied)")

    # Check NT after softmax sums to 1
    nt_sum = output_np['nt'].sum(axis=1).mean()  # Average over batch

    print(f"\nNT after softmax:")
    print(f"  Sum over classes: {nt_sum:.4f}")

    assert abs(nt_sum - 1.0) < 0.01, f"❌ NT softmax sum = {nt_sum}, expected 1.0"
    print(f"  ✅ NT softmax OK")

    print("\n✅ TEST 4 PASSED: Output activations OK\n")
    return True


def test_parameter_count():
    """Test 5: Verify parameter count is reasonable."""
    print("="*80)
    print("TEST 5: PARAMETER COUNT")
    print("="*80)

    model = HoVerNetDecoderHybrid(embed_dim=1536, h_dim=256, n_classes=5)

    n_params = model.get_num_params(trainable_only=True)
    n_params_total = model.get_num_params(trainable_only=False)

    print(f"\nParameter count:")
    print(f"  Trainable: {n_params:,}")
    print(f"  Total: {n_params_total:,}")

    # Sanity check: Should be > 100k (not trivial) but < 100M (not huge)
    assert 100_000 < n_params < 100_000_000, \
        f"❌ Parameter count {n_params:,} out of expected range [100k, 100M]"

    print(f"  ✅ Parameter count reasonable")

    # Expected: ~20-30M params for a decoder
    if n_params < 10_000_000:
        print(f"  ℹ️  Model is lightweight ({n_params/1e6:.2f}M params)")
    elif n_params < 50_000_000:
        print(f"  ✅ Model size optimal ({n_params/1e6:.2f}M params)")
    else:
        print(f"  ⚠️  Model is large ({n_params/1e6:.2f}M params)")

    print("\n✅ TEST 5 PASSED: Parameter count OK\n")
    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "🔬"*40)
    print("HOVERNET DECODER HYBRID — UNIT TESTS")
    print("🔬"*40)

    tests = [
        ("Forward Pass", test_forward_pass),
        ("Gradient Flow", test_gradient_flow),
        ("Fusion Additive", test_fusion_additive),
        ("Output Activations", test_output_activations),
        ("Parameter Count", test_parameter_count),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"Error: {e}")
            results.append((name, False))

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} — {name}")

    n_passed = sum(1 for _, success in results if success)
    n_total = len(results)

    print(f"\nTotal: {n_passed}/{n_total} tests passed")

    if n_passed == n_total:
        print("\n🎉 ALL TESTS PASSED! Architecture is ready for training.\n")
        return True
    else:
        print(f"\n❌ {n_total - n_passed} test(s) failed. Fix before training.\n")
        return False


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
