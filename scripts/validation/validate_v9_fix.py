#!/usr/bin/env python3
"""
Validation rapide du FIX v9: NUCLEI ONLY (exclut tissu).

Vérifie que:
1. NP coverage ~10-15% (pas 86%)
2. Channel 5 est exclu
3. Channel 0 est utilisé
4. Instances séparées correctement

Usage:
    python scripts/validation/validate_v9_fix.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    print("=" * 80)
    print("🔍 VALIDATION FIX v9: NUCLEI ONLY")
    print("=" * 80)

    # Load PanNuke test sample
    pannuke_dir = Path("/home/amar/data/PanNuke")
    if not pannuke_dir.exists():
        pannuke_dir = Path("data/PanNuke")

    masks_path = pannuke_dir / "fold2" / "masks.npy"
    if not masks_path.exists():
        print(f"❌ PanNuke non trouvé: {masks_path}")
        return

    masks = np.load(masks_path, mmap_mode='r')
    mask = masks[0]  # Premier échantillon

    print(f"\n📦 Test sample loaded: {mask.shape}")

    # ============================================================================
    # TEST 1: NP Coverage (AVANT vs APRÈS)
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 1: NP Coverage")
    print("=" * 80)

    # v8 (BUG): Inclut Channel 5
    np_v8 = mask[:, :, 1:].sum(axis=-1) > 0
    coverage_v8 = np_v8.mean() * 100

    # v9 (FIX): Exclut Channel 5
    np_v9 = mask[:, :, :5].sum(axis=-1) > 0
    coverage_v9 = np_v9.mean() * 100

    print(f"\n❌ v8 (BUG):  NP coverage = {coverage_v8:.2f}% (inclut tissu)")
    print(f"✅ v9 (FIX):  NP coverage = {coverage_v9:.2f}% (noyaux uniquement)")

    if coverage_v9 < 20:
        print(f"\n✅ PASS: Coverage v9 < 20% (noyaux seulement)")
    else:
        print(f"\n❌ FAIL: Coverage v9 > 20% (tissu encore inclus?)")

    # ============================================================================
    # TEST 2: Nombre d'Instances (AVANT vs APRÈS)
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Nombre d'Instances")
    print("=" * 80)

    # v8: connectedComponents sur union avec tissu
    inst_v8, n_v8 = label(np_v8)

    # v9: Channel 0 direct (IDs natifs)
    channel_0 = mask[:, :, 0]
    inst_ids_0 = np.unique(channel_0)
    inst_ids_0 = inst_ids_0[inst_ids_0 > 0]
    n_v9_channel0 = len(inst_ids_0)

    # v9: connectedComponents sur union sans tissu (fallback)
    inst_v9, n_v9_fallback = label(np_v9)

    print(f"\n❌ v8 (BUG):        {n_v8} instances (tissue fusionné)")
    print(f"✅ v9 Channel 0:   {n_v9_channel0} instances (IDs natifs)")
    print(f"   v9 Fallback:    {n_v9_fallback} instances (connectedComponents)")

    if n_v9_channel0 >= 5:
        print(f"\n✅ PASS: v9 détecte {n_v9_channel0} instances séparées")
    else:
        print(f"\n⚠️  WARNING: v9 détecte seulement {n_v9_channel0} instances (attendu >5)")

    # ============================================================================
    # TEST 3: Channel 5 Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Channel 5 (Tissue) Analysis")
    print("=" * 80)

    channel_5 = mask[:, :, 5]
    channel_5_pixels = (channel_5 > 0).sum()
    channel_5_coverage = (channel_5 > 0).mean() * 100
    channel_5_max = channel_5.max()

    print(f"\nChannel 5 (Epithelial/Tissue):")
    print(f"  Pixels: {channel_5_pixels}")
    print(f"  Coverage: {channel_5_coverage:.2f}%")
    print(f"  Max value: {channel_5_max}")

    if channel_5_max == 1:
        print(f"\n✅ CONFIRMÉ: Channel 5 est un MASQUE BINAIRE (tissue)")
    else:
        print(f"\n⚠️  Channel 5 a des IDs multiples (max={channel_5_max})")

    # ============================================================================
    # TEST 4: Channel 0 Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Channel 0 (Multi-type Instances) Analysis")
    print("=" * 80)

    channel_0 = mask[:, :, 0]
    channel_0_pixels = (channel_0 > 0).sum()
    channel_0_coverage = (channel_0 > 0).mean() * 100
    channel_0_unique = np.unique(channel_0)
    channel_0_unique = channel_0_unique[channel_0_unique > 0]

    print(f"\nChannel 0 (Multi-type instances):")
    print(f"  Pixels: {channel_0_pixels}")
    print(f"  Coverage: {channel_0_coverage:.2f}%")
    print(f"  Unique IDs: {len(channel_0_unique)}")
    print(f"  IDs: {channel_0_unique[:10].tolist()}...")

    if len(channel_0_unique) >= 5:
        print(f"\n✅ CONFIRMÉ: Channel 0 contient {len(channel_0_unique)} instances séparées")
    else:
        print(f"\n⚠️  Channel 0 a seulement {len(channel_0_unique)} instances")

    # ============================================================================
    # TEST 5: Comparison Matrix
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Comparison Matrix")
    print("=" * 80)

    print(f"\n{'Métrique':<30} {'v8 (BUG)':<20} {'v9 (FIX)':<20}")
    print("-" * 70)
    print(f"{'NP Coverage':<30} {coverage_v8:>6.2f}%{'':<13} {coverage_v9:>6.2f}%{'':<13}")
    print(f"{'Instances (connectedComp)':<30} {n_v8:>6}{'':<14} {n_v9_fallback:>6}{'':<14}")
    print(f"{'Instances (Channel 0)':<30} {'N/A':<20} {n_v9_channel0:>6}{'':<14}")
    print(f"{'Includes Channel 5?':<30} {'YES ❌':<20} {'NO ✅':<20}")

    # ============================================================================
    # VERDICT FINAL
    # ============================================================================
    print("\n" + "=" * 80)
    print("🎯 VERDICT FINAL")
    print("=" * 80)

    all_pass = True

    # Check 1: Coverage réduit
    if coverage_v9 < coverage_v8 * 0.3:  # Au moins 70% de réduction
        print(f"\n✅ CHECK 1: NP coverage réduit de {coverage_v8:.1f}% → {coverage_v9:.1f}% (-{(coverage_v8-coverage_v9)/coverage_v8*100:.0f}%)")
    else:
        print(f"\n❌ CHECK 1: NP coverage pas assez réduit ({coverage_v8:.1f}% → {coverage_v9:.1f}%)")
        all_pass = False

    # Check 2: Plus d'instances avec v9
    if n_v9_channel0 >= n_v8:
        print(f"✅ CHECK 2: Plus d'instances avec v9 ({n_v8} → {n_v9_channel0})")
    else:
        print(f"❌ CHECK 2: Moins d'instances avec v9 ({n_v8} → {n_v9_channel0})")
        all_pass = False

    # Check 3: Channel 5 est binaire
    if channel_5_max <= 1:
        print(f"✅ CHECK 3: Channel 5 est binaire (tissue mask)")
    else:
        print(f"❌ CHECK 3: Channel 5 n'est pas binaire")
        all_pass = False

    # Check 4: Channel 0 a des instances
    if n_v9_channel0 >= 5:
        print(f"✅ CHECK 4: Channel 0 contient {n_v9_channel0} instances séparées")
    else:
        print(f"⚠️  CHECK 4: Channel 0 a seulement {n_v9_channel0} instances")

    if all_pass:
        print("\n" + "=" * 80)
        print("✅ TOUS LES TESTS PASSENT - FIX v9 VALIDÉ")
        print("=" * 80)
        print("\nProchaine étape:")
        print("  python scripts/preprocessing/prepare_family_data_FIXED_v9_NUCLEI_ONLY.py --family epidermal")
    else:
        print("\n" + "=" * 80)
        print("❌ CERTAINS TESTS ÉCHOUENT - VÉRIFIER FIX v9")
        print("=" * 80)


if __name__ == "__main__":
    main()
