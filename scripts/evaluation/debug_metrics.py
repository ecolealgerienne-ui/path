#!/usr/bin/env python3
"""
Debug script pour diagnostiquer le problème de métriques.
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader

def main():
    # Charger 1 échantillon
    data_dir = Path("data/cache/family_data_FIXED")

    features_data = np.load(data_dir / "glandular_features.npz")
    targets_data = np.load(data_dir / "glandular_targets.npz")

    features = features_data['features']
    np_targets = targets_data['np_targets']
    hv_targets = targets_data['hv_targets']
    nt_targets = targets_data['nt_targets']

    print("=" * 80)
    print("SHAPES ET RANGES")
    print("=" * 80)

    print(f"\nFeatures: {features.shape} ({features.dtype})")
    print(f"NP targets: {np_targets.shape} ({np_targets.dtype}) [{np_targets.min():.3f}, {np_targets.max():.3f}]")
    print(f"HV targets: {hv_targets.shape} ({hv_targets.dtype}) [{hv_targets.min():.3f}, {hv_targets.max():.3f}]")
    print(f"NT targets: {nt_targets.shape} ({nt_targets.dtype}) [{nt_targets.min()}, {nt_targets.max()}]")

    # Charger modèle
    print("\n" + "=" * 80)
    print("CHARGEMENT MODÈLE")
    print("=" * 80)

    checkpoint_path = "models/checkpoints/hovernet_glandular_best.pth"
    hovernet = ModelLoader.load_hovernet(checkpoint_path, device="cuda")
    hovernet.eval()
    print(f"✅ Modèle chargé: {checkpoint_path}")

    # Test sur 1er échantillon
    print("\n" + "=" * 80)
    print("TEST ÉCHANTILLON 0")
    print("=" * 80)

    feat = features[0]  # (261, 1536)
    np_target_256 = np_targets[0]  # (256, 256)
    hv_target_256 = hv_targets[0]  # (2, 256, 256)
    nt_target_256 = nt_targets[0]  # (256, 256)

    print(f"\nTargets 256×256:")
    print(f"  NP: {np_target_256.shape} [{np_target_256.min():.3f}, {np_target_256.max():.3f}]")
    print(f"  HV: {hv_target_256.shape} [{hv_target_256.min():.3f}, {hv_target_256.max():.3f}]")
    print(f"  NT: {nt_target_256.shape} [{nt_target_256.min()}, {nt_target_256.max()}]")

    # Resize 256→224
    import torch.nn.functional as F

    np_target_t = torch.from_numpy(np_target_256).float().unsqueeze(0).unsqueeze(0)
    hv_target_t = torch.from_numpy(hv_target_256).float().unsqueeze(0)
    nt_target_t = torch.from_numpy(nt_target_256).float().unsqueeze(0).unsqueeze(0)

    np_target_t = F.interpolate(np_target_t, size=(224, 224), mode='nearest').squeeze()
    hv_target_t = F.interpolate(hv_target_t, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    nt_target_t = F.interpolate(nt_target_t, size=(224, 224), mode='nearest').squeeze().long()

    np_target_224 = np_target_t.numpy()
    hv_target_224 = hv_target_t.numpy()
    nt_target_224 = nt_target_t.numpy()

    print(f"\nTargets 224×224 (après resize):")
    print(f"  NP: {np_target_224.shape} [{np_target_224.min():.3f}, {np_target_224.max():.3f}]")
    print(f"  HV: {hv_target_224.shape} [{hv_target_224.min():.3f}, {hv_target_224.max():.3f}]")
    print(f"  NT: {nt_target_224.shape} [{nt_target_224.min()}, {nt_target_224.max()}]")

    # Inférence
    feat_tensor = torch.from_numpy(feat).unsqueeze(0).cuda()
    patch_tokens = feat_tensor[:, 1:257, :]

    with torch.no_grad():
        np_out, hv_out, nt_out = hovernet(patch_tokens)

    print(f"\nSorties modèle (AVANT activations):")
    print(f"  NP: {np_out.shape} [{np_out.min().item():.3f}, {np_out.max().item():.3f}]")
    print(f"  HV: {hv_out.shape} [{hv_out.min().item():.3f}, {hv_out.max().item():.3f}]")
    print(f"  NT: {nt_out.shape} [{nt_out.min().item():.3f}, {nt_out.max().item():.3f}]")

    # Activations
    np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]
    hv_pred = hv_out.cpu().numpy()[0]
    nt_pred = torch.softmax(nt_out, dim=1).cpu().numpy()[0]

    print(f"\nPrédictions (APRÈS activations):")
    print(f"  NP: {np_pred.shape} [{np_pred.min():.3f}, {np_pred.max():.3f}]")
    print(f"  HV: {hv_pred.shape} [{hv_pred.min():.3f}, {hv_pred.max():.3f}]")
    print(f"  NT: {nt_pred.shape} [{nt_pred.min():.3f}, {nt_pred.max():.3f}]")

    # Statistiques NP
    print(f"\nNP Coverage:")
    print(f"  Target > 0: {(np_target_224 > 0).sum()} pixels ({(np_target_224 > 0).mean() * 100:.1f}%)")
    print(f"  Pred > 0.5: {(np_pred > 0.5).sum()} pixels ({(np_pred > 0.5).mean() * 100:.1f}%)")

    # Dice NP
    pred_binary = np_pred > 0.5
    target_binary = np_target_224 > 0
    intersection = (pred_binary & target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    dice = 2 * intersection / (union + 1e-8)

    print(f"\nNP Dice:")
    print(f"  Intersection: {intersection}")
    print(f"  Union: {union}")
    print(f"  Dice: {dice:.4f}")

    # HV MSE - CORRECT vs BUGGY
    mask = np_target_224 > 0
    print(f"\nHV MSE Comparison:")
    print(f"  Mask pixels: {mask.sum()} ({mask.mean() * 100:.1f}%)")

    # Version BUGGY (actuelle)
    try:
        buggy_mse = ((hv_pred[:, mask] - hv_target_224[:, mask]) ** 2).mean()
        print(f"  BUGGY indexing: {buggy_mse:.4f}")
    except Exception as e:
        print(f"  BUGGY indexing: ERROR - {e}")

    # Version CORRECTE
    mask_flat = mask.flatten()
    hv_pred_flat = hv_pred.reshape(2, -1)
    hv_target_flat = hv_target_224.reshape(2, -1)
    correct_mse = ((hv_pred_flat[:, mask_flat] - hv_target_flat[:, mask_flat]) ** 2).mean()
    print(f"  CORRECT indexing: {correct_mse:.4f}")

    # NT Accuracy
    pred_class = nt_pred.argmax(axis=0)
    nt_acc = (pred_class[mask] == nt_target_224[mask]).mean()
    print(f"\nNT Accuracy: {nt_acc:.4f}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC")
    print("=" * 80)

    if dice < 0.1:
        print("\n❌ NP Dice TRÈS BAS - Problèmes possibles:")
        print("   1. Prédictions toutes à 0 ou toutes à 1")
        print("   2. Targets corrompus")
        print("   3. Mauvais checkpoint chargé")

    if correct_mse > 0.1:
        print("\n❌ HV MSE ÉLEVÉ - Problèmes possibles:")
        print("   1. HV predictions non normalisées [-1, 1]")
        print("   2. HV targets corrompus")

    if nt_acc > 0.8:
        print("\n✅ NT Accuracy OK - Le modèle fonctionne!")


if __name__ == "__main__":
    main()
