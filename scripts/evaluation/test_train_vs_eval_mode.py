#!/usr/bin/env python3
"""
Compare les sorties HV en mode train() vs eval() pour diagnostiquer
pourquoi HV MSE = 0.0001 pendant training mais 0.32 pendant eval.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader

def compute_hv_mse_like_training(hv_pred, hv_target, np_target):
    """Calcul EXACT comme dans train_hovernet_family.py ligne 208-221."""
    mask = np_target.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    if mask.sum() == 0:
        return 0.0

    hv_pred_t = torch.from_numpy(hv_pred).unsqueeze(0)  # (1, 2, H, W)
    hv_target_t = torch.from_numpy(hv_target).unsqueeze(0)  # (1, 2, H, W)

    diff = (hv_pred_t - hv_target_t) ** 2
    masked_diff = diff * mask
    mse = masked_diff.sum() / (mask.sum() * 2)  # *2 car 2 canaux

    return mse.item()

def main():
    print("=" * 80)
    print("TEST TRAIN() VS EVAL() MODE")
    print("=" * 80)
    print("")

    # Charger données
    data_dir = Path("data/cache/family_data_FIXED")

    features_data = np.load(data_dir / "glandular_features.npz")
    targets_data = np.load(data_dir / "glandular_targets.npz")

    features = features_data['features']
    np_targets = targets_data['np_targets']
    hv_targets = targets_data['hv_targets']

    # Charger modèle
    checkpoint_path = "models/checkpoints/hovernet_glandular_best.pth"
    hovernet = ModelLoader.load_hovernet(checkpoint_path, device="cuda")

    # Test sur 1er échantillon
    feat = features[0]
    np_target_256 = np_targets[0]
    hv_target_256 = hv_targets[0]

    # Resize 256→224 (comme pendant training)
    import torch.nn.functional as F

    np_target_t = torch.from_numpy(np_target_256).float().unsqueeze(0).unsqueeze(0)
    hv_target_t = torch.from_numpy(hv_target_256).float().unsqueeze(0)

    np_target_t = F.interpolate(np_target_t, size=(224, 224), mode='nearest').squeeze()
    hv_target_t = F.interpolate(hv_target_t, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

    np_target_224 = np_target_t.numpy()
    hv_target_224 = hv_target_t.numpy()

    print(f"Targets 224×224:")
    print(f"  NP: {np_target_224.shape} [{np_target_224.min():.3f}, {np_target_224.max():.3f}]")
    print(f"  HV: {hv_target_224.shape} [{hv_target_224.min():.3f}, {hv_target_224.max():.3f}]")
    print("")

    # Features
    feat_tensor = torch.from_numpy(feat).unsqueeze(0).cuda()
    patch_tokens = feat_tensor[:, 1:257, :]

    # ==================== MODE EVAL ====================
    print("=" * 80)
    print("MODE EVAL() - BatchNorm utilise running_mean/running_var")
    print("=" * 80)
    print("")

    hovernet.eval()
    with torch.no_grad():
        np_out_eval, hv_out_eval, nt_out_eval = hovernet(patch_tokens)

    hv_pred_eval = hv_out_eval.cpu().numpy()[0]

    print(f"HV output (eval): {hv_pred_eval.shape} [{hv_pred_eval.min():.3f}, {hv_pred_eval.max():.3f}]")

    mse_eval = compute_hv_mse_like_training(hv_pred_eval, hv_target_224, np_target_224)
    print(f"HV MSE (eval):    {mse_eval:.6f}")
    print("")

    # ==================== MODE TRAIN ====================
    print("=" * 80)
    print("MODE TRAIN() - BatchNorm utilise statistiques du batch")
    print("=" * 80)
    print("")

    hovernet.train()
    with torch.no_grad():  # Pas de gradient mais mode train
        np_out_train, hv_out_train, nt_out_train = hovernet(patch_tokens)

    hv_pred_train = hv_out_train.cpu().numpy()[0]

    print(f"HV output (train): {hv_pred_train.shape} [{hv_pred_train.min():.3f}, {hv_pred_train.max():.3f}]")

    mse_train = compute_hv_mse_like_training(hv_pred_train, hv_target_224, np_target_224)
    print(f"HV MSE (train):    {mse_train:.6f}")
    print("")

    # ==================== DIAGNOSTIC ====================
    print("=" * 80)
    print("DIAGNOSTIC")
    print("=" * 80)
    print("")

    print(f"Différence MSE: {abs(mse_eval - mse_train):.6f}")
    print(f"Différence range HV: eval [{hv_pred_eval.min():.3f}, {hv_pred_eval.max():.3f}] vs train [{hv_pred_train.min():.3f}, {hv_pred_train.max():.3f}]")
    print("")

    if abs(mse_eval - mse_train) > 0.01:
        print("⚠️  MODE TRAIN() vs EVAL() DIFFÈRENT SIGNIFICATIVEMENT")
        print("   → BatchNorm change les sorties HV")
        print("   → Pendant training, metrics étaient calculés en mode train()")
        print("   → Maintenant, test en mode eval() → MSE différent")
    else:
        print("✅ MODE TRAIN() vs EVAL() SIMILAIRES")
        print("   → Le problème ne vient PAS de BatchNorm")

    print("")

    # Vérifier si targets HV sont vraiment en [-1, 1]
    print("=" * 80)
    print("VÉRIFICATION TARGETS HV")
    print("=" * 80)
    print("")

    print(f"Targets HV range: [{hv_target_224.min():.3f}, {hv_target_224.max():.3f}]")
    print(f"Targets HV mean:  {hv_target_224.mean():.3f}")
    print(f"Targets HV std:   {hv_target_224.std():.3f}")

    # Vérifier coverage
    mask = np_target_224 > 0
    print(f"\nMask coverage: {mask.sum()} / {mask.size} = {mask.mean()*100:.1f}%")

    if mask.sum() > 0:
        print(f"HV dans mask: [{hv_target_224[:, mask].min():.3f}, {hv_target_224[:, mask].max():.3f}]")

if __name__ == "__main__":
    main()
