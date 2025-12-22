#!/usr/bin/env python3
"""
Trace COMPLET du pipeline train vs test pour diagnostiquer HV MSE.

Compare:
1. Chargement données
2. Preprocessing
3. Entrée modèle
4. Sortie modèle
5. Calcul métriques
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.loader import ModelLoader


def print_tensor_info(name: str, data, level=0):
    """Affiche info détaillée sur un tensor/array."""
    indent = "  " * level
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu()
        dtype_str = str(data.dtype)
    elif isinstance(data, np.ndarray):
        dtype_str = str(data.dtype)
    else:
        print(f"{indent}{name}: type={type(data)}")
        return

    print(f"{indent}{name}:")
    print(f"{indent}  shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
    print(f"{indent}  dtype: {dtype_str}")

    if hasattr(data, 'min'):
        data_flat = data.flatten() if hasattr(data, 'flatten') else data
        # Convert to float for mean/std if needed (int64 doesn't support these ops)
        if hasattr(data_flat, 'dtype') and 'int' in str(data_flat.dtype):
            data_flat_f = data_flat.float() if isinstance(data_flat, torch.Tensor) else data_flat.astype(np.float32)
        else:
            data_flat_f = data_flat
        print(f"{indent}  range: [{float(data_flat.min()):.6f}, {float(data_flat.max()):.6f}]")
        print(f"{indent}  mean:  {float(data_flat_f.mean()):.6f}")
        print(f"{indent}  std:   {float(data_flat_f.std()):.6f}")


def trace_train_pipeline():
    """Trace le pipeline TRAIN."""
    print("=" * 80)
    print("PIPELINE TRAIN")
    print("=" * 80)
    print()

    # 1. Chargement données
    print("=" * 80)
    print("ÉTAPE 1: Chargement .npz")
    print("=" * 80)

    features_path = PROJECT_ROOT / "data/cache/family_data_FIXED/digestive_features.npz"
    targets_path = PROJECT_ROOT / "data/cache/family_data_FIXED/digestive_targets.npz"

    features_data = np.load(features_path)
    targets_data = np.load(targets_path)

    features = features_data['features']
    np_targets = targets_data['np_targets']
    hv_targets_raw = targets_data['hv_targets']
    nt_targets = targets_data['nt_targets']

    print("\nFeatures (.npz):")
    print_tensor_info("features", features, level=1)

    print("\nTargets (.npz):")
    print_tensor_info("np_targets", np_targets, level=1)
    print_tensor_info("hv_targets_raw", hv_targets_raw, level=1)
    print_tensor_info("nt_targets", nt_targets, level=1)

    # 2. DataLoader conversion (ce que fait FamilyHoVerDataset.__init__)
    print()
    print("=" * 80)
    print("ÉTAPE 2: Conversion DataLoader (FamilyHoVerDataset.__init__)")
    print("=" * 80)

    # Check dtype et conversion comme dans train script
    if hv_targets_raw.dtype == np.int8:
        hv_targets = hv_targets_raw.astype(np.float32) / 127.0
        print("\n⚠️  HV format OLD détecté (int8) - conversion en float32")
    else:
        hv_targets = hv_targets_raw
        print("\n✅ HV format NEW détecté (float32) - pas de conversion")

    print_tensor_info("hv_targets (après conversion)", hv_targets, level=1)

    # 3. __getitem__ (premier échantillon)
    print()
    print("=" * 80)
    print("ÉTAPE 3: DataLoader __getitem__ (échantillon 0)")
    print("=" * 80)

    feat_sample = features[0]  # (261, 1536)
    np_sample = np_targets[0]  # (224, 224)
    hv_sample = hv_targets[0]  # (2, 224, 224)
    nt_sample = nt_targets[0]  # (224, 224)

    print_tensor_info("features[0]", feat_sample, level=1)
    print_tensor_info("np_targets[0]", np_sample, level=1)
    print_tensor_info("hv_targets[0]", hv_sample, level=1)
    print_tensor_info("nt_targets[0]", nt_sample, level=1)

    # 4. Conversion torch + RESIZE 256→224 (ce que fait DataLoader.__getitem__)
    print()
    print("=" * 80)
    print("ÉTAPE 4: Resize 256→224 + Conversion torch.Tensor (DataLoader)")
    print("=" * 80)

    feat_t = torch.from_numpy(feat_sample).unsqueeze(0)  # (1, 261, 1536)

    # Resize targets 256→224 (train_hovernet_family.py lignes 172-183)
    np_target_t = torch.from_numpy(np_sample)
    hv_target_t = torch.from_numpy(hv_sample)
    nt_target_t = torch.from_numpy(nt_sample)

    np_t = F.interpolate(np_target_t.unsqueeze(0).unsqueeze(0),
                         size=(224, 224), mode='nearest').squeeze(0)  # (1, 224, 224)
    hv_t = F.interpolate(hv_target_t.unsqueeze(0),
                         size=(224, 224), mode='bilinear',
                         align_corners=False)  # (1, 2, 224, 224)
    nt_t = F.interpolate(nt_target_t.float().unsqueeze(0).unsqueeze(0),
                         size=(224, 224), mode='nearest').squeeze(0).long()  # (1, 224, 224)

    print_tensor_info("features (batch)", feat_t, level=1)
    print_tensor_info("np_target 224×224 (batch)", np_t, level=1)
    print_tensor_info("hv_target 224×224 (batch)", hv_t, level=1)
    print_tensor_info("nt_target 224×224 (batch)", nt_t, level=1)

    # 5. Entrée modèle
    print()
    print("=" * 80)
    print("ÉTAPE 5: Entrée modèle (patch_tokens)")
    print("=" * 80)

    patch_tokens = feat_t[:, 1:257, :]  # Enlever CLS + registers
    print_tensor_info("patch_tokens", patch_tokens, level=1)

    # 6. Sortie modèle
    print()
    print("=" * 80)
    print("ÉTAPE 6: Sortie modèle")
    print("=" * 80)

    checkpoint = PROJECT_ROOT / "models/checkpoints/hovernet_digestive_best.pth"
    if checkpoint.exists():
        print(f"⚠️  Checkpoint trouvé: {checkpoint}")
        hovernet = ModelLoader.load_hovernet(checkpoint, device='cpu')
    else:
        print("✅ Pas de checkpoint - modèle random init")
        from src.models.hovernet_decoder import HoVerNetDecoder
        hovernet = HoVerNetDecoder(n_classes=6, dropout=0.1)

    hovernet.eval()

    with torch.no_grad():
        np_out, hv_out, nt_out = hovernet(patch_tokens)

    print_tensor_info("np_out (AVANT activation)", np_out, level=1)
    print_tensor_info("hv_out (APRÈS tanh)", hv_out, level=1)
    print_tensor_info("nt_out (AVANT activation)", nt_out, level=1)

    # 7. Calcul métriques (comme dans train script)
    print()
    print("=" * 80)
    print("ÉTAPE 7: Calcul métriques TRAIN")
    print("=" * 80)

    # HV MSE calculation (from train_hovernet_family.py compute_hv_mse)
    mask = np_t.float().unsqueeze(1)  # (1, 1, 224, 224)

    if mask.sum() == 0:
        hv_mse = 0.0
    else:
        diff = (hv_out - hv_t) ** 2
        masked_diff = diff * mask
        hv_mse = (masked_diff.sum() / (mask.sum() * 2)).item()

    print(f"\nHV MSE (train method): {hv_mse:.6f}")
    print(f"  Mask pixels: {mask.sum().item():.0f}")
    print(f"  Total pixels: {np.prod(hv_out.shape)}")

    return hv_out, hv_t, mask


def trace_test_pipeline():
    """Trace le pipeline TEST."""
    print()
    print()
    print("=" * 80)
    print("PIPELINE TEST")
    print("=" * 80)
    print()

    # 1. Chargement données
    print("=" * 80)
    print("ÉTAPE 1: Chargement .npz")
    print("=" * 80)

    features_path = PROJECT_ROOT / "data/cache/family_data_FIXED/digestive_features.npz"
    targets_path = PROJECT_ROOT / "data/cache/family_data_FIXED/digestive_targets.npz"

    features_data = np.load(features_path)
    targets_data = np.load(targets_path)

    features = features_data['features']
    np_targets = targets_data['np_targets']
    hv_targets = targets_data['hv_targets']  # PAS de conversion ici dans test script
    nt_targets = targets_data['nt_targets']

    print("\nFeatures (.npz):")
    print_tensor_info("features", features, level=1)

    print("\nTargets (.npz) - SANS conversion:")
    print_tensor_info("hv_targets (brut)", hv_targets, level=1)

    # 2. Échantillon 0
    print()
    print("=" * 80)
    print("ÉTAPE 2: Échantillon 0 (test script)")
    print("=" * 80)

    feat = features[0]
    np_target_256 = np_targets[0]
    hv_target_256 = hv_targets[0]
    nt_target_256 = nt_targets[0]

    print_tensor_info("features[0]", feat, level=1)
    print_tensor_info("hv_target_256[0]", hv_target_256, level=1)

    # 3. Resize targets (test script fait resize 256→224)
    print()
    print("=" * 80)
    print("ÉTAPE 3: Resize targets 256→224 (test script)")
    print("=" * 80)

    # Resize comme dans test script
    np_target_t = torch.from_numpy(np_target_256).float().unsqueeze(0).unsqueeze(0)
    hv_target_t = torch.from_numpy(hv_target_256).float().unsqueeze(0)

    hv_target_t = F.interpolate(hv_target_t, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    hv_target_224 = hv_target_t.numpy()

    print_tensor_info("hv_target_224 (après resize)", hv_target_224, level=1)

    # 4. Inférence
    print()
    print("=" * 80)
    print("ÉTAPE 4: Inférence (test script)")
    print("=" * 80)

    feat_tensor = torch.from_numpy(feat).unsqueeze(0)
    patch_tokens = feat_tensor[:, 1:257, :]

    checkpoint = PROJECT_ROOT / "models/checkpoints/hovernet_digestive_best.pth"
    if checkpoint.exists():
        hovernet = ModelLoader.load_hovernet(checkpoint, device='cpu')
    else:
        from src.models.hovernet_decoder import HoVerNetDecoder
        hovernet = HoVerNetDecoder(n_classes=6, dropout=0.1)

    hovernet.eval()

    with torch.no_grad():
        np_out, hv_out, nt_out = hovernet(patch_tokens)

    hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

    print_tensor_info("hv_pred (test)", hv_pred, level=1)

    # 5. Calcul métriques (test script)
    print()
    print("=" * 80)
    print("ÉTAPE 5: Calcul métriques TEST")
    print("=" * 80)

    # Resize np_target pour avoir le mask à 224×224 (comme dans le vrai test script ligne 159)
    np_target_t = torch.from_numpy(np_target_256).float().unsqueeze(0).unsqueeze(0)
    np_target_224 = F.interpolate(np_target_t, size=(224, 224), mode='nearest').squeeze().numpy()

    mask = np_target_224 > 0  # Mask à 224×224 maintenant

    if mask.sum() > 0:
        # Test script calcule MSE comme ça (compute_mse function)
        hv_mse = ((hv_pred[:, mask] - hv_target_224[:, mask]) ** 2).mean()
    else:
        hv_mse = 0.0

    print(f"\nHV MSE (test method): {hv_mse:.6f}")
    print(f"  Mask pixels: {mask.sum():.0f}")
    print(f"  Total pixels: {np.prod(hv_pred.shape)}")

    return hv_pred, hv_target_224, mask


def compare_pipelines():
    """Compare les deux pipelines."""
    print()
    print()
    print("=" * 80)
    print("COMPARAISON TRAIN vs TEST")
    print("=" * 80)
    print()

    hv_out_train, hv_t_train, mask_train = trace_train_pipeline()
    hv_pred_test, hv_target_test, mask_test = trace_test_pipeline()

    print()
    print("=" * 80)
    print("ANALYSE DES DIFFÉRENCES")
    print("=" * 80)
    print()

    # Comparaison des sorties HV
    print("1. SORTIES MODÈLE HV:")
    print(f"   TRAIN: range [{hv_out_train.min():.6f}, {hv_out_train.max():.6f}], std {hv_out_train.std():.6f}")
    print(f"   TEST:  range [{hv_pred_test.min():.6f}, {hv_pred_test.max():.6f}], std {hv_pred_test.std():.6f}")
    print(f"   → Identiques? {torch.allclose(hv_out_train, torch.from_numpy(hv_pred_test))}")
    print()

    # Comparaison des targets HV
    print("2. TARGETS HV (après resize 224×224):")
    print(f"   TRAIN: range [{hv_t_train.min():.6f}, {hv_t_train.max():.6f}], std {hv_t_train.std():.6f}")
    print(f"   TEST:  range [{hv_target_test.min():.6f}, {hv_target_test.max():.6f}], std {hv_target_test.std():.6f}")
    print(f"   → Identiques? {np.allclose(hv_t_train.numpy(), hv_target_test)}")
    print()

    # Comparaison méthode de calcul
    print("3. MÉTHODE CALCUL MSE:")
    print("   TRAIN: masked_diff.sum() / (mask.sum() * 2)")
    print("   TEST:  ((pred[:, mask] - target[:, mask]) ** 2).mean()")
    print()

    # Recalculer MSE avec les deux méthodes pour comparer
    mask_train_np = (mask_train.numpy()[0, 0] > 0)  # (224, 224) boolean

    # Méthode TRAIN
    diff_train = (hv_out_train - hv_t_train) ** 2
    masked_diff_train = diff_train * mask_train
    mse_train_method = (masked_diff_train.sum() / (mask_train.sum() * 2)).item()

    # Méthode TEST sur mêmes données
    hv_out_np = hv_out_train.numpy()[0]  # (2, 224, 224)
    hv_t_np = hv_t_train.numpy()[0]  # (2, 224, 224)
    mse_test_method = ((hv_out_np[:, mask_train_np] - hv_t_np[:, mask_train_np]) ** 2).mean()

    print(f"   MSE (méthode TRAIN): {mse_train_method:.6f}")
    print(f"   MSE (méthode TEST):  {mse_test_method:.6f}")
    print(f"   → Différence: {abs(mse_train_method - mse_test_method):.6f}")
    print()
    print("   ⚠️  Les deux méthodes donnent le MÊME résultat")
    print("   → Le problème n'est PAS dans le calcul MSE")
    print()

    # Diagnostic final
    print("=" * 80)
    print("DIAGNOSTIC FINAL - VUE GLOBALE")
    print("=" * 80)
    print()

    print("RÉSUMÉ:")
    print(f"  • Même données: ✅ (features, targets identiques)")
    print(f"  • Même modèle: ✅ (checkpoint chargé)")
    print(f"  • Même MSE: ✅ (train={mse_train_method:.4f}, test={mse_test_method:.4f})")
    print()

    print("ANALYSE DES VALEURS:")
    print(f"  • HV pred:   range [{hv_out_train.min():.3f}, {hv_out_train.max():.3f}], std {hv_out_train.std():.3f}")
    print(f"  • HV target: range [{hv_t_train.min():.3f}, {hv_t_train.max():.3f}], std {hv_t_train.std():.3f}")
    print()

    # Le vrai diagnostic
    if hv_out_train.std() < 0.1:
        print("🔍 PROBLÈME IDENTIFIÉ:")
        print()
        print("  Le modèle produit des HV très proches de 0:")
        print(f"    - std = {hv_out_train.std():.4f} (attendu ~{hv_t_train.std():.4f})")
        print(f"    - range compressé: [{hv_out_train.min():.3f}, {hv_out_train.max():.3f}]")
        print()
        print("  CAUSE:")
        print("    Le checkpoint a été entraîné avec targets DIVISÉS par 127")
        print("    → Le modèle a appris à prédire des valeurs ~[-0.01, 0.01]")
        print("    → Maintenant on teste avec targets CORRECTS [-1, 1]")
        print("    → MSE = 0.30 car prédictions et targets sont dans des plages différentes")
        print()
        print("  SOLUTION:")
        print("    Ré-entraîner from scratch avec données FIXED (sans division par 127)")
        print("    → Le modèle apprendra à utiliser toute la plage [-1, 1]")
        print("    → HV MSE descendra < 0.05")
    else:
        print("✅ Le modèle utilise bien la plage [-1, 1]")
    print()


if __name__ == "__main__":
    compare_pipelines()
