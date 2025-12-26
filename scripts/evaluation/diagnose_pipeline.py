#!/usr/bin/env python3
"""
DIAGNOSTIC COMPLET DU PIPELINE - Trace chaque étape avec dimensions et valeurs.

Usage:
    python scripts/evaluation/diagnose_pipeline.py \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform


def main():
    parser = argparse.ArgumentParser(description="Diagnostic complet pipeline")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint HoVer-Net")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    print("=" * 80)
    print("🔍 DIAGNOSTIC COMPLET DU PIPELINE")
    print("=" * 80)

    # ========================================================================
    # ÉTAPE 1: Charger les données
    # ========================================================================
    print("\n" + "=" * 80)
    print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
    print("=" * 80)

    # Chercher le fichier v12
    v12_file = Path("data/family_FIXED/epidermal_data_FIXED_v12_COHERENT.npz")
    if not v12_file.exists():
        print(f"❌ Fichier v12 non trouvé: {v12_file}")
        return

    print(f"\n📂 Fichier v12: {v12_file}")
    data = np.load(v12_file)

    print(f"\n📦 Contenu du fichier v12:")
    for key in data.keys():
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.dtype in [np.float32, np.float64]:
            print(f"       range=[{arr.min():.4f}, {arr.max():.4f}]")
        elif arr.dtype in [np.int32, np.int64]:
            print(f"       unique values: {np.unique(arr)[:10]}...")

    images = data['images']
    np_targets = data['np_targets']
    hv_targets = data['hv_targets']
    nt_targets = data['nt_targets']
    fold_ids = data.get('fold_ids', np.zeros(len(images), dtype=np.int32))
    image_ids = data.get('image_ids', np.arange(len(images)))

    # ========================================================================
    # ÉTAPE 2: Charger GT PanNuke pour comparaison
    # ========================================================================
    print("\n" + "=" * 80)
    print("ÉTAPE 2: CHARGER GT PANNUKE")
    print("=" * 80)

    pannuke_dir = Path("/home/amar/data/PanNuke")
    if not pannuke_dir.exists():
        pannuke_dir = Path("data/PanNuke")

    if not pannuke_dir.exists():
        print(f"❌ PanNuke non trouvé: {pannuke_dir}")
        return

    # ========================================================================
    # ÉTAPE 3: Charger modèle
    # ========================================================================
    print("\n" + "=" * 80)
    print("ÉTAPE 3: CHARGER MODÈLE")
    print("=" * 80)

    print(f"\n🔧 Chargement backbone H-optimus-0...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    backbone.eval()

    print(f"🔧 Chargement HoVer-Net depuis {args.checkpoint}...")
    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    print(f"\n📊 Infos checkpoint:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Dice: {checkpoint.get('val_dice', 'N/A')}")
    print(f"  HV MSE: {checkpoint.get('val_hv_mse', 'N/A')}")

    transform = create_hoptimus_transform()

    # ========================================================================
    # ÉTAPE 4: Analyser UN échantillon en détail
    # ========================================================================
    print("\n" + "=" * 80)
    print("ÉTAPE 4: ANALYSE DÉTAILLÉE D'UN ÉCHANTILLON")
    print("=" * 80)

    idx = 0  # Premier échantillon
    image = images[idx]
    np_target_v12 = np_targets[idx]
    hv_target_v12 = hv_targets[idx]
    nt_target_v12 = nt_targets[idx]
    fold_id = fold_ids[idx]
    img_id = image_ids[idx]

    print(f"\n📋 Échantillon {idx}:")
    print(f"  fold_id: {fold_id}")
    print(f"  image_id: {img_id}")
    print(f"  image shape: {image.shape}, dtype: {image.dtype}")

    # ========================================================================
    # ÉTAPE 4a: Targets v12 (utilisés pour training)
    # ========================================================================
    print(f"\n📊 TARGETS V12 (utilisés pour training):")
    print(f"  NP target shape: {np_target_v12.shape}")
    print(f"  NP target dtype: {np_target_v12.dtype}")
    print(f"  NP coverage: {np_target_v12.mean() * 100:.2f}%")
    print(f"  NP non-zero pixels: {(np_target_v12 > 0).sum()}")

    print(f"\n  HV target shape: {hv_target_v12.shape}")
    print(f"  HV target dtype: {hv_target_v12.dtype}")
    print(f"  HV range: [{hv_target_v12.min():.4f}, {hv_target_v12.max():.4f}]")

    print(f"\n  NT target shape: {nt_target_v12.shape}")
    print(f"  NT target dtype: {nt_target_v12.dtype}")
    print(f"  NT unique classes: {np.unique(nt_target_v12)}")
    print(f"  NT class 1 pixels: {(nt_target_v12 == 1).sum()}")

    # ========================================================================
    # ÉTAPE 4b: GT PanNuke (utilisé pour test)
    # ========================================================================
    print(f"\n📊 GT PANNUKE (utilisé pour test):")

    fold_masks_path = pannuke_dir / f"fold{fold_id}" / "masks.npy"
    print(f"  Loading from: {fold_masks_path}")

    fold_masks = np.load(fold_masks_path, mmap_mode='r')
    gt_mask = np.array(fold_masks[img_id])

    print(f"  GT mask shape: {gt_mask.shape}")
    print(f"  GT mask dtype: {gt_mask.dtype}")

    print(f"\n  Analyse par canal:")
    for c in range(gt_mask.shape[2]):
        channel = gt_mask[:, :, c]
        nonzero = (channel > 0).sum()
        unique_ids = len(np.unique(channel)) - (1 if 0 in channel else 0)
        print(f"    Canal {c}: {nonzero} pixels non-nuls, {unique_ids} IDs uniques")

    # Calculer GT instances comme dans le test
    inst_map = gt_mask[:, :, 0].astype(np.int32)
    current_max_id = inst_map.max()
    for c in range(1, 5):
        channel_map = gt_mask[:, :, c].astype(np.int32)
        if channel_map.max() > 0:
            mask = (inst_map == 0) & (channel_map > 0)
            unique_ids = np.unique(channel_map[mask])
            unique_ids = unique_ids[unique_ids > 0]
            for new_id, old_id in enumerate(unique_ids, start=current_max_id + 1):
                inst_map[(inst_map == 0) & (channel_map == old_id)] = new_id
            current_max_id = inst_map.max()

    gt_binary = (inst_map > 0).astype(np.float32)
    n_gt_instances = len(np.unique(inst_map)) - 1

    print(f"\n  GT instances: {n_gt_instances}")
    print(f"  GT binary coverage: {gt_binary.mean() * 100:.2f}%")
    print(f"  GT binary non-zero pixels: {gt_binary.sum():.0f}")

    # ========================================================================
    # ÉTAPE 4c: Comparer v12 target vs GT PanNuke
    # ========================================================================
    print(f"\n📊 COMPARAISON V12 vs GT PANNUKE:")

    v12_binary = (np_target_v12 > 0).astype(np.float32)

    # Dice entre v12 et GT
    intersection = (v12_binary * gt_binary).sum()
    union = v12_binary.sum() + gt_binary.sum()
    dice_v12_gt = (2 * intersection / union) if union > 0 else 1.0

    print(f"  V12 coverage: {v12_binary.mean() * 100:.2f}%")
    print(f"  GT coverage: {gt_binary.mean() * 100:.2f}%")
    print(f"  Dice(V12, GT): {dice_v12_gt:.4f}")

    # Pixels en commun vs différents
    common = ((v12_binary > 0) & (gt_binary > 0)).sum()
    v12_only = ((v12_binary > 0) & (gt_binary == 0)).sum()
    gt_only = ((v12_binary == 0) & (gt_binary > 0)).sum()

    print(f"\n  Pixels communs: {common}")
    print(f"  Pixels V12 uniquement: {v12_only}")
    print(f"  Pixels GT uniquement: {gt_only}")

    # ========================================================================
    # ÉTAPE 5: Inférence modèle
    # ========================================================================
    print("\n" + "=" * 80)
    print("ÉTAPE 5: INFÉRENCE MODÈLE")
    print("=" * 80)

    # Preprocessing
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)

    print(f"\n  Image après conversion: dtype={image.dtype}, shape={image.shape}")

    tensor = transform(image).unsqueeze(0).to(args.device)
    print(f"  Tensor après transform: shape={tuple(tensor.shape)}")

    # Features
    with torch.no_grad():
        features = backbone.forward_features(tensor)

    print(f"\n  Features shape: {tuple(features.shape)}")
    cls_token = features[:, 0, :]
    print(f"  CLS token std: {cls_token.std().item():.4f}")

    patch_tokens = features[:, 1:257, :]
    print(f"  Patch tokens shape: {tuple(patch_tokens.shape)}")

    # Prédiction
    with torch.no_grad():
        np_out, hv_out, nt_out = hovernet(patch_tokens)

    print(f"\n  NP out shape: {tuple(np_out.shape)}")
    print(f"  HV out shape: {tuple(hv_out.shape)}")
    print(f"  NT out shape: {tuple(nt_out.shape)}")

    # Conversion
    np_probs = torch.softmax(np_out, dim=1)[0].cpu().numpy()  # (2, 224, 224)
    hv_pred = hv_out[0].cpu().numpy()  # (2, 224, 224)

    print(f"\n  NP probs shape: {np_probs.shape}")
    print(f"  NP channel 0 (bg) range: [{np_probs[0].min():.4f}, {np_probs[0].max():.4f}]")
    print(f"  NP channel 1 (fg) range: [{np_probs[1].min():.4f}, {np_probs[1].max():.4f}]")
    print(f"  NP channel 1 mean: {np_probs[1].mean():.4f}")
    print(f"  NP channel 1 > 0.5: {(np_probs[1] > 0.5).sum()} pixels")

    print(f"\n  HV pred shape: {hv_pred.shape}")
    print(f"  HV range: [{hv_pred.min():.4f}, {hv_pred.max():.4f}]")

    # ========================================================================
    # ÉTAPE 6: Resize 224 → 256
    # ========================================================================
    print("\n" + "=" * 80)
    print("ÉTAPE 6: RESIZE 224 → 256")
    print("=" * 80)

    # Transpose pour cv2: (C, H, W) → (H, W, C)
    np_probs_hwc = np_probs.transpose(1, 2, 0)  # (224, 224, 2)
    hv_pred_hwc = hv_pred.transpose(1, 2, 0)  # (224, 224, 2)

    print(f"\n  Avant resize:")
    print(f"    NP shape: {np_probs_hwc.shape}")
    print(f"    HV shape: {hv_pred_hwc.shape}")

    # Resize
    np_pred_256 = cv2.resize(np_probs_hwc, (256, 256), interpolation=cv2.INTER_LINEAR)
    hv_pred_256 = np.zeros((256, 256, 2), dtype=hv_pred_hwc.dtype)
    hv_pred_256[:, :, 0] = cv2.resize(hv_pred_hwc[:, :, 0], (256, 256), interpolation=cv2.INTER_LINEAR)
    hv_pred_256[:, :, 1] = cv2.resize(hv_pred_hwc[:, :, 1], (256, 256), interpolation=cv2.INTER_LINEAR)

    print(f"\n  Après resize:")
    print(f"    NP shape: {np_pred_256.shape}")
    print(f"    HV shape: {hv_pred_256.shape}")

    prob_map = np_pred_256[:, :, 1]  # Canal foreground
    print(f"\n  prob_map (canal 1) shape: {prob_map.shape}")
    print(f"  prob_map range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    print(f"  prob_map mean: {prob_map.mean():.4f}")
    print(f"  prob_map > 0.5: {(prob_map > 0.5).sum()} pixels")

    # ========================================================================
    # ÉTAPE 7: Calcul Dice
    # ========================================================================
    print("\n" + "=" * 80)
    print("ÉTAPE 7: CALCUL DICE")
    print("=" * 80)

    pred_binary = (prob_map > 0.5).astype(np.float32)
    gt_binary_256 = gt_binary  # Déjà à 256×256

    print(f"\n  pred_binary non-zero: {pred_binary.sum():.0f} pixels")
    print(f"  gt_binary non-zero: {gt_binary_256.sum():.0f} pixels")

    intersection = (pred_binary * gt_binary_256).sum()
    union = pred_binary.sum() + gt_binary_256.sum()
    dice = (2 * intersection / union) if union > 0 else 1.0

    print(f"\n  Intersection: {intersection}")
    print(f"  Union: {union}")
    print(f"  ⭐ DICE: {dice:.4f}")

    # ========================================================================
    # ÉTAPE 8: Comparaison avec training target
    # ========================================================================
    print("\n" + "=" * 80)
    print("ÉTAPE 8: COMPARAISON AVEC TRAINING TARGET")
    print("=" * 80)

    # Dice entre prédiction et training target (v12)
    v12_binary_256 = (np_target_v12 > 0).astype(np.float32)

    intersection_v12 = (pred_binary * v12_binary_256).sum()
    union_v12 = pred_binary.sum() + v12_binary_256.sum()
    dice_vs_v12 = (2 * intersection_v12 / union_v12) if union_v12 > 0 else 1.0

    print(f"\n  Dice(pred, v12_target): {dice_vs_v12:.4f}")
    print(f"  Dice(pred, gt_pannuke): {dice:.4f}")
    print(f"  Dice(v12_target, gt_pannuke): {dice_v12_gt:.4f}")

    # ========================================================================
    # RÉSUMÉ
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ DIAGNOSTIC")
    print("=" * 80)

    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │ COMPARAISONS                                                 │
  ├──────────────────────────────────────────────────────────────┤
  │ Dice(Pred, GT PanNuke):   {dice:.4f}                         │
  │ Dice(Pred, V12 Target):   {dice_vs_v12:.4f}                         │
  │ Dice(V12 Target, GT):     {dice_v12_gt:.4f}                         │
  ├──────────────────────────────────────────────────────────────┤
  │ PIXELS                                                       │
  │ Prédiction (>0.5):        {pred_binary.sum():>8.0f} pixels              │
  │ V12 Target:               {v12_binary_256.sum():>8.0f} pixels              │
  │ GT PanNuke:               {gt_binary_256.sum():>8.0f} pixels              │
  └──────────────────────────────────────────────────────────────┘
""")

    if dice < 0.5:
        print("  ❌ PROBLÈME DÉTECTÉ: Dice très faible!")
        if dice_vs_v12 > 0.8:
            print("     → Modèle prédit correctement vs V12 mais V12 ≠ GT")
            print("     → Le problème est la DIFFÉRENCE entre V12 targets et GT PanNuke")
        elif prob_map.mean() < 0.1:
            print("     → prob_map moyenne très faible")
            print("     → Le modèle prédit presque tout comme background")
        else:
            print("     → Investiguer les dimensions et valeurs ci-dessus")


if __name__ == "__main__":
    main()
