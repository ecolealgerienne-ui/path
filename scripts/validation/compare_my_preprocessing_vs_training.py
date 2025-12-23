#!/usr/bin/env python3
"""
Compare mon preprocessing vs les données d'entraînement stockées.

Pour trouver la différence qui cause AJI 0.30 vs 0.94.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import create_hoptimus_transform
from src.models.loader import ModelLoader


def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """COPIÉ de prepare_family_data_FIXED.py"""
    hv_map = np.zeros((2, 256, 256), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        coords = np.argwhere(inst_mask)
        if len(coords) == 0:
            continue

        center_y, center_x = coords.mean(axis=0)

        for y, x in coords:
            h = (x - center_x) / (inst_mask.shape[1] / 2)
            v = (y - center_y) / (inst_mask.shape[0] / 2)
            h = np.clip(h, -1.0, 1.0)
            v = np.clip(v, -1.0, 1.0)
            hv_map[0, y, x] = h
            hv_map[1, y, x] = v

    return hv_map


def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """COPIÉ de prepare_family_data_FIXED.py"""
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    epithelial_mask = mask[:, :, 5]
    if epithelial_mask.max() > 0:
        _, epithelial_labels = cv2.connectedComponents(epithelial_mask.astype(np.uint8))
        epithelial_ids = np.unique(epithelial_labels)
        epithelial_ids = epithelial_ids[epithelial_ids > 0]

        for inst_id in epithelial_ids:
            inst_mask = epithelial_labels == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    return inst_map


def main():
    family = "epidermal"
    fold = 0  # Utiliser fold 0 (entraînement)
    sample_idx = 0  # Premier échantillon

    print("="*70)
    print("COMPARAISON PREPROCESSING MON SCRIPT vs TRAINING")
    print("="*70)

    # 1. Charger données d'entraînement stockées
    print("\n1. Chargement données d'entraînement stockées...")

    features_train_path = Path(f"data/cache/family_data/{family}_features.npz")
    targets_train_path = Path(f"data/cache/family_data/{family}_targets.npz")

    if not features_train_path.exists():
        print(f"❌ {features_train_path} n'existe pas")
        return

    features_train_data = np.load(features_train_path)
    targets_train_data = np.load(targets_train_path)

    # Afficher les clés disponibles
    print(f"   Clés features: {list(features_train_data.keys())}")
    print(f"   Clés targets: {list(targets_train_data.keys())}")

    features_train = features_train_data['features'][sample_idx]  # (261, 1536)
    np_target_train = targets_train_data['np_targets'][sample_idx]  # (256, 256)
    hv_target_train = targets_train_data['hv_targets'][sample_idx]  # (2, 256, 256)
    nt_target_train = targets_train_data['nt_targets'][sample_idx]  # (256, 256)

    # CRITIQUE: Récupérer l'index PanNuke original (si disponible)
    if 'fold_ids' in targets_train_data.keys():
        fold_id_train = targets_train_data['fold_ids'][sample_idx]
        image_id_train = targets_train_data['image_ids'][sample_idx]
        print(f"   📌 Fold ID: {fold_id_train}, Image ID: {image_id_train}")
    else:
        print(f"   ⚠️ fold_ids/image_ids non disponibles - impossible de retrouver l'image exacte")
        print(f"   → Les données ont été générées sans ces métadonnées")
        return

    print(f"   Features shape: {features_train.shape}")
    print(f"   NP target shape: {np_target_train.shape}")
    print(f"   HV target shape: {hv_target_train.shape}")
    print(f"   NT target shape: {nt_target_train.shape}")

    # 2. Charger LA MÊME image brute PanNuke (en utilisant fold_id et image_id)
    print("\n2. Chargement LA MÊME image brute PanNuke...")

    pannuke_path = Path("/home/amar/data/PanNuke")
    fold_name = f"fold{fold_id_train}"

    images_path = pannuke_path / fold_name / "images.npy"
    masks_path = pannuke_path / fold_name / "masks.npy"
    types_path = pannuke_path / fold_name / "types.npy"

    images = np.load(images_path)
    masks = np.load(masks_path)
    types = np.load(types_path)

    # Utiliser l'image_id stocké dans le .npz
    actual_idx = image_id_train

    image = images[actual_idx]  # (256, 256, 3)
    mask = masks[actual_idx]    # (256, 256, 6)

    print(f"   Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"   Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"   Organ: {types[actual_idx]}")
    print(f"   📌 PanNuke Index: {actual_idx} (fold {fold_id_train})")

    # 3. Mon preprocessing
    print("\n3. Mon preprocessing...")

    # Image preprocessing
    if image.dtype != np.uint8:
        image_uint8 = image.clip(0, 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()

    transform = create_hoptimus_transform()
    tensor = transform(image_uint8).unsqueeze(0)

    # Features extraction
    backbone = ModelLoader.load_hoptimus0(device='cuda')
    with torch.no_grad():
        features_mine = backbone.forward_features(tensor.cuda())
    features_mine = features_mine.cpu().numpy()[0]  # (261, 1536)

    # GT targets
    inst_map = extract_pannuke_instances(mask)
    np_target_mine = (inst_map > 0).astype(np.float32)
    hv_target_mine = compute_hv_maps(inst_map)
    nt_target_mine = np.argmax(mask[:, :, 1:], axis=-1).astype(np.int64)

    print(f"   Features shape: {features_mine.shape}")
    print(f"   NP target shape: {np_target_mine.shape}")
    print(f"   HV target shape: {hv_target_mine.shape}")
    print(f"   NT target shape: {nt_target_mine.shape}")

    # 4. Comparaison
    print("\n" + "="*70)
    print("RÉSULTATS COMPARAISON")
    print("="*70)

    print(f"\n📌 VÉRIFICATION MÊME IMAGE:")
    print(f"   Fold ID: {fold_id_train}")
    print(f"   Image ID: {image_id_train}")
    print(f"   Organ: {types[actual_idx]}")

    # Features
    print("\n📊 FEATURES H-optimus-0:")
    features_diff = np.abs(features_train - features_mine).max()
    features_mse = ((features_train - features_mine) ** 2).mean()
    print(f"   Max diff: {features_diff:.6f}")
    print(f"   MSE: {features_mse:.6f}")
    if features_diff < 1e-4:
        print("   ✅ IDENTIQUES")
    else:
        print("   ❌ DIFFÉRENTES")

    # NP target
    print("\n📊 NP TARGET:")
    np_diff = np.abs(np_target_train - np_target_mine).max()
    np_match = (np_target_train == np_target_mine).all()
    print(f"   Max diff: {np_diff:.6f}")
    print(f"   Match: {np_match}")
    if np_match:
        print("   ✅ IDENTIQUES")
    else:
        print("   ❌ DIFFÉRENTES")
        print(f"   Train: {np.unique(np_target_train)}")
        print(f"   Mine:  {np.unique(np_target_mine)}")

    # HV target
    print("\n📊 HV TARGET:")
    hv_diff = np.abs(hv_target_train - hv_target_mine).max()
    hv_mse = ((hv_target_train - hv_target_mine) ** 2).mean()
    print(f"   Max diff: {hv_diff:.6f}")
    print(f"   MSE: {hv_mse:.6f}")
    if hv_diff < 1e-4:
        print("   ✅ IDENTIQUES")
    else:
        print("   ❌ DIFFÉRENTES")
        print(f"   Train range: [{hv_target_train.min():.3f}, {hv_target_train.max():.3f}]")
        print(f"   Mine range:  [{hv_target_mine.min():.3f}, {hv_target_mine.max():.3f}]")

    # NT target
    print("\n📊 NT TARGET:")
    nt_match = (nt_target_train == nt_target_mine).all()
    nt_diff = (nt_target_train != nt_target_mine).sum()
    print(f"   Match: {nt_match}")
    print(f"   Diff pixels: {nt_diff}")
    if nt_match:
        print("   ✅ IDENTIQUES")
    else:
        print("   ❌ DIFFÉRENTES")

    # Instances count
    print("\n📊 INSTANCES:")
    inst_from_train = len(np.unique(np_target_train)) - 1
    inst_from_mine = len(np.unique(inst_map)) - 1
    print(f"   Train (depuis NP binaire): {inst_from_train}")
    print(f"   Mine (depuis inst_map):    {inst_from_mine}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
