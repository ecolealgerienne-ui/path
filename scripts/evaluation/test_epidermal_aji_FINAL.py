#!/usr/bin/env python3
"""
Test AJI FINAL pour epidermal avec HV magnitude (pas Sobel).

Usage:
    python scripts/evaluation/test_epidermal_aji_FINAL.py \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
        --n_samples 50
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
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE
from src.metrics.ground_truth_metrics import compute_aji, compute_dice, compute_panoptic_quality
from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform


def extract_instances_hv_magnitude(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    min_size: int = 20,
    dist_threshold: int = 4
) -> np.ndarray:
    """
    Extrait instances avec HV MAGNITUDE (PAS Sobel).

    Méthode: HoVer-Net original (Graham et al. 2019)

    FIX #3 (Expert 2025-12-23): Paramètres ajustés pour réduire sur-segmentation
    - min_size: 10 → 20 pixels (supprimer petits faux positifs)
    - dist_threshold: 2 → 4 (espacer marqueurs watershed)
    """
    # 1. Binariser NP
    binary_mask = (np_pred > 0.5).astype(np.uint8)

    # 2. Remove small noise
    labeled, num = ndimage.label(binary_mask)
    if num == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    sizes = ndimage.sum(binary_mask, labeled, range(1, num + 1))
    mask_size = sizes >= min_size
    remove_small = mask_size[labeled - 1]
    remove_small[labeled == 0] = 0
    binary_mask = remove_small.astype(np.uint8)

    # 3. HV MAGNITUDE (direct, pas Sobel)
    energy = np.sqrt(hv_pred[0]**2 + hv_pred[1]**2)

    # 4. Find peaks
    local_max = peak_local_max(
        energy,
        min_distance=dist_threshold,
        labels=binary_mask.astype(int),
        exclude_border=False,
    )

    # 5. Create markers
    markers = np.zeros_like(binary_mask, dtype=int)
    if len(local_max) > 0:
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

    # 6. Watershed
    if markers.max() > 0:
        inst_map = watershed(-energy, markers, mask=binary_mask)
    else:
        inst_map = ndimage.label(binary_mask)[0]

    return inst_map.astype(np.int32)


def compute_gt_instances(mask: np.ndarray) -> np.ndarray:
    """
    Compute GT instances depuis masque PanNuke.

    IMPORTANT: Utilise vraies instances PanNuke (canaux 1-4),
    PAS connectedComponents qui fusionne les cellules touchantes.
    """
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: vraies instances annotées
    for c in range(1, 5):
        class_instances = mask[:, :, c]
        inst_ids = np.unique(class_instances)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = class_instances == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
    epithelial_binary = mask[:, :, 5] > 0
    if epithelial_binary.any():
        _, epithelial_labels = cv2.connectedComponents(epithelial_binary.astype(np.uint8))
        epithelial_ids = np.unique(epithelial_labels)
        epithelial_ids = epithelial_ids[epithelial_ids > 0]

        for epi_id in epithelial_ids:
            epi_mask = epithelial_labels == epi_id
            inst_map[epi_mask] = instance_counter
            instance_counter += 1

    return inst_map


def main():
    parser = argparse.ArgumentParser(description="Test AJI FINAL epidermal")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint HoVer-Net")
    parser.add_argument("--n_samples", type=int, default=50, help="Nombre échantillons test")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    print("=" * 80)
    print("🎯 TEST AJI FINAL - EPIDERMAL")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Samples: {args.n_samples}")
    print(f"Device: {args.device}")

    # Load model
    print("\n🔧 Chargement modèle...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    backbone.eval()

    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    transform = create_hoptimus_transform()

    # Load epidermal test data
    print("\n📦 Chargement données epidermal...")
    # ⚠️ FIX GHOST PATH BUG: Chercher UN SEUL endroit (source de vérité)
    # AVANT: Cherchait dans data/cache/family_data/ (ancien cache, peut être corrompu)
    # APRÈS: Cherche UNIQUEMENT dans data/family_FIXED/ (dernière version v4)
    data_file = Path("data/family_FIXED/epidermal_data_FIXED.npz")
    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        print("Exécutez d'abord: python scripts/preprocessing/prepare_family_data_FIXED_v4.py --family epidermal")
        return

    data = np.load(data_file)
    images = data['images']

    # PanNuke base directory
    pannuke_dir = Path("/home/amar/data/PanNuke")
    if not pannuke_dir.exists():
        pannuke_dir = Path("data/PanNuke")  # Fallback

    masks = np.load(pannuke_dir / "fold0" / "masks.npy", mmap_mode='r')  # GT complet

    # Use fold IDs to get correct GT masks
    fold_ids = data.get('fold_ids', np.zeros(len(images), dtype=np.int32))
    image_ids = data.get('image_ids', np.arange(len(images)))

    n_test = min(args.n_samples, len(images))
    test_indices = np.random.choice(len(images), n_test, replace=False)

    print(f"  → {n_test} échantillons sélectionnés")

    # Evaluate
    print("\n🧪 Évaluation...")
    all_aji = []
    all_dice = []
    all_pq = []

    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Testing"):
            # Get image and GT
            image = images[idx]
            fold_id = fold_ids[idx]
            img_id = image_ids[idx]

            # Load correct GT mask
            if fold_id == 0:
                gt_mask = masks[img_id]
            else:
                # If fold 1 or 2, load from correct fold
                fold_masks = np.load(pannuke_dir / f"fold{fold_id}" / "masks.npy", mmap_mode='r')
                gt_mask = fold_masks[img_id]

            # Preprocess
            if image.dtype != np.uint8:
                image = image.clip(0, 255).astype(np.uint8)

            tensor = transform(image).unsqueeze(0).to(args.device)

            # Extract features
            features = backbone.forward_features(tensor)
            patch_tokens = features[:, 1:257, :]  # (1, 256, 1536)

            # Predict
            np_out, hv_out, nt_out = hovernet(patch_tokens)

            # Convert to numpy with sigmoid
            np_pred_sigmoid = torch.sigmoid(np_out).cpu().numpy()[0]  # (2, 224, 224)
            hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

            # DEBUG: Check which channel has nuclei
            if idx == test_indices[0]:  # Print once
                print(f"\n🔍 DEBUG (first sample):")
                print(f"  NP channel 0 max: {np_pred_sigmoid[0].max():.4f}")
                print(f"  NP channel 1 max: {np_pred_sigmoid[1].max():.4f}")
                print(f"  HV max: {hv_pred.max():.4f}")
                print(f"  Using channel: 1 (nuclei)")

            # Take channel 1 (nuclei) - channel 0 is background
            np_pred_native = np_pred_sigmoid[1]  # (224, 224)

            # CRITICAL FIX: Extract instances at NATIVE resolution (224×224)
            # BEFORE resizing (resize smooths HV gradients → kills peaks)
            pred_inst_native = extract_instances_hv_magnitude(np_pred_native, hv_pred)

            # Resize instance map to 256×256 with NEAREST (preserves instance IDs)
            pred_inst = cv2.resize(pred_inst_native.astype(np.float32), (256, 256),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)

            # Compute GT instances
            gt_inst = compute_gt_instances(gt_mask)

            # DEBUG: Print instance counts
            if idx == test_indices[0]:  # Print once
                n_pred = len(np.unique(pred_inst)) - 1  # -1 for background
                n_gt = len(np.unique(gt_inst)) - 1
                print(f"  Instances Pred: {n_pred} | GT: {n_gt}")

            # Compute metrics
            aji = compute_aji(pred_inst, gt_inst)

            # Resize NP pred to 256×256 for Dice calculation
            np_pred_256 = cv2.resize(np_pred_native, (256, 256), interpolation=cv2.INTER_LINEAR)
            dice = compute_dice((np_pred_256 > 0.5).astype(np.uint8), (gt_inst > 0).astype(np.uint8))

            pq, dq, sq, _ = compute_panoptic_quality(pred_inst, gt_inst)

            all_aji.append(aji)
            all_dice.append(dice)
            all_pq.append(pq)

    # Results
    print("\n" + "=" * 80)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 80)
    print(f"\n✅ Dice:  {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"✅ AJI:   {np.mean(all_aji):.4f} ± {np.std(all_aji):.4f}")
    print(f"✅ PQ:    {np.mean(all_pq):.4f} ± {np.std(all_pq):.4f}")

    print("\n🎯 Objectifs:")
    print(f"  Dice: >0.90  → {'✅ ATTEINT' if np.mean(all_dice) > 0.90 else '❌ PAS ATTEINT'}")
    print(f"  AJI:  >0.60  → {'✅ ATTEINT' if np.mean(all_aji) > 0.60 else '❌ PAS ATTEINT'}")
    print(f"  PQ:   >0.65  → {'✅ ATTEINT' if np.mean(all_pq) > 0.65 else '❌ PAS ATTEINT'}")


if __name__ == "__main__":
    main()
