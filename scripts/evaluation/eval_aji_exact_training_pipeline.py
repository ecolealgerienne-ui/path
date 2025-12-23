#!/usr/bin/env python3
"""
Évaluation AJI avec pipeline EXACT du training.

Copie ligne par ligne:
  - extract_features.py: preprocessing image → features
  - prepare_family_data_FIXED.py: mask → GT targets (NP, HV, NT)
  - train_hovernet_family.py: resize 256→224, compute metrics

Usage:
    python scripts/evaluation/eval_aji_exact_training_pipeline.py \
        --family epidermal \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
        --data_dir /home/amar/data/PanNuke \
        --fold 2 \
        --n_samples 20
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import create_hoptimus_transform
from src.models.loader import ModelLoader
from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.organ_families import ORGAN_TO_FAMILY
from src.metrics.ground_truth_metrics import compute_aji, compute_dice


def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """
    COPIÉ EXACTEMENT de prepare_family_data_FIXED.py lignes 29-76.

    Calcule les cartes Horizontal/Vertical pour séparation d'instances.
    """
    hv_map = np.zeros((2, 256, 256), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id

        # Trouver le centre de l'instance
        coords = np.argwhere(inst_mask)
        if len(coords) == 0:
            continue

        center_y, center_x = coords.mean(axis=0)

        # Pour chaque pixel de l'instance
        for y, x in coords:
            # Distance normalisée au centre [-1, 1]
            h = (x - center_x) / (inst_mask.shape[1] / 2)
            v = (y - center_y) / (inst_mask.shape[0] / 2)

            # Clamper à [-1, 1]
            h = np.clip(h, -1.0, 1.0)
            v = np.clip(v, -1.0, 1.0)

            hv_map[0, y, x] = h
            hv_map[1, y, x] = v

    return hv_map


def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """
    COPIÉ EXACTEMENT de prepare_family_data_FIXED.py lignes 79-134.

    Extrait les vraies instances de PanNuke (FIXED).
    """
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: IDs d'instances natifs PanNuke
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
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


def watershed_from_hv(np_pred: np.ndarray, hv_pred: np.ndarray) -> np.ndarray:
    """Watershed basique depuis HV maps."""
    np_binary = (np_pred > 0.5).astype(np.uint8)

    if np_binary.sum() == 0:
        return np.zeros_like(np_binary, dtype=np.int32)

    dist = ndimage.distance_transform_edt(np_binary)

    if dist.max() > 0:
        dist_norm = dist / dist.max()
        markers_mask = dist_norm > 0.5
        markers, _ = ndimage.label(markers_mask)

        if markers.max() == 0:
            _, inst_map = cv2.connectedComponents(np_binary)
            return inst_map.astype(np.int32)

        markers_ws = markers.astype(np.int32)
        energy = (255 * (1 - dist_norm) * np_binary).astype(np.uint8)
        cv2.watershed(cv2.cvtColor(energy, cv2.COLOR_GRAY2BGR), markers_ws)

        inst_map = markers_ws.copy()
        inst_map[inst_map == -1] = 0
        inst_map[np_binary == 0] = 0

        return inst_map
    else:
        _, inst_map = cv2.connectedComponents(np_binary)
        return inst_map.astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Load PanNuke
    fold_name = f"fold{args.fold}"
    images_path = args.data_dir / fold_name / "images.npy"
    masks_path = args.data_dir / fold_name / "masks.npy"
    types_path = args.data_dir / fold_name / "types.npy"

    print(f"Chargement PanNuke fold {args.fold}...")
    images = np.load(images_path)
    masks = np.load(masks_path)
    types = np.load(types_path)

    # Filter by family
    family_organs = [organ for organ, fam in ORGAN_TO_FAMILY.items() if fam == args.family]
    indices = [i for i, organ in enumerate(types) if organ in family_organs]

    images_fam = images[indices][:args.n_samples]
    masks_fam = masks[indices][:args.n_samples]

    print(f"Test sur {len(images_fam)} échantillons {args.family}")

    # ÉTAPE 1: Load models (comme extract_features.py + train_hovernet_family.py)
    print("\nChargement modèles...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)

    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    # ÉTAPE 2: Preprocessing transform (comme extract_features.py ligne 166)
    transform = create_hoptimus_transform()

    # Evaluate
    ajis = []
    dices = []
    n_instances_gt = []
    n_instances_pred = []

    print(f"\nÉvaluation...")
    for i in tqdm(range(len(images_fam)), desc="Évaluation"):
        image = images_fam[i]  # (256, 256, 3)
        mask = masks_fam[i]    # (256, 256, 6)

        # ÉTAPE 3: Preprocessing image (comme extract_features.py lignes 161-166)
        if image.dtype != np.uint8:
            image = image.clip(0, 255).astype(np.uint8)

        tensor = transform(image).unsqueeze(0).to(args.device)  # (1, 3, 224, 224)

        # ÉTAPE 4: Extract features (comme extract_features.py ligne 77)
        with torch.no_grad():
            features = backbone.forward_features(tensor)  # (1, 261, 1536)

        # ÉTAPE 5: Compute GT targets (comme prepare_family_data_FIXED.py lignes 230-239)
        inst_map_256 = extract_pannuke_instances(mask)
        np_target_256 = (inst_map_256 > 0).astype(np.float32)
        hv_target_256 = compute_hv_maps(inst_map_256)

        # ÉTAPE 6: Resize targets 256→224 (comme train_hovernet_family.py lignes 172-183)
        np_target_t = torch.from_numpy(np_target_256)
        hv_target_t = torch.from_numpy(hv_target_256)

        np_target_t = F.interpolate(np_target_t.unsqueeze(0).unsqueeze(0),
                                    size=(224, 224), mode='nearest').squeeze()
        hv_target_t = F.interpolate(hv_target_t.unsqueeze(0),
                                    size=(224, 224), mode='bilinear',
                                    align_corners=False).squeeze(0)

        np_target = np_target_t.numpy()
        hv_target = hv_target_t.numpy()

        # Recalculer inst_gt à 224×224
        inst_gt = cv2.resize(inst_map_256.astype(np.float32), (224, 224),
                           interpolation=cv2.INTER_NEAREST).astype(np.int32)

        # ÉTAPE 7: HoVer-Net prediction (comme train_hovernet_family.py ligne 352)
        with torch.no_grad():
            np_out, hv_out, nt_out = hovernet(features)

        # ÉTAPE 8: To numpy (comme train_hovernet_family.py ligne 427)
        np_pred_logits = np_out.cpu().numpy()[0]  # (2, 224, 224)
        np_pred = (np_pred_logits.argmax(axis=0)).astype(np.float32)  # (224, 224) [0, 1]
        hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

        # ÉTAPE 9: Watershed
        inst_pred = watershed_from_hv(np_pred, hv_pred)

        # ÉTAPE 10: Metrics (comme train_hovernet_family.py lignes 427-438)
        aji = compute_aji(inst_pred, inst_gt)
        dice = compute_dice(np_pred, np_target)

        ajis.append(aji)
        dices.append(dice)
        n_instances_gt.append(len(np.unique(inst_gt)) - 1)
        n_instances_pred.append(len(np.unique(inst_pred)) - 1)

    # Results
    ajis = np.array(ajis)
    dices = np.array(dices)
    n_instances_gt = np.array(n_instances_gt)
    n_instances_pred = np.array(n_instances_pred)

    print(f"\n{'='*70}")
    print("RÉSULTATS")
    print(f"{'='*70}\n")

    print(f"AJI:                {ajis.mean():.4f} ± {ajis.std():.4f}")
    print(f"Dice (binary):      {dices.mean():.4f} ± {dices.std():.4f}")
    print(f"Instances GT:       {n_instances_gt.mean():.1f} par image")
    print(f"Instances Pred:     {n_instances_pred.mean():.1f} par image")

    print(f"\nDistribution AJI:")
    print(f"  Min:  {ajis.min():.4f}")
    print(f"  Q25:  {np.percentile(ajis, 25):.4f}")
    print(f"  Médiane: {np.median(ajis):.4f}")
    print(f"  Q75:  {np.percentile(ajis, 75):.4f}")
    print(f"  Max:  {ajis.max():.4f}")
    print()


if __name__ == '__main__':
    main()
