#!/usr/bin/env python3
"""
Évaluation AJI utilisant les MÊMES données que l'entraînement.

Reproduit exactement le pipeline training pour éviter les bugs de preprocessing.

Usage:
    python scripts/evaluation/eval_aji_from_training_data.py \
        --family epidermal \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
        --n_samples 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.metrics.ground_truth_metrics import compute_aji, compute_dice
from src.models.hovernet_decoder import HoVerNetDecoder


PROJECT_ROOT = Path(__file__).parent.parent.parent


def watershed_from_hv(np_pred: np.ndarray, hv_pred: np.ndarray) -> np.ndarray:
    """
    Watershed basique depuis HV maps.

    Version simplifiée pour tester si le problème vient du watershed
    ou des données.
    """
    import cv2
    from scipy import ndimage

    # Binariser NP
    np_binary = (np_pred > 0.5).astype(np.uint8)

    if np_binary.sum() == 0:
        return np.zeros_like(np_binary, dtype=np.int32)

    # Distance transform pour markers
    dist = ndimage.distance_transform_edt(np_binary)

    # Markers = peaks de distance
    if dist.max() > 0:
        dist_norm = dist / dist.max()
        markers_mask = dist_norm > 0.5
        markers, _ = ndimage.label(markers_mask)

        if markers.max() == 0:
            # Fallback: connected components
            inst_map, _ = cv2.connectedComponents(np_binary)
            return inst_map.astype(np.int32)

        # Watershed
        markers_ws = markers.astype(np.int32)
        # Use inverted distance as "energy"
        energy = (255 * (1 - dist_norm) * np_binary).astype(np.uint8)
        cv2.watershed(cv2.cvtColor(energy, cv2.COLOR_GRAY2BGR), markers_ws)

        # Clean up
        inst_map = markers_ws.copy()
        inst_map[inst_map == -1] = 0
        inst_map[np_binary == 0] = 0

        return inst_map
    else:
        # Fallback
        inst_map, _ = cv2.connectedComponents(np_binary)
        return inst_map.astype(np.int32)


def extract_gt_instances(np_target: np.ndarray, nt_target: np.ndarray) -> np.ndarray:
    """
    Extrait instances GT depuis les targets.

    Args:
        np_target: (H, W) binaire [0, 1]
        nt_target: (H, W) labels [0, 1, 2, 3, 4, 5]

    Returns:
        inst_map: (H, W) avec IDs instances
    """
    import cv2

    # Pour l'instant, utiliser connected components sur NP binaire
    # TODO: Si les targets contiennent déjà les instance IDs, les utiliser directement
    np_binary = (np_target > 0.5).astype(np.uint8)
    inst_map, _ = cv2.connectedComponents(np_binary)

    return inst_map.astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache_dir', type=Path, default=None,
                        help="Répertoire cache (défaut: data/cache/family_data)")
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"ÉVALUATION AJI - Famille {args.family.upper()}")
    print(f"(Utilise données training - AUCUN preprocessing)")
    print(f"{'='*70}\n")

    # Cache dir
    if args.cache_dir is None:
        cache_dir = PROJECT_ROOT / "data" / "cache" / "family_data"
    else:
        cache_dir = args.cache_dir

    # Load data (EXACTEMENT comme le training!)
    features_path = cache_dir / f"{args.family}_features.npz"
    targets_path = cache_dir / f"{args.family}_targets.npz"

    if not features_path.exists():
        raise FileNotFoundError(f"Features non trouvées: {features_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets non trouvés: {targets_path}")

    print(f"📥 Chargement données training...")
    features_data = np.load(features_path)
    targets_data = np.load(targets_path)

    features = features_data['features']  # (N, 261, 1536)
    np_targets = targets_data['np_targets']  # (N, 256, 256)
    hv_targets = targets_data['hv_targets']  # (N, 2, 256, 256)
    nt_targets = targets_data['nt_targets']  # (N, 256, 256)

    n_total = len(features)
    n_samples = min(args.n_samples, n_total)

    print(f"   ✅ {n_total} échantillons disponibles")
    print(f"   → Évaluation sur {n_samples} échantillons\n")

    # Load model
    print(f"📥 Chargement modèle...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model = HoVerNetDecoder(embed_dim=1536, n_classes=5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    print(f"   ✅ Epoch {checkpoint.get('epoch', '?')}\n")

    # Evaluate
    all_aji = []
    all_dice = []
    all_n_instances_gt = []
    all_n_instances_pred = []

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Évaluation"):
            # Features → Model (EXACTEMENT comme training)
            feat = torch.from_numpy(features[i]).unsqueeze(0).to(args.device)
            np_out, hv_out, nt_out = model(feat)

            # To numpy (EXACTEMENT comme training validation)
            np_pred_logits = np_out.cpu().numpy()[0]  # (2, 224, 224)
            np_pred = np_pred_logits.argmax(axis=0).astype(np.float32)  # (224, 224) [0, 1]
            hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

            # Resize predictions 224 → 256 pour matcher GT
            from src.utils.image_utils import prepare_predictions_for_evaluation
            np_pred_256, hv_pred_256, _ = prepare_predictions_for_evaluation(
                np_pred, hv_pred, np.zeros((5, 224, 224)),
                target_size=256
            )

            # GT à 256×256 (directement depuis targets)
            np_gt = np_targets[i]  # (256, 256)
            hv_gt = hv_targets[i]  # (2, 256, 256)
            nt_gt = nt_targets[i]  # (256, 256)

            # Extract instances
            inst_pred = watershed_from_hv(np_pred_256, hv_pred_256)
            inst_gt = extract_gt_instances(np_gt, nt_gt)

            # Metrics
            aji = compute_aji(inst_pred, inst_gt)
            dice = compute_dice(np_pred_256 > 0.5, np_gt > 0.5)

            all_aji.append(aji)
            all_dice.append(dice)
            all_n_instances_gt.append(len(np.unique(inst_gt)) - 1)
            all_n_instances_pred.append(len(np.unique(inst_pred)) - 1)

    # Results
    print(f"\n{'='*70}")
    print("RÉSULTATS")
    print(f"{'='*70}\n")

    print(f"AJI:                {np.mean(all_aji):.4f} ± {np.std(all_aji):.4f}")
    print(f"Dice (binary):      {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Instances GT:       {np.mean(all_n_instances_gt):.1f} par image")
    print(f"Instances Pred:     {np.mean(all_n_instances_pred):.1f} par image")

    print(f"\nDistribution AJI:")
    print(f"  Min:  {min(all_aji):.4f}")
    print(f"  Q25:  {np.percentile(all_aji, 25):.4f}")
    print(f"  Médiane: {np.median(all_aji):.4f}")
    print(f"  Q75:  {np.percentile(all_aji, 75):.4f}")
    print(f"  Max:  {max(all_aji):.4f}")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
