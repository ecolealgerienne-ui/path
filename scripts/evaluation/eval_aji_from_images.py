#!/usr/bin/env python3
"""
Évaluation AJI sur images brutes (jamais vues).

Pipeline EXACT comme l'entraînement:
  1. Image brute (256×256 uint8)
  2. Preprocessing H-optimus-0 (src.preprocessing)
  3. Features extraction (forward_features)
  4. Validation CLS std [0.70-0.90]
  5. HoVer-Net prediction
  6. Resize 224 → 256
  7. Watershed
  8. AJI vs GT

Usage:
    # Test sur PanNuke fold2 (données jamais vues)
    python scripts/evaluation/eval_aji_from_images.py \
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
from scipy import ndimage
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import preprocess_image, validate_features
from src.models.loader import ModelLoader
from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.organ_families import ORGAN_TO_FAMILY
from src.metrics.ground_truth_metrics import compute_aji, compute_dice


PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_pannuke_data(data_dir: Path, fold: int):
    """
    Charge images et masks PanNuke pour un fold.

    Returns:
        images: (N, 256, 256, 3) uint8
        masks: (N, 256, 256, 6) - channels [background, type1, type2, type3, type4, type5]
        types: (N,) organ names
    """
    fold_name = f"fold{fold}"

    images_path = data_dir / "fold1" / "images" / fold_name / "images.npy"
    masks_path = data_dir / "fold1" / "masks" / fold_name / "masks.npy"
    types_path = data_dir / "fold1" / "images" / fold_name / "types.npy"

    print(f"Chargement PanNuke fold {fold}...")
    print(f"  Images: {images_path}")
    print(f"  Masks:  {masks_path}")
    print(f"  Types:  {types_path}")

    images = np.load(images_path)
    masks = np.load(masks_path)
    types = np.load(types_path)

    print(f"  Chargé: {len(images)} images")

    return images, masks, types


def filter_by_family(images, masks, types, family: str):
    """
    Filtre les images par famille d'organes.

    Args:
        images: (N, 256, 256, 3)
        masks: (N, 256, 256, 6)
        types: (N,) organ names
        family: 'glandular', 'digestive', etc.

    Returns:
        Filtered (images, masks, types, indices)
    """
    # Organes de cette famille
    family_organs = [organ for organ, fam in ORGAN_TO_FAMILY.items() if fam == family]

    # Indices des images de cette famille
    indices = [i for i, organ in enumerate(types) if organ in family_organs]

    print(f"\nFamille '{family}':")
    print(f"  Organes: {family_organs}")
    print(f"  Échantillons trouvés: {len(indices)}")

    return images[indices], masks[indices], types[indices], indices


def extract_gt_instances(mask: np.ndarray) -> np.ndarray:
    """
    Extrait instances GT depuis mask PanNuke (méthode FIXED).

    IMPORTANT: Utilise les IDs natifs PanNuke (canaux 1-4), PAS connectedComponents.

    Args:
        mask: (256, 256, 6) - [background, type1, type2, type3, type4, type5]

    Returns:
        inst_map: (256, 256) avec IDs instances
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
    epithelial_binary = (mask[:, :, 5] > 0).astype(np.uint8)
    if epithelial_binary.sum() > 0:
        _, epithelial_labels = cv2.connectedComponents(epithelial_binary)
        epithelial_ids = np.unique(epithelial_labels)
        epithelial_ids = epithelial_ids[epithelial_ids > 0]

        for inst_id in epithelial_ids:
            inst_mask = epithelial_labels == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    return inst_map


def watershed_from_hv(np_pred: np.ndarray, hv_pred: np.ndarray) -> np.ndarray:
    """
    Watershed depuis HV maps.

    Args:
        np_pred: (256, 256) binary [0, 1]
        hv_pred: (2, 256, 256) float32 [-1, 1]

    Returns:
        inst_map: (256, 256) instance IDs
    """
    # Binariser NP
    np_binary = (np_pred > 0.5).astype(np.uint8)

    if np_binary.sum() == 0:
        return np.zeros_like(np_binary, dtype=np.int32)

    # Distance transform pour markers
    dist = ndimage.distance_transform_edt(np_binary)

    # Markers = local maxima
    if dist.max() > 0:
        dist_norm = dist / dist.max()
        markers_mask = dist_norm > 0.5
        markers, _ = ndimage.label(markers_mask)

        if markers.max() == 0:
            # Fallback: connected components
            _, inst_map = cv2.connectedComponents(np_binary)
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
        _, inst_map = cv2.connectedComponents(np_binary)
        return inst_map.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Évaluation AJI sur images brutes PanNuke")
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help="Checkpoint HoVer-Net (e.g., models/checkpoints/hovernet_epidermal_best.pth)")
    parser.add_argument('--data_dir', type=Path, required=True,
                        help="Répertoire PanNuke (e.g., /home/amar/data/PanNuke)")
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2],
                        help="Fold PanNuke à tester (défaut: 2 = jamais vu)")
    parser.add_argument('--n_samples', type=int, default=20,
                        help="Nombre d'échantillons à tester")
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Load PanNuke data
    images, masks, types = load_pannuke_data(args.data_dir, args.fold)

    # Filter by family
    images_fam, masks_fam, types_fam, indices_fam = filter_by_family(
        images, masks, types, args.family
    )

    # Limit samples
    n_test = min(args.n_samples, len(images_fam))
    images_fam = images_fam[:n_test]
    masks_fam = masks_fam[:n_test]
    types_fam = types_fam[:n_test]

    print(f"\n{'='*70}")
    print(f"Test sur {n_test} échantillons {args.family} (fold {args.fold})")
    print(f"{'='*70}\n")

    # Load models
    print("Chargement modèles...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)

    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    print(f"✅ Backbone: H-optimus-0")
    print(f"✅ HoVer-Net: {args.checkpoint.name}")

    # Evaluate
    ajis = []
    dices = []
    n_instances_gt = []
    n_instances_pred = []

    print(f"\nÉvaluation...")
    for i in tqdm(range(n_test), desc="Évaluation"):
        image = images_fam[i]  # (256, 256, 3) uint8
        mask = masks_fam[i]    # (256, 256, 6)

        # 1. Preprocessing (EXACTEMENT comme training)
        tensor = preprocess_image(image, device=args.device)  # (1, 3, 224, 224)

        # 2. Features extraction
        with torch.no_grad():
            features = backbone.forward_features(tensor)  # (1, 261, 1536)

        # 3. Validation CLS std
        try:
            validate_features(features)
        except ValueError as e:
            print(f"\n⚠️ Warning image {i}: {e}")
            continue

        # 4. HoVer-Net prediction
        with torch.no_grad():
            np_out, hv_out, nt_out = hovernet(features)

        # 5. To numpy (EXACTEMENT comme training)
        np_pred_logits = np_out.cpu().numpy()[0]  # (2, 224, 224)
        np_pred_224 = (np_pred_logits.argmax(axis=0)).astype(np.float32)  # (224, 224) [0, 1]

        hv_pred_224 = hv_out.cpu().numpy()[0]  # (2, 224, 224)

        # 6. Resize 224 → 256 (pour comparer avec GT)
        np_pred = cv2.resize(np_pred_224, (256, 256), interpolation=cv2.INTER_NEAREST)
        hv_pred = np.stack([
            cv2.resize(hv_pred_224[0], (256, 256), interpolation=cv2.INTER_LINEAR),
            cv2.resize(hv_pred_224[1], (256, 256), interpolation=cv2.INTER_LINEAR)
        ], axis=0)

        # 7. Watershed
        inst_pred = watershed_from_hv(np_pred, hv_pred)

        # 8. GT instances (méthode FIXED)
        inst_gt = extract_gt_instances(mask)

        # 9. Metrics
        aji = compute_aji(inst_pred, inst_gt)
        dice = compute_dice(np_pred, (inst_gt > 0).astype(np.float32))

        ajis.append(aji)
        dices.append(dice)
        n_instances_gt.append(len(np.unique(inst_gt)) - 1)  # -1 pour enlever background
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
