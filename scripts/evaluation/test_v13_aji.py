#!/usr/bin/env python3
"""
Test AJI isolé pour V13 Multi-Crop.

Évalue le modèle V13 sur des échantillons test avec visualisation
pour debugger le pipeline complet.

Usage:
    python scripts/evaluation/test_v13_aji.py \
        --checkpoint models/checkpoints_v13/hovernet_epidermal_v13_best.pth \
        --family epidermal \
        --n_samples 10
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE, ORGAN_TO_FAMILY
from src.metrics.ground_truth_metrics import compute_aji, compute_dice
from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform


def load_pannuke_test_samples(
    pannuke_dir: Path,
    family: str,
    fold: int = 2,
    n_samples: int = 10
) -> Dict[str, np.ndarray]:
    """
    Charge des échantillons test depuis PanNuke fold 2 (non utilisé pour training).

    Args:
        pannuke_dir: Répertoire PanNuke
        family: Famille tissulaire
        fold: Fold à utiliser (défaut: 2 = test)
        n_samples: Nombre d'échantillons

    Returns:
        Dict avec images, masks, types, organs
    """
    fold_dir = pannuke_dir / f"fold{fold}"

    print(f"\n📂 Chargement PanNuke fold {fold}...")
    images = np.load(fold_dir / "images.npy", mmap_mode='r')
    masks = np.load(fold_dir / "masks.npy", mmap_mode='r')
    types = np.load(fold_dir / "types.npy")

    # Filtrer par famille
    organs = [org for org, fam in ORGAN_TO_FAMILY.items() if fam == family]
    family_indices = [i for i, org in enumerate(types) if org in organs]

    print(f"  Organes {family}: {', '.join(organs)}")
    print(f"  Échantillons disponibles: {len(family_indices)}")

    # Sélectionner n_samples
    n_select = min(n_samples, len(family_indices))
    np.random.seed(42)
    selected = np.random.choice(family_indices, n_select, replace=False)

    print(f"  Sélectionnés: {n_select} échantillons\n")

    return {
        'images': images[selected],
        'masks': masks[selected],
        'types': types[selected],
        'indices': selected,
    }


def extract_ground_truth_instances(mask: np.ndarray) -> np.ndarray:
    """
    Extrait instance map depuis mask PanNuke (canaux 0-4).

    Args:
        mask: Mask PanNuke (256, 256, 6)

    Returns:
        Instance map (256, 256) avec IDs uniques
    """
    inst_map = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int32)
    instance_counter = 1

    # Canaux 0-4: instances déjà annotées
    for c in range(5):
        channel_instances = mask[:, :, c]
        inst_ids = np.unique(channel_instances)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_instances == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    return inst_map


def post_process_v13_predictions(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    min_size: int = 10,
    dist_threshold: float = 0.4,
    edge_threshold: float = 0.5
) -> np.ndarray:
    """
    Post-processing HoVer-Net V13 → instance map.

    Args:
        np_pred: Nuclear Presence (224, 224) - float [0, 1]
        hv_pred: HV maps (2, 224, 224) - float [-1, 1]
        min_size: Taille minimale instance (pixels)
        dist_threshold: Seuil distance transform
        edge_threshold: Seuil détection contours

    Returns:
        Instance map (224, 224) avec IDs uniques
    """
    # 1. Segmentation binaire
    binary_mask = (np_pred > 0.5).astype(np.uint8)

    # 2. Calcul gradient HV (magnitude)
    h_grad = hv_pred[0]  # (224, 224)
    v_grad = hv_pred[1]
    gradient_mag = np.sqrt(h_grad**2 + v_grad**2)

    # 3. Détection contours (gradient élevé)
    edges = (gradient_mag > edge_threshold).astype(np.uint8)

    # 4. Distance transform pour markers
    dist = ndimage.distance_transform_edt(binary_mask)
    local_max = (dist > dist_threshold)
    markers, _ = ndimage.label(local_max)

    # 5. Watershed
    watershed_input = -dist
    watershed_input[edges == 1] = 0  # Imposer contours

    instances = ndimage.watershed_ift(watershed_input.astype(np.int16), markers)
    instances = instances.astype(np.int32)
    instances[binary_mask == 0] = 0  # Remove background

    # 6. Filtrer petites instances
    for inst_id in np.unique(instances):
        if inst_id == 0:
            continue
        if (instances == inst_id).sum() < min_size:
            instances[inst_id == instances] = 0

    # 7. Ré-numéroter
    unique_ids = np.unique(instances)
    unique_ids = unique_ids[unique_ids > 0]

    inst_map = np.zeros_like(instances)
    for new_id, old_id in enumerate(unique_ids, start=1):
        inst_map[instances == old_id] = new_id

    return inst_map


def run_v13_inference(
    model: torch.nn.Module,
    backbone: torch.nn.Module,
    image: np.ndarray,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inférence V13: Image 256×256 → Crop central 224×224 → Prédictions.

    Args:
        model: HoVer-Net decoder V13
        backbone: H-optimus-0
        image: Image PanNuke (256, 256, 3) - uint8
        device: Device

    Returns:
        (np_pred, hv_pred) - Arrays (224, 224) et (2, 224, 224)
    """
    # 1. Crop central 224×224 (identique training V13)
    x1, y1 = 16, 16
    x2, y2 = 240, 240
    crop = image[y1:y2, x1:x2]  # (224, 224, 3)

    # 2. Preprocessing
    transform = create_hoptimus_transform()
    if crop.dtype != np.uint8:
        crop = crop.clip(0, 255).astype(np.uint8)

    tensor = transform(crop).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # 3. Extract features
    with torch.no_grad():
        features = backbone.forward_features(tensor)  # (1, 261, 1536)

    # 4. HoVer-Net decoder
    with torch.no_grad():
        np_out, hv_out, nt_out = model(features.float())

    # 5. Activations
    np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]  # (224, 224)
    hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

    return np_pred, hv_pred


def main():
    parser = argparse.ArgumentParser(description="Test AJI isolé V13")
    parser.add_argument('--checkpoint', type=Path, required=True, help="Checkpoint V13")
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--pannuke_dir', type=Path, default=Path('/home/amar/data/PanNuke'))
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=Path, default=Path('results/v13_aji_test'))
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("="*70)
    print("TEST AJI V13 - ÉVALUATION ISOLÉE")
    print("="*70)

    # 1. Charger données test
    test_data = load_pannuke_test_samples(
        args.pannuke_dir,
        args.family,
        fold=2,
        n_samples=args.n_samples
    )

    # 2. Charger modèles
    print("🔧 Chargement modèles...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    model = HoVerNetDecoder(embed_dim=1536, n_classes=2, dropout=0.0)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"  ✅ Checkpoint: {args.checkpoint.name}")
    print(f"  ✅ Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 3. Évaluation
    print(f"\n📊 Évaluation sur {len(test_data['images'])} échantillons...")

    all_dice = []
    all_aji = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, (image, mask) in enumerate(tqdm(
        zip(test_data['images'], test_data['masks']),
        total=len(test_data['images']),
        desc="Samples"
    )):
        # Image numpy array
        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask)

        # Ground Truth (crop central pour matcher prédiction)
        x1, y1 = 16, 16
        x2, y2 = 240, 240
        gt_inst_full = extract_ground_truth_instances(mask)  # (256, 256)
        gt_inst_crop = gt_inst_full[y1:y2, x1:x2]  # (224, 224)

        if gt_inst_crop.max() == 0:
            print(f"  ⚠️  Sample {i}: Pas d'instances GT - skip")
            continue

        # Prédiction
        np_pred, hv_pred = run_v13_inference(model, backbone, image, args.device)
        pred_inst = post_process_v13_predictions(np_pred, hv_pred)

        # Métriques
        dice = compute_dice(pred_inst, gt_inst_crop)
        aji = compute_aji(pred_inst, gt_inst_crop)

        all_dice.append(dice)
        all_aji.append(aji)

        # Visualisation (premiers 5 échantillons)
        if i < 5:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # Image originale (crop)
            axes[0].imshow(image[y1:y2, x1:x2])
            axes[0].set_title("Image (crop 224×224)")
            axes[0].axis('off')

            # GT instances
            axes[1].imshow(gt_inst_crop, cmap='tab20')
            axes[1].set_title(f"GT ({gt_inst_crop.max()} instances)")
            axes[1].axis('off')

            # Prédictions instances
            axes[2].imshow(pred_inst, cmap='tab20')
            axes[2].set_title(f"Pred ({pred_inst.max()} instances)")
            axes[2].axis('off')

            # Overlay
            overlay = image[y1:y2, x1:x2].copy()
            contours_gt, _ = cv2.findContours((gt_inst_crop > 0).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_pred, _ = cv2.findContours((pred_inst > 0).astype(np.uint8),
                                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours_gt, -1, (0, 255, 0), 1)  # Vert = GT
            cv2.drawContours(overlay, contours_pred, -1, (255, 0, 0), 1)  # Rouge = Pred
            axes[3].imshow(overlay)
            axes[3].set_title(f"Overlay (Dice: {dice:.3f}, AJI: {aji:.3f})")
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(args.output_dir / f"sample_{i:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()

    # 4. Rapport
    print("\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)
    print(f"Échantillons valides: {len(all_dice)}")
    print(f"Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"AJI:  {np.mean(all_aji):.4f} ± {np.std(all_aji):.4f}")
    print("="*70)

    # Sauvegarder résultats
    results_file = args.output_dir / f"results_{args.family}.txt"
    with open(results_file, 'w') as f:
        f.write(f"V13 AJI Test - {args.family.upper()}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Échantillons: {len(all_dice)}\n\n")
        f.write(f"Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}\n")
        f.write(f"AJI:  {np.mean(all_aji):.4f} ± {np.std(all_aji):.4f}\n")

    print(f"\n✅ Rapport: {results_file}")
    print(f"✅ Visualisations: {args.output_dir}/")


if __name__ == '__main__':
    main()
