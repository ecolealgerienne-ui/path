#!/usr/bin/env python3
"""
Évaluation AJI rapide pour comparer SmoothL1 vs MSE loss.

Usage:
    python scripts/evaluation/quick_aji_test.py \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
        --family epidermal \
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE
from src.metrics.ground_truth_metrics import compute_aji, compute_dice
from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform


def get_gradient_hv(hv_map: np.ndarray) -> np.ndarray:
    """
    Calcule la magnitude des gradients HV (Sobel 5×5).

    Args:
        hv_map: (2, H, W) - Horizontal et Vertical maps [-1, 1]

    Returns:
        gradient_mag: (H, W) - Magnitude des gradients
    """
    h_map = hv_map[0]
    v_map = hv_map[1]

    # Sobel 5×5 pour gradients plus lisses
    sobelh_h = cv2.Sobel(h_map, cv2.CV_64F, 1, 0, ksize=5)
    sobelv_h = cv2.Sobel(h_map, cv2.CV_64F, 0, 1, ksize=5)

    sobelh_v = cv2.Sobel(v_map, cv2.CV_64F, 1, 0, ksize=5)
    sobelv_v = cv2.Sobel(v_map, cv2.CV_64F, 0, 1, ksize=5)

    # Magnitude totale des gradients
    gradient_mag = np.sqrt(sobelh_h**2 + sobelv_h**2 + sobelh_v**2 + sobelv_v**2)

    return gradient_mag


def remove_small_objects(binary_mask: np.ndarray, min_size: int = 10) -> np.ndarray:
    """Supprime les objets plus petits que min_size pixels."""
    labeled, num = ndimage.label(binary_mask)

    # Si aucun objet détecté, retourner masque vide
    if num == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    sizes = ndimage.sum(binary_mask, labeled, range(1, num + 1))

    mask_size = sizes >= min_size
    remove_small = mask_size[labeled - 1]
    remove_small[labeled == 0] = 0

    return remove_small.astype(np.uint8)


def watershed_instance_extraction(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    min_size: int = 10,
    edge_threshold: float = 0.5,
    dist_threshold: float = 0.5
) -> np.ndarray:
    """
    Extrait les instances avec watershed basé sur les gradients HV.

    Args:
        np_pred: (H, W) - Nuclear Presence mask [0, 1]
        hv_pred: (2, H, W) - HV maps [-1, 1]
        min_size: Taille minimale d'instance (pixels)
        edge_threshold: Seuil pour détection frontières (gradient magnitude)
        dist_threshold: Seuil pour markers (distance transform)

    Returns:
        inst_map: (H, W) - Instance map (0=bg, 1..N=instances)
    """
    # 1. Binariser NP mask
    binary_mask = (np_pred > 0.5).astype(np.uint8)

    # Remove small noise
    binary_mask = remove_small_objects(binary_mask, min_size=min_size)

    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    # 2. Calculer gradients HV (détection frontières)
    gradient_mag = get_gradient_hv(hv_pred)
    gradient_mag = (gradient_mag - gradient_mag.min()) / (gradient_mag.max() - gradient_mag.min() + 1e-8)

    # 3. Créer markers (centres de cellules)
    # Distance transform sur le masque binaire
    dist_transform = ndimage.distance_transform_edt(binary_mask)

    # Normaliser distance transform
    if dist_transform.max() > 0:
        dist_transform = dist_transform / dist_transform.max()

    # Markers = peaks de distance transform
    markers_mask = dist_transform > dist_threshold
    markers, num_markers = ndimage.label(markers_mask)

    if num_markers == 0:
        # Fallback: utiliser connected components sur masque binaire
        inst_map, _ = cv2.connectedComponents(binary_mask)
        return inst_map.astype(np.int32)

    # 4. Watershed avec gradient comme "bassin"
    # Créer une "énergie" haute aux frontières
    energy = (1 - gradient_mag) * binary_mask
    energy = (energy * 255).astype(np.uint8)

    # Watershed OpenCV
    markers_watershed = markers.astype(np.int32)
    cv2.watershed(cv2.cvtColor(energy, cv2.COLOR_GRAY2BGR), markers_watershed)

    # Remove background (-1) et borders (-1)
    inst_map = markers_watershed.copy()
    inst_map[inst_map == -1] = 0  # Borders → background
    inst_map[binary_mask == 0] = 0  # Forcer background hors du masque

    return inst_map


def load_pannuke_samples(data_dir: Path, family: str, fold: int = 2, n_samples: int = 20):
    """
    Charge quelques échantillons PanNuke pour évaluation.

    Args:
        data_dir: Répertoire PanNuke
        family: Famille d'organes
        fold: Fold PanNuke (0, 1, ou 2)
        n_samples: Nombre d'échantillons à charger

    Returns:
        List of dicts avec 'image', 'mask', 'organ'
    """
    from src.models.organ_families import ORGAN_TO_FAMILY

    # Mapping inverse famille → organes
    family_organs = [organ for organ, fam in ORGAN_TO_FAMILY.items() if fam == family]

    # Charger fold
    fold_dir = data_dir / f"fold{fold}"
    images_path = fold_dir / "images.npy"
    masks_path = fold_dir / "masks.npy"
    types_path = fold_dir / "types.npy"

    if not images_path.exists():
        raise FileNotFoundError(f"PanNuke fold {fold} non trouvé: {images_path}")

    print(f"📥 Chargement PanNuke fold {fold}...")
    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')
    types = np.load(types_path)

    # Filtrer par famille
    samples = []
    for i, organ in enumerate(types):
        if organ in family_organs:
            samples.append({
                'image': images[i].copy(),
                'mask': masks[i].copy(),
                'organ': organ,
                'index': i
            })

        if len(samples) >= n_samples:
            break

    print(f"   ✅ {len(samples)} échantillons chargés ({', '.join(set(s['organ'] for s in samples))})")

    return samples


def evaluate_model_aji(
    model: HoVerNetDecoder,
    backbone: torch.nn.Module,
    samples: list,
    device: str = 'cuda'
) -> dict:
    """
    Évalue AJI sur un ensemble d'échantillons.

    Returns:
        dict avec métriques moyennes
    """
    model.eval()
    backbone.eval()

    all_aji = []
    all_dice = []
    all_n_instances_gt = []
    all_n_instances_pred = []

    transform = create_hoptimus_transform()

    with torch.no_grad():
        for sample in tqdm(samples, desc="Évaluation AJI"):
            # Preprocessing image
            image = sample['image']  # (256, 256, 3) uint8
            mask = sample['mask']    # (256, 256, 6) int32

            # H-optimus-0 features
            tensor = transform(image).unsqueeze(0).to(device)
            features = backbone.forward_features(tensor)  # (1, 261, 1536)

            # HoVer-Net inference
            np_out, hv_out, nt_out = model(features)

            # To numpy
            np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]  # (224, 224)
            hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

            # Resize predictions to match GT (256×256)
            from src.utils.image_utils import prepare_predictions_for_evaluation
            np_pred_resized, hv_pred_resized, _ = prepare_predictions_for_evaluation(
                np_pred, hv_pred, np.zeros((5, 224, 224)),  # nt_pred pas utilisé pour AJI
                target_size=PANNUKE_IMAGE_SIZE
            )

            # Extract instances avec watershed
            inst_pred = watershed_instance_extraction(
                np_pred_resized,
                hv_pred_resized,
                min_size=10,
                edge_threshold=0.5,
                dist_threshold=0.5
            )

            # Ground truth instances
            # PanNuke: canaux 1-5 contiennent les IDs d'instances par classe
            inst_gt = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int32)
            instance_counter = 1

            for c in range(1, 6):  # Canaux 1-5
                class_instances = mask[:, :, c]
                inst_ids = np.unique(class_instances)
                inst_ids = inst_ids[inst_ids > 0]

                for inst_id in inst_ids:
                    inst_mask = class_instances == inst_id
                    inst_gt[inst_mask] = instance_counter
                    instance_counter += 1

            # Compute AJI
            aji = compute_aji(inst_pred, inst_gt)
            all_aji.append(aji)

            # Compute Dice (binary)
            np_gt = mask[:, :, 1:].sum(axis=-1) > 0
            dice = compute_dice(np_pred_resized > 0.5, np_gt)
            all_dice.append(dice)

            # Count instances
            all_n_instances_gt.append(len(np.unique(inst_gt)) - 1)  # -1 pour background
            all_n_instances_pred.append(len(np.unique(inst_pred)) - 1)

    return {
        'aji_mean': np.mean(all_aji),
        'aji_std': np.std(all_aji),
        'dice_mean': np.mean(all_dice),
        'dice_std': np.std(all_dice),
        'n_instances_gt_mean': np.mean(all_n_instances_gt),
        'n_instances_pred_mean': np.mean(all_n_instances_pred),
        'all_aji': all_aji,
    }


def main():
    parser = argparse.ArgumentParser(description="Évaluation AJI rapide MSE vs SmoothL1")
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help="Checkpoint HoVer-Net à évaluer")
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--data_dir', type=Path, default=Path('/home/amar/data/PanNuke'),
                        help="Répertoire PanNuke")
    parser.add_argument('--n_samples', type=int, default=20,
                        help="Nombre d'échantillons à évaluer")
    parser.add_argument('--fold', type=int, default=2,
                        help="Fold PanNuke (0, 1, ou 2)")
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"ÉVALUATION AJI - Famille {args.family.upper()}")
    print(f"{'='*70}\n")

    # Load model
    print("📥 Chargement modèle...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    backbone = ModelLoader.load_hoptimus0(device=args.device)

    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet = hovernet.to(args.device)
    hovernet.eval()

    print(f"   ✅ Modèle chargé (epoch {checkpoint.get('epoch', '?')})\n")

    # Load samples
    samples = load_pannuke_samples(
        args.data_dir,
        args.family,
        fold=args.fold,
        n_samples=args.n_samples
    )

    # Evaluate
    print(f"\n🔍 Évaluation sur {len(samples)} échantillons...")
    metrics = evaluate_model_aji(hovernet, backbone, samples, args.device)

    # Print results
    print(f"\n{'='*70}")
    print("RÉSULTATS")
    print(f"{'='*70}\n")

    print(f"AJI:                {metrics['aji_mean']:.4f} ± {metrics['aji_std']:.4f}")
    print(f"Dice (binary):      {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"Instances GT:       {metrics['n_instances_gt_mean']:.1f} par image")
    print(f"Instances Pred:     {metrics['n_instances_pred_mean']:.1f} par image")

    print(f"\nDistribution AJI:")
    print(f"  Min:  {min(metrics['all_aji']):.4f}")
    print(f"  Q25:  {np.percentile(metrics['all_aji'], 25):.4f}")
    print(f"  Médiane: {np.median(metrics['all_aji']):.4f}")
    print(f"  Q75:  {np.percentile(metrics['all_aji'], 75):.4f}")
    print(f"  Max:  {max(metrics['all_aji']):.4f}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRÉTATION")
    print(f"{'='*70}\n")

    aji = metrics['aji_mean']

    if aji >= 0.60:
        print("✅ EXCELLENT - AJI ≥0.60")
        print("   → Séparation d'instances de qualité comparable à HoVer-Net original")
        print("   → RECOMMANDATION: Ré-entraîner toutes les familles avec MSE loss")
    elif aji >= 0.40:
        print("⚠️  BON - AJI 0.40-0.60")
        print("   → Amélioration par rapport à baseline attendue")
        print("   → RECOMMANDATION: Tester MSE + MSGE (Sobel 5×5) pour optimiser")
    elif aji >= 0.20:
        print("⚠️  MODÉRÉ - AJI 0.20-0.40")
        print("   → Amélioration partielle")
        print("   → RECOMMANDATION: Vérifier paramètres watershed, tester MSGE")
    else:
        print("❌ FAIBLE - AJI <0.20")
        print("   → Hypothèse MSE loss à reconsidérer")
        print("   → RECOMMANDATION: Analyser prédictions HV, watershed post-processing")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
