#!/usr/bin/env python3
"""
Comparaison V12 (Resize) vs V13 (Multi-Crop Statique).

Évalue les deux versions sur le même test set et génère un rapport comparatif.

Métriques comparées:
- AJI (Aggregated Jaccard Index) - Métrique primaire
- Dice (Overlap global)
- PQ (Panoptic Quality)

Usage:
    python scripts/evaluation/compare_v12_v13.py \
        --family epidermal \
        --v12_checkpoint models/checkpoints/hovernet_epidermal_best.pth \
        --v13_checkpoint models/checkpoints_v13/hovernet_epidermal_v13_best.pth \
        --n_samples 50 \
        --output_dir results/v12_vs_v13
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE
from src.metrics.ground_truth_metrics import compute_aji, compute_pq
from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform


def post_process_predictions(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    min_size: int = 10,
    dist_threshold: float = 0.4,
    edge_threshold: float = 0.5
) -> np.ndarray:
    """
    Post-processing HoVer-Net: NP + HV → instances séparées via watershed.

    Args:
        np_pred: Nuclear Presence (H, W) float [0, 1]
        hv_pred: HV maps (2, H, W) float [-1, 1]
        min_size: Taille minimale d'instance
        dist_threshold: Seuil pour distance transform
        edge_threshold: Seuil pour détection frontières HV

    Returns:
        Instance map (H, W) int32
    """
    # 1. Binariser NP
    binary_mask = (np_pred > 0.5).astype(np.uint8)

    # 2. Calculer magnitude HV (détection frontières)
    h_map = hv_pred[0]
    v_map = hv_pred[1]
    hv_magnitude = np.sqrt(h_map ** 2 + v_map ** 2)

    # 3. Marqueurs: distance transform (centres noyaux)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    _, markers = cv2.connectedComponents((dist_transform > dist_threshold * dist_transform.max()).astype(np.uint8))

    # 4. Watershed avec frontières HV
    edges = (hv_magnitude > edge_threshold).astype(np.uint8)
    watershed_input = binary_mask.copy()
    watershed_input[edges > 0] = 0  # Supprimer frontières HV

    inst_map = cv2.watershed(
        cv2.merge([watershed_input]*3),
        markers.astype(np.int32)
    )

    # 5. Nettoyer
    inst_map[inst_map == -1] = 0  # Supprimer frontières watershed
    inst_map = inst_map.astype(np.int32)

    # Filtrer petites instances
    for label in np.unique(inst_map):
        if label == 0:
            continue
        if (inst_map == label).sum() < min_size:
            inst_map[inst_map == label] = 0

    # Réindexer
    unique_labels = np.unique(inst_map)
    unique_labels = unique_labels[unique_labels > 0]
    new_inst_map = np.zeros_like(inst_map)

    for new_id, old_id in enumerate(unique_labels, start=1):
        new_inst_map[inst_map == old_id] = new_id

    return new_inst_map


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice score."""
    intersection = (pred > 0) & (target > 0)
    union = (pred > 0).sum() + (target > 0).sum()

    if union == 0:
        return 1.0

    return 2.0 * intersection.sum() / union


def evaluate_model(
    model: torch.nn.Module,
    backbone: torch.nn.Module,
    images: np.ndarray,
    gt_inst_maps: np.ndarray,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Évalue un modèle sur un set d'images.

    Args:
        model: HoVerNetDecoder
        backbone: H-optimus-0
        images: Images (N, 256, 256, 3) uint8
        gt_inst_maps: Ground truth instance maps (N, 256, 256) int32
        device: Device

    Returns:
        Métriques moyennes
    """
    model.eval()
    backbone.eval()

    transform = create_hoptimus_transform()

    all_dice = []
    all_aji = []
    all_pq = []

    with torch.no_grad():
        for i in tqdm(range(len(images)), desc="Évaluation"):
            # 1. Extraire features H-optimus-0
            image = images[i]
            gt_inst = gt_inst_maps[i]

            # Center crop 224×224 (pour match avec V13)
            h, w = image.shape[:2]
            crop_h, crop_w = 224, 224
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            image_224 = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
            gt_inst_224 = gt_inst[start_h:start_h+crop_h, start_w:start_w+crop_w]

            # Transform
            tensor = transform(image_224).unsqueeze(0).to(device)
            features = backbone.forward_features(tensor)

            # 2. Inférence HoVer-Net
            np_out, hv_out, nt_out = model(features)

            # Convertir en numpy
            np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 1]  # (224, 224)
            hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

            # 3. Post-processing
            pred_inst = post_process_predictions(np_pred, hv_pred)

            # 4. Métriques
            if gt_inst_224.max() > 0:  # Au moins 1 instance GT
                dice = compute_dice(pred_inst, gt_inst_224)
                aji = compute_aji(pred_inst, gt_inst_224)
                pq = compute_pq(pred_inst, gt_inst_224)

                all_dice.append(dice)
                all_aji.append(aji)
                all_pq.append(pq)

    return {
        'dice': np.mean(all_dice),
        'aji': np.mean(all_aji),
        'pq': np.mean(all_pq),
        'dice_std': np.std(all_dice),
        'aji_std': np.std(all_aji),
        'pq_std': np.std(all_pq),
        'n_samples': len(all_dice),
    }


def load_test_samples(
    family: str,
    n_samples: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge échantillons de test depuis données PanNuke.

    Args:
        family: Famille tissulaire
        n_samples: Nombre d'échantillons

    Returns:
        (images, gt_inst_maps)
    """
    # Charger depuis V12 COHERENT data (source de vérité)
    data_path = Path(f"data/family_FIXED/{family}_data_FIXED_v12_COHERENT.npz")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Données V12 COHERENT non trouvées: {data_path}\n"
            f"Lancez d'abord:\n"
            f"  python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py --family {family}"
        )

    print(f"\n📂 Chargement test samples: {data_path}")
    data = np.load(data_path)

    images = data['images']
    np_targets = data['np_targets']

    # Créer instance maps depuis NP targets
    n_total = len(images)
    n_test = min(n_samples, n_total)

    # Prendre échantillons aléatoires (seed fixe pour reproductibilité)
    np.random.seed(42)
    test_indices = np.random.choice(n_total, n_test, replace=False)

    test_images = images[test_indices]
    test_np_targets = np_targets[test_indices]

    # Convertir NP → instance maps
    gt_inst_maps = []
    for np_target in test_np_targets:
        binary_mask = (np_target > 0.5).astype(np.uint8)
        inst_map, _ = ndimage.label(binary_mask)
        gt_inst_maps.append(inst_map.astype(np.int32))

    gt_inst_maps = np.array(gt_inst_maps)

    print(f"✅ {n_test} échantillons chargés")

    return test_images, gt_inst_maps


def generate_comparison_report(
    v12_metrics: Dict[str, float],
    v13_metrics: Dict[str, float],
    output_path: Path
):
    """Génère rapport comparatif V12 vs V13."""
    report = []

    report.append("=" * 70)
    report.append("RAPPORT COMPARATIF V12 (Resize) vs V13 (Multi-Crop)")
    report.append("=" * 70)
    report.append("")

    report.append(f"Échantillons testés: {v12_metrics['n_samples']}")
    report.append("")

    # Tableau comparatif
    report.append("╔═══════════════════════════════════════════════════════════════════╗")
    report.append("║                     MÉTRIQUES COMPARATIVES                        ║")
    report.append("╠════════════════╦══════════════════╦══════════════════╦════════════╣")
    report.append("║    Métrique    ║   V12 (Resize)   ║ V13 (Multi-Crop) ║   Gain     ║")
    report.append("╠════════════════╬══════════════════╬══════════════════╬════════════╣")

    for metric in ['dice', 'aji', 'pq']:
        v12_val = v12_metrics[metric]
        v13_val = v13_metrics[metric]
        gain = ((v13_val - v12_val) / v12_val * 100) if v12_val > 0 else 0

        v12_std = v12_metrics.get(f'{metric}_std', 0)
        v13_std = v13_metrics.get(f'{metric}_std', 0)

        metric_name = metric.upper()
        v12_str = f"{v12_val:.4f} ± {v12_std:.4f}"
        v13_str = f"{v13_val:.4f} ± {v13_std:.4f}"
        gain_str = f"{gain:+.2f}%"

        # Color code gain
        if gain > 0:
            gain_indicator = "✅"
        elif gain < 0:
            gain_indicator = "❌"
        else:
            gain_indicator = "="

        report.append(f"║ {metric_name:14s} ║ {v12_str:16s} ║ {v13_str:16s} ║ {gain_str:8s} {gain_indicator} ║")

    report.append("╚════════════════╩══════════════════╩══════════════════╩════════════╝")
    report.append("")

    # Verdict
    aji_gain = ((v13_metrics['aji'] - v12_metrics['aji']) / v12_metrics['aji'] * 100) if v12_metrics['aji'] > 0 else 0

    report.append("VERDICT:")
    if v13_metrics['aji'] >= 0.43:  # Objectif Epidermal
        report.append(f"✅ OBJECTIF ATTEINT - AJI V13: {v13_metrics['aji']:.4f} ≥ 0.43")
    else:
        report.append(f"⚠️  OBJECTIF NON ATTEINT - AJI V13: {v13_metrics['aji']:.4f} < 0.43")

    if aji_gain > 0:
        report.append(f"✅ AMÉLIORATION - Multi-Crop apporte un gain de {aji_gain:+.2f}% sur AJI")
    else:
        report.append(f"❌ RÉGRESSION - Multi-Crop n'améliore pas les performances ({aji_gain:.2f}%)")

    report_text = "\n".join(report)

    # Afficher
    print(f"\n{report_text}")

    # Sauvegarder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\n✅ Rapport sauvegardé: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comparaison V12 (Resize) vs V13 (Multi-Crop)"
    )
    parser.add_argument(
        '--family',
        type=str,
        required=True,
        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
        help="Famille tissulaire"
    )
    parser.add_argument(
        '--v12_checkpoint',
        type=Path,
        required=True,
        help="Checkpoint V12"
    )
    parser.add_argument(
        '--v13_checkpoint',
        type=Path,
        required=True,
        help="Checkpoint V13"
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=50,
        help="Nombre d'échantillons de test"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results/v12_vs_v13'),
        help="Répertoire de sortie"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"COMPARAISON V12 vs V13 - Famille: {args.family.upper()}")
    print(f"{'='*70}\n")

    # Charger backbone H-optimus-0
    print("🔧 Chargement H-optimus-0...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)

    # Charger modèles V12 et V13
    print("\n🔧 Chargement modèles...")

    v12_model = HoVerNetDecoder(embed_dim=1536, n_classes=2, dropout=0.4)
    v12_checkpoint = torch.load(args.v12_checkpoint, map_location=args.device)
    v12_model.load_state_dict(v12_checkpoint['model_state_dict'])
    v12_model.to(args.device)
    print(f"  ✅ V12 chargé: {args.v12_checkpoint}")

    v13_model = HoVerNetDecoder(embed_dim=1536, n_classes=2, dropout=0.4)
    v13_checkpoint = torch.load(args.v13_checkpoint, map_location=args.device)
    v13_model.load_state_dict(v13_checkpoint['model_state_dict'])
    v13_model.to(args.device)
    print(f"  ✅ V13 chargé: {args.v13_checkpoint}")

    # Charger test samples
    test_images, gt_inst_maps = load_test_samples(args.family, args.n_samples)

    # Évaluer V12
    print("\n📊 Évaluation V12 (Resize)...")
    v12_metrics = evaluate_model(
        v12_model, backbone, test_images, gt_inst_maps, args.device
    )

    # Évaluer V13
    print("\n📊 Évaluation V13 (Multi-Crop)...")
    v13_metrics = evaluate_model(
        v13_model, backbone, test_images, gt_inst_maps, args.device
    )

    # Générer rapport
    output_file = args.output_dir / f"comparison_{args.family}.txt"
    generate_comparison_report(v12_metrics, v13_metrics, output_file)

    print(f"\n{'='*70}")
    print(f"✅ COMPARAISON COMPLÈTE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
