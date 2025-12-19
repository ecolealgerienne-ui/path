#!/usr/bin/env python3
"""
Métriques d'évaluation pour la segmentation cellulaire.

Métriques implémentées:
- Dice Score
- IoU (Intersection over Union)
- Panoptic Quality (PQ)
- F1-Score par classe

Usage:
    python scripts/evaluation/metrics_segmentation.py --pred pred.npy --gt gt.npy
"""

import argparse
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import ndimage


def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calcule le Dice Score.

    Args:
        pred: Masque prédit (binaire)
        gt: Masque ground truth (binaire)
        smooth: Terme de lissage

    Returns:
        Dice score [0, 1]
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()

    return (2 * intersection + smooth) / (union + smooth)


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calcule l'IoU (Intersection over Union).

    Args:
        pred: Masque prédit (binaire)
        gt: Masque ground truth (binaire)
        smooth: Terme de lissage

    Returns:
        IoU score [0, 1]
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    return (intersection + smooth) / (union + smooth)


def panoptic_quality(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    pred_classes: Optional[np.ndarray] = None,
    gt_classes: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcule le Panoptic Quality (PQ).

    PQ = SQ * RQ
    - SQ (Segmentation Quality): IoU moyen des vrais positifs
    - RQ (Recognition Quality): F1-score des instances

    Args:
        pred_instances: Masque d'instances prédites
        gt_instances: Masque d'instances ground truth
        pred_classes: Classes prédites par instance (optionnel)
        gt_classes: Classes ground truth par instance (optionnel)
        iou_threshold: Seuil IoU pour correspondance

    Returns:
        Dict avec PQ, SQ, RQ
    """
    pred_ids = np.unique(pred_instances)
    gt_ids = np.unique(gt_instances)

    # Ignorer le fond (0)
    pred_ids = pred_ids[pred_ids != 0]
    gt_ids = gt_ids[gt_ids != 0]

    # Matrice d'IoU
    matched_gt = set()
    matched_pred = set()
    iou_sum = 0.0

    for pred_id in pred_ids:
        pred_mask = pred_instances == pred_id
        best_iou = 0.0
        best_gt_id = None

        for gt_id in gt_ids:
            if gt_id in matched_gt:
                continue

            gt_mask = gt_instances == gt_id
            iou = iou_score(pred_mask, gt_mask)

            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        if best_iou >= iou_threshold:
            matched_gt.add(best_gt_id)
            matched_pred.add(pred_id)
            iou_sum += best_iou

    tp = len(matched_pred)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp

    # Calcul des métriques
    if tp == 0:
        sq = 0.0
        rq = 0.0
        pq = 0.0
    else:
        sq = iou_sum / tp
        rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        pq = sq * rq

    return {
        'PQ': pq,
        'SQ': sq,
        'RQ': rq,
        'TP': tp,
        'FP': fp,
        'FN': fn,
    }


def f1_per_class(
    pred: np.ndarray,
    gt: np.ndarray,
    n_classes: int = 5
) -> Dict[int, Dict[str, float]]:
    """
    Calcule F1-score par classe.

    Args:
        pred: Masque de classes prédites (H, W)
        gt: Masque de classes ground truth (H, W)
        n_classes: Nombre de classes

    Returns:
        Dict par classe avec precision, recall, f1
    """
    results = {}

    for c in range(1, n_classes + 1):  # Skip background (0)
        pred_c = pred == c
        gt_c = gt == c

        tp = np.logical_and(pred_c, gt_c).sum()
        fp = np.logical_and(pred_c, ~gt_c).sum()
        fn = np.logical_and(~pred_c, gt_c).sum()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        results[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    return results


def detection_metrics(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcule les métriques de détection d'instances.

    Args:
        pred_instances: Masque d'instances prédites
        gt_instances: Masque d'instances ground truth
        iou_threshold: Seuil IoU

    Returns:
        Dict avec precision, recall, f1
    """
    pq_result = panoptic_quality(pred_instances, gt_instances, iou_threshold=iou_threshold)

    tp = pq_result['TP']
    fp = pq_result['FP']
    fn = pq_result['FN']

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_pred': tp + fp,
        'n_gt': tp + fn,
    }


def evaluate_segmentation(
    pred: np.ndarray,
    gt: np.ndarray,
    n_classes: int = 5
) -> Dict:
    """
    Évaluation complète de la segmentation.

    Args:
        pred: Masque multi-canal prédit (H, W, C) ou (H, W)
        gt: Masque multi-canal ground truth

    Returns:
        Dict avec toutes les métriques
    """
    results = {}

    # Si multi-canal, calculer par canal
    if len(pred.shape) == 3:
        # Dice et IoU par classe
        for c in range(min(pred.shape[2], n_classes)):
            pred_c = pred[:, :, c] > 0
            gt_c = gt[:, :, c] > 0

            results[f'dice_class_{c}'] = dice_score(pred_c, gt_c)
            results[f'iou_class_{c}'] = iou_score(pred_c, gt_c)

        # Moyenne
        dice_values = [v for k, v in results.items() if 'dice_class' in k]
        iou_values = [v for k, v in results.items() if 'iou_class' in k]

        results['dice_mean'] = np.mean(dice_values)
        results['iou_mean'] = np.mean(iou_values)

    else:
        # Binaire
        results['dice'] = dice_score(pred > 0, gt > 0)
        results['iou'] = iou_score(pred > 0, gt > 0)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation de la segmentation cellulaire"
    )
    parser.add_argument("--pred", type=str, required=True, help="Prédiction (.npy)")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth (.npy)")
    parser.add_argument("--n-classes", type=int, default=5, help="Nombre de classes")

    args = parser.parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.gt)

    print(f"Pred shape: {pred.shape}")
    print(f"GT shape: {gt.shape}")

    results = evaluate_segmentation(pred, gt, args.n_classes)

    print("\n" + "="*40)
    print("Résultats:")
    print("="*40)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
