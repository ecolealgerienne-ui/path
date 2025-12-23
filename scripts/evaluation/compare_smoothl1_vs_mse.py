#!/usr/bin/env python3
"""
Compare les performances SmoothL1Loss vs MSE sur famille de test.

Usage:
    python scripts/evaluation/compare_smoothl1_vs_mse.py \
        --family epidermal \
        --checkpoint_smoothl1 models/checkpoints/hovernet_epidermal_best.pth \
        --checkpoint_mse models/checkpoints/hovernet_epidermal_mse_test.pth \
        --n_samples 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader


def compute_instance_metrics(np_pred, hv_pred, nt_pred, np_gt, hv_gt, nt_gt):
    """
    Calcule AJI, PQ approximatifs (version simplifiée).
    Pour une évaluation complète, utiliser evaluate_ground_truth.py
    """
    # Approximation AJI: compter les pixels bien classés dans les zones de noyaux
    np_pred_binary = (np_pred > 0.5).astype(np.uint8)
    np_gt_binary = (np_gt > 0.5).astype(np.uint8)

    # Intersection et union
    intersection = np.logical_and(np_pred_binary, np_gt_binary).sum()
    union = np.logical_or(np_pred_binary, np_gt_binary).sum()

    # AJI approximatif (IoU binaire comme proxy)
    aji_approx = intersection / (union + 1e-8)

    # Dice
    dice = 2 * intersection / (np_pred_binary.sum() + np_gt_binary.sum() + 1e-8)

    # HV MSE (sur pixels de noyaux uniquement)
    mask = np_gt_binary.astype(bool)
    if mask.sum() > 0:
        hv_mse = ((hv_pred[:, mask] - hv_gt[:, mask]) ** 2).mean()
    else:
        hv_mse = 0.0

    # NT Accuracy
    if mask.sum() > 0:
        nt_pred_labels = nt_pred.argmax(axis=0)[mask]
        nt_gt_labels = nt_gt[mask]
        nt_acc = (nt_pred_labels == nt_gt_labels).mean()
    else:
        nt_acc = 0.0

    return {
        'aji_approx': aji_approx,
        'dice': dice,
        'hv_mse': hv_mse,
        'nt_acc': nt_acc,
    }


def load_model(checkpoint_path, device='cuda'):
    """Charge un checkpoint HoVer-Net."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = HoVerNetDecoder(embed_dim=1536, n_classes=5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def evaluate_model(model, data_loader, device='cuda'):
    """Évalue un modèle sur un dataset."""
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            features = batch['features'].to(device)
            np_target = batch['np_target'].numpy()[0]
            hv_target = batch['hv_target'].numpy()[0]
            nt_target = batch['nt_target'].numpy()[0]

            # Forward
            np_out, hv_out, nt_out = model(features)

            # To numpy
            np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]
            hv_pred = hv_out.cpu().numpy()[0]
            nt_pred = torch.softmax(nt_out, dim=1).cpu().numpy()[0]

            # Compute metrics
            metrics = compute_instance_metrics(
                np_pred, hv_pred, nt_pred,
                np_target, hv_target, nt_target
            )
            all_metrics.append(metrics)

    # Aggregate
    return {
        'aji_approx': np.mean([m['aji_approx'] for m in all_metrics]),
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'hv_mse': np.mean([m['hv_mse'] for m in all_metrics]),
        'nt_acc': np.mean([m['nt_acc'] for m in all_metrics]),
        'aji_std': np.std([m['aji_approx'] for m in all_metrics]),
        'dice_std': np.std([m['dice'] for m in all_metrics]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--checkpoint_smoothl1', type=Path, required=True)
    parser.add_argument('--checkpoint_mse', type=Path, required=True)
    parser.add_argument('--n_samples', type=int, default=20,
                        help="Nombre d'échantillons de test")
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"COMPARAISON SMOOTHL1 vs MSE - Famille {args.family.upper()}")
    print(f"{'='*70}\n")

    # Load data (TODO: implement actual data loading)
    # For now, this is a placeholder
    print("⚠️  Ce script nécessite un DataLoader pour la famille de test")
    print("    Utiliser scripts/evaluation/test_family_models_isolated.py pour évaluation complète\n")

    # Load models
    print("📥 Chargement modèles...")
    model_smoothl1 = load_model(args.checkpoint_smoothl1, args.device)
    model_mse = load_model(args.checkpoint_mse, args.device)
    print("   ✅ Modèles chargés\n")

    # Evaluate (placeholder)
    print("🔍 Évaluation SmoothL1Loss (baseline)...")
    # metrics_smoothl1 = evaluate_model(model_smoothl1, test_loader, args.device)

    print("🔍 Évaluation MSE Loss (test)...")
    # metrics_mse = evaluate_model(model_mse, test_loader, args.device)

    # Compare
    print(f"\n{'='*70}")
    print("RÉSULTATS COMPARATIFS")
    print(f"{'='*70}\n")

    # TODO: Print actual comparison
    print("Métrique         | SmoothL1 (baseline) | MSE (test)  | Amélioration")
    print("-" * 70)
    # print(f"AJI (approx)     | {metrics_smoothl1['aji_approx']:.4f}          | {metrics_mse['aji_approx']:.4f}     | {improvement_aji:+.1%}")
    # print(f"Dice             | {metrics_smoothl1['dice']:.4f}          | {metrics_mse['dice']:.4f}     | {improvement_dice:+.1%}")
    # print(f"HV MSE           | {metrics_smoothl1['hv_mse']:.4f}          | {metrics_mse['hv_mse']:.4f}     | {improvement_hv:+.1%}")

    print("\n💡 Pour évaluation complète avec AJI réel, utiliser:")
    print("   python scripts/evaluation/evaluate_ground_truth.py \\")
    print(f"       --checkpoint {args.checkpoint_mse} \\")
    print("       --dataset consep")


if __name__ == '__main__':
    main()
