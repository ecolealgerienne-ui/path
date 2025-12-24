#!/usr/bin/env python3
"""
Compare MSE vs SmoothL1Loss sur un batch réel.

Objectif: Déterminer si SmoothL1 produit des gradients plus faibles que MSE

Usage:
    python scripts/validation/compare_mse_vs_smoothl1.py \
        --data_file data/cache/family_data/glandular_targets.npz \
        --n_samples 100
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

# Ajouter le projet au path AVANT les imports src.*
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import module centralisé pour preprocessing
from src.data.preprocessing import load_targets


def compare_losses(hv_pred: torch.Tensor, hv_target: torch.Tensor, np_mask: torch.Tensor):
    """
    Compare MSE vs SmoothL1Loss sur un batch.

    Args:
        hv_pred: Prédictions HV (B, 2, H, W) float32 [-1, 1]
        hv_target: Targets HV (B, 2, H, W) float32 [-1, 1]
        np_mask: Masque noyaux (B, H, W) binary
    """
    mask = np_mask.float().unsqueeze(1)  # (B, 1, H, W)

    # Masquer pred et target
    hv_pred_masked = hv_pred * mask
    hv_target_masked = hv_target * mask

    # MSE (comme HoVer-Net)
    mse_loss = F.mse_loss(hv_pred_masked, hv_target_masked, reduction='sum')
    mse_loss = mse_loss / (mask.sum() * 2 + 1e-8)

    # SmoothL1Loss (notre système)
    smooth_l1_loss = F.smooth_l1_loss(hv_pred_masked, hv_target_masked, reduction='sum')
    smooth_l1_loss = smooth_l1_loss / (mask.sum() * 2 + 1e-8)

    # Calculer les gradients
    hv_pred.requires_grad = True

    # Backward MSE
    mse_loss_for_grad = F.mse_loss(hv_pred * mask, hv_target * mask, reduction='sum') / (mask.sum() * 2 + 1e-8)
    grad_mse = torch.autograd.grad(mse_loss_for_grad, hv_pred, retain_graph=True)[0]
    grad_mse_norm = grad_mse.norm().item()

    # Backward SmoothL1
    hv_pred.grad = None
    smooth_l1_loss_for_grad = F.smooth_l1_loss(hv_pred * mask, hv_target * mask, reduction='sum') / (mask.sum() * 2 + 1e-8)
    grad_smooth_l1 = torch.autograd.grad(smooth_l1_loss_for_grad, hv_pred)[0]
    grad_smooth_l1_norm = grad_smooth_l1.norm().item()

    return {
        'mse_loss': mse_loss.item(),
        'smooth_l1_loss': smooth_l1_loss.item(),
        'grad_mse_norm': grad_mse_norm,
        'grad_smooth_l1_norm': grad_smooth_l1_norm,
        'grad_ratio': grad_smooth_l1_norm / (grad_mse_norm + 1e-8),
    }


def generate_synthetic_batch(batch_size: int = 4, img_size: int = 224, n_nuclei: int = 50):
    """Génère un batch synthétique pour tests."""
    hv_pred = torch.randn(batch_size, 2, img_size, img_size) * 0.5
    hv_target = torch.randn(batch_size, 2, img_size, img_size) * 0.5

    # Créer masques de noyaux réalistes
    np_mask = torch.zeros(batch_size, img_size, img_size)
    for b in range(batch_size):
        for _ in range(n_nuclei):
            cx = np.random.randint(20, img_size - 20)
            cy = np.random.randint(20, img_size - 20)
            r = np.random.randint(5, 15)

            y, x = np.ogrid[:img_size, :img_size]
            mask = ((y - cy)**2 + (x - cx)**2) <= r**2
            np_mask[b, mask] = 1

    return hv_pred, hv_target, np_mask


def load_real_batch(data_file: Path, n_samples: int = 100):
    """Charge un batch réel depuis les données d'entraînement."""
    print(f"\n📥 Chargement données réelles: {data_file}")

    if not data_file.exists():
        raise FileNotFoundError(f"Fichier introuvable: {data_file}")

    # Utilisation du module centralisé pour chargement + validation
    np_targets, hv_targets, _ = load_targets(
        data_file,
        validate=True,          # Valide automatiquement
        auto_convert_hv=True    # Convertit int8 → float32 si nécessaire
    )

    print(f"\n📊 Format HV Targets:")
    print(f"   Dtype: {hv_targets.dtype}")
    print(f"   Shape: {hv_targets.shape}")
    print(f"   Range: [{hv_targets.min():.4f}, {hv_targets.max():.4f}]")

    # Prendre n_samples aléatoires
    n_available = len(hv_targets)
    n_samples = min(n_samples, n_available)
    indices = np.random.choice(n_available, n_samples, replace=False)

    hv_target = torch.from_numpy(hv_targets[indices])
    np_mask = torch.from_numpy(np_targets[indices])

    # Générer prédictions synthétiques (avec bruit)
    hv_pred = hv_target + torch.randn_like(hv_target) * 0.1

    print(f"   Échantillons chargés: {n_samples}")

    return hv_pred, hv_target, np_mask


def main():
    parser = argparse.ArgumentParser(
        description="Compare MSE vs SmoothL1Loss sur données réelles"
    )
    parser.add_argument(
        '--data_file',
        type=Path,
        help='Fichier .npz de targets (glandular_targets.npz)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=100,
        help='Nombre d\'échantillons à tester (default: 100)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Utiliser données synthétiques au lieu de réelles'
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"COMPARAISON MSE vs SMOOTHL1LOSS")
    print(f"{'='*70}")

    # Charger données
    if args.synthetic or not args.data_file:
        print(f"\n🔧 Génération données synthétiques...")
        hv_pred, hv_target, np_mask = generate_synthetic_batch(
            batch_size=args.n_samples,
            img_size=224,
            n_nuclei=50
        )
    else:
        hv_pred, hv_target, np_mask = load_real_batch(args.data_file, args.n_samples)

    # Comparer loss functions
    print(f"\n⚖️ Comparaison des Loss Functions\n")

    results = compare_losses(hv_pred, hv_target, np_mask)

    print(f"MSE Loss:        {results['mse_loss']:.6f}")
    print(f"SmoothL1 Loss:   {results['smooth_l1_loss']:.6f}")
    print(f"Ratio (S/M):     {results['smooth_l1_loss'] / (results['mse_loss'] + 1e-8):.4f}")

    print(f"\n📊 Magnitude des Gradients:")
    print(f"MSE Gradient Norm:       {results['grad_mse_norm']:.6f}")
    print(f"SmoothL1 Gradient Norm:  {results['grad_smooth_l1_norm']:.6f}")
    print(f"Ratio (S/M):             {results['grad_ratio']:.4f}")

    print(f"\n🎯 VERDICT:")
    if results['grad_ratio'] < 0.9:
        print(f"   ❌ SmoothL1 produit des gradients {(1-results['grad_ratio'])*100:.1f}% plus FAIBLES que MSE")
        print(f"   → Cela peut expliquer des HV maps moins précises")
        print(f"   → RECOMMANDATION: Tester MSE comme HoVer-Net original")
    elif results['grad_ratio'] > 1.1:
        print(f"   ⚠️  SmoothL1 produit des gradients {(results['grad_ratio']-1)*100:.1f}% plus FORTS que MSE")
        print(f"   → Comportement inattendu, vérifier les données")
    else:
        print(f"   ✅ Gradients similaires (ratio ≈ 1.0)")
        print(f"   → SmoothL1 et MSE devraient donner des résultats comparables")

    # Visualisation
    print(f"\n📈 Génération graphique...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Loss values
    axes[0, 0].bar(['MSE', 'SmoothL1'], [results['mse_loss'], results['smooth_l1_loss']])
    axes[0, 0].set_title('Loss Values')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Gradient norms
    axes[0, 1].bar(['MSE', 'SmoothL1'], [results['grad_mse_norm'], results['grad_smooth_l1_norm']])
    axes[0, 1].set_title('Gradient Norms')
    axes[0, 1].set_ylabel('Norm')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Loss curve comparison
    x = np.linspace(-2, 2, 1000)
    mse = x ** 2
    smooth_l1 = np.where(np.abs(x) < 1, 0.5 * x**2, np.abs(x) - 0.5)

    axes[1, 0].plot(x, mse, label='MSE', linewidth=2)
    axes[1, 0].plot(x, smooth_l1, label='SmoothL1', linewidth=2)
    axes[1, 0].set_title('Loss Functions')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Gradient comparison
    grad_mse = 2 * x
    grad_smooth_l1 = np.where(np.abs(x) < 1, x, np.sign(x))

    axes[1, 1].plot(x, grad_mse, label='MSE gradient', linewidth=2)
    axes[1, 1].plot(x, grad_smooth_l1, label='SmoothL1 gradient', linewidth=2)
    axes[1, 1].set_title('Gradients')
    axes[1, 1].set_xlabel('Error')
    axes[1, 1].set_ylabel('Gradient')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path('results/mse_vs_smoothl1_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Sauvegardé: {output_path}")

    print(f"\n✅ Analyse terminée!")


if __name__ == '__main__':
    main()
