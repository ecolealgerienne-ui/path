#!/usr/bin/env python3
"""
Diagnostic visuel pour comprendre l'échec Ground Truth.

Génère des visualisations comparatives GT vs Prédictions
pour identifier la cause du problème.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily


def visualize_comparison(
    image: np.ndarray,
    gt_inst: np.ndarray,
    gt_type: np.ndarray,
    pred_inst: np.ndarray,
    pred_type: np.ndarray,
    pred_np: np.ndarray,
    pred_hv: np.ndarray,
    output_path: Path,
):
    """Visualisation comparative complète."""

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Row 1: Image + GT
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image Originale", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_inst, cmap='nipy_spectral')
    axes[0, 1].set_title(f"GT Instances\n{gt_inst.max()} instances", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gt_type, cmap='tab10', vmin=0, vmax=5)
    axes[0, 2].set_title("GT Types", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Histogramme GT types
    gt_type_hist = np.bincount(gt_type[gt_type > 0], minlength=6)
    axes[0, 3].bar(range(1, 6), gt_type_hist[1:])
    axes[0, 3].set_title("GT Type Distribution", fontsize=14, fontweight='bold')
    axes[0, 3].set_xticks(range(1, 6))
    axes[0, 3].set_xticklabels(['Neo', 'Inf', 'Con', 'Dead', 'Epi'], rotation=45)

    # Row 2: Prédictions
    axes[1, 0].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f"Pred NP Prob\nMax: {pred_np.max():.3f}", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_inst, cmap='nipy_spectral')
    axes[1, 1].set_title(f"Pred Instances\n{pred_inst.max()} instances", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pred_type, cmap='tab10', vmin=0, vmax=5)
    axes[1, 2].set_title("Pred Types", fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    # Histogramme Pred types
    pred_type_hist = np.bincount(pred_type[pred_type > 0], minlength=6)
    axes[1, 3].bar(range(1, 6), pred_type_hist[1:], alpha=0.7, label='Pred')
    axes[1, 3].bar(range(1, 6), gt_type_hist[1:], alpha=0.5, label='GT')
    axes[1, 3].set_title("Type Distribution Comparison", fontsize=14, fontweight='bold')
    axes[1, 3].set_xticks(range(1, 6))
    axes[1, 3].set_xticklabels(['Neo', 'Inf', 'Con', 'Dead', 'Epi'], rotation=45)
    axes[1, 3].legend()

    # Row 3: HV maps + gradients
    axes[2, 0].imshow(pred_hv[0], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2, 0].set_title(f"HV-H\n[{pred_hv[0].min():.3f}, {pred_hv[0].max():.3f}]", fontsize=14, fontweight='bold')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(pred_hv[1], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2, 1].set_title(f"HV-V\n[{pred_hv[1].min():.3f}, {pred_hv[1].max():.3f}]", fontsize=14, fontweight='bold')
    axes[2, 1].axis('off')

    # Gradients HV
    h_grad = np.abs(cv2.Sobel(pred_hv[0], cv2.CV_64F, 1, 0, ksize=3))
    v_grad = np.abs(cv2.Sobel(pred_hv[1], cv2.CV_64F, 0, 1, ksize=3))
    grad_magnitude = h_grad + v_grad

    axes[2, 2].imshow(grad_magnitude, cmap='hot')
    axes[2, 2].set_title(f"HV Gradient Magnitude\nMax: {grad_magnitude.max():.3f}", fontsize=14, fontweight='bold')
    axes[2, 2].axis('off')

    # Overlay GT vs Pred instances
    from matplotlib.colors import ListedColormap

    # Créer overlay
    overlay = image.copy()

    # GT en vert (contours)
    gt_contours = cv2.Canny((gt_inst > 0).astype(np.uint8) * 255, 50, 150)
    overlay[gt_contours > 0] = [0, 255, 0]  # Vert

    # Pred en rouge (contours)
    pred_contours = cv2.Canny((pred_inst > 0).astype(np.uint8) * 255, 50, 150)
    overlay[pred_contours > 0] = [255, 0, 0]  # Rouge

    axes[2, 3].imshow(overlay)
    axes[2, 3].set_title("Overlay: GT (green) vs Pred (red)", fontsize=14, fontweight='bold')
    axes[2, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Diagnostic saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose GT failure")
    parser.add_argument("--npz_file", type=str, required=True, help="Path to .npz GT file")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints_FIXED")
    parser.add_argument("--output_dir", type=str, default="results/diagnostic_gt")

    args = parser.parse_args()

    npz_file = Path(args.npz_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("DIAGNOSTIC ÉCHEC GROUND TRUTH")
    print("="*70)
    print(f"\nFichier: {npz_file.name}")

    # Load GT
    data = np.load(npz_file)
    image = data['image']
    gt_inst = data['inst_map']
    gt_type = data['type_map']

    print(f"\nGT:")
    print(f"  Image: {image.shape}")
    print(f"  Instances: {gt_inst.max()}")
    print(f"  Types: {np.unique(gt_type[gt_type > 0])}")

    # Load model
    print(f"\nChargement modèle: {args.checkpoint_dir}")
    model = OptimusGateInferenceMultiFamily(
        checkpoint_dir=args.checkpoint_dir,
        device='cuda',
    )

    # Predict
    pil_image = Image.fromarray(image)
    result = model.predict(pil_image)

    pred_inst = result['instance_map']
    pred_type = result['type_map']
    pred_np = result.get('np_prob', np.zeros_like(pred_inst))
    pred_hv = result.get('hv_map', np.zeros((2, *pred_inst.shape)))

    print(f"\nPrédictions:")
    print(f"  Instances: {pred_inst.max()}")
    print(f"  Types: {np.unique(pred_type[pred_type > 0])}")
    print(f"  HV range: [{pred_hv.min():.3f}, {pred_hv.max():.3f}]")

    # Resize pred to match GT if needed
    if pred_inst.shape != gt_inst.shape:
        print(f"\n⚠️  Resize pred {pred_inst.shape} → {gt_inst.shape}")
        pred_inst = cv2.resize(pred_inst, gt_inst.shape[::-1], interpolation=cv2.INTER_NEAREST)
        pred_type = cv2.resize(pred_type, gt_inst.shape[::-1], interpolation=cv2.INTER_NEAREST)
        pred_np = cv2.resize(pred_np, gt_inst.shape[::-1], interpolation=cv2.INTER_LINEAR)

        pred_hv_resized = np.zeros((2, *gt_inst.shape))
        for c in range(2):
            pred_hv_resized[c] = cv2.resize(pred_hv[c], gt_inst.shape[::-1], interpolation=cv2.INTER_LINEAR)
        pred_hv = pred_hv_resized

    # Visualize
    output_path = output_dir / f"diagnostic_{npz_file.stem}.png"
    visualize_comparison(
        image, gt_inst, gt_type,
        pred_inst, pred_type,
        pred_np, pred_hv,
        output_path
    )

    # Stats
    print("\n" + "="*70)
    print("STATISTIQUES COMPARATIVES")
    print("="*70)

    print(f"\nNombre d'instances:")
    print(f"  GT:   {gt_inst.max()}")
    print(f"  Pred: {pred_inst.max()}")
    print(f"  Ratio: {pred_inst.max() / max(gt_inst.max(), 1):.2f}x")

    print(f"\nDistribution types (GT):")
    for i, name in enumerate(['', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial'], 1):
        if i == 0:
            continue
        count = (gt_type == i).sum()
        print(f"  {name:15s}: {count:4d} pixels")

    print(f"\nDistribution types (Pred):")
    for i, name in enumerate(['', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial'], 1):
        if i == 0:
            continue
        count = (pred_type == i).sum()
        print(f"  {name:15s}: {count:4d} pixels")

    print(f"\n✅ Diagnostic terminé: {output_path}")
    print(f"\n💡 Analyser visuellement pour identifier le problème")


if __name__ == "__main__":
    main()
