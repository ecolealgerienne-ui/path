#!/usr/bin/env python3
"""
Visualize raw model predictions (NP mask, HV maps) before post-processing.

This helps diagnose if the problem is:
1. Bad HV predictions (no gradients at cell boundaries)
2. Bad NP mask (missing cells)
3. Bad watershed post-processing (wrong thresholds)

Usage:
    python scripts/evaluation/visualize_raw_predictions.py \
        --npz_file data/evaluation/pannuke_fold2_converted/image_00002.npz \
        --checkpoint_dir models/checkpoints
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily
from src.models.organ_families import get_family


def visualize_raw_predictions(
    npz_file: Path,
    checkpoint_dir: Path,
    output_dir: Path
) -> None:
    """Visualize raw predictions before watershed."""
    print(f"\n{'='*70}")
    print(f"VISUALIZING RAW PREDICTIONS")
    print(f"{'='*70}\n")

    # Load image
    data = np.load(npz_file)
    image = data['image']
    print(f"📥 Image: {image.shape}, dtype={image.dtype}")

    # Load model
    print(f"🤖 Loading model...")
    model = OptimusGateInferenceMultiFamily(checkpoint_dir=str(checkpoint_dir))

    # Get raw predictions
    print(f"🔮 Running prediction...")

    # We need to access the model internals to get raw outputs
    # Preprocess image
    tensor = model.preprocess(image)

    # Extract features
    features = model.extract_features(tensor)

    # Get organ prediction
    cls_token = features[:, 0, :]
    patch_tokens = features[:, 1:257, :]

    predictions, probs = model.model.organ_head.predict(cls_token)
    organ_idx = predictions[0].item()
    organ_name = model.model.organ_head.organ_names[organ_idx]
    family = get_family(organ_name)

    print(f"   Organ: {organ_name}")
    print(f"   Family: {family}")

    # Get HoVer-Net decoder for this family
    hovernet = model.model.hovernet_decoders[family]

    # Reshape patch tokens for decoder
    patch_tokens_2d = patch_tokens.reshape(1, 16, 16, 1536).permute(0, 3, 1, 2)

    # Get raw predictions (before post-processing)
    with torch.no_grad():
        np_pred, hv_pred, nt_pred = hovernet(patch_tokens_2d)

    # Convert to numpy
    np_pred = torch.sigmoid(np_pred).cpu().numpy()[0, 0]  # (256, 256)
    hv_pred = hv_pred.cpu().numpy()[0]  # (2, 256, 256)
    nt_pred = torch.softmax(nt_pred, dim=1).cpu().numpy()[0]  # (5, 256, 256)

    print(f"\n📊 Raw predictions:")
    print(f"   NP shape: {np_pred.shape}, range: [{np_pred.min():.3f}, {np_pred.max():.3f}]")
    print(f"   HV shape: {hv_pred.shape}, range: [{hv_pred.min():.3f}, {hv_pred.max():.3f}]")
    print(f"   NT shape: {nt_pred.shape}")

    # Compute HV gradients
    h_grad = np.abs(cv2.Sobel(hv_pred[0], cv2.CV_64F, 1, 0, ksize=3))
    v_grad = np.abs(cv2.Sobel(hv_pred[1], cv2.CV_64F, 0, 1, ksize=3))
    edge = h_grad + v_grad
    edge_norm = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

    print(f"   HV gradient range: [{edge.min():.3f}, {edge.max():.3f}]")

    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Row 1: Original, NP mask, NP thresholded
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np_pred, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'NP Prediction (raw)\nRange: [{np_pred.min():.2f}, {np_pred.max():.2f}]')
    axes[0, 1].axis('off')

    np_binary = np_pred > 0.5
    axes[0, 2].imshow(np_binary, cmap='gray')
    axes[0, 2].set_title(f'NP Thresholded (>0.5)\nPixels: {np_binary.sum()}')
    axes[0, 2].axis('off')

    # Row 2: HV maps
    axes[1, 0].imshow(hv_pred[0], cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 0].set_title('HV Map - Horizontal')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(hv_pred[1], cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('HV Map - Vertical')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(edge_norm, cmap='hot')
    axes[1, 2].set_title(f'HV Gradient (edges)\nMax: {edge.max():.3f}')
    axes[1, 2].axis('off')

    # Row 3: Edge thresholds and watershed
    for i, thresh in enumerate([0.1, 0.2, 0.3]):
        ax = axes[2, i]
        edge_mask = edge_norm > thresh
        ax.imshow(edge_mask, cmap='gray')
        ax.set_title(f'Edge > {thresh}\nPixels: {edge_mask.sum()}')
        ax.axis('off')

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{npz_file.stem}_raw_preds.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_file}")

    # Print analysis
    print(f"\n📊 ANALYSIS:")
    print(f"   NP mask covers {np_binary.sum()} pixels ({100*np_binary.sum()/(256*256):.1f}%)")
    print(f"   HV gradient max: {edge.max():.3f}")
    print(f"   Edge pixels (>0.1): {(edge_norm > 0.1).sum()}")
    print(f"   Edge pixels (>0.2): {(edge_norm > 0.2).sum()}")
    print(f"   Edge pixels (>0.3): {(edge_norm > 0.3).sum()}")

    if edge.max() < 0.5:
        print(f"\n⚠️  WARNING: HV gradients are very weak (max={edge.max():.3f})")
        print(f"   This means the model doesn't predict clear boundaries between cells!")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize raw predictions")
    parser.add_argument("--npz_file", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/raw_preds"))

    args = parser.parse_args()
    visualize_raw_predictions(args.npz_file, args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
