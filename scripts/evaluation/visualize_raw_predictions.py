#!/usr/bin/env python3
"""
Visualise les prédictions brutes pour UNE image.
Permet de voir le "blob géant" au lieu des noyaux séparés.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.loader import ModelLoader
from src.models.hovernet_decoder import HoVerNetDecoder
from src.preprocessing import create_hoptimus_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_npz", required=True)
    parser.add_argument("--output", default="raw_predictions.png")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load models
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    # Load image
    data = np.load(args.image_npz)
    image = data['image']
    mask = data['mask'] if 'mask' in data else None

    # Preprocess
    transform = create_hoptimus_transform()
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    tensor = transform(image).unsqueeze(0).to(args.device)

    # Inference
    with torch.no_grad():
        features = backbone.forward_features(tensor)
        patch_tokens = features[:, 1:257, :]
        np_out, hv_out, nt_out = hovernet(patch_tokens)

        # To numpy
        np_logits = np_out.cpu().numpy()[0]
        hv_pred = hv_out.cpu().numpy()[0]
        nt_logits = nt_out.cpu().numpy()[0]

        # Activations
        np_probs = 1 / (1 + np.exp(-np_logits))
        np_binary = np_probs[1]
        nt_probs = np.exp(nt_logits) / np.exp(nt_logits).sum(axis=0, keepdims=True)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Input + NP + HV magnitude
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image H&E")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np_binary, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title(f"NP Prediction\nMax={np_binary.max():.3f}")
    axes[0, 1].axis('off')

    hv_magnitude = np.sqrt(hv_pred[0]**2 + hv_pred[1]**2)
    axes[0, 2].imshow(hv_magnitude, cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title(f"HV Magnitude\nMax={hv_magnitude.max():.3f}")
    axes[0, 2].axis('off')

    # Row 2: HV H + HV V + NT
    axes[1, 0].imshow(hv_pred[0], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title(f"HV Horizontal\nRange=[{hv_pred[0].min():.2f}, {hv_pred[0].max():.2f}]")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(hv_pred[1], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_title(f"HV Vertical\nRange=[{hv_pred[1].min():.2f}, {hv_pred[1].max():.2f}]")
    axes[1, 1].axis('off')

    nt_argmax = nt_probs.argmax(axis=0)
    cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'cyan'])
    axes[1, 2].imshow(nt_argmax, cmap=cmap, vmin=0, vmax=5)
    axes[1, 2].set_title("NT Classification")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {args.output}")

    # Print stats
    print(f"\n📊 STATISTIQUES:")
    print(f"  NP max:  {np_binary.max():.4f}")
    print(f"  NP mean: {np_binary.mean():.4f}")
    print(f"  HV max:  {np.abs(hv_pred).max():.4f}")
    print(f"  HV mean: {np.abs(hv_pred).mean():.4f}")
    print(f"  HV magnitude max: {hv_magnitude.max():.4f}")

    if np.abs(hv_pred).max() < 0.3:
        print(f"\n  🔴 TANH SATURE: HV max = {np.abs(hv_pred).max():.3f} < 0.3")
    elif hv_magnitude.max() < 0.2:
        print(f"\n  ⚠️  HV MAGNITUDE FAIBLE: {hv_magnitude.max():.3f} < 0.2")
        print(f"     → Watershed ne verra pas de contours nets")


if __name__ == "__main__":
    main()
