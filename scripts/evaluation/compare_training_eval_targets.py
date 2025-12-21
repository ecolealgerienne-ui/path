#!/usr/bin/env python3
"""
Compare les targets d'entraînement avec le GT d'évaluation.

Vérifie si les données d'entraînement étaient cohérentes avec le GT.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def compare_training_vs_eval_targets(
    pannuke_dir: Path,
    family_data_dir: Path,
    fold: int = 2,
    image_idx: int = 2
):
    """Compare training targets vs eval GT for one image."""

    print("=" * 70)
    print("COMPARAISON TRAINING TARGETS VS EVAL GT")
    print("=" * 70)

    # 1. Charger l'image et mask PanNuke bruts
    masks_path = pannuke_dir / f"fold{fold}" / "masks.npy"
    images_path = pannuke_dir / f"fold{fold}" / "images.npy"

    masks = np.load(masks_path)  # (N, 256, 256, 6)
    images = np.load(images_path)  # (N, 256, 256, 3)

    mask = masks[image_idx]
    image = images[image_idx]

    print(f"\n📥 PanNuke Raw Data:")
    print(f"   Image: {image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
    print(f"   Mask: {mask.shape}, dtype={mask.dtype}")

    # 2. Générer le NP mask comme pendant l'entraînement
    np_mask_train = mask[:, :, 1:].sum(axis=-1) > 0  # Canaux 1-5

    print(f"\n🎯 TRAINING NP Target (sum(channels 1-5) > 0):")
    print(f"   Shape: {np_mask_train.shape}")
    print(f"   Coverage: {np_mask_train.sum() / np_mask_train.size * 100:.2f}%")
    print(f"   dtype: {np_mask_train.dtype}")

    # 3. Générer le GT comme pendant l'évaluation
    np_mask_eval = mask[:, :, 1:6].sum(axis=-1) > 0  # Canaux 1-5 (identique)
    _, inst_map_eval = cv2.connectedComponents(np_mask_eval.astype(np.uint8))

    print(f"\n📊 EVAL GT (connectedComponents):")
    print(f"   Binary mask coverage: {np_mask_eval.sum() / np_mask_eval.size * 100:.2f}%")
    print(f"   Instances: {len(np.unique(inst_map_eval)) - 1}")

    # 4. Comparer
    print(f"\n🔍 COMPARISON:")
    print(f"   Training NP == Eval NP: {np.array_equal(np_mask_train, np_mask_eval)}")
    print(f"   Difference: {np.abs(np_mask_train.astype(int) - np_mask_eval.astype(int)).sum()} pixels")

    # 5. Visualiser
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Image, Training NP, Eval NP
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image (PanNuke)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np_mask_train, cmap='gray')
    axes[0, 1].set_title(f"Training NP Target\n{np_mask_train.sum() / np_mask_train.size * 100:.2f}%")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np_mask_eval, cmap='gray')
    axes[0, 2].set_title(f"Eval NP GT\n{np_mask_eval.sum() / np_mask_eval.size * 100:.2f}%")
    axes[0, 2].axis('off')

    # Row 2: Difference, Instances, Channels breakdown
    diff = np.abs(np_mask_train.astype(int) - np_mask_eval.astype(int))
    axes[1, 0].imshow(diff, cmap='Reds')
    axes[1, 0].set_title(f"Difference\n{diff.sum()} pixels")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(inst_map_eval, cmap='tab20')
    axes[1, 1].set_title(f"Eval Instances\n{len(np.unique(inst_map_eval)) - 1} instances")
    axes[1, 1].axis('off')

    # Channel breakdown
    channel_names = ['BG', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
    channel_counts = []
    for c in range(6):
        count = (mask[:, :, c] > 0).sum()
        channel_counts.append(count)
        print(f"   Channel {c} ({channel_names[c]:15}): {count:5} pixels ({count / (256*256) * 100:5.2f}%)")

    axes[1, 2].bar(range(6), channel_counts, color=['gray', 'red', 'green', 'blue', 'yellow', 'cyan'])
    axes[1, 2].set_xticks(range(6))
    axes[1, 2].set_xticklabels([c[:4] for c in channel_names], rotation=45)
    axes[1, 2].set_title("Pixels per Channel")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path("results/training_vs_eval_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")

    # 6. Vérifier si les données d'entraînement existantes sont cohérentes
    if family_data_dir.exists():
        print(f"\n📂 Checking existing family training data...")
        for family in ['glandular', 'digestive', 'urologic', 'respiratory', 'epidermal']:
            targets_path = family_data_dir / f"{family}_targets.npz"
            if targets_path.exists():
                data = np.load(targets_path)
                if 'np_targets' in data:
                    np_targets = data['np_targets']
                    print(f"   {family:12}: {np_targets.shape}, coverage={np_targets.mean() * 100:.2f}%")

    print("=" * 70)

    return np_mask_train, np_mask_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pannuke_dir", type=Path, default=Path("/home/amar/data/PanNuke"))
    parser.add_argument("--family_data_dir", type=Path, default=Path("data/family"))
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--image_idx", type=int, default=2)

    args = parser.parse_args()

    compare_training_vs_eval_targets(
        args.pannuke_dir,
        args.family_data_dir,
        args.fold,
        args.image_idx
    )


if __name__ == "__main__":
    main()
