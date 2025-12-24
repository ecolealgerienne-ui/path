#!/usr/bin/env python3
"""
Extrait N échantillons de PanNuke fold2 (.npy) vers .npz individuels.
Utilisé pour diagnostic rapide sans convertir tout le fold.
"""
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold2_dir", required=True, help="Path to fold2 directory with .npy files")
    parser.add_argument("--output_dir", default="data/temp_fold2_samples", help="Output directory for .npz files")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to extract")
    args = parser.parse_args()

    fold2_dir = Path(args.fold2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔧 Extraction {args.num_samples} échantillons de PanNuke fold2...")
    print(f"   Source: {fold2_dir}")
    print(f"   Destination: {output_dir}")

    # Load fold2 data (memory-mapped pour économiser RAM)
    print("\n📂 Chargement données (memory-mapped)...")
    images = np.load(fold2_dir / "images.npy", mmap_mode='r')
    masks = np.load(fold2_dir / "masks.npy", mmap_mode='r')
    types = np.load(fold2_dir / "types.npy")

    print(f"   Images: {images.shape}")
    print(f"   Masks: {masks.shape}")
    print(f"   Types: {types.shape}")

    # Extract samples
    n_samples = min(args.num_samples, len(images))

    print(f"\n💾 Extraction {n_samples} échantillons...")
    for i in tqdm(range(n_samples)):
        # Copy to memory (otherwise save will fail on mmap)
        image = images[i].copy()
        mask = masks[i].copy()
        organ_type = types[i]

        # Save as .npz
        output_path = output_dir / f"sample_{i:05d}.npz"
        np.savez_compressed(
            output_path,
            image=image,
            mask=mask,
            organ_type=organ_type
        )

    print(f"\n✅ {n_samples} échantillons extraits dans {output_dir}")
    print(f"\n🚀 Commande pour diagnostic:")
    print(f"   python scripts/evaluation/diagnose_predictions.py \\")
    print(f"       --checkpoint models/checkpoints/hovernet_epidermal_best.pth \\")
    print(f"       --dataset_dir {output_dir} \\")
    print(f"       --num_samples {n_samples}")


if __name__ == "__main__":
    main()
