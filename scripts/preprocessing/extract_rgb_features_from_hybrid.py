#!/usr/bin/env python3
"""
Extract RGB features from V13-Hybrid dataset using H-optimus-0.

This script extracts H-optimus-0 features from the images in the hybrid dataset.

Usage: python extract_rgb_features_from_hybrid.py --family epidermal
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import timm
from torchvision import transforms


# H-optimus-0 normalization (from src/constants.py)
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


def create_hoptimus_transform():
    """Create canonical H-optimus-0 transform."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])


def load_hoptimus0(device='cuda'):
    """Load H-optimus-0 backbone."""
    print("Loading H-optimus-0 backbone...")
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False
    )
    model = model.to(device)
    model.eval()

    # Freeze
    for param in model.parameters():
        param.requires_grad = False

    print(f"  ✅ H-optimus-0 loaded on {device}")
    return model


def extract_features_batch(backbone, images_batch, transform, device):
    """Extract features for a batch of images."""
    batch_size = len(images_batch)

    # Preprocess images
    tensors = []
    for img in images_batch:
        tensor = transform(img)  # (3, 224, 224)
        tensors.append(tensor)

    batch_tensor = torch.stack(tensors).to(device)  # (B, 3, 224, 224)

    # Extract features
    with torch.no_grad():
        features = backbone.forward_features(batch_tensor)  # (B, 261, 1536)

    return features.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Extract RGB features from hybrid dataset")
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--hybrid_data_dir', type=Path, default=Path('data/family_data_v13_hybrid'))
    parser.add_argument('--output_dir', type=Path, default=Path('data/cache/family_data'))
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("="*80)
    print(f"EXTRACTING RGB FEATURES: {args.family.upper()}")
    print("="*80)

    # Paths
    hybrid_data_file = args.hybrid_data_dir / f"{args.family}_data_v13_hybrid.npz"

    if not hybrid_data_file.exists():
        raise FileNotFoundError(f"Hybrid data not found: {hybrid_data_file}")

    # Load hybrid dataset
    print(f"\n📂 Loading hybrid dataset: {hybrid_data_file}")
    data = np.load(hybrid_data_file)

    images_224 = data['images_224']  # (N, 224, 224, 3) uint8
    fold_ids = data['fold_ids']  # (N,) int32

    n_samples = len(images_224)
    print(f"  ✅ Loaded {n_samples} samples")
    print(f"  Images: {images_224.shape}, {images_224.dtype}")

    data.close()

    # Load H-optimus-0
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    backbone = load_hoptimus0(device)
    transform = create_hoptimus_transform()

    # Extract features
    print(f"\n🔬 Extracting RGB features...")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")

    all_features = []
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(0, n_samples, args.batch_size), desc="Processing batches", total=n_batches):
        batch_images = images_224[i:i+args.batch_size]
        features_batch = extract_features_batch(backbone, batch_images, transform, device)
        all_features.append(features_batch)

    # Concatenate
    all_features = np.concatenate(all_features, axis=0)  # (N, 261, 1536)

    print(f"  ✅ RGB features extracted: {all_features.shape}, {all_features.dtype}")

    # Validate CLS std (should be ~0.70-0.90)
    cls_tokens = all_features[:, 0, :]  # (N, 1536)
    cls_std = cls_tokens.std()

    print(f"\n🔍 Validation:")
    print(f"  CLS token std: {cls_std:.4f}")

    if not (0.70 <= cls_std <= 0.90):
        print(f"  ⚠️  WARNING: CLS std out of range [0.70, 0.90]")
    else:
        print(f"  ✅ CLS std OK")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"{args.family}_rgb_features_v13.npz"

    print(f"\n💾 Saving to: {output_file}")
    np.savez_compressed(
        output_file,
        features=all_features,
        fold_ids=fold_ids,
        family=args.family
    )

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"  ✅ Saved: {file_size_mb:.2f} MB")

    print("\n" + "="*80)
    print(f"✅ RGB FEATURES EXTRACTION COMPLETE: {args.family.upper()}")
    print("="*80)


if __name__ == '__main__':
    main()
