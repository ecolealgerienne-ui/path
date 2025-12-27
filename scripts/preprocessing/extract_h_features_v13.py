#!/usr/bin/env python3
"""
Extract H-Channel Features using lightweight CNN adapter.

Pipeline:
1. Load hybrid dataset (with H-channels)
2. Initialize lightweight CNN (3 Conv + Pool + FC → 256-dim)
3. Extract features for all H-channels
4. Save features for training

Author: CellViT-Optimus Team
Date: 2025-12-26
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class HChannelCNN(nn.Module):
    """
    Lightweight CNN for H-channel feature extraction.

    Architecture:
        Input: (B, 1, 224, 224)
        Conv1: 1 → 32, kernel=7, stride=2, padding=3  → (B, 32, 112, 112)
        Conv2: 32 → 64, kernel=5, stride=2, padding=2  → (B, 64, 56, 56)
        Conv3: 64 → 128, kernel=3, stride=2, padding=1 → (B, 128, 28, 28)
        AdaptiveAvgPool2d(1)                           → (B, 128, 1, 1)
        FC: 128 → 256                                  → (B, 256)

    Total params: ~148k (negligible vs 1.1B H-optimus-0)
    """

    def __init__(self, output_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, output_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, h_channel):
        """
        Forward pass.

        Args:
            h_channel: (B, 1, 224, 224) - H-channel images

        Returns:
            features: (B, 256) - Extracted features
        """
        # Conv layers with ReLU and BatchNorm
        x = F.relu(self.bn1(self.conv1(h_channel)))  # (B, 32, 112, 112)
        x = F.relu(self.bn2(self.conv2(x)))          # (B, 64, 56, 56)
        x = F.relu(self.bn3(self.conv3(x)))          # (B, 128, 28, 28)

        # Global pooling
        x = self.pool(x)  # (B, 128, 1, 1)
        x = x.flatten(1)  # (B, 128)

        # FC projection
        features = self.fc(x)  # (B, 256)

        return features

    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def extract_h_features(
    data_file: Path,
    output_dir: Path,
    family: str,
    batch_size: int = 32,
    device: str = 'cuda'
):
    """
    Extract H-channel features using CNN.

    Args:
        data_file: Path to hybrid dataset .npz
        output_dir: Output directory for features
        family: Family name (for output filename)
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING H-CHANNEL FEATURES: {family.upper()}")
    print(f"{'='*80}\n")

    # Device setup
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'

    # Load data
    print(f"📂 Loading hybrid dataset: {data_file}")
    data = np.load(data_file)

    h_channels_224 = data['h_channels_224']  # (N, 224, 224) uint8
    n_crops = h_channels_224.shape[0]

    print(f"  ✅ Loaded {n_crops} H-channels")
    print(f"  Shape: {h_channels_224.shape}, dtype: {h_channels_224.dtype}")

    # Initialize CNN
    print(f"\n🔧 Initializing H-Channel CNN...")
    model = HChannelCNN(output_dim=256).to(device)
    model.eval()  # Evaluation mode (no dropout, BatchNorm in eval)

    n_params = model.get_num_params()
    print(f"  ✅ CNN initialized: {n_params:,} parameters")
    print(f"  Architecture: 3 Conv + Pool + FC → 256-dim")

    # Extract features
    print(f"\n🔬 Extracting features...")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")

    h_features_list = []

    with torch.no_grad():
        for i in tqdm(range(0, n_crops, batch_size), desc="Processing batches"):
            # Get batch
            batch_h = h_channels_224[i:i+batch_size]

            # Convert to tensor (B, 1, 224, 224)
            # Normalize to [0, 1]
            batch_tensor = torch.from_numpy(batch_h).float() / 255.0
            batch_tensor = batch_tensor.unsqueeze(1)  # Add channel dimension

            # Move to device
            batch_tensor = batch_tensor.to(device)

            # Forward pass
            features = model(batch_tensor)  # (B, 256)

            # Store features (CPU)
            h_features_list.append(features.cpu().numpy())

    # Concatenate all batches
    h_features = np.concatenate(h_features_list, axis=0)  # (N, 256)

    print(f"  ✅ H-features extracted: {h_features.shape}, {h_features.dtype}")

    # Compute statistics
    h_mean = h_features.mean()
    h_std = h_features.std()
    h_min = h_features.min()
    h_max = h_features.max()

    print(f"\n📊 H-features statistics:")
    print(f"  Mean: {h_mean:.4f}")
    print(f"  Std: {h_std:.4f}")
    print(f"  Range: [{h_min:.4f}, {h_max:.4f}]")

    # Validation
    if h_std < 0.01:
        print(f"  ⚠️  WARNING: Very low std ({h_std:.4f}). CNN might be outputting near-zero.")
    elif h_std > 10.0:
        print(f"  ⚠️  WARNING: Very high std ({h_std:.4f}). Check CNN initialization.")
    else:
        print(f"  ✅ H-features std looks reasonable")

    # Save features
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{family}_h_features_v13.npz"

    print(f"\n💾 Saving to: {output_file}")
    np.savez_compressed(
        output_file,
        h_features=h_features,
        # Metadata
        cnn_params=n_params,
        feature_mean=h_mean,
        feature_std=h_std
    )

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"  ✅ Saved: {file_size_mb:.2f} MB")

    print(f"\n{'='*80}")
    print(f"✅ H-CHANNEL FEATURES EXTRACTION COMPLETE: {family.upper()}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract H-channel features using CNN")
    parser.add_argument('--data_file', type=Path,
                        help='Path to hybrid dataset .npz file')
    parser.add_argument('--family', type=str, default='epidermal',
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
                        help='Family name (used if --data_file not specified)')
    parser.add_argument('--hybrid_data_dir', type=Path, default=Path('data/family_data_v13_hybrid'),
                        help='Directory containing hybrid data (if --data_file not specified)')
    parser.add_argument('--output_dir', type=Path, default=Path('data/cache/family_data'),
                        help='Output directory for features')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    # Determine data file
    if args.data_file is None:
        args.data_file = args.hybrid_data_dir / f"{args.family}_data_v13_hybrid.npz"

    if not args.data_file.exists():
        raise FileNotFoundError(f"Hybrid dataset file not found: {args.data_file}")

    # Extract features
    extract_h_features(
        data_file=args.data_file,
        output_dir=args.output_dir,
        family=args.family,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
