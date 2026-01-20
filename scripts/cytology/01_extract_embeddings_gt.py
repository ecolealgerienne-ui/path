"""
H-Optimus-0 Embeddings Extraction — Using Ground Truth Masks

Ce script extrait les embeddings H-Optimus-0 pour le pipeline V14 Cytologie:
1. Charge images SIPaKMeD preprocessées (224×224)
2. Utilise les masques GT (pas CellPose) — Décision stratégique V14
3. Extrait embeddings H-Optimus-0 (1536-dim)
4. Sauvegarde pour entraînement MLP

Décision Stratégique (2026-01-20):
- CellPose inadapté aux cellules isolées SIPaKMeD
- Masques GT utilisés pour validation architecture
- CellPose sera utilisé en production sur lames réelles (groupes cellulaires)

Author: V14 Cytology Branch
Date: 2026-01-20
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# H-Optimus-0 normalization constants (from src/preprocessing)
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224


# =============================================================================
#  DATASET
# =============================================================================

class SIPaKMeDDataset(Dataset):
    """
    Dataset pour images SIPaKMeD preprocessées avec masques GT
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Args:
            data_dir: data/processed/sipakmed/
            split: 'train' or 'val'
        """
        self.split_dir = os.path.join(data_dir, split)
        self.metadata_path = os.path.join(self.split_dir, 'metadata.json')

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"Loaded {len(self.metadata)} samples from {split} split")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata[idx]

        # Load image
        image_path = os.path.join(self.split_dir, sample['image_path'])
        image = np.array(Image.open(image_path).convert('RGB'))

        # Load GT mask
        mask_path = os.path.join(self.split_dir, sample['mask_path'])
        mask = np.array(Image.open(mask_path))
        mask = (mask > 0).astype(np.uint8)  # Binary mask

        # Normalize for H-Optimus-0
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(HOPTIMUS_MEAN)) / np.array(HOPTIMUS_STD)

        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'label': sample['label'],
            'class_name': sample['class_name'],
            'filename': sample['filename']
        }


# =============================================================================
#  H-OPTIMUS-0 MODEL
# =============================================================================

def load_hoptimus_model(device: str = 'cuda'):
    """
    Charge H-Optimus-0 depuis HuggingFace

    Returns:
        model: H-Optimus-0 model (frozen)
        transform: Preprocessing transform
    """
    try:
        import timm
        from huggingface_hub import login

        print("Loading H-Optimus-0 from HuggingFace...")

        # Load model
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False
        )

        model = model.to(device)
        model.eval()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        print(f"H-Optimus-0 loaded on {device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        return model

    except Exception as e:
        print(f"Error loading H-Optimus-0: {e}")
        print("Make sure you have: pip install timm huggingface_hub")
        print("And logged in: huggingface-cli login")
        raise


def extract_features(model, images: torch.Tensor, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Extrait les features H-Optimus-0

    Args:
        model: H-Optimus-0 model
        images: Batch of images (B, 3, 224, 224)
        device: Device

    Returns:
        features: Dict with 'cls_token' and 'patch_tokens'
    """
    images = images.to(device)

    with torch.no_grad():
        # Forward pass - get all tokens
        outputs = model.forward_features(images)

        # outputs shape: (B, 261, 1536)
        # - [:, 0, :] = CLS token
        # - [:, 1:5, :] = 4 Register tokens (ignored)
        # - [:, 5:261, :] = 256 Patch tokens

        cls_token = outputs[:, 0, :]  # (B, 1536)
        patch_tokens = outputs[:, 5:261, :]  # (B, 256, 1536)

    return {
        'cls_token': cls_token.cpu(),
        'patch_tokens': patch_tokens.cpu()
    }


# =============================================================================
#  MAIN EXTRACTION
# =============================================================================

def extract_embeddings_for_split(
    data_dir: str,
    split: str,
    output_dir: str,
    batch_size: int = 16,
    device: str = 'cuda'
):
    """
    Extrait les embeddings pour un split complet

    Args:
        data_dir: data/processed/sipakmed/
        split: 'train' or 'val'
        output_dir: Output directory for embeddings
        batch_size: Batch size
        device: Device
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING EMBEDDINGS — {split.upper()} SPLIT")
    print(f"{'='*80}")

    # Create dataset and dataloader
    dataset = SIPaKMeDDataset(data_dir, split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    model = load_hoptimus_model(device)

    # Storage
    all_cls_tokens = []
    all_patch_tokens = []
    all_labels = []
    all_class_names = []
    all_filenames = []
    all_masks = []

    # Extract
    print(f"\nExtracting embeddings for {len(dataset)} samples...")

    for batch in tqdm(dataloader, desc=f"  {split}"):
        images = batch['image']

        # Extract features
        features = extract_features(model, images, device)

        # Store
        all_cls_tokens.append(features['cls_token'])
        all_patch_tokens.append(features['patch_tokens'])
        all_labels.extend(batch['label'].tolist())
        all_class_names.extend(batch['class_name'])
        all_filenames.extend(batch['filename'])
        all_masks.append(batch['mask'])

    # Concatenate
    cls_tokens = torch.cat(all_cls_tokens, dim=0)  # (N, 1536)
    patch_tokens = torch.cat(all_patch_tokens, dim=0)  # (N, 256, 1536)
    masks = torch.cat(all_masks, dim=0)  # (N, 224, 224)
    labels = torch.tensor(all_labels)  # (N,)

    print(f"\nEmbeddings extracted:")
    print(f"  CLS tokens:   {cls_tokens.shape}")
    print(f"  Patch tokens: {patch_tokens.shape}")
    print(f"  Masks:        {masks.shape}")
    print(f"  Labels:       {labels.shape}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'sipakmed_{split}_embeddings.pt')

    torch.save({
        'cls_tokens': cls_tokens,
        'patch_tokens': patch_tokens,
        'masks': masks,
        'labels': labels,
        'class_names': all_class_names,
        'filenames': all_filenames,
        'metadata': {
            'model': 'H-Optimus-0',
            'input_size': HOPTIMUS_INPUT_SIZE,
            'cls_dim': 1536,
            'patch_dim': 1536,
            'n_patches': 256,
            'normalization': {
                'mean': HOPTIMUS_MEAN,
                'std': HOPTIMUS_STD
            },
            'strategy': 'GT_masks (CellPose skipped for SIPaKMeD)'
        }
    }, output_path)

    print(f"\n  Saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    return cls_tokens, patch_tokens, labels


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract H-Optimus-0 embeddings using GT masks (V14 Cytology)"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/sipakmed',
        help='Preprocessed data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/embeddings/sipakmed',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='both',
        choices=['train', 'val', 'both'],
        help='Which split to process'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for extraction'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("H-OPTIMUS-0 EMBEDDINGS EXTRACTION — V14 Cytology")
    print("=" * 80)
    print(f"Data directory:   {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split:            {args.split}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Device:           {args.device}")
    print("")
    print("Strategy: Using GT masks (CellPose skipped for SIPaKMeD)")
    print("  - CellPose inadapté aux cellules isolées")
    print("  - Masques GT pour validation architecture")
    print("  - CellPose sera utilisé en production (lames réelles)")
    print("=" * 80)

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Extract embeddings
    splits = ['train', 'val'] if args.split == 'both' else [args.split]

    for split in splits:
        extract_embeddings_for_split(
            data_dir=args.data_dir,
            split=split,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device
        )

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Embeddings saved to: {args.output_dir}")
    print("\nNext step:")
    print("  python scripts/cytology/02_compute_morphometry.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
