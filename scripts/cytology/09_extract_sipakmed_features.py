"""
SIPaKMeD Feature Extraction — H-Optimus-0 Embeddings with Bethesda Mapping

Ce script extrait les features H-Optimus de SIPaKMeD et les mappe vers les classes Bethesda
pour fusion avec APCData.

Mapping SIPaKMeD (7 classes) → Bethesda (6 classes):
    normal_columnar      → NILM (0)
    normal_intermediate  → NILM (0)
    normal_superficiel   → NILM (0)
    light_dysplastic     → LSIL (3)
    moderate_dysplastic  → HSIL (4)  # CIN2 = High-grade
    severe_dysplastic    → HSIL (4)  # CIN3 = High-grade
    carcinoma_in_situ    → SCC (5)

Note: SIPaKMeD n'a pas d'équivalent ASCUS (1) ou ASCH (2).

Usage:
    # Étape 1: Préprocesser SIPaKMeD (si pas déjà fait)
    python scripts/cytology/00_preprocess_sipakmed.py

    # Étape 2: Extraire features H-Optimus
    python scripts/cytology/09_extract_sipakmed_features.py

    # Étape 3: Entraîner avec données fusionnées
    python scripts/cytology/10_train_multihead_combined.py

Author: V15 Cytology Branch
Date: 2026-01-23
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import json

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
#  CONFIGURATION
# =============================================================================

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224

# SIPaKMeD classes
SIPAKMED_CLASSES = [
    'normal_columnar',
    'normal_intermediate',
    'normal_superficiel',
    'light_dysplastic',
    'moderate_dysplastic',
    'severe_dysplastic',
    'carcinoma_in_situ'
]

# Mapping SIPaKMeD → Bethesda
# Bethesda: 0=NILM, 1=ASCUS, 2=ASCH, 3=LSIL, 4=HSIL, 5=SCC
SIPAKMED_TO_BETHESDA = {
    'normal_columnar': 0,      # NILM
    'normal_intermediate': 0,  # NILM
    'normal_superficiel': 0,   # NILM
    'light_dysplastic': 3,     # LSIL (CIN1)
    'moderate_dysplastic': 4,  # HSIL (CIN2)
    'severe_dysplastic': 4,    # HSIL (CIN3)
    'carcinoma_in_situ': 5     # SCC
}

BETHESDA_CLASSES = {
    0: "NILM",
    1: "ASCUS",
    2: "ASCH",
    3: "LSIL",
    4: "HSIL",
    5: "SCC"
}

# Binary mapping: Normal (0) vs Abnormal (1)
BETHESDA_TO_BINARY = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

# Severity mapping: Low-grade (0) vs High-grade (1)
# NILM = N/A (-1), ASCUS/LSIL = Low (0), ASCH/HSIL/SCC = High (1)
BETHESDA_TO_SEVERITY = {0: -1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1}


# =============================================================================
#  DATASET
# =============================================================================

class SIPaKMeDDataset(Dataset):
    """
    Dataset pour SIPaKMeD préprocessé (224×224 PNG).
    """

    def __init__(self, processed_dir: Path, split: str = 'train'):
        self.processed_dir = Path(processed_dir) / split
        self.images_dir = self.processed_dir / 'images'

        # Load metadata
        metadata_path = self.processed_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        print(f"  [INFO] Loaded {len(self.metadata)} samples from {split}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # Load image
        img_path = self.processed_dir / item['image_path']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize for H-Optimus
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(HOPTIMUS_MEAN)) / np.array(HOPTIMUS_STD)

        # To tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Get Bethesda label from SIPaKMeD class
        sipakmed_class = item['class_name']
        bethesda_label = SIPAKMED_TO_BETHESDA[sipakmed_class]
        binary_label = BETHESDA_TO_BINARY[bethesda_label]
        severity_label = BETHESDA_TO_SEVERITY[bethesda_label]

        return image, bethesda_label, binary_label, severity_label, item['filename']


class RawSIPaKMeDDataset(Dataset):
    """
    Dataset pour SIPaKMeD brut (BMP directement depuis raw/).
    Utilise si preprocess n'a pas été fait.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.samples = []

        print(f"  [INFO] Loading raw SIPaKMeD from {raw_dir}")

        for class_name in SIPAKMED_CLASSES:
            class_dir = self.raw_dir / class_name
            if not class_dir.exists():
                print(f"  [WARN] Class dir not found: {class_dir}")
                continue

            # Find .BMP files (uppercase)
            bmp_files = list(class_dir.glob("*.BMP"))
            for bmp_file in bmp_files:
                self.samples.append({
                    'path': bmp_file,
                    'class_name': class_name,
                    'bethesda': SIPAKMED_TO_BETHESDA[class_name]
                })

        print(f"  [INFO] Found {len(self.samples)} samples")

        # Class distribution
        from collections import Counter
        class_counts = Counter(s['class_name'] for s in self.samples)
        for cls, count in sorted(class_counts.items()):
            print(f"    {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Load image
        image = cv2.imread(str(item['path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 224×224 with padding
        h, w = image.shape[:2]
        if h != HOPTIMUS_INPUT_SIZE or w != HOPTIMUS_INPUT_SIZE:
            # Pad to square
            max_dim = max(h, w)
            padded = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
            y_off = (max_dim - h) // 2
            x_off = (max_dim - w) // 2
            padded[y_off:y_off+h, x_off:x_off+w] = image
            # Resize
            image = cv2.resize(padded, (HOPTIMUS_INPUT_SIZE, HOPTIMUS_INPUT_SIZE))

        # Normalize for H-Optimus
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(HOPTIMUS_MEAN)) / np.array(HOPTIMUS_STD)

        # To tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Labels
        bethesda_label = item['bethesda']
        binary_label = BETHESDA_TO_BINARY[bethesda_label]
        severity_label = BETHESDA_TO_SEVERITY[bethesda_label]

        return image, bethesda_label, binary_label, severity_label, item['path'].stem


# =============================================================================
#  FEATURE EXTRACTION
# =============================================================================

def extract_features_hoptimus(
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract H-Optimus features for all samples.

    Returns:
        features: (N, 1536)
        class_labels: (N,) - Bethesda 6-class labels
        binary_labels: (N,) - Normal/Abnormal
        severity_labels: (N,) - Low/High grade (-1 for NILM)
    """
    print("  [INFO] Loading H-Optimus-0...")

    try:
        import timm

        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False
        )
        model = model.to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        print(f"  [OK] H-Optimus-0 loaded on {device}")

    except Exception as e:
        print(f"  [ERROR] Failed to load H-Optimus-0: {e}")
        return None, None, None, None

    all_features = []
    all_class_labels = []
    all_binary_labels = []
    all_severity_labels = []

    print("  [INFO] Extracting features...")

    with torch.no_grad():
        for images, class_labels, binary_labels, severity_labels, filenames in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)

            # Extract features
            features = model.forward_features(images)

            # Get CLS token
            if len(features.shape) == 3:
                cls_tokens = features[:, 0, :]
            else:
                cls_tokens = features

            all_features.append(cls_tokens.cpu())
            all_class_labels.append(class_labels)
            all_binary_labels.append(binary_labels)
            all_severity_labels.append(severity_labels)

    features = torch.cat(all_features, dim=0)
    class_labels = torch.cat(all_class_labels, dim=0)
    binary_labels = torch.cat(all_binary_labels, dim=0)
    severity_labels = torch.cat(all_severity_labels, dim=0)

    print(f"  [OK] Extracted features: {features.shape}")

    return features, class_labels, binary_labels, severity_labels


# =============================================================================
#  MAIN
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract H-Optimus features from SIPaKMeD"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw/sipakmed/pictures",
        help="Path to raw SIPaKMeD directory"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed/sipakmed",
        help="Path to preprocessed SIPaKMeD directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/sipakmed_features",
        help="Output directory for features"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--use_raw",
        action="store_true",
        help="Use raw BMP files directly (skip preprocessing)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  SIPAKMED FEATURE EXTRACTION — H-Optimus-0")
    print("  V15 Cytology Pipeline")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"  [INFO] Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which data source to use
    processed_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir)

    if args.use_raw or not processed_dir.exists():
        if not raw_dir.exists():
            print(f"\n  [ERROR] Raw SIPaKMeD not found: {raw_dir}")
            print("\n  Please download SIPaKMeD first:")
            print("    1. Register at: https://www.cs.uoi.gr/~marina/sipakmed.html")
            print("    2. Download and extract to: data/raw/sipakmed/pictures/")
            print("    3. Run this script again")
            return 1

        print_header("LOADING RAW SIPAKMED")
        dataset = RawSIPaKMeDDataset(raw_dir)
        split_name = "all"
    else:
        print_header("LOADING PREPROCESSED SIPAKMED")
        # Process both train and val
        for split in ['train', 'val']:
            print(f"\n  Processing {split} split...")

            try:
                dataset = SIPaKMeDDataset(processed_dir, split)
            except FileNotFoundError as e:
                print(f"  [ERROR] {e}")
                print("  Run preprocessing first: python scripts/cytology/00_preprocess_sipakmed.py")
                return 1

            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            # Extract features
            features, class_labels, binary_labels, severity_labels = extract_features_hoptimus(dataloader, device)

            if features is None:
                return 1

            # Save
            output_path = output_dir / f"sipakmed_{split}_features.pt"
            torch.save({
                'features': features,
                'class_labels': class_labels,
                'binary_labels': binary_labels,
                'severity_labels': severity_labels,
                'mapping': SIPAKMED_TO_BETHESDA,
                'bethesda_classes': BETHESDA_CLASSES
            }, output_path)
            print(f"  [OK] Saved to {output_path}")

        print_header("SUMMARY")
        print(f"  Output directory: {output_dir}")
        print(f"  Files created:")
        print(f"    - sipakmed_train_features.pt")
        print(f"    - sipakmed_val_features.pt")
        print("\n  Next step:")
        print("    python scripts/cytology/10_train_multihead_combined.py")
        return 0

    # For raw data, process everything together then split
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print_header("EXTRACTING FEATURES")
    features, class_labels, binary_labels, severity_labels = extract_features_hoptimus(dataloader, device)

    if features is None:
        return 1

    # Split 80/20
    print_header("SPLITTING DATA")
    n_samples = len(features)
    n_train = int(n_samples * 0.8)

    # Shuffle indices
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Save train
    train_path = output_dir / "sipakmed_train_features.pt"
    torch.save({
        'features': features[train_idx],
        'class_labels': class_labels[train_idx],
        'binary_labels': binary_labels[train_idx],
        'severity_labels': severity_labels[train_idx],
        'mapping': SIPAKMED_TO_BETHESDA,
        'bethesda_classes': BETHESDA_CLASSES
    }, train_path)
    print(f"  [OK] Train: {len(train_idx)} samples → {train_path}")

    # Save val
    val_path = output_dir / "sipakmed_val_features.pt"
    torch.save({
        'features': features[val_idx],
        'class_labels': class_labels[val_idx],
        'binary_labels': binary_labels[val_idx],
        'severity_labels': severity_labels[val_idx],
        'mapping': SIPAKMED_TO_BETHESDA,
        'bethesda_classes': BETHESDA_CLASSES
    }, val_path)
    print(f"  [OK] Val: {len(val_idx)} samples → {val_path}")

    # Class distribution
    print_header("CLASS DISTRIBUTION (Bethesda)")
    for split_name, labels in [("Train", class_labels[train_idx]), ("Val", class_labels[val_idx])]:
        print(f"\n  {split_name}:")
        for class_id, class_name in BETHESDA_CLASSES.items():
            count = (labels == class_id).sum().item()
            print(f"    {class_name}: {count}")

    print_header("COMPLETE")
    print(f"  Total samples: {n_samples}")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val: {len(val_idx)}")
    print(f"\n  Next step:")
    print(f"    python scripts/cytology/10_train_multihead_combined.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
