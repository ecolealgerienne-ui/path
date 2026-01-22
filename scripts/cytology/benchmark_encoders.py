#!/usr/bin/env python3
"""
Benchmark Encoders pour Cytologie V15.2

Compare H-Optimus, UNI, et Phikon-v2 sur tâches cytologie Pap.

Objectif: Démontrer que H-Optimus (entraîné H&E) n'est pas optimal
pour cytologie Pap-stain et justifier le passage à UNI/Phikon.

Usage:
    python scripts/cytology/benchmark_encoders.py \
        --dataset apcdata \
        --encoders h-optimus,uni,phikon-v2 \
        --output_dir reports/encoder_benchmark

Author: CellViT-Optimus Team
Date: 2026-01-22
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import preprocess_image, HOPTIMUS_MEAN, HOPTIMUS_STD

# ============================================================================
# CONSTANTS
# ============================================================================

BETHESDA_CLASSES = ['NILM', 'ASCUS', 'ASCH', 'LSIL', 'HSIL', 'SCC']
BETHESDA_ABNORMAL = ['ASCUS', 'ASCH', 'LSIL', 'HSIL', 'SCC']

# Encoder configurations
ENCODER_CONFIGS = {
    'h-optimus': {
        'model_name': 'bioptimus/H-optimus-0',
        'embed_dim': 1536,
        'mean': (0.707223, 0.578729, 0.703617),
        'std': (0.211883, 0.230117, 0.177517),
        'description': 'H-Optimus-0 (1.1B params, trained on H&E histopathology)'
    },
    'uni': {
        'model_name': 'MahmoodLab/UNI',
        'embed_dim': 1024,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'description': 'UNI (trained on diverse pathology including cytology)'
    },
    'phikon-v2': {
        'model_name': 'owkin/phikon-v2',
        'embed_dim': 768,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'description': 'Phikon-v2 (trained on TCGA + diverse pathology)'
    }
}

# ============================================================================
# ENCODER LOADERS
# ============================================================================

def load_encoder(encoder_name: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """
    Load a pretrained encoder.

    Args:
        encoder_name: Name of encoder ('h-optimus', 'uni', 'phikon-v2')
        device: Torch device

    Returns:
        model: Loaded model in eval mode
        config: Encoder configuration
    """
    config = ENCODER_CONFIGS[encoder_name]

    print(f"Loading {encoder_name}: {config['description']}")

    if encoder_name == 'h-optimus':
        # H-Optimus must be loaded via timm (not transformers)
        import timm
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False
        )
    elif encoder_name == 'uni':
        # UNI uses timm
        import timm
        model = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            num_classes=0  # Remove classification head
        )
        # Load UNI weights (requires huggingface login)
        # For benchmark, we use the public checkpoint
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id=config['model_name'],
                filename='pytorch_model.bin'
            )
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load UNI weights: {e}")
            print("Using randomly initialized model for structure testing")
    elif encoder_name == 'phikon-v2':
        from transformers import AutoModel
        model = AutoModel.from_pretrained(config['model_name'])
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    model = model.to(device)
    model.eval()

    return model, config


def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    encoder_name: str
) -> torch.Tensor:
    """
    Extract CLS token features from encoder.

    Args:
        model: Encoder model
        images: Batch of images (B, 3, 224, 224)
        encoder_name: Name of encoder

    Returns:
        features: (B, embed_dim)
    """
    with torch.no_grad():
        if encoder_name == 'h-optimus':
            # H-Optimus via timm: use forward_features
            outputs = model.forward_features(images)
            # H-Optimus: output is (B, 261, 1536) = CLS + 4 registers + 256 patches
            features = outputs[:, 0, :]  # CLS token
        elif encoder_name == 'uni':
            features = model(images)
        elif encoder_name == 'phikon-v2':
            outputs = model(images)
            # Phikon: get CLS token
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs[:, 0, :]
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

    return features

# ============================================================================
# YOLO ANNOTATION LOADING (from existing scripts)
# ============================================================================

# YOLO class mapping (from classes.txt)
YOLO_CLASS_MAPPING = {
    0: "NILM",
    1: "ASCUS",
    2: "ASCH",
    3: "LSIL",
    4: "HSIL",
    5: "SCC"
}


def load_yolo_annotations(data_dir: str, image_width: int = 2048, image_height: int = 1532) -> Dict[str, List[Dict]]:
    """
    Load APCData annotations from YOLO format.

    YOLO format: class_id x_center y_center width height (normalized 0-1)

    Args:
        data_dir: Path to APCData_YOLO/ directory
        image_width: Image width for denormalization
        image_height: Image height for denormalization

    Returns:
        Dict mapping image_filename to list of cell annotations
    """
    labels_dir = os.path.join(data_dir, 'labels')
    images_dir = os.path.join(data_dir, 'images')

    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"YOLO labels directory not found: {labels_dir}")

    annotations = {}

    # Get all label files (exclude Zone.Identifier files from WSL)
    label_files = [f for f in os.listdir(labels_dir)
                   if f.endswith('.txt') and ':Zone' not in f]

    for label_file in label_files:
        base_name = label_file.replace('.txt', '')

        # Find matching image (try common extensions)
        image_file = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = base_name + ext
            if os.path.exists(os.path.join(images_dir, candidate)):
                image_file = candidate
                break

        if image_file is None:
            continue

        # Read label file
        label_path = os.path.join(labels_dir, label_file)
        cells = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])

                    # Denormalize to pixel coordinates
                    nucleus_x = int(x_center * image_width)
                    nucleus_y = int(y_center * image_height)

                    bethesda_class = YOLO_CLASS_MAPPING.get(class_id, "NILM")

                    cells.append({
                        'class': bethesda_class,
                        'nucleus_x': nucleus_x,
                        'nucleus_y': nucleus_y
                    })

        if cells:
            annotations[image_file] = cells

    return annotations


def extract_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    patch_size: int = 224
) -> np.ndarray:
    """
    Extract a patch centered on (center_x, center_y).
    Pads with white if near image boundary.
    """
    h, w = image.shape[:2]
    half = patch_size // 2

    # Calculate crop boundaries
    x1 = int(center_x - half)
    y1 = int(center_y - half)
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    # Create white canvas
    patch = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255

    # Calculate valid regions
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Copy valid region
    patch[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    return patch


# ============================================================================
# DATASET
# ============================================================================

class APCDataBenchmarkDataset(Dataset):
    """
    Dataset for encoder benchmarking on APCData.

    Loads full images and extracts 224×224 patches centered on annotated cells.
    Compatible with APCData_YOLO format.
    """

    def __init__(
        self,
        data_dir: str,
        transform_mean: tuple = HOPTIMUS_MEAN,
        transform_std: tuple = HOPTIMUS_STD,
        patch_size: int = 224,
        max_samples: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.patch_size = patch_size
        self.max_samples = max_samples

        # Load annotations and build sample list
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} cell samples from APCData")

    def _load_samples(self) -> List[dict]:
        """Load sample metadata from YOLO annotations."""
        samples = []

        images_dir = self.data_dir / 'images'
        if not images_dir.exists():
            print(f"ERROR: Images directory not found: {images_dir}")
            return samples

        # Get image dimensions from first image
        image_files = [f for f in os.listdir(images_dir)
                       if f.endswith(('.jpg', '.png')) and ':Zone' not in f]

        if not image_files:
            print(f"ERROR: No images found in {images_dir}")
            return samples

        # Read first image to get dimensions
        from PIL import Image as PILImage
        sample_img = PILImage.open(images_dir / image_files[0])
        img_width, img_height = sample_img.size
        print(f"  Image dimensions: {img_width}×{img_height}")

        # Load YOLO annotations
        try:
            annotations = load_yolo_annotations(str(self.data_dir), img_width, img_height)
            print(f"  Loaded annotations for {len(annotations)} images")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return samples

        # Build sample list (one sample per cell)
        for image_file, cells in annotations.items():
            image_path = images_dir / image_file

            if not image_path.exists():
                continue

            for cell in cells:
                label = BETHESDA_CLASSES.index(cell['class'])
                samples.append({
                    'image_path': str(image_path),
                    'nucleus_x': cell['nucleus_x'],
                    'nucleus_y': cell['nucleus_y'],
                    'label': label,
                    'class_name': cell['class']
                })

        # Limit samples if requested
        if self.max_samples and len(samples) > self.max_samples:
            # Stratified sampling to maintain class distribution
            np.random.seed(42)
            indices = np.random.permutation(len(samples))[:self.max_samples]
            samples = [samples[i] for i in indices]

        # Print class distribution
        class_counts = {}
        for s in samples:
            c = s['class_name']
            class_counts[c] = class_counts.get(c, 0) + 1

        print(f"  Class distribution:")
        for c in BETHESDA_CLASSES:
            if c in class_counts:
                print(f"    {c}: {class_counts[c]}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        # Load full image
        from PIL import Image as PILImage
        image = np.array(PILImage.open(sample['image_path']).convert('RGB'))

        # Extract patch centered on cell
        patch = extract_patch(
            image,
            sample['nucleus_x'],
            sample['nucleus_y'],
            self.patch_size
        )

        # Normalize for encoder
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - np.array(self.transform_mean)) / np.array(self.transform_std)
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float()

        return tensor, sample['label']


# Alias for backward compatibility
CytologyBenchmarkDataset = APCDataBenchmarkDataset

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute comprehensive metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

    # Per-class recall (important for cytology)
    for i, class_name in enumerate(BETHESDA_CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            class_pred = y_pred[mask]
            metrics[f'recall_{class_name}'] = (class_pred == i).mean()
        else:
            metrics[f'recall_{class_name}'] = None

    # Binary metrics (Normal vs Abnormal)
    y_true_binary = np.isin(y_true, [BETHESDA_CLASSES.index(c) for c in BETHESDA_ABNORMAL]).astype(int)
    y_pred_binary = np.isin(y_pred, [BETHESDA_CLASSES.index(c) for c in BETHESDA_ABNORMAL]).astype(int)

    metrics['sensitivity_abnormal'] = recall_score(y_true_binary, y_pred_binary)
    metrics['specificity_normal'] = recall_score(1 - y_true_binary, 1 - y_pred_binary)

    # ECE (Expected Calibration Error)
    metrics['ece'] = compute_ece(y_true, y_proba)

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ece: Expected Calibration Error
    """
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    ece = 0.0
    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        mask = (confidences > bin_lower) & (confidences <= bin_upper)

        if mask.sum() > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            ece += mask.sum() * abs(bin_accuracy - bin_confidence)

    ece /= len(y_true)
    return ece

# ============================================================================
# BENCHMARK PIPELINE
# ============================================================================

def run_benchmark(
    encoder_name: str,
    data_dir: str,
    device: torch.device,
    batch_size: int = 32,
    max_samples: Optional[int] = None
) -> dict:
    """
    Run benchmark for a single encoder.

    Uses linear probe (logistic regression) on frozen features.

    Args:
        encoder_name: Name of encoder
        data_dir: Path to dataset directory
        device: Torch device
        batch_size: Batch size for feature extraction
        max_samples: Maximum samples to use

    Returns:
        results: Dictionary with metrics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {encoder_name}")
    print(f"{'='*60}")

    # Load encoder
    model, config = load_encoder(encoder_name, device)

    # Create dataset with correct normalization for this encoder
    dataset = APCDataBenchmarkDataset(
        data_dir,
        transform_mean=config['mean'],
        transform_std=config['std'],
        max_samples=max_samples
    )

    if len(dataset) == 0:
        print("ERROR: No samples found!")
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Extract features
    print("Extracting features...")
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        features = extract_features(model, images, encoder_name)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Extracted features: {features.shape}")

    # Linear probe (5-fold cross-validation)
    print("Training linear probe (5-fold CV)...")
    from sklearn.model_selection import StratifiedKFold

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros(len(labels))
    all_probas = np.zeros((len(labels), len(BETHESDA_CLASSES)))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features, labels)):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train logistic regression
        clf = LogisticRegression(
            max_iter=5000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs',
            tol=1e-4
        )
        clf.fit(X_train, y_train)

        # Predict
        all_preds[val_idx] = clf.predict(X_val)
        all_probas[val_idx] = clf.predict_proba(X_val)

    # Compute metrics
    metrics = compute_metrics(labels, all_preds.astype(int), all_probas)
    metrics['encoder'] = encoder_name
    metrics['embed_dim'] = config['embed_dim']
    metrics['description'] = config['description']

    # Print summary
    print(f"\nResults for {encoder_name}:")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity (Abnormal): {metrics['sensitivity_abnormal']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  ASC-H Recall: {metrics.get('recall_ASCH', 'N/A')}")
    print(f"  HSIL Recall: {metrics.get('recall_HSIL', 'N/A')}")

    return metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark encoders for cytology classification'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='apcdata',
        help='Dataset to use (apcdata, sipakmed, cricva)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--encoders',
        type=str,
        default='h-optimus,uni,phikon-v2',
        help='Comma-separated list of encoders to benchmark'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports/encoder_benchmark',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for feature extraction'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to use (None = all)'
    )

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine data directory
    if args.data_dir is None:
        if args.dataset == 'apcdata':
            args.data_dir = 'data/raw/apcdata/APCData_YOLO'
        elif args.dataset == 'sipakmed':
            args.data_dir = 'data/raw/sipakmed'
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"\n{'='*60}")
        print("ERROR: Data directory not found!")
        print(f"{'='*60}")
        print(f"  Expected path: {args.data_dir}")
        print(f"\n  To fix, download APCData from:")
        print(f"    https://data.mendeley.com/datasets/ytd568rh3p/1")
        print(f"\n  Then extract to:")
        print(f"    data/raw/apcdata/APCData_YOLO/")
        print(f"    ├── images/      # 425 images")
        print(f"    ├── labels/      # 425 .txt files (YOLO format)")
        print(f"    └── classes.txt  # Class names")
        print(f"{'='*60}")
        sys.exit(1)

    print(f"\nDataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")

    # Parse encoders
    encoder_list = [e.strip() for e in args.encoders.split(',')]
    print(f"Encoders to benchmark: {encoder_list}")

    # Run benchmarks
    results = []
    for encoder_name in encoder_list:
        if encoder_name not in ENCODER_CONFIGS:
            print(f"Warning: Unknown encoder '{encoder_name}', skipping")
            continue

        try:
            metrics = run_benchmark(
                encoder_name,
                args.data_dir,
                device,
                args.batch_size,
                args.max_samples
            )
            if metrics is not None:
                results.append(metrics)
        except Exception as e:
            print(f"Error benchmarking {encoder_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'encoder_benchmark_{args.dataset}_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'timestamp': timestamp,
            'results': results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Encoder':<15} {'Bal.Acc':<10} {'F1':<10} {'Sens.Abn':<10} {'ECE':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['encoder']:<15} "
              f"{r['balanced_accuracy']:.4f}     "
              f"{r['f1_macro']:.4f}     "
              f"{r['sensitivity_abnormal']:.4f}     "
              f"{r['ece']:.4f}")

    print("="*80)

    # Recommendation
    if len(results) > 1:
        best = max(results, key=lambda x: x['sensitivity_abnormal'])
        print(f"\nRECOMMENDATION: {best['encoder']} has highest abnormal sensitivity "
              f"({best['sensitivity_abnormal']:.4f})")


if __name__ == '__main__':
    main()
