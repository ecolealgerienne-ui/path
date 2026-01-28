#!/usr/bin/env python3
"""
Train Local Nucleus Classifier on PanNuke Dataset.

Trains a Random Forest classifier on morphological features extracted from
PanNuke ground truth data. The resulting model provides context-independent
nucleus classification for consistent WSI visualization.

Usage:
    python scripts/training/train_local_classifier.py \
        --pannuke_dir /path/to/PanNuke \
        --output models/local_classifier_rf.pkl \
        --n_samples 5000

Output:
    - Trained RF model (.pkl)
    - Feature importance plot (.png)
    - Classification report (.txt)

Author: CellViT-Optimus
Date: 2026-01-28
"""

import argparse
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Tuple, List, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.morphological_features import (
    extract_nucleus_features,
    MorphologicalFeatures,
)
from src.preprocessing.stain_separation import ruifrok_extract_h_channel

# Import existing preprocessing functions for consistency
# Using importlib to handle non-package script import
import importlib.util

_prep_script_path = PROJECT_ROOT / "scripts" / "preprocessing" / "prepare_v13_smart_crops.py"
_spec = importlib.util.spec_from_file_location("prepare_v13_smart_crops", _prep_script_path)
_prep_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep_module)

# Import functions from existing preprocessing pipeline (SINGLE SOURCE OF TRUTH)
extract_pannuke_instances = _prep_module.extract_pannuke_instances
normalize_mask_format = _prep_module.normalize_mask_format

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTS
# ==============================================================================

PANNUKE_CLASSES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_pannuke_fold(pannuke_dir: Path, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a PanNuke fold.

    Returns:
        (images, masks, types) where:
        - images: (N, 256, 256, 3) RGB
        - masks: (N, 256, 256, 6) instance masks per type
        - types: (N, 256, 256) type map
    """
    fold_dir = pannuke_dir / f"fold{fold}"

    images = np.load(fold_dir / "images.npy")
    masks = np.load(fold_dir / "masks.npy")
    types = np.load(fold_dir / "types.npy") if (fold_dir / "types.npy").exists() else None

    logger.info(f"Loaded fold {fold}: {len(images)} images")

    return images, masks, types


def create_instance_and_type_maps(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create instance map and type map from PanNuke masks.

    Official PanNuke format (masks shape: 256, 256, 6):
    - Channel 0: Neoplastic → type 1
    - Channel 1: Inflammatory → type 2
    - Channel 2: Connective → type 3
    - Channel 3: Dead → type 4
    - Channel 4: Epithelial → type 5
    - Channel 5: Background (always 0)

    This matches compute_np_target which uses channels 0-4 for nuclei.

    Returns:
        (instance_map, type_map) where type_map uses labels 1-5
    """
    masks = normalize_mask_format(masks)
    h, w = masks.shape[:2]

    instance_map = np.zeros((h, w), dtype=np.int32)
    type_map = np.zeros((h, w), dtype=np.int32)

    current_id = 1

    # Official PanNuke: channels 0-4 are cell types (Neoplastic to Epithelial)
    for type_idx in range(5):  # 0, 1, 2, 3, 4
        channel = masks[:, :, type_idx]
        unique_ids = np.unique(channel)
        unique_ids = unique_ids[unique_ids > 0]

        for uid in unique_ids:
            mask = channel == uid
            # Only assign if not already assigned (avoid overlap issues)
            new_mask = mask & (instance_map == 0)
            if new_mask.sum() > 0:
                instance_map[new_mask] = current_id
                type_map[new_mask] = type_idx + 1  # Labels 1-5
                current_id += 1

    return instance_map, type_map


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

def extract_features_from_pannuke(
    pannuke_dir: Path,
    n_samples: int = 5000,
    folds: List[int] = [0, 1, 2],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract morphological features from PanNuke dataset.

    Args:
        pannuke_dir: Path to PanNuke dataset
        n_samples: Maximum number of samples (nuclei) to extract
        folds: Which folds to use

    Returns:
        (features, labels) where features is (N, 6) and labels is (N,)
    """
    all_features = []
    all_labels = []

    samples_per_fold = n_samples // len(folds)

    for fold in folds:
        logger.info(f"Processing fold {fold}...")

        images, masks, _ = load_pannuke_fold(pannuke_dir, fold)

        fold_features = []
        fold_labels = []

        for img_idx in range(len(images)):
            if len(fold_features) >= samples_per_fold:
                break

            image = images[img_idx]
            mask = masks[img_idx]

            # Create instance and type maps
            instance_map, type_map = create_instance_and_type_maps(mask)

            # Pre-compute H-channel
            h_channel = ruifrok_extract_h_channel(image)

            # Extract features for each nucleus
            unique_ids = np.unique(instance_map)
            unique_ids = unique_ids[unique_ids > 0]

            for nid in unique_ids:
                if len(fold_features) >= samples_per_fold:
                    break

                # Get nucleus type from type_map
                nucleus_mask = instance_map == nid
                nucleus_type = int(np.median(type_map[nucleus_mask]))

                if nucleus_type == 0:  # Skip background
                    continue

                # Extract features
                feat = extract_nucleus_features(
                    image, instance_map, int(nid), h_channel
                )

                if feat is not None:
                    fold_features.append(feat.to_vector())
                    fold_labels.append(nucleus_type)

            if img_idx % 100 == 0:
                logger.info(f"  Image {img_idx}/{len(images)}, {len(fold_features)} nuclei extracted")

        all_features.extend(fold_features)
        all_labels.extend(fold_labels)

        logger.info(f"Fold {fold}: {len(fold_features)} nuclei extracted")

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)

    logger.info(f"Total: {len(features)} nuclei extracted")

    # Log class distribution
    for class_id in range(1, 6):
        count = np.sum(labels == class_id)
        logger.info(f"  {PANNUKE_CLASSES[class_id]}: {count} ({100*count/len(labels):.1f}%)")

    return features, labels


# ==============================================================================
# TRAINING
# ==============================================================================

def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 100,
    test_size: float = 0.2,
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train Random Forest classifier.

    Args:
        features: (N, 6) feature matrix
        labels: (N,) class labels
        n_estimators: Number of trees
        test_size: Fraction for test set

    Returns:
        (trained_model, metrics_dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )

    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

    # Train RF
    logger.info(f"Training Random Forest with {n_estimators} trees...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    logger.info(f"Test accuracy: {accuracy:.4f}")

    # Classification report
    class_names = [PANNUKE_CLASSES[i] for i in range(1, 6)]
    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    feature_names = MorphologicalFeatures.feature_names()
    importances = rf.feature_importances_

    logger.info("\nFeature Importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        logger.info(f"  {name}: {imp:.4f}")

    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'feature_importance': dict(zip(feature_names, importances)),
    }

    return rf, metrics


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train local nucleus classifier")
    parser.add_argument("--pannuke_dir", type=str, required=True,
                        help="Path to PanNuke dataset")
    parser.add_argument("--output", type=str, default="models/local_classifier_rf.pkl",
                        help="Output path for trained model")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of nuclei samples to use")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of RF trees")
    parser.add_argument("--folds", type=str, default="0,1,2",
                        help="Comma-separated fold indices to use")
    args = parser.parse_args()

    pannuke_dir = Path(args.pannuke_dir)
    output_path = Path(args.output)
    folds = [int(f) for f in args.folds.split(",")]

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract features
    logger.info("=== Feature Extraction ===")
    features, labels = extract_features_from_pannuke(
        pannuke_dir, n_samples=args.n_samples, folds=folds
    )

    # Train classifier
    logger.info("\n=== Training ===")
    rf_model, metrics = train_random_forest(
        features, labels, n_estimators=args.n_estimators
    )

    # Save model
    logger.info(f"\n=== Saving model to {output_path} ===")
    with open(output_path, "wb") as f:
        pickle.dump(rf_model, f)

    # Save report
    report_path = output_path.with_suffix(".txt")
    with open(report_path, "w") as f:
        f.write(f"Local Nucleus Classifier - Training Report\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Samples: {len(features)}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
        f.write(f"Classification Report:\n{metrics['classification_report']}\n")
        f.write(f"\nFeature Importance:\n")
        for name, imp in sorted(metrics['feature_importance'].items(), key=lambda x: -x[1]):
            f.write(f"  {name}: {imp:.4f}\n")

    logger.info(f"Report saved to {report_path}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
