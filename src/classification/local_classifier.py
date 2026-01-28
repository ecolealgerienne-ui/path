#!/usr/bin/env python3
"""
Local Nucleus Classifier for Consistent WSI Visualization.

Provides context-independent classification of nuclei based on morphological
features. Uses Random Forest trained on PanNuke, with rule-based fallback
if no trained model is available.

Classes (PanNuke):
1 - Neoplastic (tumoral)
2 - Inflammatory (immune cells)
3 - Connective (stromal)
4 - Dead (apoptotic/necrotic)
5 - Epithelial (normal epithelial)

Architecture:
- Primary: Random Forest trained on PanNuke morphological features
- Fallback: Rule-based classification using expert thresholds

Reference:
- Expert recommendation: RF > SVM for robustness and interpretability
- Industrial systems (QuPath, PathAI) use similar morphological approaches

Author: CellViT-Optimus
Date: 2026-01-28
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
import pickle

logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTS
# ==============================================================================

# PanNuke class mapping
PANNUKE_CLASSES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "local_classifier_rf.pkl"


# ==============================================================================
# LOCAL NUCLEUS CLASSIFIER
# ==============================================================================

class LocalNucleusClassifier:
    """
    Local nucleus classifier using morphological features.

    Provides consistent classification independent of patch context,
    essential for proper WSI visualization.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize classifier.

        Args:
            model_path: Path to trained RF model (.pkl). If None, uses fallback rules.
        """
        self.rf_model = None
        self.use_rules = True

        # Try to load trained model
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    self.rf_model = pickle.load(f)
                self.use_rules = False
                logger.info(f"Loaded RF classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load RF model: {e}. Using rule-based fallback.")
        else:
            logger.info("No trained RF model found. Using rule-based classification.")

    def classify(self, features: np.ndarray) -> int:
        """
        Classify a single nucleus.

        Args:
            features: Feature vector (6,) from MorphologicalFeatures.to_vector()

        Returns:
            Class ID (1-5)
        """
        if self.rf_model is not None:
            return int(self.rf_model.predict(features.reshape(1, -1))[0])
        else:
            return self._rule_based_classify(features)

    def classify_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Classify multiple nuclei.

        Args:
            features: Feature matrix (N, 6)

        Returns:
            Class IDs array (N,)
        """
        if len(features) == 0:
            return np.array([], dtype=np.int32)

        if self.rf_model is not None:
            return self.rf_model.predict(features).astype(np.int32)
        else:
            return np.array([
                self._rule_based_classify(f) for f in features
            ], dtype=np.int32)

    def classify_with_proba(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify with probability estimates.

        Args:
            features: Feature matrix (N, 6)

        Returns:
            (class_ids, probabilities) where probabilities is (N, 5)
        """
        if len(features) == 0:
            return np.array([], dtype=np.int32), np.zeros((0, 5), dtype=np.float32)

        if self.rf_model is not None:
            classes = self.rf_model.predict(features).astype(np.int32)
            probas = self.rf_model.predict_proba(features).astype(np.float32)
            return classes, probas
        else:
            classes = self.classify_batch(features)
            # Create dummy probabilities for rule-based
            probas = np.zeros((len(features), 5), dtype=np.float32)
            for i, c in enumerate(classes):
                probas[i, c - 1] = 0.8  # High confidence for predicted class
            return classes, probas

    def _rule_based_classify(self, features: np.ndarray) -> int:
        """
        Rule-based classification using expert thresholds.

        Features order: [area_norm, circularity, eccentricity, solidity, h_mean_norm, h_std_norm]

        Rules based on biological characteristics:
        - Inflammatory: small, round (high circularity)
        - Neoplastic: large, irregular (low circularity), high chromatin
        - Connective: elongated (high eccentricity)
        - Epithelial: medium size, regular shape
        - Dead: low chromatin intensity, irregular

        Args:
            features: Normalized feature vector (6,)

        Returns:
            Class ID (1-5)
        """
        area_norm = features[0]       # Normalized area (/ 1000)
        circularity = features[1]     # 0-1
        eccentricity = features[2]    # 0-1
        solidity = features[3]        # 0-1
        h_mean_norm = features[4]     # 0-1 (H-channel intensity)
        h_std_norm = features[5]      # 0-1 (H-channel variability)

        # Convert normalized area back to approximate pixel area
        area_px = area_norm * 1000

        # === INFLAMMATORY (2) ===
        # Small, round nuclei (lymphocytes, macrophages)
        if area_px < 300 and circularity > 0.7 and h_mean_norm > 0.5:
            return 2  # Inflammatory

        # === CONNECTIVE (3) ===
        # Elongated nuclei (fibroblasts, myofibroblasts)
        if eccentricity > 0.7 and area_px < 600:
            return 3  # Connective

        # === DEAD (4) ===
        # Low chromatin, irregular shape (apoptotic, necrotic)
        if h_mean_norm < 0.3 or (solidity < 0.7 and h_std_norm > 0.4):
            return 4  # Dead

        # === NEOPLASTIC (1) ===
        # Large, irregular, high chromatin
        if area_px > 500 and (circularity < 0.6 or h_mean_norm > 0.6):
            return 1  # Neoplastic

        # === EPITHELIAL (5) - Default ===
        # Medium size, regular shape
        return 5  # Epithelial

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from RF model.

        Returns:
            Dict mapping feature names to importance scores, or None if no RF model.
        """
        if self.rf_model is None:
            return None

        from .morphological_features import MorphologicalFeatures

        names = MorphologicalFeatures.feature_names()
        importances = self.rf_model.feature_importances_

        return {name: float(imp) for name, imp in zip(names, importances)}


def load_classifier(model_path: Optional[Path] = None) -> LocalNucleusClassifier:
    """
    Load a local nucleus classifier.

    Args:
        model_path: Path to trained model, or None for default/fallback.

    Returns:
        LocalNucleusClassifier instance
    """
    return LocalNucleusClassifier(model_path)
