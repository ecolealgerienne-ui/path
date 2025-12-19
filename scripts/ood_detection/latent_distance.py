#!/usr/bin/env python3
"""
Détection Out-of-Distribution par distance latente.

Méthodes implémentées:
- Distance de Mahalanobis
- Distance euclidienne
- Distance cosinus

Conforme aux specs: "Détection OOD (distance Mahalanobis latente)"

Usage:
    python scripts/ood_detection/latent_distance.py --train-features train.npy --test-features test.npy
"""

import argparse
import numpy as np
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import mahalanobis, cdist
from sklearn.covariance import EmpiricalCovariance, LedoitWolf


class LatentDistanceOOD:
    """
    Détecteur OOD basé sur la distance dans l'espace latent.

    Utilise les embeddings H-optimus-0 (1536-dim) pour détecter
    les échantillons hors distribution.
    """

    def __init__(
        self,
        method: str = "mahalanobis",
        regularization: str = "ledoit-wolf"
    ):
        """
        Args:
            method: 'mahalanobis', 'euclidean', 'cosine'
            regularization: 'empirical', 'ledoit-wolf' (pour covariance)
        """
        self.method = method
        self.regularization = regularization
        self.mean = None
        self.cov_inv = None
        self.threshold = None

    def fit(self, features: np.ndarray, percentile: float = 95.0):
        """
        Ajuste le détecteur sur les données d'entraînement.

        Args:
            features: Embeddings (N, D)
            percentile: Percentile pour le seuil automatique
        """
        self.mean = features.mean(axis=0)

        if self.method == "mahalanobis":
            # Estimation de la covariance
            if self.regularization == "ledoit-wolf":
                cov_estimator = LedoitWolf()
            else:
                cov_estimator = EmpiricalCovariance()

            cov_estimator.fit(features)
            self.cov_inv = np.linalg.pinv(cov_estimator.covariance_)

        # Calculer les scores sur le training set pour le seuil
        scores = self.score(features)
        self.threshold = np.percentile(scores, percentile)

        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Calcule les scores OOD.

        Args:
            features: Embeddings (N, D)

        Returns:
            Scores (N,) - plus élevé = plus OOD
        """
        if self.method == "mahalanobis":
            # Distance de Mahalanobis
            diff = features - self.mean
            scores = np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))

        elif self.method == "euclidean":
            # Distance euclidienne au centroïde
            scores = np.linalg.norm(features - self.mean, axis=1)

        elif self.method == "cosine":
            # Distance cosinus
            norm_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            norm_mean = self.mean / (np.linalg.norm(self.mean) + 1e-8)
            scores = 1 - np.dot(norm_features, norm_mean)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return scores

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Prédit si les échantillons sont OOD.

        Args:
            features: Embeddings (N, D)

        Returns:
            Prédictions (N,) - True = OOD
        """
        scores = self.score(features)
        return scores > self.threshold

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """
        Retourne la distance au seuil (négatif = in-distribution).
        """
        scores = self.score(features)
        return scores - self.threshold


class ClassConditionalOOD:
    """
    Détecteur OOD conditionnel par classe.

    Calcule la distance à la distribution de chaque classe,
    puis prend le minimum.
    """

    def __init__(self, method: str = "mahalanobis"):
        self.method = method
        self.class_detectors: Dict[int, LatentDistanceOOD] = {}
        self.threshold = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        percentile: float = 95.0
    ):
        """
        Ajuste un détecteur par classe.

        Args:
            features: Embeddings (N, D)
            labels: Labels (N,)
            percentile: Percentile pour le seuil
        """
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            class_features = features[mask]

            detector = LatentDistanceOOD(method=self.method)
            detector.fit(class_features, percentile=100)  # Pas de seuil interne

            self.class_detectors[label] = detector

        # Calculer les scores minimaux sur le training
        min_scores = self.score(features)
        self.threshold = np.percentile(min_scores, percentile)

        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Calcule le score OOD minimal (distance à la classe la plus proche).
        """
        all_scores = []

        for detector in self.class_detectors.values():
            scores = detector.score(features)
            all_scores.append(scores)

        all_scores = np.stack(all_scores, axis=1)
        min_scores = all_scores.min(axis=1)

        return min_scores

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Prédit si OOD."""
        scores = self.score(features)
        return scores > self.threshold


def evaluate_ood_detection(
    in_scores: np.ndarray,
    out_scores: np.ndarray
) -> Dict[str, float]:
    """
    Évalue la performance de détection OOD.

    Args:
        in_scores: Scores des échantillons in-distribution
        out_scores: Scores des échantillons OOD

    Returns:
        Dict avec AUROC, FPR@95TPR, etc.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # Labels: 0 = in, 1 = out
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    scores = np.concatenate([in_scores, out_scores])

    # AUROC
    auroc = roc_auc_score(labels, scores)

    # FPR @ 95% TPR
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[idx]

    # Detection accuracy
    threshold = np.median(np.concatenate([in_scores, out_scores]))
    in_correct = (in_scores < threshold).mean()
    out_correct = (out_scores >= threshold).mean()
    detection_acc = (in_correct + out_correct) / 2

    return {
        'AUROC': auroc,
        'FPR@95TPR': fpr_at_95tpr,
        'Detection_Accuracy': detection_acc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Détection OOD par distance latente"
    )
    parser.add_argument("--train-features", type=str, required=True,
                        help="Features d'entraînement (.npy)")
    parser.add_argument("--test-features", type=str, required=True,
                        help="Features de test (.npy)")
    parser.add_argument("--ood-features", type=str,
                        help="Features OOD pour évaluation (.npy)")
    parser.add_argument("--method", type=str, default="mahalanobis",
                        choices=["mahalanobis", "euclidean", "cosine"])
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Percentile pour le seuil")

    args = parser.parse_args()

    train_features = np.load(args.train_features)
    test_features = np.load(args.test_features)

    print(f"Train features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")

    # Ajuster le détecteur
    detector = LatentDistanceOOD(method=args.method)
    detector.fit(train_features, percentile=args.percentile)

    print(f"\nMéthode: {args.method}")
    print(f"Seuil: {detector.threshold:.4f}")

    # Scores sur le test set
    test_scores = detector.score(test_features)
    test_predictions = detector.predict(test_features)

    print(f"\nTest set:")
    print(f"  Score moyen: {test_scores.mean():.4f}")
    print(f"  Score min/max: {test_scores.min():.4f} / {test_scores.max():.4f}")
    print(f"  OOD détectés: {test_predictions.sum()} / {len(test_predictions)}")

    # Évaluation si OOD features fournis
    if args.ood_features:
        ood_features = np.load(args.ood_features)
        print(f"\nOOD features: {ood_features.shape}")

        ood_scores = detector.score(ood_features)

        metrics = evaluate_ood_detection(test_scores, ood_scores)

        print("\nMétriques OOD:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
