#!/usr/bin/env python3
"""
Détection OOD par scoring d'entropie.

Conforme aux specs: "Incertitude aléatorique (entropie NP/HV)"

L'entropie des prédictions indique l'incertitude du modèle.
Haute entropie → potentiellement OOD ou cas difficile.

Usage:
    python scripts/ood_detection/entropy_scoring.py --probs predictions.npy
"""

import argparse
import numpy as np
from typing import Dict, Tuple


def entropy(probs: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    """
    Calcule l'entropie de Shannon.

    Args:
        probs: Probabilités (N, C) ou (N, H, W, C)
        axis: Axe des classes
        eps: Epsilon pour stabilité numérique

    Returns:
        Entropie (N,) ou (N, H, W)
    """
    probs = np.clip(probs, eps, 1 - eps)
    return -np.sum(probs * np.log(probs), axis=axis)


def normalized_entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Entropie normalisée [0, 1].

    H_norm = H / log(n_classes)
    """
    n_classes = probs.shape[axis]
    max_entropy = np.log(n_classes)
    return entropy(probs, axis) / max_entropy


def max_prob(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Probabilité maximale (confiance).

    Faible max_prob → haute incertitude
    """
    return np.max(probs, axis=axis)


def margin(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Marge entre les deux probabilités les plus hautes.

    Faible marge → haute incertitude
    """
    sorted_probs = np.sort(probs, axis=axis)
    return sorted_probs[..., -1] - sorted_probs[..., -2]


class EntropyOODDetector:
    """
    Détecteur OOD basé sur l'entropie des prédictions.
    """

    def __init__(
        self,
        method: str = "entropy",
        threshold: float = None
    ):
        """
        Args:
            method: 'entropy', 'normalized_entropy', 'max_prob', 'margin'
            threshold: Seuil de décision (optionnel)
        """
        self.method = method
        self.threshold = threshold

    def fit(self, probs: np.ndarray, percentile: float = 95.0):
        """
        Ajuste le seuil sur les prédictions d'entraînement.
        """
        scores = self.score(probs)

        if self.method in ['entropy', 'normalized_entropy']:
            # Haut = OOD
            self.threshold = np.percentile(scores, percentile)
        else:
            # Bas = OOD
            self.threshold = np.percentile(scores, 100 - percentile)

        return self

    def score(self, probs: np.ndarray) -> np.ndarray:
        """
        Calcule les scores d'incertitude.

        Pour entropy: haut = incertain
        Pour max_prob/margin: bas = incertain
        """
        if self.method == "entropy":
            return entropy(probs)
        elif self.method == "normalized_entropy":
            return normalized_entropy(probs)
        elif self.method == "max_prob":
            return max_prob(probs)
        elif self.method == "margin":
            return margin(probs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Prédit si les échantillons sont incertains/OOD.
        """
        scores = self.score(probs)

        if self.method in ['entropy', 'normalized_entropy']:
            return scores > self.threshold
        else:
            return scores < self.threshold

    def get_uncertainty_map(
        self,
        probs: np.ndarray
    ) -> np.ndarray:
        """
        Génère une carte d'incertitude pour les masques de segmentation.

        Args:
            probs: Probabilités (H, W, C) ou (N, H, W, C)

        Returns:
            Carte d'incertitude (H, W) ou (N, H, W)
        """
        return normalized_entropy(probs)


def compute_uncertainty_stats(
    probs: np.ndarray,
    labels: np.ndarray = None
) -> Dict:
    """
    Calcule les statistiques d'incertitude.

    Args:
        probs: Probabilités (N, C)
        labels: Labels ground truth (optionnel)

    Returns:
        Dict avec statistiques
    """
    ent = normalized_entropy(probs)
    conf = max_prob(probs)
    marg = margin(probs)

    stats = {
        'entropy_mean': ent.mean(),
        'entropy_std': ent.std(),
        'confidence_mean': conf.mean(),
        'confidence_std': conf.std(),
        'margin_mean': marg.mean(),
        'margin_std': marg.std(),
    }

    if labels is not None:
        predictions = probs.argmax(axis=-1)
        correct = predictions == labels
        incorrect = ~correct

        if correct.sum() > 0:
            stats['entropy_correct'] = ent[correct].mean()
            stats['confidence_correct'] = conf[correct].mean()

        if incorrect.sum() > 0:
            stats['entropy_incorrect'] = ent[incorrect].mean()
            stats['confidence_incorrect'] = conf[incorrect].mean()

    return stats


def classify_predictions(
    probs: np.ndarray,
    entropy_threshold: float = 0.5,
    confidence_threshold: float = 0.7
) -> Dict[str, np.ndarray]:
    """
    Classifie les prédictions selon l'incertitude.

    Conforme aux specs:
    - Fiable: haute confiance, basse entropie
    - À revoir: incertitude moyenne
    - Hors domaine: haute entropie ou très faible confiance

    Returns:
        Dict avec masques pour chaque catégorie
    """
    ent = normalized_entropy(probs)
    conf = max_prob(probs)

    # Critères
    fiable = (ent < entropy_threshold) & (conf > confidence_threshold)
    hors_domaine = (ent > entropy_threshold * 1.5) | (conf < confidence_threshold * 0.5)
    a_revoir = ~fiable & ~hors_domaine

    return {
        'fiable': fiable,
        'a_revoir': a_revoir,
        'hors_domaine': hors_domaine,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Détection OOD par entropie"
    )
    parser.add_argument("--probs", type=str, required=True,
                        help="Probabilités (.npy)")
    parser.add_argument("--labels", type=str,
                        help="Labels optionnels (.npy)")
    parser.add_argument("--method", type=str, default="entropy",
                        choices=["entropy", "normalized_entropy", "max_prob", "margin"])

    args = parser.parse_args()

    probs = np.load(args.probs)
    print(f"Probs shape: {probs.shape}")

    labels = None
    if args.labels:
        labels = np.load(args.labels)
        print(f"Labels shape: {labels.shape}")

    # Statistiques
    stats = compute_uncertainty_stats(probs, labels)

    print("\nStatistiques d'incertitude:")
    print("-" * 40)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Classification
    classification = classify_predictions(probs)

    print("\nClassification des prédictions:")
    print("-" * 40)
    total = len(probs)
    print(f"  Fiable: {classification['fiable'].sum()} ({classification['fiable'].mean()*100:.1f}%)")
    print(f"  À revoir: {classification['a_revoir'].sum()} ({classification['a_revoir'].mean()*100:.1f}%)")
    print(f"  Hors domaine: {classification['hors_domaine'].sum()} ({classification['hors_domaine'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
