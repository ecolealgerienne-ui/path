#!/usr/bin/env python3
"""
Temperature Scaling pour calibration des prédictions.

Référence: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.

Usage:
    python scripts/calibration/temperature_scaling.py --logits logits.npy --labels labels.npy
"""

import argparse
import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize_scalar
from scipy.special import softmax


class TemperatureScaler:
    """
    Calibration par Temperature Scaling.

    Divise les logits par une température T avant le softmax.
    T > 1: Réduit la confiance (moins peaky)
    T < 1: Augmente la confiance (plus peaky)
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Applique le scaling."""
        return logits / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Retourne les probabilités calibrées."""
        scaled = self.scale(logits)
        return softmax(scaled, axis=-1)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        method: str = "nll"
    ) -> float:
        """
        Optimise la température.

        Args:
            logits: Logits non-calibrés (N, C)
            labels: Labels ground truth (N,)
            method: 'nll' (Negative Log Likelihood) ou 'ece' (Expected Calibration Error)

        Returns:
            Température optimale
        """
        def objective(T):
            if T <= 0:
                return np.inf

            scaled = logits / T
            probs = softmax(scaled, axis=-1)

            if method == "nll":
                # Negative Log Likelihood
                correct_probs = probs[np.arange(len(labels)), labels]
                nll = -np.log(correct_probs + 1e-10).mean()
                return nll
            else:
                # Expected Calibration Error
                return self._compute_ece(probs, labels)

        result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x

        return self.temperature

    def _compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15
    ) -> float:
        """Calcule l'Expected Calibration Error."""
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() * np.abs(bin_acc - bin_conf)

        return ece / len(labels)


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Données pour un diagramme de fiabilité.

    Returns:
        bin_centers: Centres des bins
        bin_accuracies: Accuracy par bin
        bin_counts: Nombre d'échantillons par bin
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            bin_accuracies[i] = accuracies[mask].mean()
            bin_counts[i] = mask.sum()

    return bin_centers, bin_accuracies, bin_counts


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    Calcule les métriques de calibration.

    Returns:
        Dict avec ECE, MCE, accuracy, etc.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)

    # ECE
    scaler = TemperatureScaler()
    ece = scaler._compute_ece(probs, labels)

    # MCE (Maximum Calibration Error)
    bins = np.linspace(0, 1, 16)
    mce = 0.0
    for i in range(15):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            mce = max(mce, np.abs(bin_acc - bin_conf))

    # Brier Score
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    brier = ((probs - one_hot) ** 2).mean()

    return {
        'accuracy': accuracies.mean(),
        'mean_confidence': confidences.mean(),
        'ECE': ece,
        'MCE': mce,
        'Brier': brier,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Temperature Scaling pour calibration"
    )
    parser.add_argument("--logits", type=str, required=True, help="Logits (.npy)")
    parser.add_argument("--labels", type=str, required=True, help="Labels (.npy)")
    parser.add_argument("--output", type=str, help="Fichier de sortie pour température")

    args = parser.parse_args()

    logits = np.load(args.logits)
    labels = np.load(args.labels)

    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")

    # Avant calibration
    probs_before = softmax(logits, axis=-1)
    metrics_before = compute_calibration_metrics(probs_before, labels)

    print("\nAvant calibration:")
    for k, v in metrics_before.items():
        print(f"  {k}: {v:.4f}")

    # Calibration
    scaler = TemperatureScaler()
    T = scaler.fit(logits, labels)

    print(f"\nTempérature optimale: {T:.4f}")

    # Après calibration
    probs_after = scaler.predict_proba(logits)
    metrics_after = compute_calibration_metrics(probs_after, labels)

    print("\nAprès calibration:")
    for k, v in metrics_after.items():
        print(f"  {k}: {v:.4f}")

    if args.output:
        np.save(args.output, T)
        print(f"\nTempérature sauvegardée: {args.output}")


if __name__ == "__main__":
    main()
