#!/usr/bin/env python3
"""
Tests unitaires pour la calibration des modèles.

Usage:
    pytest tests/unit/test_calibration.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.calibration.temperature_scaling import (
    TemperatureScaler,
    reliability_diagram,
    compute_calibration_metrics,
)


class TestTemperatureScaler:
    """Tests pour TemperatureScaler."""

    def test_initialization(self):
        """Initialisation correcte."""
        scaler = TemperatureScaler()
        assert scaler.temperature == 1.0

    def test_scale(self):
        """Scale divise par température."""
        scaler = TemperatureScaler(temperature=2.0)

        logits = np.array([[1.0, 2.0, 3.0]])
        scaled = scaler.scale(logits)

        expected = np.array([[0.5, 1.0, 1.5]])
        np.testing.assert_array_almost_equal(scaled, expected)

    def test_predict_proba_valid(self):
        """predict_proba retourne des probabilités valides."""
        scaler = TemperatureScaler(temperature=1.5)

        logits = np.random.randn(10, 5)
        probs = scaler.predict_proba(logits)

        # Vérifie que ce sont des probabilités
        assert probs.shape == logits.shape
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(10))

    def test_fit_returns_positive_temperature(self):
        """Fit retourne une température positive."""
        logits = np.random.randn(100, 5) * 2
        labels = np.random.randint(0, 5, 100)

        scaler = TemperatureScaler()
        T = scaler.fit(logits, labels)

        assert T > 0
        assert scaler.temperature == T

    def test_fit_nll_method(self):
        """Fit avec méthode NLL."""
        logits = np.random.randn(100, 5) * 3
        labels = np.random.randint(0, 5, 100)

        scaler = TemperatureScaler()
        T = scaler.fit(logits, labels, method="nll")

        assert T > 0

    def test_fit_ece_method(self):
        """Fit avec méthode ECE."""
        logits = np.random.randn(100, 5) * 3
        labels = np.random.randint(0, 5, 100)

        scaler = TemperatureScaler()
        T = scaler.fit(logits, labels, method="ece")

        assert T > 0


class TestReliabilityDiagram:
    """Tests pour reliability_diagram."""

    def test_output_structure(self):
        """Structure de sortie correcte."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        bin_centers, bin_accs, bin_counts = reliability_diagram(probs, labels, n_bins=10)

        assert len(bin_centers) == 10
        assert len(bin_accs) == 10
        assert len(bin_counts) == 10

    def test_bin_centers_range(self):
        """Centres des bins dans [0, 1]."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        bin_centers, _, _ = reliability_diagram(probs, labels)

        assert np.all(bin_centers >= 0)
        assert np.all(bin_centers <= 1)

    def test_bin_accuracies_range(self):
        """Accuracies dans [0, 1]."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        _, bin_accs, bin_counts = reliability_diagram(probs, labels)

        # Seulement vérifier les bins non-vides
        non_empty = bin_counts > 0
        assert np.all(bin_accs[non_empty] >= 0)
        assert np.all(bin_accs[non_empty] <= 1)


class TestCalibrationMetrics:
    """Tests pour compute_calibration_metrics."""

    def test_metrics_structure(self):
        """Structure des métriques correcte."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        metrics = compute_calibration_metrics(probs, labels)

        assert 'accuracy' in metrics
        assert 'ECE' in metrics
        assert 'MCE' in metrics
        assert 'Brier' in metrics
        assert 'mean_confidence' in metrics

    def test_ece_range(self):
        """ECE dans [0, 1]."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        metrics = compute_calibration_metrics(probs, labels)

        assert 0 <= metrics['ECE'] <= 1

    def test_mce_range(self):
        """MCE dans [0, 1]."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        metrics = compute_calibration_metrics(probs, labels)

        assert 0 <= metrics['MCE'] <= 1

    def test_accuracy_range(self):
        """Accuracy dans [0, 1]."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        metrics = compute_calibration_metrics(probs, labels)

        assert 0 <= metrics['accuracy'] <= 1


class TestIntegration:
    """Tests d'intégration pour le pipeline de calibration."""

    def test_full_pipeline(self):
        """Pipeline complet: fit -> transform -> metrics."""
        n_train = 200
        n_classes = 5

        # Données
        logits = np.random.randn(n_train, n_classes) * 3
        labels = np.random.randint(0, n_classes, n_train)

        # Calibration
        scaler = TemperatureScaler()
        T = scaler.fit(logits, labels)

        assert T > 0

        # Prédictions calibrées
        probs_calibrated = scaler.predict_proba(logits)

        # Métriques
        metrics = compute_calibration_metrics(probs_calibrated, labels)

        assert 0 <= metrics['ECE'] <= 1
        assert 0 <= metrics['accuracy'] <= 1

    def test_calibration_reduces_ece(self):
        """La calibration devrait réduire ou maintenir l'ECE."""
        from scipy.special import softmax

        # Logits sur-confiants
        logits = np.random.randn(200, 5) * 5
        labels = np.random.randint(0, 5, 200)

        # ECE avant
        probs_before = softmax(logits, axis=-1)
        metrics_before = compute_calibration_metrics(probs_before, labels)

        # Calibration
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        probs_after = scaler.predict_proba(logits)

        # ECE après
        metrics_after = compute_calibration_metrics(probs_after, labels)

        # L'ECE devrait diminuer ou rester stable
        # (peut ne pas toujours diminuer sur données aléatoires)
        assert metrics_after['ECE'] <= metrics_before['ECE'] + 0.1


class TestEdgeCases:
    """Tests pour cas limites."""

    def test_single_class(self):
        """Gère le cas où toutes les prédictions sont la même classe."""
        probs = np.zeros((10, 5))
        probs[:, 0] = 1.0
        labels = np.zeros(10, dtype=int)

        metrics = compute_calibration_metrics(probs, labels)

        assert metrics['accuracy'] == 1.0
        assert metrics['ECE'] < 0.1

    def test_small_dataset(self):
        """Gère les petits datasets."""
        probs = np.random.rand(5, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 3, 5)

        metrics = compute_calibration_metrics(probs, labels)

        assert 'ECE' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
