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
    TemperatureScaling,
    compute_ece,
    reliability_diagram_data,
)


class TestTemperatureScaling:
    """Tests pour Temperature Scaling."""

    def test_initialization(self):
        """Initialisation correcte."""
        ts = TemperatureScaling()
        assert ts.temperature == 1.0
        assert not ts.fitted

    def test_fit_improves_calibration(self):
        """Le fit améliore la calibration."""
        # Simuler des logits sur-confiants
        n_samples = 200
        n_classes = 5

        # Logits avec biais (trop confiants)
        logits = np.random.randn(n_samples, n_classes) * 3
        labels = np.random.randint(0, n_classes, n_samples)

        ts = TemperatureScaling()
        ts.fit(logits, labels)

        assert ts.fitted
        assert ts.temperature > 0

    def test_temperature_scales_properly(self):
        """La température scale correctement les logits."""
        ts = TemperatureScaling()
        ts.temperature = 2.0
        ts.fitted = True

        logits = np.array([[1.0, 2.0, 3.0]])
        scaled = ts.transform(logits)

        expected = logits / 2.0
        np.testing.assert_array_almost_equal(scaled, expected)

    def test_predict_proba(self):
        """predict_proba retourne des probabilités valides."""
        ts = TemperatureScaling()
        ts.temperature = 1.5
        ts.fitted = True

        logits = np.random.randn(10, 5)
        probs = ts.predict_proba(logits)

        # Vérifie que ce sont des probabilités
        assert probs.shape == logits.shape
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(10))


class TestECE:
    """Tests pour Expected Calibration Error."""

    def test_perfect_calibration(self):
        """ECE = 0 pour calibration parfaite."""
        n_samples = 100

        # Prédictions parfaitement calibrées
        confidences = np.linspace(0.1, 0.9, n_samples)
        # Pour chaque confiance c, on a c% de corrects
        accuracies = (np.random.rand(n_samples) < confidences).astype(float)
        predictions = np.argmax(np.random.rand(n_samples, 5), axis=1)
        labels = np.where(accuracies, predictions, (predictions + 1) % 5)

        probs = np.zeros((n_samples, 5))
        probs[np.arange(n_samples), predictions] = confidences
        probs = probs / probs.sum(axis=1, keepdims=True)

        # ECE devrait être relativement bas
        ece = compute_ece(probs, labels, n_bins=10)
        assert ece < 0.3  # Tolérance pour la variance

    def test_overconfident_model(self):
        """ECE élevé pour modèle sur-confiant."""
        n_samples = 100
        n_classes = 5

        # Modèle très confiant mais souvent faux
        probs = np.zeros((n_samples, n_classes))
        probs[:, 0] = 0.95  # Toujours 95% confiant sur classe 0
        probs[:, 1:] = 0.05 / (n_classes - 1)

        # Mais la vraie classe est aléatoire
        labels = np.random.randint(0, n_classes, n_samples)

        ece = compute_ece(probs, labels, n_bins=10)
        # ECE devrait être élevé car confiance >> accuracy
        assert ece > 0.5

    def test_ece_bounds(self):
        """ECE est dans [0, 1]."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        ece = compute_ece(probs, labels)
        assert 0 <= ece <= 1


class TestReliabilityDiagram:
    """Tests pour le diagramme de fiabilité."""

    def test_output_structure(self):
        """Structure de sortie correcte."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        bin_edges, bin_accs, bin_confs, bin_counts = reliability_diagram_data(
            probs, labels, n_bins=10
        )

        assert len(bin_edges) == 11  # n_bins + 1 edges
        assert len(bin_accs) == 10
        assert len(bin_confs) == 10
        assert len(bin_counts) == 10

    def test_bin_accuracies_in_range(self):
        """Précisions des bins dans [0, 1]."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 5, 100)

        _, bin_accs, _, _ = reliability_diagram_data(probs, labels)

        valid_accs = bin_accs[~np.isnan(bin_accs)]
        assert np.all(valid_accs >= 0)
        assert np.all(valid_accs <= 1)


class TestIntegration:
    """Tests d'intégration pour le pipeline de calibration."""

    def test_full_pipeline(self):
        """Pipeline complet: fit -> transform -> ECE."""
        n_train = 500
        n_test = 100
        n_classes = 5

        # Données d'entraînement
        train_logits = np.random.randn(n_train, n_classes) * 2
        train_labels = np.random.randint(0, n_classes, n_train)

        # Données de test
        test_logits = np.random.randn(n_test, n_classes) * 2
        test_labels = np.random.randint(0, n_classes, n_test)

        # Calibration
        ts = TemperatureScaling()
        ts.fit(train_logits, train_labels)

        # Prédictions calibrées
        calibrated_probs = ts.predict_proba(test_logits)

        # ECE sur prédictions calibrées
        ece = compute_ece(calibrated_probs, test_labels)

        assert 0 <= ece <= 1
        assert ts.temperature > 0


class TestEdgeCases:
    """Tests pour cas limites."""

    def test_single_class(self):
        """Gère le cas mono-classe."""
        probs = np.ones((10, 1))
        labels = np.zeros(10, dtype=int)

        ece = compute_ece(probs, labels)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_empty_bins(self):
        """Gère les bins vides."""
        # Toutes les confidences dans un seul bin
        probs = np.zeros((10, 5))
        probs[:, 0] = 0.95
        probs[:, 1:] = 0.05 / 4
        labels = np.zeros(10, dtype=int)

        _, bin_accs, _, bin_counts = reliability_diagram_data(probs, labels, n_bins=10)

        # Certains bins devraient être vides
        empty_bins = bin_counts == 0
        assert np.any(empty_bins)
        # Les bins vides ont NaN accuracy
        assert np.all(np.isnan(bin_accs[empty_bins]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
