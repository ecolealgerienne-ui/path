#!/usr/bin/env python3
"""
Tests unitaires pour la détection Out-of-Distribution (OOD).

Usage:
    pytest tests/unit/test_ood.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.ood_detection.latent_distance import (
    compute_mahalanobis_distance,
    fit_reference_distribution,
    detect_ood,
)
from scripts.ood_detection.entropy_scoring import (
    compute_entropy,
    classify_confidence,
    CONFIDENCE_LEVELS,
)


class TestMahalanobisDistance:
    """Tests pour la distance de Mahalanobis."""

    def test_zero_distance_for_mean(self):
        """Distance = 0 pour un point au centre de la distribution."""
        # Distribution simple
        embeddings = np.random.randn(100, 64)
        mean, cov_inv = fit_reference_distribution(embeddings)

        distance = compute_mahalanobis_distance(mean.reshape(1, -1), mean, cov_inv)
        assert distance[0] == pytest.approx(0.0, abs=1e-5)

    def test_distance_increases_with_deviation(self):
        """Distance augmente avec l'éloignement du centre."""
        embeddings = np.random.randn(100, 64)
        mean, cov_inv = fit_reference_distribution(embeddings)

        # Points de plus en plus éloignés
        close_point = mean + 0.1 * np.std(embeddings, axis=0)
        far_point = mean + 3.0 * np.std(embeddings, axis=0)

        d_close = compute_mahalanobis_distance(close_point.reshape(1, -1), mean, cov_inv)
        d_far = compute_mahalanobis_distance(far_point.reshape(1, -1), mean, cov_inv)

        assert d_far[0] > d_close[0]

    def test_batch_processing(self):
        """Traitement par batch fonctionne."""
        embeddings = np.random.randn(100, 64)
        mean, cov_inv = fit_reference_distribution(embeddings)

        test_batch = np.random.randn(10, 64)
        distances = compute_mahalanobis_distance(test_batch, mean, cov_inv)

        assert distances.shape == (10,)
        assert all(d >= 0 for d in distances)


class TestOODDetection:
    """Tests pour la détection OOD complète."""

    def test_in_distribution_samples(self):
        """Échantillons in-distribution détectés comme tels."""
        # Créer une distribution de référence
        reference = np.random.randn(100, 64)

        # Échantillons similaires (in-distribution)
        in_dist = np.random.randn(10, 64)

        mean, cov_inv = fit_reference_distribution(reference)
        is_ood, distances = detect_ood(in_dist, mean, cov_inv, threshold_percentile=95)

        # La plupart devraient être in-distribution
        ood_ratio = np.mean(is_ood)
        assert ood_ratio < 0.3  # Moins de 30% détectés OOD

    def test_out_of_distribution_samples(self):
        """Échantillons OOD détectés comme tels."""
        # Distribution de référence centrée
        reference = np.random.randn(100, 64)

        # Échantillons très différents (OOD)
        ood_samples = np.random.randn(10, 64) * 10 + 50  # Très éloignés

        mean, cov_inv = fit_reference_distribution(reference)
        is_ood, distances = detect_ood(ood_samples, mean, cov_inv, threshold_percentile=95)

        # La plupart devraient être OOD
        ood_ratio = np.mean(is_ood)
        assert ood_ratio > 0.7  # Plus de 70% détectés OOD


class TestEntropyScoring:
    """Tests pour le scoring par entropie."""

    def test_low_entropy_confident(self):
        """Basse entropie = haute confiance."""
        # Distribution très piquée (confiant)
        probs = np.array([0.99, 0.005, 0.005])
        entropy = compute_entropy(probs)
        assert entropy < 0.1

    def test_high_entropy_uncertain(self):
        """Haute entropie = incertitude."""
        # Distribution uniforme (incertain)
        probs = np.array([0.33, 0.33, 0.34])
        entropy = compute_entropy(probs)
        assert entropy > 1.0

    def test_entropy_bounds(self):
        """Entropie dans les bornes attendues."""
        # Entropie max pour n classes = log(n)
        n_classes = 5
        uniform = np.ones(n_classes) / n_classes
        entropy = compute_entropy(uniform)
        max_entropy = np.log(n_classes)
        assert entropy == pytest.approx(max_entropy, abs=0.01)

    def test_zero_probability_handled(self):
        """Gère les probabilités nulles sans erreur."""
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = compute_entropy(probs)
        assert entropy == pytest.approx(0.0, abs=1e-5)


class TestConfidenceClassification:
    """Tests pour classification de confiance."""

    def test_fiable_classification(self):
        """Haute confiance classée 'Fiable'."""
        # Très confiant
        probs = np.array([0.98, 0.01, 0.01])
        level = classify_confidence(probs)
        assert level == CONFIDENCE_LEVELS["FIABLE"]

    def test_a_revoir_classification(self):
        """Confiance moyenne classée 'À revoir'."""
        # Moyennement confiant
        probs = np.array([0.6, 0.3, 0.1])
        level = classify_confidence(probs)
        assert level == CONFIDENCE_LEVELS["A_REVOIR"]

    def test_hors_domaine_classification(self):
        """Basse confiance classée 'Hors domaine'."""
        # Très incertain (uniforme)
        probs = np.array([0.34, 0.33, 0.33])
        level = classify_confidence(probs)
        assert level == CONFIDENCE_LEVELS["HORS_DOMAINE"]


class TestEdgeCases:
    """Tests pour cas limites."""

    def test_single_embedding(self):
        """Gère un seul embedding."""
        single = np.random.randn(1, 64)
        # Ne devrait pas crasher
        try:
            mean, cov_inv = fit_reference_distribution(single)
        except (ValueError, np.linalg.LinAlgError):
            pass  # Attendu - pas assez de données

    def test_high_dimensional(self):
        """Gère les hautes dimensions (1536 pour H-optimus-0)."""
        embeddings = np.random.randn(50, 1536)
        mean, cov_inv = fit_reference_distribution(embeddings)

        test_point = np.random.randn(1, 1536)
        distance = compute_mahalanobis_distance(test_point, mean, cov_inv)

        assert distance.shape == (1,)
        assert distance[0] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
