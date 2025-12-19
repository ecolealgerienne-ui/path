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
    LatentDistanceOOD,
    ClassConditionalOOD,
    evaluate_ood_detection,
)
from scripts.ood_detection.entropy_scoring import (
    entropy,
    normalized_entropy,
    max_prob,
    margin,
    EntropyOODDetector,
    classify_predictions,
)


class TestLatentDistanceOOD:
    """Tests pour LatentDistanceOOD."""

    def test_initialization(self):
        """Initialisation correcte."""
        detector = LatentDistanceOOD(method="mahalanobis")
        assert detector.method == "mahalanobis"
        assert detector.mean is None

    def test_fit(self):
        """Fit ajuste mean et threshold."""
        embeddings = np.random.randn(100, 64)
        detector = LatentDistanceOOD()
        detector.fit(embeddings)

        assert detector.mean is not None
        assert detector.threshold is not None
        assert detector.mean.shape == (64,)

    def test_score_shape(self):
        """Score retourne la bonne forme."""
        embeddings = np.random.randn(100, 64)
        detector = LatentDistanceOOD()
        detector.fit(embeddings)

        test_data = np.random.randn(10, 64)
        scores = detector.score(test_data)

        assert scores.shape == (10,)
        assert all(s >= 0 for s in scores)

    def test_predict(self):
        """Predict retourne des booléens."""
        embeddings = np.random.randn(100, 64)
        detector = LatentDistanceOOD()
        detector.fit(embeddings, percentile=90)

        test_data = np.random.randn(10, 64)
        predictions = detector.predict(test_data)

        assert predictions.dtype == bool
        assert predictions.shape == (10,)

    def test_in_distribution_low_score(self):
        """Échantillons in-distribution ont des scores bas."""
        embeddings = np.random.randn(100, 64)
        detector = LatentDistanceOOD()
        detector.fit(embeddings)

        # Points similaires
        in_dist = np.random.randn(10, 64)
        in_scores = detector.score(in_dist)

        # Points très éloignés
        ood = np.random.randn(10, 64) * 10 + 50
        ood_scores = detector.score(ood)

        # OOD devrait avoir des scores plus élevés
        assert np.mean(ood_scores) > np.mean(in_scores)

    def test_different_methods(self):
        """Test des différentes méthodes."""
        embeddings = np.random.randn(100, 64)

        for method in ["mahalanobis", "euclidean", "cosine"]:
            detector = LatentDistanceOOD(method=method)
            detector.fit(embeddings)

            test_data = np.random.randn(10, 64)
            scores = detector.score(test_data)

            assert scores.shape == (10,)


class TestClassConditionalOOD:
    """Tests pour ClassConditionalOOD."""

    def test_fit_with_labels(self):
        """Fit avec labels crée un détecteur par classe."""
        features = np.random.randn(100, 64)
        labels = np.random.randint(0, 3, 100)

        detector = ClassConditionalOOD()
        detector.fit(features, labels)

        assert len(detector.class_detectors) == 3
        assert detector.threshold is not None


class TestEntropyFunctions:
    """Tests pour les fonctions d'entropie."""

    def test_entropy_uniform(self):
        """Entropie max pour distribution uniforme."""
        n_classes = 5
        uniform = np.ones((10, n_classes)) / n_classes
        ent = entropy(uniform)

        expected = np.log(n_classes)
        np.testing.assert_array_almost_equal(ent, expected)

    def test_entropy_peaked(self):
        """Entropie basse pour distribution piquée."""
        probs = np.array([[0.99, 0.005, 0.005]])
        ent = entropy(probs)

        assert ent[0] < 0.1

    def test_normalized_entropy_range(self):
        """Entropie normalisée dans [0, 1]."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)

        norm_ent = normalized_entropy(probs)

        assert np.all(norm_ent >= 0)
        assert np.all(norm_ent <= 1)

    def test_max_prob(self):
        """max_prob retourne la probabilité max."""
        probs = np.array([[0.1, 0.6, 0.3], [0.8, 0.1, 0.1]])
        mp = max_prob(probs)

        np.testing.assert_array_almost_equal(mp, [0.6, 0.8])

    def test_margin(self):
        """margin calcule la différence entre top-2."""
        probs = np.array([[0.1, 0.6, 0.3]])
        m = margin(probs)

        assert m[0] == pytest.approx(0.3, abs=1e-5)  # 0.6 - 0.3


class TestEntropyOODDetector:
    """Tests pour EntropyOODDetector."""

    def test_fit_sets_threshold(self):
        """Fit établit le seuil."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)

        detector = EntropyOODDetector()
        detector.fit(probs)

        assert detector.threshold is not None

    def test_predict_shape(self):
        """Predict retourne la bonne forme."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)

        detector = EntropyOODDetector()
        detector.fit(probs)

        test_probs = np.random.rand(10, 5)
        test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

        predictions = detector.predict(test_probs)

        assert predictions.dtype == bool
        assert predictions.shape == (10,)


class TestClassifyPredictions:
    """Tests pour classify_predictions."""

    def test_classification_coverage(self):
        """Toutes les prédictions sont classifiées."""
        probs = np.random.rand(100, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)

        result = classify_predictions(probs)

        # Chaque échantillon dans exactement une catégorie
        total = result['fiable'] | result['a_revoir'] | result['hors_domaine']
        assert total.all()

    def test_high_confidence_is_fiable(self):
        """Haute confiance classée fiable."""
        # Très confiant
        probs = np.zeros((10, 5))
        probs[:, 0] = 0.95
        probs[:, 1:] = 0.05 / 4

        result = classify_predictions(probs)

        assert result['fiable'].sum() > 5  # La plupart fiables


class TestEvaluateOODDetection:
    """Tests pour evaluate_ood_detection."""

    def test_perfect_separation(self):
        """AUROC = 1 pour séparation parfaite."""
        in_scores = np.array([0.1, 0.2, 0.3, 0.4])
        out_scores = np.array([0.9, 0.95, 0.85, 0.92])

        metrics = evaluate_ood_detection(in_scores, out_scores)

        assert metrics['AUROC'] == pytest.approx(1.0, abs=0.01)

    def test_random_scores(self):
        """AUROC ~ 0.5 pour scores aléatoires."""
        in_scores = np.random.rand(100)
        out_scores = np.random.rand(100)

        metrics = evaluate_ood_detection(in_scores, out_scores)

        assert 0.3 < metrics['AUROC'] < 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
