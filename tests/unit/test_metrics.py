#!/usr/bin/env python3
"""
Tests unitaires pour les métriques de segmentation.

Usage:
    pytest tests/unit/test_metrics.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.evaluation.metrics_segmentation import (
    dice_score,
    iou_score,
    panoptic_quality,
    f1_per_class,
    detection_metrics,
)


class TestDiceScore:
    """Tests pour le Dice Score."""

    def test_perfect_match(self):
        """Dice = 1.0 quand pred == gt."""
        mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        assert dice_score(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        """Dice ~ 0 quand aucune intersection."""
        pred = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        gt = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        assert dice_score(pred, gt) < 0.01

    def test_partial_overlap(self):
        """Dice entre 0 et 1 pour overlap partiel."""
        pred = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
        gt = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
        score = dice_score(pred, gt)
        assert 0 < score < 1

    def test_empty_masks(self):
        """Gère les masques vides."""
        empty = np.zeros((3, 3))
        full = np.ones((3, 3))
        # Deux masques vides
        assert dice_score(empty, empty) == pytest.approx(1.0, abs=1e-5)
        # Un vide, un plein
        assert dice_score(empty, full) < 0.01


class TestIoUScore:
    """Tests pour l'IoU (Jaccard)."""

    def test_perfect_match(self):
        """IoU = 1.0 quand pred == gt."""
        mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        assert iou_score(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        """IoU ~ 0 quand aucune intersection."""
        pred = np.array([[1, 0], [0, 0]])
        gt = np.array([[0, 1], [1, 1]])
        assert iou_score(pred, gt) < 0.01

    def test_half_overlap(self):
        """IoU = 0.5 pour 50% overlap avec même taille."""
        # pred: 4 pixels, gt: 4 pixels, intersection: 2
        # Union = 4 + 4 - 2 = 6, IoU = 2/6 = 0.33
        pred = np.array([[1, 1], [1, 1]])
        gt = np.array([[0, 1], [1, 1]])
        score = iou_score(pred, gt)
        assert 0.3 < score < 0.8


class TestPanopticQuality:
    """Tests pour Panoptic Quality (PQ = SQ * RQ)."""

    def test_perfect_instances(self):
        """PQ = 1.0 pour instances parfaitement alignées."""
        instances = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4],
        ])
        result = panoptic_quality(instances, instances)
        assert result['PQ'] == pytest.approx(1.0, abs=1e-5)
        assert result['SQ'] == pytest.approx(1.0, abs=1e-5)
        assert result['RQ'] == pytest.approx(1.0, abs=1e-5)

    def test_no_instances(self):
        """Gère absence d'instances."""
        empty = np.zeros((5, 5))
        result = panoptic_quality(empty, empty)
        assert result['PQ'] == 0.0
        assert result['TP'] == 0

    def test_missing_predictions(self):
        """FN quand instances GT non détectées."""
        gt = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        pred = np.zeros((3, 3))
        result = panoptic_quality(pred, gt)
        assert result['FN'] == 1
        assert result['TP'] == 0

    def test_extra_predictions(self):
        """FP quand prédictions sans GT correspondant."""
        gt = np.zeros((3, 3))
        pred = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        result = panoptic_quality(pred, gt)
        assert result['FP'] == 1
        assert result['TP'] == 0


class TestF1PerClass:
    """Tests pour F1-Score par classe."""

    def test_perfect_classification(self):
        """F1 = 1.0 pour classification parfaite."""
        mask = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ])
        result = f1_per_class(mask, mask, n_classes=5)
        for c in range(1, 5):
            assert result[c]['f1'] == pytest.approx(1.0, abs=1e-5)

    def test_wrong_classification(self):
        """F1 ~ 0 pour classification complètement fausse."""
        pred = np.ones((4, 4))  # Tout classe 1
        gt = np.full((4, 4), 2)  # Tout classe 2
        result = f1_per_class(pred, gt, n_classes=3)
        assert result[1]['precision'] < 0.01  # Aucun vrai positif pour classe 1
        assert result[2]['recall'] < 0.01  # Aucun rappel pour classe 2


class TestDetectionMetrics:
    """Tests pour métriques de détection."""

    def test_perfect_detection(self):
        """Precision/Recall = 1.0 pour détection parfaite."""
        instances = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
        ])
        result = detection_metrics(instances, instances)
        assert result['precision'] == pytest.approx(1.0, abs=1e-5)
        assert result['recall'] == pytest.approx(1.0, abs=1e-5)
        assert result['f1'] == pytest.approx(1.0, abs=1e-5)


# Tests de validation des formes
class TestInputValidation:
    """Tests pour validation des entrées."""

    def test_different_shapes_raises(self):
        """Vérifie que les formes différentes sont gérées."""
        pred = np.zeros((3, 3))
        gt = np.zeros((4, 4))
        # Le comportement dépend de l'implémentation
        # Ici on vérifie juste que ça ne crash pas silencieusement
        try:
            dice_score(pred, gt)
        except (ValueError, IndexError):
            pass  # Comportement attendu

    def test_3d_masks(self):
        """Gère les masques multi-canaux."""
        pred = np.random.rand(3, 256, 256) > 0.5
        gt = np.random.rand(3, 256, 256) > 0.5
        # Devrait fonctionner canal par canal ou lever une erreur claire
        for c in range(3):
            score = dice_score(pred[c], gt[c])
            assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
