#!/usr/bin/env python3
"""
Tests d'intégration end-to-end pour le pipeline CellViT-Optimus.

Usage:
    pytest tests/integration/test_pipeline_e2e.py -v

Note: Ces tests nécessitent les modèles et données.
      Utilisez --skip-slow pour ignorer les tests lents.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ajouter le chemin du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CellViT"))


# Markers pour tests conditionnels
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "requires_model: mark test as requiring model files")


class TestCellViTInference:
    """Tests d'inférence CellViT-256."""

    @pytest.fixture
    def sample_image(self):
        """Image synthétique pour tests."""
        # Image 256x256 RGB simulant du tissu H&E
        img = np.random.randint(150, 250, (256, 256, 3), dtype=np.uint8)
        # Ajouter des "noyaux" (zones plus sombres)
        for _ in range(20):
            cx, cy = np.random.randint(20, 236, 2)
            r = np.random.randint(5, 15)
            y, x = np.ogrid[:256, :256]
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            img[mask] = np.random.randint(50, 100, 3)
        return img

    @pytest.mark.requires_model
    def test_cellvit_loads(self):
        """CellViT-256 se charge correctement."""
        checkpoint_path = PROJECT_ROOT / "models" / "pretrained" / "CellViT-256.pth"
        if not checkpoint_path.exists():
            pytest.skip("Checkpoint CellViT-256.pth non trouvé")

        from src.inference.cellvit_official import CellViTOfficial
        model = CellViTOfficial(str(checkpoint_path))
        assert model is not None

    @pytest.mark.requires_model
    @pytest.mark.requires_gpu
    def test_cellvit_inference(self, sample_image):
        """Inférence CellViT-256 fonctionne."""
        checkpoint_path = PROJECT_ROOT / "models" / "pretrained" / "CellViT-256.pth"
        if not checkpoint_path.exists():
            pytest.skip("Checkpoint CellViT-256.pth non trouvé")

        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU non disponible")

        from src.inference.cellvit_official import CellViTOfficial
        model = CellViTOfficial(str(checkpoint_path))
        result = model.predict(sample_image)

        assert "cells" in result
        assert "cell_types" in result
        assert isinstance(result["cells"], list)

    @pytest.mark.requires_model
    @pytest.mark.requires_gpu
    def test_cellvit_output_format(self, sample_image):
        """Format de sortie CellViT correct."""
        checkpoint_path = PROJECT_ROOT / "models" / "pretrained" / "CellViT-256.pth"
        if not checkpoint_path.exists():
            pytest.skip("Checkpoint CellViT-256.pth non trouvé")

        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU non disponible")

        from src.inference.cellvit_official import CellViTOfficial
        model = CellViTOfficial(str(checkpoint_path))
        result = model.predict(sample_image)

        # Vérifier structure des cellules
        if len(result["cells"]) > 0:
            cell = result["cells"][0]
            assert "centroid" in cell
            assert "type" in cell
            assert "bbox" in cell or "contour" in cell


class TestGradioDemo:
    """Tests pour l'interface Gradio."""

    def test_demo_imports(self):
        """Les imports de la démo fonctionnent."""
        try:
            from scripts.demo.gradio_demo import (
                analyze_image,
                CELLVIT_AVAILABLE,
            )
            # Import réussi
            assert True
        except ImportError as e:
            pytest.fail(f"Import échoué: {e}")

    def test_synthetic_generation(self):
        """Génération d'images synthétiques fonctionne."""
        from scripts.demo.synthetic_cells import generate_synthetic_tissue

        img, cells = generate_synthetic_tissue(
            size=(256, 256),
            n_cells=50,
            cell_types=["neoplastic", "inflammatory", "connective"]
        )

        assert img.shape == (256, 256, 3)
        assert len(cells) == 50
        assert all("type" in c for c in cells)
        assert all("centroid" in c for c in cells)


class TestMetricsPipeline:
    """Tests pour le pipeline de métriques."""

    def test_metrics_computation(self):
        """Calcul des métriques sur données synthétiques."""
        from scripts.evaluation.metrics_segmentation import (
            dice_score,
            iou_score,
            panoptic_quality,
        )

        # Données de test
        pred = np.random.randint(0, 2, (256, 256))
        gt = np.random.randint(0, 2, (256, 256))

        # Calculs
        d = dice_score(pred, gt)
        iou = iou_score(pred, gt)

        assert 0 <= d <= 1
        assert 0 <= iou <= 1

    def test_pq_computation(self):
        """Calcul PQ sur instances."""
        from scripts.evaluation.metrics_segmentation import panoptic_quality

        # Instances simples
        pred = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
        ])
        gt = pred.copy()

        result = panoptic_quality(pred, gt)
        assert result["PQ"] == pytest.approx(1.0, abs=0.01)


class TestOODPipeline:
    """Tests pour le pipeline OOD."""

    def test_ood_detection_pipeline(self):
        """Pipeline OOD complet."""
        from scripts.ood_detection.latent_distance import (
            fit_reference_distribution,
            detect_ood,
        )

        # Référence
        reference = np.random.randn(100, 64)
        mean, cov_inv = fit_reference_distribution(reference)

        # Test in-distribution
        in_dist = np.random.randn(10, 64)
        is_ood_in, _ = detect_ood(in_dist, mean, cov_inv)

        # Test OOD
        ood = np.random.randn(10, 64) * 10 + 50
        is_ood_out, _ = detect_ood(ood, mean, cov_inv)

        # Plus d'OOD dans le set OOD
        assert np.mean(is_ood_out) > np.mean(is_ood_in)


class TestCalibrationPipeline:
    """Tests pour le pipeline de calibration."""

    def test_temperature_scaling_pipeline(self):
        """Pipeline Temperature Scaling complet."""
        from scripts.calibration.temperature_scaling import (
            TemperatureScaling,
            compute_ece,
        )

        # Données
        logits = np.random.randn(100, 5) * 2
        labels = np.random.randint(0, 5, 100)

        # Calibration
        ts = TemperatureScaling()
        ts.fit(logits, labels)
        probs = ts.predict_proba(logits)

        # ECE
        ece = compute_ece(probs, labels)
        assert 0 <= ece <= 1


class TestEndToEnd:
    """Tests end-to-end complets."""

    @pytest.mark.slow
    @pytest.mark.requires_model
    @pytest.mark.requires_gpu
    def test_full_analysis_pipeline(self):
        """Pipeline complet d'analyse d'image."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU non disponible")

        checkpoint_path = PROJECT_ROOT / "models" / "pretrained" / "CellViT-256.pth"
        if not checkpoint_path.exists():
            pytest.skip("Checkpoint non trouvé")

        from scripts.demo.synthetic_cells import generate_synthetic_tissue
        from src.inference.cellvit_official import CellViTOfficial

        # 1. Générer image
        img, _ = generate_synthetic_tissue(size=(256, 256), n_cells=30)

        # 2. Inférence
        model = CellViTOfficial(str(checkpoint_path))
        result = model.predict(img)

        # 3. Vérifier résultat
        assert "cells" in result
        assert "cell_types" in result

        # 4. Générer rapport
        report = model.generate_report(result)
        assert "Total" in report or "cellules" in report.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
