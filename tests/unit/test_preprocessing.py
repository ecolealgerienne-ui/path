"""
Tests unitaires pour le module src.preprocessing.

Ces tests garantissent que le preprocessing est correct et cohérent,
évitant ainsi les bugs historiques (ToPILImage, LayerNorm, etc.).
"""

import pytest
import torch
import numpy as np

from src.preprocessing import (
    HOPTIMUS_MEAN,
    HOPTIMUS_STD,
    HOPTIMUS_IMAGE_SIZE,
    create_hoptimus_transform,
    preprocess_image,
    validate_features,
    get_preprocessing_info,
)


class TestConstants:
    """Tests sur les constantes de normalisation."""

    def test_constants_are_tuples(self):
        """Les constantes doivent être des tuples (pas np.array)."""
        assert isinstance(HOPTIMUS_MEAN, tuple), "HOPTIMUS_MEAN doit être un tuple"
        assert isinstance(HOPTIMUS_STD, tuple), "HOPTIMUS_STD doit être un tuple"

    def test_constants_have_correct_length(self):
        """Les constantes doivent avoir 3 valeurs (RGB)."""
        assert len(HOPTIMUS_MEAN) == 3, "HOPTIMUS_MEAN doit avoir 3 valeurs (RGB)"
        assert len(HOPTIMUS_STD) == 3, "HOPTIMUS_STD doit avoir 3 valeurs (RGB)"

    def test_constants_are_immutable(self):
        """Les constantes ne doivent pas être modifiables."""
        with pytest.raises(TypeError):
            HOPTIMUS_MEAN[0] = 0.5  # Tuples sont immutables

    def test_constants_values(self):
        """Vérifier les valeurs exactes (définies par Bioptimus)."""
        assert HOPTIMUS_MEAN == (0.707223, 0.578729, 0.703617)
        assert HOPTIMUS_STD == (0.211883, 0.230117, 0.177517)

    def test_image_size(self):
        """Taille d'image doit être 224."""
        assert HOPTIMUS_IMAGE_SIZE == 224


class TestCreateHoptimusTransform:
    """Tests sur create_hoptimus_transform()."""

    def test_returns_compose(self):
        """Doit retourner un transforms.Compose."""
        from torchvision import transforms
        transform = create_hoptimus_transform()
        assert isinstance(transform, transforms.Compose)

    def test_transform_is_deterministic(self):
        """Le transform doit être déterministe."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        np.random.seed(42)

        transform = create_hoptimus_transform()
        result1 = transform(image)
        result2 = transform(image)

        assert torch.allclose(result1, result2), "Transform non déterministe!"

    def test_transform_output_shape(self):
        """Sortie doit être (3, 224, 224)."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transform = create_hoptimus_transform()
        result = transform(image)

        assert result.shape == (3, 224, 224), f"Shape attendue (3, 224, 224), obtenu {result.shape}"

    def test_transform_output_range(self):
        """Sortie doit être dans une plage raisonnable après normalisation."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transform = create_hoptimus_transform()
        result = transform(image)

        # Après normalisation, valeurs typiques entre -3 et +3
        assert result.min() > -5 and result.max() < 5, "Valeurs hors plage après normalisation"


class TestPreprocessImage:
    """Tests sur preprocess_image()."""

    def test_accepts_uint8(self):
        """Doit accepter uint8 [0-255]."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        tensor = preprocess_image(image, device="cpu")
        assert tensor.shape == (1, 3, 224, 224)

    def test_accepts_float_0_1(self):
        """Doit accepter float [0-1]."""
        image = np.random.rand(256, 256, 3).astype(np.float32)
        tensor = preprocess_image(image, device="cpu")
        assert tensor.shape == (1, 3, 224, 224)

    def test_accepts_float_0_255(self):
        """Doit accepter float [0-255]."""
        image = np.random.rand(256, 256, 3).astype(np.float64) * 255
        tensor = preprocess_image(image, device="cpu")
        assert tensor.shape == (1, 3, 224, 224)

    def test_uint8_conversion_correctness(self):
        """
        BUG 2025-12-20: ToPILImage multiplie les floats par 255.
        Tester que la conversion uint8 évite ce bug.
        """
        # Image de test avec valeurs connues
        img_uint8 = np.array([[[100, 150, 200]]], dtype=np.uint8)  # (1, 1, 3)

        # Preprocessing
        tensor_uint8 = preprocess_image(img_uint8, device="cpu")

        # Image float64 équivalente
        img_float64 = img_uint8.astype(np.float64)
        tensor_float64 = preprocess_image(img_float64, device="cpu")

        # Les deux doivent être IDENTIQUES (à epsilon près)
        assert torch.allclose(tensor_uint8, tensor_float64, atol=1e-3), (
            "Conversion uint8 incorrecte! Bug ToPILImage détecté."
        )

    def test_rejects_invalid_shape(self):
        """Doit rejeter les images avec shape invalide."""
        # 2D image
        with pytest.raises(ValueError, match="Expected 3D image"):
            preprocess_image(np.random.rand(256, 256), device="cpu")

        # 4 canaux
        with pytest.raises(ValueError, match="Expected RGB image"):
            preprocess_image(np.random.rand(256, 256, 4), device="cpu")

    def test_rejects_non_numpy(self):
        """Doit rejeter les types non-numpy."""
        with pytest.raises(TypeError):
            preprocess_image([[1, 2, 3]], device="cpu")

    def test_device_placement(self):
        """Doit placer le tensor sur le bon device."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        tensor_cpu = preprocess_image(image, device="cpu")
        assert tensor_cpu.device.type == "cpu"

        # Note: test cuda seulement si disponible
        if torch.cuda.is_available():
            tensor_cuda = preprocess_image(image, device="cuda")
            assert tensor_cuda.device.type == "cuda"

    def test_invalid_device(self):
        """Doit rejeter les devices invalides."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Invalid device"):
            preprocess_image(image, device="invalid")


class TestValidateFeatures:
    """Tests sur validate_features()."""

    def test_valid_features(self):
        """Features valides doivent passer la validation."""
        # Créer features simulées avec CLS std dans la plage
        features = torch.randn(1, 261, 1536) * 0.77  # std ~0.77

        validation = validate_features(features)

        assert validation["valid"] == True
        assert 0.70 <= validation["cls_std"] <= 0.90
        assert "✅" in validation["message"]

    def test_invalid_features_low_std(self):
        """Features avec CLS std bas doivent échouer."""
        # CLS std ~0.28 (bug LayerNorm)
        features = torch.randn(1, 261, 1536) * 0.28

        validation = validate_features(features)

        assert validation["valid"] == False
        assert validation["cls_std"] < 0.40
        assert "LayerNorm" in validation["message"]

    def test_invalid_features_high_std(self):
        """Features avec CLS std élevé doivent échouer."""
        # CLS std ~1.5
        features = torch.randn(1, 261, 1536) * 1.5

        validation = validate_features(features)

        assert validation["valid"] == False
        assert validation["cls_std"] > 0.90

    def test_invalid_shape_2d(self):
        """Shape 2D doit échouer."""
        features = torch.randn(261, 1536)

        validation = validate_features(features)

        assert validation["valid"] == False
        assert "Shape invalide" in validation["message"]

    def test_invalid_shape_wrong_tokens(self):
        """Mauvais nombre de tokens doit échouer."""
        features = torch.randn(1, 197, 1536)  # ViT-B (pas H-optimus-0)

        validation = validate_features(features)

        assert validation["valid"] == False
        assert "261" in validation["message"]

    def test_invalid_shape_wrong_embed_dim(self):
        """Mauvaise dimension embedding doit échouer."""
        features = torch.randn(1, 261, 768)  # 768 au lieu de 1536

        validation = validate_features(features)

        assert validation["valid"] == False
        assert "1536" in validation["message"]

    def test_batch_size_flexible(self):
        """Doit accepter différentes tailles de batch."""
        for batch_size in [1, 4, 8]:
            features = torch.randn(batch_size, 261, 1536) * 0.77
            validation = validate_features(features)
            assert validation["valid"] == True


class TestGetPreprocessingInfo:
    """Tests sur get_preprocessing_info()."""

    def test_returns_dict(self):
        """Doit retourner un dict."""
        info = get_preprocessing_info()
        assert isinstance(info, dict)

    def test_contains_expected_keys(self):
        """Doit contenir les clés attendues."""
        info = get_preprocessing_info()

        assert "version" in info
        assert "mean" in info
        assert "std" in info
        assert "image_size" in info
        assert "method" in info

    def test_values_match_constants(self):
        """Les valeurs doivent correspondre aux constantes."""
        info = get_preprocessing_info()

        assert info["mean"] == HOPTIMUS_MEAN
        assert info["std"] == HOPTIMUS_STD
        assert info["image_size"] == HOPTIMUS_IMAGE_SIZE


# ============================================================================
# TESTS D'INTÉGRATION (avec vrai modèle)
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.slow
class TestPreprocessingWithModel:
    """
    Tests d'intégration avec le vrai modèle H-optimus-0.

    Ces tests nécessitent:
    - CUDA disponible
    - Token HuggingFace valide
    - Connexion internet

    Exécution: pytest tests/unit/test_preprocessing.py -m slow
    """

    def test_full_pipeline_cls_std(self):
        """Test complet: preprocessing + extraction + validation."""
        from src.models.loader import ModelLoader

        # Charger backbone
        backbone = ModelLoader.load_hoptimus0(device="cuda")

        # Image de test
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Preprocessing
        tensor = preprocess_image(image, device="cuda")

        # Extraction features
        features = backbone.forward_features(tensor)

        # Validation
        validation = validate_features(features)

        # CLS std doit être dans [0.70, 0.90]
        assert validation["valid"], validation["message"]
        assert 0.70 <= validation["cls_std"] <= 0.90


if __name__ == "__main__":
    # Exécution rapide (sans tests lents)
    pytest.main([__file__, "-v", "-m", "not slow"])
