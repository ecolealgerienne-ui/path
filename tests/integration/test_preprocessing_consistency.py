"""
Tests d'intégration pour la cohérence du preprocessing.

CES TESTS SONT CRITIQUES pour éviter les bugs historiques:
- 2025-12-20: ToPILImage float64 → Features corrompues
- 2025-12-21: LayerNorm mismatch (blocks[23] vs forward_features)

Ces tests garantissent qu'on n'aura PLUS JAMAIS ces bugs.
"""

import pytest
import torch
import numpy as np
import subprocess
from pathlib import Path

from src.preprocessing import preprocess_image, validate_features, HOPTIMUS_MEAN, HOPTIMUS_STD


class TestConstantsNotDuplicated:
    """
    Vérifie qu'aucune constante n'est dupliquée dans le code.

    OBJECTIF: Garantir une source unique de vérité.
    """

    def test_no_local_hoptimus_mean(self):
        """Aucun fichier ne doit définir HOPTIMUS_MEAN localement."""
        result = subprocess.run(
            [
                "grep", "-r",
                "HOPTIMUS_MEAN\\s*=",
                "src/", "scripts/",
                "--include=*.py"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent  # Project root
        )

        # Trouver toutes les définitions
        definitions = [line for line in result.stdout.split('\n') if line.strip()]

        # Filtrer pour garder seulement les vraies définitions (pas imports)
        true_definitions = [
            line for line in definitions
            if "import" not in line.lower()
            and "from src.preprocessing" not in line.lower()
        ]

        # Autoriser SEULEMENT src/preprocessing/__init__.py
        allowed_files = ["src/preprocessing/__init__.py"]
        invalid_definitions = [
            line for line in true_definitions
            if not any(allowed in line for allowed in allowed_files)
        ]

        assert len(invalid_definitions) == 0, (
            f"❌ Constantes HOPTIMUS_MEAN trouvées en dehors de src/preprocessing/!\n"
            f"Fichiers concernés:\n"
            + "\n".join(invalid_definitions) +
            "\n\nTOUTES les constantes doivent être importées depuis src.preprocessing"
        )

    def test_no_local_create_transform(self):
        """Aucun fichier ne doit définir create_hoptimus_transform localement."""
        result = subprocess.run(
            [
                "grep", "-r",
                "def create_hoptimus_transform",
                "src/", "scripts/",
                "--include=*.py"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        definitions = [line for line in result.stdout.split('\n') if line.strip()]

        # Autoriser SEULEMENT src/preprocessing/__init__.py
        allowed_files = ["src/preprocessing/__init__.py"]
        invalid_definitions = [
            line for line in definitions
            if not any(allowed in line for allowed in allowed_files)
        ]

        assert len(invalid_definitions) == 0, (
            f"❌ Fonction create_hoptimus_transform trouvée en dehors de src/preprocessing/!\n"
            f"Fichiers concernés:\n"
            + "\n".join(invalid_definitions) +
            "\n\nTOUTE la logique de transform doit être dans src.preprocessing"
        )


class TestPreprocessingDeterminism:
    """
    Vérifie que le preprocessing est déterministe.

    OBJECTIF: Garantir cohérence train/inference.
    """

    def test_same_image_same_output(self):
        """Même image doit donner même output."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        tensor1 = preprocess_image(image, device="cpu")
        tensor2 = preprocess_image(image, device="cpu")

        assert torch.allclose(tensor1, tensor2), "Preprocessing non déterministe!"

    def test_uint8_vs_float64_equivalence(self):
        """
        BUG CRITIQUE 2025-12-20: ToPILImage multiplie floats par 255.

        Ce test garantit que ce bug ne reviendra JAMAIS.
        """
        # Image de référence uint8
        img_uint8 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Preprocessing
        tensor_uint8 = preprocess_image(img_uint8, device="cpu")

        # Conversion float64 (bug ToPILImage)
        img_float64 = img_uint8.astype(np.float64)
        tensor_float64 = preprocess_image(img_float64, device="cpu")

        # Les deux DOIVENT être identiques
        assert torch.allclose(tensor_uint8, tensor_float64, atol=1e-3), (
            "❌ BUG TOPI LImage DÉTECTÉ!\n"
            "La conversion uint8 ne fonctionne pas correctement.\n"
            "Tensors uint8 vs float64 sont différents.\n"
            "\n"
            "CAUSE: ToPILImage multiplie les floats par 255\n"
            "SOLUTION: Toujours convertir en uint8 AVANT ToPILImage\n"
            "\n"
            "Voir CLAUDE.md Section '⚠️ GUIDE CRITIQUE: BUG #1 ToPILImage'"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.slow
class TestCLSStdRange:
    """
    Vérifie que CLS std est dans la plage attendue [0.70, 0.90].

    BUG CRITIQUE 2025-12-21: blocks[23] sans LayerNorm → CLS std ~0.28
    CORRECT: forward_features() avec LayerNorm → CLS std ~0.77

    Ce test garantit que ce bug ne reviendra JAMAIS.
    """

    def test_cls_std_in_expected_range(self):
        """CLS std doit être entre 0.70 et 0.90."""
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

        assert validation["valid"], (
            f"❌ BUG LAYERNORM DÉTECTÉ!\n"
            f"CLS std={validation['cls_std']:.3f} hors plage [0.70, 0.90]\n"
            "\n"
            f"DIAGNOSTIC:\n"
            f"- Si std < 0.40: LayerNorm manquant (bug blocks[23])\n"
            f"- Si std > 1.0: Preprocessing incorrect\n"
            "\n"
            f"SOLUTION: Utiliser forward_features() au lieu de blocks[X]\n"
            "\n"
            f"Voir CLAUDE.md Section '⚠️ GUIDE CRITIQUE: BUG #2 LayerNorm Mismatch'\n"
            f"\nMessage complet: {validation['message']}"
        )

    def test_cls_std_consistency_across_batches(self):
        """CLS std doit être cohérent pour différentes images."""
        from src.models.loader import ModelLoader

        backbone = ModelLoader.load_hoptimus0(device="cuda")

        cls_stds = []

        # Tester sur 10 images aléatoires
        for _ in range(10):
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            tensor = preprocess_image(image, device="cuda")
            features = backbone.forward_features(tensor)
            validation = validate_features(features)

            cls_stds.append(validation["cls_std"])

        # Tous doivent être dans la plage
        for i, std in enumerate(cls_stds):
            assert 0.70 <= std <= 0.90, (
                f"Image {i}: CLS std={std:.3f} hors plage [0.70, 0.90]"
            )

        # Variance entre images doit être raisonnable (< 0.05)
        std_variance = np.std(cls_stds)
        assert std_variance < 0.05, (
            f"Variance CLS std trop élevée: {std_variance:.4f}\n"
            f"Indique un problème de cohérence preprocessing."
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.slow
class TestEndToEndPipeline:
    """
    Tests end-to-end complets.

    Simule le pipeline complet: image → preprocessing → features → validation
    """

    def test_multiple_images_batch(self):
        """Test avec batch de plusieurs images."""
        from src.models.loader import ModelLoader

        backbone = ModelLoader.load_hoptimus0(device="cuda")

        # Batch de 4 images
        images = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(4)
        ]

        # Preprocessing
        tensors = [preprocess_image(img, device="cuda") for img in images]
        batch = torch.cat(tensors, dim=0)  # (4, 3, 224, 224)

        # Extraction features
        features = backbone.forward_features(batch)

        # Validation
        validation = validate_features(features)

        assert validation["valid"], validation["message"]
        assert features.shape == (4, 261, 1536)

    def test_real_image_if_available(self):
        """Test avec une vraie image si disponible."""
        from src.models.loader import ModelLoader

        # Chercher images de test
        test_images_dir = Path(__file__).parent.parent / "fixtures" / "sample_images"

        if not test_images_dir.exists():
            pytest.skip("Pas d'images de test disponibles")

        # Lister images
        image_files = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg"))

        if len(image_files) == 0:
            pytest.skip("Pas d'images de test disponibles")

        # Charger backbone
        backbone = ModelLoader.load_hoptimus0(device="cuda")

        # Tester chaque image
        for image_file in image_files[:5]:  # Max 5 images
            import cv2

            # Lire image
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preprocessing
            tensor = preprocess_image(image, device="cuda")

            # Extraction
            features = backbone.forward_features(tensor)

            # Validation
            validation = validate_features(features)

            assert validation["valid"], (
                f"Image {image_file.name} échoue la validation:\n"
                f"{validation['message']}"
            )


if __name__ == "__main__":
    # Exécution: tous les tests sauf slow
    pytest.main([__file__, "-v", "-m", "not slow"])
