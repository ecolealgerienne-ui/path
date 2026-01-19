"""
Module centralisé pour preprocessing des images H&E.

CE MODULE EST LA SOURCE UNIQUE DE VÉRITÉ.
TOUTES les opérations de normalisation DOIVENT passer par ici.

Historique des bugs évités par ce module:
- 2025-12-20: Bug ToPILImage float64 → Features corrompues
- 2025-12-21: Bug LayerNorm mismatch (blocks[23] vs forward_features)
- 2025-12-21: Bug instance mismatch (connectedComponents)

Règles strictes:
1. Ne JAMAIS modifier HOPTIMUS_MEAN/STD sans ré-entraîner tous les modèles
2. Ne JAMAIS utiliser autre chose que forward_features() pour H-optimus-0
3. TOUJOURS convertir en uint8 avant ToPILImage
4. TOUJOURS valider CLS std après extraction features

Usage:
    >>> from src.preprocessing import preprocess_image, validate_features
    >>> tensor = preprocess_image(image, device="cuda")
    >>> features = backbone.forward_features(tensor)
    >>> validation = validate_features(features)
    >>> assert validation["valid"], validation["message"]
"""

from torchvision import transforms
import torch
import numpy as np
from typing import Dict, Tuple

# Import stain separation functions (V13/V14)
from .stain_separation import (
    ruifrok_extract_h_channel,
    ruifrok_extract_e_channel,
    ruifrok_deconvolution,
    macenko_normalize,
    rgb_to_od,
    od_to_rgb,
    visualize_h_channel,
    compare_ruifrok_vs_macenko,
    RUIFROK_H_VECTOR,
    RUIFROK_E_VECTOR,
    RUIFROK_DAB_VECTOR
)

__all__ = [
    'HOPTIMUS_MEAN',
    'HOPTIMUS_STD',
    'HOPTIMUS_IMAGE_SIZE',
    'create_hoptimus_transform',
    'preprocess_image',
    'validate_features',
    # Stain separation (V13/V14)
    'ruifrok_extract_h_channel',
    'ruifrok_extract_e_channel',
    'ruifrok_deconvolution',
    'macenko_normalize',
    'rgb_to_od',
    'od_to_rgb',
    'visualize_h_channel',
    'compare_ruifrok_vs_macenko',
    'RUIFROK_H_VECTOR',
    'RUIFROK_E_VECTOR',
    'RUIFROK_DAB_VECTOR',
]

# ============================================================================
# CONSTANTES GLOBALES (Source Unique de Vérité)
# ============================================================================

HOPTIMUS_MEAN: Tuple[float, float, float] = (0.707223, 0.578729, 0.703617)
"""
Moyenne RGB pour normalisation H-optimus-0.

ATTENTION: Ces valeurs sont FIXES et définies par Bioptimus.
Ne JAMAIS modifier sans ré-entraîner tous les modèles.
"""

HOPTIMUS_STD: Tuple[float, float, float] = (0.211883, 0.230117, 0.177517)
"""
Écart-type RGB pour normalisation H-optimus-0.

ATTENTION: Ces valeurs sont FIXES et définies par Bioptimus.
Ne JAMAIS modifier sans ré-entraîner tous les modèles.
"""

HOPTIMUS_IMAGE_SIZE: int = 224
"""Taille d'image attendue par H-optimus-0 (224x224 pixels)."""

# ============================================================================
# TRANSFORM CANONIQUE
# ============================================================================

def create_hoptimus_transform() -> transforms.Compose:
    """
    Crée la transformation CANONIQUE pour H-optimus-0.

    Cette fonction définit le pipeline EXACT utilisé pour:
    - L'extraction de features (entraînement)
    - L'inférence (production)
    - Les tests (validation)

    Pipeline:
        1. ToPILImage() - Convertit numpy array en PIL Image
        2. Resize((224, 224)) - Redimensionne à la taille attendue
        3. ToTensor() - Convertit en tensor [0, 1] et transpose (H,W,C) → (C,H,W)
        4. Normalize(mean, std) - Normalise selon statistiques H-optimus-0

    RÈGLES STRICTES:
        - L'image d'entrée DOIT être uint8 [0-255] avant ToPILImage
          (sinon bug: ToPILImage multiplie les floats par 255)
        - Cette fonction DOIT être utilisée PARTOUT (train + inference)
        - Ne JAMAIS modifier sans ré-entraîner tous les modèles

    Returns:
        Transform torchvision composé

    Example:
        >>> transform = create_hoptimus_transform()
        >>> image_uint8 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        >>> tensor = transform(image_uint8)
        >>> tensor.shape
        torch.Size([3, 224, 224])

    See Also:
        - preprocess_image(): Wrapper qui inclut validation et conversion uint8
        - validate_features(): Valide les features extraites
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((HOPTIMUS_IMAGE_SIZE, HOPTIMUS_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])


# ============================================================================
# PREPROCESSING UNIFIÉ
# ============================================================================

def preprocess_image(
    image: np.ndarray,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Prétraite une image H&E pour inférence H-optimus-0.

    Cette fonction est la SEULE manière correcte de préparer une image
    pour H-optimus-0. Elle garantit:
    - Conversion uint8 correcte (évite bug ToPILImage)
    - Transform canonique identique train/inference
    - Validation de l'image d'entrée
    - Batch dimension et device placement

    ÉTAPES CRITIQUES:
        1. Validation image (RGB, shape correcte)
        2. Conversion uint8 (évite bug ToPILImage sur float64)
        3. Transform torchvision canonique:
           - ToPILImage() [APRÈS conversion uint8!]
           - Resize(224, 224)
           - ToTensor()
           - Normalize(HOPTIMUS_MEAN, HOPTIMUS_STD)
        4. Ajout batch dimension
        5. Transfert vers device

    Args:
        image: Image RGB (H, W, 3)
            - Formats acceptés: uint8 [0-255], float [0-1], float [0-255]
            - Sera automatiquement converti en uint8
        device: Device PyTorch ("cuda", "cpu", "mps")

    Returns:
        Tensor (1, 3, 224, 224) normalisé, prêt pour H-optimus-0

    Raises:
        ValueError: Si image n'est pas RGB ou dimensions invalides
        TypeError: Si image n'est pas numpy array

    Example:
        >>> import cv2
        >>> image = cv2.imread("breast.png")
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> tensor = preprocess_image(image, device="cuda")
        >>> features = backbone.forward_features(tensor)
        >>> validation = validate_features(features)
        >>> assert validation["valid"]

    Notes:
        - La conversion uint8 est CRITIQUE pour éviter le bug ToPILImage
        - CLS token std doit être entre 0.70-0.90 après forward_features()
        - Cette fonction est testée dans tests/unit/test_preprocessing.py

    Bugs évités:
        - 2025-12-20: ToPILImage multiplie float64 par 255 → overflow
        - 2025-12-21: Incohérence train/inference → prédictions fausses

    See Also:
        - create_hoptimus_transform(): Transform sous-jacent
        - validate_features(): Valide les features extraites
    """
    # Validation type
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(image)}")

    # Validation shape
    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image (H, W, C), got {image.ndim}D with shape {image.shape}"
        )

    if image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image (H, W, 3), got {image.shape[2]} channels"
        )

    # ÉTAPE CRITIQUE: Conversion uint8 AVANT ToPILImage
    # Bug 2025-12-20: ToPILImage multiplie les floats par 255!
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            # Image normalisée [0, 1] → [0, 255]
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            # Image déjà [0, 255] mais en float
            image = image.clip(0, 255).astype(np.uint8)

    # Transform canonique
    transform = create_hoptimus_transform()
    tensor = transform(image)

    # Batch dimension + device
    tensor = tensor.unsqueeze(0)

    # Device placement
    if device not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda', or 'mps'")

    tensor = tensor.to(device)

    return tensor


# ============================================================================
# VALIDATION FEATURES
# ============================================================================

def validate_features(features: torch.Tensor) -> Dict[str, any]:
    """
    Valide que les features H-optimus-0 sont correctes.

    Cette fonction détecte les erreurs de preprocessing en vérifiant
    que les features extraites ont les propriétés statistiques attendues.

    CRITÈRES DE VALIDATION:
        - Shape = (B, 261, 1536) pour H-optimus-0
          (1 CLS token + 256 patch tokens + 4 register tokens)
        - CLS token std ∈ [0.70, 0.90]
          (indique que LayerNorm final a été appliqué)

    CLS Token Std - Signification:
        - std < 0.40: ERREUR - LayerNorm manquant (bug blocks[23])
        - std ∈ [0.70, 0.90]: OK - forward_features() utilisé
        - std > 1.0: SUSPECT - Vérifier preprocessing

    Args:
        features: Features de H-optimus-0
            - Shape attendue: (B, 261, 1536)
            - Source: backbone.forward_features(tensor)

    Returns:
        dict avec:
            - valid (bool): True si features valides
            - cls_std (float): Écart-type du CLS token
            - shape (tuple): Shape des features
            - message (str): Message de validation

    Example:
        >>> features = backbone.forward_features(tensor)
        >>> validation = validate_features(features)
        >>> if not validation["valid"]:
        ...     raise RuntimeError(validation["message"])

    Raises:
        Aucune - Retourne toujours un dict avec validation["valid"] = bool

    Bugs détectés:
        - 2025-12-21: CLS std ~0.28 → blocks[23] sans LayerNorm
        - Valeur attendue: CLS std ~0.77 avec forward_features()

    See Also:
        - preprocess_image(): Preprocessing avant extraction
    """
    # Validation shape
    expected_shape = (features.shape[0], 261, 1536)  # (B, tokens, embed_dim)

    if features.ndim != 3:
        return {
            "valid": False,
            "cls_std": None,
            "shape": tuple(features.shape),
            "message": (
                f"❌ Shape invalide: attendu 3D (B, 261, 1536), "
                f"obtenu {features.ndim}D {tuple(features.shape)}"
            )
        }

    if features.shape[1] != 261 or features.shape[2] != 1536:
        return {
            "valid": False,
            "cls_std": None,
            "shape": tuple(features.shape),
            "message": (
                f"❌ Shape invalide: attendu (B, 261, 1536), "
                f"obtenu {tuple(features.shape)}\n"
                f"Vérifier que vous utilisez H-optimus-0"
            )
        }

    # Extraction CLS token (premier token)
    cls_token = features[:, 0, :]  # (B, 1536)

    # Calcul std
    cls_std = cls_token.std().item()

    # Validation range
    valid = 0.70 <= cls_std <= 0.90

    if valid:
        message = f"✅ Features valides (CLS std={cls_std:.3f})"
    elif cls_std < 0.40:
        message = (
            f"❌ Features CORROMPUES (CLS std={cls_std:.3f}, attendu 0.70-0.90)\n"
            f"CAUSE PROBABLE: LayerNorm manquant\n"
            f"SOLUTION: Utiliser forward_features() au lieu de blocks[23]\n"
            f"Voir CLAUDE.md Section '⚠️ GUIDE CRITIQUE: BUG #2 LayerNorm Mismatch'"
        )
    else:
        message = (
            f"⚠️ Features SUSPECTES (CLS std={cls_std:.3f}, attendu 0.70-0.90)\n"
            f"Vérifier le preprocessing (conversion uint8, normalisation)"
        )

    return {
        "valid": valid,
        "cls_std": cls_std,
        "shape": tuple(features.shape),
        "message": message
    }


# ============================================================================
# UTILITIES
# ============================================================================

def get_preprocessing_info() -> Dict[str, any]:
    """
    Retourne les informations de preprocessing actuelles.

    Utile pour debugging et logging.

    Returns:
        dict avec constantes et versions
    """
    return {
        "version": "2025-12-22-FINAL",
        "mean": HOPTIMUS_MEAN,
        "std": HOPTIMUS_STD,
        "image_size": HOPTIMUS_IMAGE_SIZE,
        "method": "forward_features_with_layernorm",
        "bugs_fixed": [
            "ToPILImage float64 overflow (2025-12-20)",
            "LayerNorm mismatch (2025-12-21)",
        ]
    }


# ============================================================================
# VERSION
# ============================================================================

__version__ = "1.0.0"
"""Version du module preprocessing."""
