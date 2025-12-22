"""
Utilitaires pour la gestion des images et conversions de taille.

Ce module gère les conversions entre les différentes résolutions utilisées
dans le projet (HoVer-Net 224×224, PanNuke 256×256).
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from src.constants import (
    HOVERNET_OUTPUT_SIZE,
    PANNUKE_IMAGE_SIZE,
    get_image_size_mismatch_info,
)


def resize_to_match_ground_truth(
    prediction: np.ndarray,
    target_size: int = PANNUKE_IMAGE_SIZE,
    interpolation: str = "nearest"
) -> np.ndarray:
    """
    Resize une prédiction HoVer-Net (224×224) vers la taille du ground truth (256×256).

    Args:
        prediction: Array à resizer
            - Shape 2D: (H, W) → (target_size, target_size)
            - Shape 3D: (C, H, W) → (C, target_size, target_size)
        target_size: Taille cible (défaut: 256 pour PanNuke)
        interpolation: Méthode d'interpolation
            - "nearest": INTER_NEAREST (pour masques binaires, labels)
            - "linear": INTER_LINEAR (pour probabilités, gradients)
            - "cubic": INTER_CUBIC (pour images RGB)

    Returns:
        Array resized

    Raises:
        ValueError: Si shape invalide ou interpolation inconnue

    Example:
        >>> np_pred = np.random.rand(224, 224)  # Prédiction HoVer-Net
        >>> np_resized = resize_to_match_ground_truth(np_pred)
        >>> assert np_resized.shape == (256, 256)
    """
    # Mapping interpolation
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }

    if interpolation not in interp_map:
        raise ValueError(
            f"Interpolation '{interpolation}' invalide. "
            f"Choix: {list(interp_map.keys())}"
        )

    interp = interp_map[interpolation]

    # Gérer 2D vs 3D
    if prediction.ndim == 2:
        # (H, W) → (target_size, target_size)
        h, w = prediction.shape
        if h == target_size and w == target_size:
            return prediction  # Déjà à la bonne taille

        resized = cv2.resize(
            prediction,
            (target_size, target_size),
            interpolation=interp
        )
        return resized

    elif prediction.ndim == 3:
        # (C, H, W) → (C, target_size, target_size)
        c, h, w = prediction.shape
        if h == target_size and w == target_size:
            return prediction  # Déjà à la bonne taille

        # Resize chaque canal séparément
        resized = np.zeros((c, target_size, target_size), dtype=prediction.dtype)
        for i in range(c):
            resized[i] = cv2.resize(
                prediction[i],
                (target_size, target_size),
                interpolation=interp
            )
        return resized

    else:
        raise ValueError(
            f"Shape invalide: {prediction.shape}. "
            f"Attendu: (H, W) ou (C, H, W)."
        )


def resize_ground_truth_to_prediction(
    ground_truth: np.ndarray,
    target_size: int = HOVERNET_OUTPUT_SIZE,
    interpolation: str = "nearest"
) -> np.ndarray:
    """
    Resize un ground truth PanNuke (256×256) vers la taille des prédictions HoVer-Net (224×224).

    Utilisé rarement (préférer resize_to_match_ground_truth pour évaluation).

    Args:
        ground_truth: Ground truth à resizer
        target_size: Taille cible (défaut: 224 pour HoVer-Net)
        interpolation: Méthode d'interpolation

    Returns:
        Ground truth resized
    """
    return resize_to_match_ground_truth(
        ground_truth,
        target_size=target_size,
        interpolation=interpolation
    )


def prepare_predictions_for_evaluation(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    nt_pred: np.ndarray,
    target_size: int = PANNUKE_IMAGE_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prépare les prédictions HoVer-Net pour évaluation contre ground truth PanNuke.

    Cette fonction est LA RÉFÉRENCE pour convertir les sorties HoVer-Net avant
    calcul des métriques. Elle gère automatiquement le resize et valide les shapes.

    Args:
        np_pred: Nuclear Presence (H, W) - float [0, 1] après sigmoid
        hv_pred: HV maps (2, H, W) - float [-1, 1]
        nt_pred: Nuclear Type (n_classes, H, W) - float [0, 1] après softmax
        target_size: Taille cible pour le resize (défaut: 256)

    Returns:
        (np_resized, hv_resized, nt_resized) - Tous à (target_size, target_size)

    Raises:
        ValueError: Si shapes invalides

    Example:
        >>> # Après inférence HoVer-Net
        >>> output = hovernet_wrapper(features)
        >>> result = output.to_numpy(apply_activations=True)
        >>>
        >>> # Préparer pour évaluation
        >>> np_eval, hv_eval, nt_eval = prepare_predictions_for_evaluation(
        ...     result["np"], result["hv"], result["nt"]
        ... )
        >>> # Maintenant compatibles avec GT PanNuke 256×256
        >>> metrics = compute_metrics(np_eval, hv_eval, nt_eval, gt_np, gt_hv, gt_nt)
    """
    # Validation des shapes d'entrée
    if np_pred.ndim != 2:
        raise ValueError(f"NP shape invalide: {np_pred.shape}. Attendu: (H, W).")

    if hv_pred.ndim != 3 or hv_pred.shape[0] != 2:
        raise ValueError(f"HV shape invalide: {hv_pred.shape}. Attendu: (2, H, W).")

    if nt_pred.ndim != 3:
        raise ValueError(f"NT shape invalide: {nt_pred.shape}. Attendu: (n_classes, H, W).")

    # Resize avec interpolation adaptée
    np_resized = resize_to_match_ground_truth(
        np_pred,
        target_size=target_size,
        interpolation="linear"  # Probabilités → linear
    )

    hv_resized = resize_to_match_ground_truth(
        hv_pred,
        target_size=target_size,
        interpolation="linear"  # Gradients → linear
    )

    nt_resized = resize_to_match_ground_truth(
        nt_pred,
        target_size=target_size,
        interpolation="linear"  # Probabilités → linear
    )

    return np_resized, hv_resized, nt_resized


def check_size_compatibility(
    pred_size: Tuple[int, int],
    gt_size: Tuple[int, int],
    auto_fix: bool = False
) -> dict:
    """
    Vérifie la compatibilité des tailles prédiction/ground truth.

    Args:
        pred_size: (H, W) de la prédiction
        gt_size: (H, W) du ground truth
        auto_fix: Si True, suggère la fonction de correction

    Returns:
        {
            "compatible": bool,
            "pred_size": (H, W),
            "gt_size": (H, W),
            "mismatch": bool,
            "fix_function": str ou None
        }

    Example:
        >>> info = check_size_compatibility((224, 224), (256, 256), auto_fix=True)
        >>> if info["mismatch"]:
        ...     print(f"Utiliser: {info['fix_function']}")
        Utiliser: prepare_predictions_for_evaluation()
    """
    compatible = pred_size == gt_size
    mismatch = not compatible

    result = {
        "compatible": compatible,
        "pred_size": pred_size,
        "gt_size": gt_size,
        "mismatch": mismatch,
        "fix_function": None,
    }

    if mismatch and auto_fix:
        if pred_size == (HOVERNET_OUTPUT_SIZE, HOVERNET_OUTPUT_SIZE) and \
           gt_size == (PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE):
            result["fix_function"] = "prepare_predictions_for_evaluation()"
        else:
            result["fix_function"] = "resize_to_match_ground_truth()"

    return result
