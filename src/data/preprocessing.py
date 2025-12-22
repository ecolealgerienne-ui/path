"""
Module centralise pour le preprocessing des donnees.

Ce module est la SOURCE UNIQUE DE VERITE pour:
- Chargement des targets
- Validation des dtypes et ranges
- Resize des targets (train et eval)
- Conversions de format

PRINCIPE: Une seule implementation reutilisee partout pour garantir la coherence.
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from pathlib import Path

from src.constants import HOPTIMUS_INPUT_SIZE, PANNUKE_IMAGE_SIZE


@dataclass
class TargetFormat:
    """
    Format attendu pour les targets HoVer-Net.

    Cette classe documente explicitement les formats attendus
    et sert de reference pour validation.
    """
    # NP (Nuclear Presence)
    np_dtype: type = np.float32
    np_min: float = 0.0
    np_max: float = 1.0

    # HV (Horizontal-Vertical)
    hv_dtype: type = np.float32
    hv_min: float = -1.0
    hv_max: float = 1.0

    # NT (Nuclear Type)
    nt_dtype: type = np.int64
    nt_min: int = 0
    nt_max: int = 4  # 5 classes [0-4]

    # Sizes
    original_size: int = PANNUKE_IMAGE_SIZE  # 256
    model_size: int = HOPTIMUS_INPUT_SIZE    # 224


def validate_targets(
    np_target: np.ndarray,
    hv_target: np.ndarray,
    nt_target: np.ndarray,
    strict: bool = True
) -> Dict[str, any]:
    """
    Valide que les targets respectent le format attendu.

    Args:
        np_target: (H, W) - Nuclear Presence
        hv_target: (2, H, W) - Horizontal-Vertical maps
        nt_target: (H, W) - Nuclear Type
        strict: Si True, raise ValueError si invalide. Sinon, retourne dict avec warnings.

    Returns:
        {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str]
        }

    Raises:
        ValueError: Si strict=True et targets invalides
    """
    fmt = TargetFormat()
    errors = []
    warnings = []

    # Validation NP
    if np_target.dtype != fmt.np_dtype:
        errors.append(f"NP dtype invalide: {np_target.dtype}, attendu: {fmt.np_dtype}")

    if np_target.min() < fmt.np_min or np_target.max() > fmt.np_max:
        errors.append(f"NP range invalide: [{np_target.min():.3f}, {np_target.max():.3f}], attendu: [{fmt.np_min}, {fmt.np_max}]")

    # Validation HV (CRITIQUE pour Bug #3)
    if hv_target.dtype != fmt.hv_dtype:
        errors.append(f"HV dtype invalide: {hv_target.dtype}, attendu: {fmt.hv_dtype}")

    if hv_target.dtype == np.int8:
        errors.append(
            "HV dtype est int8 [-127, 127] au lieu de float32 [-1, 1] ! "
            "Cela cause MSE ~4681 au lieu de ~0.01. "
            "Re-generer targets avec prepare_family_data_FIXED.py"
        )

    if hv_target.min() < fmt.hv_min - 0.1 or hv_target.max() > fmt.hv_max + 0.1:
        errors.append(f"HV range invalide: [{hv_target.min():.3f}, {hv_target.max():.3f}], attendu: [{fmt.hv_min}, {fmt.hv_max}]")

    if hv_target.shape[0] != 2:
        errors.append(f"HV doit avoir 2 canaux, trouve: {hv_target.shape[0]}")

    # Validation NT
    if nt_target.dtype not in [np.int32, np.int64]:
        errors.append(f"NT dtype invalide: {nt_target.dtype}, attendu: int64")

    if nt_target.min() < fmt.nt_min or nt_target.max() > fmt.nt_max:
        errors.append(f"NT range invalide: [{nt_target.min()}, {nt_target.max()}], attendu: [{fmt.nt_min}, {fmt.nt_max}]")

    # Shape consistency
    if np_target.shape != nt_target.shape:
        errors.append(f"NP shape {np_target.shape} != NT shape {nt_target.shape}")

    if hv_target.shape[1:] != np_target.shape:
        errors.append(f"HV spatial shape {hv_target.shape[1:]} != NP shape {np_target.shape}")

    # Verdict
    valid = len(errors) == 0

    result = {
        "valid": valid,
        "errors": errors,
        "warnings": warnings
    }

    if strict and not valid:
        error_msg = "\n".join(["VALIDATION TARGETS ECHOUEE:"] + [f"  • {e}" for e in errors])
        raise ValueError(error_msg)

    return result


def resize_targets(
    np_target: np.ndarray,
    hv_target: np.ndarray,
    nt_target: np.ndarray,
    target_size: int = HOPTIMUS_INPUT_SIZE,
    mode: str = "training"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resize les targets de 256x256 vers target_size (generalement 224x224).

    CETTE FONCTION EST LA REFERENCE pour le resize.
    Elle est utilisee IDENTIQUEMENT pendant l'entrainement et l'evaluation.

    Args:
        np_target: (H, W) - Nuclear Presence, dtype=float32, range=[0, 1]
        hv_target: (2, H, W) - HV maps, dtype=float32, range=[-1, 1]
        nt_target: (H, W) - Nuclear Type, dtype=int64, range=[0, 4]
        target_size: Taille de sortie (defaut: 224)
        mode: "training" ou "evaluation" (meme implementation)

    Returns:
        (np_resized, hv_resized, nt_resized) - Tous a (target_size, target_size)

    Raises:
        ValueError: Si targets invalides
    """
    # Validation stricte
    validate_targets(np_target, hv_target, nt_target, strict=True)

    # Conversion en tenseurs PyTorch (float pour interpolation)
    np_t = torch.from_numpy(np_target).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    hv_t = torch.from_numpy(hv_target).float().unsqueeze(0)                # (1, 2, H, W)
    nt_t = torch.from_numpy(nt_target).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Resize avec interpolations adaptees
    # IMPORTANT: Ces interpolations DOIVENT etre identiques a train_hovernet_family.py
    np_resized_t = F.interpolate(
        np_t,
        size=(target_size, target_size),
        mode='nearest'  # Binaire → nearest
    ).squeeze()

    hv_resized_t = F.interpolate(
        hv_t,
        size=(target_size, target_size),
        mode='bilinear',  # Gradients → bilinear
        align_corners=False
    ).squeeze(0)

    nt_resized_t = F.interpolate(
        nt_t,
        size=(target_size, target_size),
        mode='nearest'  # Labels → nearest
    ).squeeze().long()

    # Conversion en numpy
    np_resized = np_resized_t.numpy()
    hv_resized = hv_resized_t.numpy()
    nt_resized = nt_resized_t.numpy()

    return np_resized, hv_resized, nt_resized


def load_targets(
    targets_path: Path,
    validate: bool = True,
    auto_convert_hv: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge les targets depuis un fichier .npz.

    Args:
        targets_path: Chemin vers fichier *_targets.npz
        validate: Si True, valide les targets apres chargement
        auto_convert_hv: Si True, convertit automatiquement HV int8 → float32
                        (utile pour compatibilite anciennes donnees)

    Returns:
        (np_targets, hv_targets, nt_targets)

    Raises:
        FileNotFoundError: Si fichier introuvable
        ValueError: Si validation echoue
    """
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets introuvables: {targets_path}")

    # Chargement
    data = np.load(targets_path)

    np_targets = data['np_targets']
    hv_targets = data['hv_targets']
    nt_targets = data['nt_targets']

    # Auto-conversion HV int8 → float32 si demande
    if auto_convert_hv and hv_targets.dtype == np.int8:
        print(f"⚠️  Auto-conversion HV: int8 [-127, 127] → float32 [-1, 1]")
        hv_targets = hv_targets.astype(np.float32) / 127.0

    # Validation
    if validate:
        # Valider un seul echantillon (pour performance)
        validation = validate_targets(
            np_targets[0],
            hv_targets[0],
            nt_targets[0],
            strict=False
        )

        if not validation["valid"]:
            print(f"⚠️  ATTENTION: Targets invalides dans {targets_path}")
            for error in validation["errors"]:
                print(f"     • {error}")

            if not auto_convert_hv:
                print("")
                print("   Conseil: Utiliser auto_convert_hv=True pour conversion automatique")
                print("   ou re-generer avec prepare_family_data_FIXED.py")

            raise ValueError(f"Targets invalides: {validation['errors']}")

    return np_targets, hv_targets, nt_targets


def prepare_batch_for_training(
    np_targets: np.ndarray,
    hv_targets: np.ndarray,
    nt_targets: np.ndarray,
    indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare un batch pour l'entrainement.

    Cette fonction centralise la logique de preparation utilisee dans DataLoader.

    Args:
        np_targets: (N, 256, 256)
        hv_targets: (N, 2, 256, 256)
        nt_targets: (N, 256, 256)
        indices: Indices du batch

    Returns:
        Batch resized a (B, ..., 224, 224)
    """
    batch_size = len(indices)

    np_batch = []
    hv_batch = []
    nt_batch = []

    for idx in indices:
        np_resized, hv_resized, nt_resized = resize_targets(
            np_targets[idx],
            hv_targets[idx],
            nt_targets[idx],
            target_size=HOPTIMUS_INPUT_SIZE,
            mode="training"
        )

        np_batch.append(np_resized)
        hv_batch.append(hv_resized)
        nt_batch.append(nt_resized)

    return (
        np.array(np_batch),
        np.array(hv_batch),
        np.array(nt_batch)
    )
