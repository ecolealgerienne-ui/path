"""
Module de gestion des données.

Ce module centralise toutes les fonctions de preprocessing et de chargement
des données pour garantir la cohérence entre entraînement et évaluation.
"""

from .preprocessing import (
    # Dataclasses
    TargetFormat,

    # Validation
    validate_targets,

    # Resize
    resize_targets,

    # Loading
    load_targets,

    # Batch preparation
    prepare_batch_for_training,
)

__all__ = [
    "TargetFormat",
    "validate_targets",
    "resize_targets",
    "load_targets",
    "prepare_batch_for_training",
]
