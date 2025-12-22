"""
Module de gestion des donnees.

Ce module centralise toutes les fonctions de preprocessing et de chargement
des donnees pour garantir la coherence entre entrainement et evaluation.
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
