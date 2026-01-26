"""
Module WSI (Whole Slide Image) pour traitement industriel.

Ce module fournit les outils pour traiter différents types d'inputs:
- PATCH: Images 256×256 (PanNuke, CoNSeP) → Center crop 224×224
- TILE: Images 224×224 pré-extraites → Direct
- IMAGE: Images moyennes (<10000px) → Tiling simple
- WSI: Lames entières (.svs, .ndpi) → CLAM + HistoQC + Tiling

Architecture:
    INPUT → Input Router → Preprocessing adapté → Tiles 224×224 → V13 Inference

Standards implémentés:
    - CLAM (Mahmood Lab) pour tissue segmentation
    - HistoQC pour artifact detection
    - HistoROI pour content filtering

Usage:
    >>> from src.wsi import InputRouter, InputType
    >>> router = InputRouter()
    >>> input_type = router.detect_type("slide.svs")
    >>> tiles = router.process("slide.svs")
    >>> for tile in tiles:
    ...     # tile is 224×224 ready for V13
    ...     pass

See Also:
    - docs/specs/WSI_INDUSTRIAL_PIPELINE_SPEC.md
"""

from .input_router import (
    InputType,
    InputMetadata,
    ProcessedTile,
    InputRouter,
    detect_input_type,
    transform_pannuke_to_224,
)

__all__ = [
    'InputType',
    'InputMetadata',
    'ProcessedTile',
    'InputRouter',
    'detect_input_type',
    'transform_pannuke_to_224',
]

__version__ = "1.0.0"
