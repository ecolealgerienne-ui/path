#!/usr/bin/env python3
"""
Models for CellViT-Optimus.

Architecture "Optimus-Gate":
- HoVerNetDecoder: Local flow (cell segmentation)
- OrganHead: Global flow (organ classification + OOD)

Interface standardisée (Recommandé):
- HoVerNetWrapper: Normalise les sorties HoVer-Net
- OrganHeadWrapper: Normalise les sorties OrganHead
- BackboneWrapper: Normalise les features H-optimus-0
"""

from .hovernet_decoder import HoVerNetDecoder, HoVerNetLoss
from .organ_head import OrganHead, OrganHeadLoss, OrganPrediction, PANNUKE_ORGANS
from .model_interface import (
    HoVerNetWrapper,
    OrganHeadWrapper,
    BackboneWrapper,
    HoVerNetOutput,
    OrganHeadOutput,
    create_hovernet_wrapper,
    create_organ_head_wrapper,
    create_backbone_wrapper,
)

__all__ = [
    # HoVer-Net (Local Flow)
    'HoVerNetDecoder',
    'HoVerNetLoss',
    # Organ Head (Global Flow)
    'OrganHead',
    'OrganHeadLoss',
    'OrganPrediction',
    'PANNUKE_ORGANS',
    # Interface standardisée (RECOMMANDÉ pour nouveaux scripts)
    'HoVerNetWrapper',
    'OrganHeadWrapper',
    'BackboneWrapper',
    'HoVerNetOutput',
    'OrganHeadOutput',
    'create_hovernet_wrapper',
    'create_organ_head_wrapper',
    'create_backbone_wrapper',
]
