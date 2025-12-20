# CellViT-Optimus Inference Module

from .cellvit_inference import (
    CellViTInference,
    CellViT256Model,
    CELL_TYPES,
    CELL_COLORS,
    CELL_EMOJIS,
)

from .cellvit256_model import (
    CellViT256,
    load_cellvit256_from_checkpoint,
)

from .optimus_gate import (
    OptimusGate,
    OptimusGateResult,
    CellDetection,
)

__all__ = [
    # Legacy CellViT
    "CellViTInference",
    "CellViT256Model",
    "CellViT256",
    "load_cellvit256_from_checkpoint",
    "CELL_TYPES",
    "CELL_COLORS",
    "CELL_EMOJIS",
    # Optimus-Gate (new architecture)
    "OptimusGate",
    "OptimusGateResult",
    "CellDetection",
]
