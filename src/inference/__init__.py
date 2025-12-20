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

from .optimus_gate_inference import OptimusGateInference

from .optimus_gate_multifamily import (
    OptimusGateMultiFamily,
    MultiFamilyResult,
    ORGAN_TO_FAMILY,
    FAMILIES,
    RELIABLE_HV_FAMILIES,
)

from .optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily

__all__ = [
    # Legacy CellViT
    "CellViTInference",
    "CellViT256Model",
    "CellViT256",
    "load_cellvit256_from_checkpoint",
    "CELL_TYPES",
    "CELL_COLORS",
    "CELL_EMOJIS",
    # Optimus-Gate (architecture simple)
    "OptimusGate",
    "OptimusGateResult",
    "CellDetection",
    "OptimusGateInference",
    # Optimus-Gate Multi-Famille (5 décodeurs)
    "OptimusGateMultiFamily",
    "MultiFamilyResult",
    "OptimusGateInferenceMultiFamily",
    "ORGAN_TO_FAMILY",
    "FAMILIES",
    "RELIABLE_HV_FAMILIES",
]
