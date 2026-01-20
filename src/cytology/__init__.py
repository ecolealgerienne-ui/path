"""
V14 Cytology Module — Complete Pipeline

Ce module contient:
- Morphométrie: Calcul des 20 features à partir des masques CellPose
- Models: MLP Classification Head avec fusion multimodale

SINGLE SOURCE OF TRUTH:
Les features sont TOUJOURS calculées sur les masques générés,
JAMAIS lues depuis CSV/Excel externe.

Author: V14 Cytology Branch
Date: 2026-01-19
"""

# Morphometry functions
from .morphometry import (
    compute_single_cell_features,
    compute_batch_features,
    get_feature_names,
    validate_features,
    interpret_nc_ratio,
    interpret_chromatin_density,
)

# Classification models
from .models.cytology_classifier import (
    CytologyClassifier,
    FocalLoss,
    compute_class_weights,
    count_parameters,
)

__all__ = [
    # Morphometry
    'compute_single_cell_features',
    'compute_batch_features',
    'get_feature_names',
    'validate_features',
    'interpret_nc_ratio',
    'interpret_chromatin_density',
    # Models
    'CytologyClassifier',
    'FocalLoss',
    'compute_class_weights',
    'count_parameters',
]
