"""
CellViT-Optimus R&D Cockpit — Interface Gradio.

Module d'interface utilisateur pour l'exploration et la validation
du moteur IA de segmentation cellulaire.

Note: Ceci est un outil R&D, pas une IHM clinique.
"""

from .inference_engine import CellVitEngine, AnalysisResult
from .visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    create_uncertainty_map,
    create_type_distribution_chart,
    CELL_COLORS,
)

__all__ = [
    'CellVitEngine',
    'AnalysisResult',
    'create_segmentation_overlay',
    'create_contour_overlay',
    'create_uncertainty_map',
    'create_type_distribution_chart',
    'CELL_COLORS',
]
