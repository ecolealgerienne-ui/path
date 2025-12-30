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
    # Phase 3
    create_hotspot_overlay,
    create_mitosis_overlay,
    create_chromatin_overlay,
    create_spatial_debug_panel,
    create_phase3_combined_overlay,
)
from .spatial_analysis import (
    SpatialAnalysisResult,
    PleomorphismScore,
    ChromatinFeatures,
)
# Phase 4
from .export import (
    AuditMetadata,
    create_audit_metadata,
    export_nuclei_csv,
    export_summary_csv,
    create_report_pdf,
    BatchResult,
    process_batch,
    export_batch_summary,
    export_batch_csv,
)

__all__ = [
    'CellVitEngine',
    'AnalysisResult',
    'create_segmentation_overlay',
    'create_contour_overlay',
    'create_uncertainty_map',
    'create_type_distribution_chart',
    'CELL_COLORS',
    # Phase 3
    'SpatialAnalysisResult',
    'PleomorphismScore',
    'ChromatinFeatures',
    'create_hotspot_overlay',
    'create_mitosis_overlay',
    'create_chromatin_overlay',
    'create_spatial_debug_panel',
    'create_phase3_combined_overlay',
    # Phase 4
    'AuditMetadata',
    'create_audit_metadata',
    'export_nuclei_csv',
    'export_summary_csv',
    'create_report_pdf',
    'BatchResult',
    'process_batch',
    'export_batch_summary',
    'export_batch_csv',
]
