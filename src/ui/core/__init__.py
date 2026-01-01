"""
CellViT-Optimus UI Core — Logique partagée entre les interfaces.

Ce module contient toute la logique métier partagée entre:
- app.py (R&D Cockpit)
- app_pathologist.py (Interface Pathologiste)

Principe: La logique est unique, seul l'affichage diffère.
"""

from .engine_ops import (
    UIState,
    state,
    preload_backbone_core,  # Précharge backbone au démarrage
    load_engine_core,
    run_analysis_core,  # Fonction partagée (sans visualisations)
    analyze_image_core,  # Wrapper R&D (avec visualisations)
    change_organ_core,
    on_image_click_core,
)

from .export_ops import (
    export_pdf_core,
    export_nuclei_csv_core,
    export_summary_csv_core,
    export_json_core,
)

__all__ = [
    # State
    "UIState",
    "state",
    # Engine operations
    "preload_backbone_core",  # Précharge backbone au démarrage
    "load_engine_core",
    "run_analysis_core",   # Partagé (sans visualisations)
    "analyze_image_core",  # R&D (avec visualisations)
    "change_organ_core",
    "on_image_click_core",
    # Export operations
    "export_pdf_core",
    "export_nuclei_csv_core",
    "export_summary_csv_core",
    "export_json_core",
]
