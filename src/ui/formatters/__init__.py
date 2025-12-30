"""
CellViT-Optimus UI Formatters — Formatage d'affichage.

Ce module sépare le formatage selon l'audience:
- format_rnd.py: Interface R&D (technique, métriques détaillées)
- format_clinical.py: Interface Pathologiste (langage clinique)

Principe: Même données, affichage différent.
"""

from .format_rnd import (
    format_metrics_rnd,
    format_alerts_rnd,
    format_nucleus_info_rnd,
    format_load_status_rnd,
    format_organ_change_rnd,
)

from .format_clinical import (
    format_metrics_clinical,
    format_alerts_clinical,
    format_nucleus_info_clinical,
    format_load_status_clinical,
    format_organ_change_clinical,
    format_identification_clinical,
    format_confidence_badge,
    compute_confidence_level,
    interpret_density,
    interpret_pleomorphism,
    interpret_mitotic_index,
)

__all__ = [
    # R&D formatters
    "format_metrics_rnd",
    "format_alerts_rnd",
    "format_nucleus_info_rnd",
    "format_load_status_rnd",
    "format_organ_change_rnd",
    # Clinical formatters
    "format_metrics_clinical",
    "format_alerts_clinical",
    "format_nucleus_info_clinical",
    "format_load_status_clinical",
    "format_organ_change_clinical",
    "format_identification_clinical",
    "format_confidence_badge",
    "compute_confidence_level",
    "interpret_density",
    "interpret_pleomorphism",
    "interpret_mitotic_index",
]
