"""
CellViT-Optimus UI Core — Opérations d'export partagées.

Ce module contient la logique d'export partagée:
- PDF
- CSV noyaux
- CSV résumé
- JSON
"""

import tempfile
import os
import logging
from typing import Optional
from pathlib import Path

from .engine_ops import state
from src.ui.export import (
    create_audit_metadata,
    export_nuclei_csv,
    export_summary_csv,
    create_report_pdf,
)

logger = logging.getLogger(__name__)


def export_pdf_core() -> Optional[str]:
    """
    Génère le rapport PDF.

    Returns:
        Chemin du fichier PDF ou None si erreur.
    """
    if state.current_result is None:
        logger.warning("No result to export")
        return None

    if state.engine is None:
        logger.warning("No engine loaded")
        return None

    try:
        # Créer métadonnées audit
        metadata = create_audit_metadata(state.current_result)

        # Générer PDF
        temp_dir = tempfile.mkdtemp()
        pdf_path = Path(temp_dir) / f"cellvit_report_{metadata.analysis_id}.pdf"

        create_report_pdf(
            result=state.current_result,
            output_path=pdf_path,
            metadata=metadata,
            organ=state.engine.organ,
            family=state.engine.family,
        )

        logger.info(f"PDF exported: {pdf_path}")
        return str(pdf_path)

    except Exception as e:
        logger.error(f"PDF export error: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_nuclei_csv_core() -> Optional[str]:
    """
    Génère le CSV détaillé des noyaux.

    Returns:
        Chemin du fichier CSV ou None si erreur.
    """
    if state.current_result is None:
        logger.warning("No result to export")
        return None

    try:
        metadata = create_audit_metadata(state.current_result)

        temp_dir = tempfile.mkdtemp()
        csv_path = Path(temp_dir) / f"nuclei_{metadata.analysis_id}.csv"

        export_nuclei_csv(
            result=state.current_result,
            output_path=csv_path,
            metadata=metadata,
        )

        logger.info(f"Nuclei CSV exported: {csv_path}")
        return str(csv_path)

    except Exception as e:
        logger.error(f"Nuclei CSV export error: {e}")
        return None


def export_summary_csv_core() -> Optional[str]:
    """
    Génère le CSV résumé.

    Returns:
        Chemin du fichier CSV ou None si erreur.
    """
    if state.current_result is None:
        logger.warning("No result to export")
        return None

    if state.engine is None:
        logger.warning("No engine loaded")
        return None

    try:
        metadata = create_audit_metadata(state.current_result)

        temp_dir = tempfile.mkdtemp()
        csv_path = Path(temp_dir) / f"summary_{metadata.analysis_id}.csv"

        export_summary_csv(
            result=state.current_result,
            output_path=csv_path,
            metadata=metadata,
            organ=state.engine.organ,
            family=state.engine.family,
            watershed_params=state.engine.watershed_params,
        )

        logger.info(f"Summary CSV exported: {csv_path}")
        return str(csv_path)

    except Exception as e:
        logger.error(f"Summary CSV export error: {e}")
        return None


def export_json_core() -> str:
    """
    Génère le JSON des résultats.

    Returns:
        Chaîne JSON.
    """
    if state.current_result is None:
        return '{"error": "Aucune analyse disponible"}'

    return state.current_result.to_json(indent=2)
