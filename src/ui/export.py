"""
CellViT-Optimus R&D Cockpit — Module d'Export (Phase 4).

Ce module implémente les fonctionnalités d'export:
- Export PDF: Rapport clinique formaté
- Export CSV: Métriques tabulaires
- Traçabilité: Métadonnées d'audit

Author: CellViT-Optimus Project
Date: 2025-12-30
"""

import numpy as np
import cv2
import csv
import io
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TRAÇABILITÉ / AUDIT
# =============================================================================

@dataclass
class AuditMetadata:
    """Métadonnées de traçabilité pour chaque analyse."""
    # Identification
    analysis_id: str = ""              # UUID unique
    timestamp: str = ""                # ISO 8601

    # Contexte
    user_id: str = "anonymous"         # Identifiant utilisateur
    session_id: str = ""               # Session Gradio

    # Configuration
    model_family: str = ""             # Famille HoVer-Net
    model_checkpoint: str = ""         # Nom du checkpoint
    model_version: str = "v3.0"        # Version CellViT-Optimus

    # Paramètres
    watershed_params: Dict = field(default_factory=dict)
    device: str = "cuda"

    # Input
    image_hash: str = ""               # SHA256 de l'image
    image_size: Tuple[int, int] = (0, 0)

    # Timing
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0


def generate_analysis_id() -> str:
    """Génère un identifiant unique pour l'analyse."""
    import uuid
    return str(uuid.uuid4())[:8].upper()


def compute_image_hash(image: np.ndarray) -> str:
    """Calcule le hash SHA256 d'une image."""
    import hashlib
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def create_audit_metadata(
    result,  # AnalysisResult
    user_id: str = "anonymous",
    session_id: str = "",
) -> AuditMetadata:
    """
    Crée les métadonnées d'audit pour une analyse.

    Args:
        result: AnalysisResult de l'analyse
        user_id: Identifiant utilisateur
        session_id: Session Gradio

    Returns:
        AuditMetadata complètes
    """
    return AuditMetadata(
        analysis_id=generate_analysis_id(),
        timestamp=datetime.now().isoformat(),
        user_id=user_id,
        session_id=session_id,
        model_family=result.family,
        model_checkpoint=f"hovernet_{result.family}_v13_smart_crops_hybrid_fpn_best.pth",
        model_version="v3.0",
        watershed_params=result.watershed_params,
        device="cuda",
        image_hash=compute_image_hash(result.image_rgb),
        image_size=(result.image_rgb.shape[0], result.image_rgb.shape[1]),
        inference_time_ms=result.inference_time_ms,
        total_time_ms=result.inference_time_ms,
    )


# =============================================================================
# EXPORT CSV
# =============================================================================

def export_nuclei_csv(result, output_path: Optional[Path] = None) -> str:
    """
    Exporte les données des noyaux en CSV.

    Args:
        result: AnalysisResult
        output_path: Chemin de sortie (optionnel, retourne string si None)

    Returns:
        Contenu CSV ou chemin du fichier
    """
    output = io.StringIO()

    # Colonnes
    fieldnames = [
        "id", "centroid_y", "centroid_x", "area_um2", "perimeter_um",
        "circularity", "cell_type", "type_idx", "confidence",
        "is_uncertain", "is_mitotic",
        # Phase 2
        "is_potential_fusion", "is_potential_over_seg", "anomaly_reason",
        # Phase 3
        "chromatin_entropy", "chromatin_heterogeneous",
        "is_mitosis_candidate", "mitosis_score",
        "n_neighbors", "is_in_hotspot"
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for n in result.nucleus_info:
        writer.writerow({
            "id": n.id,
            "centroid_y": n.centroid[0],
            "centroid_x": n.centroid[1],
            "area_um2": f"{n.area_um2:.2f}",
            "perimeter_um": f"{n.perimeter_um:.2f}",
            "circularity": f"{n.circularity:.3f}",
            "cell_type": n.cell_type,
            "type_idx": n.type_idx,
            "confidence": f"{n.confidence:.3f}",
            "is_uncertain": n.is_uncertain,
            "is_mitotic": n.is_mitotic,
            "is_potential_fusion": n.is_potential_fusion,
            "is_potential_over_seg": n.is_potential_over_seg,
            "anomaly_reason": n.anomaly_reason,
            "chromatin_entropy": f"{n.chromatin_entropy:.3f}",
            "chromatin_heterogeneous": n.chromatin_heterogeneous,
            "is_mitosis_candidate": n.is_mitosis_candidate,
            "mitosis_score": f"{n.mitosis_score:.3f}",
            "n_neighbors": n.n_neighbors,
            "is_in_hotspot": n.is_in_hotspot,
        })

    csv_content = output.getvalue()

    if output_path:
        output_path.write_text(csv_content)
        return str(output_path)

    return csv_content


def export_summary_csv(result, audit: Optional[AuditMetadata] = None) -> str:
    """
    Exporte un résumé des métriques en CSV.

    Args:
        result: AnalysisResult
        audit: Métadonnées d'audit optionnelles

    Returns:
        Contenu CSV
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Métadonnées
    writer.writerow(["# CellViT-Optimus Analysis Summary"])
    writer.writerow(["# Generated", datetime.now().isoformat()])
    if audit:
        writer.writerow(["# Analysis ID", audit.analysis_id])
    writer.writerow([])

    # Section: Identification
    writer.writerow(["## Identification"])
    writer.writerow(["Organ", result.organ_name])
    writer.writerow(["Organ Confidence", f"{result.organ_confidence:.1%}"])
    writer.writerow(["Family", result.family])
    writer.writerow([])

    # Section: Counts
    writer.writerow(["## Counts"])
    writer.writerow(["Total Nuclei", result.n_nuclei])
    writer.writerow(["Fusions", result.n_fusions])
    writer.writerow(["Over-segmentations", result.n_over_seg])
    writer.writerow([])

    # Section: Morphometry
    if result.morphometry:
        m = result.morphometry
        writer.writerow(["## Morphometry"])
        writer.writerow(["Density (nuclei/mm²)", f"{m.nuclei_per_mm2:.0f}"])
        writer.writerow(["Mean Area (µm²)", f"{m.mean_area_um2:.2f}"])
        writer.writerow(["Std Area (µm²)", f"{m.std_area_um2:.2f}"])
        writer.writerow(["Mean Circularity", f"{m.mean_circularity:.3f}"])
        writer.writerow(["Mitotic Index (/10 HPF)", f"{m.mitotic_index_per_10hpf:.1f}"])
        writer.writerow(["Neoplastic Ratio", f"{m.neoplastic_ratio:.1%}"])
        writer.writerow(["TILs Status", m.til_status])
        writer.writerow(["Confidence Level", m.confidence_level])
        writer.writerow([])

    # Section: Phase 3
    if result.spatial_analysis:
        writer.writerow(["## Spatial Analysis (Phase 3)"])
        writer.writerow(["Pleomorphism Score", f"{result.pleomorphism_score}/3"])
        writer.writerow(["Pleomorphism Description", result.pleomorphism_description])
        writer.writerow(["Hotspots", result.n_hotspots])
        writer.writerow(["Mitosis Candidates", result.n_mitosis_candidates])
        writer.writerow(["Heterogeneous Chromatin", result.n_heterogeneous_nuclei])
        writer.writerow(["Mean Neighbors (Voronoi)", f"{result.mean_neighbors:.1f}"])
        writer.writerow(["Mean Chromatin Entropy", f"{result.mean_chromatin_entropy:.2f}"])
        writer.writerow([])

    # Section: Parameters
    writer.writerow(["## Watershed Parameters"])
    for k, v in result.watershed_params.items():
        writer.writerow([k, v])
    writer.writerow([])

    # Section: Timing
    writer.writerow(["## Performance"])
    writer.writerow(["Inference Time (ms)", f"{result.inference_time_ms:.0f}"])

    return output.getvalue()


# =============================================================================
# EXPORT PDF
# =============================================================================

def create_report_pdf(
    result,
    image_overlay: np.ndarray,
    audit: Optional[AuditMetadata] = None,
    output_path: Optional[Path] = None,
) -> bytes:
    """
    Crée un rapport PDF clinique formaté.

    Args:
        result: AnalysisResult
        image_overlay: Image avec segmentation overlay
        audit: Métadonnées d'audit
        output_path: Chemin de sortie (optionnel)

    Returns:
        Contenu PDF en bytes
    """
    # Créer le PDF
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # Page 1: Rapport principal
        fig = plt.figure(figsize=(8.5, 11), facecolor='white')

        # Header
        fig.text(0.5, 0.96, "RAPPORT D'ANALYSE — CellViT-Optimus",
                 ha='center', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.94, "DOCUMENT D'AIDE À LA DÉCISION — VALIDATION MÉDICALE REQUISE",
                 ha='center', fontsize=8, color='red', style='italic')

        # Ligne de séparation
        fig.add_axes([0.1, 0.92, 0.8, 0.001]).set_facecolor('black')
        fig.gca().set_xticks([])
        fig.gca().set_yticks([])

        # Zone identification
        id_text = f"""
Organe détecté: {result.organ_name} ({result.organ_confidence:.1%})
Famille: {result.family}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        if audit:
            id_text += f"ID Analyse: {audit.analysis_id}\n"

        fig.text(0.12, 0.88, id_text, fontsize=9, verticalalignment='top',
                 fontfamily='monospace')

        # Image avec overlay
        ax_img = fig.add_axes([0.1, 0.52, 0.35, 0.35])
        ax_img.imshow(image_overlay)
        ax_img.set_title("Segmentation", fontsize=10)
        ax_img.axis('off')

        # Image originale
        ax_orig = fig.add_axes([0.55, 0.52, 0.35, 0.35])
        ax_orig.imshow(result.image_rgb)
        ax_orig.set_title("Image originale", fontsize=10)
        ax_orig.axis('off')

        # Métriques globales
        metrics_text = f"""
MÉTRIQUES GLOBALES
─────────────────────
Noyaux détectés: {result.n_nuclei}
"""
        if result.morphometry:
            m = result.morphometry
            metrics_text += f"""Densité: {m.nuclei_per_mm2:.0f} noyaux/mm²
Aire moyenne: {m.mean_area_um2:.1f} ± {m.std_area_um2:.1f} µm²
Circularité: {m.mean_circularity:.2f}
Index mitotique: {m.mitotic_index_per_10hpf:.1f}/10 HPF
Ratio néoplasique: {m.neoplastic_ratio:.1%}
TILs status: {m.til_status}
"""

        fig.text(0.12, 0.48, metrics_text, fontsize=9, verticalalignment='top',
                 fontfamily='monospace')

        # Phase 3: Intelligence Spatiale
        if result.spatial_analysis:
            score_labels = {1: "Faible", 2: "Modéré", 3: "Sévère"}
            spatial_text = f"""
INTELLIGENCE SPATIALE (Phase 3)
───────────────────────────────
Pléomorphisme: {result.pleomorphism_score}/3 ({score_labels.get(result.pleomorphism_score, '')})
Hotspots: {result.n_hotspots} zones
Mitoses candidates: {result.n_mitosis_candidates}
Chromatine hétérogène: {result.n_heterogeneous_nuclei} noyaux
Voisins Voronoï (moy.): {result.mean_neighbors:.1f}
"""
            fig.text(0.55, 0.48, spatial_text, fontsize=9, verticalalignment='top',
                     fontfamily='monospace')

        # Alertes
        alerts_text = "\nALERTES\n───────\n"
        has_alerts = False

        if result.morphometry and result.morphometry.alerts:
            for alert in result.morphometry.alerts[:5]:
                alerts_text += f"• {alert}\n"
                has_alerts = True

        if result.n_fusions > 0:
            alerts_text += f"• {result.n_fusions} fusion(s) potentielle(s)\n"
            has_alerts = True
        if result.n_over_seg > 0:
            alerts_text += f"• {result.n_over_seg} sur-segmentation(s)\n"
            has_alerts = True

        if result.spatial_analysis:
            if result.pleomorphism_score >= 3:
                alerts_text += "• Pléomorphisme sévère\n"
                has_alerts = True
            if result.n_mitosis_candidates > 3:
                alerts_text += f"• {result.n_mitosis_candidates} mitoses suspectes\n"
                has_alerts = True

        if not has_alerts:
            alerts_text += "Aucune alerte\n"

        fig.text(0.12, 0.25, alerts_text, fontsize=9, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Paramètres
        params_text = f"""
PARAMÈTRES WATERSHED
────────────────────
np_threshold: {result.watershed_params.get('np_threshold', 'N/A')}
min_size: {result.watershed_params.get('min_size', 'N/A')}
beta: {result.watershed_params.get('beta', 'N/A')}
min_distance: {result.watershed_params.get('min_distance', 'N/A')}
"""
        fig.text(0.55, 0.25, params_text, fontsize=8, verticalalignment='top',
                 fontfamily='monospace')

        # Footer
        fig.text(0.5, 0.03,
                 f"CellViT-Optimus v3.0 — Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ha='center', fontsize=7, color='gray')
        fig.text(0.5, 0.01,
                 "Ce document est un outil d'aide à la décision et ne remplace pas le diagnostic médical.",
                 ha='center', fontsize=6, color='red', style='italic')

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # Page 2: Distribution des types (si morphometry disponible)
        if result.morphometry and result.morphometry.type_counts:
            fig2 = plt.figure(figsize=(8.5, 11), facecolor='white')

            fig2.text(0.5, 0.96, "DISTRIBUTION DES TYPES CELLULAIRES",
                      ha='center', fontsize=14, fontweight='bold')

            # Pie chart
            ax_pie = fig2.add_axes([0.15, 0.5, 0.7, 0.4])
            types = list(result.morphometry.type_counts.keys())
            counts = list(result.morphometry.type_counts.values())
            colors = ['#FF3232', '#32FF32', '#3232FF', '#FFFF32', '#32FFFF']

            if sum(counts) > 0:
                ax_pie.pie(counts, labels=types, autopct='%1.1f%%', colors=colors[:len(types)])
                ax_pie.set_title("Répartition des types cellulaires")

            # Table des comptages
            table_data = [[t, c, f"{c/sum(counts)*100:.1f}%"] for t, c in zip(types, counts)]
            ax_table = fig2.add_axes([0.2, 0.2, 0.6, 0.25])
            ax_table.axis('off')
            table = ax_table.table(
                cellText=table_data,
                colLabels=['Type', 'Nombre', 'Pourcentage'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            pdf.savefig(fig2, dpi=150)
            plt.close(fig2)

    pdf_content = pdf_buffer.getvalue()

    if output_path:
        output_path.write_bytes(pdf_content)
        logger.info(f"PDF report saved to {output_path}")

    return pdf_content


# =============================================================================
# BATCH PROCESSING
# =============================================================================

@dataclass
class BatchResult:
    """Résultat d'un traitement batch."""
    n_images: int = 0
    n_success: int = 0
    n_failed: int = 0
    total_nuclei: int = 0
    mean_nuclei_per_image: float = 0.0
    results: List[Any] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)
    processing_time_ms: float = 0.0


def process_batch(
    images: List[Tuple[str, np.ndarray]],  # List of (name, image)
    engine,  # CellVitEngine
    watershed_params: Optional[Dict] = None,
) -> BatchResult:
    """
    Traite un batch d'images.

    Args:
        images: Liste de tuples (nom, image)
        engine: CellVitEngine initialisé
        watershed_params: Paramètres watershed optionnels

    Returns:
        BatchResult avec tous les résultats
    """
    import time
    start = time.time()

    batch = BatchResult(n_images=len(images))

    for name, image in images:
        try:
            # Valider taille
            if image.shape[0] != 224 or image.shape[1] != 224:
                batch.errors.append((name, f"Invalid size: {image.shape[:2]}"))
                batch.n_failed += 1
                continue

            # Analyser
            result = engine.analyze(
                image,
                watershed_params=watershed_params,
                compute_morphometry=True,
                compute_uncertainty=True,
            )

            batch.results.append((name, result))
            batch.total_nuclei += result.n_nuclei
            batch.n_success += 1

        except Exception as e:
            batch.errors.append((name, str(e)))
            batch.n_failed += 1

    batch.processing_time_ms = (time.time() - start) * 1000

    if batch.n_success > 0:
        batch.mean_nuclei_per_image = batch.total_nuclei / batch.n_success

    return batch


def export_batch_summary(batch: BatchResult) -> str:
    """
    Exporte un résumé du batch en texte.

    Args:
        batch: BatchResult

    Returns:
        Résumé formaté
    """
    lines = [
        "=" * 50,
        "BATCH PROCESSING SUMMARY",
        "=" * 50,
        "",
        f"Images traitées: {batch.n_success}/{batch.n_images}",
        f"Échecs: {batch.n_failed}",
        f"Total noyaux: {batch.total_nuclei}",
        f"Moyenne noyaux/image: {batch.mean_nuclei_per_image:.1f}",
        f"Temps total: {batch.processing_time_ms:.0f} ms",
        "",
    ]

    if batch.errors:
        lines.append("ERREURS:")
        for name, error in batch.errors:
            lines.append(f"  - {name}: {error}")
        lines.append("")

    # Résumé par image
    lines.append("DÉTAILS:")
    for name, result in batch.results:
        lines.append(f"  {name}:")
        lines.append(f"    - Noyaux: {result.n_nuclei}")
        lines.append(f"    - Organe: {result.organ_name}")
        if result.spatial_analysis:
            lines.append(f"    - Pléomorphisme: {result.pleomorphism_score}/3")

    return "\n".join(lines)


def export_batch_csv(batch: BatchResult) -> str:
    """
    Exporte les résultats batch en CSV.

    Args:
        batch: BatchResult

    Returns:
        Contenu CSV
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "image_name", "n_nuclei", "organ", "organ_confidence",
        "density", "mean_area", "circularity", "mitotic_index",
        "pleomorphism_score", "n_hotspots", "n_mitosis_candidates",
        "inference_time_ms"
    ])

    for name, result in batch.results:
        row = [
            name,
            result.n_nuclei,
            result.organ_name,
            f"{result.organ_confidence:.3f}",
        ]

        if result.morphometry:
            m = result.morphometry
            row.extend([
                f"{m.nuclei_per_mm2:.0f}",
                f"{m.mean_area_um2:.2f}",
                f"{m.mean_circularity:.3f}",
                f"{m.mitotic_index_per_10hpf:.1f}",
            ])
        else:
            row.extend(["", "", "", ""])

        if result.spatial_analysis:
            row.extend([
                result.pleomorphism_score,
                result.n_hotspots,
                result.n_mitosis_candidates,
            ])
        else:
            row.extend(["", "", ""])

        row.append(f"{result.inference_time_ms:.0f}")

        writer.writerow(row)

    return output.getvalue()
