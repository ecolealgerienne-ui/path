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
        mitotic_display = f"{m.mitotic_index_per_10hpf:.1f}" if m.mitotic_index_per_10hpf is not None else "N/A (patch unique)"
        writer.writerow(["Mitotic Index (/10 HPF)", mitotic_display])
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
    selected_organ: Optional[str] = None,
) -> bytes:
    """
    Crée un rapport PDF clinique formaté.

    Améliorations v3:
    - Organe SÉLECTIONNÉ affiché en primaire (pas OrganHead)
    - OrganHead utilisé comme VALIDATION (cohérence IA)
    - Activité mitotique au lieu d'index mitotique
    - Note technique au lieu de params watershed

    Args:
        result: AnalysisResult
        image_overlay: Image avec segmentation overlay
        audit: Métadonnées d'audit
        output_path: Chemin de sortie (optionnel)
        selected_organ: Organe sélectionné par l'utilisateur (prioritaire sur OrganHead)

    Returns:
        Contenu PDF en bytes
    """
    # Couleurs pour les types cellulaires
    TYPE_COLORS = {
        'Neoplastic': '#E63946',      # Rouge
        'Inflammatory': '#457B9D',    # Bleu
        'Epithelial': '#2A9D8F',      # Vert-bleu
        'Connective': '#E9C46A',      # Jaune
        'Dead': '#6C757D',            # Gris
    }

    # Créer le PDF
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # =====================================================================
        # PAGE 1: RAPPORT PRINCIPAL
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11), facecolor='white')

        # Header avec fond coloré
        header_ax = fig.add_axes([0, 0.93, 1, 0.07])
        header_ax.set_facecolor('#2C3E50')
        header_ax.set_xticks([])
        header_ax.set_yticks([])
        for spine in header_ax.spines.values():
            spine.set_visible(False)

        fig.text(0.5, 0.965, "RAPPORT D'ANALYSE — CellViT-Optimus",
                 ha='center', fontsize=14, fontweight='bold', color='white')
        fig.text(0.5, 0.94, "DOCUMENT D'AIDE À LA DÉCISION — VALIDATION MÉDICALE REQUISE",
                 ha='center', fontsize=8, color='#FFD700', style='italic')

        # Zone identification (carte)
        id_box = fig.add_axes([0.05, 0.82, 0.42, 0.09])
        id_box.set_facecolor('#F8F9FA')
        id_box.set_xticks([])
        id_box.set_yticks([])
        for spine in id_box.spines.values():
            spine.set_color('#DEE2E6')

        # Organe = sélectionné par l'utilisateur (pas OrganHead)
        display_organ = selected_organ or result.organ_name

        # Validation OrganHead (cohérence)
        if result.organ_confidence >= 0.5:
            if result.organ_name == display_organ:
                validation = f"✓ Confirmé IA ({result.organ_confidence:.0%})"
            else:
                validation = f"⚠️ IA suggère: {result.organ_name}"
        else:
            validation = "ℹ️ Validation IA limitée"

        id_lines = [
            f"Organe: {display_organ} — {validation}",
            f"Modèle: famille {result.family}",
            f"Surface: 0.01 mm² (champ limité)",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        if audit:
            id_lines.append(f"ID: {audit.analysis_id}")

        for i, line in enumerate(id_lines):
            fig.text(0.07, 0.895 - i * 0.022, line, fontsize=8,
                     fontfamily='sans-serif')

        # Noyaux détectés (badge)
        nuclei_box = fig.add_axes([0.52, 0.82, 0.2, 0.09])
        nuclei_box.set_facecolor('#3498DB')
        nuclei_box.set_xticks([])
        nuclei_box.set_yticks([])
        for spine in nuclei_box.spines.values():
            spine.set_visible(False)

        fig.text(0.62, 0.875, f"{result.n_nuclei}", ha='center',
                 fontsize=24, fontweight='bold', color='white')
        fig.text(0.62, 0.835, "noyaux", ha='center',
                 fontsize=10, color='white')

        # Temps d'inférence (badge)
        time_box = fig.add_axes([0.75, 0.82, 0.2, 0.09])
        time_box.set_facecolor('#27AE60')
        time_box.set_xticks([])
        time_box.set_yticks([])
        for spine in time_box.spines.values():
            spine.set_visible(False)

        fig.text(0.85, 0.875, f"{result.inference_time_ms:.0f}", ha='center',
                 fontsize=24, fontweight='bold', color='white')
        fig.text(0.85, 0.835, "ms", ha='center',
                 fontsize=10, color='white')

        # Images côte à côte
        ax_orig = fig.add_axes([0.05, 0.50, 0.42, 0.30])
        ax_orig.imshow(result.image_rgb)
        ax_orig.set_title("Image H&E", fontsize=10, fontweight='bold', pad=5)
        ax_orig.axis('off')

        ax_seg = fig.add_axes([0.53, 0.50, 0.42, 0.30])
        ax_seg.imshow(image_overlay)
        ax_seg.set_title("Segmentation", fontsize=10, fontweight='bold', pad=5)
        ax_seg.axis('off')

        # Section Morphométrie (tableau)
        if result.morphometry:
            m = result.morphometry
            # Figures mitotiques suspectes (PAS un "index mitotique" clinique)
            n_cand = result.n_mitosis_candidates if result.spatial_analysis else m.mitotic_candidates
            if n_cand > 0:
                mitotic_str = f"{n_cand} figure(s) suspecte(s)"
            else:
                mitotic_str = "Aucune détectée"

            morph_data = [
                ["Densité", f"{m.nuclei_per_mm2:.0f} /mm²"],
                ["Aire moy.", f"{m.mean_area_um2:.1f} ± {m.std_area_um2:.1f} µm²"],
                ["Circularité", f"{m.mean_circularity:.2f}"],
                ["Activité mitotique", mitotic_str],
                ["Ratio néoplasique", f"{m.neoplastic_ratio:.0%}"],
                ["TILs", m.til_status],
            ]

            ax_morph = fig.add_axes([0.05, 0.28, 0.42, 0.18])
            ax_morph.axis('off')
            ax_morph.set_title("MORPHOMÉTRIE", fontsize=10, fontweight='bold',
                               loc='left', pad=2)

            table_morph = ax_morph.table(
                cellText=morph_data,
                colWidths=[0.4, 0.6],
                loc='upper left',
                cellLoc='left',
                edges='horizontal'
            )
            table_morph.auto_set_font_size(False)
            table_morph.set_fontsize(8)
            for key, cell in table_morph.get_celld().items():
                cell.set_height(0.15)
                if key[1] == 0:
                    cell.set_text_props(fontweight='bold')
                    cell.set_facecolor('#F8F9FA')

        # Section Phase 3 (tableau)
        if result.spatial_analysis:
            score_colors = {1: '#27AE60', 2: '#F39C12', 3: '#E74C3C'}
            score_labels = {1: "Faible", 2: "Modéré", 3: "Sévère"}

            phase3_data = [
                ["Pléomorphisme", f"{result.pleomorphism_score}/3 ({score_labels.get(result.pleomorphism_score, '')})"],
                ["Hotspots", f"{result.n_hotspots} zones"],
                ["Mitoses cand.", f"{result.n_mitosis_candidates}"],
                ["Chrom. hétérog.", f"{result.n_heterogeneous_nuclei} noyaux"],
                ["Voisins Voronoï", f"{result.mean_neighbors:.1f}"],
            ]

            ax_phase3 = fig.add_axes([0.53, 0.28, 0.42, 0.18])
            ax_phase3.axis('off')
            ax_phase3.set_title("INTELLIGENCE SPATIALE (Phase 3)", fontsize=10,
                                fontweight='bold', loc='left', pad=2)

            table_phase3 = ax_phase3.table(
                cellText=phase3_data,
                colWidths=[0.45, 0.55],
                loc='upper left',
                cellLoc='left',
                edges='horizontal'
            )
            table_phase3.auto_set_font_size(False)
            table_phase3.set_fontsize(8)
            for key, cell in table_phase3.get_celld().items():
                cell.set_height(0.15)
                if key[1] == 0:
                    cell.set_text_props(fontweight='bold')
                    cell.set_facecolor('#F8F9FA')

        # Section Alertes avec couleurs
        alerts = []

        if result.morphometry and result.morphometry.alerts:
            for alert in result.morphometry.alerts[:3]:
                alerts.append(('orange', alert))

        if result.n_fusions > 0:
            alerts.append(('red', f"{result.n_fusions} fusion(s) potentielle(s)"))
        if result.n_over_seg > 0:
            alerts.append(('orange', f"{result.n_over_seg} sur-segmentation(s)"))

        if result.spatial_analysis:
            if result.pleomorphism_score >= 3:
                alerts.append(('red', "Pléomorphisme SÉVÈRE"))
            if result.n_mitosis_candidates > 10:
                alerts.append(('red', f"⚠️ {result.n_mitosis_candidates} mitoses (TRÈS ÉLEVÉ)"))
            elif result.n_mitosis_candidates > 3:
                alerts.append(('red', f"{result.n_mitosis_candidates} mitoses suspectes"))

        ax_alerts = fig.add_axes([0.05, 0.10, 0.55, 0.15])
        ax_alerts.set_facecolor('#FFF9E6' if alerts else '#E8F5E9')
        ax_alerts.set_xticks([])
        ax_alerts.set_yticks([])
        for spine in ax_alerts.spines.values():
            spine.set_color('#FFE082' if alerts else '#A5D6A7')

        fig.text(0.07, 0.235, "POINTS D'ATTENTION", fontsize=9, fontweight='bold')

        if alerts:
            for i, (color, text) in enumerate(alerts[:5]):
                marker_color = '#E74C3C' if color == 'red' else '#F39C12'
                fig.text(0.07, 0.21 - i * 0.025, "●", fontsize=10, color=marker_color)
                fig.text(0.09, 0.21 - i * 0.025, text, fontsize=8)
        else:
            fig.text(0.07, 0.19, "✓ Aucune alerte particulière", fontsize=9, color='#27AE60')

        # Note technique (remplace les params watershed - plus approprié pour rapport clinique)
        ax_note = fig.add_axes([0.65, 0.10, 0.30, 0.15])
        ax_note.set_facecolor('#E8F5E9')
        ax_note.set_xticks([])
        ax_note.set_yticks([])
        for spine in ax_note.spines.values():
            spine.set_color('#A5D6A7')

        fig.text(0.67, 0.22, "NOTE TECHNIQUE", fontsize=8, fontweight='bold', color='#2E7D32')
        fig.text(0.67, 0.19, "Surface analysée:", fontsize=7, color='#558B2F')
        fig.text(0.67, 0.165, "  0.01 mm² (patch 224×224)", fontsize=7, color='#33691E')
        fig.text(0.67, 0.14, "Segmentation: HV-Watershed", fontsize=7, color='#558B2F')
        fig.text(0.67, 0.115, "Paramètres: optimisés/organe", fontsize=7, color='#558B2F')

        # Footer
        fig.text(0.5, 0.03,
                 f"CellViT-Optimus v3.0 — Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ha='center', fontsize=7, color='gray')
        fig.text(0.5, 0.01,
                 "Ce document est un outil d'aide à la décision et ne remplace pas le diagnostic médical.",
                 ha='center', fontsize=6, color='#E74C3C', style='italic')

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # =====================================================================
        # PAGE 2: DISTRIBUTION DES TYPES CELLULAIRES
        # =====================================================================
        if result.morphometry and result.morphometry.type_counts:
            fig2 = plt.figure(figsize=(8.5, 11), facecolor='white')

            # Header
            header_ax2 = fig2.add_axes([0, 0.93, 1, 0.07])
            header_ax2.set_facecolor('#2C3E50')
            header_ax2.set_xticks([])
            header_ax2.set_yticks([])
            for spine in header_ax2.spines.values():
                spine.set_visible(False)

            fig2.text(0.5, 0.96, "DISTRIBUTION DES TYPES CELLULAIRES",
                      ha='center', fontsize=14, fontweight='bold', color='white')

            types = list(result.morphometry.type_counts.keys())
            counts = list(result.morphometry.type_counts.values())
            total = sum(counts)

            if total > 0:
                # Graphique circulaire SANS labels (légende séparée)
                ax_pie = fig2.add_axes([0.15, 0.50, 0.45, 0.38])

                colors = [TYPE_COLORS.get(t, '#95A5A6') for t in types]
                wedges, texts, autotexts = ax_pie.pie(
                    counts,
                    autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
                    colors=colors,
                    startangle=90,
                    pctdistance=0.75,
                    wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
                )

                for autotext in autotexts:
                    autotext.set_fontsize(9)
                    autotext.set_fontweight('bold')
                    autotext.set_color('white')

                ax_pie.set_title("Répartition", fontsize=11, fontweight='bold', pad=10)

                # Légende déportée à droite (évite chevauchement)
                ax_legend = fig2.add_axes([0.62, 0.55, 0.33, 0.30])
                ax_legend.axis('off')

                fig2.text(0.64, 0.84, "LÉGENDE", fontsize=10, fontweight='bold')

                for i, (t, c) in enumerate(zip(types, counts)):
                    pct = c / total * 100
                    color = TYPE_COLORS.get(t, '#95A5A6')
                    y_pos = 0.80 - i * 0.08

                    # Carré de couleur
                    color_box = fig2.add_axes([0.64, y_pos - 0.015, 0.03, 0.025])
                    color_box.set_facecolor(color)
                    color_box.set_xticks([])
                    color_box.set_yticks([])
                    for spine in color_box.spines.values():
                        spine.set_visible(False)

                    # Label
                    fig2.text(0.69, y_pos, f"{t}", fontsize=9, fontweight='bold')
                    fig2.text(0.69, y_pos - 0.025, f"{c} ({pct:.1f}%)", fontsize=8, color='gray')

            # Tableau récapitulatif
            table_data = [[t, c, f"{c/total*100:.1f}%"] for t, c in zip(types, counts)]
            ax_table = fig2.add_axes([0.15, 0.20, 0.70, 0.25])
            ax_table.axis('off')

            table = ax_table.table(
                cellText=table_data,
                colLabels=['Type cellulaire', 'Nombre', 'Pourcentage'],
                loc='upper center',
                cellLoc='center',
                colColours=['#2C3E50'] * 3,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.8)

            # Style des cellules
            for key, cell in table.get_celld().items():
                if key[0] == 0:  # Header
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    if key[1] == 0:  # Colonne type
                        cell.set_text_props(fontweight='bold')
                    cell.set_facecolor('#F8F9FA' if key[0] % 2 == 0 else 'white')

            # Footer page 2
            fig2.text(0.5, 0.03, "Page 2/2", ha='center', fontsize=8, color='gray')

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
            mitotic_val = f"{m.mitotic_index_per_10hpf:.1f}" if m.mitotic_index_per_10hpf is not None else "N/A"
            row.extend([
                f"{m.nuclei_per_mm2:.0f}",
                f"{m.mean_area_um2:.2f}",
                f"{m.mean_circularity:.3f}",
                mitotic_val,
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
