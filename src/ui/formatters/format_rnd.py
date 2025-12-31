"""
CellViT-Optimus UI Formatters — Format R&D (technique).

Ce module formate les résultats pour l'interface R&D:
- Langage technique
- Métriques détaillées
- Debug visible

Usage:
    from src.ui.formatters import format_metrics_rnd, format_alerts_rnd
"""

from typing import Optional, Dict, Any
from src.ui.inference_engine import AnalysisResult
from src.ui.visualizations import TYPE_NAMES


def format_metrics_rnd(
    result: AnalysisResult,
    organ: Optional[str] = None,
    family: Optional[str] = None,
    is_dedicated: bool = False,
) -> str:
    """
    Formate les métriques en texte (style R&D technique).

    Args:
        result: Résultat d'analyse
        organ: Nom de l'organe
        family: Famille du modèle
        is_dedicated: True si modèle dédié
    """
    # Afficher le modèle utilisé
    if is_dedicated:
        model_info = f"### Modèle: **{organ}** ★ (dédié)"
        family_info = f"*Famille: {family}*"
    else:
        model_info = f"### Modèle: {family or 'N/A'}"
        family_info = f"*Organe sélectionné: {organ or 'N/A'}*"

    # Message contextuel pour organe non déterminé
    if is_dedicated:
        organ_display = f"### Organe: **{organ}** (modèle dédié)"
    elif result.organ_name == "Unknown" or result.organ_confidence < 0.1:
        if result.n_nuclei < 20:
            organ_display = "### Organe détecté (IA): *Non déterminable* (surface insuffisante)"
        else:
            organ_display = "### Organe détecté (IA): *Non déterminable* (contexte architectural limité)"
    else:
        organ_display = f"### Organe détecté (IA): {result.organ_name} ({result.organ_confidence:.1%})"

    lines = [
        organ_display,
        model_info,
        family_info,
        "",
        f"**Noyaux détectés:** {result.n_nuclei}",
        f"**Temps d'inférence:** {result.inference_time_ms:.0f} ms",
        "",
    ]

    if result.morphometry:
        m = result.morphometry

        # Ratio I/E: logique métier complète
        inflammatory_count = m.type_counts.get("Inflammatory", 0)
        epithelial_count = m.type_counts.get("Epithelial", 0)
        ie_denominator = inflammatory_count + epithelial_count

        if m.neoplastic_ratio >= 0.95:
            ie_display = "*non applicable* (foyer tumoral pur)"
        elif ie_denominator < 3:
            ie_display = "*non interprétable* (effectif insuffisant)"
        else:
            ie_display = f"**{m.immuno_epithelial_ratio:.2f}**"

        # Activité mitotique: Signal IA (pas un "index mitotique" clinique)
        # Note: L'index mitotique clinique requiert comptage sur 10 HPF par pathologiste
        n_mitosis = result.n_mitosis_candidates if result.spatial_analysis else 0

        if n_mitosis == 0:
            mitotic_display = "Aucune figure suspecte"
        elif n_mitosis > result.n_nuclei * 0.5:
            mitotic_display = f"⚠️ **{n_mitosis} figures suspectes** (activité élevée)"
        elif n_mitosis > 3:
            mitotic_display = f"**{n_mitosis} figures suspectes** (activité modérée)"
        else:
            mitotic_display = f"{n_mitosis} figure(s) suspecte(s)"

        lines.extend([
            "---",
            "### Morphométrie",
            f"- Densité: **{m.nuclei_per_mm2:.0f}** noyaux/mm²",
            f"- Aire moyenne: **{m.mean_area_um2:.1f}** ± {m.std_area_um2:.1f} µm²",
            f"- Circularité: **{m.mean_circularity:.2f}** ± {m.std_circularity:.2f}",
            f"- Hypercellularité: **{m.nuclear_density_percent:.1f}%**",
            "",
            "### Activité & Ratios",
            f"- Activité mitotique: {mitotic_display}",
            f"- Ratio néoplasique: **{m.neoplastic_ratio:.1%}**",
            f"- Ratio I/E: {ie_display}",
            f"- TILs status: **{m.til_status}**",
            "",
            "### Distribution",
        ])

        for t in TYPE_NAMES:
            count = m.type_counts.get(t, 0)
            pct = m.type_percentages.get(t, 0)
            lines.append(f"- {t}: {count} ({pct:.1f}%)")

        lines.extend([
            "",
            f"**Confiance:** {m.confidence_level}",
        ])

    # Phase 3: Intelligence Spatiale
    if result.spatial_analysis:
        score_labels = {1: "Faible", 2: "Modéré", 3: "Sévère"}
        score_emoji = {1: "🟢", 2: "🟡", 3: "🔴"}

        if result.n_nuclei < 20:
            phase3_title = "### Phase 3 — Intelligence Spatiale 🛈 *(surface limitée)*"
        else:
            phase3_title = "### Phase 3 — Intelligence Spatiale"

        # Chromatine: message contextuel explicite
        if result.n_heterogeneous_nuclei == 0:
            if result.mean_chromatin_entropy > 4.0:
                chromatin_display = "**homogène** *(entropie élevée mais peu variable entre noyaux)*"
            else:
                chromatin_display = "**homogène** *(texture régulière)*"
        elif result.n_heterogeneous_nuclei < 3:
            chromatin_display = f"**{result.n_heterogeneous_nuclei}** noyau(x) atypique(s)"
        else:
            chromatin_display = f"**{result.n_heterogeneous_nuclei}** noyaux hétérogènes"

        lines.extend([
            "",
            "---",
            phase3_title,
            f"- Pléomorphisme: **{result.pleomorphism_score}/3** {score_emoji.get(result.pleomorphism_score, '')} ({score_labels.get(result.pleomorphism_score, '')})",
            f"- Hotspots: **{result.n_hotspots}** zones haute densité",
            f"- Mitoses candidates: **{result.n_mitosis_candidates}**",
            f"- Chromatine: {chromatin_display}",
            f"- Voisins moyens (Voronoï): **{result.mean_neighbors:.1f}**",
            f"- Entropie chromatine: **{result.mean_chromatin_entropy:.2f}**",
        ])

    return "\n".join(lines)


def format_alerts_rnd(result: AnalysisResult) -> str:
    """
    Formate les alertes en texte (style R&D).

    Inclut:
    - Alertes morphométriques
    - Anomalies Phase 2 (fusions, sur-segmentations)
    - Intelligence spatiale Phase 3
    """
    lines = ["### Points d'attention", ""]

    # Alertes morphométriques
    if result.morphometry and result.morphometry.alerts:
        for alert in result.morphometry.alerts:
            lines.append(f"- {alert}")

    # Phase 2: Alertes anomalies
    if result.n_fusions > 0:
        lines.append(f"- **{result.n_fusions} fusion(s) potentielle(s)** (aire > 2× moyenne)")
    if result.n_over_seg > 0:
        lines.append(f"- **{result.n_over_seg} sur-segmentation(s)** (aire < 0.5× moyenne)")

    # Phase 3: Alertes intelligence spatiale
    if result.spatial_analysis:
        if result.pleomorphism_score >= 3:
            lines.append("- 🔴 **Pléomorphisme sévère** — anisocaryose marquée")
        elif result.pleomorphism_score == 2:
            lines.append("- 🟡 **Pléomorphisme modéré** — variation notable")

        if result.n_mitosis_candidates > 3:
            lines.append(f"- 🔴 **{result.n_mitosis_candidates} mitoses suspectes** — activité proliférative")
        elif result.n_mitosis_candidates > 0:
            # Note: index peut être None (surface insuffisante)
            lines.append(f"- 🟡 **{result.n_mitosis_candidates} mitose(s) candidate(s)** *(patch unique)*")

        if result.n_hotspots > 0:
            lines.append(f"- 🟠 **{result.n_hotspots} hotspot(s)** — zones haute densité")

        if result.n_heterogeneous_nuclei > 5:
            lines.append(f"- 🟣 **{result.n_heterogeneous_nuclei} noyaux chromatine hétérogène**")

    if len(lines) == 2:  # Seulement le titre
        return "Aucune alerte"

    return "\n".join(lines)


def format_nucleus_info_rnd(nucleus_data: Dict[str, Any]) -> str:
    """
    Formate les informations d'un noyau sélectionné (style R&D).
    """
    if not nucleus_data.get("found"):
        if nucleus_data.get("clicked_background"):
            return "*Clic sur le fond (pas de noyau)*"
        return "*Cliquer sur un noyau*"

    lines = [
        f"**Noyau #{nucleus_data['nucleus_id']}**",
        f"- Type: {nucleus_data.get('cell_type', 'Unknown')}",
    ]

    if nucleus_data.get("area_um2"):
        lines.append(f"- Aire: {nucleus_data['area_um2']:.1f} µm²")
    if nucleus_data.get("circularity"):
        lines.append(f"- Circularité: {nucleus_data['circularity']:.2f}")
    if nucleus_data.get("position"):
        x, y = nucleus_data["position"]
        lines.append(f"- Position: ({x}, {y})")

    return "\n".join(lines)


def format_load_status_rnd(load_result: Dict[str, Any]) -> str:
    """Formate le message de chargement du moteur (style R&D)."""
    if load_result["success"]:
        return f"Moteur chargé : {load_result['organ']} ({load_result['model_type']}) sur {load_result['device']}"
    else:
        return f"Erreur : {load_result['error']}"


def format_organ_change_rnd(change_result: Dict[str, Any]) -> str:
    """Formate le message de changement d'organe (style R&D)."""
    if change_result["success"]:
        model_type = "dédié ★" if "dédié" in change_result["model_type"] else change_result["model_type"]
        base_msg = f"Organe: {change_result['organ']} — Modèle {model_type}"
        # Inclure les params watershed si disponibles
        if change_result.get("watershed_params"):
            return f"{base_msg}\nParams: {change_result['watershed_params']}"
        return base_msg
    else:
        return f"Erreur : {change_result['error']}"
