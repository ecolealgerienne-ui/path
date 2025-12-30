"""
CellViT-Optimus UI Formatters — Format Clinical (pathologiste).

Ce module formate les résultats pour l'interface Pathologiste:
- Langage clinique (pas de jargon technique)
- Métriques interprétées
- Valeurs brutes masquées

Usage:
    from src.ui.formatters import format_metrics_clinical, format_alerts_clinical
"""

from typing import Optional, Dict, Any, Tuple
from src.ui.inference_engine import AnalysisResult


# ==============================================================================
# DICTIONNAIRE DE CORRESPONDANCE CLINIQUE (Lexique IA -> Pathologie)
# Version "Factuelle" — Évite les verbes d'interprétation (suspicion, suggère)
# Utilise: "corrélé à", "associé à", "observé dans" (faits bibliographiques)
# ==============================================================================

CLINICAL_INTERPRETATIONS = {
    # Pléomorphisme — Langage factuel (corrélation, pas suspicion)
    "pleomorphism_3": (
        "🔴 **Pléomorphisme sévère (3/3)** — "
        "Critère morphologique corrélé au grade nucléaire élevé (Nottingham/Elston)"
    ),
    "pleomorphism_2": (
        "🟡 **Pléomorphisme modéré (2/3)** — "
        "Variation notable de taille et forme nucléaire"
    ),

    # Mitoses — Faits observés (pas "processus tumoral agressif")
    "mitosis_very_high": (
        "🔴 **Activité mitotique très élevée ({count})** — "
        "Index prolifératif associé aux tumeurs à croissance rapide dans la littérature"
    ),
    "mitosis_high": (
        "🟡 **Activité mitotique élevée ({count})** — "
        "Figures évocatrices de mitoses identifiées"
    ),
    "mitosis_present": (
        "ℹ️ **Mitoses détectées ({count})** — "
        "Figure(s) évocatrice(s) à confirmer visuellement"
    ),

    # Chromatine — Description technique (pas "instabilité génétique")
    "chromatin_heterogeneous": (
        "🔍 **Chromatine hétérogène ({percent:.0f}% des noyaux)** — "
        "Texture nucléaire irrégulière, critère observé dans les cellules à activité métabolique élevée"
    ),

    # Architecture — Observations quantifiées (pas "fortement suspect")
    "neoplastic_predominance": (
        "🔍 **Prédominance néoplasique ({ratio:.0f}%)** — "
        "Ratio cellules néoplasiques/total supérieur au seuil d'attention (70%)"
    ),
    "hypercellularity": (
        "🔍 **Hypercellularité ({density:.0f}%)** — "
        "Densité nucléaire élevée, critère associé aux proliférations cellulaires denses"
    ),
    "hotspots": (
        "🟠 **Zones hypercellulaires ({count})** — "
        "Cluster(s) de haute densité identifié(s)"
    ),

    # Anisocaryose — Mesure objective (pas "dysplasie")
    "anisocaryose_marked": (
        "🔍 **Anisocaryose marquée (CV={cv:.2f})** — "
        "Coefficient de variation de l'aire nucléaire > 0.5, indicateur d'hétérogénéité morphologique"
    ),

    # TILs — Description spatiale (neutre)
    "til_cold": (
        "❄️ **Infiltrat lymphocytaire périphérique** — "
        "TILs localisés en bordure, pattern associé à l'immuno-exclusion tumorale"
    ),
    "til_excluded": (
        "🚫 **TILs distants** — "
        "Lymphocytes éloignés du compartiment tumoral"
    ),
}


# ==============================================================================
# FONCTIONS D'INTERPRÉTATION CLINIQUE
# ==============================================================================

def compute_confidence_level(result: AnalysisResult) -> Tuple[str, str]:
    """
    Calcule le niveau de confiance global de l'IA.

    Returns:
        (niveau, couleur) - ex: ("Élevée", "green")
    """
    if result.uncertainty_map is None:
        return "Non disponible", "gray"

    # Moyenne d'incertitude
    mean_uncertainty = result.uncertainty_map.mean()

    # Confiance organe
    organ_conf = result.organ_confidence

    # Score combiné
    if mean_uncertainty < 0.3 and organ_conf > 0.9:
        return "Élevée", "green"
    elif mean_uncertainty < 0.5 and organ_conf > 0.7:
        return "Modérée", "orange"
    else:
        return "Faible", "red"


def interpret_density(density: float) -> str:
    """Interprète la densité en langage clinique."""
    if density < 1000:
        return "Faible"
    elif density < 2000:
        return "Normale"
    elif density < 3500:
        return "Élevée"
    else:
        return "Très élevée"


def interpret_pleomorphism(score: int) -> str:
    """Interprète le score de pléomorphisme."""
    interpretations = {
        1: "Faible (compatible grade I)",
        2: "Modéré (compatible grade II)",
        3: "Sévère (compatible grade III)",
    }
    return interpretations.get(score, "Non évalué")


def interpret_mitotic_index(index: Optional[float], n_candidates: int = 0) -> str:
    """
    Interprète l'index mitotique.

    Args:
        index: Index mitotique (None si surface insuffisante)
        n_candidates: Nombre de candidats mitose (Phase 3)
    """
    if index is None:
        # Surface insuffisante pour extrapolation HPF
        if n_candidates == 0:
            return "N/A (patch unique)"
        else:
            return f"{n_candidates} candidat(s) *(patch unique)*"
    elif index < 3:
        return f"{index:.0f}/10 HPF (Faible)"
    elif index < 8:
        return f"{index:.0f}/10 HPF (Modéré)"
    else:
        return f"{index:.0f}/10 HPF (Élevé)"


# ==============================================================================
# FONCTIONS DE FORMATAGE CLINIQUE
# ==============================================================================

def format_identification_clinical(
    result: AnalysisResult,
    organ: Optional[str] = None,
    family: Optional[str] = None,
    is_dedicated: bool = False,
) -> str:
    """
    Formate l'identification de l'organe (style clinique).

    Args:
        result: Résultat d'analyse
        organ: Nom de l'organe sélectionné
        family: Famille du modèle
        is_dedicated: True si modèle dédié
    """
    # Afficher le modèle utilisé
    if is_dedicated:
        model_line = f"**Modèle:** {organ} ★ (dédié)"
    else:
        model_line = f"**Modèle:** {family} (famille)\n*Organe: {organ}*"

    return f"""### {result.organ_name}
**Confiance IA:** {result.organ_confidence:.0%}
{model_line}"""


def format_metrics_clinical(
    result: AnalysisResult,
    organ: Optional[str] = None,
    family: Optional[str] = None,
    is_dedicated: bool = False,
) -> str:
    """
    Formate les métriques en langage clinique (pas de valeurs brutes techniques).

    Args:
        result: Résultat d'analyse
        organ: Nom de l'organe
        family: Famille du modèle
        is_dedicated: True si modèle dédié
    """
    lines = [
        f"**Noyaux détectés:** {result.n_nuclei}",
        "",
    ]

    if result.morphometry:
        m = result.morphometry

        # Densité interprétée
        density_label = interpret_density(m.nuclei_per_mm2)
        lines.append(f"**Densité cellulaire:** {density_label} ({m.nuclei_per_mm2:.0f}/mm²)")

        # Index mitotique interprété (avec n_candidates pour affichage si index=None)
        n_candidates = result.n_mitosis_candidates if result.spatial_analysis else m.mitotic_candidates
        mitotic_label = interpret_mitotic_index(m.mitotic_index_per_10hpf, n_candidates)
        lines.append(f"**Index mitotique:** {mitotic_label}")

        # Ratio néoplasique
        if m.neoplastic_ratio > 0.5:
            lines.append(f"**Ratio néoplasique:** Élevé ({m.neoplastic_ratio:.0%})")
        elif m.neoplastic_ratio > 0.2:
            lines.append(f"**Ratio néoplasique:** Modéré ({m.neoplastic_ratio:.0%})")
        else:
            lines.append(f"**Ratio néoplasique:** Faible ({m.neoplastic_ratio:.0%})")

        # TILs
        lines.append(f"**TILs:** {m.til_status}")

    # Phase 3: Pléomorphisme (interprété)
    if result.spatial_analysis:
        pleo_label = interpret_pleomorphism(result.pleomorphism_score)
        lines.append("")
        lines.append(f"**Pléomorphisme:** {pleo_label}")

    return "\n".join(lines)


def format_alerts_clinical(result: AnalysisResult) -> str:
    """
    Formate les alertes avec enrichissement clinique descriptif.

    Utilise le dictionnaire CLINICAL_INTERPRETATIONS pour transformer
    les métriques brutes en observations cliniques factuelles.

    Principe: "corrélé à", "associé à", "observé dans" (pas "suspicion de")
    """
    alerts = []

    # ==========================================================================
    # Phase 3: Pléomorphisme
    # ==========================================================================
    if result.spatial_analysis:
        if result.pleomorphism_score >= 3:
            alerts.append(CLINICAL_INTERPRETATIONS["pleomorphism_3"])
        elif result.pleomorphism_score == 2:
            alerts.append(CLINICAL_INTERPRETATIONS["pleomorphism_2"])

        # Chromatine hétérogène
        n_heterogeneous = len(result.spatial_analysis.heterogeneous_nuclei_ids)
        if n_heterogeneous > 0 and result.n_nuclei > 0:
            percent = (n_heterogeneous / result.n_nuclei) * 100
            if percent > 10:  # Seuil significatif
                alerts.append(
                    CLINICAL_INTERPRETATIONS["chromatin_heterogeneous"].format(percent=percent)
                )

        # Mitoses
        n_mitosis = result.n_mitosis_candidates
        if n_mitosis > 10:
            alerts.append(
                CLINICAL_INTERPRETATIONS["mitosis_very_high"].format(count=n_mitosis)
            )
        elif n_mitosis > 3:
            alerts.append(
                CLINICAL_INTERPRETATIONS["mitosis_high"].format(count=n_mitosis)
            )
        elif n_mitosis > 0:
            alerts.append(
                CLINICAL_INTERPRETATIONS["mitosis_present"].format(count=n_mitosis)
            )

        # Hotspots
        if result.n_hotspots > 0:
            alerts.append(
                CLINICAL_INTERPRETATIONS["hotspots"].format(count=result.n_hotspots)
            )

    # ==========================================================================
    # Morphométrie et Architecture
    # ==========================================================================
    if result.morphometry:
        m = result.morphometry

        # Prédominance néoplasique
        if m.neoplastic_ratio > 0.7:
            alerts.append(
                CLINICAL_INTERPRETATIONS["neoplastic_predominance"].format(
                    ratio=m.neoplastic_ratio * 100
                )
            )

        # Hypercellularité (densité nucléaire > 40%)
        if hasattr(m, 'nuclear_density_percent') and m.nuclear_density_percent > 40:
            alerts.append(
                CLINICAL_INTERPRETATIONS["hypercellularity"].format(
                    density=m.nuclear_density_percent
                )
            )

        # Anisocaryose marquée (CV > 0.5)
        if m.mean_area_um2 > 0:
            cv_area = m.std_area_um2 / m.mean_area_um2
            if cv_area > 0.5:
                alerts.append(
                    CLINICAL_INTERPRETATIONS["anisocaryose_marked"].format(cv=cv_area)
                )

        # Statut TILs
        if m.til_status == "froid":
            alerts.append(CLINICAL_INTERPRETATIONS["til_cold"])
        elif m.til_status == "exclu":
            alerts.append(CLINICAL_INTERPRETATIONS["til_excluded"])

    if not alerts:
        return "✅ Aucune anomalie majeure détectée par l'IA"

    return "\n\n".join(alerts)


def format_confidence_badge(result: AnalysisResult) -> str:
    """Crée le badge de confiance HTML."""
    level, color = compute_confidence_level(result)

    color_map = {
        "green": "#28a745",
        "orange": "#fd7e14",
        "red": "#dc3545",
        "gray": "#6c757d",
    }

    bg_color = color_map.get(color, "#6c757d")

    return f"""
    <div style="
        display: inline-block;
        background-color: {bg_color};
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    ">
        Confiance IA : {level}
    </div>
    """


def format_nucleus_info_clinical(nucleus_data: Dict[str, Any]) -> str:
    """
    Formate les informations d'un noyau sélectionné (style clinique simplifié).
    """
    if not nucleus_data.get("found"):
        if nucleus_data.get("clicked_background"):
            return "*Cliquer sur un noyau pour voir ses détails*"
        return "*Cliquer sur un noyau pour voir ses détails*"

    lines = [
        f"### Noyau #{nucleus_data['nucleus_id']}",
        "",
        f"**Type:** {nucleus_data.get('cell_type', 'Unknown')}",
    ]

    if nucleus_data.get("area_um2"):
        lines.append(f"**Aire:** {nucleus_data['area_um2']:.1f} µm²")

    if nucleus_data.get("circularity"):
        circ = nucleus_data["circularity"]
        shape = "Régulière" if circ > 0.7 else "Irrégulière"
        lines.append(f"**Forme:** {shape}")

    # Alertes simplifiées
    if nucleus_data.get("is_mitosis_candidate"):
        lines.append("")
        lines.append("**Mitose suspecte**")

    if nucleus_data.get("is_in_hotspot"):
        lines.append("**Zone hypercellulaire**")

    return "\n".join(lines)


def format_load_status_clinical(load_result: Dict[str, Any]) -> str:
    """Formate le message de chargement du moteur (style clinique)."""
    if load_result["success"]:
        return f"Prêt : {load_result['organ']} ({load_result['model_type']})"
    else:
        return f"Erreur : {load_result['error']}"


def format_organ_change_clinical(change_result: Dict[str, Any]) -> str:
    """Formate le message de changement d'organe (style clinique)."""
    if change_result["success"]:
        model_display = "dédié ★" if "dédié" in change_result["model_type"] else change_result["model_type"]
        return f"Organe: {change_result['organ']} — {model_display}"
    else:
        return f"Erreur: {change_result['error']}"
