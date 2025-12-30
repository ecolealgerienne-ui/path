"""
Configuration des organes pour CellViT-Optimus.

Ce fichier est la SOURCE UNIQUE DE VÉRITÉ pour le mapping organe → modèle.

Usage:
    from src.ui.organ_config import ORGANS, get_model_for_organ, get_organ_choices

Logique:
    - Si l'organe a un modèle dédié → utiliser ce modèle
    - Sinon → utiliser le modèle de la famille d'appartenance
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class OrganConfig:
    """Configuration d'un organe."""
    name: str                      # Nom PanNuke (ex: "Breast", "Lung")
    family: str                    # Famille d'appartenance
    has_dedicated_model: bool      # True si modèle spécifique entraîné
    display_name: str              # Nom affiché dans l'UI


# =============================================================================
# MAPPING ORGANES → FAMILLES (Source: PanNuke dataset)
# =============================================================================
# Les 19 organes de PanNuke groupés par famille

ORGAN_TO_FAMILY = {
    # Glandular (5 organes)
    "Breast": "glandular",
    "Prostate": "glandular",
    "Thyroid": "glandular",
    "Pancreatic": "glandular",
    "Adrenal_gland": "glandular",

    # Digestive (4 organes)
    "Colon": "digestive",
    "Stomach": "digestive",
    "Esophagus": "digestive",
    "Bile-duct": "digestive",

    # Urologic (6 organes)
    "Kidney": "urologic",
    "Bladder": "urologic",
    "Testis": "urologic",
    "Ovarian": "urologic",
    "Uterus": "urologic",
    "Cervix": "urologic",

    # Respiratory (2 organes)
    "Lung": "respiratory",
    "Liver": "respiratory",

    # Epidermal (2 organes)
    "Skin": "epidermal",
    "HeadNeck": "epidermal",
}


# =============================================================================
# ORGANES AVEC MODÈLE DÉDIÉ
# =============================================================================
# Ajouter ici les organes pour lesquels un modèle spécifique a été entraîné.
# Les autres utiliseront le modèle de leur famille.

ORGANS_WITH_DEDICATED_MODEL = {
    "Breast",
    "Colon",
}


# =============================================================================
# CHEMINS DES CHECKPOINTS
# =============================================================================

CHECKPOINT_BASE_PATH = "models/checkpoints_v13_smart_crops"

FAMILY_CHECKPOINTS = {
    "respiratory": f"{CHECKPOINT_BASE_PATH}/hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth",
    "urologic": f"{CHECKPOINT_BASE_PATH}/hovernet_urologic_v13_smart_crops_hybrid_fpn_best.pth",
    "epidermal": f"{CHECKPOINT_BASE_PATH}/hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth",
    "digestive": f"{CHECKPOINT_BASE_PATH}/hovernet_digestive_v13_smart_crops_hybrid_fpn_best.pth",
    "glandular": f"{CHECKPOINT_BASE_PATH}/hovernet_glandular_v13_smart_crops_hybrid_fpn_best.pth",
}

ORGAN_CHECKPOINTS = {
    "Breast": f"{CHECKPOINT_BASE_PATH}/hovernet_Breast_v13_smart_crops_hybrid_fpn_best.pth",
    "Colon": f"{CHECKPOINT_BASE_PATH}/hovernet_Colon_v13_smart_crops_hybrid_fpn_best.pth",
}


# =============================================================================
# PARAMÈTRES WATERSHED PAR FAMILLE
# =============================================================================
# Source: CLAUDE.md - Paramètres optimisés par famille

FAMILY_WATERSHED_PARAMS = {
    "respiratory": {"np_threshold": 0.40, "min_size": 30, "beta": 0.50, "min_distance": 5},
    "urologic": {"np_threshold": 0.45, "min_size": 30, "beta": 0.50, "min_distance": 2},
    "epidermal": {"np_threshold": 0.45, "min_size": 20, "beta": 1.00, "min_distance": 3},
    "digestive": {"np_threshold": 0.45, "min_size": 60, "beta": 2.00, "min_distance": 5},
    "glandular": {"np_threshold": 0.40, "min_size": 30, "beta": 0.50, "min_distance": 5},
}

# Override par organe (optionnel - si vide, utilise les params de la famille)
ORGAN_WATERSHED_PARAMS = {
    # Exemple: si Breast nécessite des params différents de glandular
    # "Breast": {"np_threshold": 0.42, "min_size": 25, "beta": 0.45, "min_distance": 4},
}


# =============================================================================
# CONFIGURATION COMPLÈTE DES ORGANES
# =============================================================================

def _build_organs_config() -> Dict[str, OrganConfig]:
    """Construit la configuration complète de tous les organes."""
    organs = {}

    for organ_name, family in ORGAN_TO_FAMILY.items():
        has_dedicated = organ_name in ORGANS_WITH_DEDICATED_MODEL

        # Nom d'affichage: ajouter indicateur si modèle dédié
        if has_dedicated:
            display_name = f"{organ_name} ★"  # Étoile pour modèle dédié
        else:
            display_name = f"{organ_name} ({family})"

        organs[organ_name] = OrganConfig(
            name=organ_name,
            family=family,
            has_dedicated_model=has_dedicated,
            display_name=display_name,
        )

    return organs


ORGANS: Dict[str, OrganConfig] = _build_organs_config()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_organ_choices() -> List[str]:
    """
    Retourne la liste des organes pour l'UI, triée avec modèles dédiés en premier.

    Returns:
        Liste des noms d'organes pour le dropdown
    """
    # Trier: modèles dédiés d'abord, puis alphabétique par famille
    dedicated = sorted([o for o in ORGANS.keys() if ORGANS[o].has_dedicated_model])
    others = sorted(
        [o for o in ORGANS.keys() if not ORGANS[o].has_dedicated_model],
        key=lambda x: (ORGANS[x].family, x)
    )
    return dedicated + others


def get_organ_display_choices() -> List[str]:
    """
    Retourne la liste des noms d'affichage pour l'UI.

    Returns:
        Liste des display_name pour le dropdown
    """
    organ_names = get_organ_choices()
    return [ORGANS[o].display_name for o in organ_names]


def get_model_for_organ(organ_name: str) -> Dict:
    """
    Retourne les informations du modèle à utiliser pour un organe.

    Args:
        organ_name: Nom de l'organe (ex: "Breast", "Lung")

    Returns:
        Dict avec:
            - checkpoint_path: Chemin du checkpoint
            - family: Famille (pour watershed params si pas de modèle dédié)
            - is_dedicated: True si modèle spécifique
            - watershed_params: Paramètres watershed
    """
    if organ_name not in ORGANS:
        raise ValueError(f"Unknown organ: {organ_name}. Valid: {list(ORGANS.keys())}")

    organ_config = ORGANS[organ_name]
    family = organ_config.family

    # Déterminer le checkpoint
    if organ_config.has_dedicated_model:
        checkpoint = ORGAN_CHECKPOINTS.get(organ_name)
        is_dedicated = True
    else:
        checkpoint = FAMILY_CHECKPOINTS.get(family)
        is_dedicated = False

    # Déterminer les params watershed (organe-spécifique ou famille)
    if organ_name in ORGAN_WATERSHED_PARAMS:
        watershed_params = ORGAN_WATERSHED_PARAMS[organ_name]
    else:
        watershed_params = FAMILY_WATERSHED_PARAMS.get(family, FAMILY_WATERSHED_PARAMS["respiratory"])

    return {
        "checkpoint_path": checkpoint,
        "family": family,
        "is_dedicated": is_dedicated,
        "watershed_params": watershed_params,
        "display_name": organ_config.display_name,
    }


def get_family_for_organ(organ_name: str) -> str:
    """
    Retourne la famille d'un organe.

    Args:
        organ_name: Nom de l'organe

    Returns:
        Nom de la famille
    """
    if organ_name not in ORGAN_TO_FAMILY:
        raise ValueError(f"Unknown organ: {organ_name}")
    return ORGAN_TO_FAMILY[organ_name]


def organ_has_dedicated_model(organ_name: str) -> bool:
    """
    Vérifie si un organe a un modèle dédié.

    Args:
        organ_name: Nom de l'organe

    Returns:
        True si modèle dédié existe
    """
    return organ_name in ORGANS_WITH_DEDICATED_MODEL


def get_organs_by_family(family: str) -> List[str]:
    """
    Retourne tous les organes d'une famille.

    Args:
        family: Nom de la famille

    Returns:
        Liste des noms d'organes
    """
    return [organ for organ, fam in ORGAN_TO_FAMILY.items() if fam == family]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Valide la cohérence de la configuration."""
    errors = []

    # Vérifier que tous les organes avec modèle dédié ont un checkpoint
    for organ in ORGANS_WITH_DEDICATED_MODEL:
        if organ not in ORGAN_CHECKPOINTS:
            errors.append(f"Missing checkpoint for dedicated organ: {organ}")

    # Vérifier que toutes les familles ont un checkpoint
    families = set(ORGAN_TO_FAMILY.values())
    for family in families:
        if family not in FAMILY_CHECKPOINTS:
            errors.append(f"Missing checkpoint for family: {family}")
        if family not in FAMILY_WATERSHED_PARAMS:
            errors.append(f"Missing watershed params for family: {family}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))

    return True


# Valider à l'import
validate_config()
