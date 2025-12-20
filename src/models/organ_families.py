"""
Mapping des organes vers les familles pour HoVer-Net spécialisés.

Basé sur la littérature scientifique (MICCAI, Nature Communications sur PanNuke):
- Feature Sharing: Les noyaux partagent des propriétés physiques
- Domain-Specific Variance: L'erreur augmente entre organes de textures différentes
- Domain Adaptation: Le transfert fonctionne mieux entre organes de même famille embryologique
"""

from typing import Dict, List

# Mapping organe → famille
ORGAN_TO_FAMILY: Dict[str, str] = {
    # Glandulaire & Hormonale (acini, sécrétions)
    "Breast": "glandular",
    "Prostate": "glandular",
    "Thyroid": "glandular",
    "Pancreatic": "glandular",
    "Adrenal_gland": "glandular",

    # Digestive (formes tubulaires)
    "Colon": "digestive",
    "Stomach": "digestive",
    "Esophagus": "digestive",
    "Bile-duct": "digestive",

    # Urologique & Reproductif (densité nucléaire)
    "Kidney": "urologic",
    "Bladder": "urologic",
    "Testis": "urologic",
    "Ovarian": "urologic",
    "Uterus": "urologic",
    "Cervix": "urologic",

    # Respiratoire & Hépatique (structures ouvertes, travées)
    "Lung": "respiratory",
    "Liver": "respiratory",

    # Épidermoïde (couches stratifiées, kératine)
    "Skin": "epidermal",
    "HeadNeck": "epidermal",
}

# Liste des familles
FAMILIES: List[str] = ["glandular", "digestive", "urologic", "respiratory", "epidermal"]

# Mapping inverse: famille → liste d'organes
FAMILY_TO_ORGANS: Dict[str, List[str]] = {
    "glandular": ["Breast", "Prostate", "Thyroid", "Pancreatic", "Adrenal_gland"],
    "digestive": ["Colon", "Stomach", "Esophagus", "Bile-duct"],
    "urologic": ["Kidney", "Bladder", "Testis", "Ovarian", "Uterus", "Cervix"],
    "respiratory": ["Lung", "Liver"],
    "epidermal": ["Skin", "HeadNeck"],
}

# Descriptions des familles (pour documentation)
FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "glandular": "Glandulaire & Hormonale - Focus sur acini et sécrétions",
    "digestive": "Digestive - Formes tubulaires cohérentes",
    "urologic": "Urologique & Reproductif - Focus sur densité nucléaire",
    "respiratory": "Respiratoire & Hépatique - Structures ouvertes et travées",
    "epidermal": "Épidermoïde - Couches stratifiées et kératine",
}


def get_family(organ: str) -> str:
    """Retourne la famille d'un organe."""
    # Normaliser le nom (gérer les variations)
    organ_normalized = organ.strip()

    if organ_normalized in ORGAN_TO_FAMILY:
        return ORGAN_TO_FAMILY[organ_normalized]

    # Chercher correspondance partielle (case insensitive)
    organ_lower = organ_normalized.lower().replace("_", "").replace("-", "")
    for org, family in ORGAN_TO_FAMILY.items():
        if org.lower().replace("_", "") == organ_lower:
            return family

    raise ValueError(f"Organe non reconnu: {organ}")


def get_organs(family: str) -> List[str]:
    """Retourne la liste des organes d'une famille."""
    if family not in FAMILY_TO_ORGANS:
        raise ValueError(f"Famille non reconnue: {family}. Choix: {FAMILIES}")
    return FAMILY_TO_ORGANS[family]
