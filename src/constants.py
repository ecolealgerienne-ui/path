"""
Constantes globales du projet.

Ce fichier est la SOURCE UNIQUE DE VÉRITÉ pour toutes les dimensions,
résolutions et paramètres fixes du projet.

Principe: Une constante définie ICI est utilisée PARTOUT, jamais redéfinie.
"""

# =============================================================================
# TAILLES D'IMAGES
# =============================================================================

# H-optimus-0 backbone (ViT-Giant/14)
HOPTIMUS_INPUT_SIZE = 224  # Taille d'entrée fixe du modèle
HOPTIMUS_PATCH_SIZE = 14   # Taille des patches ViT
HOPTIMUS_NUM_PATCHES = 256  # (224 / 14)^2 = 256 patches
HOPTIMUS_EMBED_DIM = 1536  # Dimension des embeddings

# PanNuke dataset
PANNUKE_IMAGE_SIZE = 256  # Taille originale des images PanNuke
PANNUKE_NUM_CLASSES = 5   # Neoplastic, Inflammatory, Connective, Dead, Epithelial
PANNUKE_NUM_ORGANS = 19   # 19 organes dans PanNuke

# HoVer-Net decoder
HOVERNET_OUTPUT_SIZE = HOPTIMUS_INPUT_SIZE  # Sorties à la même taille que l'input (224×224)

# =============================================================================
# NORMALISATION H-OPTIMUS-0
# =============================================================================

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

# Validation features
HOPTIMUS_CLS_STD_MIN = 0.70  # Minimum attendu pour CLS std (détecte Bug #2 LayerNorm)
HOPTIMUS_CLS_STD_MAX = 0.90  # Maximum attendu

# =============================================================================
# ARCHITECTURE HOVERNET
# =============================================================================

# Branches HoVer-Net
HOVERNET_NP_CHANNELS = 1   # Nuclear Presence (binaire)
HOVERNET_HV_CHANNELS = 2   # Horizontal + Vertical gradients
HOVERNET_NT_CHANNELS = PANNUKE_NUM_CLASSES  # Nuclear Type (5 classes)

# HV Maps range
HV_MAP_MIN = -1.0  # Minimum des cartes HV (distance normalisée)
HV_MAP_MAX = 1.0   # Maximum des cartes HV

# =============================================================================
# ORGAN FAMILIES
# =============================================================================

FAMILIES = ["glandular", "digestive", "urologic", "respiratory", "epidermal"]
NUM_FAMILIES = len(FAMILIES)

# =============================================================================
# DEVICE & PERFORMANCE
# =============================================================================

DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 8
MAX_VRAM_GB = 12  # RTX 4070 SUPER

# =============================================================================
# PATHS (relative to project root)
# =============================================================================

# Source data
DEFAULT_PANNUKE_DIR = "/home/amar/data/PanNuke"

# Checkpoints
DEFAULT_CHECKPOINT_DIR = "models/checkpoints"

# Features cache
DEFAULT_FEATURES_CACHE_DIR = "data/cache/pannuke_features"

# ⚠️ CRITICAL: Family data path (Bug #6 fix)
# This is the SINGLE SOURCE OF TRUTH for family features/targets location
# Used by: train_hovernet_family.py, test_on_training_data.py, test_aji_v8.py, etc.
DEFAULT_FAMILY_DATA_DIR = "data/family_data"  # ← Validated path (exists, CLS std 0.770)

# FIXED data (v8 with proper HV normalization)
DEFAULT_FAMILY_FIXED_DIR = "data/family_FIXED"

# =============================================================================
# TEMPERATURE SCALING (CALIBRATION)
# =============================================================================

DEFAULT_TEMPERATURE_ORGAN_HEAD = 0.5  # Calibration OrganHead (empirique)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_image_size_mismatch_info() -> dict:
    """
    Retourne les informations de mismatch entre HoVer-Net et PanNuke.

    Returns:
        {
            "hovernet_size": 224,
            "pannuke_size": 256,
            "needs_resize": True,
            "resize_direction": "predictions → ground_truth"
        }
    """
    return {
        "hovernet_size": HOVERNET_OUTPUT_SIZE,
        "pannuke_size": PANNUKE_IMAGE_SIZE,
        "needs_resize": HOVERNET_OUTPUT_SIZE != PANNUKE_IMAGE_SIZE,
        "resize_direction": "predictions → ground_truth" if HOVERNET_OUTPUT_SIZE < PANNUKE_IMAGE_SIZE else "ground_truth → predictions"
    }


def validate_image_size(image_size: int, expected: str = "hovernet") -> bool:
    """
    Valide qu'une taille d'image correspond à l'attendu.

    Args:
        image_size: Taille à vérifier
        expected: "hovernet" (224) ou "pannuke" (256)

    Returns:
        True si la taille correspond

    Raises:
        ValueError: Si taille incorrecte avec message explicatif
    """
    if expected == "hovernet":
        if image_size != HOVERNET_OUTPUT_SIZE:
            raise ValueError(
                f"Taille HoVer-Net incorrecte: {image_size} "
                f"(attendu: {HOVERNET_OUTPUT_SIZE}). "
                f"Vérifier que l'input H-optimus-0 est bien à {HOPTIMUS_INPUT_SIZE}×{HOPTIMUS_INPUT_SIZE}."
            )
    elif expected == "pannuke":
        if image_size != PANNUKE_IMAGE_SIZE:
            raise ValueError(
                f"Taille PanNuke incorrecte: {image_size} "
                f"(attendu: {PANNUKE_IMAGE_SIZE}). "
                f"Vérifier que le ground truth vient bien de PanNuke."
            )
    else:
        raise ValueError(f"Type attendu invalide: {expected}. Utiliser 'hovernet' ou 'pannuke'.")

    return True
