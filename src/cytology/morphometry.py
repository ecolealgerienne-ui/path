"""
V14 Cytology — Morphometric Features Extraction

Calcul des 20 features morphométriques à partir des masques CellPose.
SINGLE SOURCE OF TRUTH: Les features sont calculées sur les masques générés,
JAMAIS lues depuis CSV/Excel externe.

Basé sur:
- ISBI 2014 MITOS-ATYPIA (6 Universal Criteria)
- Specs Expert V14 (20 features)
- Paris System (N/C ratio > 0.7)
- Bethesda System (Thyroid cytology)

Author: V14 Cytology Branch
Date: 2026-01-19
"""

import numpy as np
from typing import Dict, Optional, Tuple
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import convex_hull_image
import warnings

from src.preprocessing.stain_separation import ruifrok_extract_h_channel


# ═════════════════════════════════════════════════════════════════════════════
#  MORPHOMETRIC FEATURES EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_single_cell_features(
    image_rgb: np.ndarray,
    mask_nucleus: np.ndarray,
    mask_cytoplasm: Optional[np.ndarray] = None,
    pixel_size_um: float = 0.25
) -> np.ndarray:
    """
    Calcule 20 features morphométriques pour UNE cellule

    Args:
        image_rgb: Image RGB (H, W, 3), valeurs [0, 255]
        mask_nucleus: Masque binaire noyau (H, W), valeurs {0, 1}
        mask_cytoplasm: Masque binaire cytoplasme (optionnel)
        pixel_size_um: Taille pixel en microns (0.25 = 0.5 MPP)

    Returns:
        features: np.ndarray shape (20,)
                  Valeurs: np.nan si feature non calculable

    Features List:
        1. area_nucleus (µm²)
        2. perimeter (µm)
        3. eccentricity (sans unité)
        4. solidity (sans unité)
        5. major_axis_length (µm)
        6. minor_axis_length (µm)
        7. extent (sans unité)
        8. convex_area (µm²)
        9. circularity (sans unité) — Criterion 5 (Regularity)
        10. compactness (sans unité)
        11. mean_intensity (0-255)
        12. integrated_od (sans unité)
        13. mean_h_channel (OD) — Criterion 3 (Chromatin Density)
        14. haralick_contrast (sans unité)
        15. haralick_energy (sans unité)
        16. haralick_correlation (sans unité)
        17. area_cytoplasm (µm²)
        18. nc_ratio (sans unité) — CRITICAL (Paris/Bethesda)
        19. feret_diameter_max (µm)
        20. roundness (sans unité)

    Notes:
        - Features 17-19: Requièrent mask_cytoplasm, sinon np.nan
        - Feature 1 (Size of Nuclei) → ISBI Criterion 1
        - Feature 13 (Chromatin Density) → ISBI Criterion 3
        - Feature 18 (N/C Ratio) → Paris System (> 0.7 = High Grade)
    """
    features = {}

    # Validation masque
    if mask_nucleus.sum() == 0:
        warnings.warn("Empty nucleus mask — returning NaN features")
        return np.full(20, np.nan, dtype=np.float32)

    # Conversion facteur pixel → microns
    pixel_to_um2 = pixel_size_um ** 2

    # ═════════════════════════════════════════════════════════════════════════
    #  REGIONPROPS (Features 1-8, 19)
    # ═════════════════════════════════════════════════════════════════════════

    # Label le masque (au cas où multiple components, prendre le plus grand)
    labeled_nucleus = label(mask_nucleus)
    props_list = regionprops(labeled_nucleus, intensity_image=image_rgb[:, :, 0])

    if len(props_list) == 0:
        return np.full(20, np.nan, dtype=np.float32)

    # Prendre la région la plus grande (si fragmentation)
    props_nucleus = max(props_list, key=lambda p: p.area)

    # 1. Area (µm²) — ISBI Criterion 1
    features['area_nucleus'] = props_nucleus.area * pixel_to_um2

    # 2. Perimeter (µm)
    features['perimeter'] = props_nucleus.perimeter * pixel_size_um

    # 3. Eccentricity (0 = circle, 1 = line)
    features['eccentricity'] = props_nucleus.eccentricity

    # 4. Solidity (area / convex_area) — ISBI Criterion 5 (Regularity)
    features['solidity'] = props_nucleus.solidity

    # 5-6. Major/Minor Axis (µm)
    features['major_axis_length'] = props_nucleus.major_axis_length * pixel_size_um
    features['minor_axis_length'] = props_nucleus.minor_axis_length * pixel_size_um

    # 7. Extent (area / bbox_area)
    features['extent'] = props_nucleus.extent

    # 8. Convex Area (µm²)
    # Note: regionprops.convex_area est déjà calculé
    features['convex_area'] = props_nucleus.convex_area * pixel_to_um2

    # 19. Feret Diameter Max (µm)
    # Note: Disponible dans scikit-image >= 0.20
    if hasattr(props_nucleus, 'feret_diameter_max'):
        features['feret_diameter_max'] = props_nucleus.feret_diameter_max * pixel_size_um
    else:
        # Approximation: major_axis_length
        features['feret_diameter_max'] = props_nucleus.major_axis_length * pixel_size_um

    # ═════════════════════════════════════════════════════════════════════════
    #  DERIVED GEOMETRIC FEATURES (9-10, 20)
    # ═════════════════════════════════════════════════════════════════════════

    # 9. Circularity (4π × area / perimeter²) — ISBI Criterion 5
    # Valeur: 1.0 = cercle parfait, < 1.0 = irrégulier
    if props_nucleus.perimeter > 0:
        features['circularity'] = (
            4 * np.pi * props_nucleus.area / (props_nucleus.perimeter ** 2)
        )
    else:
        features['circularity'] = np.nan

    # 10. Compactness (perimeter² / area)
    # Inverse de circularity (normalisé)
    if props_nucleus.area > 0:
        features['compactness'] = props_nucleus.perimeter ** 2 / props_nucleus.area
    else:
        features['compactness'] = np.nan

    # 20. Roundness (4 × area / (π × major_axis²))
    # Alternative à circularity
    if props_nucleus.major_axis_length > 0:
        features['roundness'] = (
            4 * props_nucleus.area / (np.pi * props_nucleus.major_axis_length ** 2)
        )
    else:
        features['roundness'] = np.nan

    # ═════════════════════════════════════════════════════════════════════════
    #  INTENSITY FEATURES (11-13)
    # ═════════════════════════════════════════════════════════════════════════

    # 11. Mean Intensity (canal rouge)
    features['mean_intensity'] = props_nucleus.mean_intensity

    # 12-13. Canal H (Ruifrok) — ISBI Criterion 3 (Chromatin Density)
    try:
        h_channel = ruifrok_extract_h_channel(image_rgb, normalize=True)
        h_masked = h_channel * mask_nucleus

        # 12. Integrated OD (somme totale)
        features['integrated_od'] = h_masked.sum()

        # 13. Mean H-channel OD — CRITICAL pour Criterion 3
        features['mean_h_channel'] = h_masked[mask_nucleus > 0].mean()

    except Exception as e:
        warnings.warn(f"H-channel extraction failed: {e}")
        features['integrated_od'] = np.nan
        features['mean_h_channel'] = np.nan

    # ═════════════════════════════════════════════════════════════════════════
    #  HARALICK TEXTURE FEATURES (14-16) — Chromatin Pattern
    # ═════════════════════════════════════════════════════════════════════════

    try:
        # Extraire bbox du noyau
        bbox = props_nucleus.bbox  # (min_row, min_col, max_row, max_col)
        nucleus_patch = image_rgb[bbox[0]:bbox[2], bbox[1]:bbox[3], 0]  # Canal rouge

        # Redimensionner intensités [0, 255] → [0, 63] pour GLCM (réduit compute)
        nucleus_patch_scaled = (nucleus_patch / 4).astype(np.uint8)

        # Gray-Level Co-occurrence Matrix
        glcm = graycomatrix(
            nucleus_patch_scaled,
            distances=[1],       # Distance 1 pixel
            angles=[0],          # Direction horizontale (simplification)
            levels=64,           # 64 niveaux de gris
            symmetric=True,
            normed=True
        )

        # 14. Contrast (mesure variations locales)
        features['haralick_contrast'] = graycoprops(glcm, 'contrast')[0, 0]

        # 15. Energy (homogénéité texture)
        features['haralick_energy'] = graycoprops(glcm, 'energy')[0, 0]

        # 16. Correlation (dépendance spatiale)
        features['haralick_correlation'] = graycoprops(glcm, 'correlation')[0, 0]

    except Exception as e:
        warnings.warn(f"Haralick features failed: {e}")
        features['haralick_contrast'] = np.nan
        features['haralick_energy'] = np.nan
        features['haralick_correlation'] = np.nan

    # ═════════════════════════════════════════════════════════════════════════
    #  CYTOPLASM FEATURES (17-18) — CONDITIONAL
    # ═════════════════════════════════════════════════════════════════════════

    if mask_cytoplasm is not None and mask_cytoplasm.sum() > 0:
        labeled_cyto = label(mask_cytoplasm)
        props_cyto_list = regionprops(labeled_cyto)

        if len(props_cyto_list) > 0:
            props_cyto = max(props_cyto_list, key=lambda p: p.area)

            # 17. Area Cytoplasm (µm²)
            features['area_cytoplasm'] = props_cyto.area * pixel_to_um2

            # 18. N/C Ratio — CRITICAL (Paris System, Bethesda)
            # Paris System: N/C > 0.7 → High Grade Urothelial Carcinoma
            # Bethesda: Élevé → Carcinome papillaire/folliculaire thyroïde
            if props_cyto.area > 0:
                features['nc_ratio'] = props_nucleus.area / props_cyto.area
            else:
                features['nc_ratio'] = np.nan
        else:
            features['area_cytoplasm'] = np.nan
            features['nc_ratio'] = np.nan
    else:
        # Cytoplasme non disponible (CellPose Slave skippé)
        features['area_cytoplasm'] = np.nan
        features['nc_ratio'] = np.nan

    # ═════════════════════════════════════════════════════════════════════════
    #  CONVERSION TO ARRAY
    # ═════════════════════════════════════════════════════════════════════════

    # Ordre fixe des features (IMPORTANT pour MLP)
    feature_names = [
        'area_nucleus', 'perimeter', 'eccentricity', 'solidity',
        'major_axis_length', 'minor_axis_length', 'extent', 'convex_area',
        'circularity', 'compactness',
        'mean_intensity', 'integrated_od', 'mean_h_channel',
        'haralick_contrast', 'haralick_energy', 'haralick_correlation',
        'area_cytoplasm', 'nc_ratio', 'feret_diameter_max', 'roundness'
    ]

    feature_vector = np.array(
        [features[name] for name in feature_names],
        dtype=np.float32
    )

    return feature_vector


def compute_batch_features(
    images: np.ndarray,
    masks_nuclei: np.ndarray,
    masks_cytoplasm: Optional[np.ndarray] = None,
    pixel_size_um: float = 0.25
) -> np.ndarray:
    """
    Calcule features morphométriques pour un batch de cellules

    Args:
        images: Batch d'images RGB (N, H, W, 3)
        masks_nuclei: Batch de masques noyaux (N, H, W)
        masks_cytoplasm: Batch de masques cytoplasme (optionnel) (N, H, W)
        pixel_size_um: Taille pixel en microns

    Returns:
        features: np.ndarray shape (N, 20)
    """
    N = len(images)
    features_batch = np.zeros((N, 20), dtype=np.float32)

    for i in range(N):
        mask_cyto_i = masks_cytoplasm[i] if masks_cytoplasm is not None else None

        features_batch[i] = compute_single_cell_features(
            image_rgb=images[i],
            mask_nucleus=masks_nuclei[i],
            mask_cytoplasm=mask_cyto_i,
            pixel_size_um=pixel_size_um
        )

    return features_batch


def get_feature_names() -> list:
    """
    Retourne les noms des 20 features morphométriques (ordre fixe)

    Returns:
        feature_names: List[str] (20 éléments)
    """
    return [
        'area_nucleus',
        'perimeter',
        'eccentricity',
        'solidity',
        'major_axis_length',
        'minor_axis_length',
        'extent',
        'convex_area',
        'circularity',
        'compactness',
        'mean_intensity',
        'integrated_od',
        'mean_h_channel',
        'haralick_contrast',
        'haralick_energy',
        'haralick_correlation',
        'area_cytoplasm',
        'nc_ratio',
        'feret_diameter_max',
        'roundness'
    ]


def validate_features(features: np.ndarray) -> Tuple[bool, str]:
    """
    Valide un vecteur de features morphométriques

    Args:
        features: np.ndarray shape (20,)

    Returns:
        is_valid: bool
        message: str (raison si invalide)
    """
    if features.shape != (20,):
        return False, f"Shape incorrecte: {features.shape}, attendu (20,)"

    # Vérifier nombre de NaN (max 3 acceptable: area_cytoplasm, nc_ratio, feret_diameter_max)
    n_nan = np.isnan(features).sum()
    if n_nan > 5:  # Tolérance 5 NaN max
        return False, f"Trop de NaN: {n_nan}/20 features"

    # Vérifier valeurs nucléaires critiques (doivent être présentes)
    critical_indices = [0, 1, 9, 12]  # area, perimeter, circularity, mean_h_channel
    critical_nan = np.isnan(features[critical_indices]).sum()
    if critical_nan > 0:
        return False, f"Features critiques manquantes (indices {critical_indices})"

    # Vérifier plages de valeurs
    if features[0] <= 0:  # area_nucleus
        return False, f"area_nucleus invalide: {features[0]}"

    if features[9] > 1.0:  # circularity (doit être ≤ 1.0)
        return False, f"circularity invalide: {features[9]} > 1.0"

    return True, "OK"


# ═════════════════════════════════════════════════════════════════════════════
#  CLINICAL INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════

def interpret_nc_ratio(nc_ratio: float, organ: str = "bladder") -> Dict[str, str]:
    """
    Interprète le ratio N/C selon critères cliniques

    Args:
        nc_ratio: Ratio Nucleus/Cytoplasm (0-1)
        organ: Organe ("bladder", "thyroid", "cervix")

    Returns:
        interpretation: Dict avec clés "grade", "message", "action"

    References:
        - Paris System (Bladder): N/C > 0.7 = High Grade
        - Bethesda System (Thyroid): N/C élevé = Carcinome
    """
    if np.isnan(nc_ratio):
        return {
            "grade": "UNKNOWN",
            "message": "N/C ratio non disponible (cytoplasme non segmenté)",
            "action": "Utiliser critères nucléaires seuls"
        }

    if organ.lower() == "bladder":
        # Paris System (Urine Cytology)
        if nc_ratio > 0.7:
            return {
                "grade": "HIGH_GRADE",
                "message": "⚠️ Paris System: N/C > 0.7 — Suspicion Haut Grade (HGUC)",
                "action": "Cystoscopie + Biopsie URGENTE"
            }
        elif nc_ratio > 0.5:
            return {
                "grade": "ATYPICAL",
                "message": "⚠️ N/C > 0.5 — Atypique, surveillance rapprochée",
                "action": "Contrôle cytologie 3 mois"
            }
        else:
            return {
                "grade": "BENIGN",
                "message": "✅ N/C < 0.5 — Bénin",
                "action": "Surveillance standard"
            }

    elif organ.lower() == "thyroid":
        # Bethesda System (approximation)
        if nc_ratio > 0.6:
            return {
                "grade": "SUSPICIOUS",
                "message": "⚠️ N/C élevé — Suspicion néoplasie folliculaire/papillaire",
                "action": "Biopsie chirurgicale recommandée"
            }
        else:
            return {
                "grade": "BENIGN",
                "message": "✅ N/C normal",
                "action": "Surveillance standard"
            }

    else:  # cervix, generic
        if nc_ratio > 0.8:
            return {
                "grade": "ABNORMAL",
                "message": "⚠️ N/C très élevé — Anomalie nucléaire",
                "action": "Review pathologiste"
            }
        else:
            return {
                "grade": "NORMAL",
                "message": "✅ N/C normal",
                "action": "Surveillance standard"
            }


def interpret_chromatin_density(mean_h_od: float) -> Dict[str, str]:
    """
    Interprète la densité de chromatine (ISBI Criterion 3)

    Args:
        mean_h_od: Optical Density moyenne du canal H (Ruifrok)

    Returns:
        interpretation: Dict avec clés "grade", "message"

    Reference:
        ISBI 2014 MITOS-ATYPIA Table 2 — Criterion 3
    """
    if np.isnan(mean_h_od):
        return {
            "grade": "UNKNOWN",
            "message": "Densité chromatine non calculable"
        }

    # Seuils approximatifs (à calibrer sur ISBI dataset)
    if mean_h_od > 0.6:
        return {
            "grade": "HIGH",
            "message": "⚠️ Chromatine dense (Criterion 3 positif) — Suspect malignité"
        }
    elif mean_h_od > 0.4:
        return {
            "grade": "MODERATE",
            "message": "⚠️ Chromatine modérément dense — Surveillance"
        }
    else:
        return {
            "grade": "LOW",
            "message": "✅ Chromatine claire — Bénin probable"
        }
