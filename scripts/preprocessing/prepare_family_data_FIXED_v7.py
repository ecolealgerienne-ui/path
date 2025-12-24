#!/usr/bin/env python3
"""
PREPARE FAMILY DATA - VERSION v7 (RADIAL NORMALIZATION)

Changements v6 → v7:
- ❌ v6: Normalisation par bounding box (rectangle)
      v_dist = (y_coords - center_y) / bbox_h
      h_dist = (x_coords - center_x) / bbox_w

- ✅ v7: Normalisation par distance radiale max (cercle)
      y_dist = y_coords - center_y
      x_dist = x_coords - center_x
      dist_max = max(sqrt(y_dist² + x_dist²))
      v_dist = y_dist / dist_max
      h_dist = x_dist / dist_max

Objectif: Distance centroïdes <2px (actuellement 40.51px avec v6)

Usage:
    python scripts/preprocessing/prepare_family_data_FIXED_v7.py --family epidermal
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE


# =============================================================================
# ORGAN TO FAMILY MAPPING
# =============================================================================

ORGAN_TO_FAMILY = {
    # Glandulaire & Hormonale
    "Breast": "glandular",
    "Prostate": "glandular",
    "Thyroid": "glandular",
    "Pancreatic": "glandular",
    "Adrenal_gland": "glandular",

    # Digestive
    "Colon": "digestive",
    "Stomach": "digestive",
    "Esophagus": "digestive",
    "Bile-duct": "digestive",

    # Urologique & Reproductif
    "Kidney": "urologic",
    "Bladder": "urologic",
    "Testis": "urologic",
    "Ovarian": "urologic",
    "Uterus": "urologic",
    "Cervix": "urologic",

    # Respiratoire & Hépatique
    "Lung": "respiratory",
    "Liver": "respiratory",

    # Épidermoïde
    "Skin": "epidermal",
    "HeadNeck": "epidermal",
}


# =============================================================================
# HV MAPS COMPUTATION (VERSION v7 - RADIAL NORMALIZATION)
# =============================================================================

def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """
    Calcule les cartes HV avec NORMALISATION RADIALE (v7).

    VERSION v7 - CHANGEMENT CRITIQUE:
    - ❌ v6: Normalisation par bbox (rectangle)
          v_dist = (y_coords - center_y) / bbox_h
          h_dist = (x_coords - center_x) / bbox_w

    - ✅ v7: Normalisation par distance radiale max (cercle)
          y_dist = y_coords - center_y
          x_dist = x_coords - center_x
          dist_max = max(sqrt(y_dist² + x_dist²))
          v_dist = y_dist / dist_max
          h_dist = x_dist / dist_max

    Avantages v7:
    - Invariant par forme (pas dépendant du ratio bbox_h/bbox_w)
    - Valeurs HV cohérentes quelle que soit l'orientation du noyau
    - Meilleur alignement spatial avec les vrais centroïdes

    Args:
        inst_map: Instance map (H, W) avec IDs d'instances (0 = background)

    Returns:
        hv_map: Cartes HV (2, H, W) en float32 [-1, 1]
                hv_map[0] = Vertical (Y)
                hv_map[1] = Horizontal (X)
    """
    h, w = inst_map.shape
    hv_map = np.zeros((2, h, w), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) == 0:
            continue

        # Calculer le centroïde
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)

        # ✅ v7: Distance depuis le centroïde
        y_dist = y_coords - center_y
        x_dist = x_coords - center_x

        # ✅ v7: Normalisation par distance radiale MAX (circular)
        # dist_max = maximum euclidean distance from centroid
        dist_max = np.max(np.sqrt(y_dist**2 + x_dist**2)) + 1e-7

        v_dist = y_dist / dist_max  # Range: [-1, 1]
        h_dist = x_dist / dist_max  # Range: [-1, 1]

        # Clip à [-1, 1] (sécurité)
        v_dist = np.clip(v_dist, -1.0, 1.0)
        h_dist = np.clip(h_dist, -1.0, 1.0)

        # Convention standard: hv_map[0] = V, hv_map[1] = H
        hv_map[0, y_coords, x_coords] = v_dist
        hv_map[1, y_coords, x_coords] = h_dist

    # Gaussian smoothing (sigma=0.5) pour réduire le bruit
    hv_map[0] = gaussian_filter(hv_map[0], sigma=0.5)
    hv_map[1] = gaussian_filter(hv_map[1], sigma=0.5)

    return hv_map


# =============================================================================
# MASK NORMALIZATION
# =============================================================================

def normalize_mask_format(mask: np.ndarray) -> np.ndarray:
    """
    Normalise le format du mask vers HWC (256, 256, 6).

    Args:
        mask: Mask PanNuke (peut être HWC ou CHW)

    Returns:
        mask_hwc: Mask au format (256, 256, 6)
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D with shape {mask.shape}")

    if mask.shape == (PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE, 6):
        return mask
    elif mask.shape == (6, PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE):
        mask_hwc = np.transpose(mask, (1, 2, 0))
        mask_hwc = np.ascontiguousarray(mask_hwc)
        return mask_hwc
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")


# =============================================================================
# INSTANCE MAP EXTRACTION (NATIVE PANNUKE IDS)
# =============================================================================

def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """
    Extrait les vraies instances de PanNuke avec IDs séparés.

    CRITIQUE: Utilise les IDs natifs des canaux 1-4 au lieu de label(np_mask > 0)
    qui fusionnerait les noyaux qui se touchent.

    Args:
        mask: Mask PanNuke (H, W, 6)

    Returns:
        inst_map: Instance map (H, W) avec IDs séparés (0 = background)
    """
    mask = normalize_mask_format(mask)

    inst_map = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: IDs d'instances natifs PanNuke (déjà séparés)
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
    epithelial_mask = mask[:, :, 5]
    if epithelial_mask.max() > 0:
        _, epithelial_labels = cv2.connectedComponents(epithelial_mask.astype(np.uint8))
        epithelial_ids = np.unique(epithelial_labels)
        epithelial_ids = epithelial_ids[epithelial_ids > 0]

        for epi_id in epithelial_ids:
            epi_mask = epithelial_labels == epi_id
            inst_map[epi_mask] = instance_counter
            instance_counter += 1

    return inst_map


# =============================================================================
# NP TARGET GENERATION
# =============================================================================

def compute_np_target(mask: np.ndarray) -> np.ndarray:
    """
    Génère le target NP (Nuclear Presence).

    Args:
        mask: Mask PanNuke (H, W, 6)

    Returns:
        np_target: Binary map (H, W) en float32 [0, 1]
    """
    mask = normalize_mask_format(mask)

    # Union binaire des canaux 1-5 (excluant canal 0 = background)
    np_target = mask[:, :, 1:].sum(axis=-1) > 0

    return np_target.astype(np.float32)


# =============================================================================
# NT TARGET GENERATION
# =============================================================================

def compute_nt_target(mask: np.ndarray) -> np.ndarray:
    """
    Génère le target NT (Nuclear Type).

    Args:
        mask: Mask PanNuke (H, W, 6)

    Returns:
        nt_target: Class map (H, W) en int64 [0-5]
                   0 = Background
                   1 = Neoplastic
                   2 = Inflammatory
                   3 = Connective
                   4 = Dead
                   5 = Epithelial
    """
    mask = normalize_mask_format(mask)

    nt_target = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int64)

    # Priorité: dernier canal écrase les précédents
    for c in range(1, 6):
        channel_mask = mask[:, :, c] > 0
        nt_target[channel_mask] = c

    return nt_target


# =============================================================================
# MAIN PREPARATION FUNCTION
# =============================================================================

def prepare_family_data(family: str, pannuke_dir: Path, output_dir: Path):
    """
    Prépare les données pour une famille (v7 - radial normalization).

    Args:
        family: Nom de la famille (glandular, digestive, etc.)
        pannuke_dir: Répertoire PanNuke source
        output_dir: Répertoire de sortie
    """
    print("=" * 80)
    print(f"PRÉPARATION DONNÉES FAMILLE: {family.upper()} (VERSION v7 - RADIAL)")
    print("=" * 80)
    print()

    # Sélectionner les organes de cette famille
    family_organs = [organ for organ, fam in ORGAN_TO_FAMILY.items() if fam == family]

    print(f"Organes de la famille '{family}':")
    for organ in family_organs:
        print(f"  - {organ}")
    print()

    # Charger les 3 folds
    all_images = []
    all_np_targets = []
    all_hv_targets = []
    all_nt_targets = []
    all_fold_ids = []
    all_image_ids = []

    for fold in [0, 1, 2]:
        fold_dir = pannuke_dir / f"fold{fold}"

        images_path = fold_dir / "images.npy"
        masks_path = fold_dir / "masks.npy"
        types_path = fold_dir / "types.npy"

        if not images_path.exists():
            print(f"⚠️  Fold {fold} non trouvé: {images_path}")
            continue

        # Charger fold
        images = np.load(images_path, mmap_mode='r')
        masks = np.load(masks_path, mmap_mode='r')
        types = np.load(types_path)

        print(f"Fold {fold}: {len(images)} échantillons")

        # Filtrer par famille
        for i, organ in enumerate(types):
            if organ not in family_organs:
                continue

            image = images[i]
            mask = masks[i]

            # Générer targets
            inst_map = extract_pannuke_instances(mask)
            np_target = compute_np_target(mask)
            hv_target = compute_hv_maps(inst_map)  # ✅ v7 radial normalization
            nt_target = compute_nt_target(mask)

            all_images.append(image)
            all_np_targets.append(np_target)
            all_hv_targets.append(hv_target)
            all_nt_targets.append(nt_target)
            all_fold_ids.append(fold)
            all_image_ids.append(i)

    # Convertir en arrays
    n_samples = len(all_images)

    print()
    print(f"Total échantillons famille '{family}': {n_samples}")

    if n_samples == 0:
        print("❌ Aucun échantillon trouvé pour cette famille!")
        return

    images_array = np.stack(all_images, axis=0)
    np_targets_array = np.stack(all_np_targets, axis=0)
    hv_targets_array = np.stack(all_hv_targets, axis=0)
    nt_targets_array = np.stack(all_nt_targets, axis=0)
    fold_ids_array = np.array(all_fold_ids, dtype=np.int32)
    image_ids_array = np.array(all_image_ids, dtype=np.int32)

    # Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{family}_data_FIXED.npz"

    np.savez_compressed(
        output_file,
        images=images_array,
        np_targets=np_targets_array,
        hv_targets=hv_targets_array,
        nt_targets=nt_targets_array,
        fold_ids=fold_ids_array,
        image_ids=image_ids_array,
    )

    print()
    print(f"✅ Données sauvegardées: {output_file}")
    print(f"   Taille: {output_file.stat().st_size / (1024**2):.2f} MB")
    print()

    # Vérification
    print("VÉRIFICATION:")
    print(f"  Images shape:      {images_array.shape}")
    print(f"  NP targets shape:  {np_targets_array.shape}")
    print(f"  HV targets shape:  {hv_targets_array.shape}")
    print(f"  NT targets shape:  {nt_targets_array.shape}")
    print()

    # Stats HV
    print("STATS HV (v7 - radial normalization):")
    print(f"  Dtype:  {hv_targets_array.dtype}")
    print(f"  Min:    {hv_targets_array.min():.4f}")
    print(f"  Max:    {hv_targets_array.max():.4f}")
    print(f"  Mean:   {hv_targets_array.mean():.4f}")
    print(f"  Std:    {hv_targets_array.std():.4f}")
    print()

    print("=" * 80)
    print("PRÉPARATION COMPLÈTE")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare family data (v7 - radial normalization)")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=["glandular", "digestive", "urologic", "respiratory", "epidermal"],
        help="Famille à préparer",
    )
    parser.add_argument(
        "--pannuke_dir",
        type=str,
        default="/home/amar/data/PanNuke",
        help="Répertoire PanNuke source",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_FAMILY_FIXED_DIR,
        help="Répertoire de sortie",
    )

    args = parser.parse_args()

    pannuke_dir = Path(args.pannuke_dir)
    output_dir = Path(args.output_dir)

    if not pannuke_dir.exists():
        print(f"❌ Répertoire PanNuke non trouvé: {pannuke_dir}")
        return 1

    prepare_family_data(args.family, pannuke_dir, output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
