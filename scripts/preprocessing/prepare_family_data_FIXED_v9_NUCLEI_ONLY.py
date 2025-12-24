#!/usr/bin/env python3
"""
PREPARE FAMILY DATA - VERSION v9 (FIX CRITIQUE: NUCLEI ONLY, NO TISSUE)

CHANGEMENT CRITIQUE v8 → v9:
❌ BUG v8: Inclut Channel 5 (Epithelial/Tissue) dans NP targets et inst_map
   - Channel 5: 56,475 pixels (86% de l'image) = TISSUE MASK
   - Channel 0-4: 7,411 pixels (11% de l'image) = NUCLEI
   - Résultat: Modèle apprend à segmenter le TISSU au lieu des NOYAUX
   - AJI catastrophique: 0.03-0.08 au lieu de >0.60

✅ FIX v9: Utilise UNIQUEMENT les canaux de noyaux (0-4), EXCLUT le tissu (5)
   - Channel 0: Instance IDs multi-types (SOURCE PRIMAIRE)
   - Channels 1-4: Instance IDs par classe (supplémentaires)
   - Channel 5: EXCLU (tissue mask, pas des noyaux)
   - NP target: Canaux 0-4 seulement (11% pixels = noyaux)
   - Résultat attendu: AJI 0.08 → >0.60 (+650%)

STRUCTURE PANNUKE CHANNELS:
- Channel 0: Instance IDs multi-types (Neoplastic, Inflammatory, Connective, Dead mélangés)
              Valeurs: [0, 3, 4, 12, 16, 26...68] (IDs séparés, ~11% pixels)
- Channel 1: Neoplastic instance IDs (souvent vide pour epidermal)
- Channel 2: Inflammatory instance IDs (souvent vide pour epidermal)
- Channel 3: Connective instance IDs (souvent vide pour epidermal)
- Channel 4: Dead instance IDs (souvent vide pour epidermal)
- Channel 5: Epithelial BINARY MASK (1 ou 0, ~86% pixels) = TISSUE, PAS NOYAUX

Usage:
    python scripts/preprocessing/prepare_family_data_FIXED_v9_NUCLEI_ONLY.py --family epidermal
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
# HV MAPS COMPUTATION (VERSION v8 - FIX INVERSION 180°)
# =============================================================================

def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """
    Calcule les cartes HV avec FIX INVERSION 180° (v8).

    VERSION v8 - FIX CRITIQUE:
    - ❌ v7: y_dist = y_coords - center_y  (vecteurs centrifuges)
          Erreur angulaire: 179.95° (pointent vers EXTÉRIEUR)

    - ✅ v8: y_dist = center_y - y_coords  (vecteurs centripètes)
          Erreur angulaire: ~0° (pointent vers CENTRE)

    Explication:
    - Pixel à y=110, centroïde à y=100
    - coords - centroid = +10 → vecteur vers BAS (s'éloigne) ❌
    - centroid - coords = -10 → vecteur vers HAUT (vers centre) ✅

    HoVer-Net exige un champ de force CENTRIPÈTE (attraction vers centre).
    v7 générait un champ CENTRIFUGE (répulsion depuis centre).

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

        # ✅ FIX v8: INVERSION 180° - Centroïde MOINS pixel (vers centre)
        y_dist = center_y - y_coords  # Vecteur pointe vers CENTRE (centripète)
        x_dist = center_x - x_coords

        # Normalisation RADIALE (v7 conservée)
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
# INSTANCE MAP EXTRACTION (NUCLEI ONLY - EXCLUDES TISSUE)
# =============================================================================

def extract_pannuke_instances_NUCLEI_ONLY(mask: np.ndarray) -> np.ndarray:
    """
    ✅ FIX v9: Extrait UNIQUEMENT les instances de NOYAUX (Channels 0-4).
    ❌ EXCLUT le channel 5 (Epithelial/Tissue).

    CRITIQUE: Channel 5 est un MASQUE DE TISSU (86% pixels), PAS des noyaux!
    L'inclure fait apprendre au modèle à segmenter le tissu au lieu des noyaux.

    Priorité des sources:
    1. Channel 0: Instance IDs multi-types (SOURCE PRIMAIRE)
       - Contient tous les noyaux avec IDs séparés (ex: [3, 4, 12, 16, 26...68])
       - Couvre ~11% de l'image (taille normale pour noyaux)

    2. Channels 1-4: Instance IDs par classe (SUPPLÉMENTAIRES si non vide)
       - Neoplastic (1), Inflammatory (2), Connective (3), Dead (4)
       - Souvent vides pour certaines familles (ex: epidermal)

    3. Channel 5: EXCLU ❌
       - Masque binaire de tissu épithélial (~86% pixels)
       - PAS des noyaux individuels, juste la région tissulaire

    Args:
        mask: Mask PanNuke (H, W, 6)

    Returns:
        inst_map: Instance map (H, W) avec IDs séparés (0 = background)
                  Contient UNIQUEMENT les noyaux, PAS le tissu
    """
    mask = normalize_mask_format(mask)

    inst_map = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int32)
    instance_counter = 1

    # ✅ PRIORITÉ 1: Channel 0 (multi-type instances) - SOURCE PRIMAIRE
    channel_0 = mask[:, :, 0]
    if channel_0.max() > 0:
        inst_ids_0 = np.unique(channel_0)
        inst_ids_0 = inst_ids_0[inst_ids_0 > 0]

        for inst_id in inst_ids_0:
            inst_mask = channel_0 == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # ✅ PRIORITÉ 2: Canaux 1-4 (class-specific instances) - SUPPLÉMENTAIRES
    # Note: On ajoute seulement les pixels qui n'ont PAS déjà été assignés par Channel 0
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        if channel_mask.max() > 0:
            inst_ids = np.unique(channel_mask)
            inst_ids = inst_ids[inst_ids > 0]

            for inst_id in inst_ids:
                # Masque pour cette instance
                inst_mask = channel_mask == inst_id

                # Ne garder que les pixels qui ne sont pas déjà dans inst_map
                # (évite duplication si Channel 0 et Channel 1-4 se chevauchent)
                inst_mask_new = inst_mask & (inst_map == 0)

                if inst_mask_new.sum() > 0:
                    inst_map[inst_mask_new] = instance_counter
                    instance_counter += 1

    # ❌ Channel 5 (Epithelial/Tissue): EXCLU COMPLÈTEMENT
    # Ce canal est un masque de TISSU (~86% pixels), pas des noyaux individuels
    # L'inclure ferait apprendre au modèle à segmenter le tissu entier au lieu
    # de séparer les noyaux individuels → AJI catastrophique

    return inst_map


# =============================================================================
# NP TARGET GENERATION (NUCLEI ONLY - EXCLUDES TISSUE)
# =============================================================================

def compute_np_target_NUCLEI_ONLY(mask: np.ndarray) -> np.ndarray:
    """
    ✅ FIX v9: Génère le target NP UNIQUEMENT pour les NOYAUX (Channels 0-4).
    ❌ EXCLUT le channel 5 (Epithelial/Tissue).

    AVANT (BUG v8):
        np_target = mask[:, :, 1:].sum(axis=-1) > 0
        → Inclut Channels 1, 2, 3, 4, ET 5 (tissue!)
        → 56,475 pixels (86% de l'image) = TISSU
        → Modèle apprend à segmenter le tissu, pas les noyaux

    APRÈS (FIX v9):
        np_target = mask[:, :, :5].sum(axis=-1) > 0
        → Inclut UNIQUEMENT Channels 0, 1, 2, 3, 4 (noyaux)
        → 7,411 pixels (11% de l'image) = NOYAUX
        → Modèle apprend à segmenter les noyaux correctement

    Args:
        mask: Mask PanNuke (H, W, 6)

    Returns:
        np_target: Binary map (H, W) en float32 [0, 1]
                   Valeur 1 = présence de noyau (channels 0-4)
                   Valeur 0 = background
    """
    mask = normalize_mask_format(mask)

    # ✅ Union binaire des canaux 0-4 (NOYAUX SEULEMENT)
    # [:5] signifie channels 0, 1, 2, 3, 4 (exclut 5)
    np_target = mask[:, :, :5].sum(axis=-1) > 0

    return np_target.astype(np.float32)


# =============================================================================
# NT TARGET GENERATION
# =============================================================================

def compute_nt_target(mask: np.ndarray) -> np.ndarray:
    """
    Génère le target NT (Nuclear Type) - NUCLEI ONLY (exclut Channel 5).

    ✅ v9 FIX: Exclut Channel 5 (Epithelial) pour cohérence avec NP/HV.

    Classes PanNuke utilisées:
    - 0: Background
    - 1: Neoplastic
    - 2: Inflammatory
    - 3: Connective
    - 4: Dead

    Channel 5 (Epithelial) EXCLU:
    - Pour epidermal, les noyaux épithéliaux seront classés comme background (classe 0)
    - Cohérent avec l'exclusion de Channel 5 pour NP/HV
    - Modèle HoVer-Net a n_classes=5 (0-4), pas 6

    Args:
        mask: Mask PanNuke (H, W, 6)

    Returns:
        nt_target: Class map (H, W) en int64 [0-4]
    """
    mask = normalize_mask_format(mask)

    # Initialiser avec classe 0 (background)
    nt_target = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int64)

    # ✅ Pour chaque canal 1-4 SEULEMENT (exclut 5)
    for class_id in range(1, 5):  # 1, 2, 3, 4 (PAS 5)
        channel_mask = mask[:, :, class_id] > 0
        nt_target[channel_mask] = class_id

    return nt_target


# =============================================================================
# MAIN PREPARATION FUNCTION
# =============================================================================

def prepare_family_data_v9(
    pannuke_dir: Path,
    output_dir: Path,
    family: str,
    folds: list = None
):
    """
    Prépare les données d'entraînement pour une famille d'organes.

    VERSION v9: NUCLEI ONLY (exclut Channel 5 tissue)

    Args:
        pannuke_dir: Répertoire PanNuke (/home/amar/data/PanNuke)
        output_dir: Répertoire de sortie (data/family_FIXED)
        family: Famille d'organes (ex: "epidermal")
        folds: Liste des folds à traiter (défaut: [0, 1, 2])
    """
    if folds is None:
        folds = [0, 1, 2]

    print("=" * 80)
    print(f"PRÉPARATION FAMILLE: {family.upper()} (VERSION v9 - NUCLEI ONLY)")
    print("=" * 80)

    # Trouver les organes de cette famille
    organs = [org for org, fam in ORGAN_TO_FAMILY.items() if fam == family]
    print(f"\nOrganes: {', '.join(organs)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{family}_data_FIXED_v9_NUCLEI_ONLY.npz"

    # Collecter les échantillons
    all_images = []
    all_np_targets = []
    all_hv_targets = []
    all_nt_targets = []
    all_fold_ids = []
    all_image_ids = []

    total_samples = 0

    for fold in folds:
        fold_dir = pannuke_dir / f"fold{fold}"
        images_path = fold_dir / "images.npy"
        masks_path = fold_dir / "masks.npy"
        types_path = fold_dir / "types.npy"

        if not images_path.exists():
            print(f"\n⚠️  Fold {fold}: fichiers manquants, skip")
            continue

        # Charger avec mmap
        images = np.load(images_path, mmap_mode='r')
        masks = np.load(masks_path, mmap_mode='r')
        types = np.load(types_path)

        # Filtrer par famille
        fold_samples = 0
        for i in range(len(images)):
            organ_name = types[i].decode('utf-8') if isinstance(types[i], bytes) else types[i]
            if organ_name not in organs:
                continue

            # Charger image et mask
            image = np.array(images[i], dtype=np.uint8)
            mask = np.array(masks[i])

            # ✅ v9: Extraction instances NUCLEI ONLY (exclut tissue)
            inst_map = extract_pannuke_instances_NUCLEI_ONLY(mask)

            # ✅ v9: NP target NUCLEI ONLY (exclut tissue)
            np_target = compute_np_target_NUCLEI_ONLY(mask)

            # HV targets (centripètes v8)
            hv_target = compute_hv_maps(inst_map)

            # NT target
            nt_target = compute_nt_target(mask)

            # Stocker
            all_images.append(image)
            all_np_targets.append(np_target)
            all_hv_targets.append(hv_target)
            all_nt_targets.append(nt_target)
            all_fold_ids.append(fold)
            all_image_ids.append(i)

            fold_samples += 1

        total_samples += fold_samples
        print(f"  Fold {fold}: {fold_samples} samples")

    if total_samples == 0:
        print(f"\n⚠️  Aucun échantillon trouvé pour famille {family}")
        return

    print(f"\n  Total: {total_samples} samples")

    # Convertir en arrays
    print("\n💾 Conversion en arrays...")
    images_array = np.stack(all_images, axis=0)
    np_targets_array = np.stack(all_np_targets, axis=0)
    hv_targets_array = np.stack(all_hv_targets, axis=0)
    nt_targets_array = np.stack(all_nt_targets, axis=0)
    fold_ids_array = np.array(all_fold_ids, dtype=np.int32)
    image_ids_array = np.array(all_image_ids, dtype=np.int32)

    # Sauvegarder
    print(f"\n💾 Sauvegarde: {output_file}")
    np.savez_compressed(
        output_file,
        images=images_array,
        np_targets=np_targets_array,
        hv_targets=hv_targets_array,
        nt_targets=nt_targets_array,
        fold_ids=fold_ids_array,
        image_ids=image_ids_array,
    )

    # Statistiques
    print(f"\n📊 Statistiques:")
    print(f"  Images shape:      {images_array.shape}")
    print(f"  NP targets shape:  {np_targets_array.shape}")
    print(f"  HV targets shape:  {hv_targets_array.shape}")
    print(f"  NT targets shape:  {nt_targets_array.shape}")
    print(f"  NP coverage:       {np_targets_array.mean() * 100:.2f}%")
    print(f"  HV range:          [{hv_targets_array.min():.3f}, {hv_targets_array.max():.3f}]")
    print(f"  Taille fichier:    {output_file.stat().st_size / 1e6:.1f} MB")

    print("\n✅ TERMINÉ")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prépare données par famille (VERSION v9 - NUCLEI ONLY, EXCLUT TISSUE)"
    )
    parser.add_argument(
        "--pannuke_dir",
        type=Path,
        default=Path("/home/amar/data/PanNuke"),
        help="Répertoire PanNuke"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/family_FIXED"),
        help="Répertoire de sortie"
    )
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=["glandular", "digestive", "urologic", "respiratory", "epidermal"],
        help="Famille d'organes"
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help="Folds à traiter (défaut: 0 1 2)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CHANGEMENTS CRITIQUES v8 → v9:")
    print("=" * 80)
    print("\n❌ BUG v8:")
    print("  - np_target = mask[:, :, 1:].sum() → Inclut Channel 5 (tissue)")
    print("  - Channel 5: 56,475 pixels (86% image) = TISSU ENTIER")
    print("  - Résultat: Modèle apprend à segmenter tissu, pas noyaux")
    print("  - AJI: 0.03-0.08 (catastrophique)")
    print("\n✅ FIX v9:")
    print("  - np_target = mask[:, :, :5].sum() → Channels 0-4 UNIQUEMENT")
    print("  - Channels 0-4: 7,411 pixels (11% image) = NOYAUX")
    print("  - Résultat attendu: Modèle apprend à segmenter noyaux")
    print("  - AJI attendu: >0.60 (gain +650%)")
    print("\n" + "=" * 80)

    prepare_family_data_v9(
        args.pannuke_dir,
        args.output_dir,
        args.family,
        args.folds
    )


if __name__ == "__main__":
    main()
