#!/usr/bin/env python3
"""
PREPARE FAMILY DATA - VERSION v12 (FIX CRITIQUE: COHÉRENCE NP/NT)

CHANGEMENT CRITIQUE v11 → v12:
❌ PROBLÈME v11: Incohérence logique NP vs NT
   - NP utilise: mask[:, :, :5].sum(axis=-1) > 0  (union channels 0-4)
   - NT utilise: channel_0 > 0                    (channel 0 SEUL)
   - Pixels dans channels 1-4 mais PAS dans channel 0:
     * NP = 1 (présent dans l'union)
     * NT = 0 (absent de channel 0)
   - MISMATCH 45.35%: Le modèle reçoit des ordres contradictoires!
   - Résultat: Conflit NP/NT persistant malgré training OK (Dice 0.95)

✅ FIX v12: Cohérence PARFAITE NP et NT
   - NT utilise EXACTEMENT la même logique que NP: mask[:, :, :5].sum(axis=-1) > 0
   - TOUS les pixels avec NP=1 auront NT=1
   - Conflit NP/NT = 0.00% GARANTI

Usage:
    python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py --family epidermal
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
# RÉSOLUTION CIBLE H-OPTIMUS-0
# =============================================================================
# CRITIQUE: H-optimus-0 travaille en 224x224.
# Les HV targets doivent être calculés APRÈS resize pour éviter l'interpolation
# qui floute les gradients et "tue" la branche HV.
HOPTIMUS_SIZE = 224


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

    HoVer-Net exige un champ de force CENTRIPÈTE (attraction vers centre).

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

        # Normalisation RADIALE
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
    Normalise le format du mask vers HWC (H, W, 6).
    Supporte 256x256 (PanNuke original) et 224x224 (H-optimus-0).
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D with shape {mask.shape}")

    # Accepter les deux résolutions: 256 (original) ou 224 (H-optimus-0)
    h, w = mask.shape[:2] if mask.shape[0] != 6 else mask.shape[1:]

    if mask.shape == (h, w, 6):
        return mask
    elif mask.shape == (6, h, w):
        mask_hwc = np.transpose(mask, (1, 2, 0))
        mask_hwc = np.ascontiguousarray(mask_hwc)
        return mask_hwc
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}. Expected (H, W, 6) or (6, H, W).")


# =============================================================================
# NUCLEI MASK - SOURCE UNIQUE DE VÉRITÉ (v12)
# =============================================================================

def compute_nuclei_mask_v12(mask: np.ndarray) -> np.ndarray:
    """
    ✅ v12: Calcule le masque binaire des noyaux.

    C'est LA SEULE définition des noyaux, utilisée par NP ET NT.

    Définition: Union des channels 0-4 (exclut channel 5 = tissue)

    Returns:
        nuclei_mask: Boolean mask (H, W) - True = noyau, False = background
    """
    mask = normalize_mask_format(mask)

    # ✅ Union binaire des canaux 0-4 (NOYAUX SEULEMENT)
    # [:5] signifie channels 0, 1, 2, 3, 4 (exclut 5 = tissue)
    nuclei_mask = mask[:, :, :5].sum(axis=-1) > 0

    return nuclei_mask


# =============================================================================
# INSTANCE MAP EXTRACTION (NUCLEI ONLY - EXCLUDES TISSUE)
# =============================================================================

def extract_pannuke_instances_NUCLEI_ONLY(mask: np.ndarray) -> np.ndarray:
    """
    ✅ FIX v9: Extrait UNIQUEMENT les instances de NOYAUX (Channels 0-4).
    ❌ EXCLUT le channel 5 (Epithelial/Tissue).

    Supporte les deux résolutions: 256x256 (PanNuke) et 224x224 (H-optimus-0).
    """
    mask = normalize_mask_format(mask)

    # Utiliser la taille du masque (supporte 256 et 224)
    h, w = mask.shape[:2]
    inst_map = np.zeros((h, w), dtype=np.int32)
    instance_counter = 1

    # PRIORITÉ 1: Channel 0 (multi-type instances) - SOURCE PRIMAIRE
    channel_0 = mask[:, :, 0]
    if channel_0.max() > 0:
        inst_ids_0 = np.unique(channel_0)
        inst_ids_0 = inst_ids_0[inst_ids_0 > 0]

        for inst_id in inst_ids_0:
            inst_mask = channel_0 == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # PRIORITÉ 2: Canaux 1-4 (class-specific instances) - SUPPLÉMENTAIRES
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        if channel_mask.max() > 0:
            inst_ids = np.unique(channel_mask)
            inst_ids = inst_ids[inst_ids > 0]

            for inst_id in inst_ids:
                inst_mask = channel_mask == inst_id
                inst_mask_new = inst_mask & (inst_map == 0)

                if inst_mask_new.sum() > 0:
                    inst_map[inst_mask_new] = instance_counter
                    instance_counter += 1

    # ❌ Channel 5 (Epithelial/Tissue): EXCLU COMPLÈTEMENT

    return inst_map


# =============================================================================
# NP TARGET GENERATION - v12 COHÉRENT
# =============================================================================

def compute_np_target_v12(mask: np.ndarray) -> np.ndarray:
    """
    ✅ v12: Génère le target NP à partir du masque commun.

    Utilise compute_nuclei_mask_v12() pour garantir cohérence avec NT.

    Returns:
        np_target: Binary map (H, W) en float32 [0, 1]
    """
    nuclei_mask = compute_nuclei_mask_v12(mask)
    return nuclei_mask.astype(np.float32)


# =============================================================================
# NT TARGET GENERATION - v12 COHÉRENT (FIX CRITIQUE)
# =============================================================================

def compute_nt_target_v12(mask: np.ndarray) -> np.ndarray:
    """
    ✅ v12 FIX CRITIQUE: NT utilise EXACTEMENT la même logique que NP.

    PROBLÈME v11:
    - NP utilisait: mask[:, :, :5].sum(axis=-1) > 0  (union channels 0-4)
    - NT utilisait: channel_0 > 0                    (channel 0 SEUL)
    - Résultat: 45.35% de conflit (pixels dans 1-4 mais pas 0)

    FIX v12:
    - NT utilise compute_nuclei_mask_v12() = MÊME masque que NP
    - Conflit NP/NT = 0.00% GARANTI

    Classes simplifiées:
    - 0: Background (pas de noyau)
    - 1: Nucleus (noyau, peu importe le type)

    Returns:
        nt_target: Class map (H, W) en int64 [0-1]
    """
    # ✅ v12: Utilise la MÊME logique que NP
    nuclei_mask = compute_nuclei_mask_v12(mask)

    # Initialiser avec classe 0 (background)
    nt_target = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int64)

    # ✅ v12: Force NT=1 pour TOUS les pixels du masque commun
    nt_target[nuclei_mask] = 1

    return nt_target


# =============================================================================
# MAIN PREPARATION FUNCTION
# =============================================================================

def prepare_family_data_v12(
    pannuke_dir: Path,
    output_dir: Path,
    family: str,
    folds: list = None
):
    """
    Prépare les données d'entraînement pour une famille d'organes.

    VERSION v12: COHÉRENCE PARFAITE NP/NT (0% conflit garanti)

    IMPORTANT: Les données sont resizées en 224x224 (résolution H-optimus-0)
    AVANT le calcul des HV targets pour éviter l'interpolation qui floute
    les gradients et "tue" la branche HV.

    Output:
        - images: (N, 224, 224, 3) uint8
        - np_targets: (N, 224, 224) float32 [0, 1]
        - hv_targets: (N, 2, 224, 224) float32 [-1, 1]
        - nt_targets: (N, 224, 224) int64 [0, 1]
    """
    if folds is None:
        folds = [0, 1, 2]

    print("=" * 80)
    print(f"PRÉPARATION FAMILLE: {family.upper()} (VERSION v12 - COHÉRENCE NP/NT)")
    print("=" * 80)
    print(f"\n⚠️  RÉSOLUTION CIBLE: {HOPTIMUS_SIZE}x{HOPTIMUS_SIZE} (H-optimus-0)")
    print("   Les HV targets sont calculés APRÈS resize pour des gradients nets.")

    # Trouver les organes de cette famille
    organs = [org for org, fam in ORGAN_TO_FAMILY.items() if fam == family]
    print(f"\nOrganes: {', '.join(organs)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{family}_data_FIXED_v12_COHERENT.npz"

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

            # Charger image et mask (256x256 original)
            image = np.array(images[i], dtype=np.uint8)
            mask = np.array(masks[i])

            # =========================================================================
            # RESIZE PRÉVENTIF 256→224 (CRITIQUE pour H-optimus-0)
            # =========================================================================
            # On resize AVANT de calculer les HV targets pour éviter l'interpolation
            # qui floute les gradients et "tue" la branche HV.
            # - INTER_AREA pour l'image (meilleur pour réduire)
            # - INTER_NEAREST pour le masque (préserve les IDs d'instances)
            # =========================================================================
            image = cv2.resize(image, (HOPTIMUS_SIZE, HOPTIMUS_SIZE), interpolation=cv2.INTER_AREA)

            # Resize masque canal par canal (préserve les IDs avec INTER_NEAREST)
            mask_224 = np.zeros((HOPTIMUS_SIZE, HOPTIMUS_SIZE, 6), dtype=mask.dtype)
            for c in range(6):
                mask_224[:, :, c] = cv2.resize(
                    mask[:, :, c].astype(np.float32),
                    (HOPTIMUS_SIZE, HOPTIMUS_SIZE),
                    interpolation=cv2.INTER_NEAREST
                ).astype(mask.dtype)
            mask = mask_224

            # ✅ v9: Extraction instances NUCLEI ONLY (exclut tissue)
            inst_map = extract_pannuke_instances_NUCLEI_ONLY(mask)

            # ✅ v12: NP target COHÉRENT (même masque que NT)
            np_target = compute_np_target_v12(mask)

            # HV targets (centripètes v8)
            hv_target = compute_hv_maps(inst_map)

            # ✅ v12: NT target COHÉRENT (même masque que NP)
            nt_target = compute_nt_target_v12(mask)

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

    # Statistiques et VÉRIFICATION CRITIQUE
    print(f"\n📊 Statistiques:")
    print(f"  Images shape:      {images_array.shape}")
    print(f"  NP targets shape:  {np_targets_array.shape}")
    print(f"  HV targets shape:  {hv_targets_array.shape}")
    print(f"  NT targets shape:  {nt_targets_array.shape}")

    # VÉRIFICATION COHÉRENCE NP/NT (DOIT être 0%)
    np_coverage = np_targets_array.mean() * 100
    nt_nuclei_pixels = (nt_targets_array == 1).sum()
    nt_nuclei_pct = nt_nuclei_pixels / nt_targets_array.size * 100

    print(f"\n🔍 VÉRIFICATION COHÉRENCE NP/NT:")
    print(f"  NP coverage:       {np_coverage:.4f}%")
    print(f"  NT nuclei (cl=1):  {nt_nuclei_pct:.4f}%")
    print(f"  Différence NP-NT:  {abs(np_coverage - nt_nuclei_pct):.6f}%")

    # VÉRIFICATION CONFLIT (doit être 0)
    np_positive = np_targets_array > 0
    nt_background = nt_targets_array == 0
    conflict_pixels = (np_positive & nt_background).sum()
    n_np_positive = np_positive.sum()
    conflict_pct = (conflict_pixels / n_np_positive * 100) if n_np_positive > 0 else 0

    print(f"\n🎯 CONFLIT NP/NT (CRITIQUE):")
    print(f"  Pixels NP=1:           {n_np_positive}")
    print(f"  Pixels NP=1 & NT=0:    {conflict_pixels}")
    print(f"  Conflit:               {conflict_pct:.4f}%")

    if conflict_pct < 0.01:
        print(f"\n  ✅ COHÉRENCE PARFAITE NP/NT (conflit < 0.01%)")
    else:
        print(f"\n  ❌ ERREUR: Conflit {conflict_pct:.2f}% détecté!")
        print(f"     Le script v12 devrait produire 0% conflit.")
        print(f"     Vérifiez compute_nuclei_mask_v12().")

    print(f"\n  HV range:          [{hv_targets_array.min():.3f}, {hv_targets_array.max():.3f}]")
    print(f"  NT classes:        {sorted(np.unique(nt_targets_array))}")
    print(f"  Taille fichier:    {output_file.stat().st_size / 1e6:.1f} MB")

    print("\n✅ TERMINÉ")
    print("\n" + "=" * 80)
    print("🎯 OBJECTIF v12:")
    print("=" * 80)
    print("  Cohérence PARFAITE NP/NT (conflit = 0.00%)")
    print("  Classification binaire: nucleus (1) vs background (0)")
    print("  NP et NT utilisent EXACTEMENT le même masque")
    print("  Résultat attendu: AJI > 0.60 après training")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prépare données par famille (VERSION v12 - COHÉRENCE NP/NT)"
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
    print("CHANGEMENTS CRITIQUES v11 → v12:")
    print("=" * 80)
    print("\n❌ PROBLÈME v11 (Incohérence NP/NT):")
    print("  - NP utilise: mask[:, :, :5].sum(axis=-1) > 0  (union 0-4)")
    print("  - NT utilise: channel_0 > 0                    (channel 0 SEUL)")
    print("  - Résultat: 45.35% de conflit!")
    print("    * Pixels dans channels 1-4 mais PAS dans channel 0")
    print("    * NP = 1 (présent dans l'union)")
    print("    * NT = 0 (absent de channel 0)")
    print("\n✅ FIX v12 (Cohérence parfaite):")
    print("  - compute_nuclei_mask_v12(): SOURCE UNIQUE pour NP et NT")
    print("  - NP et NT utilisent EXACTEMENT le même masque")
    print("  - Conflit NP/NT = 0.00% GARANTI")
    print("  - Résultat attendu: AJI > 0.60 après training")
    print("\n" + "=" * 80)

    prepare_family_data_v12(
        args.pannuke_dir,
        args.output_dir,
        args.family,
        args.folds
    )


if __name__ == "__main__":
    main()
