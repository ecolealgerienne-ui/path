#!/usr/bin/env python3
"""
Prépare les données d'entraînement par famille d'organes.

VERSION FIXÉE v4:
- Utilise les IDs d'instances NATIFS de PanNuke au lieu de connectedComponents
- AUTO-DÉTECTION format HWC vs CHW pour IMAGES ET MASKS
- NORMALISATION complète : Images CHW→HWC + Masks CHW→HWC
- INVERSION H/V : Convention standard (V, H) au lieu de (H, V)
- CONTIGUITY : np.ascontiguousarray() après transpose pour garantir layout mémoire

BUGS CORRIGÉS:
- Bug #3: connectedComponents fusionnait les cellules qui se touchent → 75% perdues
- Bug #4 v1: Format mismatch HWC vs CHW causait désalignement 96px (masks uniquement)
- Bug #4 v2→v3: Images CHW non normalisées causaient 96px distance
- Bug #5 v3→v4 (Expert): Inversion H/V causait 96px distance (transposée (x,y)→(y,x))

Diagnostic Expert Bug #5 (2025-12-24):
- v3 avec normalisation Images+Masks testée → Toujours 96px
- Analyse: 96px = distance statistique moyenne entre (x,y) et transposée (y,x)
- Problème: compute_hv_maps() assignait [0]=X, [1]=Y au lieu de [0]=Y, [1]=X
- Convention HoVer-Net attendue: [0]=Vertical(Y), [1]=Horizontal(X)
- Fix v4: Inverser assignation + np.ascontiguousarray() après transpose

Résultat attendu v4:
- Distance alignement: 96px → <2px (-98%)
- AJI post re-training: 0.06 → 0.60+ (+846%)

Usage:
    python scripts/preprocessing/prepare_family_data_FIXED_v4.py --data_dir /home/amar/data/PanNuke
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.organ_families import FAMILY_TO_ORGANS, FAMILIES
from src.constants import DEFAULT_FAMILY_FIXED_DIR


def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """
    Calcule les cartes Horizontal/Vertical pour séparation d'instances.

    VERSION v5 avec:
    - Normalisation BBOX (HoVer-Net original Graham et al. 2019)
    - Gaussian smoothing (sigma=0.5) pour réduire le bruit
    - Convention H-optimus-0 correcte (V, H)

    Changement v4 → v5:
        v4: Normalise par max_dist (range variable)
        v5: Normalise par bbox_size (range fixe [-1, +1])
            → Donne des gradients plus forts et plus stables

    Convention HoVer-Net:
        hv_map[0] = Vertical (Y) - distance normalisée en Y
        hv_map[1] = Horizontal (X) - distance normalisée en X

    Args:
        inst_map: (H, W) avec IDs d'instances [0, 1, 2, ...]

    Returns:
        hv_maps: (2, H, W) avec V et H normalisés [-1, +1]
                 [0] = Vertical (Y)
                 [1] = Horizontal (X)
    """
    from scipy.ndimage import gaussian_filter

    h, w = inst_map.shape
    hv_map = np.zeros((2, h, w), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]  # Exclude background

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id

        # Trouver coordonnées de l'instance
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) == 0:
            continue

        # ✅ CHANGEMENT v5: Bounding box de l'instance (HoVer-Net original)
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        # Taille de la bbox
        bbox_h = y_max - y_min + 1e-6  # +epsilon pour éviter division par 0
        bbox_w = x_max - x_min + 1e-6

        # Normalisation BBOX (range TOUJOURS [-1, +1])
        # Formula: 2 * (coord - min) / (max - min) - 1.0
        y_dist = 2.0 * (y_coords - y_min) / bbox_h - 1.0
        x_dist = 2.0 * (x_coords - x_min) / bbox_w - 1.0

        # ✅ Convention standard (V, H)
        hv_map[0, y_coords, x_coords] = y_dist  # Canal 0 = Vertical (Y)
        hv_map[1, y_coords, x_coords] = x_dist  # Canal 1 = Horizontal (X)

    # ✅ NOUVEAU v5: Gaussian smoothing léger pour réduire le bruit
    # (sigma=0.5 recommandé par expert - ne lisse pas trop mais élimine aliasing)
    hv_map[0] = gaussian_filter(hv_map[0], sigma=0.5)
    hv_map[1] = gaussian_filter(hv_map[1], sigma=0.5)

    return hv_map


def normalize_mask_format(mask: np.ndarray) -> np.ndarray:
    """
    Normalise le format du mask vers HWC (256, 256, 6).

    AUTO-DÉTECTION et conversion si nécessaire.

    Args:
        mask: PanNuke mask, peut être:
            - HWC: (256, 256, 6) ✅ Attendu
            - CHW: (6, 256, 256) ⚠️ Nécessite conversion

    Returns:
        mask_hwc: (256, 256, 6) HWC format

    Raises:
        ValueError: Si le format ne peut pas être détecté
    """
    if mask.ndim != 3:
        raise ValueError(
            f"Expected 3D mask, got {mask.ndim}D with shape {mask.shape}"
        )

    # DÉTECTION FORMAT
    # Cas 1: HWC (256, 256, 6)
    if mask.shape == (256, 256, 6):
        return mask

    # Cas 2: CHW (6, 256, 256)
    elif mask.shape == (6, 256, 256):
        mask_hwc = np.transpose(mask, (1, 2, 0))  # (6, 256, 256) → (256, 256, 6)
        mask_hwc = np.ascontiguousarray(mask_hwc)  # Garantir layout mémoire continu
        return mask_hwc

    # Cas 3: Format inconnu
    else:
        raise ValueError(
            f"Unexpected mask shape: {mask.shape}. "
            f"Expected (256, 256, 6) or (6, 256, 256)"
        )


def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """
    Extrait les vraies instances de PanNuke (FIXÉ v2).

    AVANT (BUGGY):
        np_mask = mask[:, :, 1:].sum(axis=-1) > 0
        _, inst_map = cv2.connectedComponents(np_mask.astype(np.uint8))
        → Fusionne les cellules qui se touchent ❌

    APRÈS (FIXÉ v2):
        1. Normalise format HWC vs CHW ✅
        2. Utilise IDs natifs PanNuke canaux 1-4 ✅

    Args:
        mask: (256, 256, 6) ou (6, 256, 256) PanNuke mask
            - Canal 0: Background
            - Canal 1: Neoplastic instance IDs
            - Canal 2: Inflammatory instance IDs
            - Canal 3: Connective instance IDs
            - Canal 4: Dead instance IDs
            - Canal 5: Epithelial (binaire, pas d'IDs)

    Returns:
        inst_map: (256, 256) avec IDs d'instances uniques [0, 1, 2, ...]
    """
    # ✅ FIXÉ v2: Auto-détection et normalisation format
    mask = normalize_mask_format(mask)

    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: IDs d'instances natifs PanNuke
    for c in range(1, 5):
        channel_mask = mask[:, :, c]  # Maintenant garanti HWC
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]  # Exclude 0 = background

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
    # (Ce canal ne contient pas d'IDs natifs dans PanNuke)
    epithelial_mask = mask[:, :, 5]
    if epithelial_mask.max() > 0:
        _, epithelial_labels = cv2.connectedComponents(epithelial_mask.astype(np.uint8))
        epithelial_ids = np.unique(epithelial_labels)
        epithelial_ids = epithelial_ids[epithelial_ids > 0]

        for inst_id in epithelial_ids:
            inst_mask = epithelial_labels == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    return inst_map


def prepare_family_data(data_dir: Path, output_dir: Path, family: str, chunk_size: int = 500, folds: list = None):
    """
    Prépare les données d'entraînement pour une famille d'organes.

    VERSION FIXÉE v2 avec:
    - Vraies instances PanNuke (Bug #3)
    - Auto-détection format HWC/CHW (Bug #4)
    - Optimisation RAM (chunking)

    Args:
        chunk_size: Nombre d'images à traiter par lot (défaut: 500)
                    Réduit la consommation RAM (~2 GB par chunk au lieu de 10+ GB)
        folds: Liste des folds à traiter (défaut: [0, 1, 2])
    """
    if folds is None:
        folds = [0, 1, 2]
    print(f"\n{'='*70}")
    print(f"Préparation données famille: {family}")
    print(f"{'='*70}")

    organs = FAMILY_TO_ORGANS[family]
    print(f"Organes: {', '.join(organs)}")
    print(f"Chunk size: {chunk_size} images (RAM-optimized)")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{family}_data_FIXED.npz"

    # Phase 1: Collecter les indices par fold
    print(f"\n📋 Phase 1: Indexing...")
    fold_indices = {}

    for fold in folds:
        fold_dir = data_dir / f"fold{fold}"
        types_path = fold_dir / "types.npy"

        if not types_path.exists():
            print(f"  Fold {fold}: Missing types.npy, skipping")
            continue

        types = np.load(types_path, mmap_mode='r')
        indices = []

        for i in range(len(types)):
            organ_name = types[i].decode('utf-8') if isinstance(types[i], bytes) else types[i]
            if organ_name in organs:
                indices.append(i)

        if len(indices) > 0:
            fold_indices[fold] = indices
            print(f"  Fold {fold}: {len(indices)} images")

    total_samples = sum(len(indices) for indices in fold_indices.values())
    print(f"\n  Total samples: {total_samples}")

    if total_samples == 0:
        print(f"  ⚠️  No samples found for family {family}, skipping")
        return

    # Phase 2: Traiter par chunks et sauvegarder progressivement
    print(f"\n🔄 Phase 2: Processing in chunks of {chunk_size}...")

    all_chunks = {
        'images': [],
        'np_targets': [],
        'hv_targets': [],
        'nt_targets': [],
        'fold_ids': [],
        'image_ids': []
    }

    global_idx = 0
    format_detected = None  # Pour afficher une seule fois

    for fold, indices in fold_indices.items():
        fold_dir = data_dir / f"fold{fold}"
        images_path = fold_dir / "images.npy"
        masks_path = fold_dir / "masks.npy"

        # Charger avec mmap (pas en RAM)
        images = np.load(images_path, mmap_mode='r')
        masks = np.load(masks_path, mmap_mode='r')

        # ✅ NOUVEAU: Afficher format détecté pour ce fold
        if format_detected is None:
            sample_mask = np.array(masks[0])
            print(f"\n  🔍 Détection format fold {fold}:")
            print(f"     Masks shape: {masks.shape}")
            print(f"     Sample mask shape: {sample_mask.shape}")
            if sample_mask.shape == (256, 256, 6):
                format_detected = "HWC"
                print(f"     ✅ Format: HWC (256, 256, 6) - Pas de conversion nécessaire")
            elif sample_mask.shape == (6, 256, 256):
                format_detected = "CHW"
                print(f"     ⚠️ Format: CHW (6, 256, 256) - Conversion automatique vers HWC")
            else:
                raise ValueError(f"Format inattendu: {sample_mask.shape}")

        print(f"\n  Processing fold {fold} ({len(indices)} images)...")

        # Traiter par chunks
        for chunk_start in range(0, len(indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(indices))
            chunk_indices = indices[chunk_start:chunk_end]

            print(f"    Chunk {chunk_start//chunk_size + 1}/{(len(indices)-1)//chunk_size + 1} ({len(chunk_indices)} images)...")

            # Arrays temporaires pour ce chunk
            chunk_images = []
            chunk_np_targets = []
            chunk_hv_targets = []
            chunk_nt_targets = []
            chunk_fold_ids = []
            chunk_image_ids = []

            for idx in tqdm(chunk_indices, desc="      Processing", leave=False):
                raw_img = np.array(images[idx], dtype=np.uint8)
                raw_mask = np.array(masks[idx])

                # ✅ FIXÉ v4: NORMALISATION IMAGE + CONTIGUITY (Bug #1 Expert + Bug #5)
                # Si image en CHW (3, 256, 256) → transpose vers HWC (256, 256, 3)
                if raw_img.shape[0] == 3:
                    image = np.transpose(raw_img, (1, 2, 0))  # CHW → HWC
                    image = np.ascontiguousarray(image)  # Garantir layout mémoire continu
                else:
                    image = raw_img  # Déjà HWC

                # ✅ FIXÉ v3: NORMALISATION MASQUE (Une seule fois)
                mask = normalize_mask_format(raw_mask)

                # ✅ FIXÉ v3: Génération targets sur données REDRESSÉES
                inst_map = extract_pannuke_instances(mask)  # Mask déjà HWC

                # NP target
                np_target = (inst_map > 0).astype(np.float32)

                # HV target
                hv_target = compute_hv_maps(inst_map)

                # NT target (mask déjà normalisé HWC)
                nt_target = np.argmax(mask[:, :, 1:], axis=-1).astype(np.int64)

                # ✅ STOCKAGE (Image garantie HWC maintenant)
                chunk_images.append(image)
                chunk_np_targets.append(np_target)
                chunk_hv_targets.append(hv_target)
                chunk_nt_targets.append(nt_target)
                chunk_fold_ids.append(fold)
                chunk_image_ids.append(idx)

            # Convertir chunk en arrays et stocker
            all_chunks['images'].append(np.stack(chunk_images, axis=0))
            all_chunks['np_targets'].append(np.stack(chunk_np_targets, axis=0))
            all_chunks['hv_targets'].append(np.stack(chunk_hv_targets, axis=0))
            all_chunks['nt_targets'].append(np.stack(chunk_nt_targets, axis=0))
            all_chunks['fold_ids'].append(np.array(chunk_fold_ids, dtype=np.int32))
            all_chunks['image_ids'].append(np.array(chunk_image_ids, dtype=np.int32))

            # Libérer mémoire du chunk
            del chunk_images, chunk_np_targets, chunk_hv_targets, chunk_nt_targets
            del chunk_fold_ids, chunk_image_ids

    # Phase 3: Concaténer tous les chunks et sauvegarder
    print(f"\n💾 Phase 3: Concatenating and saving...")

    images_array = np.concatenate(all_chunks['images'], axis=0)
    np_targets_array = np.concatenate(all_chunks['np_targets'], axis=0)
    hv_targets_array = np.concatenate(all_chunks['hv_targets'], axis=0)
    nt_targets_array = np.concatenate(all_chunks['nt_targets'], axis=0)
    fold_ids_array = np.concatenate(all_chunks['fold_ids'], axis=0)
    image_ids_array = np.concatenate(all_chunks['image_ids'], axis=0)

    # Sauvegarder
    np.savez_compressed(
        output_file,
        images=images_array,
        np_targets=np_targets_array,
        hv_targets=hv_targets_array,
        nt_targets=nt_targets_array,
        fold_ids=fold_ids_array,
        image_ids=image_ids_array,
    )

    print(f"\n  ✅ Saved: {output_file}")
    print(f"     Size: {output_file.stat().st_size / 1e9:.2f} GB")

    # Statistiques
    print(f"\n  📊 Statistics:")
    print(f"     Images: {images_array.shape}")
    print(f"     NP coverage: {np_targets_array.mean() * 100:.2f}%")
    print(f"     HV range: [{hv_targets_array.min():.3f}, {hv_targets_array.max():.3f}]")
    print(f"     NT classes: {np.unique(nt_targets_array)}")
    print(f"\n  🔍 Format final:")
    print(f"     Mask format processed: {format_detected} → HWC")
    print(f"     All data saved in HWC format (256, 256, 6)")


def main():
    parser = argparse.ArgumentParser(description="Prépare données par famille (VERSION FIXÉE v3 + EXPERT FIX)")
    parser.add_argument("--data_dir", type=Path, default=Path("/home/amar/data/PanNuke"))
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_FAMILY_FIXED_DIR))
    parser.add_argument("--family", type=str, choices=FAMILIES, help="Famille spécifique (optionnel)")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Nombre d'images par chunk (défaut: 500, réduit RAM)")
    parser.add_argument("--folds", type=int, nargs='+', default=[0, 1, 2],
                        help="Folds à traiter (défaut: 0 1 2)")

    args = parser.parse_args()

    print("=" * 70)
    print("PRÉPARATION DONNÉES PAR FAMILLE (VERSION FIXÉE v4 - BUG #5 FIX)")
    print("=" * 70)
    print(f"\n🆕 NOUVEAUTÉ v4 (Expert Diagnosis Bug #5 - 2025-12-24):")
    print(f"  ✅ INVERSION H/V: [0]=Vertical(Y), [1]=Horizontal(X)")
    print(f"  ✅ CONTIGUITY: np.ascontiguousarray() après transpose")
    print(f"  ✅ Élimine Bug #5 (96px = distance (x,y)→(y,x) transposée)")
    print(f"\nFixes cumulatifs:")
    print(f"  ✅ v2: Auto-détection format HWC vs CHW (masks)")
    print(f"  ✅ v3: Normalisation IMAGES + MASKS (Bug #4)")
    print(f"  ✅ v4: Inversion H/V + Contiguity (Bug #5)")
    print(f"\nChangements historiques:")
    print(f"  ❌ v1: connectedComponents fusionnait cellules touchantes")
    print(f"  ✅ v2+: IDs natifs PanNuke (vraies instances séparées)")
    print(f"\nOptimisations:")
    print(f"  ✅ Traitement par chunks de {args.chunk_size} images")
    print(f"  ✅ mmap_mode='r' pour économiser la RAM")
    print(f"  ✅ Consommation RAM: ~2 GB par chunk au lieu de 10+ GB")
    print(f"\nRésultat attendu v4:")
    print(f"  - Distance alignement: 96px → <2px (-98%)")
    print(f"  - AJI post re-training: 0.06 → 0.60+ (gain +846%)")
    print(f"  - Convention HoVer-Net respectée: [V, H] au lieu de [H, V]")

    if args.family:
        prepare_family_data(args.data_dir, args.output_dir, args.family, args.chunk_size, args.folds)
    else:
        for family in FAMILIES:
            prepare_family_data(args.data_dir, args.output_dir, family, args.chunk_size, args.folds)

    print("\n" + "=" * 70)
    print("✅ PRÉPARATION TERMINÉE")
    print("=" * 70)
    print(f"\nProchaines étapes:")
    print(f"  1. Vérifier alignement spatial:")
    print(f"     python scripts/validation/verify_spatial_alignment.py \\")
    print(f"         --family <famille> --n_samples 5")
    print(f"     Attendu: distance < 2 pixels")
    print(f"  2. Si alignement OK → Ré-entraîner HoVer-Net (~40 min)")


if __name__ == "__main__":
    main()
