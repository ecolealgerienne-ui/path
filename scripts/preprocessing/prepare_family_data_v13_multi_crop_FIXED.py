#!/usr/bin/env python3
"""
Préparation données V13 - Multi-Crop Statique (5 crops par image).

⚠️ VERSION CORRIGÉE: Charge depuis PanNuke originales 256×256

Génère 5 crops fixes (224×224) depuis chaque patch source (256×256):
- Center: (16, 16) → (240, 240)
- Top-Left: (0, 0) → (224, 224)
- Top-Right: (32, 0) → (256, 224)
- Bottom-Left: (0, 32) → (224, 256)
- Bottom-Right: (32, 32) → (256, 256)

Filtre les crops avec GT vide (aucune instance détectée).

Usage:
    python scripts/preprocessing/prepare_family_data_v13_multi_crop_FIXED.py \
        --family epidermal \
        --pannuke_dir /home/amar/data/PanNuke \
        --output_dir data/family_V13
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE
from src.models.organ_families import ORGAN_TO_FAMILY

# Positions de crop fixes (5 crops par image 256×256)
CROP_POSITIONS = {
    'center':       (16, 16, 240, 240),  # (x_start, y_start, x_end, y_end)
    'top_left':     (0,  0,  224, 224),
    'top_right':    (32, 0,  256, 224),
    'bottom_left':  (0,  32, 224, 256),
    'bottom_right': (32, 32, 256, 256),
}

CROP_SIZE = 224


def normalize_mask_format(mask: np.ndarray) -> np.ndarray:
    """Normalise format masque PanNuke (H, W, 6)."""
    if mask.ndim == 2:
        # Masque binaire → Ajouter dimension
        mask = np.expand_dims(mask, axis=-1)
    return mask


def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """
    Extrait instance map depuis masque PanNuke.

    Args:
        mask: Masque PanNuke (H, W, 6)

    Returns:
        inst_map: Instance map (H, W) int32
    """
    mask = normalize_mask_format(mask)

    h, w = mask.shape[:2]
    inst_map = np.zeros((h, w), dtype=np.int32)
    instance_counter = 1

    # Channel 0: Multi-type instances (source primaire)
    channel_0 = mask[:, :, 0]
    if channel_0.max() > 0:
        inst_ids_0 = np.unique(channel_0)
        inst_ids_0 = inst_ids_0[inst_ids_0 > 0]

        for inst_id in inst_ids_0:
            inst_mask = channel_0 == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canaux 1-4: Class-specific instances
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

    return inst_map


def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """
    Calcule cartes HV (Horizontal/Vertical) centripètes.

    Args:
        inst_map: Instance map (H, W) int32

    Returns:
        hv_map: (2, H, W) float32 [-1, 1]
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

        # Centroïde
        cy = y_coords.mean()
        cx = x_coords.mean()

        # Max distances pour normalisation
        max_dist_y = max(abs(y_coords - cy).max(), 1e-6)
        max_dist_x = max(abs(x_coords - cx).max(), 1e-6)

        # Vecteurs centripètes normalisés [-1, 1]
        v_map = (y_coords - cy) / max_dist_y
        h_map = (x_coords - cx) / max_dist_x

        hv_map[0, y_coords, x_coords] = v_map  # Vertical
        hv_map[1, y_coords, x_coords] = h_map  # Horizontal

    return hv_map


def compute_np_target(mask: np.ndarray) -> np.ndarray:
    """Génère target NP (Nuclear Presence) binaire."""
    nuclei_mask = mask[:, :, :5].sum(axis=-1) > 0
    return nuclei_mask.astype(np.float32)


def compute_nt_target(mask: np.ndarray) -> np.ndarray:
    """Génère target NT (Nuclear Type) binaire simplifié."""
    nuclei_mask = mask[:, :, :5].sum(axis=-1) > 0
    nt_target = np.zeros(mask.shape[:2], dtype=np.int64)
    nt_target[nuclei_mask] = 1  # Classe 1: nucleus, Classe 0: background
    return nt_target


def extract_crop(
    image: np.ndarray,
    np_target: np.ndarray,
    hv_target: np.ndarray,
    nt_target: np.ndarray,
    x1: int, y1: int, x2: int, y2: int
) -> Dict[str, np.ndarray]:
    """Extrait un crop 224×224 et ses targets."""
    crop_image = image[y1:y2, x1:x2]
    crop_np = np_target[y1:y2, x1:x2]
    crop_hv = hv_target[:, y1:y2, x1:x2]
    crop_nt = nt_target[y1:y2, x1:x2]

    # Validation
    assert crop_image.shape == (CROP_SIZE, CROP_SIZE, 3), f"Image shape: {crop_image.shape}"
    assert crop_np.shape == (CROP_SIZE, CROP_SIZE), f"NP shape: {crop_np.shape}"
    assert crop_hv.shape == (2, CROP_SIZE, CROP_SIZE), f"HV shape: {crop_hv.shape}"
    assert crop_nt.shape == (CROP_SIZE, CROP_SIZE), f"NT shape: {crop_nt.shape}"

    return {
        'image': crop_image,
        'np_target': crop_np,
        'hv_target': crop_hv,
        'nt_target': crop_nt,
    }


def is_valid_crop(np_target: np.ndarray, nt_target: np.ndarray) -> Tuple[bool, int]:
    """Vérifie si un crop contient au moins 1 instance."""
    binary_mask = (np_target > 0.5).astype(np.uint8)
    inst_map, num_instances = ndimage.label(binary_mask)
    unique_labels = np.unique(inst_map)
    is_valid = len(unique_labels) > 1  # Plus que background
    return is_valid, num_instances


def generate_multi_crops_from_pannuke(
    pannuke_dir: Path,
    output_dir: Path,
    family: str,
    folds: list = None
) -> Dict[str, int]:
    """
    Génère 5 crops fixes depuis données PanNuke originales 256×256.

    Args:
        pannuke_dir: Répertoire PanNuke (/home/amar/data/PanNuke)
        output_dir: Répertoire de sortie
        family: Famille tissulaire
        folds: Liste des folds (défaut: [0, 1, 2])

    Returns:
        Statistiques de génération
    """
    if folds is None:
        folds = [0, 1, 2]

    print(f"\n{'='*70}")
    print(f"GÉNÉRATION MULTI-CROPS V13 - Famille: {family.upper()}")
    print(f"{'='*70}\n")

    # Organes de cette famille
    organs = [org for org, fam in ORGAN_TO_FAMILY.items() if fam == family]
    print(f"Organes: {', '.join(organs)}\n")

    # Accumulateurs
    crops_data = {
        'images': [],
        'np_targets': [],
        'hv_targets': [],
        'nt_targets': [],
        'source_image_ids': [],
        'crop_positions': [],
        'fold_ids': [],
    }

    stats = {
        'total_source_images': 0,
        'total_crops_generated': 0,
        'total_crops_filtered': 0,
        'total_crops_kept': 0,
        'crops_per_position': {pos: {'generated': 0, 'kept': 0} for pos in CROP_POSITIONS},
    }

    # Traiter chaque fold
    for fold in folds:
        fold_dir = pannuke_dir / f"fold{fold}"
        images_path = fold_dir / "images.npy"
        masks_path = fold_dir / "masks.npy"
        types_path = fold_dir / "types.npy"

        if not images_path.exists():
            print(f"⚠️  Fold {fold}: fichiers manquants, skip")
            continue

        print(f"📂 Fold {fold}: Chargement...")

        # Charger avec mmap
        images = np.load(images_path, mmap_mode='r')
        masks = np.load(masks_path, mmap_mode='r')
        types = np.load(types_path)

        fold_source_images = 0

        # Filtrer par famille
        for i in tqdm(range(len(images)), desc=f"  Fold {fold}"):
            organ_name = types[i].decode('utf-8') if isinstance(types[i], bytes) else types[i]
            if organ_name not in organs:
                continue

            # Charger image et mask (256×256 original)
            image = np.array(images[i], dtype=np.uint8)
            mask = np.array(masks[i])

            # Validation
            assert image.shape == (PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE, 3), \
                f"Image shape invalide: {image.shape}"
            assert mask.shape == (PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE, 6), \
                f"Mask shape invalide: {mask.shape}"

            fold_source_images += 1

            # Générer targets à 256×256 (AVANT crop)
            inst_map = extract_pannuke_instances(mask)
            np_target = compute_np_target(mask)
            hv_target = compute_hv_maps(inst_map)
            nt_target = compute_nt_target(mask)

            # Générer 5 crops
            for pos_name, (x1, y1, x2, y2) in CROP_POSITIONS.items():
                stats['total_crops_generated'] += 1
                stats['crops_per_position'][pos_name]['generated'] += 1

                # Extraire crop
                crop = extract_crop(image, np_target, hv_target, nt_target, x1, y1, x2, y2)

                # Filtrer si GT vide
                is_valid, num_instances = is_valid_crop(crop['np_target'], crop['nt_target'])

                if is_valid:
                    crops_data['images'].append(crop['image'])
                    crops_data['np_targets'].append(crop['np_target'])
                    crops_data['hv_targets'].append(crop['hv_target'])
                    crops_data['nt_targets'].append(crop['nt_target'])
                    crops_data['source_image_ids'].append(i)
                    crops_data['crop_positions'].append(pos_name)
                    crops_data['fold_ids'].append(fold)

                    stats['total_crops_kept'] += 1
                    stats['crops_per_position'][pos_name]['kept'] += 1
                else:
                    stats['total_crops_filtered'] += 1

        stats['total_source_images'] += fold_source_images
        print(f"  → {fold_source_images} images source traitées")

    # Statistiques
    print(f"\n📊 Statistiques Génération:")
    print(f"  Images source:        {stats['total_source_images']}")
    print(f"  Crops générés:        {stats['total_crops_generated']}")
    print(f"  Crops filtrés (vide): {stats['total_crops_filtered']} ({100*stats['total_crops_filtered']/stats['total_crops_generated']:.1f}%)")
    print(f"  Crops conservés:      {stats['total_crops_kept']} ({100*stats['total_crops_kept']/stats['total_crops_generated']:.1f}%)")

    print(f"\n📊 Répartition par position:")
    for pos_name, pos_stats in stats['crops_per_position'].items():
        kept = pos_stats['kept']
        gen = pos_stats['generated']
        pct = 100 * kept / gen if gen > 0 else 0
        print(f"  {pos_name:15s}: {kept:4d}/{gen:4d} conservés ({pct:5.1f}%)")

    # Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{family}_data_v13_crops.npz"

    print(f"\n💾 Conversion en arrays...")
    images_array = np.stack(crops_data['images'], axis=0)
    np_targets_array = np.stack(crops_data['np_targets'], axis=0)
    hv_targets_array = np.stack(crops_data['hv_targets'], axis=0)
    nt_targets_array = np.stack(crops_data['nt_targets'], axis=0)
    source_ids_array = np.array(crops_data['source_image_ids'], dtype=np.int32)
    crop_positions_array = np.array(crops_data['crop_positions'])
    fold_ids_array = np.array(crops_data['fold_ids'], dtype=np.int32)

    print(f"💾 Sauvegarde: {output_file}")
    np.savez_compressed(
        output_file,
        images=images_array,
        np_targets=np_targets_array,
        hv_targets=hv_targets_array,
        nt_targets=nt_targets_array,
        source_image_ids=source_ids_array,
        crop_positions=crop_positions_array,
        fold_ids=fold_ids_array,
    )

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Fichier créé: {file_size_mb:.1f} MB")

    print(f"\n✅ GÉNÉRATION COMPLÈTE - {stats['total_crops_kept']} crops sauvegardés")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Génération Multi-Crops V13 depuis PanNuke originales"
    )
    parser.add_argument(
        '--pannuke_dir',
        type=Path,
        default=Path('/home/amar/data/PanNuke'),
        help="Répertoire PanNuke"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('data/family_V13'),
        help="Répertoire de sortie"
    )
    parser.add_argument(
        '--family',
        type=str,
        required=True,
        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
        help="Famille tissulaire"
    )
    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help="Folds à traiter (défaut: 0 1 2)"
    )

    args = parser.parse_args()

    # Validation
    if not args.pannuke_dir.exists():
        print(f"❌ ERREUR: PanNuke directory non trouvé: {args.pannuke_dir}")
        sys.exit(1)

    # Génération
    stats = generate_multi_crops_from_pannuke(
        pannuke_dir=args.pannuke_dir,
        output_dir=args.output_dir,
        family=args.family,
        folds=args.folds,
    )


if __name__ == '__main__':
    main()
