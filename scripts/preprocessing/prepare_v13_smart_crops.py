#!/usr/bin/env python3
"""
Préparation données V13 Smart Crops - 5 crops + 5 rotations déterministes.

⚠️ APPROCHE SPLIT-FIRST-THEN-ROTATE (CTO-validated):
1. Split train/val par source_image_ids (80/20)
2. Apply 5 crops to TRAIN → train crops
3. Apply 5 crops to VAL → val crops
4. Apply 5 rotations to train crops → train dataset
5. Apply 5 rotations to val crops → val dataset

Cette approche garantit ZÉRO data leakage (aucune source partagée).

Usage:
    python scripts/preprocessing/prepare_v13_smart_crops.py \
        --family epidermal \
        --pannuke_dir /home/amar/data/PanNuke \
        --output_dir data/family_data_v13_smart_crops
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE
from src.models.organ_families import ORGAN_TO_FAMILY

# Positions de crop fixes (5 crops par image 256×256)
CROP_POSITIONS = {
    'center':       (16, 16, 240, 240),
    'top_left':     (0,  0,  224, 224),
    'top_right':    (32, 0,  256, 224),
    'bottom_left':  (0,  32, 224, 256),
    'bottom_right': (32, 32, 256, 256),
}

CROP_SIZE = 224


def normalize_mask_format(mask: np.ndarray) -> np.ndarray:
    """Normalise format masque PanNuke (H, W, 6)."""
    if mask.ndim == 2:
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

    # Channel 0: Multi-type instances
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
    nt_target[nuclei_mask] = 1
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
    is_valid = len(unique_labels) > 1
    return is_valid, num_instances


def apply_rotation(
    image: np.ndarray,
    np_target: np.ndarray,
    hv_target: np.ndarray,
    nt_target: np.ndarray,
    rotation: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applique rotation déterministe avec HV component swapping.

    Args:
        image: (224, 224, 3)
        np_target: (224, 224)
        hv_target: (2, 224, 224) - [V, H]
        nt_target: (224, 224)
        rotation: '0', '90', '180', '270', 'flip_h'

    Returns:
        (image_rot, np_rot, hv_rot, nt_rot)
    """
    if rotation == '0':
        return image, np_target, hv_target, nt_target

    elif rotation == '90':
        # Rotation 90° clockwise
        image_rot = np.rot90(image, k=-1, axes=(0, 1))
        np_rot = np.rot90(np_target, k=-1, axes=(0, 1))
        nt_rot = np.rot90(nt_target, k=-1, axes=(0, 1))

        # HV component swapping: H' = V, V' = -H
        v_old, h_old = hv_target[0], hv_target[1]
        v_rot = -np.rot90(h_old, k=-1, axes=(0, 1))      # V' = -H
        h_rot = np.rot90(v_old, k=-1, axes=(0, 1))       # H' = V
        hv_rot = np.stack([v_rot, h_rot], axis=0)

        return image_rot, np_rot, hv_rot, nt_rot

    elif rotation == '180':
        # Rotation 180°
        image_rot = np.rot90(image, k=2, axes=(0, 1))
        np_rot = np.rot90(np_target, k=2, axes=(0, 1))
        nt_rot = np.rot90(nt_target, k=2, axes=(0, 1))

        # HV negation: H' = -H, V' = -V
        v_rot = -np.rot90(hv_target[0], k=2, axes=(0, 1))
        h_rot = -np.rot90(hv_target[1], k=2, axes=(0, 1))
        hv_rot = np.stack([v_rot, h_rot], axis=0)

        return image_rot, np_rot, hv_rot, nt_rot

    elif rotation == '270':
        # Rotation 270° clockwise (= 90° counter-clockwise)
        image_rot = np.rot90(image, k=1, axes=(0, 1))
        np_rot = np.rot90(np_target, k=1, axes=(0, 1))
        nt_rot = np.rot90(nt_target, k=1, axes=(0, 1))

        # HV component swapping: H' = -V, V' = H
        v_old, h_old = hv_target[0], hv_target[1]
        v_rot = np.rot90(h_old, k=1, axes=(0, 1))        # V' = H
        h_rot = -np.rot90(v_old, k=1, axes=(0, 1))       # H' = -V
        hv_rot = np.stack([v_rot, h_rot], axis=0)

        return image_rot, np_rot, hv_rot, nt_rot

    elif rotation == 'flip_h':
        # Flip horizontal
        image_rot = np.fliplr(image)
        np_rot = np.fliplr(np_target)
        nt_rot = np.fliplr(nt_target)

        # HV flip: H' = -H, V' = V
        v_rot = np.fliplr(hv_target[0])
        h_rot = -np.fliplr(hv_target[1])
        hv_rot = np.stack([v_rot, h_rot], axis=0)

        return image_rot, np_rot, hv_rot, nt_rot

    else:
        raise ValueError(f"Unknown rotation: {rotation}")


def generate_smart_crops_from_pannuke(
    pannuke_dir: Path,
    output_dir: Path,
    family: str,
    folds: list = None,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, int]:
    """
    Génère crops avec split-first-then-rotate strategy.

    Args:
        pannuke_dir: Répertoire PanNuke
        output_dir: Répertoire de sortie
        family: Famille tissulaire
        folds: Liste des folds (défaut: [0, 1, 2])
        train_ratio: Ratio train/val (défaut: 0.8)
        seed: Seed pour reproductibilité

    Returns:
        Statistiques de génération
    """
    if folds is None:
        folds = [0, 1, 2]

    print(f"\n{'='*70}")
    print(f"GÉNÉRATION V13 SMART CROPS - Famille: {family.upper()}")
    print(f"{'='*70}\n")

    # Organes de cette famille
    organs = [org for org, fam in ORGAN_TO_FAMILY.items() if fam == family]
    print(f"Organes: {', '.join(organs)}\n")

    # ========== ÉTAPE 1: Collecter toutes les images sources ==========
    all_source_images = []
    all_source_masks = []
    all_source_ids = []
    all_fold_ids = []

    for fold in folds:
        fold_dir = pannuke_dir / f"fold{fold}"
        images_path = fold_dir / "images.npy"
        masks_path = fold_dir / "masks.npy"
        types_path = fold_dir / "types.npy"

        if not images_path.exists():
            print(f"⚠️  Fold {fold}: fichiers manquants, skip")
            continue

        print(f"📂 Fold {fold}: Chargement...")

        images = np.load(images_path, mmap_mode='r')
        masks = np.load(masks_path, mmap_mode='r')
        types = np.load(types_path)

        for i in range(len(images)):
            organ_name = types[i].decode('utf-8') if isinstance(types[i], bytes) else types[i]
            if organ_name not in organs:
                continue

            # Charger en mémoire uniquement les images de cette famille
            image = np.array(images[i], dtype=np.uint8)
            mask = np.array(masks[i])

            all_source_images.append(image)
            all_source_masks.append(mask)
            all_source_ids.append(i)
            all_fold_ids.append(fold)

    n_total = len(all_source_images)
    print(f"✅ Total images sources collectées: {n_total}\n")

    # ========== ÉTAPE 2: Split train/val par source images ==========
    np.random.seed(seed)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    n_train = int(train_ratio * n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    print(f"📊 Split train/val:")
    print(f"  Train: {len(train_indices)} images sources ({100*len(train_indices)/n_total:.1f}%)")
    print(f"  Val:   {len(val_indices)} images sources ({100*len(val_indices)/n_total:.1f}%)\n")

    # ========== ÉTAPE 3: Traiter train et val séparément ==========
    rotations = ['0', '90', '180', '270', 'flip_h']

    for split_name, split_indices in [('train', train_indices), ('val', val_indices)]:
        print(f"{'='*70}")
        print(f"Traitement split: {split_name.upper()}")
        print(f"{'='*70}\n")

        crops_data = {
            'images': [],
            'np_targets': [],
            'hv_targets': [],
            'nt_targets': [],
            'source_image_ids': [],
            'crop_positions': [],
            'fold_ids': [],
            'rotations': [],
        }

        stats = {
            'total_crops_kept': 0,
            'total_crops_filtered': 0,
        }

        # Traiter chaque image source du split
        for idx in tqdm(split_indices, desc=f"  {split_name}"):
            image = all_source_images[idx]
            mask = all_source_masks[idx]
            source_id = all_source_ids[idx]
            fold_id = all_fold_ids[idx]

            # Validation
            assert image.shape == (PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE, 3)
            assert mask.shape == (PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE, 6)

            # Générer targets à 256×256 (AVANT crop)
            inst_map = extract_pannuke_instances(mask)
            np_target = compute_np_target(mask)
            hv_target = compute_hv_maps(inst_map)
            nt_target = compute_nt_target(mask)

            # Générer 5 crops
            for pos_name, (x1, y1, x2, y2) in CROP_POSITIONS.items():
                crop = extract_crop(image, np_target, hv_target, nt_target, x1, y1, x2, y2)

                # Filtrer si GT vide
                is_valid, _ = is_valid_crop(crop['np_target'], crop['nt_target'])

                if not is_valid:
                    stats['total_crops_filtered'] += 1
                    continue

                # Appliquer 5 rotations à ce crop
                for rotation in rotations:
                    img_rot, np_rot, hv_rot, nt_rot = apply_rotation(
                        crop['image'],
                        crop['np_target'],
                        crop['hv_target'],
                        crop['nt_target'],
                        rotation
                    )

                    crops_data['images'].append(img_rot)
                    crops_data['np_targets'].append(np_rot)
                    crops_data['hv_targets'].append(hv_rot)
                    crops_data['nt_targets'].append(nt_rot)
                    crops_data['source_image_ids'].append(source_id)
                    crops_data['crop_positions'].append(pos_name)
                    crops_data['fold_ids'].append(fold_id)
                    crops_data['rotations'].append(rotation)

                    stats['total_crops_kept'] += 1

        # Statistiques split
        print(f"\n📊 Statistiques {split_name}:")
        print(f"  Crops conservés:      {stats['total_crops_kept']}")
        print(f"  Crops filtrés (vide): {stats['total_crops_filtered']}")

        # Sauvegarder split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{family}_{split_name}_v13_smart_crops.npz"

        print(f"\n💾 Conversion en arrays...")
        images_array = np.stack(crops_data['images'], axis=0)
        np_targets_array = np.stack(crops_data['np_targets'], axis=0)
        hv_targets_array = np.stack(crops_data['hv_targets'], axis=0)
        nt_targets_array = np.stack(crops_data['nt_targets'], axis=0)
        source_ids_array = np.array(crops_data['source_image_ids'], dtype=np.int32)
        crop_positions_array = np.array(crops_data['crop_positions'])
        fold_ids_array = np.array(crops_data['fold_ids'], dtype=np.int32)
        rotations_array = np.array(crops_data['rotations'])

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
            rotations=rotations_array,
        )

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ Fichier créé: {file_size_mb:.1f} MB\n")

    print(f"✅ GÉNÉRATION COMPLÈTE - Train et Val sauvegardés séparément")

    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Génération V13 Smart Crops avec split-first-then-rotate"
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
        default=Path('data/family_data_v13_smart_crops'),
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
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help="Ratio train/val (défaut: 0.8)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Seed pour reproductibilité (défaut: 42)"
    )

    args = parser.parse_args()

    # Validation
    if not args.pannuke_dir.exists():
        print(f"❌ ERREUR: PanNuke directory non trouvé: {args.pannuke_dir}")
        sys.exit(1)

    # Génération
    stats = generate_smart_crops_from_pannuke(
        pannuke_dir=args.pannuke_dir,
        output_dir=args.output_dir,
        family=args.family,
        folds=args.folds,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
