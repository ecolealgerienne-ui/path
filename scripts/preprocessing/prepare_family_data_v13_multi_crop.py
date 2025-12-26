#!/usr/bin/env python3
"""
Préparation données V13 - Multi-Crop Statique (5 crops par image).

Génère 5 crops fixes (224×224) depuis chaque patch source (256×256):
- Center: (16, 16) → (240, 240)
- Top-Left: (0, 0) → (224, 224)
- Top-Right: (32, 0) → (256, 224)
- Bottom-Left: (0, 32) → (224, 256)
- Bottom-Right: (32, 32) → (256, 256)

Filtre les crops avec GT vide (aucune instance détectée).

Usage:
    python scripts/preprocessing/prepare_family_data_v13_multi_crop.py \
        --family epidermal \
        --input_file data/family_FIXED/epidermal_data_FIXED_v12_COHERENT.npz \
        --output_dir data/family_V13
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE

# Positions de crop fixes (5 crops par image 256×256)
CROP_POSITIONS = {
    'center':       (16, 16, 240, 240),  # (x_start, y_start, x_end, y_end)
    'top_left':     (0,  0,  224, 224),
    'top_right':    (32, 0,  256, 224),
    'bottom_left':  (0,  32, 224, 256),
    'bottom_right': (32, 32, 256, 256),
}

CROP_SIZE = 224


def extract_crop(
    image: np.ndarray,
    np_target: np.ndarray,
    hv_target: np.ndarray,
    nt_target: np.ndarray,
    x1: int, y1: int, x2: int, y2: int
) -> Dict[str, np.ndarray]:
    """
    Extrait un crop 224×224 depuis une image 256×256 et ses targets.

    Args:
        image: Image RGB (256, 256, 3) uint8
        np_target: Nuclear Presence (256, 256) float32
        hv_target: HV maps (2, 256, 256) float32
        nt_target: Nuclear Type (256, 256) int64
        x1, y1, x2, y2: Coordonnées du crop

    Returns:
        Dict contenant les crops (tous à 224×224)
    """
    # ⚠️ CRITIQUE: Slicing identique pour garantir alignement bit-à-bit
    crop_image = image[y1:y2, x1:x2]          # (224, 224, 3)
    crop_np = np_target[y1:y2, x1:x2]         # (224, 224)
    crop_hv = hv_target[:, y1:y2, x1:x2]      # (2, 224, 224) ← Note le slicing spécial
    crop_nt = nt_target[y1:y2, x1:x2]         # (224, 224)

    # Validation des shapes
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
    """
    Vérifie si un crop contient des instances valides.

    Critère de filtrage (Option B validée):
    - len(np.unique(inst_map)) > 1 (au moins 1 instance + background)

    Args:
        np_target: Nuclear Presence (224, 224) float32
        nt_target: Nuclear Type (224, 224) int64

    Returns:
        (is_valid, num_instances)
    """
    # Créer instance map depuis NP (masque binaire)
    binary_mask = (np_target > 0.5).astype(np.uint8)

    # Compter instances via connected components
    from scipy import ndimage
    inst_map, num_instances = ndimage.label(binary_mask)

    # Filtrage: Au moins 1 instance (num_instances > 0)
    # inst_map contient: [0 (background), 1, 2, ..., num_instances]
    # Donc len(unique) = num_instances + 1
    unique_labels = np.unique(inst_map)
    is_valid = len(unique_labels) > 1  # Plus que juste background

    return is_valid, num_instances


def generate_multi_crops(
    input_file: Path,
    output_dir: Path,
    family: str
) -> Dict[str, int]:
    """
    Génère 5 crops par image et filtre les crops vides.

    Args:
        input_file: Fichier .npz V12 (256×256)
        output_dir: Répertoire de sortie V13
        family: Nom de la famille

    Returns:
        Statistiques de génération
    """
    print(f"\n{'='*70}")
    print(f"GÉNÉRATION MULTI-CROPS V13 - Famille: {family.upper()}")
    print(f"{'='*70}\n")

    # 1. Charger données V12
    print(f"📂 Chargement données V12: {input_file}")
    data = np.load(input_file)

    images = data['images']          # (N, 256, 256, 3) uint8
    np_targets = data['np_targets']  # (N, 256, 256) float32
    hv_targets = data['hv_targets']  # (N, 2, 256, 256) float32
    nt_targets = data['nt_targets']  # (N, 256, 256) int64
    fold_ids = data.get('fold_ids', np.zeros(len(images), dtype=np.int32))
    image_ids = data.get('image_ids', np.arange(len(images), dtype=np.int32))

    n_images = len(images)
    print(f"✅ {n_images} images chargées (shape: {images.shape})")

    # Validation shape
    assert images.shape[1:3] == (PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), \
        f"Expected 256×256, got {images.shape[1:3]}"

    # 2. Générer crops
    print(f"\n🔧 Génération de 5 crops par image...")

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
        'total_crops_generated': 0,
        'total_crops_filtered': 0,
        'total_crops_kept': 0,
        'crops_per_position': {pos: {'generated': 0, 'kept': 0} for pos in CROP_POSITIONS},
    }

    for img_idx in tqdm(range(n_images), desc="Processing images"):
        image = images[img_idx]
        np_target = np_targets[img_idx]
        hv_target = hv_targets[img_idx]
        nt_target = nt_targets[img_idx]
        fold_id = fold_ids[img_idx]
        image_id = image_ids[img_idx]

        # Générer les 5 crops
        for pos_name, (x1, y1, x2, y2) in CROP_POSITIONS.items():
            stats['total_crops_generated'] += 1
            stats['crops_per_position'][pos_name]['generated'] += 1

            # Extraire crop
            crop = extract_crop(image, np_target, hv_target, nt_target, x1, y1, x2, y2)

            # Filtrer si GT vide
            is_valid, num_instances = is_valid_crop(crop['np_target'], crop['nt_target'])

            if is_valid:
                # Garder ce crop
                crops_data['images'].append(crop['image'])
                crops_data['np_targets'].append(crop['np_target'])
                crops_data['hv_targets'].append(crop['hv_target'])
                crops_data['nt_targets'].append(crop['nt_target'])
                crops_data['source_image_ids'].append(image_id)
                crops_data['crop_positions'].append(pos_name)
                crops_data['fold_ids'].append(fold_id)

                stats['total_crops_kept'] += 1
                stats['crops_per_position'][pos_name]['kept'] += 1
            else:
                # Filtré (GT vide)
                stats['total_crops_filtered'] += 1

    # 3. Convertir en arrays
    print(f"\n📦 Conversion en arrays NumPy...")

    crops_data['images'] = np.array(crops_data['images'], dtype=np.uint8)
    crops_data['np_targets'] = np.array(crops_data['np_targets'], dtype=np.float32)
    crops_data['hv_targets'] = np.array(crops_data['hv_targets'], dtype=np.float32)
    crops_data['nt_targets'] = np.array(crops_data['nt_targets'], dtype=np.int64)
    crops_data['source_image_ids'] = np.array(crops_data['source_image_ids'], dtype=np.int32)
    crops_data['crop_positions'] = np.array(crops_data['crop_positions'], dtype='U20')
    crops_data['fold_ids'] = np.array(crops_data['fold_ids'], dtype=np.int32)

    n_crops_kept = len(crops_data['images'])
    print(f"✅ {n_crops_kept} crops valides (shape: {crops_data['images'].shape})")

    # Validation finale
    assert crops_data['images'].shape == (n_crops_kept, CROP_SIZE, CROP_SIZE, 3)
    assert crops_data['np_targets'].shape == (n_crops_kept, CROP_SIZE, CROP_SIZE)
    assert crops_data['hv_targets'].shape == (n_crops_kept, 2, CROP_SIZE, CROP_SIZE)
    assert crops_data['nt_targets'].shape == (n_crops_kept, CROP_SIZE, CROP_SIZE)

    # 4. Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{family}_data_v13_crops.npz"

    print(f"\n💾 Sauvegarde: {output_file}")
    np.savez_compressed(
        output_file,
        **crops_data
    )

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Fichier créé: {file_size_mb:.1f} MB")

    # 5. Afficher statistiques
    print(f"\n{'='*70}")
    print(f"STATISTIQUES DE GÉNÉRATION")
    print(f"{'='*70}\n")

    print(f"Images source (V12):        {n_images}")
    print(f"Crops générés (total):      {stats['total_crops_generated']}")
    print(f"Crops filtrés (GT vide):    {stats['total_crops_filtered']} "
          f"({100*stats['total_crops_filtered']/stats['total_crops_generated']:.1f}%)")
    print(f"Crops conservés:            {stats['total_crops_kept']} "
          f"({100*stats['total_crops_kept']/stats['total_crops_generated']:.1f}%)")

    print(f"\n📊 Répartition par position:")
    for pos_name in CROP_POSITIONS:
        generated = stats['crops_per_position'][pos_name]['generated']
        kept = stats['crops_per_position'][pos_name]['kept']
        ratio = 100 * kept / generated if generated > 0 else 0
        print(f"  {pos_name:15s}: {kept:4d}/{generated:4d} ({ratio:5.1f}% conservés)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Génération Multi-Crops V13 (5 crops fixes par image)"
    )
    parser.add_argument(
        '--family',
        type=str,
        required=True,
        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
        help="Famille tissulaire"
    )
    parser.add_argument(
        '--input_file',
        type=Path,
        required=True,
        help="Fichier .npz V12 (256×256)"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('data/family_V13'),
        help="Répertoire de sortie V13"
    )

    args = parser.parse_args()

    # Validation fichier d'entrée
    if not args.input_file.exists():
        print(f"❌ ERREUR: Fichier introuvable: {args.input_file}")
        sys.exit(1)

    # Génération
    stats = generate_multi_crops(
        input_file=args.input_file,
        output_dir=args.output_dir,
        family=args.family
    )

    print(f"\n{'='*70}")
    print(f"✅ GÉNÉRATION COMPLÈTE - {stats['total_crops_kept']} crops prêts pour extraction features")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
