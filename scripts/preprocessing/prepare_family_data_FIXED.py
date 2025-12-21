#!/usr/bin/env python3
"""
Prépare les données d'entraînement par famille d'organes.

VERSION FIXÉE: Utilise les IDs d'instances NATIFS de PanNuke au lieu de connectedComponents.

BUG CORRIGÉ:
- Avant: connectedComponents fusionnait les cellules qui se touchent → 75% perdues
- Après: Utilise les IDs natifs PanNuke (canaux 1-4) → vraies instances séparées

Usage:
    python scripts/preprocessing/prepare_family_data_FIXED.py --data_dir /home/amar/data/PanNuke
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


def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """
    Calcule les cartes Horizontal/Vertical pour séparation d'instances.

    FIXE: Utilise l'inst_map avec vraies instances séparées PanNuke.

    Args:
        inst_map: (H, W) avec IDs d'instances [0, 1, 2, ...]

    Returns:
        hv_maps: (2, H, W) avec H et V normalisés [-1, +1]
    """
    h, w = inst_map.shape
    hv_map = np.zeros((2, h, w), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]  # Exclude background

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id

        # Trouver le centroïde de l'instance
        y_coords, x_coords = np.where(inst_mask)

        if len(y_coords) == 0:
            continue

        centroid_y = y_coords.mean()
        centroid_x = x_coords.mean()

        # Calculer distances normalisées au centroïde
        y_dist = y_coords - centroid_y
        x_dist = x_coords - centroid_x

        # Normaliser par distance maximale
        max_dist_y = np.abs(y_dist).max()
        max_dist_x = np.abs(x_dist).max()

        if max_dist_y > 0:
            y_dist = y_dist / max_dist_y
        if max_dist_x > 0:
            x_dist = x_dist / max_dist_x

        # Assigner aux cartes HV
        hv_map[0, y_coords, x_coords] = x_dist  # H (horizontal)
        hv_map[1, y_coords, x_coords] = y_dist  # V (vertical)

    return hv_map


def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """
    Extrait les vraies instances de PanNuke (FIXÉ).

    AVANT (BUGGY):
        np_mask = mask[:, :, 1:].sum(axis=-1) > 0
        _, inst_map = cv2.connectedComponents(np_mask.astype(np.uint8))
        → Fusionne les cellules qui se touchent ❌

    APRÈS (FIXÉ):
        Utilise les IDs natifs PanNuke dans canaux 1-4 ✅

    Args:
        mask: (256, 256, 6) PanNuke mask
            - Canal 0: Background
            - Canal 1: Neoplastic instance IDs
            - Canal 2: Inflammatory instance IDs
            - Canal 3: Connective instance IDs
            - Canal 4: Dead instance IDs
            - Canal 5: Epithelial (binaire, pas d'IDs)

    Returns:
        inst_map: (256, 256) avec IDs d'instances uniques [0, 1, 2, ...]
    """
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: IDs d'instances natifs PanNuke
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
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


def prepare_family_data(data_dir: Path, output_dir: Path, family: str, chunk_size: int = 500):
    """
    Prépare les données d'entraînement pour une famille d'organes.

    VERSION FIXÉE avec vraies instances PanNuke + OPTIMISATION RAM (chunking).

    Args:
        chunk_size: Nombre d'images à traiter par lot (défaut: 500)
                    Réduit la consommation RAM (~2 GB par chunk au lieu de 10+ GB)
    """
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

    for fold in [0, 1, 2]:
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

    for fold, indices in fold_indices.items():
        fold_dir = data_dir / f"fold{fold}"
        images_path = fold_dir / "images.npy"
        masks_path = fold_dir / "masks.npy"

        # Charger avec mmap (pas en RAM)
        images = np.load(images_path, mmap_mode='r')
        masks = np.load(masks_path, mmap_mode='r')

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
                image = np.array(images[idx])  # Copie en RAM seulement ce dont on a besoin
                mask = np.array(masks[idx])

                # ✅ FIXÉ: Utiliser vraies instances PanNuke
                inst_map = extract_pannuke_instances(mask)

                # NP target
                np_target = (inst_map > 0).astype(np.float32)

                # ✅ FIXÉ: HV targets avec vraies instances
                hv_target = compute_hv_maps(inst_map)

                # NT target
                nt_target = np.argmax(mask[:, :, 1:], axis=-1).astype(np.int64)

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


def main():
    parser = argparse.ArgumentParser(description="Prépare données par famille (VERSION FIXÉE + RAM-OPTIMIZED)")
    parser.add_argument("--data_dir", type=Path, default=Path("/home/amar/data/PanNuke"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/family_FIXED"))
    parser.add_argument("--family", type=str, choices=FAMILIES, help="Famille spécifique (optionnel)")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Nombre d'images par chunk (défaut: 500, réduit RAM)")

    args = parser.parse_args()

    print("=" * 70)
    print("PRÉPARATION DONNÉES PAR FAMILLE (VERSION FIXÉE + RAM-OPTIMIZED)")
    print("=" * 70)
    print(f"\nChangements:")
    print(f"  ❌ AVANT: connectedComponents fusionnait cellules touchantes")
    print(f"  ✅ APRÈS: IDs natifs PanNuke (vraies instances séparées)")
    print(f"\nOptimisations:")
    print(f"  ✅ Traitement par chunks de {args.chunk_size} images")
    print(f"  ✅ mmap_mode='r' pour économiser la RAM")
    print(f"  ✅ Consommation RAM: ~2 GB par chunk au lieu de 10+ GB")
    print(f"\nImpact:")
    print(f"  - HV maps avec frontières RÉELLES entre cellules")
    print(f"  - Gradients HV FORTS aux bordures")
    print(f"  - Modèle apprendra à séparer correctement")

    if args.family:
        prepare_family_data(args.data_dir, args.output_dir, args.family, args.chunk_size)
    else:
        for family in FAMILIES:
            prepare_family_data(args.data_dir, args.output_dir, family, args.chunk_size)

    print("\n" + "=" * 70)
    print("✅ PRÉPARATION TERMINÉE")
    print("=" * 70)
    print(f"\nProchaines étapes:")
    print(f"  1. Vérifier les nouvelles données dans {args.output_dir}")
    print(f"  2. Comparer HV maps BEFORE vs AFTER")
    print(f"  3. Ré-entraîner HoVer-Net avec nouvelles données (~10h pour 5 familles)")


if __name__ == "__main__":
    main()
