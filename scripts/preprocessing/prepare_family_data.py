#!/usr/bin/env python3
"""
Prépare les données par famille d'organes pour entraînement HoVer-Net.

Filtre les features et masks par famille et sauvegarde des fichiers séparés.
Cela évite de charger toutes les données en RAM pendant l'entraînement.

Usage:
    # Préparer toutes les familles
    python scripts/preprocessing/prepare_family_data.py --data_dir /home/amar/data/PanNuke

    # Préparer une seule famille
    python scripts/preprocessing/prepare_family_data.py --data_dir /home/amar/data/PanNuke --family glandular
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.organ_families import FAMILIES, FAMILY_TO_ORGANS, FAMILY_DESCRIPTIONS


def prepare_family(data_dir: Path, output_dir: Path, family: str, folds: list = None):
    """
    Prépare les données pour une famille d'organes.

    Sauvegarde:
    - {family}_features.npz : features H-optimus-0 filtrées
    - {family}_masks.npz : masks + indices originaux
    """
    if folds is None:
        folds = [0, 1, 2]

    family_organs = set(FAMILY_TO_ORGANS[family])
    print(f"\n{'='*60}")
    print(f"FAMILLE: {family.upper()}")
    print(f"{'='*60}")
    print(f"Organes: {', '.join(family_organs)}")
    print(f"Description: {FAMILY_DESCRIPTIONS[family]}")

    all_features = []
    all_masks = []
    all_fold_ids = []
    all_original_indices = []

    for fold in folds:
        # Charger les features
        features_path = PROJECT_ROOT / "data" / "cache" / "pannuke_features" / f"fold{fold}_features.npz"
        if not features_path.exists():
            print(f"  ⚠️ Features fold {fold} non trouvées, ignoré")
            continue

        print(f"\nFold {fold}:")

        # Charger les types pour filtrer
        types_path = data_dir / f"fold{fold}" / "types.npy"
        types = np.load(types_path)

        # Trouver les indices de cette famille
        indices = []
        for i, organ in enumerate(types):
            organ_str = str(organ).strip()
            if organ_str in family_organs:
                indices.append(i)

        if not indices:
            print(f"  → 0 samples pour cette famille")
            continue

        indices = np.array(indices)
        print(f"  → {len(indices)} samples sélectionnés")

        # Charger features (memory-mapped pour économiser la RAM)
        data = np.load(features_path, mmap_mode='r')
        features = data['layer_24'] if 'layer_24' in data else data['layer_23']

        # Charger masks (memory-mapped)
        masks_path = data_dir / f"fold{fold}" / "masks.npy"
        masks = np.load(masks_path, mmap_mode='r')

        # Extraire seulement les samples de cette famille
        family_features = features[indices].copy()  # Copie nécessaire car mmap
        family_masks = masks[indices].copy()

        all_features.append(family_features)
        all_masks.append(family_masks)
        all_fold_ids.extend([fold] * len(indices))
        all_original_indices.extend(indices.tolist())

        print(f"  → Features: {family_features.shape}")
        print(f"  → Masks: {family_masks.shape}")

    if not all_features:
        print(f"\n❌ Aucun sample trouvé pour la famille {family}")
        return

    # Concaténer
    features = np.concatenate(all_features, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    fold_ids = np.array(all_fold_ids)
    original_indices = np.array(all_original_indices)

    print(f"\n📊 Total famille {family}:")
    print(f"  → {len(features)} samples")
    print(f"  → Features: {features.shape} ({features.nbytes / 1e9:.2f} GB)")
    print(f"  → Masks: {masks.shape} ({masks.nbytes / 1e9:.2f} GB)")

    # Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Features (compressées pour économiser l'espace)
    features_path = output_dir / f"{family}_features.npz"
    print(f"\nSauvegarde features: {features_path}")
    np.savez(features_path,
             layer_24=features,
             fold_ids=fold_ids,
             original_indices=original_indices)

    # Masks (compressées)
    masks_path = output_dir / f"{family}_masks.npz"
    print(f"Sauvegarde masks: {masks_path}")
    np.savez(masks_path, masks=masks)

    # Afficher les tailles
    features_size = features_path.stat().st_size / 1e9
    masks_size = masks_path.stat().st_size / 1e9
    print(f"\n✅ Famille {family} préparée:")
    print(f"   Features: {features_size:.2f} GB")
    print(f"   Masks: {masks_size:.2f} GB")
    print(f"   Total: {features_size + masks_size:.2f} GB")

    return len(features)


def main():
    parser = argparse.ArgumentParser(description="Préparer données par famille")
    parser.add_argument('--data_dir', type=str, default='/home/amar/data/PanNuke',
                       help='Répertoire PanNuke')
    parser.add_argument('--output_dir', type=str, default='data/cache/family_data',
                       help='Répertoire de sortie')
    parser.add_argument('--family', type=str, choices=FAMILIES, default=None,
                       help='Famille spécifique (sinon toutes)')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2],
                       help='Folds à utiliser')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("="*60)
    print("PRÉPARATION DES DONNÉES PAR FAMILLE")
    print("="*60)

    families_to_process = [args.family] if args.family else FAMILIES

    total_samples = {}
    for family in families_to_process:
        n_samples = prepare_family(data_dir, output_dir, family, args.folds)
        if n_samples:
            total_samples[family] = n_samples

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    for family, n in total_samples.items():
        print(f"  {family:15}: {n:5} samples")
    print(f"  {'TOTAL':15}: {sum(total_samples.values()):5} samples")
    print(f"\nFichiers dans: {output_dir}")


if __name__ == "__main__":
    main()
