#!/usr/bin/env python3
"""
Prépare les données par famille d'organes pour entraînement HoVer-Net.

Pré-calcule les targets HV (horizontal/vertical maps) pour éviter le calcul
coûteux pendant l'entraînement.

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
import cv2

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.organ_families import FAMILIES, FAMILY_TO_ORGANS, FAMILY_DESCRIPTIONS


def compute_hv_maps(binary_mask: np.ndarray) -> np.ndarray:
    """Calcule les cartes H/V depuis un masque binaire."""
    hv = np.zeros((2, 256, 256), dtype=np.float32)

    if not binary_mask.any():
        return hv

    binary_uint8 = (binary_mask * 255).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary_uint8)

    for label_id in range(1, n_labels):
        instance_mask = labels == label_id
        coords = np.where(instance_mask)

        if len(coords[0]) == 0:
            continue

        cy = coords[0].mean()
        cx = coords[1].mean()

        for y, x in zip(coords[0], coords[1]):
            h_dist = (x - cx)
            v_dist = (y - cy)
            radius = max(np.sqrt(len(coords[0]) / np.pi), 1)
            hv[0, y, x] = h_dist / radius
            hv[1, y, x] = v_dist / radius

    hv = np.clip(hv, -1, 1)
    return hv


def prepare_targets_chunk(masks: np.ndarray, start_idx: int = 0) -> tuple:
    """
    Pré-calcule les targets HoVer-Net pour un chunk de masks.

    Returns:
        np_targets: (N, 256, 256) float32 - binary nuclei mask
        hv_targets: (N, 2, 256, 256) float32 - horizontal/vertical maps
        nt_targets: (N, 256, 256) int64 - nuclei type
    """
    N = len(masks)
    np_targets = np.zeros((N, 256, 256), dtype=np.float32)
    hv_targets = np.zeros((N, 2, 256, 256), dtype=np.float32)
    nt_targets = np.zeros((N, 256, 256), dtype=np.int64)

    for i in tqdm(range(N), desc=f"  HV maps [{start_idx}:{start_idx+N}]"):
        mask = masks[i]

        # NP: union de tous les types
        np_mask = mask[:, :, 1:].sum(axis=-1) > 0
        np_targets[i] = np_mask.astype(np.float32)

        # NT: argmax sur canaux 1-5
        for c in range(5):
            type_mask = mask[:, :, c + 1] > 0
            nt_targets[i][type_mask] = c

        # HV: cartes horizontal/vertical (le plus coûteux)
        hv_targets[i] = compute_hv_maps(np_mask)

    return np_targets, hv_targets, nt_targets


def prepare_family(data_dir: Path, output_dir: Path, family: str, folds: list = None, chunk_size: int = None):
    """
    Prépare les données pour une famille d'organes.

    Traite fold par fold pour éviter de surcharger la RAM.

    Sauvegarde:
    - {family}_features.npz : features H-optimus-0 filtrées
    - {family}_targets.npz : targets pré-calculés (NP, HV, NT)
    """
    if folds is None:
        folds = [0, 1, 2]

    family_organs = set(FAMILY_TO_ORGANS[family])
    print(f"\n{'='*60}")
    print(f"FAMILLE: {family.upper()}")
    print(f"{'='*60}")
    print(f"Organes: {', '.join(family_organs)}")
    print(f"Description: {FAMILY_DESCRIPTIONS[family]}")

    # Listes pour accumuler les résultats
    all_features = []
    all_np_targets = []
    all_hv_targets = []
    all_nt_targets = []
    all_fold_ids = []
    all_original_indices = []

    for fold in folds:
        # Charger les features
        features_path = PROJECT_ROOT / "data" / "cache" / "pannuke_features" / f"fold{fold}_features.npz"
        if not features_path.exists():
            print(f"  ⚠️ Features fold {fold} non trouvées, ignoré")
            continue

        print(f"\n--- Fold {fold} ---")

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

        # Charger features (memory-mapped)
        print(f"  Chargement features...")
        data = np.load(features_path, mmap_mode='r')
        features = data['layer_24'] if 'layer_24' in data else data['layer_23']
        family_features = features[indices].copy()
        print(f"    Features: {family_features.shape}")

        # Charger masks (memory-mapped)
        print(f"  Chargement masks...")
        masks_path = data_dir / f"fold{fold}" / "masks.npy"
        masks = np.load(masks_path, mmap_mode='r')
        family_masks = masks[indices].copy()

        # Calculer targets pour CE FOLD uniquement
        print(f"  Calcul targets HV...")
        np_targets, hv_targets, nt_targets = prepare_targets_chunk(family_masks, start_idx=0)

        # Convertir HV en int8 immédiatement pour économiser la RAM
        hv_targets_int8 = (hv_targets * 127).astype(np.int8)
        del hv_targets  # Libérer la mémoire

        # Accumuler
        all_features.append(family_features)
        all_np_targets.append(np_targets)
        all_hv_targets.append(hv_targets_int8)
        all_nt_targets.append(nt_targets)
        all_fold_ids.extend([fold] * len(indices))
        all_original_indices.extend(indices.tolist())

        # Libérer la mémoire du fold
        del family_masks, np_targets, nt_targets
        print(f"  ✓ Fold {fold} traité")

    if not all_features:
        print(f"\n❌ Aucun sample trouvé pour la famille {family}")
        return

    # Concaténer les résultats
    print(f"\n📊 Concaténation des {len(folds)} folds...")
    features = np.concatenate(all_features, axis=0)
    np_targets = np.concatenate(all_np_targets, axis=0)
    hv_targets_int8 = np.concatenate(all_hv_targets, axis=0)
    nt_targets = np.concatenate(all_nt_targets, axis=0)
    fold_ids = np.array(all_fold_ids)
    original_indices = np.array(all_original_indices)

    print(f"  → {len(features)} samples total")
    print(f"  → Features: {features.nbytes / 1e9:.2f} GB")
    print(f"  → Targets: {(np_targets.nbytes + hv_targets_int8.nbytes + nt_targets.nbytes) / 1e9:.2f} GB")

    # Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Features
    features_path = output_dir / f"{family}_features.npz"
    print(f"\nSauvegarde features: {features_path}")
    np.savez(features_path,
             layer_24=features,
             fold_ids=fold_ids,
             original_indices=original_indices)

    # Targets pré-calculés (NP, HV en int8, NT)
    targets_path = output_dir / f"{family}_targets.npz"
    print(f"Sauvegarde targets: {targets_path}")
    np.savez(targets_path,
             np_targets=np_targets,
             hv_targets=hv_targets_int8,
             nt_targets=nt_targets)

    # Afficher les tailles
    features_size = features_path.stat().st_size / 1e9
    targets_size = targets_path.stat().st_size / 1e9
    print(f"\n✅ Famille {family} préparée:")
    print(f"   Features: {features_size:.2f} GB")
    print(f"   Targets: {targets_size:.2f} GB")
    print(f"   Total: {features_size + targets_size:.2f} GB")

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
