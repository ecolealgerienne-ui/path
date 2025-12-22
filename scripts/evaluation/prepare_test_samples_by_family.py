#!/usr/bin/env python3
"""
Prépare des échantillons de test par famille pour validation isolée.

Ce script extrait des échantillons de PanNuke Fold 2 (non utilisé pour entraînement)
et les organise par famille pour tester chaque modèle HoVer-Net indépendamment.

Objectif: Isoler le problème
- Si un modèle de famille échoue sur ses propres données → Problème d'entraînement
- Si tous les modèles fonctionnent bien → Problème de routage ou d'intégration
- Si les résultats varient par famille → Problème de données ou de samples

Usage:
    python scripts/evaluation/prepare_test_samples_by_family.py \
        --pannuke_dir /path/to/PanNuke \
        --fold 2 \
        --samples_per_organ 10 \
        --output_dir data/test_samples_by_family
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.organ_families import (
    ORGAN_TO_FAMILY,
    FAMILY_TO_ORGANS,
    FAMILIES,
    get_family
)


def prepare_test_samples(
    pannuke_dir: Path,
    fold: int,
    max_samples: int,
    output_dir: Path
):
    """
    Prépare des échantillons de test organisés par famille.

    Stratégie optimisée:
    - Prend les N premiers échantillons du fold (pas toutes les données)
    - Organise par famille ce qui est trouvé
    - Évite de charger tout le dataset en mémoire

    Args:
        pannuke_dir: Répertoire PanNuke
        fold: Fold à utiliser (typiquement 2 pour test)
        max_samples: Nombre maximum d'échantillons à extraire (ex: 500)
        output_dir: Répertoire de sortie
    """
    print("=" * 70)
    print("PRÉPARATION DES ÉCHANTILLONS DE TEST PAR FAMILLE")
    print("=" * 70)
    print(f"Source: {pannuke_dir}/fold{fold}")
    print(f"Max samples: {max_samples}")
    print(f"Output: {output_dir}")
    print()

    # Charger les données du fold
    fold_dir = pannuke_dir / f"fold{fold}"

    images_path = fold_dir / "images.npy"
    masks_path = fold_dir / "masks.npy"
    types_path = fold_dir / "types.npy"

    # Vérifier existence
    for path in [images_path, masks_path, types_path]:
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant: {path}")

    print("Chargement des données...")
    # Charger UNIQUEMENT les N premiers échantillons
    images_full = np.load(images_path, mmap_mode='r')
    masks_full = np.load(masks_path, mmap_mode='r')
    types_full = np.load(types_path)

    # Limiter au max_samples
    n_available = len(types_full)
    n_to_load = min(max_samples, n_available)

    print(f"Disponibles: {n_available}, Extraction: {n_to_load}")

    # Copier les N premiers en mémoire
    images = images_full[:n_to_load].copy()
    masks = masks_full[:n_to_load].copy()
    types = types_full[:n_to_load]

    print(f"✅ Chargé: {len(types)} images")
    print()

    # Créer structure de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistiques par famille
    family_stats = {family: {"organs": {}, "total": 0} for family in FAMILIES}

    # D'abord grouper par organe
    organ_samples = {}

    print("Organisation par organe...")
    for i in range(len(types)):
        organ = str(types[i]).strip()

        if organ not in organ_samples:
            organ_samples[organ] = []

        sample_data = {
            "image": images[i],
            "mask": masks[i],
            "organ": organ,
            "original_index": int(i),
            "fold": fold,
        }

        organ_samples[organ].append(sample_data)

    # Afficher distribution brute
    print()
    print(f"Organes trouvés dans les {n_to_load} premiers échantillons:")
    for organ, samples in sorted(organ_samples.items()):
        print(f"  {organ:20s}: {len(samples):3d} échantillons")
    print()

    # Sélectionner max 10 par organe
    samples_per_organ = 10
    organ_samples_limited = {}

    for organ, samples in organ_samples.items():
        n_available = len(samples)
        n_to_select = min(samples_per_organ, n_available)

        # Sélection aléatoire
        np.random.seed(42)  # Reproductibilité
        if n_to_select < n_available:
            selected_indices = np.random.choice(
                len(samples),
                size=n_to_select,
                replace=False
            )
            selected_samples = [samples[i] for i in selected_indices]
        else:
            selected_samples = samples

        organ_samples_limited[organ] = selected_samples

    # Grouper par famille
    family_samples_dict = {family: [] for family in FAMILIES}

    print(f"Sélection de max {samples_per_organ} échantillons par organe:")
    for organ, samples in sorted(organ_samples_limited.items()):
        try:
            family = get_family(organ)

            # Ajouter à la famille
            for sample in samples:
                sample["family"] = family
                family_samples_dict[family].append(sample)

            # Statistiques
            if organ not in family_stats[family]["organs"]:
                family_stats[family]["organs"][organ] = 0
            family_stats[family]["organs"][organ] = len(samples)
            family_stats[family]["total"] += len(samples)

            print(f"  {organ:20s} → {family:15s} ({len(samples):2d} échantillons)")

        except ValueError:
            # Organe inconnu, ignorer
            print(f"  ⚠️  {organ:20s} → UNKNOWN (ignoré)")
            continue

    # Afficher statistiques
    print()
    print("Distribution trouvée:")
    for family in FAMILIES:
        n_samples = family_stats[family]["total"]
        if n_samples > 0:
            print(f"  {family:15s}: {n_samples:3d} échantillons")
            for organ, count in sorted(family_stats[family]["organs"].items()):
                print(f"     - {organ:20s}: {count:2d}")
    print()

    # Sauvegarder par famille
    print("Sauvegarde des échantillons...")

    for family in FAMILIES:
        family_samples = family_samples_dict[family]

        if not family_samples:
            print(f"  ⚠️  {family}: Aucun échantillon")
            continue

        print(f"\n{'=' * 70}")
        print(f"FAMILLE: {family.upper()}")
        print(f"{'=' * 70}")

        family_dir = output_dir / family
        family_dir.mkdir(exist_ok=True)

        # Sauvegarder tous les échantillons de la famille
        if family_samples:
            print(f"\n  💾 Sauvegarde de {len(family_samples)} échantillons...")

            # Créer arrays numpy
            n_samples = len(family_samples)
            images_array = np.stack([s["image"] for s in family_samples])
            masks_array = np.stack([s["mask"] for s in family_samples])
            organs = np.array([s["organ"] for s in family_samples])
            indices = np.array([s["original_index"] for s in family_samples])

            # Sauvegarder
            np.savez_compressed(
                family_dir / "test_samples.npz",
                images=images_array,
                masks=masks_array,
                organs=organs,
                indices=indices,
            )

            # Métadonnées JSON
            metadata = {
                "family": family,
                "fold": fold,
                "n_samples": n_samples,
                "organs": family_stats[family]["organs"],
            }

            with open(family_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  ✅ Sauvegardé: {family_dir / 'test_samples.npz'}")
        else:
            print(f"  ⚠️  Aucun échantillon pour cette famille")

    # Rapport final
    print()
    print("=" * 70)
    print("RAPPORT FINAL")
    print("=" * 70)

    total_samples = 0
    for family in FAMILIES:
        n_samples = family_stats[family]["total"]
        total_samples += n_samples

        status = "✅" if n_samples > 0 else "⚠️"
        print(f"{status} {family:15s}: {n_samples:3d} échantillons")

        for organ, count in sorted(family_stats[family]["organs"].items()):
            print(f"     - {organ:20s}: {count:2d}")

    print()
    print(f"Total: {total_samples} échantillons extraits")
    print(f"Sauvegardés dans: {output_dir}")
    print()

    # Sauvegarder rapport global
    global_report = {
        "fold": fold,
        "max_samples_loaded": n_to_load,
        "samples_per_organ": 10,  # Fixé à 10
        "total_samples": total_samples,
        "families": family_stats,
    }

    with open(output_dir / "global_report.json", "w") as f:
        json.dump(global_report, f, indent=2)

    print(f"✅ Rapport global: {output_dir / 'global_report.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Prépare des échantillons de test par famille"
    )
    parser.add_argument(
        "--pannuke_dir",
        type=Path,
        required=True,
        help="Répertoire PanNuke"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=2,
        help="Fold à utiliser (défaut: 2 pour test)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Nombre maximum d'échantillons à charger du fold (défaut: 500)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/test_samples_by_family"),
        help="Répertoire de sortie"
    )

    args = parser.parse_args()

    try:
        prepare_test_samples(
            pannuke_dir=args.pannuke_dir,
            fold=args.fold,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )

        print("=" * 70)
        print("✅ PRÉPARATION TERMINÉE AVEC SUCCÈS")
        print("=" * 70)
        print()
        print("Prochaine étape:")
        print("  python scripts/evaluation/test_family_models_isolated.py \\")
        print(f"      --test_samples_dir {args.output_dir} \\")
        print("      --checkpoint_dir models/checkpoints")
        print()

    except Exception as e:
        print(f"\n❌ ERREUR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
