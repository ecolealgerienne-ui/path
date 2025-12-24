#!/usr/bin/env python3
"""
Nettoyage complet pour repartir de zéro.

GARDE:
    - /home/amar/data/PanNuke (données originales)
    - data/family_FIXED/ (targets v4 corrigés)
    - models/pretrained/ (H-optimus-0)

SUPPRIME:
    - models/checkpoints (entraînés avec features corrompues)
    - models/checkpoints_FIXED (entraînés avec features corrompues)
    - data/cache/pannuke_features (features corrompues)
    - results/ (résultats obsolètes)

Usage:
    python scripts/utils/cleanup_start_fresh.py --dry-run  # Voir
    python scripts/utils/cleanup_start_fresh.py            # Supprimer
"""

import argparse
from pathlib import Path
import shutil


def get_dir_size(directory: Path) -> float:
    """Retourne la taille d'un répertoire en MB."""
    if not directory.exists():
        return 0.0

    total_size = 0
    for item in directory.rglob('*'):
        if item.is_file():
            try:
                total_size += item.stat().st_size
            except:
                pass

    return total_size / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Afficher ce qui serait supprimé sans supprimer')
    args = parser.parse_args()

    print("=" * 80)
    print("NETTOYAGE COMPLET - REPARTIR DE ZÉRO")
    print("=" * 80)

    # Répertoires à GARDER
    keep_dirs = {
        'PanNuke Original': Path('/home/amar/data/PanNuke'),
        'Family Data FIXED v4': Path('data/family_FIXED'),
        'Pretrained Models': Path('models/pretrained'),
    }

    # Répertoires à SUPPRIMER
    delete_dirs = {
        'Checkpoints OLD': Path('models/checkpoints'),
        'Checkpoints FIXED': Path('models/checkpoints_FIXED'),
        'Features Cache': Path('data/cache/pannuke_features'),
        'Results': Path('results'),
    }

    # Afficher ce qui est gardé
    print("\n✅ FICHIERS À GARDER:")
    print("-" * 80)

    total_keep = 0
    for name, directory in keep_dirs.items():
        if directory.exists():
            size_mb = get_dir_size(directory)
            total_keep += size_mb
            print(f"   {name:30s} {size_mb:>10.2f} MB")
        else:
            print(f"   {name:30s} {'N/A':>10s} (n'existe pas)")

    print(f"\n   {'TOTAL À GARDER':30s} {total_keep:>10.2f} MB")

    # Afficher ce qui sera supprimé
    print("\n🗑️  FICHIERS À SUPPRIMER:")
    print("-" * 80)

    total_delete = 0
    existing_deletes = []

    for name, directory in delete_dirs.items():
        if directory.exists():
            size_mb = get_dir_size(directory)
            total_delete += size_mb
            existing_deletes.append((name, directory, size_mb))
            print(f"   {name:30s} {size_mb:>10.2f} MB")
        else:
            print(f"   {name:30s} {'0.00':>10s} MB (n'existe pas)")

    print(f"\n   {'TOTAL À SUPPRIMER':30s} {total_delete:>10.2f} MB")

    # Dry-run
    if args.dry_run:
        print("\n⚠️  MODE DRY-RUN: Aucune suppression")
        print(f"   Libération potentielle: {total_delete:.2f} MB")
        return 0

    if not existing_deletes:
        print("\n✅ Rien à supprimer")
        return 0

    # Confirmation
    print("\n" + "=" * 80)
    print("⚠️  ATTENTION: Vous allez SUPPRIMER DÉFINITIVEMENT:")
    print("-" * 80)

    for name, directory, size_mb in existing_deletes:
        print(f"   {directory} ({size_mb:.2f} MB)")

    print("\nRAISON: Repartir de zéro avec données corrigées")
    print("\n✅ GARDÉS:")
    print("   - PanNuke original (/home/amar/data/PanNuke)")
    print("   - Family Data FIXED v4 (data/family_FIXED/)")
    print("   - Pretrained models (models/pretrained/)")

    response = input("\nContinuer ? (tapez 'OUI'): ")

    if response.strip().upper() != "OUI":
        print("\n❌ Annulé")
        return 1

    # Suppression
    print("\n" + "=" * 80)
    print("🗑️  SUPPRESSION EN COURS...")
    print("-" * 80)

    deleted_mb = 0

    for name, directory, size_mb in existing_deletes:
        print(f"\n🗑️  Suppression de {directory}...")
        shutil.rmtree(directory)
        deleted_mb += size_mb
        print(f"   ✅ Supprimé ({size_mb:.2f} MB)")

    # Résumé final
    print("\n" + "=" * 80)
    print("✅ NETTOYAGE TERMINÉ")
    print("=" * 80)
    print(f"\n💾 Espace libéré: {deleted_mb:.2f} MB")
    print(f"📍 Espace restant: {total_keep:.2f} MB")

    print("\n📝 PROCHAINES ÉTAPES:")
    print("   1. Vérifier alignement spatial (data/family_FIXED/)")
    print("   2. SI OK: Régénérer features fold 0 (20 min)")
    print("   3. Re-training epidermal (40 min)")
    print("   4. Test AJI final (attendu: 0.06 → 0.60+)")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
