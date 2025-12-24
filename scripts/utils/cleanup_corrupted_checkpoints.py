#!/usr/bin/env python3
"""
Supprime les checkpoints entraînés avec features corrompues.

Stratégie:
    1. Supprimer models/checkpoints/ (entraînés AVANT Bug #1/#2 fixes)
    2. Garder models/checkpoints_FIXED/ (entraînés APRÈS fixes)

Usage:
    python scripts/utils/cleanup_corrupted_checkpoints.py --dry-run  # Voir
    python scripts/utils/cleanup_corrupted_checkpoints.py            # Supprimer
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
            total_size += item.stat().st_size

    return total_size / (1024 * 1024)


def list_files(directory: Path) -> list:
    """Liste tous les fichiers dans un répertoire."""
    if not directory.exists():
        return []

    return list(directory.rglob('*'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print("=" * 80)
    print("NETTOYAGE CHECKPOINTS CORROMPUS")
    print("=" * 80)

    corrupted_dir = Path("models/checkpoints")
    fixed_dir = Path("models/checkpoints_FIXED")

    # Analyser
    print(f"\n📂 CHECKPOINTS CORROMPUS: {corrupted_dir}")
    if corrupted_dir.exists():
        size_mb = get_dir_size(corrupted_dir)
        files = [f for f in list_files(corrupted_dir) if f.is_file()]

        print(f"   Taille: {size_mb:.2f} MB")
        print(f"   Fichiers: {len(files)}")

        for f in files:
            print(f"      - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print("   ❌ N'existe pas")
        return 0

    print(f"\n✅ CHECKPOINTS FIXED: {fixed_dir}")
    if fixed_dir.exists():
        size_mb = get_dir_size(fixed_dir)
        files = [f for f in list_files(fixed_dir) if f.is_file()]

        print(f"   Taille: {size_mb:.2f} MB")
        print(f"   Fichiers: {len(files)}")
    else:
        print("   ⚠️  N'existe pas - sera créé après re-training")

    # Dry-run
    if args.dry_run:
        print("\n⚠️  MODE DRY-RUN: Aucune suppression")
        print(f"   Libération potentielle: {get_dir_size(corrupted_dir):.2f} MB")
        return 0

    # Confirmation
    size_to_delete = get_dir_size(corrupted_dir)

    print("\n" + "=" * 80)
    print("⚠️  ATTENTION: Vous allez SUPPRIMER:")
    print("-" * 80)
    print(f"   {corrupted_dir} ({size_to_delete:.2f} MB)")
    print("\nRaison: Entraînés avec features corrompues (Bug #1/#2)")

    response = input("\nContinuer ? (tapez 'OUI'): ")

    if response.strip().upper() != "OUI":
        print("\n❌ Annulé")
        return 1

    # Suppression
    print("\n🗑️  Suppression...")
    shutil.rmtree(corrupted_dir)
    print(f"✅ Supprimé ({size_to_delete:.2f} MB)")

    print("\n" + "=" * 80)
    print("✅ NETTOYAGE TERMINÉ")
    print(f"💾 Espace libéré: {size_to_delete:.2f} MB")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
