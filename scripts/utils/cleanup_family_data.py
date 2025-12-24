#!/usr/bin/env python3
"""
Nettoyage des données de famille pour libérer l'espace disque.

Stratégie:
    1. Garder UNIQUEMENT data/family_FIXED/ comme source de vérité
    2. Supprimer tous les fichiers dans data/cache/family_data/
    3. Supprimer tous les fichiers dans data/cache/family_data_FIXED/
    4. Afficher l'espace libéré

Usage:
    python scripts/utils/cleanup_family_data.py --dry-run  # Voir ce qui serait supprimé
    python scripts/utils/cleanup_family_data.py            # Supprimer réellement
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

    return total_size / (1024 * 1024)  # Bytes → MB


def list_files(directory: Path) -> list:
    """Liste tous les fichiers .npz dans un répertoire."""
    if not directory.exists():
        return []

    return list(directory.glob('*.npz'))


def main():
    parser = argparse.ArgumentParser(description="Nettoyer les doublons de données de famille")
    parser.add_argument('--dry-run', action='store_true',
                        help='Afficher ce qui serait supprimé sans supprimer')
    args = parser.parse_args()

    print("=" * 80)
    print("NETTOYAGE DES DONNÉES DE FAMILLE")
    print("=" * 80)

    # Définir les répertoires
    source_of_truth = Path("data/family_FIXED")
    cache_dirs = [
        Path("data/cache/family_data"),
        Path("data/cache/family_data_FIXED"),
    ]

    # Anciens répertoires (autres versions)
    old_dirs = [
        Path("data/cache/family_data_OLD_int8_20251222_163212"),
        Path("data/cache/family_data_OLD_CORRUPTED_20251223"),
    ]

    print(f"\n📍 SOURCE DE VÉRITÉ: {source_of_truth}")
    print(f"   Taille: {get_dir_size(source_of_truth):.2f} MB")
    print(f"   Fichiers: {len(list_files(source_of_truth))}")

    # Analyser les caches
    print("\n" + "=" * 80)
    print("ANALYSE DES CACHES À SUPPRIMER:")
    print("-" * 80)

    total_to_delete_mb = 0

    for cache_dir in cache_dirs + old_dirs:
        if not cache_dir.exists():
            print(f"\n❌ {cache_dir} (n'existe pas)")
            continue

        size_mb = get_dir_size(cache_dir)
        files = list_files(cache_dir)

        if size_mb > 0:
            total_to_delete_mb += size_mb

            print(f"\n📂 {cache_dir}")
            print(f"   Taille: {size_mb:.2f} MB")
            print(f"   Fichiers: {len(files)}")

            if files:
                for f in files:
                    print(f"      - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

    # Résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ:")
    print("-" * 80)
    print(f"\n💾 SOURCE DE VÉRITÉ (à garder): {get_dir_size(source_of_truth):.2f} MB")
    print(f"🗑️  À SUPPRIMER: {total_to_delete_mb:.2f} MB")
    print(f"✅ ESPACE LIBÉRÉ: {total_to_delete_mb:.2f} MB")

    # Confirmation
    if args.dry_run:
        print("\n⚠️  MODE DRY-RUN: Aucune suppression effectuée")
        print("   Relancer sans --dry-run pour supprimer réellement")
        return 0

    print("\n" + "=" * 80)
    print("⚠️  ATTENTION: Vous allez SUPPRIMER définitivement:")
    print("-" * 80)

    for cache_dir in cache_dirs + old_dirs:
        if cache_dir.exists() and get_dir_size(cache_dir) > 0:
            print(f"   - {cache_dir} ({get_dir_size(cache_dir):.2f} MB)")

    response = input("\nContinuer ? (tapez 'OUI' pour confirmer): ")

    if response.strip().upper() != "OUI":
        print("\n❌ Annulé par l'utilisateur")
        return 1

    # Suppression
    print("\n" + "=" * 80)
    print("SUPPRESSION EN COURS...")
    print("-" * 80)

    deleted_mb = 0

    for cache_dir in cache_dirs + old_dirs:
        if not cache_dir.exists():
            continue

        size_mb = get_dir_size(cache_dir)

        if size_mb > 0:
            print(f"\n🗑️  Suppression de {cache_dir}...")
            shutil.rmtree(cache_dir)
            deleted_mb += size_mb
            print(f"   ✅ Supprimé ({size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("✅ NETTOYAGE TERMINÉ")
    print("-" * 80)
    print(f"\n💾 Espace libéré: {deleted_mb:.2f} MB")
    print(f"📍 Source de vérité unique: {source_of_truth}")
    print(f"   {len(list_files(source_of_truth))} fichier(s) conservé(s)")

    print("\n📝 RECOMMANDATION:")
    print("   Modifier verify_spatial_alignment.py pour chercher UNIQUEMENT dans:")
    print(f"   {source_of_truth}")

    return 0


if __name__ == "__main__":
    exit(main())
