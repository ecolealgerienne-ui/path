#!/usr/bin/env python3
"""
Rapport complet d'utilisation disque du projet.

Scanne tous les répertoires pour identifier les gros fichiers et doublons.
Focus sur les fichiers originaux PanNuke et les caches générés.

Usage:
    python scripts/utils/disk_usage_report.py
"""

from pathlib import Path
import hashlib
from collections import defaultdict


def compute_file_hash(filepath: Path) -> str:
    """Calcule le hash MD5 d'un fichier."""
    md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()
    except:
        return "ERROR"


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


def format_size(size_bytes: int) -> str:
    """Formate une taille en bytes vers une unité lisible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def scan_directory(directory: Path, extensions: list = None) -> list:
    """
    Scanne un répertoire et retourne la liste des fichiers avec leurs infos.

    Args:
        directory: Répertoire à scanner
        extensions: Liste d'extensions à filtrer (ex: ['.npz', '.pth'])

    Returns:
        Liste de dicts: {'path': Path, 'size': int, 'hash': str}
    """
    if not directory.exists():
        return []

    files = []

    for item in directory.rglob('*'):
        if not item.is_file():
            continue

        if extensions:
            if item.suffix not in extensions:
                continue

        try:
            size = item.stat().st_size
            files.append({
                'path': item,
                'size': size,
                'hash': None  # Calculé à la demande pour les gros fichiers
            })
        except:
            pass

    return files


def main():
    import argparse
from src.constants import DEFAULT_FAMILY_FIXED_DIR

    parser = argparse.ArgumentParser(description="Rapport d'utilisation disque")
    parser.add_argument('--min-size', type=int, default=10,
                        help='Taille minimale en MB pour afficher les fichiers (défaut: 10)')
    args = parser.parse_args()

    min_size_mb = args.min_size
    min_size_bytes = min_size_mb * 1024 * 1024

    print("=" * 80)
    print(f"RAPPORT D'UTILISATION DISQUE (> {min_size_mb} MB)")
    print("=" * 80)

    # Définir les répertoires à scanner (UNIQUEMENT fichiers générés)
    directories = {
        'Features Cache': Path('data/cache/pannuke_features'),
        'Family Data FIXED': Path(DEFAULT_FAMILY_FIXED_DIR),
        'Checkpoints': Path('models/checkpoints'),
        'Checkpoints FIXED': Path('models/checkpoints_FIXED'),
        'Pretrained': Path('models/pretrained'),
        'Results': Path('results'),
        'Data Cache (autres)': Path('data/cache'),
    }

    # Scanner tous les répertoires
    total_usage = {}
    file_details = {}

    print("\n📊 SCAN EN COURS...\n")

    for name, directory in directories.items():
        if not directory.exists():
            print(f"❌ {name}: {directory} (n'existe pas)")
            continue

        size_mb = get_dir_size(directory)
        total_usage[name] = size_mb

        # Pour les gros répertoires, lister les fichiers
        if name in ['Features Cache', 'Family Data FIXED', 'Checkpoints', 'Checkpoints FIXED']:
            files = scan_directory(directory, extensions=['.npz', '.npy', '.pth'])
            file_details[name] = files

        print(f"✅ {name}: {size_mb:.2f} MB")

    # Rapport par catégorie
    print("\n" + "=" * 80)
    print("USAGE PAR CATÉGORIE:")
    print("-" * 80)

    # Trier par taille décroissante
    sorted_usage = sorted(total_usage.items(), key=lambda x: x[1], reverse=True)

    total_all = sum(total_usage.values())

    for name, size_mb in sorted_usage:
        percentage = (size_mb / total_all * 100) if total_all > 0 else 0
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{name:25s} {bar} {size_mb:>10.2f} MB ({percentage:>5.1f}%)")

    print("-" * 80)
    print(f"{'TOTAL':25s} {'':50s} {total_all:>10.2f} MB (100.0%)")

    # Détails Features Cache
    if 'Features Cache' in file_details:
        print("\n" + "=" * 80)
        print(f"DÉTAILS: Features Cache (> {min_size_mb} MB)")
        print("-" * 80)

        files = file_details['Features Cache']
        if files:
            # Filtrer uniquement fichiers > min_size
            big_files = [f for f in files if f['size'] > min_size_bytes]
            if big_files:
                for f in sorted(big_files, key=lambda x: x['size'], reverse=True):
                    print(f"   {f['path'].name:40s} {format_size(f['size']):>12s}")
            else:
                print(f"   (Aucun fichier > {min_size_mb} MB)")

            # Analyser les doublons (même taille = suspect)
            size_groups = defaultdict(list)
            for f in files:
                size_groups[f['size']].append(f)

            duplicates = {size: files for size, files in size_groups.items() if len(files) > 1}
            if duplicates:
                print("\n⚠️  DOUBLONS POTENTIELS (même taille):")
                for size, dup_files in duplicates.items():
                    print(f"\n   Taille: {format_size(size)}")
                    for f in dup_files:
                        print(f"      - {f['path']}")

    # Détails Family Data FIXED
    if 'Family Data FIXED' in file_details:
        print("\n" + "=" * 80)
        print(f"DÉTAILS: Family Data FIXED (> {min_size_mb} MB)")
        print("-" * 80)

        files = file_details['Family Data FIXED']
        if files:
            # Filtrer uniquement fichiers > min_size
            big_files = [f for f in files if f['size'] > min_size_bytes]
            if big_files:
                for f in sorted(big_files, key=lambda x: x['size'], reverse=True):
                    print(f"   {f['path'].name:40s} {format_size(f['size']):>12s}")
            else:
                print(f"   (Aucun fichier > {min_size_mb} MB)")

    # Recherche de gros fichiers
    print("\n" + "=" * 80)
    print(f"GROS FICHIERS (> {min_size_mb} MB):")
    print("-" * 80)

    big_files = []
    for name, files in file_details.items():
        for f in files:
            if f['size'] > min_size_bytes:
                big_files.append((name, f))

    if big_files:
        for category, f in sorted(big_files, key=lambda x: x[1]['size'], reverse=True):
            print(f"{category:25s} {f['path'].name:40s} {format_size(f['size']):>12s}")
    else:
        print(f"   (Aucun fichier > {min_size_mb} MB)")

    # Recommandations
    print("\n" + "=" * 80)
    print("RECOMMANDATIONS:")
    print("-" * 80)

    # Features cache
    features_size = total_usage.get('Features Cache', 0)
    if features_size > 0:
        print(f"\n💾 Features Cache ({features_size:.2f} MB):")
        print("   ❌ PEUT ÊTRE SUPPRIMÉ si corrompues (Bug #1/#2)")
        print("   ✅ À régénérer avec preprocessing corrigé")
        print(f"   Libération: {features_size:.2f} MB")

    # Family Data FIXED
    family_size = total_usage.get('Family Data FIXED', 0)
    if family_size > 0:
        print(f"\n✅ Family Data FIXED ({family_size:.2f} MB):")
        print("   ✅ GARDER - Source de vérité après Bug #3/#4/#5 fixes")

    # Checkpoints
    checkpoints_size = total_usage.get('Checkpoints', 0)
    checkpoints_fixed_size = total_usage.get('Checkpoints FIXED', 0)
    if checkpoints_size > 0 and checkpoints_fixed_size > 0:
        print(f"\n🔧 Checkpoints:")
        print(f"   models/checkpoints ({checkpoints_size:.2f} MB) ❌ Peut être supprimé (entraînés avec features corrompues)")
        print(f"   models/checkpoints_FIXED ({checkpoints_fixed_size:.2f} MB) ✅ GARDER")

    print("\n" + "=" * 80)
    print(f"💾 ESPACE TOTAL UTILISÉ: {total_all:.2f} MB ({total_all/1024:.2f} GB)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
