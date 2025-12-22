#!/usr/bin/env python3
"""
Script pour identifier les fichiers de données redondants.

Analyse le répertoire du projet pour trouver :
- Fichiers .npz/.npy dupliqués
- Anciens fichiers de features
- Données temporaires
- Checkpoints obsolètes

Aide à libérer de l'espace disque.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import hashlib


def get_file_size_mb(path: Path) -> float:
    """Retourne la taille du fichier en MB."""
    return path.stat().st_size / (1024 * 1024)


def get_file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Calcule le hash MD5 d'un fichier (pour détecter duplicates exacts)."""
    md5 = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return None


def scan_directory(root_dir: Path, extensions: list) -> dict:
    """
    Scanne un répertoire et retourne des statistiques.

    Returns:
        {
            'files': [(path, size_mb, mtime), ...],
            'total_size_mb': float,
            'by_extension': {'.npz': size_mb, ...},
            'duplicates': {hash: [paths], ...}
        }
    """
    files = []
    by_extension = defaultdict(float)
    by_hash = defaultdict(list)

    for ext in extensions:
        for path in root_dir.rglob(f"*{ext}"):
            if path.is_file():
                size_mb = get_file_size_mb(path)
                mtime = path.stat().st_mtime

                files.append((path, size_mb, mtime))
                by_extension[ext] += size_mb

                # Calculer hash seulement pour fichiers < 500 MB (trop lent sinon)
                if size_mb < 500:
                    file_hash = get_file_hash(path)
                    if file_hash:
                        by_hash[file_hash].append(path)

    # Filtrer pour garder seulement les vrais duplicates
    duplicates = {h: paths for h, paths in by_hash.items() if len(paths) > 1}

    return {
        'files': sorted(files, key=lambda x: x[1], reverse=True),  # Tri par taille
        'total_size_mb': sum(by_extension.values()),
        'by_extension': dict(by_extension),
        'duplicates': duplicates,
    }


def identify_redundant_patterns(files: list) -> dict:
    """
    Identifie les patterns de fichiers redondants basés sur les noms.

    Patterns recherchés :
    - *_OLD_*.npz
    - *_backup_*.npz
    - *_test_*.npz
    - Fold0 vs fold_0 (casse différente)
    """
    patterns = {
        'old_versions': [],      # *_OLD_*
        'backups': [],           # *_backup_*, *_bak_*
        'test_files': [],        # *_test_*, *_debug_*
        'temp_files': [],        # *_temp_*, *_tmp_*
        'layer_24_legacy': [],   # Anciens fichiers avec clé 'layer_24'
    }

    for path, size_mb, mtime in files:
        name_lower = path.name.lower()

        if '_old_' in name_lower or name_lower.startswith('old_'):
            patterns['old_versions'].append((path, size_mb))

        if '_backup' in name_lower or '_bak' in name_lower:
            patterns['backups'].append((path, size_mb))

        if '_test' in name_lower or '_debug' in name_lower:
            patterns['test_files'].append((path, size_mb))

        if '_temp' in name_lower or '_tmp' in name_lower:
            patterns['temp_files'].append((path, size_mb))

        # Vérifier si c'est un ancien fichier de features avec clé 'layer_24'
        if path.suffix == '.npz' and 'features' in name_lower:
            # On ne peut pas facilement vérifier les clés sans charger le fichier
            # Donc on ne marque que ceux avec des noms suspects
            if 'layer' in name_lower or path.parent.name == 'pannuke_features_old':
                patterns['layer_24_legacy'].append((path, size_mb))

    return patterns


def print_report(stats: dict, patterns: dict, root_dir: Path):
    """Affiche un rapport détaillé."""
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║              DIAGNOSTIC DES FICHIERS DE DONNÉES REDONDANTS              ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"Répertoire scanné: {root_dir}")
    print()

    # Vue d'ensemble
    print("=" * 80)
    print("VUE D'ENSEMBLE")
    print("=" * 80)
    print(f"Nombre total de fichiers: {len(stats['files'])}")
    print(f"Espace disque utilisé: {stats['total_size_mb']:.1f} MB ({stats['total_size_mb']/1024:.2f} GB)")
    print()

    print("Par extension:")
    for ext, size_mb in sorted(stats['by_extension'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext:8s}: {size_mb:8.1f} MB ({size_mb/1024:6.2f} GB)")
    print()

    # Top 10 fichiers les plus volumineux
    print("=" * 80)
    print("TOP 10 FICHIERS LES PLUS VOLUMINEUX")
    print("=" * 80)
    for i, (path, size_mb, mtime) in enumerate(stats['files'][:10], 1):
        rel_path = path.relative_to(root_dir)
        print(f"{i:2d}. {size_mb:8.1f} MB - {rel_path}")
    print()

    # Duplicates exacts
    if stats['duplicates']:
        print("=" * 80)
        print("DUPLICATES EXACTS (même contenu)")
        print("=" * 80)
        total_wasted = 0
        for file_hash, paths in stats['duplicates'].items():
            size_mb = get_file_size_mb(paths[0])
            wasted_mb = size_mb * (len(paths) - 1)
            total_wasted += wasted_mb

            print(f"\n🔄 {len(paths)} copies identiques ({size_mb:.1f} MB chacune, {wasted_mb:.1f} MB gaspillés):")
            for path in paths:
                rel_path = path.relative_to(root_dir)
                print(f"   - {rel_path}")

        print(f"\n💾 Espace récupérable: {total_wasted:.1f} MB ({total_wasted/1024:.2f} GB)")
        print()

    # Patterns de redondance
    print("=" * 80)
    print("PATTERNS DE REDONDANCE")
    print("=" * 80)

    total_pattern_size = 0

    for pattern_name, files in patterns.items():
        if files:
            size_mb = sum(size for _, size in files)
            total_pattern_size += size_mb

            labels = {
                'old_versions': '🗑️  Anciennes versions (*_OLD_*)',
                'backups': '💾 Backups (*_backup_*, *_bak_*)',
                'test_files': '🧪 Fichiers de test',
                'temp_files': '⏳ Fichiers temporaires',
                'layer_24_legacy': '🔧 Features legacy (clé layer_24)',
            }

            print(f"\n{labels[pattern_name]}")
            print(f"   Nombre: {len(files)}, Taille totale: {size_mb:.1f} MB")

            # Afficher les 5 plus gros
            for path, file_size_mb in sorted(files, key=lambda x: x[1], reverse=True)[:5]:
                rel_path = path.relative_to(root_dir)
                print(f"     - {file_size_mb:7.1f} MB: {rel_path}")

            if len(files) > 5:
                print(f"     ... et {len(files) - 5} autres fichiers")

    if total_pattern_size > 0:
        print(f"\n💾 Espace récupérable (patterns): {total_pattern_size:.1f} MB ({total_pattern_size/1024:.2f} GB)")
    print()

    # Recommandations
    print("=" * 80)
    print("RECOMMANDATIONS")
    print("=" * 80)
    print()

    if stats['duplicates'] or total_pattern_size > 0:
        print("✅ NETTOYAGE RECOMMANDÉ:")
        print()

        if stats['duplicates']:
            print("1. Supprimer les duplicates exacts:")
            print("   for file_hash, paths in duplicates.items():")
            print("       # Garder le premier, supprimer les autres")
            print("       for path in paths[1:]:")
            print("           path.unlink()")
            print()

        if patterns['old_versions']:
            print("2. Supprimer les anciennes versions (*_OLD_*):")
            print("   Ces fichiers sont généralement des backups automatiques")
            print()

        if patterns['backups']:
            print("3. Supprimer les backups manuels (si les données principales existent)")
            print()

        if patterns['test_files']:
            print("4. Supprimer les fichiers de test (après validation)")
            print()

        if patterns['temp_files']:
            print("5. Supprimer les fichiers temporaires")
            print()

        if patterns['layer_24_legacy']:
            print("6. Migrer ou supprimer les anciennes features (clé 'layer_24' → 'features')")
            print()

        print("⚠️  ATTENTION: Toujours faire un backup avant de supprimer !")
        print()

    else:
        print("✅ Aucun fichier redondant détecté.")
        print("   Le répertoire semble propre.")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Identifier les fichiers de données redondants")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Répertoire racine à scanner (défaut: répertoire courant)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".npz", ".npy", ".pth"],
        help="Extensions à scanner (défaut: .npz .npy .pth)"
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()

    if not root_dir.exists():
        print(f"❌ ERREUR: {root_dir} introuvable")
        sys.exit(1)

    print(f"🔍 Scanning {root_dir}...")
    print(f"   Extensions: {', '.join(args.extensions)}")
    print()

    stats = scan_directory(root_dir, args.extensions)
    patterns = identify_redundant_patterns(stats['files'])

    print_report(stats, patterns, root_dir)


if __name__ == "__main__":
    main()
