#!/usr/bin/env python3
"""
Script d'audit pour analyser la structure des données et identifier les duplications.

Usage:
    python scripts/audit/audit_data.py --output audit_report_data.md
"""

import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import subprocess
from src.constants import DEFAULT_FAMILY_DATA_DIR, DEFAULT_FAMILY_FIXED_DIR

def get_directory_size(path: Path) -> int:
    """Calcule la taille totale d'un répertoire en bytes."""
    if not path.exists():
        return 0

    total = 0
    try:
        result = subprocess.run(
            ['du', '-sb', str(path)],
            capture_output=True,
            text=True,
            check=True
        )
        total = int(result.stdout.split()[0])
    except Exception as e:
        # Fallback: compter manuellement
        for item in path.rglob('*'):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except:
                    pass

    return total

def format_size(bytes: int) -> str:
    """Formate la taille en unités lisibles."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"

def scan_directory(path: Path, max_depth: int = 3) -> Dict:
    """Scanne un répertoire et collecte les métadonnées."""
    if not path.exists():
        return {'exists': False}

    result = {
        'exists': True,
        'path': str(path),
        'size_bytes': get_directory_size(path),
        'size_human': None,
        'files': [],
        'subdirs': {},
        'file_types': defaultdict(lambda: {'count': 0, 'size': 0}),
    }

    result['size_human'] = format_size(result['size_bytes'])

    # Lister les fichiers
    try:
        for item in path.iterdir():
            if item.is_file():
                file_size = item.stat().st_size
                file_ext = item.suffix or 'no_ext'

                result['files'].append({
                    'name': item.name,
                    'size': file_size,
                    'size_human': format_size(file_size),
                    'ext': file_ext,
                })

                result['file_types'][file_ext]['count'] += 1
                result['file_types'][file_ext]['size'] += file_size

            elif item.is_dir() and max_depth > 0:
                result['subdirs'][item.name] = scan_directory(item, max_depth - 1)

    except PermissionError:
        result['error'] = 'Permission denied'

    return result

def find_duplicate_files(paths: List[Path]) -> Dict[str, List[str]]:
    """Trouve les fichiers dupliqués par hash MD5."""
    file_hashes = defaultdict(list)

    for path in paths:
        if not path.exists():
            continue

        for item in path.rglob('*'):
            if item.is_file() and item.stat().st_size > 0:
                try:
                    # Calculer hash seulement pour fichiers < 500MB
                    if item.stat().st_size < 500 * 1024 * 1024:
                        with open(item, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            file_hashes[file_hash].append(str(item))
                except Exception:
                    pass

    # Garder seulement les duplications (> 1 fichier)
    duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}

    return duplicates

def audit_data_structure() -> Dict:
    """Audite la structure complète des données."""

    # Répertoires à auditer
    data_dirs = {
        'data_root': Path('data'),
        'cache': Path('data/cache'),
        'pannuke_features': Path('data/cache/pannuke_features'),
        'family_data': Path(DEFAULT_FAMILY_DATA_DIR),
        'family_data_FIXED': Path(DEFAULT_FAMILY_FIXED_DIR),
        'evaluation': Path('data/evaluation'),
        'samples': Path('data/samples'),
        'snapshots': Path('data/snapshots'),
        'feedback': Path('data/feedback'),
        'models_pretrained': Path('models/pretrained'),
        'models_checkpoints': Path('models/checkpoints'),
        'models_checkpoints_FIXED': Path('models/checkpoints_FIXED'),
        'results': Path('results'),
    }

    results = {}

    print("🔍 Scanning data directories...\n")

    for name, path in data_dirs.items():
        print(f"   Scanning {name}: {path}")
        results[name] = scan_directory(path, max_depth=2)

    return results

def generate_report(results: Dict, output_path: Path):
    """Génère le rapport markdown."""

    report = []
    report.append("# Rapport d'Audit - Données\n")
    report.append(f"**Date:** 2025-12-22\n\n")
    report.append("---\n")

    # Section 1: Vue d'ensemble
    report.append("\n## 1. Vue d'Ensemble\n\n")

    total_size = 0
    dirs_exist = 0
    dirs_total = len(results)

    for name, data in results.items():
        if data['exists']:
            dirs_exist += 1
            total_size += data['size_bytes']

    report.append(f"**Répertoires existants:** {dirs_exist}/{dirs_total}\n")
    report.append(f"**Espace disque total:** {format_size(total_size)}\n\n")

    # Section 2: Détails par répertoire
    report.append("\n## 2. Détails par Répertoire\n")

    # Trier par taille décroissante
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['size_bytes'] if x[1]['exists'] else 0,
        reverse=True
    )

    for name, data in sorted_results:
        if not data['exists']:
            report.append(f"\n### `{name}` ❌ N'existe pas\n\n")
            continue

        report.append(f"\n### `{name}`\n\n")
        report.append(f"**Chemin:** `{data['path']}`\n")
        report.append(f"**Taille:** {data['size_human']} ({data['size_bytes']:,} bytes)\n")
        report.append(f"**Fichiers:** {len(data['files'])}\n\n")

        # Types de fichiers
        if data['file_types']:
            report.append("**Types de fichiers:**\n\n")
            report.append("| Extension | Nombre | Taille Totale |\n")
            report.append("|-----------|--------|---------------|\n")

            sorted_types = sorted(
                data['file_types'].items(),
                key=lambda x: x[1]['size'],
                reverse=True
            )

            for ext, stats in sorted_types[:10]:  # Top 10
                report.append(f"| `{ext}` | {stats['count']} | {format_size(stats['size'])} |\n")

            report.append("\n")

        # Sous-répertoires
        if data['subdirs']:
            report.append("**Sous-répertoires:**\n\n")
            for subdir_name, subdir_data in data['subdirs'].items():
                if subdir_data['exists']:
                    report.append(f"- `{subdir_name}/`: {subdir_data['size_human']}\n")

            report.append("\n")

    # Section 3: Analyse des duplications
    report.append("\n## 3. Analyse des Duplications\n\n")

    # Identifier les répertoires suspects
    report.append("### Répertoires Suspects de Duplication\n\n")

    if results['family_data']['exists'] and results['family_data_FIXED']['exists']:
        size_old = results['family_data']['size_bytes']
        size_new = results['family_data_FIXED']['size_bytes']
        total_dup = size_old + size_new

        report.append(f"⚠️  **family_data vs family_FIXED**\n")
        report.append(f"- `family_data/`: {format_size(size_old)}\n")
        report.append(f"- `family_FIXED/`: {format_size(size_new)}\n")
        report.append(f"- **Total potentiellement dupliqué:** {format_size(total_dup)}\n\n")
        report.append(f"💡 **Recommandation:** Supprimer l'ancien répertoire si FIXED est validé\n\n")

    if results['models_checkpoints']['exists'] and results['models_checkpoints_FIXED']['exists']:
        size_old = results['models_checkpoints']['size_bytes']
        size_new = results['models_checkpoints_FIXED']['size_bytes']
        total_dup = size_old + size_new

        report.append(f"⚠️  **checkpoints vs checkpoints_FIXED**\n")
        report.append(f"- `checkpoints/`: {format_size(size_old)}\n")
        report.append(f"- `checkpoints_FIXED/`: {format_size(size_new)}\n")
        report.append(f"- **Total potentiellement dupliqué:** {format_size(total_dup)}\n\n")
        report.append(f"💡 **Recommandation:** Archiver ou supprimer les anciens checkpoints\n\n")

    # Section 4: Recommandations
    report.append("\n## 4. Recommandations de Nettoyage\n\n")

    report.append("### Actions Immédiates\n\n")

    report.append("1. **Supprimer les duplications validées**\n")
    report.append("   ```bash\n")
    report.append("   # Si family_FIXED est validé\n")
    report.append("   rm -rf data/family_data\n")
    report.append("   mv data/family_FIXED data/family_data\n\n")
    report.append("   # Si checkpoints_FIXED est validé\n")
    report.append("   rm -rf models/checkpoints\n")
    report.append("   mv models/checkpoints_FIXED models/checkpoints\n")
    report.append("   ```\n\n")

    report.append("2. **Centraliser les données pré-extraites**\n")
    report.append("   - Créer `data/preprocessed/` pour TOUTES les features H-optimus-0\n")
    report.append("   - Structure:\n")
    report.append("     ```\n")
    report.append("     data/preprocessed/\n")
    report.append("     ├── pannuke_features/  ← Features H-optimus-0\n")
    report.append("     ├── family_data/       ← Targets NP/HV/NT par famille\n")
    report.append("     └── metadata.json      ← Versions, hashes, dates\n")
    report.append("     ```\n\n")

    report.append("3. **Versioning des données**\n")
    report.append("   - Ajouter `metadata.json` dans chaque cache:\n")
    report.append("     ```json\n")
    report.append("     {\n")
    report.append('       "version": "2025-12-21-FIXED",\n')
    report.append('       "backbone": "H-optimus-0",\n')
    report.append('       "preprocessing": "forward_features_with_layernorm",\n')
    report.append('       "created_at": "2025-12-21T10:30:00",\n')
    report.append('       "num_samples": 7900,\n')
    report.append('       "hash": "a1b2c3d4"\n')
    report.append("     }\n")
    report.append("     ```\n\n")

    report.append("4. **Pipeline de génération unique**\n")
    report.append("   - Script `scripts/preprocessing/generate_all_data.py`:\n")
    report.append("     1. Extrait features H-optimus-0 (une fois)\n")
    report.append("     2. Génère family_data (une fois)\n")
    report.append("     3. Sauvegarde metadata\n")
    report.append("     4. Tous les scripts utilisent ces données\n\n")

    report.append("### Estimation d'Économie d'Espace\n\n")

    # Calculer les économies potentielles
    savings = 0
    if results['family_data']['exists'] and results['family_data_FIXED']['exists']:
        savings += min(results['family_data']['size_bytes'], results['family_data_FIXED']['size_bytes'])

    if results['models_checkpoints']['exists'] and results['models_checkpoints_FIXED']['exists']:
        savings += min(results['models_checkpoints']['size_bytes'], results['models_checkpoints_FIXED']['size_bytes'])

    if savings > 0:
        report.append(f"**Économie potentielle estimée:** {format_size(savings)}\n\n")

    # Écrire le rapport
    with open(output_path, 'w') as f:
        f.write(''.join(report))

    print(f"\n✅ Rapport généré: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Audit de la structure des données")
    parser.add_argument('--output', default='audit_report_data.md', help='Fichier de sortie')
    args = parser.parse_args()

    print("=" * 60)
    print("AUDIT DE LA STRUCTURE DES DONNÉES")
    print("=" * 60)
    print()

    results = audit_data_structure()

    output_path = Path(args.output)
    generate_report(results, output_path)

    print("\n" + "=" * 60)
    print("AUDIT TERMINÉ")
    print("=" * 60)

if __name__ == "__main__":
    main()
