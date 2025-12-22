#!/usr/bin/env python3
"""
Script d'audit pour identifier les différences exactes dans le preprocessing.

Usage:
    python scripts/audit/audit_preprocessing.py --output audit_report_code.md
"""

import argparse
import ast
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Fichiers à auditer
FILES_TO_AUDIT = [
    "scripts/demo/gradio_demo.py",
    "scripts/evaluation/compare_train_vs_inference.py",
    "scripts/preprocessing/extract_features.py",
    "scripts/preprocessing/extract_fold_features.py",
    "scripts/preprocessing/prepare_family_data.py",
    "scripts/preprocessing/prepare_family_data_FIXED.py",
    "scripts/validation/diagnose_organ_prediction.py",
    "scripts/validation/test_organ_prediction_batch.py",
    "scripts/validation/verify_features.py",
    "src/inference/cellvit_inference.py",
    "src/inference/cellvit_official.py",
    "src/inference/hoptimus_hovernet.py",
    "src/inference/hoptimus_unetr.py",
    "src/inference/optimus_gate_inference.py",
    "src/inference/optimus_gate_inference_multifamily.py",
]

def extract_constants(filepath: Path) -> Dict[str, any]:
    """Extrait les constantes HOPTIMUS_MEAN, HOPTIMUS_STD."""
    constants = {}

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Chercher HOPTIMUS_MEAN
        if 'HOPTIMUS_MEAN' in content:
            for line in content.split('\n'):
                if 'HOPTIMUS_MEAN' in line and '=' in line:
                    constants['HOPTIMUS_MEAN'] = line.strip()
                    break

        # Chercher HOPTIMUS_STD
        if 'HOPTIMUS_STD' in content:
            for line in content.split('\n'):
                if 'HOPTIMUS_STD' in line and '=' in line:
                    constants['HOPTIMUS_STD'] = line.strip()
                    break

    except Exception as e:
        constants['error'] = str(e)

    return constants

def extract_function_code(filepath: Path, func_names: List[str]) -> Dict[str, str]:
    """Extrait le code des fonctions spécifiées."""
    functions = {}

    try:
        with open(filepath, 'r') as f:
            content = f.read()
            tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in func_names:
                    # Extraire le code de la fonction
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10

                    func_code = '\n'.join(content.split('\n')[start_line-1:end_line])
                    functions[node.name] = func_code

    except Exception as e:
        functions['error'] = str(e)

    return functions

def hash_code(code: str) -> str:
    """Calcule un hash du code (pour détecter duplications exactes)."""
    # Normaliser les espaces pour éviter les faux positifs
    normalized = ' '.join(code.split())
    return hashlib.md5(normalized.encode()).hexdigest()[:8]

def audit_files() -> Dict:
    """Audite tous les fichiers."""
    results = {
        'constants': defaultdict(list),
        'functions': defaultdict(list),
        'duplicates': defaultdict(list),
    }

    for filepath_str in FILES_TO_AUDIT:
        filepath = Path(filepath_str)

        if not filepath.exists():
            print(f"⚠️  SKIP: {filepath} (not found)")
            continue

        print(f"🔍 Auditing: {filepath}")

        # Extraire constantes
        constants = extract_constants(filepath)
        for const_name, const_value in constants.items():
            results['constants'][const_name].append({
                'file': str(filepath),
                'value': const_value
            })

        # Extraire fonctions clés
        func_names = [
            'create_hoptimus_transform',
            'preprocess',
            'preprocess_image',
            'load_hoptimus',
        ]

        functions = extract_function_code(filepath, func_names)
        for func_name, func_code in functions.items():
            if func_name != 'error':
                code_hash = hash_code(func_code)
                results['functions'][func_name].append({
                    'file': str(filepath),
                    'hash': code_hash,
                    'lines': len(func_code.split('\n'))
                })

                # Grouper par hash pour détecter duplications
                results['duplicates'][f"{func_name}_{code_hash}"].append(str(filepath))

    return results

def generate_report(results: Dict, output_path: Path):
    """Génère le rapport markdown."""

    report = []
    report.append("# Rapport d'Audit - Code Preprocessing\n")
    report.append(f"**Date:** 2025-12-22\n")
    report.append(f"**Fichiers auditĂ©s:** {len(FILES_TO_AUDIT)}\n")
    report.append("\n---\n")

    # Section 1: Constantes
    report.append("\n## 1. Constantes de Normalisation\n")

    for const_name, occurrences in results['constants'].items():
        report.append(f"\n### {const_name}\n")
        report.append(f"**Occurrences:** {len(occurrences)}\n\n")

        # Grouper par valeur
        by_value = defaultdict(list)
        for occ in occurrences:
            by_value[occ['value']].append(occ['file'])

        if len(by_value) == 1:
            report.append("✅ **COHÉRENT** - Tous les fichiers utilisent la même valeur\n\n")
        else:
            report.append("❌ **INCOHÉRENT** - Valeurs différentes détectées!\n\n")

        for value, files in by_value.items():
            report.append(f"**Valeur:** `{value}`\n")
            report.append(f"**Fichiers ({len(files)}):**\n")
            for f in files:
                report.append(f"- {f}\n")
            report.append("\n")

    # Section 2: Fonctions
    report.append("\n## 2. Fonctions de Preprocessing\n")

    for func_name, implementations in results['functions'].items():
        if func_name == 'error':
            continue

        report.append(f"\n### `{func_name}()`\n")
        report.append(f"**Implémentations trouvées:** {len(implementations)}\n\n")

        # Grouper par hash
        by_hash = defaultdict(list)
        for impl in implementations:
            by_hash[impl['hash']].append(impl['file'])

        report.append(f"**Versions uniques:** {len(by_hash)}\n\n")

        if len(by_hash) == 1:
            report.append("✅ **CODE IDENTIQUE** dans tous les fichiers\n\n")
        else:
            report.append(f"❌ **{len(by_hash)} VERSIONS DIFFÉRENTES** détectées!\n\n")

        for code_hash, files in by_hash.items():
            report.append(f"**Version {code_hash}** ({len(files)} fichiers):\n")
            for f in files:
                report.append(f"- {f}\n")
            report.append("\n")

    # Section 3: Duplications Exactes
    report.append("\n## 3. Duplications Exactes\n")
    report.append("\nFonctions dupliquées identiques (même code, plusieurs endroits):\n\n")

    exact_duplicates = {k: v for k, v in results['duplicates'].items() if len(v) > 1}

    if exact_duplicates:
        for dup_key, files in exact_duplicates.items():
            func_name = dup_key.split('_')[0]
            report.append(f"### `{func_name}()` - {len(files)} copies exactes\n\n")
            for f in files:
                report.append(f"- {f}\n")
            report.append("\n")
    else:
        report.append("✅ Aucune duplication exacte détectée\n\n")

    # Section 4: Recommandations
    report.append("\n## 4. Recommandations\n")

    # Compter total de duplications
    total_const_files = sum(len(occ) for occ in results['constants'].values())
    total_func_files = sum(len(impl) for impl in results['functions'].values() if isinstance(impl, list))

    report.append(f"\n### Statistiques\n\n")
    report.append(f"- **Constantes dupliquées:** {total_const_files} occurrences\n")
    report.append(f"- **Fonctions dupliquées:** {total_func_files} implémentations\n")
    report.append(f"- **Duplications exactes:** {len(exact_duplicates)} fonctions\n\n")

    report.append("### Actions Prioritaires\n\n")
    report.append("1. **Créer `src/preprocessing/__init__.py`**\n")
    report.append("   - Centraliser HOPTIMUS_MEAN, HOPTIMUS_STD\n")
    report.append("   - Centraliser create_hoptimus_transform()\n")
    report.append("   - Centraliser preprocess_image()\n\n")

    report.append("2. **Mettre à jour tous les fichiers**\n")
    report.append("   - Remplacer les constantes locales par imports\n")
    report.append("   - Remplacer les fonctions locales par imports\n\n")

    report.append("3. **Ajouter tests de cohérence**\n")
    report.append("   - Vérifier que tous les fichiers utilisent le même preprocessing\n\n")

    # Écrire le rapport
    with open(output_path, 'w') as f:
        f.write(''.join(report))

    print(f"\n✅ Rapport généré: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Audit des fichiers de preprocessing")
    parser.add_argument('--output', default='audit_report_code.md', help='Fichier de sortie')
    args = parser.parse_args()

    print("=" * 60)
    print("AUDIT DES FICHIERS DE PREPROCESSING")
    print("=" * 60)
    print()

    results = audit_files()

    output_path = Path(args.output)
    generate_report(results, output_path)

    print("\n" + "=" * 60)
    print("AUDIT TERMINÉ")
    print("=" * 60)

if __name__ == "__main__":
    main()
