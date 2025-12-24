#!/usr/bin/env python3
"""
Migre automatiquement tous les scripts vers les paths centralisés.

Remplace:
- DEFAULT_FAMILY_DATA_DIR → DEFAULT_FAMILY_DATA_DIR
- DEFAULT_FAMILY_DATA_DIR → DEFAULT_FAMILY_DATA_DIR
- DEFAULT_FAMILY_FIXED_DIR → DEFAULT_FAMILY_FIXED_DIR

Ajoute l'import nécessaire si manquant.
"""

import argparse
import re
from pathlib import Path
import sys

def migrate_script(script_path: Path, dry_run: bool = True) -> dict:
    """
    Migre un script vers les paths centralisés.

    Returns:
        {
            "path": str,
            "modified": bool,
            "replacements": int,
            "imports_added": bool
        }
    """
    content = script_path.read_text()
    original = content

    # Patterns à remplacer
    patterns = [
        (r'DEFAULT_FAMILY_DATA_DIR', 'DEFAULT_FAMILY_DATA_DIR'),
        (r'DEFAULT_FAMILY_DATA_DIR', 'DEFAULT_FAMILY_DATA_DIR'),
        (r'DEFAULT_FAMILY_FIXED_DIR', 'DEFAULT_FAMILY_FIXED_DIR'),
        (r"DEFAULT_FAMILY_DATA_DIR", 'DEFAULT_FAMILY_DATA_DIR'),
        (r"DEFAULT_FAMILY_DATA_DIR", 'DEFAULT_FAMILY_DATA_DIR'),
        (r"DEFAULT_FAMILY_FIXED_DIR", 'DEFAULT_FAMILY_FIXED_DIR'),
    ]

    replacements = 0
    for pattern, replacement in patterns:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            replacements += len(matches)

    if replacements == 0:
        return {
            "path": str(script_path),
            "modified": False,
            "replacements": 0,
            "imports_added": False
        }

    # Vérifier si les imports sont présents
    has_constants_import = (
        "from src.constants import" in content or
        "import src.constants" in content
    )

    imports_added = False
    if not has_constants_import:
        # Trouver où insérer l'import (après les autres imports src.*)
        lines = content.split('\n')
        insert_pos = None

        # Chercher la dernière ligne d'import src.*
        for i, line in enumerate(lines):
            if line.strip().startswith("from src.") or line.strip().startswith("import src."):
                insert_pos = i + 1

        # Si pas d'import src.*, chercher après sys.path.insert
        if insert_pos is None:
            for i, line in enumerate(lines):
                if "sys.path.insert" in line:
                    insert_pos = i + 1
                    break

        # Si toujours rien, chercher après les imports stdlib
        if insert_pos is None:
            for i, line in enumerate(lines):
                if line.strip().startswith("import ") or line.strip().startswith("from "):
                    insert_pos = i + 1

        if insert_pos is not None:
            # Construire la liste des imports nécessaires
            needed_imports = set()
            if "DEFAULT_FAMILY_DATA_DIR" in content:
                needed_imports.add("DEFAULT_FAMILY_DATA_DIR")
            if "DEFAULT_FAMILY_FIXED_DIR" in content:
                needed_imports.add("DEFAULT_FAMILY_FIXED_DIR")

            import_line = f"from src.constants import {', '.join(sorted(needed_imports))}"
            lines.insert(insert_pos, import_line)
            content = '\n'.join(lines)
            imports_added = True

    # Sauvegarder si pas dry-run
    if not dry_run:
        script_path.write_text(content)

    return {
        "path": str(script_path),
        "modified": True,
        "replacements": replacements,
        "imports_added": imports_added,
        "changes": len(content) - len(original)
    }

def main():
    parser = argparse.ArgumentParser(description="Migre scripts vers paths centralisés")
    parser.add_argument("--dry-run", action="store_true", help="Afficher seulement, ne pas modifier")
    parser.add_argument("--scripts-dir", default="scripts", help="Répertoire scripts")
    args = parser.parse_args()

    scripts_dir = Path(args.scripts_dir)

    # Trouver tous les scripts Python avec paths hardcodés
    all_scripts = list(scripts_dir.rglob("*.py"))

    print("=" * 80)
    print(f"MIGRATION VERS PATHS CENTRALISÉS")
    print("=" * 80)
    print("")
    print(f"Dry-run: {args.dry_run}")
    print(f"Scripts à analyser: {len(all_scripts)}")
    print("")

    results = []
    for script_path in all_scripts:
        result = migrate_script(script_path, dry_run=args.dry_run)
        if result["modified"]:
            results.append(result)

    # Afficher résultats
    print("")
    print("=" * 80)
    print(f"RÉSULTATS: {len(results)} scripts modifiés")
    print("=" * 80)
    print("")

    for result in results:
        print(f"✓ {result['path']}")
        print(f"  Remplacements: {result['replacements']}")
        if result['imports_added']:
            print(f"  ✅ Import ajouté: from src.constants import ...")
        print("")

    if args.dry_run:
        print("⚠️  DRY-RUN MODE - Aucun fichier modifié")
        print("   Lancez sans --dry-run pour appliquer les changements")
    else:
        print(f"✅ {len(results)} scripts migrés avec succès!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
