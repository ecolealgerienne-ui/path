#!/usr/bin/env python3
"""
Migration automatique: sigmoid → softmax pour NP predictions.

Le décodeur HoVer-Net produit 2 canaux pour NP (background/foreground)
avec CrossEntropyLoss, pas 1 canal avec BCELoss.

Il faut donc utiliser softmax[:, 1] au lieu de sigmoid[:, 0].

Usage:
    python scripts/utils/migrate_np_predictions.py --dry-run
    python scripts/utils/migrate_np_predictions.py
"""

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Pattern à remplacer
PATTERNS = [
    # Pattern 1: torch.sigmoid(np_out).cpu().numpy()[0, 0]
    {
        "old": r'torch\.sigmoid\(np_out\)\.cpu\(\)\.numpy\(\)\[0, 0\]',
        "new": 'torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]',
        "comment": "  # Channel 1 = foreground (after softmax)",
    },
    # Pattern 2: torch.sigmoid(np_logits[:, 1, :, :])
    # Ce pattern est CORRECT (prend déjà le bon canal), pas besoin de changer

    # Pattern 3: torch.sigmoid(np_out).cpu().numpy()[0]
    {
        "old": r'torch\.sigmoid\(np_out\)\.cpu\(\)\.numpy\(\)\[0\]',
        "new": 'torch.softmax(np_out, dim=1).cpu().numpy()[0]',
        "comment": "  # Softmax over channels (0=bg, 1=fg)",
    },
]

def migrate_file(file_path: Path, dry_run: bool = True) -> dict:
    """Migre un fichier."""
    content = file_path.read_text()
    original = content

    changes = []

    for pattern in PATTERNS:
        old_pattern = pattern["old"]
        new_code = pattern["new"]
        comment = pattern.get("comment", "")

        matches = list(re.finditer(old_pattern, content))

        if matches:
            # Remplacer avec commentaire
            content = re.sub(old_pattern, new_code + comment, content)

            for match in matches:
                changes.append({
                    "line": content[:match.start()].count('\n') + 1,
                    "old": match.group(0),
                    "new": new_code + comment,
                })

    if not changes:
        return {"changed": False}

    if not dry_run:
        file_path.write_text(content)

    return {
        "changed": True,
        "changes": changes,
        "original": original,
        "new": content,
    }

def main():
    parser = argparse.ArgumentParser(description="Migrer sigmoid → softmax pour NP")
    parser.add_argument("--dry-run", action="store_true",
                       help="Afficher changements sans modifier")
    args = parser.parse_args()

    print("=" * 80)
    print("MIGRATION: sigmoid → softmax POUR NP PREDICTIONS")
    print("=" * 80)
    print()

    # Scanner tous les fichiers Python dans scripts/
    scripts_dir = PROJECT_ROOT / "scripts"
    python_files = list(scripts_dir.rglob("*.py"))

    migrated = []

    for file_path in python_files:
        result = migrate_file(file_path, dry_run=args.dry_run)

        if result["changed"]:
            migrated.append((file_path, result))

    if not migrated:
        print("✅ Aucun fichier à migrer - Tous les scripts sont déjà corrects")
        return 0

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Fichiers à migrer: {len(migrated)}")
    print()

    for file_path, result in migrated:
        rel_path = file_path.relative_to(PROJECT_ROOT)
        print(f"📄 {rel_path}")

        for change in result["changes"]:
            print(f"   Ligne {change['line']}:")
            print(f"     AVANT: {change['old']}")
            print(f"     APRÈS: {change['new']}")
        print()

    if args.dry_run:
        print("Pour exécuter réellement:")
        print("  python scripts/utils/migrate_np_predictions.py")
    else:
        print(f"✅ {len(migrated)} fichiers migrés avec succès")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
