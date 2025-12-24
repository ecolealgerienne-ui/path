#!/usr/bin/env python3
"""
Nettoyage automatisé des données redondantes du projet.

Supprime:
- Données obsolètes/dupliquées
- Cache features corrompus
- Checkpoints invalides

Conserve:
- Données les plus récentes
- Checkpoints validés
- Sources FIXED

Usage:
    python scripts/utils/cleanup_project_data.py --dry-run  # Voir ce qui sera supprimé
    python scripts/utils/cleanup_project_data.py            # Exécuter le nettoyage
"""

import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import DEFAULT_FAMILY_DATA_DIR, DEFAULT_FAMILY_FIXED_DIR

def identify_redundant_data():
    """Identifie les données redondantes."""
    project_root = Path(__file__).parent.parent.parent

    candidates = {
        "data/cache/family_data": {
            "path": project_root / "data" / "cache" / "family_data",
            "reason": "Redondant avec data/family_data (source de vérité)",
            "priority": "HIGH",
        },
        "data/cache/pannuke_features": {
            "path": project_root / "data" / "cache" / "pannuke_features",
            "reason": "Features OLD (avant fix Bug #1 et #2) - CLS std incorrect",
            "priority": "HIGH",
        },
    }

    to_delete = {}

    for name, info in candidates.items():
        path = info["path"]

        if path.exists():
            # Calculer taille
            total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

            to_delete[name] = {
                "path": str(path),
                "exists": True,
                "size_mb": total_size / 1e6,
                "reason": info["reason"],
                "priority": info["priority"],
            }

    return to_delete

def backup_before_delete(path: Path, backup_root: Path):
    """Crée une sauvegarde avant suppression."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.name}_BACKUP_{timestamp}"
    backup_path = backup_root / backup_name

    if path.exists():
        shutil.move(str(path), str(backup_path))
        return backup_path

    return None

def cleanup_data(to_delete: dict, dry_run: bool = True, create_backup: bool = False):
    """Exécute le nettoyage."""
    project_root = Path(__file__).parent.parent.parent
    backup_root = project_root / "data" / "backups"

    results = {
        "deleted": [],
        "backed_up": [],
        "errors": [],
        "total_freed_mb": 0,
    }

    for name, info in sorted(to_delete.items(), key=lambda x: x[1]["priority"], reverse=True):
        path = Path(info["path"])

        if not path.exists():
            continue

        print(f"\n{'[DRY-RUN] ' if dry_run else ''}Traitement: {name}")
        print(f"  Path: {path}")
        print(f"  Taille: {info['size_mb']:.1f} MB")
        print(f"  Raison: {info['reason']}")

        if dry_run:
            print(f"  → Sera supprimé")
            results["total_freed_mb"] += info["size_mb"]
            continue

        try:
            if create_backup:
                backup_root.mkdir(parents=True, exist_ok=True)
                backup_path = backup_before_delete(path, backup_root)
                if backup_path:
                    print(f"  ✅ Sauvegardé: {backup_path}")
                    results["backed_up"].append(str(backup_path))
            else:
                shutil.rmtree(path)
                print(f"  ✅ Supprimé")

            results["deleted"].append(name)
            results["total_freed_mb"] += info["size_mb"]

        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            results["errors"].append({
                "path": name,
                "error": str(e),
            })

    return results

def verify_essential_data():
    """Vérifie que les données essentielles sont présentes."""
    project_root = Path(__file__).parent.parent.parent

    essential = {
        "data/family_data": {
            "path": project_root / DEFAULT_FAMILY_DATA_DIR,
            "required_files": ["*_features.npz", "*_targets.npz"],
        },
        "data/family_FIXED": {
            "path": project_root / DEFAULT_FAMILY_FIXED_DIR,
            "required_files": ["*_data_FIXED.npz"],
        },
    }

    status = {}

    for name, info in essential.items():
        path = info["path"]

        if not path.exists():
            status[name] = {
                "exists": False,
                "complete": False,
            }
            continue

        # Vérifier fichiers requis
        found = []
        for pattern in info["required_files"]:
            files = list(path.glob(pattern))
            found.extend([f.name for f in files])

        status[name] = {
            "exists": True,
            "complete": len(found) > 0,
            "files": found,
        }

    return status

def main():
    parser = argparse.ArgumentParser(description="Nettoyage données projet")
    parser.add_argument("--dry-run", action="store_true",
                       help="Afficher ce qui sera supprimé sans supprimer")
    parser.add_argument("--backup", action="store_true",
                       help="Créer sauvegarde avant suppression")
    parser.add_argument("--force", action="store_true",
                       help="Supprimer sans confirmation")
    args = parser.parse_args()

    print("=" * 80)
    print("NETTOYAGE DONNÉES PROJET")
    print("=" * 80)
    print()

    # Vérifier données essentielles
    print("1. VÉRIFICATION DONNÉES ESSENTIELLES")
    print("-" * 80)
    essential_status = verify_essential_data()

    all_essential_ok = True

    for name, status in essential_status.items():
        if not status["exists"] or not status["complete"]:
            print(f"❌ {name}: MANQUANT")
            all_essential_ok = False
        else:
            print(f"✅ {name}: OK ({len(status['files'])} fichiers)")

    print()

    if not all_essential_ok:
        print("⚠️  ATTENTION: Données essentielles manquantes!")
        print("   Lancez d'abord:")
        print("   python scripts/preprocessing/extract_features_from_fixed.py --family <FAMILY>")
        print()
        if not args.force:
            return 1

    # Identifier données à supprimer
    print("2. DONNÉES REDONDANTES IDENTIFIÉES")
    print("-" * 80)
    to_delete = identify_redundant_data()

    if not to_delete:
        print("✅ Aucune donnée redondante trouvée - Projet propre!")
        print()
        return 0

    total_size = sum(info["size_mb"] for info in to_delete.values())

    for name, info in to_delete.items():
        priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        emoji = priority_emoji.get(info["priority"], "⚪")
        print(f"{emoji} {name}:")
        print(f"   Taille: {info['size_mb']:.1f} MB")
        print(f"   Raison: {info['reason']}")
        print()

    print(f"💾 Espace total à libérer: {total_size:.1f} MB")
    print()

    # Confirmation
    if not args.dry_run and not args.force:
        print("⚠️  Cette action est IRRÉVERSIBLE (sans --backup)!")
        response = input("Continuer? (oui/non): ")
        if response.lower() not in ["oui", "yes", "y"]:
            print("Annulé.")
            return 0
        print()

    # Exécuter nettoyage
    print("3. EXÉCUTION NETTOYAGE")
    print("-" * 80)
    results = cleanup_data(to_delete, dry_run=args.dry_run, create_backup=args.backup)

    print()
    print("=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    if args.dry_run:
        print(f"[DRY-RUN MODE]")
        print(f"Espace qui sera libéré: {results['total_freed_mb']:.1f} MB")
        print(f"Fichiers qui seront supprimés: {len(to_delete)}")
        print()
        print("Pour exécuter réellement:")
        print("  python scripts/utils/cleanup_project_data.py")
        print()
        print("Pour créer des sauvegardes:")
        print("  python scripts/utils/cleanup_project_data.py --backup")
    else:
        print(f"✅ Supprimés: {len(results['deleted'])}")
        if results["backed_up"]:
            print(f"💾 Sauvegardés: {len(results['backed_up'])}")
        if results["errors"]:
            print(f"❌ Erreurs: {len(results['errors'])}")
        print(f"💾 Espace libéré: {results['total_freed_mb']:.1f} MB")

    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
