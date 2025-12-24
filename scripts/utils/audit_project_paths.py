#!/usr/bin/env python3
"""
Audit complet des paths utilisés dans le projet.

Vérifie:
1. Tous les scripts Python et leurs références de paths
2. Les données existantes sur disque
3. Les incohérences et doublons
4. Génère un plan de nettoyage

Usage:
    python scripts/utils/audit_project_paths.py
"""

import sys
from pathlib import Path
import re
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import DEFAULT_FAMILY_DATA_DIR, DEFAULT_FAMILY_FIXED_DIR

def scan_python_files():
    """Scanne tous les fichiers Python pour trouver les références de paths."""
    project_root = Path(__file__).parent.parent.parent

    patterns = {
        "hardcoded_data": r'["\']data/(?:cache/)?family_(?:data|FIXED)["\']',
        "constant_import": r'from src\.constants import.*(?:DEFAULT_FAMILY_DATA_DIR|DEFAULT_FAMILY_FIXED_DIR)',
        "path_join": r'(?:Path|os\.path\.join).*family_(?:data|FIXED)',
    }

    results = {
        "scripts_using_constants": [],
        "scripts_with_hardcoded": [],
        "scripts_clean": [],
    }

    for py_file in project_root.rglob("*.py"):
        if ".git" in str(py_file) or "__pycache__" in str(py_file):
            continue

        content = py_file.read_text()

        has_constant = bool(re.search(patterns["constant_import"], content))
        has_hardcoded = bool(re.search(patterns["hardcoded_data"], content))

        rel_path = str(py_file.relative_to(project_root))

        if has_constant:
            results["scripts_using_constants"].append(rel_path)
        if has_hardcoded:
            results["scripts_with_hardcoded"].append(rel_path)
        if not has_constant and not has_hardcoded:
            # Check si le script manipule des données famille
            if "family" in content.lower() and ("features" in content or "targets" in content):
                results["scripts_clean"].append(rel_path)

    return results

def scan_data_directories():
    """Scanne les répertoires de données pour identifier ce qui existe."""
    project_root = Path(__file__).parent.parent.parent

    data_locations = {
        "data/family_data": {
            "path": project_root / "data" / "family_data",
            "expected": ["*_features.npz", "*_targets.npz"],
        },
        "data/family_FIXED": {
            "path": project_root / "data" / "family_FIXED",
            "expected": ["*_data_FIXED.npz"],
        },
        "data/cache/family_data": {
            "path": project_root / "data" / "cache" / "family_data",
            "expected": ["*_features.npz", "*_targets.npz"],
        },
        "data/cache/pannuke_features": {
            "path": project_root / "data" / "cache" / "pannuke_features",
            "expected": ["fold*_features.npz"],
        },
    }

    results = {}

    for name, info in data_locations.items():
        path = info["path"]

        if not path.exists():
            results[name] = {
                "exists": False,
                "files": [],
                "total_size_mb": 0,
            }
            continue

        files = []
        total_size = 0

        for pattern in info["expected"]:
            for file in path.glob(pattern):
                size = file.stat().st_size
                files.append({
                    "name": file.name,
                    "size_mb": size / 1e6,
                    "modified": file.stat().st_mtime,
                })
                total_size += size

        results[name] = {
            "exists": True,
            "files": files,
            "total_size_mb": total_size / 1e6,
        }

    return results

def generate_cleanup_plan(script_audit, data_audit):
    """Génère un plan de nettoyage basé sur l'audit."""
    plan = {
        "scripts_to_fix": [],
        "data_to_keep": {},
        "data_to_delete": {},
        "actions": [],
    }

    # Scripts avec hardcoded paths
    if script_audit["scripts_with_hardcoded"]:
        plan["scripts_to_fix"] = script_audit["scripts_with_hardcoded"]
        plan["actions"].append({
            "priority": "HIGH",
            "action": "Migrer scripts hardcodés vers constantes",
            "count": len(script_audit["scripts_with_hardcoded"]),
            "script": "scripts/utils/migrate_to_centralized_paths.py",
        })

    # Données redondantes
    has_data = data_audit.get("data/family_data", {}).get("exists", False)
    has_cache = data_audit.get("data/cache/family_data", {}).get("exists", False)

    if has_data and has_cache:
        # Comparer timestamps pour voir lequel est le plus récent
        data_files = data_audit["data/family_data"]["files"]
        cache_files = data_audit["data/cache/family_data"]["files"]

        if data_files and cache_files:
            data_newest = max(f["modified"] for f in data_files)
            cache_newest = max(f["modified"] for f in cache_files)

            if data_newest > cache_newest:
                plan["data_to_keep"]["data/family_data"] = "Plus récent"
                plan["data_to_delete"]["data/cache/family_data"] = {
                    "reason": "Obsolète (plus ancien)",
                    "size_mb": data_audit["data/cache/family_data"]["total_size_mb"],
                }
            else:
                plan["data_to_keep"]["data/cache/family_data"] = "Plus récent"
                plan["data_to_delete"]["data/family_data"] = {
                    "reason": "Obsolète (plus ancien)",
                    "size_mb": data_audit["data/family_data"]["total_size_mb"],
                }

        plan["actions"].append({
            "priority": "MEDIUM",
            "action": "Supprimer données redondantes",
            "savings_mb": plan["data_to_delete"].get("data/family_data", plan["data_to_delete"].get("data/cache/family_data", {})).get("size_mb", 0),
        })

    return plan

def print_report(script_audit, data_audit, cleanup_plan):
    """Affiche le rapport d'audit."""
    print("=" * 80)
    print("AUDIT COMPLET DU PROJET - PATHS ET DONNÉES")
    print("=" * 80)
    print()

    # Scripts
    print("1. SCRIPTS PYTHON")
    print("-" * 80)
    print(f"✅ Utilisant constantes: {len(script_audit['scripts_using_constants'])}")
    for script in script_audit['scripts_using_constants'][:5]:
        print(f"   - {script}")
    if len(script_audit['scripts_using_constants']) > 5:
        print(f"   ... et {len(script_audit['scripts_using_constants']) - 5} autres")
    print()

    if script_audit['scripts_with_hardcoded']:
        print(f"⚠️  Avec paths hardcodés: {len(script_audit['scripts_with_hardcoded'])}")
        for script in script_audit['scripts_with_hardcoded']:
            print(f"   - {script}")
        print()
    else:
        print("✅ Aucun script avec paths hardcodés")
        print()

    # Données
    print("2. RÉPERTOIRES DE DONNÉES")
    print("-" * 80)
    for name, info in data_audit.items():
        if info["exists"]:
            print(f"✅ {name}:")
            print(f"   Taille: {info['total_size_mb']:.1f} MB")
            print(f"   Fichiers: {len(info['files'])}")
            for f in info["files"][:3]:
                print(f"     - {f['name']} ({f['size_mb']:.1f} MB)")
            if len(info["files"]) > 3:
                print(f"     ... et {len(info['files']) - 3} autres")
        else:
            print(f"❌ {name}: N'EXISTE PAS")
        print()

    # Plan de nettoyage
    print("3. PLAN DE NETTOYAGE")
    print("-" * 80)

    if not cleanup_plan["actions"]:
        print("✅ Aucune action requise - Projet propre!")
        print()
        return

    for action in cleanup_plan["actions"]:
        priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        emoji = priority_emoji.get(action["priority"], "⚪")
        print(f"{emoji} {action['priority']}: {action['action']}")
        if "count" in action:
            print(f"   Scripts à migrer: {action['count']}")
        if "savings_mb" in action:
            print(f"   Espace libéré: {action['savings_mb']:.1f} MB")
        if "script" in action:
            print(f"   Commande: python {action['script']}")
        print()

    # Actions recommandées
    print("4. ACTIONS RECOMMANDÉES (ORDRE)")
    print("-" * 80)

    step = 1

    if cleanup_plan["scripts_to_fix"]:
        print(f"{step}. Migrer scripts hardcodés:")
        print("   python scripts/utils/migrate_to_centralized_paths.py")
        step += 1

    if cleanup_plan["data_to_delete"]:
        print(f"{step}. Supprimer données redondantes:")
        for path, info in cleanup_plan["data_to_delete"].items():
            print(f"   rm -rf {path}")
            print(f"   # Libère {info['size_mb']:.1f} MB ({info['reason']})")
        step += 1

    if cleanup_plan["data_to_keep"]:
        print(f"{step}. Vérifier données conservées:")
        for path, reason in cleanup_plan["data_to_keep"].items():
            print(f"   ✅ {path} ({reason})")
        step += 1

    print(f"{step}. Lancer re-train sur bases propres:")
    print("   python scripts/training/train_hovernet_family.py --family epidermal --epochs 50 --augment")
    print()

def main():
    print("Démarrage audit...")
    print()

    script_audit = scan_python_files()
    data_audit = scan_data_directories()
    cleanup_plan = generate_cleanup_plan(script_audit, data_audit)

    print_report(script_audit, data_audit, cleanup_plan)

    # Sauvegarder rapport JSON
    report_path = Path("results") / "audit_project_paths.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "scripts": script_audit,
        "data": data_audit,
        "cleanup_plan": cleanup_plan,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📄 Rapport détaillé sauvegardé: {report_path}")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
