#!/usr/bin/env python3
"""
Script d'Inspection de l'Environnement CellViT-Optimus

Ce script collecte TOUTES les informations nécessaires pour que Claude puisse
comprendre votre environnement SANS jamais avoir à tester lui-même.

Usage:
    python scripts/utils/inspect_environment.py > environment_report.txt

Ensuite, copiez le contenu de environment_report.txt à Claude.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import platform

def run_command(cmd, shell=False):
    """Exécute une commande et retourne la sortie (ou None si erreur)."""
    try:
        result = subprocess.run(
            cmd if shell else cmd.split(),
            capture_output=True,
            text=True,
            timeout=10,
            shell=shell
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as e:
        return f"ERROR: {str(e)}"

def get_directory_size(path):
    """Retourne la taille d'un répertoire en GB."""
    try:
        result = run_command(f"du -sh {path}", shell=True)
        if result and result != "ERROR":
            return result.split()[0]
        return "N/A"
    except:
        return "N/A"

def count_files(path, pattern="*"):
    """Compte le nombre de fichiers dans un répertoire."""
    try:
        p = Path(path)
        if not p.exists():
            return 0
        return len(list(p.glob(pattern)))
    except:
        return 0

def check_file_exists(path):
    """Vérifie si un fichier existe et retourne ses infos."""
    p = Path(path)
    if not p.exists():
        return {"exists": False}

    stat = p.stat()
    return {
        "exists": True,
        "size": f"{stat.st_size / (1024**2):.2f} MB" if stat.st_size > 1024**2 else f"{stat.st_size / 1024:.2f} KB",
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    }

def inspect_environment():
    """Collecte toutes les informations d'environnement."""

    print("="*80)
    print("RAPPORT D'INSPECTION ENVIRONNEMENT CellViT-Optimus")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Répertoire: {os.getcwd()}")
    print("="*80)
    print()

    # ========================================================================
    # SECTION 1: SYSTÈME
    # ========================================================================
    print("### 1. SYSTÈME ###")
    print("-" * 40)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Hostname: {platform.node()}")

    # Espace disque
    df_output = run_command("df -h .", shell=True)
    if df_output:
        lines = df_output.split('\n')
        if len(lines) > 1:
            print(f"Espace disque:\n{lines[0]}\n{lines[1]}")
    print()

    # ========================================================================
    # SECTION 2: GPU & CUDA
    # ========================================================================
    print("### 2. GPU & CUDA ###")
    print("-" * 40)
    nvidia_smi = run_command("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader")
    if nvidia_smi and nvidia_smi != "ERROR":
        print(f"GPU Info:\n{nvidia_smi}")
    else:
        print("❌ nvidia-smi non disponible")

    nvcc_version = run_command("nvcc --version", shell=True)
    if nvcc_version and "release" in nvcc_version:
        # Extraire la ligne avec la version
        for line in nvcc_version.split('\n'):
            if "release" in line:
                print(f"CUDA Compiler: {line.strip()}")
    print()

    # ========================================================================
    # SECTION 3: ENVIRONNEMENT PYTHON
    # ========================================================================
    print("### 3. ENVIRONNEMENT PYTHON ###")
    print("-" * 40)

    # Conda
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'N/A')
    print(f"Conda Env: {conda_env}")

    # Packages critiques
    critical_packages = [
        'torch', 'torchvision', 'timm', 'transformers',
        'numpy', 'scipy', 'scikit-learn', 'opencv-python',
        'matplotlib', 'pandas'
    ]

    print("\nPackages Critiques:")
    for pkg in critical_packages:
        try:
            result = run_command(f"python -c \"import {pkg}; print({pkg}.__version__)\"", shell=True)
            if result and result != "ERROR":
                print(f"  ✅ {pkg:20s} {result}")
            else:
                print(f"  ❌ {pkg:20s} NOT INSTALLED")
        except:
            print(f"  ❌ {pkg:20s} ERROR")
    print()

    # ========================================================================
    # SECTION 4: DONNÉES PanNuke
    # ========================================================================
    print("### 4. DONNÉES PANNUKE ###")
    print("-" * 40)

    pannuke_paths = [
        "/home/amar/data/PanNuke",
        "data/PanNuke",
        "/data/PanNuke"
    ]

    pannuke_found = False
    for path in pannuke_paths:
        if Path(path).exists():
            print(f"✅ PanNuke trouvé: {path}")
            print(f"   Taille: {get_directory_size(path)}")

            # Lister les folds
            p = Path(path)
            folds = sorted([f.name for f in p.iterdir() if f.is_dir() and 'fold' in f.name.lower()])
            if folds:
                print(f"   Folds: {', '.join(folds)}")

                # Pour chaque fold, compter les fichiers
                for fold in folds:
                    fold_path = p / fold
                    images_count = count_files(fold_path, "images.npy") + count_files(fold_path, "*.png") + count_files(fold_path, "*.jpg")
                    masks_count = count_files(fold_path, "masks.npy") + count_files(fold_path, "*mask*.npy")
                    print(f"      {fold}: ~{images_count} images, ~{masks_count} masks")

            pannuke_found = True
            break

    if not pannuke_found:
        print("❌ PanNuke NON TROUVÉ dans les emplacements standards")
        print(f"   Cherché dans: {', '.join(pannuke_paths)}")
    print()

    # ========================================================================
    # SECTION 5: CACHES & FEATURES
    # ========================================================================
    print("### 5. CACHES & FEATURES ###")
    print("-" * 40)

    cache_dirs = [
        ("data/cache/pannuke_features", "Features H-optimus-0 (folds 0,1,2)"),
        ("data/cache/family_data", "Targets par famille (FIXED/OLD)"),
        ("data/cache/family_data_FIXED", "Targets FIXED (float32)"),
        ("data/cache/family_data_OLD_int8_20251222_163212", "Targets OLD (int8, corrompu)"),
        ("data/cache/pannuke_features_OLD_CORRUPTED_20251223", "Features corrompues (Bug #4)")
    ]

    for cache_path, description in cache_dirs:
        p = Path(cache_path)
        if p.exists():
            print(f"✅ {cache_path}")
            print(f"   {description}")
            print(f"   Taille: {get_directory_size(cache_path)}")

            # Compter les fichiers .npz
            npz_count = count_files(cache_path, "*.npz")
            print(f"   Fichiers .npz: {npz_count}")

            # Lister les premiers fichiers
            npz_files = sorted(p.glob("*.npz"))[:5]
            if npz_files:
                print(f"   Exemples: {', '.join([f.name for f in npz_files])}")
        else:
            print(f"❌ {cache_path} (n'existe pas)")
    print()

    # ========================================================================
    # SECTION 6: CHECKPOINTS MODÈLES
    # ========================================================================
    print("### 6. CHECKPOINTS MODÈLES ###")
    print("-" * 40)

    checkpoint_files = [
        "models/checkpoints/organ_head_best.pth",
        "models/checkpoints/hovernet_glandular_best.pth",
        "models/checkpoints/hovernet_digestive_best.pth",
        "models/checkpoints/hovernet_urologic_best.pth",
        "models/checkpoints/hovernet_epidermal_best.pth",
        "models/checkpoints/hovernet_respiratory_best.pth",
        "models/pretrained/CellViT-256.pth"
    ]

    for ckpt in checkpoint_files:
        info = check_file_exists(ckpt)
        if info["exists"]:
            print(f"✅ {ckpt}")
            print(f"   Taille: {info['size']}, Modifié: {info['modified']}")
        else:
            print(f"❌ {ckpt} (n'existe pas)")
    print()

    # ========================================================================
    # SECTION 7: STRUCTURE PROJET
    # ========================================================================
    print("### 7. STRUCTURE PROJET ###")
    print("-" * 40)

    # Arborescence src/
    print("\n📁 src/")
    src_path = Path("src")
    if src_path.exists():
        for item in sorted(src_path.rglob("*.py")):
            rel_path = item.relative_to(src_path)
            depth = len(rel_path.parts) - 1
            indent = "  " * depth
            print(f"{indent}├── {rel_path.name}")

    # Compter les scripts
    print("\n📁 scripts/")
    scripts_path = Path("scripts")
    if scripts_path.exists():
        for subdir in sorted(scripts_path.iterdir()):
            if subdir.is_dir():
                py_count = count_files(subdir, "*.py")
                print(f"  ├── {subdir.name}/ ({py_count} scripts)")
    print()

    # ========================================================================
    # SECTION 8: ÉTAT GIT
    # ========================================================================
    print("### 8. ÉTAT GIT ###")
    print("-" * 40)

    branch = run_command("git branch --show-current")
    print(f"Branche actuelle: {branch if branch else 'N/A'}")

    status = run_command("git status --short")
    if status:
        print(f"Fichiers modifiés:\n{status}")
    else:
        print("Working tree clean")

    last_commits = run_command("git log --oneline -5")
    if last_commits:
        print(f"\nDerniers commits:\n{last_commits}")
    print()

    # ========================================================================
    # SECTION 9: TESTS CRITIQUES
    # ========================================================================
    print("### 9. TESTS CRITIQUES ###")
    print("-" * 40)

    # Test import PyTorch + CUDA
    print("Test PyTorch + CUDA:")
    torch_test = run_command(
        'python -c "import torch; print(f\'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}\')\"',
        shell=True
    )
    if torch_test and torch_test != "ERROR":
        print(f"  {torch_test}")
    else:
        print(f"  ❌ Erreur import PyTorch")

    # Test import timm (H-optimus-0)
    print("\nTest timm (H-optimus-0):")
    timm_test = run_command('python -c "import timm; print(f\'timm: {timm.__version__}\')"', shell=True)
    if timm_test and timm_test != "ERROR":
        print(f"  {timm_test}")
    else:
        print(f"  ❌ Erreur import timm")

    # Test modules custom
    print("\nTest modules custom:")
    custom_imports = [
        "from src.preprocessing import create_hoptimus_transform",
        "from src.models.loader import ModelLoader",
        "from src.data.preprocessing import validate_targets"
    ]
    for import_stmt in custom_imports:
        test = run_command(f'python -c "{import_stmt}; print(\'OK\')"', shell=True)
        if test == "OK":
            print(f"  ✅ {import_stmt}")
        else:
            print(f"  ❌ {import_stmt} ({test})")
    print()

    # ========================================================================
    # SECTION 10: RÉSUMÉ & RECOMMANDATIONS
    # ========================================================================
    print("="*80)
    print("### RÉSUMÉ ###")
    print("="*80)

    # Calcul du statut global
    checks = {
        "GPU disponible": nvidia_smi and nvidia_smi != "ERROR",
        "PanNuke trouvé": pannuke_found,
        "PyTorch + CUDA": torch_test and "CUDA: True" in torch_test if torch_test else False,
        "Modules custom OK": all([
            run_command(f'python -c "{imp}; print(\'OK\')"', shell=True) == "OK"
            for imp in custom_imports
        ])
    }

    print("\nChecks Environnement:")
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {check}")

    all_ok = all(checks.values())

    if all_ok:
        print("\n🎉 ENVIRONNEMENT PRÊT pour entraînement/évaluation")
    else:
        print("\n⚠️ PROBLÈMES DÉTECTÉS - Voir sections ci-dessus")

    print("\n" + "="*80)
    print("FIN DU RAPPORT")
    print("="*80)
    print("\n💡 Copiez TOUT ce rapport et envoyez-le à Claude pour analyse.")

if __name__ == "__main__":
    try:
        inspect_environment()
    except KeyboardInterrupt:
        print("\n\n❌ Inspection interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERREUR CRITIQUE: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
