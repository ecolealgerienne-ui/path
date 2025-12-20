#!/usr/bin/env python3
"""
Télécharge et prépare PanNuke pour l'entraînement.

Ce script:
1. Télécharge les 3 folds de PanNuke depuis Warwick
2. Réorganise la structure pour correspondre à extract_features.py
3. Vérifie l'intégrité des données

Structure attendue après exécution:
    data_dir/
    ├── fold0/
    │   ├── images.npy    (N, 256, 256, 3) float64
    │   ├── masks.npy     (N, 256, 256, 6)
    │   └── types.npy     (N,) string
    ├── fold1/
    │   └── ...
    └── fold2/
        └── ...

Usage:
    python scripts/setup/download_and_prepare_pannuke.py --output_dir /home/amar/data/PanNuke
    python scripts/setup/download_and_prepare_pannuke.py --output_dir /home/amar/data/PanNuke --fold 1
"""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np


# URLs officielles PanNuke (Warwick University)
PANNUKE_URLS = {
    1: "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip",
    2: "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip",
    3: "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip",
}

# Mapping Fold officiel → fold interne (0-indexed)
FOLD_MAPPING = {1: 0, 2: 1, 3: 2}


def download_file(url: str, output_path: Path) -> bool:
    """Télécharge un fichier avec wget, curl ou requests."""
    print(f"📥 Téléchargement: {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Essayer wget
    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(output_path), url],
            check=True
        )
        if output_path.exists() and output_path.stat().st_size > 1000:
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Essayer curl
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", str(output_path), "--progress-bar", url],
            check=True
        )
        if output_path.exists() and output_path.stat().st_size > 1000:
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Essayer requests
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)", end="")
            print()
        return output_path.exists() and output_path.stat().st_size > 1000
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def extract_and_reorganize(zip_path: Path, output_dir: Path, fold_official: int) -> bool:
    """
    Extrait le ZIP et réorganise pour correspondre à extract_features.py.

    PanNuke officiel (structure réelle):
        Fold 1/images/fold1/images.npy
        Fold 1/images/fold1/types.npy
        Fold 1/masks/fold1/masks.npy

    Notre structure:
        fold0/images.npy
        fold0/masks.npy
        fold0/types.npy
    """
    fold_internal = FOLD_MAPPING[fold_official]
    target_dir = output_dir / f"fold{fold_internal}"

    print(f"📦 Extraction Fold{fold_official} → fold{fold_internal}...")

    # Extraire dans un répertoire temporaire
    temp_dir = output_dir / f"_temp_fold{fold_official}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)

        # Chercher les fichiers .npy dans l'extraction
        npy_files = list(temp_dir.rglob("*.npy"))
        print(f"  Fichiers trouvés: {len(npy_files)}")

        # Trouver images.npy, masks.npy, types.npy
        images_file = None
        masks_file = None
        types_file = None

        for f in npy_files:
            if f.name == "images.npy":
                images_file = f
            elif f.name == "masks.npy":
                masks_file = f
            elif f.name == "types.npy":
                types_file = f

        if images_file is None or masks_file is None or types_file is None:
            print(f"  ❌ Fichiers manquants:")
            print(f"     images.npy: {images_file}")
            print(f"     masks.npy: {masks_file}")
            print(f"     types.npy: {types_file}")
            raise FileNotFoundError("Fichiers PanNuke incomplets")

        print(f"  📄 images.npy: {images_file}")
        print(f"  📄 masks.npy: {masks_file}")
        print(f"  📄 types.npy: {types_file}")

        # Créer le répertoire cible
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copier les fichiers
        shutil.copy(images_file, target_dir / "images.npy")
        shutil.copy(masks_file, target_dir / "masks.npy")
        shutil.copy(types_file, target_dir / "types.npy")

        print(f"  ✅ Réorganisé dans {target_dir}")

        # Nettoyer
        shutil.rmtree(temp_dir)
        return True

    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False


def verify_fold(data_dir: Path, fold_internal: int) -> dict:
    """Vérifie l'intégrité d'un fold."""
    fold_dir = data_dir / f"fold{fold_internal}"

    result = {
        "fold": fold_internal,
        "valid": False,
        "images": None,
        "masks": None,
        "types": None,
    }

    if not fold_dir.exists():
        return result

    try:
        # Images
        images_path = fold_dir / "images.npy"
        if images_path.exists():
            images = np.load(images_path)
            result["images"] = {
                "shape": images.shape,
                "dtype": str(images.dtype),
                "range": f"[{images.min():.1f}, {images.max():.1f}]",
            }

        # Masks
        masks_path = fold_dir / "masks.npy"
        if masks_path.exists():
            masks = np.load(masks_path)
            result["masks"] = {
                "shape": masks.shape,
                "dtype": str(masks.dtype),
            }

        # Types
        types_path = fold_dir / "types.npy"
        if types_path.exists():
            types = np.load(types_path)
            unique_types = np.unique(types)
            result["types"] = {
                "count": len(types),
                "unique_organs": len(unique_types),
                "organs": list(unique_types)[:5],  # Premiers 5
            }

        # Valide si les 3 fichiers existent
        result["valid"] = all([
            result["images"] is not None,
            result["masks"] is not None,
            result["types"] is not None,
        ])

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge et prépare PanNuke pour CellViT-Optimus"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Répertoire de destination (ex: /home/amar/data/PanNuke)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3],
        help="Fold spécifique à télécharger (1, 2, ou 3)"
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Conserver les fichiers ZIP après extraction"
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Vérifier uniquement les données existantes"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mode vérification
    if args.verify_only:
        print("\n" + "=" * 60)
        print("VÉRIFICATION DES DONNÉES PANNUKE")
        print("=" * 60)

        all_valid = True
        for fold_internal in [0, 1, 2]:
            result = verify_fold(output_dir, fold_internal)
            fold_official = fold_internal + 1

            print(f"\nFold {fold_internal} (Fold{fold_official} officiel):")
            if result["valid"]:
                print(f"  ✅ Images: {result['images']['shape']} {result['images']['dtype']}")
                print(f"     Range: {result['images']['range']}")
                print(f"  ✅ Masks: {result['masks']['shape']}")
                print(f"  ✅ Types: {result['types']['count']} samples, {result['types']['unique_organs']} organes")
            else:
                print(f"  ❌ Données manquantes ou invalides")
                all_valid = False

        print("\n" + "=" * 60)
        if all_valid:
            print("✅ Toutes les données sont valides!")
        else:
            print("⚠️  Certains folds sont manquants ou invalides")
        return

    # Téléchargement
    folds_to_download = [args.fold] if args.fold else [1, 2, 3]

    print("\n" + "=" * 60)
    print("TÉLÉCHARGEMENT PANNUKE")
    print("=" * 60)
    print(f"Destination: {output_dir}")
    print(f"Folds: {folds_to_download}")
    print("=" * 60)

    for fold_official in folds_to_download:
        fold_internal = FOLD_MAPPING[fold_official]
        target_dir = output_dir / f"fold{fold_internal}"

        # Vérifier si déjà présent
        if (target_dir / "images.npy").exists():
            print(f"\n✅ fold{fold_internal} déjà présent, skip")
            continue

        print(f"\n--- Fold{fold_official} → fold{fold_internal} ---")

        # Télécharger
        url = PANNUKE_URLS[fold_official]
        zip_path = output_dir / f"fold_{fold_official}.zip"

        if not zip_path.exists():
            success = download_file(url, zip_path)
            if not success:
                print(f"❌ Échec du téléchargement de Fold{fold_official}")
                print(f"   Téléchargez manuellement: {url}")
                continue
        else:
            print(f"📦 ZIP déjà présent: {zip_path}")

        # Extraire et réorganiser
        success = extract_and_reorganize(zip_path, output_dir, fold_official)

        # Supprimer le ZIP si demandé
        if success and not args.keep_zip and zip_path.exists():
            print(f"🗑️  Suppression du ZIP")
            zip_path.unlink()

    # Vérification finale
    print("\n" + "=" * 60)
    print("VÉRIFICATION FINALE")
    print("=" * 60)

    all_valid = True
    for fold_internal in [0, 1, 2]:
        result = verify_fold(output_dir, fold_internal)
        status = "✅" if result["valid"] else "❌"
        if result["valid"]:
            print(f"{status} fold{fold_internal}: {result['images']['shape'][0]} images, {result['types']['unique_organs']} organes")
        else:
            print(f"{status} fold{fold_internal}: MANQUANT")
            all_valid = False

    print("\n" + "=" * 60)
    if all_valid:
        print("🎉 PanNuke prêt pour l'extraction de features!")
        print(f"\nProchaine étape:")
        print(f"  python scripts/preprocessing/extract_features.py \\")
        print(f"      --data_dir {output_dir} \\")
        print(f"      --fold 0 --all_layers")
    else:
        print("⚠️  Certains folds sont manquants")


if __name__ == "__main__":
    main()
