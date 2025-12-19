#!/usr/bin/env python3
"""
Téléchargement des datasets pour CellViT-Optimus.

Datasets disponibles:
- PanNuke: Segmentation cellulaire (Fold 1, 2, 3)
- MoNuSeG: Multi-organes segmentation
- CoNSeP: Morphologie colique

Usage:
    python scripts/setup/download_datasets.py --dataset pannuke --fold 1
    python scripts/setup/download_datasets.py --dataset pannuke --all-folds
"""

import argparse
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Optional


# Configuration des datasets
DATASETS = {
    "pannuke": {
        "base_url": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke",
        "folds": {
            1: "fold_1.zip",
            2: "fold_2.zip",
            3: "fold_3.zip",
        },
        "description": "PanNuke: ~200k noyaux, 5 types, 19 organes",
        "size": "~668 MB par fold",
    },
    "monuseg": {
        "url": "https://monuseg.grand-challenge.org/Data/",
        "description": "MoNuSeG: Multi-organes segmentation",
        "note": "Téléchargement manuel requis (inscription)",
    },
    "consep": {
        "url": "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/",
        "description": "CoNSeP: Morphologie colique",
        "note": "Téléchargement manuel recommandé",
    },
}


def download_file(url: str, output_path: Path) -> bool:
    """Télécharge un fichier avec wget ou curl."""
    print(f"Téléchargement: {url}")
    print(f"Destination: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Essayer wget d'abord
    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(output_path), url],
            check=True
        )
        return output_path.exists()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Essayer curl
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", str(output_path), url],
            check=True
        )
        return output_path.exists()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Essayer requests Python
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
        return output_path.exists()
    except Exception as e:
        print(f"Erreur: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extrait un fichier ZIP."""
    print(f"Extraction: {zip_path}")
    print(f"Vers: {output_dir}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        print("✓ Extraction terminée")
        return True
    except Exception as e:
        print(f"Erreur extraction: {e}")
        return False


def download_pannuke(
    fold: int,
    output_dir: Path,
    keep_zip: bool = False
) -> bool:
    """Télécharge un fold de PanNuke."""
    config = DATASETS["pannuke"]

    if fold not in config["folds"]:
        print(f"Fold invalide: {fold}. Disponibles: {list(config['folds'].keys())}")
        return False

    filename = config["folds"][fold]
    url = f"{config['base_url']}/{filename}"
    zip_path = output_dir / filename
    extract_dir = output_dir / f"Fold{fold}"

    print(f"\n{'='*50}")
    print(f"PanNuke Fold {fold}")
    print(f"{'='*50}")

    # Vérifier si déjà extrait
    if extract_dir.exists() and (extract_dir / "images").exists():
        print(f"✓ Déjà téléchargé et extrait: {extract_dir}")
        return True

    # Télécharger
    if not zip_path.exists():
        success = download_file(url, zip_path)
        if not success:
            print("✗ Échec du téléchargement")
            print(f"  Téléchargez manuellement: {url}")
            return False

    # Extraire
    extract_dir.mkdir(parents=True, exist_ok=True)
    success = extract_zip(zip_path, extract_dir)

    if success and not keep_zip:
        print(f"Suppression du ZIP: {zip_path}")
        zip_path.unlink()

    return success


def list_datasets():
    """Affiche la liste des datasets disponibles."""
    print("\nDatasets disponibles:")
    print("-" * 70)
    for name, config in DATASETS.items():
        print(f"\n  {name.upper()}")
        print(f"    {config['description']}")
        if "size" in config:
            print(f"    Taille: {config['size']}")
        if "note" in config:
            print(f"    Note: {config['note']}")
    print("-" * 70)


def verify_pannuke(data_dir: Path, fold: int) -> dict:
    """Vérifie l'intégrité d'un fold PanNuke."""
    fold_dir = data_dir / f"Fold{fold}"

    result = {
        "fold": fold,
        "exists": fold_dir.exists(),
        "images": None,
        "masks": None,
        "types": None,
    }

    if not fold_dir.exists():
        return result

    # Vérifier les fichiers
    images_path = fold_dir / "images" / "images.npy"
    masks_path = fold_dir / "masks" / "masks.npy"
    types_path = fold_dir / "images" / "types.npy"

    if images_path.exists():
        import numpy as np
        images = np.load(images_path)
        result["images"] = {
            "shape": images.shape,
            "dtype": str(images.dtype),
        }

    if masks_path.exists():
        import numpy as np
        masks = np.load(masks_path)
        result["masks"] = {
            "shape": masks.shape,
            "dtype": str(masks.dtype),
        }

    if types_path.exists():
        import numpy as np
        types = np.load(types_path)
        result["types"] = {
            "shape": types.shape,
            "unique": len(np.unique(types)),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge les datasets pour CellViT-Optimus"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        help="Dataset à télécharger"
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3],
        help="Fold PanNuke à télécharger"
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Télécharge tous les folds PanNuke"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liste les datasets disponibles"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Vérifie l'intégrité des données téléchargées"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/pannuke",
        help="Répertoire de destination"
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Garde les fichiers ZIP après extraction"
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    output_dir = Path(args.output_dir)

    if args.verify:
        print("Vérification des données PanNuke...")
        for fold in [1, 2, 3]:
            result = verify_pannuke(output_dir, fold)
            print(f"\nFold {fold}:")
            if result["exists"]:
                if result["images"]:
                    print(f"  ✓ Images: {result['images']['shape']}")
                if result["masks"]:
                    print(f"  ✓ Masks: {result['masks']['shape']}")
                if result["types"]:
                    print(f"  ✓ Types: {result['types']['unique']} organes")
            else:
                print(f"  ✗ Non trouvé")
        return

    if args.dataset == "pannuke":
        if args.all_folds:
            for fold in [1, 2, 3]:
                download_pannuke(fold, output_dir, args.keep_zip)
        elif args.fold:
            download_pannuke(args.fold, output_dir, args.keep_zip)
        else:
            print("Spécifiez --fold ou --all-folds pour PanNuke")
            return
    elif args.dataset in ["monuseg", "consep"]:
        config = DATASETS[args.dataset]
        print(f"\n{args.dataset.upper()}")
        print(f"  {config['description']}")
        print(f"  URL: {config['url']}")
        print(f"  Note: {config.get('note', 'N/A')}")
        print("\n  Téléchargement manuel requis.")
    else:
        parser.print_help()
        print("\n")
        list_datasets()
        return

    print("\n✅ Terminé!")


if __name__ == "__main__":
    main()
