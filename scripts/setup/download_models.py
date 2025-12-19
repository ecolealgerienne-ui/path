#!/usr/bin/env python3
"""
Téléchargement des modèles pré-entraînés.

Modèles disponibles:
- CellViT-256: Segmentation cellulaire légère (~4-6 GB VRAM)
- CellViT-SAM-H: Segmentation haute précision (~24 GB VRAM)
- H-optimus-0: Backbone foundation (via HuggingFace)

Usage:
    python scripts/setup/download_models.py --model cellvit-256
    python scripts/setup/download_models.py --all
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional


# Configuration des modèles
MODELS = {
    "cellvit-256": {
        "gdrive_id": "1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q",
        "filename": "CellViT-256.pth",
        "description": "CellViT avec ViT-256 encoder (~22M params)",
        "vram": "4-6 GB",
    },
    "cellvit-256-x20": {
        "gdrive_id": "1WRgz_ViIBSzfVDMKb7P89wIf6He3VKfH",
        "filename": "CellViT-256-x20.pth",
        "description": "CellViT-256 pour images 20x",
        "vram": "4-6 GB",
    },
    "cellvit-sam-h": {
        "gdrive_id": "1w4gxiX1abPjHdNq4pIPqNwhbkyJT3Igt",
        "filename": "CellViT-SAM-H.pth",
        "description": "CellViT avec SAM ViT-H encoder (~632M params)",
        "vram": "24+ GB",
    },
}


def check_gdown():
    """Vérifie si gdown est installé."""
    try:
        import gdown
        return True
    except ImportError:
        return False


def install_gdown():
    """Installe gdown."""
    print("Installation de gdown...")
    subprocess.run(["pip", "install", "gdown", "-q"], check=True)
    print("✓ gdown installé")


def download_from_gdrive(file_id: str, output_path: Path) -> bool:
    """Télécharge un fichier depuis Google Drive."""
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Téléchargement depuis Google Drive...")
    print(f"  ID: {file_id}")
    print(f"  Destination: {output_path}")

    try:
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists()
    except Exception as e:
        print(f"Erreur: {e}")
        return False


def download_model(
    model_name: str,
    output_dir: Path,
    force: bool = False
) -> bool:
    """Télécharge un modèle spécifique."""
    if model_name not in MODELS:
        print(f"Modèle inconnu: {model_name}")
        print(f"Modèles disponibles: {list(MODELS.keys())}")
        return False

    config = MODELS[model_name]
    output_path = output_dir / config["filename"]

    print(f"\n{'='*50}")
    print(f"Modèle: {model_name}")
    print(f"Description: {config['description']}")
    print(f"VRAM requise: {config['vram']}")
    print(f"{'='*50}")

    if output_path.exists() and not force:
        print(f"✓ Déjà téléchargé: {output_path}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    success = download_from_gdrive(config["gdrive_id"], output_path)

    if success:
        print(f"✓ Téléchargé: {output_path}")
        print(f"  Taille: {output_path.stat().st_size / 1e6:.1f} MB")
    else:
        print(f"✗ Échec du téléchargement")

    return success


def download_hoptimus(cache_dir: Optional[Path] = None):
    """Télécharge H-optimus-0 via HuggingFace."""
    print("\n" + "="*50)
    print("Téléchargement H-optimus-0 (via HuggingFace)")
    print("="*50)

    try:
        import timm
        print("Chargement du modèle (téléchargement si nécessaire)...")
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        print(f"✓ H-optimus-0 chargé ({sum(p.numel() for p in model.parameters()):,} params)")
        return True
    except Exception as e:
        print(f"✗ Erreur: {e}")
        print("  Vérifiez votre connexion HuggingFace: huggingface-cli login")
        return False


def list_models():
    """Affiche la liste des modèles disponibles."""
    print("\nModèles disponibles:")
    print("-" * 60)
    for name, config in MODELS.items():
        print(f"  {name:<20} | VRAM: {config['vram']:<10} | {config['description']}")
    print("-" * 60)
    print("  h-optimus-0          | VRAM: 4-6 GB    | Backbone foundation (HuggingFace)")


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge les modèles pré-entraînés"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()) + ["h-optimus-0"],
        help="Modèle à télécharger"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Télécharge tous les modèles"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liste les modèles disponibles"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/pretrained",
        help="Répertoire de destination"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force le re-téléchargement"
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.model and not args.all:
        parser.print_help()
        print("\n")
        list_models()
        return

    # Vérifier/installer gdown
    if not check_gdown():
        install_gdown()

    output_dir = Path(args.output_dir)

    if args.all:
        print("Téléchargement de tous les modèles...")
        for model_name in MODELS:
            download_model(model_name, output_dir, args.force)
        download_hoptimus()
    elif args.model == "h-optimus-0":
        download_hoptimus()
    else:
        download_model(args.model, output_dir, args.force)

    print("\n✅ Terminé!")


if __name__ == "__main__":
    main()
