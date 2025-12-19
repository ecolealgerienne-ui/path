#!/usr/bin/env python3
"""
Script d'extraction de features avec H-optimus-0 sur PanNuke.

Ce script :
1. Charge les images PanNuke
2. Les prétraite pour H-optimus-0 (resize 224x224, normalisation)
3. Extrait les embeddings 1536-dim
4. Sauvegarde les features pour usage ultérieur

Usage:
    python scripts/preprocessing/extract_features.py --num_images 100
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from tqdm import tqdm


# Normalisation spécifique H-optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


def load_pannuke_images(data_dir: Path, fold: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Charge les images et types de PanNuke."""
    fold_dir = data_dir / f"Fold {fold}" / "images" / f"fold{fold}"

    images_path = fold_dir / "images.npy"
    types_path = fold_dir / "types.npy"

    print(f"Chargement des images depuis {images_path}...")
    images = np.load(images_path)
    types = np.load(types_path)

    print(f"  → {len(images)} images chargées")
    print(f"  → Shape: {images.shape}")
    print(f"  → Types uniques: {np.unique(types)}")

    return images, types


def create_transform():
    """Crée la transformation pour H-optimus-0."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # H-optimus-0 attend 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])


def load_model():
    """Charge H-optimus-0 en mode évaluation."""
    print("Chargement de H-optimus-0...")
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False
    )
    model = model.eval().cuda().half()

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  → Modèle chargé: {num_params:.2f}B paramètres")

    return model


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    images: np.ndarray,
    transform: transforms.Compose,
    batch_size: int = 16,
    num_images: int = None
) -> np.ndarray:
    """Extrait les features pour toutes les images."""

    if num_images is not None:
        images = images[:num_images]

    n_images = len(images)
    features = []

    print(f"Extraction de features pour {n_images} images...")

    # Traitement par batch
    for i in tqdm(range(0, n_images, batch_size)):
        batch_images = images[i:i + batch_size]

        # Prétraitement
        batch_tensors = []
        for img in batch_images:
            tensor = transform(img)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).cuda().half()

        # Extraction
        out = model(batch)
        features.append(out.cpu().float().numpy())

    features = np.concatenate(features, axis=0)
    print(f"  → Features extraites: {features.shape}")

    return features


def main():
    parser = argparse.ArgumentParser(description="Extraction de features H-optimus-0")
    parser.add_argument("--data_dir", type=str, default="data/raw/pannuke",
                        help="Chemin vers les données PanNuke")
    parser.add_argument("--output_dir", type=str, default="data/cache",
                        help="Chemin de sortie pour les features")
    parser.add_argument("--num_images", type=int, default=100,
                        help="Nombre d'images à traiter (None = toutes)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Taille des batchs")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold PanNuke à utiliser")
    args = parser.parse_args()

    # Chemins
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger les images
    images, types = load_pannuke_images(data_dir, fold=args.fold)

    # Charger le modèle
    model = load_model()

    # Créer la transformation
    transform = create_transform()

    # Mesurer le temps
    start_time = time.time()

    # Extraire les features
    features = extract_features(
        model,
        images,
        transform,
        batch_size=args.batch_size,
        num_images=args.num_images
    )

    elapsed = time.time() - start_time

    # Stats
    print("\n" + "=" * 50)
    print("RÉSUMÉ")
    print("=" * 50)
    print(f"Images traitées: {len(features)}")
    print(f"Shape features: {features.shape}")
    print(f"Temps total: {elapsed:.1f}s")
    print(f"Temps par image: {elapsed / len(features) * 1000:.1f}ms")
    print(f"Images par seconde: {len(features) / elapsed:.1f}")

    # Sauvegarder
    output_path = output_dir / f"pannuke_fold{args.fold}_features.npy"
    np.save(output_path, features)
    print(f"\nFeatures sauvegardées: {output_path}")

    # Sauvegarder aussi les types correspondants
    types_subset = types[:args.num_images] if args.num_images else types
    types_path = output_dir / f"pannuke_fold{args.fold}_types.npy"
    np.save(types_path, types_subset)
    print(f"Types sauvegardés: {types_path}")

    # Vérification mémoire GPU
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPic mémoire GPU: {mem_used:.2f} GB")


if __name__ == "__main__":
    main()
