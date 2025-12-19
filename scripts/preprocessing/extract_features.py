#!/usr/bin/env python3
"""
Script d'extraction de features avec H-optimus-0 sur PanNuke.

Ce script :
1. Charge les images PanNuke
2. Les prétraite pour H-optimus-0 (resize 224x224, normalisation)
3. Extrait les features des couches 6, 12, 18, 24 (pour UNETR)
4. Sauvegarde les features pour entraînement UNETR

Usage:
    python scripts/preprocessing/extract_features.py \
        --data_dir /home/amar/data/PanNuke \
        --fold 0 \
        --output_dir data/cache/pannuke_features
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

# Couches à extraire pour UNETR (0-indexed)
EXTRACT_LAYERS = [5, 11, 17, 23]  # Couches 6, 12, 18, 24


def load_pannuke_fold(data_dir: Path, fold: int = 0):
    """Charge un fold PanNuke complet (images + masks)."""
    fold_dir = data_dir / f"fold{fold}"

    images_path = fold_dir / "images.npy"
    masks_path = fold_dir / "masks.npy"
    types_path = fold_dir / "types.npy"

    if not images_path.exists():
        raise FileNotFoundError(f"Images non trouvées: {images_path}")

    print(f"Chargement fold{fold} depuis {fold_dir}...")
    images = np.load(images_path)
    masks = np.load(masks_path) if masks_path.exists() else None
    types = np.load(types_path) if types_path.exists() else None

    print(f"  → {len(images)} images")
    print(f"  → Shape images: {images.shape}")
    if masks is not None:
        print(f"  → Shape masks: {masks.shape}")

    return images, masks, types


def create_transform():
    """Crée la transformation pour H-optimus-0."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])


class HOptimusFeatureExtractor:
    """Extracteur de features multi-couches pour H-optimus-0."""

    def __init__(self, device="cuda"):
        print("Chargement de H-optimus-0...")
        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False
        )
        self.model = self.model.eval().to(device).half()
        self.device = device

        # Hooks pour extraire les features intermédiaires
        self.features = {}
        self._register_hooks()

        num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"  → Modèle chargé: {num_params:.2f}B paramètres")
        print(f"  → Couches extraites: {[i+1 for i in EXTRACT_LAYERS]}")

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.cpu().float()
            return hook

        for idx in EXTRACT_LAYERS:
            self.model.blocks[idx].register_forward_hook(get_hook(f'layer_{idx}'))

    @torch.no_grad()
    def extract(self, batch: torch.Tensor):
        """Extrait les features des 4 couches."""
        self.features = {}
        batch = batch.to(self.device).half()
        _ = self.model.forward_features(batch)

        return {
            'layer_6': self.features['layer_5'].numpy(),
            'layer_12': self.features['layer_11'].numpy(),
            'layer_18': self.features['layer_17'].numpy(),
            'layer_24': self.features['layer_23'].numpy(),
        }


@torch.no_grad()
def extract_features(
    extractor: HOptimusFeatureExtractor,
    images: np.ndarray,
    transform,
    batch_size: int = 8,
):
    """Extrait les features multi-couches pour toutes les images."""
    n_images = len(images)

    all_features = {
        'layer_6': [],
        'layer_12': [],
        'layer_18': [],
        'layer_24': [],
    }

    print(f"Extraction de features pour {n_images} images...")

    for i in tqdm(range(0, n_images, batch_size)):
        batch_images = images[i:i + batch_size]

        # Prétraitement
        batch_tensors = torch.stack([transform(img) for img in batch_images])

        # Extraction
        features = extractor.extract(batch_tensors)

        for key in all_features:
            all_features[key].append(features[key])

    # Concaténer
    for key in all_features:
        all_features[key] = np.concatenate(all_features[key], axis=0)
        print(f"  → {key}: {all_features[key].shape}")

    return all_features


def main():
    parser = argparse.ArgumentParser(description="Extraction features H-optimus-0 pour UNETR")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Chemin vers PanNuke (ex: /home/amar/data/PanNuke)")
    parser.add_argument("--output_dir", type=str, default="data/cache/pannuke_features",
                        help="Chemin de sortie pour les features")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold PanNuke (0, 1, ou 2)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Taille des batchs (8 recommandé pour 12GB VRAM)")
    args = parser.parse_args()

    # Chemins
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger les données
    images, masks, types = load_pannuke_fold(data_dir, fold=args.fold)

    # Créer l'extracteur
    extractor = HOptimusFeatureExtractor()
    transform = create_transform()

    # Mesurer le temps
    start_time = time.time()

    # Extraire les features multi-couches
    features = extract_features(
        extractor,
        images,
        transform,
        batch_size=args.batch_size,
    )

    elapsed = time.time() - start_time

    # Stats
    n_images = len(images)
    print("\n" + "=" * 50)
    print("RÉSUMÉ EXTRACTION")
    print("=" * 50)
    print(f"Images traitées: {n_images}")
    print(f"Temps total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Temps par image: {elapsed / n_images * 1000:.1f}ms")

    # Sauvegarder les features (format NPZ sans compression - plus rapide)
    output_path = output_dir / f"fold{args.fold}_features.npz"
    print(f"\nSauvegarde features (sans compression)...")
    np.savez(
        output_path,
        layer_6=features['layer_6'],
        layer_12=features['layer_12'],
        layer_18=features['layer_18'],
        layer_24=features['layer_24'],
    )
    print(f"Features sauvegardées: {output_path}")

    # Sauvegarder les masks (pour entraînement)
    if masks is not None:
        masks_path = output_dir / f"fold{args.fold}_masks.npy"
        np.save(masks_path, masks)
        print(f"Masks sauvegardés: {masks_path}")

    # Taille des fichiers
    features_size = output_path.stat().st_size / 1e9
    print(f"\nTaille features: {features_size:.2f} GB")

    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"Pic mémoire GPU: {mem_used:.2f} GB")


if __name__ == "__main__":
    main()
