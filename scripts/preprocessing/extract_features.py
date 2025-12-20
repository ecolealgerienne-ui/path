#!/usr/bin/env python3
"""
Extraction des features H-optimus-0 pour PanNuke.

Version optimisée pour faible consommation RAM:
- Utilise mmap pour ne pas charger toutes les images en RAM
- Traite par chunks et sauvegarde incrémentalement
- Libère la mémoire entre les chunks

Usage:
    python scripts/preprocessing/extract_features.py \
        --data_dir /home/amar/data/PanNuke \
        --fold 0 --all_layers --batch_size 8
"""

import argparse
import time
import gc
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Installez timm: pip install timm")

# Normalisation spécifique H-optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

# Couches à extraire pour UNETR (0-indexed)
EXTRACT_LAYERS = [5, 11, 17, 23]  # Couches 6, 12, 18, 24


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

    def __init__(self, device="cuda", all_layers=False):
        print("Chargement de H-optimus-0...")
        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False
        )
        self.model = self.model.eval().to(device).half()
        self.device = device
        self.all_layers = all_layers

        if all_layers:
            self.extract_layers = [5, 11, 17, 23]
        else:
            self.extract_layers = [23]

        self.features = {}
        self._register_hooks()

        num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"  → Modèle chargé: {num_params:.2f}B paramètres")
        print(f"  → Couches extraites: {[i+1 for i in self.extract_layers]}")

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.cpu().float()
            return hook

        for idx in self.extract_layers:
            self.model.blocks[idx].register_forward_hook(get_hook(f'layer_{idx}'))

    @torch.no_grad()
    def extract(self, batch: torch.Tensor):
        """Extrait les features des couches configurées."""
        self.features = {}
        batch = batch.to(self.device).half()
        _ = self.model.forward_features(batch)

        if self.all_layers:
            return {
                'layer_6': self.features['layer_5'].numpy(),
                'layer_12': self.features['layer_11'].numpy(),
                'layer_18': self.features['layer_17'].numpy(),
                'layer_24': self.features['layer_23'].numpy(),
            }
        else:
            return {
                'layer_24': self.features['layer_23'].numpy(),
            }


def extract_features_chunked(
    extractor: HOptimusFeatureExtractor,
    images_mmap: np.ndarray,
    transform,
    output_path: Path,
    batch_size: int = 8,
    chunk_size: int = 500,
):
    """
    Extrait les features par chunks pour économiser la RAM.

    Sauvegarde incrémentalement dans un fichier temporaire.
    """
    n_images = len(images_mmap)
    n_chunks = (n_images + chunk_size - 1) // chunk_size

    print(f"Extraction de features pour {n_images} images...")
    print(f"  → {n_chunks} chunks de {chunk_size} images max")
    print(f"  → Batch size: {batch_size}")

    # Déterminer les clés de features
    if extractor.all_layers:
        feature_keys = ['layer_6', 'layer_12', 'layer_18', 'layer_24']
    else:
        feature_keys = ['layer_24']

    # Premier batch pour déterminer la shape
    first_img = images_mmap[0]
    if first_img.dtype != np.uint8:
        first_img = first_img.clip(0, 255).astype(np.uint8)
    first_tensor = transform(first_img).unsqueeze(0)
    first_features = extractor.extract(first_tensor)

    feature_shape = first_features[feature_keys[0]].shape[1:]  # (261, 1536)
    print(f"  → Shape features: {feature_shape}")

    # Pré-allouer les arrays de sortie sur disque (memory-mapped)
    temp_files = {}
    all_features = {}

    for key in feature_keys:
        temp_path = output_path.parent / f"_temp_{key}.npy"
        arr = np.lib.format.open_memmap(
            temp_path,
            mode='w+',
            dtype=np.float32,
            shape=(n_images,) + feature_shape
        )
        temp_files[key] = temp_path
        all_features[key] = arr

    # Traiter par chunks
    global_idx = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_images)

        print(f"\n📦 Chunk {chunk_idx+1}/{n_chunks} [{start}:{end}]")

        # Charger le chunk en RAM (seulement ce chunk)
        chunk_images = np.array(images_mmap[start:end])

        # Traiter par batches
        for i in tqdm(range(0, len(chunk_images), batch_size), desc="  Batches"):
            batch_end = min(i + batch_size, len(chunk_images))
            batch_images = chunk_images[i:batch_end]

            # Convertir en uint8 pour ToPILImage
            batch_images_uint8 = []
            for img in batch_images:
                if img.dtype != np.uint8:
                    img = img.clip(0, 255).astype(np.uint8)
                batch_images_uint8.append(img)

            # Prétraitement
            batch_tensors = torch.stack([transform(img) for img in batch_images_uint8])

            # Extraction
            features = extractor.extract(batch_tensors)

            # Sauvegarder dans les arrays memory-mapped
            for key in feature_keys:
                all_features[key][global_idx:global_idx + len(batch_images)] = features[key]

            global_idx += len(batch_images)

        # Libérer la mémoire du chunk
        del chunk_images
        gc.collect()
        torch.cuda.empty_cache()

    # Forcer l'écriture sur disque
    for key in feature_keys:
        all_features[key].flush()

    print(f"\n✅ Extraction terminée: {global_idx} images")

    return all_features, temp_files


def main():
    parser = argparse.ArgumentParser(description="Extraction features H-optimus-0")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Chemin vers PanNuke (ex: /home/amar/data/PanNuke)")
    parser.add_argument("--output_dir", type=str, default="data/cache/pannuke_features",
                        help="Chemin de sortie pour les features")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold PanNuke (0, 1, ou 2)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Taille des batchs (8 recommandé pour 12GB VRAM)")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Nombre d'images par chunk RAM (500 = ~1GB)")
    parser.add_argument("--all_layers", action="store_true",
                        help="Extraire 4 couches (6,12,18,24) au lieu de layer_24 seule")
    args = parser.parse_args()

    # Chemins
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dir = data_dir / f"fold{args.fold}"
    images_path = fold_dir / "images.npy"

    if not images_path.exists():
        raise FileNotFoundError(f"Images non trouvées: {images_path}")

    # Charger les images en mode memory-mapped (ne charge PAS en RAM)
    print(f"Ouverture de {images_path} en mode mmap...")
    images_mmap = np.load(images_path, mmap_mode='r')
    print(f"  → {len(images_mmap)} images, shape: {images_mmap.shape}")
    print(f"  → dtype: {images_mmap.dtype}")

    # Créer l'extracteur
    extractor = HOptimusFeatureExtractor(all_layers=args.all_layers)
    transform = create_transform()

    # Mesurer le temps
    start_time = time.time()

    # Extraire les features par chunks
    output_path = output_dir / f"fold{args.fold}_features.npz"
    all_features, temp_files = extract_features_chunked(
        extractor,
        images_mmap,
        transform,
        output_path,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    elapsed = time.time() - start_time

    # Stats
    n_images = len(images_mmap)
    print("\n" + "=" * 50)
    print("RÉSUMÉ EXTRACTION")
    print("=" * 50)
    print(f"Images traitées: {n_images}")
    print(f"Temps total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Temps par image: {elapsed / n_images * 1000:.1f}ms")

    # Convertir les mmap en arrays normaux pour sauvegarde NPZ
    print(f"\nConsolidation des features...")
    final_features = {}
    for key in all_features:
        # Lire depuis le mmap
        final_features[key] = np.array(all_features[key])
        print(f"  → {key}: {final_features[key].shape}")

    # Sauvegarder
    print(f"\nSauvegarde vers {output_path}...")
    np.savez(output_path, **final_features)
    print(f"✅ Features sauvegardées: {output_path}")

    # Nettoyer les fichiers temporaires
    print("Nettoyage des fichiers temporaires...")
    for key, temp_path in temp_files.items():
        if temp_path.exists():
            temp_path.unlink()
            print(f"  🗑️  {temp_path.name}")

    # Vérification finale
    print("\n" + "=" * 50)
    print("VÉRIFICATION")
    print("=" * 50)
    loaded = np.load(output_path)
    for key in loaded.files:
        arr = loaded[key]
        print(f"  {key}: {arr.shape}, dtype={arr.dtype}")
        print(f"    range: [{arr.min():.4f}, {arr.max():.4f}]")

    print(f"\n🎉 Extraction fold{args.fold} terminée!")


if __name__ == "__main__":
    main()
