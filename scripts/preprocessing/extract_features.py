#!/usr/bin/env python3
"""
Extraction des features H-optimus-0 pour PanNuke.

IMPORTANT - SOLUTION CIBLE (2025-12-21):
=========================================
Ce script utilise forward_features() qui inclut le LayerNorm final.
Toute l'inférence doit aussi utiliser forward_features() pour cohérence.

Version optimisée pour faible consommation RAM:
- Utilise mmap pour ne pas charger toutes les images en RAM
- Traite par chunks et sauvegarde incrémentalement
- Libère la mémoire entre les chunks

Usage:
    python scripts/preprocessing/extract_features.py \
        --data_dir /home/amar/data/PanNuke \
        --fold 0 --batch_size 8

Vérification attendue:
    - CLS token std: ~0.75-0.85 (avec LayerNorm)
    - Si std ~0.28, c'est SANS LayerNorm = ERREUR
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

# Valeurs de référence pour vérification (avec LayerNorm final)
EXPECTED_CLS_STD_MIN = 0.70  # Minimum attendu
EXPECTED_CLS_STD_MAX = 0.90  # Maximum attendu


def create_transform():
    """Crée la transformation pour H-optimus-0."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])


class HOptimusFeatureExtractor:
    """
    Extracteur de features pour H-optimus-0.

    IMPORTANT: Utilise forward_features() qui inclut le LayerNorm final.
    C'est l'API officielle de timm/ViT et garantit des features normalisées.

    Les features extraites ont un CLS token avec std ~0.75-0.85.
    Si std ~0.28, c'est une erreur (LayerNorm manquant).
    """

    def __init__(self, device="cuda"):
        print("Chargement de H-optimus-0...")
        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False
        )
        self.model = self.model.eval().to(device)
        self.device = device

        num_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"  → Modèle chargé: {num_params:.2f}B paramètres")
        print(f"  → Méthode: forward_features() (avec LayerNorm final)")

    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> np.ndarray:
        """
        Extrait les features via forward_features().

        Args:
            batch: Tensor (B, 3, 224, 224)

        Returns:
            Features (B, 261, 1536) - CLS token + 256 patch tokens
        """
        batch = batch.to(self.device)

        # forward_features() inclut le LayerNorm final
        features = self.model.forward_features(batch)

        return features.cpu().float().numpy()

    def verify_features(self, features: np.ndarray) -> bool:
        """
        Vérifie que les features sont cohérentes (LayerNorm appliqué).

        Args:
            features: (B, 261, 1536)

        Returns:
            True si les features sont valides
        """
        cls_tokens = features[:, 0, :]  # (B, 1536)
        std = cls_tokens.std()

        if std < EXPECTED_CLS_STD_MIN:
            print(f"  ⚠️ ATTENTION: CLS std={std:.4f} < {EXPECTED_CLS_STD_MIN}")
            print(f"     → Cela suggère que le LayerNorm n'est PAS appliqué!")
            return False
        elif std > EXPECTED_CLS_STD_MAX:
            print(f"  ⚠️ ATTENTION: CLS std={std:.4f} > {EXPECTED_CLS_STD_MAX}")
            return False
        else:
            print(f"  ✓ CLS std={std:.4f} (attendu: {EXPECTED_CLS_STD_MIN}-{EXPECTED_CLS_STD_MAX})")
            return True


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

    IMPORTANT: Utilise forward_features() qui inclut le LayerNorm final.
    Les features ont un CLS token avec std ~0.75-0.85.

    Sauvegarde incrémentalement dans un fichier temporaire.
    """
    n_images = len(images_mmap)
    n_chunks = (n_images + chunk_size - 1) // chunk_size

    print(f"Extraction de features pour {n_images} images...")
    print(f"  → {n_chunks} chunks de {chunk_size} images max")
    print(f"  → Batch size: {batch_size}")
    print(f"  → Méthode: forward_features() (avec LayerNorm final)")

    # Premier batch pour déterminer la shape
    first_img = images_mmap[0]
    if first_img.dtype != np.uint8:
        first_img = first_img.clip(0, 255).astype(np.uint8)
    first_tensor = transform(first_img).unsqueeze(0)
    first_features = extractor.extract(first_tensor)

    feature_shape = first_features.shape[1:]  # (261, 1536)
    print(f"  → Shape features: {feature_shape}")

    # Vérifier les features du premier batch
    extractor.verify_features(first_features)

    # Pré-allouer l'array de sortie sur disque (memory-mapped)
    temp_path = output_path.parent / "_temp_features.npy"
    all_features = np.lib.format.open_memmap(
        temp_path,
        mode='w+',
        dtype=np.float32,
        shape=(n_images,) + feature_shape
    )

    # Stocker les premiers features
    all_features[0:1] = first_features

    # Traiter par chunks
    global_idx = 1  # On a déjà traité la première image

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_images)

        # Skip la première image déjà traitée
        if start == 0:
            start = 1
            if start >= end:
                continue

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

            # Extraction via forward_features()
            features = extractor.extract(batch_tensors)

            # Sauvegarder dans l'array memory-mapped
            all_features[global_idx:global_idx + len(batch_images)] = features

            global_idx += len(batch_images)

        # Libérer la mémoire du chunk
        del chunk_images
        gc.collect()
        torch.cuda.empty_cache()

    # Forcer l'écriture sur disque
    all_features.flush()

    # Vérification finale sur un échantillon
    print(f"\n✅ Extraction terminée: {global_idx} images")
    print("\n🔍 Vérification finale...")
    sample_indices = np.random.choice(n_images, min(100, n_images), replace=False)
    sample_features = all_features[sample_indices]
    extractor.verify_features(sample_features)

    return all_features, temp_path


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

    # Créer l'extracteur (utilise forward_features avec LayerNorm)
    extractor = HOptimusFeatureExtractor()
    transform = create_transform()

    # Mesurer le temps
    start_time = time.time()

    # Extraire les features par chunks
    output_path = output_dir / f"fold{args.fold}_features.npz"
    all_features, temp_path = extract_features_chunked(
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

    # Convertir le mmap en array normal pour sauvegarde NPZ
    print(f"\nConsolidation des features...")
    features_array = np.array(all_features)
    print(f"  → features: {features_array.shape}")

    # Calculer les statistiques CLS pour vérification
    cls_tokens = features_array[:, 0, :]  # (N, 1536)
    cls_std = cls_tokens.std()
    cls_mean = cls_tokens.mean()
    print(f"  → CLS token stats: mean={cls_mean:.4f}, std={cls_std:.4f}")

    if cls_std < EXPECTED_CLS_STD_MIN or cls_std > EXPECTED_CLS_STD_MAX:
        print(f"  ⚠️ ATTENTION: CLS std hors plage attendue [{EXPECTED_CLS_STD_MIN}, {EXPECTED_CLS_STD_MAX}]!")
    else:
        print(f"  ✓ CLS std dans la plage attendue")

    # Sauvegarder avec clé 'features' pour compatibilité
    print(f"\nSauvegarde vers {output_path}...")
    np.savez(output_path, features=features_array)
    print(f"✅ Features sauvegardées: {output_path}")

    # Nettoyer le fichier temporaire
    print("Nettoyage des fichiers temporaires...")
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
        if key == 'features':
            cls = arr[:, 0, :]
            print(f"    CLS std: {cls.std():.4f}")

    print(f"\n🎉 Extraction fold{args.fold} terminée!")


if __name__ == "__main__":
    main()
