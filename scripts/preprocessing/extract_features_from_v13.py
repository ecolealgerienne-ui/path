#!/usr/bin/env python3
"""
Extraction Features H-optimus-0 depuis Multi-Crops V13.

Charge les crops pré-générés (224×224) et extrait les features du backbone gelé.

⚠️ IMPORTANT: Utilise le preprocessing centralisé (src.preprocessing) pour
garantir cohérence avec l'entraînement.

Usage:
    python scripts/preprocessing/extract_features_from_v13.py \
        --input_file data/family_V13/epidermal_data_v13_crops.npz \
        --output_dir data/cache/family_features_v13 \
        --family epidermal \
        --batch_size 16 \
        --chunk_size 500
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import HOPTIMUS_INPUT_SIZE
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform, validate_features


def extract_features_batch(
    backbone: torch.nn.Module,
    images: np.ndarray,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extrait features H-optimus-0 pour un batch d'images.

    Args:
        backbone: Modèle H-optimus-0
        images: Images (B, 224, 224, 3) uint8 [0-255]
        device: Device

    Returns:
        Features (B, 261, 1536) float32
    """
    transform = create_hoptimus_transform()
    batch_features = []

    for img in images:
        # ✅ Preprocessing centralisé (garantit cohérence avec training)
        # Images V13 sont déjà 224×224, le transform gère la normalisation
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # ✅ forward_features() inclut LayerNorm final
            features = backbone.forward_features(tensor)

        batch_features.append(features.cpu().float().numpy())

    return np.vstack(batch_features)


def extract_features_chunked(
    input_file: Path,
    output_dir: Path,
    family: str,
    batch_size: int = 16,
    chunk_size: int = 500,
    device: str = "cuda"
) -> Dict[str, int]:
    """
    Extrait features H-optimus-0 depuis crops V13 avec chunking pour économiser RAM.

    Args:
        input_file: Fichier .npz V13 (crops 224×224)
        output_dir: Répertoire de sortie
        family: Nom de la famille
        batch_size: Taille des batches GPU
        chunk_size: Nombre de crops par chunk (économie RAM)
        device: Device

    Returns:
        Statistiques d'extraction
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTION FEATURES H-OPTIMUS-0 V13 - Famille: {family.upper()}")
    print(f"{'='*70}\n")

    # 1. Charger métadonnées (memory-mapped pour économiser RAM)
    print(f"📂 Chargement métadonnées: {input_file}")
    data = np.load(input_file)

    images_mmap = data['images']  # Memory-mapped, pas chargé en RAM
    source_ids = data['source_image_ids']
    crop_positions = data['crop_positions']
    fold_ids = data['fold_ids']

    n_crops = len(images_mmap)
    print(f"✅ {n_crops} crops détectés")

    # Validation shapes
    print(f"\n📏 Validation des shapes:")
    print(f"  images: {images_mmap.shape} (attendu: (N, 224, 224, 3))")
    assert images_mmap.shape[1:] == (HOPTIMUS_INPUT_SIZE, HOPTIMUS_INPUT_SIZE, 3), \
        f"Shape images invalide: {images_mmap.shape}"
    print(f"✅ Shape images correcte")

    # 2. Charger backbone H-optimus-0
    print(f"\n🔧 Chargement H-optimus-0 (backbone gelé)...")
    backbone = ModelLoader.load_hoptimus0(device=device)
    print(f"✅ Backbone chargé sur {device}")

    # 3. Extraction par chunks (économie RAM)
    print(f"\n🚀 Extraction features par chunks (chunk_size={chunk_size})...")

    all_features = []
    n_chunks = (n_crops + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_crops)
        chunk_n = end_idx - start_idx

        print(f"\n📦 Chunk {chunk_idx + 1}/{n_chunks} ({chunk_n} crops)")

        # Charger chunk en mémoire
        images_chunk = images_mmap[start_idx:end_idx].copy()

        # Extraction par batches GPU
        chunk_features = []
        n_batches = (chunk_n + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc=f"  Extraction batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, chunk_n)

            images_batch = images_chunk[batch_start:batch_end]
            features_batch = extract_features_batch(backbone, images_batch, device)

            chunk_features.append(features_batch)

        # Concaténer features du chunk
        chunk_features = np.vstack(chunk_features)
        all_features.append(chunk_features)

        # Libérer mémoire chunk
        del images_chunk
        del chunk_features

    # 4. Concaténer tous les chunks
    print(f"\n📦 Concaténation de {n_chunks} chunks...")
    all_features = np.vstack(all_features)

    print(f"✅ Features extraites: {all_features.shape}")
    assert all_features.shape == (n_crops, 261, 1536), f"Shape invalide: {all_features.shape}"

    # 5. Validation CLS std (CRITIQUE)
    print(f"\n🔍 Validation CLS std (détection bugs preprocessing)...")

    cls_tokens = all_features[:, 0, :]  # (N, 1536)
    validation_result = validate_features(torch.from_numpy(cls_tokens))

    if not validation_result['valid']:
        print(f"❌ ERREUR: {validation_result['message']}")
        print(f"   CLS std: {validation_result['cls_std']:.4f}")
        print(f"   Attendu: [{validation_result['expected_range'][0]}, {validation_result['expected_range'][1]}]")
        sys.exit(1)

    print(f"✅ CLS std: {validation_result['cls_std']:.4f} (valide)")

    # 6. Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{family}_features_v13.npz"

    print(f"\n💾 Sauvegarde: {output_file}")
    np.savez_compressed(
        output_file,
        features=all_features,
        source_image_ids=source_ids,
        crop_positions=crop_positions,
        fold_ids=fold_ids,
    )

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Fichier créé: {file_size_mb:.1f} MB")

    # 7. Statistiques
    print(f"\n{'='*70}")
    print(f"STATISTIQUES D'EXTRACTION")
    print(f"{'='*70}\n")

    print(f"Crops traités:         {n_crops}")
    print(f"Features shape:        {all_features.shape}")
    print(f"CLS std:               {validation_result['cls_std']:.4f}")
    print(f"Dtype:                 {all_features.dtype}")
    print(f"Taille fichier:        {file_size_mb:.1f} MB")

    print(f"\n📊 Répartition par position:")
    unique_positions, counts = np.unique(crop_positions, return_counts=True)
    for pos, count in zip(unique_positions, counts):
        pct = 100 * count / n_crops
        print(f"  {pos:15s}: {count:4d} ({pct:5.1f}%)")

    return {
        'n_crops': n_crops,
        'cls_std': validation_result['cls_std'],
        'file_size_mb': file_size_mb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extraction Features H-optimus-0 depuis Multi-Crops V13"
    )
    parser.add_argument(
        '--input_file',
        type=Path,
        required=True,
        help="Fichier .npz V13 (crops 224×224)"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('data/cache/family_features_v13'),
        help="Répertoire de sortie pour features"
    )
    parser.add_argument(
        '--family',
        type=str,
        required=True,
        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
        help="Famille tissulaire"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="Taille des batches GPU (défaut: 16)"
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=500,
        help="Nombre de crops par chunk RAM (défaut: 500)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    # Validation fichier d'entrée
    if not args.input_file.exists():
        print(f"❌ ERREUR: Fichier introuvable: {args.input_file}")
        sys.exit(1)

    # Extraction
    stats = extract_features_chunked(
        input_file=args.input_file,
        output_dir=args.output_dir,
        family=args.family,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=args.device,
    )

    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION COMPLÈTE - {stats['n_crops']} crops → features")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
