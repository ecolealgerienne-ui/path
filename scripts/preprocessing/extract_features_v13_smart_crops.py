#!/usr/bin/env python3
"""
Extrait les features H-optimus-0 depuis les fichiers V13 Smart Crops.

Ce script supporte l'extraction séparée des splits train/val pour prévenir
le data leakage et respecter la stratégie split-first-then-rotate.

Usage:
    # Train features
    python scripts/preprocessing/extract_features_v13_smart_crops.py \
        --family epidermal \
        --split train

    # Val features
    python scripts/preprocessing/extract_features_v13_smart_crops.py \
        --family epidermal \
        --split val
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import create_hoptimus_transform, validate_features
from src.models.loader import ModelLoader

def main():
    parser = argparse.ArgumentParser(
        description="Extraire features H-optimus-0 depuis V13 Smart Crops (split séparés)"
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"],
        help="Famille d'organes"
    )
    parser.add_argument(
        "--organ",
        type=str,
        default=None,
        help="Organe spécifique (optionnel). Si spécifié, charge {organ}_{split}.npz au lieu de {family}_{split}.npz"
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val"],
        help="Split à extraire (train ou val)"
    )
    parser.add_argument(
        "--data_dir",
        default="data/family_data_v13_smart_crops",
        help="Répertoire des données V13 Smart Crops"
    )
    parser.add_argument(
        "--output_dir",
        default="data/cache/family_data",
        help="Répertoire de sortie"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size pour extraction"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"]
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Préfixe pour les fichiers de données (organe ou famille)
    data_prefix = args.organ.lower() if args.organ else args.family

    # Chemins fichiers
    data_file = data_dir / f"{data_prefix}_{args.split}_v13_smart_crops.npz"
    features_file = output_dir / f"{data_prefix}_rgb_features_v13_smart_crops_{args.split}.npz"

    if not data_file.exists():
        print(f"❌ ERREUR: {data_file} introuvable")
        print(f"\nVous devez d'abord générer les données V13 Smart Crops:")
        if args.organ:
            print(f"  python scripts/preprocessing/prepare_v13_smart_crops.py --family {args.family} --organ {args.organ}")
        else:
            print(f"  python scripts/preprocessing/prepare_v13_smart_crops.py --family {args.family}")
        return 1

    print("=" * 80)
    organ_info = f" (Organe: {args.organ})" if args.organ else ""
    print(f"EXTRACTION FEATURES H-OPTIMUS-0: {data_prefix}{organ_info} ({args.split})")
    print("=" * 80)
    print("")

    # Charger données V13 Smart Crops
    print(f"Chargement {data_file.name}...")
    data = np.load(data_file)

    images = data['images']  # (N_crops, 224, 224, 3) uint8
    np_targets = data['np_targets']  # (N_crops, 224, 224) float32
    hv_targets = data['hv_targets']  # (N_crops, 2, 224, 224) float32
    nt_targets = data['nt_targets']  # (N_crops, 224, 224) int64
    source_image_ids = data['source_image_ids']  # (N_crops,) int32
    crop_positions = data['crop_positions']  # (N_crops,) str
    fold_ids = data['fold_ids']  # (N_crops,) int32

    n_crops = len(images)
    n_source_images = len(np.unique(source_image_ids))

    print(f"  → {n_crops} crops depuis {n_source_images} images sources")
    print(f"  → Images: {images.shape} ({images.dtype})")
    print(f"  → HV targets: {hv_targets.shape} ({hv_targets.dtype})")
    print(f"  → Source IDs: {source_image_ids.shape}")
    print(f"  → Crop positions: {crop_positions.shape}")

    # Vérifier HV dtype et range
    print(f"\nVérification HV targets:")
    print(f"  Dtype: {hv_targets.dtype}")
    print(f"  Range: [{hv_targets.min():.3f}, {hv_targets.max():.3f}]")

    if hv_targets.dtype != np.float32:
        print(f"  ⚠️  WARNING: HV dtype devrait être float32, pas {hv_targets.dtype}")
    if not (-1.0 <= hv_targets.min() <= hv_targets.max() <= 1.0):
        print(f"  ⚠️  WARNING: HV range devrait être [-1, 1]")
    else:
        print(f"  ✅ HV targets valides")

    # Vérification data leakage (informationnel)
    print(f"\n📊 Statistiques split:")
    print(f"  Split: {args.split}")
    print(f"  Source images uniques: {n_source_images}")
    print(f"  Crops totaux: {n_crops}")
    print(f"  Ratio amplification: {n_crops / n_source_images:.1f}×")
    print("")

    # Charger H-optimus-0
    print("Chargement H-optimus-0...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    backbone.eval()
    print(f"  → Device: {args.device}")
    print("")

    # Transform
    transform = create_hoptimus_transform()

    # Extraire features par batch
    print(f"Extraction features (batch_size={args.batch_size})...")
    all_features = []

    for i in tqdm(range(0, n_crops, args.batch_size), desc="Extraction"):
        batch_images = images[i:i+args.batch_size]
        batch_tensors = []

        # Préprocesser batch
        for img in batch_images:
            # Images sont déjà en uint8 [0, 255], resize à 224×224
            tensor = transform(img)  # (3, 224, 224) normalized
            batch_tensors.append(tensor)

        batch_tensor = torch.stack(batch_tensors).to(args.device)  # (B, 3, 224, 224)

        # Extraire features
        with torch.no_grad():
            features = backbone.forward_features(batch_tensor)  # (B, 261, 1536)

        all_features.append(features.cpu().numpy())

    # Concaténer
    all_features = np.concatenate(all_features, axis=0)  # (N_crops, 261, 1536)
    print(f"  → Features extraites: {all_features.shape}")
    print("")

    # Validation CLS std
    cls_tokens = all_features[:, 0, :]  # (N_crops, 1536)
    cls_std = cls_tokens.std()
    print(f"Validation features:")
    print(f"  CLS std: {cls_std:.4f} (attendu: 0.70-0.90)")

    if not (0.70 <= cls_std <= 0.90):
        print(f"  ⚠️  WARNING: CLS std hors range attendu!")
        print(f"  Cela indique un problème de preprocessing (Bug #1 ou #2)")
    else:
        print(f"  ✅ CLS std OK")
    print("")

    # Sauvegarder features avec metadata
    print(f"Sauvegarde features: {features_file.name}")
    np.savez_compressed(
        features_file,
        features=all_features,  # (N_crops, 261, 1536)
        source_image_ids=source_image_ids,  # Traceability
        crop_positions=crop_positions,  # Quel crop (center, top_left, etc.)
        fold_ids=fold_ids,  # Quel fold PanNuke original
        split=args.split,  # train ou val
        family=args.family  # Quelle famille
    )

    file_size_gb = features_file.stat().st_size / 1e9
    print(f"  → {file_size_gb:.2f} GB")
    print("")

    print("=" * 80)
    print("✅ EXTRACTION TERMINÉE")
    print("=" * 80)
    print("")
    print(f"Fichier créé:")
    print(f"  - {features_file}")
    print("")
    # Construire les flags pour les commandes suivantes
    organ_flag = f" --organ {args.organ}" if args.organ else ""

    print(f"Prochaine étape:")
    if args.split == "train":
        print(f"  1. Extraire features VAL:")
        print(f"     python scripts/preprocessing/extract_features_v13_smart_crops.py \\")
        print(f"         --family {args.family}{organ_flag} --split val")
        print(f"")
        print(f"  2. Lancer entraînement:")
        print(f"     python scripts/training/train_hovernet_family_v13_smart_crops.py \\")
        print(f"         --family {args.family}{organ_flag} --epochs 30")
    else:
        print(f"  Lancer entraînement:")
        print(f"     python scripts/training/train_hovernet_family_v13_smart_crops.py \\")
        print(f"         --family {args.family}{organ_flag} --epochs 30")
    print("")

    return 0

if __name__ == "__main__":
    sys.exit(main())
