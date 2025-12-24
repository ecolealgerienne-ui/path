#!/usr/bin/env python3
"""
Extrait les features H-optimus-0 depuis les fichiers *_data_FIXED.npz

Usage:
    python scripts/preprocessing/extract_features_from_fixed.py --family glandular
    python scripts/preprocessing/extract_features_from_fixed.py --family digestive
    # etc.
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
    parser = argparse.ArgumentParser(description="Extraire features H-optimus-0 depuis FIXED data")
    parser.add_argument("--family", required=True, choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"])
    parser.add_argument("--data_dir", default="data/cache/family_data", help="Répertoire des données FIXED")
    parser.add_argument("--output_dir", default=None, help="Répertoire de sortie (défaut: même que data_dir)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size pour extraction")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chemins fichiers
    data_file = data_dir / f"{args.family}_data_FIXED.npz"
    features_file = output_dir / f"{args.family}_features.npz"
    targets_file = output_dir / f"{args.family}_targets.npz"

    if not data_file.exists():
        print(f"❌ ERREUR: {data_file} introuvable")
        return 1

    print("=" * 80)
    print(f"EXTRACTION FEATURES H-OPTIMUS-0: {args.family}")
    print("=" * 80)
    print("")

    # Charger données FIXED
    print(f"Chargement {data_file.name}...")
    data = np.load(data_file)
    images = data['images']  # (N, 256, 256, 3) uint8
    np_targets = data['np_targets']  # (N, 256, 256) float32
    hv_targets = data['hv_targets']  # (N, 2, 256, 256) float32
    nt_targets = data['nt_targets']  # (N, 256, 256) int64
    inst_maps = data['inst_maps']  # (N, 256, 256) int32 - ✅ INSTANCES NATIVES

    n_samples = len(images)
    print(f"  → {n_samples} samples")
    print(f"  → Images: {images.shape} ({images.dtype})")
    print(f"  → HV targets: {hv_targets.shape} ({hv_targets.dtype})")
    print(f"  → Inst maps: {inst_maps.shape} ({inst_maps.dtype})")

    # Conversion uint8 si nécessaire (économie 8× espace + évite Bug #1)
    if images.dtype != np.uint8:
        print(f"\n⚠️  WARNING: Images en {images.dtype} au lieu de uint8")
        print(f"   Conversion uint8 (économie ~{images.nbytes / 1e9:.1f} GB → {images.nbytes / 8 / 1e9:.1f} GB)")
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.clip(0, 255).astype(np.uint8)
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

    for i in tqdm(range(0, n_samples, args.batch_size)):
        batch_images = images[i:i+args.batch_size]
        batch_tensors = []

        # Préprocesser batch
        for img in batch_images:
            # Images sont déjà en uint8 [0, 255]
            tensor = transform(img)  # (3, 224, 224) normalized
            batch_tensors.append(tensor)

        batch_tensor = torch.stack(batch_tensors).to(args.device)  # (B, 3, 224, 224)

        # Extraire features
        with torch.no_grad():
            features = backbone.forward_features(batch_tensor)  # (B, 261, 1536)

        all_features.append(features.cpu().numpy())

    # Concaténer
    all_features = np.concatenate(all_features, axis=0)  # (N, 261, 1536)
    print(f"  → Features extraites: {all_features.shape}")
    print("")

    # Validation CLS std
    cls_tokens = all_features[:, 0, :]  # (N, 1536)
    cls_std = cls_tokens.std()
    print(f"Validation features:")
    print(f"  CLS std: {cls_std:.3f} (attendu: 0.70-0.90)")

    if not (0.70 <= cls_std <= 0.90):
        print(f"  ⚠️  WARNING: CLS std hors range attendu!")
    else:
        print(f"  ✅ CLS std OK")
    print("")

    # Sauvegarder features
    print(f"Sauvegarde features: {features_file.name}")
    np.savez_compressed(features_file, features=all_features)
    print(f"  → {features_file.stat().st_size / 1e9:.2f} GB")
    print("")

    # Vérifier et corriger NT targets (doit être 0-4, pas 0-5)
    nt_unique = np.unique(nt_targets)
    print(f"\nVérification NT targets:")
    print(f"  Valeurs uniques: {sorted(nt_unique)}")

    if nt_targets.max() == 5:
        print(f"  ⚠️  Conversion NT: [0-5] → [0-4] (shift -1 pour non-background)")
        # Shift -1 pour tous les pixels non-background
        nt_targets_corrected = nt_targets.copy()
        mask = nt_targets > 0
        nt_targets_corrected[mask] = nt_targets[mask] - 1
        nt_targets = nt_targets_corrected
        print(f"  ✅ Après correction: {sorted(np.unique(nt_targets))}")

    # Sauvegarder targets (✅ INCLUT inst_maps maintenant - Solution B)
    print(f"\nSauvegarde targets: {targets_file.name}")
    np.savez_compressed(
        targets_file,
        np_targets=np_targets,
        hv_targets=hv_targets,
        nt_targets=nt_targets,
        inst_maps=inst_maps  # ✅ Instances natives PanNuke (0=bg, 1..N=instances)
    )
    print(f"  → {targets_file.stat().st_size / 1e9:.2f} GB")
    print(f"  → inst_maps sauvegardés ({len(np.unique(inst_maps[0])) - 1} instances moyenne dans 1er échantillon)")
    print("")

    print("=" * 80)
    print("✅ EXTRACTION TERMINÉE")
    print("=" * 80)
    print("")
    print(f"Fichiers créés:")
    print(f"  - {features_file}")
    print(f"  - {targets_file}")
    print("")
    print("Vous pouvez maintenant lancer l'entraînement:")
    print(f"  python scripts/training/train_hovernet_family.py --family {args.family} --epochs 50 --augment --dropout 0.1")
    print("")

    return 0

if __name__ == "__main__":
    sys.exit(main())
