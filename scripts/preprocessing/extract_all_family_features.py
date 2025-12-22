#!/usr/bin/env python3
"""
Extrait les features H-optimus-0 pour TOUTES les familles en une seule passe.
Charge le modèle UNE SEULE FOIS pour gagner du temps.

Usage:
    python scripts/preprocessing/extract_all_family_features.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import create_hoptimus_transform, validate_features
from src.models.loader import ModelLoader

FAMILIES = ["glandular", "digestive", "urologic", "epidermal", "respiratory"]

def extract_family_features(
    family: str,
    data_dir: Path,
    output_dir: Path,
    backbone,
    transform,
    batch_size: int,
    device: str
):
    """Extrait features pour une famille donnée."""

    # Chemins fichiers
    data_file = data_dir / f"{family}_data_FIXED.npz"
    features_file = output_dir / f"{family}_features.npz"
    targets_file = output_dir / f"{family}_targets.npz"

    if not data_file.exists():
        print(f"  ❌ SKIP: {data_file} introuvable")
        return False

    print(f"\n{'='*80}")
    print(f"📦 {family.upper()}")
    print(f"{'='*80}")

    # Charger données FIXED
    print(f"  Chargement {data_file.name}...")
    data = np.load(data_file)
    images = data['images']  # (N, 256, 256, 3) uint8
    np_targets = data['np_targets']  # (N, 256, 256) float32
    hv_targets = data['hv_targets']  # (N, 2, 256, 256) float32
    nt_targets = data['nt_targets']  # (N, 256, 256) int64

    n_samples = len(images)
    print(f"    → {n_samples} samples")
    print(f"    → Images: {images.dtype}, HV: {hv_targets.dtype}")

    # Conversion uint8 si nécessaire
    if images.dtype != np.uint8:
        print(f"    ⚠️  Conversion uint8...")
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.clip(0, 255).astype(np.uint8)

    # Extraire features par batch
    print(f"  Extraction features (batch_size={batch_size})...")
    all_features = []

    for i in tqdm(range(0, n_samples, batch_size), desc=f"  {family}"):
        batch_images = images[i:i+batch_size]
        batch_tensors = []

        # Préprocesser batch
        for img in batch_images:
            tensor = transform(img)  # (3, 224, 224) normalized
            batch_tensors.append(tensor)

        batch_tensor = torch.stack(batch_tensors).to(device)  # (B, 3, 224, 224)

        # Extraire features
        with torch.no_grad():
            features = backbone.forward_features(batch_tensor)  # (B, 261, 1536)

        all_features.append(features.cpu().numpy())

    # Concaténer
    all_features = np.concatenate(all_features, axis=0)  # (N, 261, 1536)

    # Validation CLS std
    cls_tokens = all_features[:, 0, :]  # (N, 1536)
    cls_std = cls_tokens.std()

    if not (0.70 <= cls_std <= 0.90):
        print(f"    ⚠️  CLS std: {cls_std:.3f} (hors range 0.70-0.90)")
    else:
        print(f"    ✅ CLS std: {cls_std:.3f}")

    # Sauvegarder features
    print(f"  Sauvegarde...")
    np.savez_compressed(features_file, features=all_features)

    # Sauvegarder targets
    np.savez_compressed(
        targets_file,
        np_targets=np_targets,
        hv_targets=hv_targets,
        nt_targets=nt_targets
    )

    feat_size = features_file.stat().st_size / 1e9
    targ_size = targets_file.stat().st_size / 1e9
    print(f"    → Features: {feat_size:.2f} GB")
    print(f"    → Targets:  {targ_size:.2f} GB")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extraire features pour toutes les familles")
    parser.add_argument("--data_dir", default="data/cache/family_data_FIXED", help="Répertoire des données FIXED")
    parser.add_argument("--output_dir", default=None, help="Répertoire de sortie (défaut: même que data_dir)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size pour extraction")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--families", nargs="+", default=None, help="Familles à extraire (défaut: toutes)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    families = args.families if args.families else FAMILIES

    print("=" * 80)
    print("🧬 EXTRACTION FEATURES H-OPTIMUS-0 - TOUTES FAMILLES")
    print("=" * 80)
    print(f"\nFamilles: {', '.join(families)}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {args.device}")
    print("")

    # ⏱️ CHARGER H-OPTIMUS-0 UNE SEULE FOIS
    print("Chargement H-optimus-0...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    backbone.eval()
    print("  ✅ Modèle chargé\n")

    # Transform
    transform = create_hoptimus_transform()

    # Extraire chaque famille
    success_count = 0
    for family in families:
        success = extract_family_features(
            family=family,
            data_dir=data_dir,
            output_dir=output_dir,
            backbone=backbone,
            transform=transform,
            batch_size=args.batch_size,
            device=args.device
        )
        if success:
            success_count += 1

    # Résumé
    print(f"\n{'='*80}")
    print(f"✅ EXTRACTION TERMINÉE: {success_count}/{len(families)} familles")
    print(f"{'='*80}\n")

    print("Fichiers créés dans:", output_dir)
    for family in families:
        features_file = output_dir / f"{family}_features.npz"
        targets_file = output_dir / f"{family}_targets.npz"
        if features_file.exists() and targets_file.exists():
            print(f"  ✅ {family}: {family}_features.npz + {family}_targets.npz")

    print("\n🚀 Vous pouvez maintenant lancer l'entraînement:")
    print("  python scripts/training/train_hovernet_family.py --family glandular --epochs 50 --augment --dropout 0.1 --cache_dir data/cache/family_data_FIXED")
    print("")

    return 0 if success_count == len(families) else 1


if __name__ == "__main__":
    sys.exit(main())
