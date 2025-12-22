#!/usr/bin/env python3
"""
Extraction des features H-optimus-0 pour un fold PanNuke.

Sauvegarde les features en float16 pour économiser l'espace disque.
Après extraction, les images originales peuvent être supprimées.

Usage:
    python scripts/preprocessing/extract_fold_features.py \
        --data_dir /home/amar/data/PanNuke \
        --fold 0 \
        --output_dir data/features

Structure de sortie:
    data/features/
    ├── fold0/
    │   ├── cls_tokens.npy      # (N, 1536) float16 - pour OrganHead
    │   ├── patch_tokens.npy    # (N, 256, 1536) float16 - pour HoVerNet
    │   ├── types.npy           # (N,) - labels organes
    │   ├── masks.npy           # (N, 256, 256, 6) - masks segmentation
    │   └── metadata.json       # infos extraction
    └── fold1/
        └── ...
"""

import numpy as np
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports des modules centralisés (Phase 1 Refactoring)
from src.preprocessing import preprocess_image
from src.models.loader import ModelLoader


def preprocess_batch(images: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Prétraite un batch d'images pour H-optimus-0.

    Utilise le preprocessing centralisé pour cohérence.

    Args:
        images: (B, 256, 256, 3) uint8 ou float
        device: Device pour les tensors

    Returns:
        tensor: (B, 3, 224, 224) normalized
    """
    # Utiliser la fonction centralisée pour chaque image (Phase 1 Refactoring)
    batch_tensors = [preprocess_image(img, device=device).squeeze(0) for img in images]
    return torch.stack(batch_tensors)


@torch.no_grad()
def extract_features(
    model,
    images: np.ndarray,
    batch_size: int = 16,
    device: str = "cuda",
) -> tuple:
    """
    Extrait les features H-optimus-0.

    Args:
        model: H-optimus-0 backbone
        images: (N, 256, 256, 3)
        batch_size: Taille des batches
        device: cuda ou cpu

    Returns:
        cls_tokens: (N, 1536) float16
        patch_tokens: (N, 256, 1536) float16
    """
    n_images = len(images)
    cls_tokens = []
    patch_tokens = []

    for i in tqdm(range(0, n_images, batch_size), desc="Extraction"):
        batch = images[i:i+batch_size]
        # Utiliser preprocessing centralisé (Phase 1 Refactoring)
        x = preprocess_batch(batch, device=device)

        # Forward via forward_features() (inclut LayerNorm final)
        features = model.forward_features(x)  # (B, 261, 1536)

        # Séparer CLS et patches
        cls = features[:, 0, :].cpu().numpy()  # (B, 1536)
        patches = features[:, 1:257, :].cpu().numpy()  # (B, 256, 1536)

        cls_tokens.append(cls)
        patch_tokens.append(patches)

    # Concatenate et convertir en float16
    cls_tokens = np.concatenate(cls_tokens, axis=0).astype(np.float16)
    patch_tokens = np.concatenate(patch_tokens, axis=0).astype(np.float16)

    return cls_tokens, patch_tokens


def main():
    parser = argparse.ArgumentParser(description="Extraire les features d'un fold PanNuke")
    parser.add_argument("--data_dir", type=str, default="/home/amar/data/PanNuke",
                        help="Chemin vers PanNuke")
    parser.add_argument("--fold", type=int, default=0,
                        help="Numéro du fold (0, 1, 2)")
    parser.add_argument("--output_dir", type=str, default="data/features",
                        help="Dossier de sortie")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Taille des batches")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--skip_patches", action="store_true",
                        help="Ne pas sauvegarder les patch tokens (économie espace)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / f"fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chercher le fold
    fold_names = [f"fold{args.fold}", f"fold_{args.fold}"]
    fold_dir = None
    for name in fold_names:
        if (data_dir / name).exists():
            fold_dir = data_dir / name
            break

    if fold_dir is None:
        print(f"❌ Fold {args.fold} non trouvé dans {data_dir}")
        print(f"   Chemins testés: {fold_names}")
        return

    print(f"📂 Chargement des données depuis {fold_dir}...")

    # Charger les données
    images = np.load(fold_dir / "images.npy")
    types = np.load(fold_dir / "types.npy")
    masks = np.load(fold_dir / "masks.npy")

    print(f"   Images: {images.shape}")
    print(f"   Types: {types.shape}")
    print(f"   Masks: {masks.shape}")

    # Charger le backbone (Phase 1 Refactoring - chargeur centralisé)
    print("⏳ Chargement H-optimus-0 (1.1B params)...")
    model = ModelLoader.load_hoptimus0(device=args.device)
    print("✅ H-optimus-0 chargé")

    # Extraire les features
    print(f"\n🔄 Extraction des features (batch_size={args.batch_size})...")
    cls_tokens, patch_tokens = extract_features(
        model, images,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"   CLS tokens: {cls_tokens.shape} ({cls_tokens.nbytes / 1024 / 1024:.1f} MB)")
    print(f"   Patch tokens: {patch_tokens.shape} ({patch_tokens.nbytes / 1024 / 1024:.1f} MB)")

    # Sauvegarder
    print(f"\n💾 Sauvegarde dans {output_dir}...")

    np.save(output_dir / "cls_tokens.npy", cls_tokens)
    print(f"   ✓ cls_tokens.npy")

    if not args.skip_patches:
        np.save(output_dir / "patch_tokens.npy", patch_tokens)
        print(f"   ✓ patch_tokens.npy")

    np.save(output_dir / "types.npy", types)
    print(f"   ✓ types.npy")

    np.save(output_dir / "masks.npy", masks)
    print(f"   ✓ masks.npy")

    # Metadata
    metadata = {
        "fold": args.fold,
        "n_images": len(images),
        "extraction_date": datetime.now().isoformat(),
        "backbone": "H-optimus-0",
        "dtype": "float16",
        "cls_shape": list(cls_tokens.shape),
        "patch_shape": list(patch_tokens.shape) if not args.skip_patches else None,
        "organs": list(np.unique(types)),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ metadata.json")

    # Résumé espace
    total_size = 0
    for f in output_dir.glob("*.npy"):
        total_size += f.stat().st_size

    original_size = images.nbytes + masks.nbytes

    print(f"\n📊 Résumé:")
    print(f"   Images originales: {original_size / 1024 / 1024:.1f} MB")
    print(f"   Features extraites: {total_size / 1024 / 1024:.1f} MB")
    print(f"   Ratio: {total_size / original_size:.2%}")

    print(f"\n✅ Extraction terminée!")
    print(f"\n🗑️  Vous pouvez maintenant supprimer les images originales:")
    print(f"   rm -rf {fold_dir}/images.npy")
    print(f"\n📚 Pour entraîner avec les features:")
    print(f"   python scripts/training/train_organ_head.py --features_dir {output_dir}")


if __name__ == "__main__":
    main()
