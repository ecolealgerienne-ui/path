#!/usr/bin/env python3
"""
Extrait les features H-optimus-0 depuis les données v12 (224×224).

⚠️ SPÉCIFIQUE À V12:
   Les données v12 sont DÉJÀ à 224×224 (résolution native H-optimus-0).
   Les HV targets ont été calculés APRÈS resize (pas d'interpolation floue).

Ce script génère les fichiers attendus par train_hovernet_family.py:
- {family}_features.npz (features H-optimus-0)
- {family}_targets.npz (NP/HV/NT targets)

Pipeline complet:
    1. prepare_family_data_FIXED_v12_COHERENT.py → *_data_FIXED_v12_COHERENT.npz
    2. extract_features_from_v12.py              → *_features.npz + *_targets.npz
    3. train_hovernet_family.py                  → hovernet_*_best.pth

Usage:
    python scripts/preprocessing/extract_features_from_v12.py --family epidermal
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing import create_hoptimus_transform, validate_features
from src.models.loader import ModelLoader
from src.constants import (
    HOPTIMUS_INPUT_SIZE,
    HV_MAP_MIN,
    HV_MAP_MAX,
    FAMILIES,
    DEFAULT_FAMILY_DATA_DIR,
    DEFAULT_FAMILY_FIXED_DIR,
)

# Version spécifique de ce script
V12_VERSION = "v12_COHERENT"


def get_v12_input_path(family: str) -> Path:
    """Retourne le chemin du fichier v12 pour une famille."""
    return Path(DEFAULT_FAMILY_FIXED_DIR) / f"{family}_data_FIXED_{V12_VERSION}.npz"


def validate_v12_data(images: np.ndarray, hv_targets: np.ndarray) -> dict:
    """
    Valide que les données sont bien au format v12 (224×224, HV float32).

    Returns:
        dict avec 'valid' (bool) et 'messages' (list)
    """
    messages = []
    valid = True

    # Vérifier taille des images
    if images.shape[1:3] != (HOPTIMUS_INPUT_SIZE, HOPTIMUS_INPUT_SIZE):
        messages.append(
            f"❌ ERREUR: Images sont {images.shape[1]}×{images.shape[2]} "
            f"au lieu de {HOPTIMUS_INPUT_SIZE}×{HOPTIMUS_INPUT_SIZE}"
        )
        valid = False
    else:
        messages.append(f"✅ Images: {HOPTIMUS_INPUT_SIZE}×{HOPTIMUS_INPUT_SIZE}")

    # Vérifier dtype HV
    if hv_targets.dtype != np.float32:
        messages.append(f"❌ ERREUR: HV dtype est {hv_targets.dtype} au lieu de float32")
        valid = False
    else:
        messages.append(f"✅ HV dtype: float32")

    # Vérifier range HV
    hv_min, hv_max = hv_targets.min(), hv_targets.max()
    if hv_min < HV_MAP_MIN - 0.1 or hv_max > HV_MAP_MAX + 0.1:
        messages.append(
            f"❌ ERREUR: HV range [{hv_min:.3f}, {hv_max:.3f}] "
            f"hors limites [{HV_MAP_MIN}, {HV_MAP_MAX}]"
        )
        valid = False
    else:
        messages.append(f"✅ HV range: [{hv_min:.3f}, {hv_max:.3f}]")

    return {'valid': valid, 'messages': messages}


def main():
    parser = argparse.ArgumentParser(
        description="Extrait features H-optimus-0 depuis données v12 (224×224)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
📌 Version: {V12_VERSION}

⚠️ IMPORTANT:
   Ce script est spécifique aux données v12 qui sont à 224×224.
   Les HV targets ont été calculés APRÈS resize pour éviter l'interpolation floue.

Exemples:
    python scripts/preprocessing/extract_features_from_v12.py --family epidermal
    python scripts/preprocessing/extract_features_from_v12.py --family glandular --batch_size 16
        """
    )
    parser.add_argument(
        '--family',
        type=str,
        required=True,
        choices=FAMILIES,
        help='Famille d\'organes'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_FAMILY_DATA_DIR,
        help=f'Répertoire de sortie (défaut: {DEFAULT_FAMILY_DATA_DIR})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Taille de batch pour extraction'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda ou cpu)'
    )

    args = parser.parse_args()

    # Déterminer le fichier input v12
    input_file = get_v12_input_path(args.family)

    if not input_file.exists():
        print(f"❌ Fichier v12 non trouvé: {input_file}")
        print(f"\n💡 Exécutez d'abord:")
        print(f"   python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py --family {args.family}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_file = output_dir / f"{args.family}_features.npz"
    targets_file = output_dir / f"{args.family}_targets.npz"

    print("=" * 80)
    print(f"EXTRACTION FEATURES H-optimus-0 - v12 - Famille: {args.family.upper()}")
    print("=" * 80)

    # Load v12 data
    print(f"\n📦 Chargement données v12: {input_file}")
    data = np.load(input_file)

    images = data['images']
    np_targets = data['np_targets']
    hv_targets = data['hv_targets']
    nt_targets = data['nt_targets']

    n_samples = len(images)
    print(f"  Samples: {n_samples}")
    print(f"  Images shape: {images.shape}")
    print(f"  NP targets shape: {np_targets.shape}")
    print(f"  HV targets shape: {hv_targets.shape}")
    print(f"  NT targets shape: {nt_targets.shape}")

    # ⚠️ VALIDATION SPÉCIFIQUE v12
    print(f"\n🔍 Validation format v12...")
    validation = validate_v12_data(images, hv_targets)

    for msg in validation['messages']:
        print(f"  {msg}")

    if not validation['valid']:
        print(f"\n❌ ÉCHEC: Données non conformes au format v12")
        print(f"   Régénérez avec prepare_family_data_FIXED_v12_COHERENT.py")
        return

    # Load H-optimus-0
    print(f"\n🔧 Chargement H-optimus-0...")
    device_str = args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    backbone = ModelLoader.load_hoptimus0(device=device_str)
    backbone.eval()
    print(f"  Device: {device}")

    # Create transform
    # Note: Le Resize(224, 224) est un no-op car les images sont déjà 224×224
    transform = create_hoptimus_transform()

    # Extract features
    print(f"\n🚀 Extraction features (batch_size={args.batch_size})...")
    all_features = []

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, args.batch_size), desc="Extraction"):
            batch_end = min(i + args.batch_size, n_samples)
            batch_images = images[i:batch_end]

            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                # Convert to uint8 if needed
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                # Apply transform (Resize est no-op car déjà 224×224)
                tensor = transform(img)
                batch_tensors.append(tensor)

            # Stack batch
            batch = torch.stack(batch_tensors).to(device)

            # Extract features
            features = backbone.forward_features(batch)  # (B, 261, 1536)

            # Store
            all_features.append(features.cpu().numpy())

    # Concatenate
    print(f"\n💾 Concatenation features...")
    features_array = np.concatenate(all_features, axis=0)

    # Validate features
    print(f"\n✅ Validation features H-optimus-0...")
    feat_validation = validate_features(torch.from_numpy(features_array))

    if feat_validation['valid']:
        print(f"  CLS std: {feat_validation['cls_std']:.4f} (attendu: 0.70-0.90)")
        print(f"  ✅ Features valides!")
    else:
        print(f"  ⚠️ WARNING: {feat_validation['message']}")
        print(f"  CLS std: {feat_validation['cls_std']:.4f}")

    # Save features
    print(f"\n💾 Sauvegarde features: {features_file}")
    np.savez_compressed(
        features_file,
        features=features_array
    )

    # Save targets
    print(f"💾 Sauvegarde targets: {targets_file}")
    np.savez_compressed(
        targets_file,
        np_targets=np_targets,
        hv_targets=hv_targets,
        nt_targets=nt_targets
    )

    # Stats
    print(f"\n📊 Statistiques:")
    print(f"  Features shape: {features_array.shape}")
    print(f"  Features size: {features_file.stat().st_size / 1e6:.1f} MB")
    print(f"  Targets size: {targets_file.stat().st_size / 1e6:.1f} MB")

    # HV stats pour vérification
    print(f"\n📊 Stats HV targets (pour vérification):")
    print(f"  Min: {hv_targets.min():.4f}")
    print(f"  Max: {hv_targets.max():.4f}")
    print(f"  Mean: {hv_targets.mean():.4f}")
    print(f"  Std: {hv_targets.std():.4f}")

    print("\n" + "=" * 80)
    print("✅ EXTRACTION v12 TERMINÉE")
    print("=" * 80)

    print(f"\n🚀 Prochaine étape - Training:")
    print(f"   python scripts/training/train_hovernet_family.py \\")
    print(f"       --family {args.family} \\")
    print(f"       --cache_dir {output_dir} \\")
    print(f"       --epochs 50 \\")
    print(f"       --augment \\")
    print(f"       --lambda_hv 2.0")


if __name__ == '__main__':
    main()
