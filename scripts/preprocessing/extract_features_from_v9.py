#!/usr/bin/env python3
"""
Extrait les features H-optimus-0 depuis les données famille.

⚠️ UTILISE AUTOMATIQUEMENT LA VERSION COURANTE définie dans src/constants.py
   (CURRENT_DATA_VERSION = "v12_COHERENT")

Génère les fichiers attendus par train_hovernet_family.py:
- {family}_features.npz (features H-optimus-0)
- {family}_targets.npz (NP/HV/NT targets)

Usage:
    python scripts/preprocessing/extract_features_from_v9.py --family epidermal
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
    CURRENT_DATA_VERSION,
    get_family_data_path,
    get_family_features_path,
    get_family_targets_path,
    DEFAULT_FAMILY_DATA_DIR,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extrait features H-optimus-0 depuis données famille",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
📌 Version courante: {CURRENT_DATA_VERSION}
   (Définie dans src/constants.py)

Exemples:
    # Utilise automatiquement la version courante (v12_COHERENT)
    python scripts/preprocessing/extract_features_from_v9.py --family epidermal

    # Spécifier un fichier différent
    python scripts/preprocessing/extract_features_from_v9.py --family epidermal \\
        --input_file data/family_FIXED/epidermal_data_FIXED_v11.npz
        """
    )
    parser.add_argument(
        '--family',
        type=str,
        required=True,
        choices=['epidermal', 'glandular', 'digestive', 'urologic', 'respiratory'],
        help='Famille d\'organes'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help=f'Fichier input (défaut: utilise CURRENT_DATA_VERSION={CURRENT_DATA_VERSION})'
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

    # ⚠️ UTILISE LA VERSION COURANTE PAR DÉFAUT (centralisée dans constants.py)
    if args.input_file is None:
        input_file = Path(get_family_data_path(args.family))
        print(f"📌 Utilisation de la version courante: {CURRENT_DATA_VERSION}")
    else:
        input_file = Path(args.input_file)
        print(f"📌 Fichier spécifié manuellement: {input_file}")

    if not input_file.exists():
        print(f"❌ Fichier non trouvé: {input_file}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_file = output_dir / f"{args.family}_features.npz"
    targets_file = output_dir / f"{args.family}_targets.npz"

    print("=" * 80)
    print(f"EXTRACTION FEATURES H-optimus-0 - Famille: {args.family.upper()}")
    print("=" * 80)

    # Load v9 data
    print(f"\n📦 Chargement données v9: {input_file}")
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

    # Load H-optimus-0
    print(f"\n🔧 Chargement H-optimus-0...")
    device_str = args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    backbone = ModelLoader.load_hoptimus0(device=device_str)  # Pass string, not torch.device
    backbone.eval()
    print(f"  Device: {device}")

    # Create transform
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

                # Apply transform
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

    # Validate
    print(f"\n✅ Validation features...")
    validation = validate_features(torch.from_numpy(features_array))

    if validation['valid']:
        print(f"  CLS std: {validation['cls_std']:.4f} (attendu: 0.70-0.90)")
        print(f"  ✅ Features valides!")
    else:
        print(f"  ❌ WARNING: {validation['message']}")
        print(f"  CLS std: {validation['cls_std']:.4f}")

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

    print("\n✅ TERMINÉ")
    print(f"\nProchaine étape:")
    print(f"  python scripts/training/train_hovernet_family.py \\")
    print(f"      --family {args.family} \\")
    print(f"      --cache_dir {output_dir} \\")
    print(f"      --epochs 50 \\")
    print(f"      --augment \\")
    print(f"      --lambda_hv 2.0")


if __name__ == '__main__':
    main()
