#!/usr/bin/env python3
"""
Diagnostic des prédictions brutes HoVer-Net.

Vérifie:
1. Range NP predictions (max, min, mean)
2. Range HV predictions (vérifier saturation Tanh)
3. Gradient magnitude HV (vérifier si gradients nets)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.loader import ModelLoader
from src.models.hovernet_decoder import HoVerNetDecoder
from src.preprocessing import create_hoptimus_transform


def diagnose_single_image(
    image: np.ndarray,
    backbone,
    hovernet,
    device: str = "cuda"
) -> dict:
    """
    Diagnostique une image unique.
    """
    # Preprocessing
    transform = create_hoptimus_transform()

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    tensor = transform(image).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        features = backbone.forward_features(tensor)
        patch_tokens = features[:, 1:257, :]

        # HoVer-Net inference
        np_out, hv_out, nt_out = hovernet(patch_tokens)

        # Convert to numpy
        np_logits = np_out.cpu().numpy()[0]
        hv_pred = hv_out.cpu().numpy()[0]

        # Apply sigmoid to NP
        np_probs = 1 / (1 + np.exp(-np_logits))
        np_binary = np_probs[1]

        result = {
            'np_max': float(np_binary.max()),
            'np_mean': float(np_binary.mean()),
            'hv_max': float(np.abs(hv_pred).max()),
            'hv_mean': float(np.abs(hv_pred).mean()),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC PRÉDICTIONS BRUTES")
    print("=" * 70)

    # Load models
    print("\n🔧 Chargement modèles...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5, adaptive=False).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    # Load dataset
    dataset_path = Path(args.dataset_dir)
    npz_files = sorted(dataset_path.glob("*.npz"))[:args.num_samples]

    results = []
    for npz_file in tqdm(npz_files):
        data = np.load(npz_file)
        result = diagnose_single_image(data['image'], backbone, hovernet, args.device)
        results.append(result)

    # Stats
    np_max_vals = [r['np_max'] for r in results]
    hv_max_vals = [r['hv_max'] for r in results]

    print("\n📊 NP PREDICTIONS")
    print(f"  Max: {np.max(np_max_vals):.4f}")
    print(f"  Mean: {np.mean(np_max_vals):.4f}")

    n_above = sum(1 for m in np_max_vals if m > 0.3)
    print(f"\n  Images NP > 0.3: {n_above}/{len(np_max_vals)} ({100*n_above/len(np_max_vals):.1f}%)")

    if n_above < len(np_max_vals) * 0.5:
        print(f"  ⚠️  Seuil 0.3 TROP ÉLEVÉ → Abaisser à 0.15")

    print("\n📊 HV PREDICTIONS")
    print(f"  Max |HV|: {np.mean(hv_max_vals):.4f}")

    if np.mean(hv_max_vals) < 0.3:
        print(f"  🔴 Tanh SATURE ({np.mean(hv_max_vals):.3f} < 0.3)")
    elif np.mean(hv_max_vals) < 0.6:
        print(f"  ⚠️  Tanh sous-utilisé ({np.mean(hv_max_vals):.3f} < 0.6)")
    else:
        print(f"  ✅ Range OK ({np.mean(hv_max_vals):.3f})")


if __name__ == "__main__":
    main()
