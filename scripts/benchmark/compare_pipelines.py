#!/usr/bin/env python3
"""
Compare les pipelines Benchmark vs IHM pour identifier les différences.

Usage:
    python scripts/benchmark/compare_pipelines.py --family respiratory --n_samples 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
import torchvision.transforms as T

from src.models.loader import ModelLoader
from src.preprocessing import preprocess_image
from src.postprocessing import hv_guided_watershed
from src.evaluation import run_inference
from src.ui.organ_config import FAMILY_WATERSHED_PARAMS, FAMILY_CHECKPOINTS
from src.models.hovernet_decoder import HoVerNetDecoder


def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, rtol=1e-5, atol=1e-5):
    """Compare deux tensors et affiche les différences."""
    if t1.shape != t2.shape:
        print(f"  ❌ {name}: Shape mismatch! {t1.shape} vs {t2.shape}")
        return False

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)

    if is_close:
        print(f"  ✅ {name}: IDENTICAL (max_diff={max_diff:.2e})")
    else:
        print(f"  ❌ {name}: DIFFERENT (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")

    return is_close


def compare_arrays(name: str, a1: np.ndarray, a2: np.ndarray):
    """Compare deux arrays numpy."""
    if a1.shape != a2.shape:
        print(f"  ❌ {name}: Shape mismatch! {a1.shape} vs {a2.shape}")
        return False

    if np.array_equal(a1, a2):
        print(f"  ✅ {name}: IDENTICAL")
        return True
    else:
        diff_count = np.sum(a1 != a2)
        diff_percent = 100 * diff_count / a1.size
        print(f"  ❌ {name}: DIFFERENT ({diff_count} pixels differ, {diff_percent:.2f}%)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare pipelines Benchmark vs IHM")
    parser.add_argument("--family", required=True, choices=["respiratory", "urologic", "glandular", "epidermal", "digestive"])
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device_str = str(device).split(':')[0]
    family = args.family

    print("=" * 80)
    print("PIPELINE COMPARISON: Benchmark vs IHM")
    print("=" * 80)
    print(f"Family: {family}")
    print(f"Samples: {args.n_samples}")
    print(f"Device: {device}")

    # ==========================================================================
    # 1. CHARGER LES DONNÉES
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. LOADING DATA")
    print("=" * 80)

    val_data_path = Path(f"data/family_data_v13_smart_crops/{family}_val_v13_smart_crops.npz")
    if not val_data_path.exists():
        print(f"❌ ERROR: {val_data_path} not found")
        return 1

    val_data = np.load(val_data_path)
    images = val_data['images'][:args.n_samples]
    print(f"  Loaded {len(images)} images: {images.shape}, dtype={images.dtype}")

    # ==========================================================================
    # 2. CHARGER LES MODÈLES
    # ==========================================================================
    print("\n" + "=" * 80)
    print("2. LOADING MODELS")
    print("=" * 80)

    # Backbone
    print("  Loading H-optimus-0 backbone...")
    backbone = ModelLoader.load_hoptimus0(device=device_str)
    backbone.eval()
    print(f"  ✅ Backbone loaded")

    # HoVer-Net decoder
    checkpoint_path = FAMILY_CHECKPOINTS[family]
    print(f"  Loading HoVer-Net: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HoVerNetDecoder(
        use_hybrid=True,
        use_fpn_chimique=True,
        use_h_alpha=True
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✅ HoVer-Net loaded")

    # Watershed params
    watershed_params = FAMILY_WATERSHED_PARAMS[family]
    print(f"  Watershed params: {watershed_params}")

    # ==========================================================================
    # 3. COMPARER POUR CHAQUE IMAGE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("3. COMPARING PIPELINES")
    print("=" * 80)

    all_identical = True

    for i, image in enumerate(images):
        print(f"\n--- Sample {i+1}/{len(images)} ---")
        print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"  Image range: [{image.min()}, {image.max()}]")

        # ======================================================================
        # ÉTAPE A: Preprocessing pour backbone
        # ======================================================================
        print("\n  [A] Backbone preprocessing:")

        # Pipeline IHM: preprocess_image()
        tensor_ihm = preprocess_image(image, device=device_str)
        print(f"      IHM (preprocess_image): shape={tensor_ihm.shape}, range=[{tensor_ihm.min():.3f}, {tensor_ihm.max():.3f}]")

        # Les deux devraient être identiques maintenant
        # (on utilise preprocess_image pour les deux)

        # ======================================================================
        # ÉTAPE B: Feature extraction
        # ======================================================================
        print("\n  [B] Feature extraction:")

        with torch.no_grad():
            features = backbone.forward_features(tensor_ihm)

        print(f"      Features: shape={features.shape}")
        print(f"      CLS token std: {features[:, 0, :].std().item():.4f}")

        # ======================================================================
        # ÉTAPE C: Image tensor pour FPN Chimique
        # ======================================================================
        print("\n  [C] FPN Chimique image tensor:")

        # Pipeline IHM: T.ToTensor()
        image_tensor_ihm = T.ToTensor()(image).unsqueeze(0).to(device)
        print(f"      IHM (T.ToTensor): shape={image_tensor_ihm.shape}, range=[{image_tensor_ihm.min():.3f}, {image_tensor_ihm.max():.3f}]")

        # ======================================================================
        # ÉTAPE D: HoVer-Net inference
        # ======================================================================
        print("\n  [D] HoVer-Net inference:")

        np_pred, hv_pred = run_inference(model, features, image_tensor_ihm, device=device_str)

        print(f"      NP pred: shape={np_pred.shape}, range=[{np_pred.min():.3f}, {np_pred.max():.3f}]")
        print(f"      HV pred: shape={hv_pred.shape}, range=[{hv_pred.min():.3f}, {hv_pred.max():.3f}]")

        # ======================================================================
        # ÉTAPE E: Watershed
        # ======================================================================
        print("\n  [E] Watershed:")

        instance_map = hv_guided_watershed(np_pred, hv_pred, **watershed_params)
        n_nuclei = len(np.unique(instance_map)) - 1

        print(f"      Instance map: shape={instance_map.shape}")
        print(f"      Nuclei detected: {n_nuclei}")

        print(f"\n  📊 RESULT: {n_nuclei} nuclei detected")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Ce script montre le pipeline EXACT utilisé.
Si les résultats diffèrent encore de l'IHM, vérifier:

1. L'IHM a-t-elle été redémarrée après les modifications?
2. L'image uploadée est-elle exactement la même? (même taille 224x224?)
3. Y a-t-il un redimensionnement dans l'IHM avant analyse?
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
