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

# Import IHM engine pour comparaison directe
from src.ui.inference_engine import CellVitEngine


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
    # 2. CHARGER LE MOTEUR IHM (CellVitEngine)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("2. LOADING IHM ENGINE (CellVitEngine)")
    print("=" * 80)

    # Déterminer l'organe par défaut pour la famille
    family_organs = {
        "respiratory": "Lung",
        "urologic": "Kidney",
        "glandular": "Breast",
        "epidermal": "Skin",
        "digestive": "Colon"
    }
    default_organ = family_organs[family]

    print(f"  Loading CellVitEngine (organ={default_organ})...")
    engine = CellVitEngine(device=device_str, organ=default_organ)
    print(f"  ✅ Engine loaded")
    print(f"  Family: {engine.family}")
    print(f"  Watershed params: {engine.watershed_params}")

    # ==========================================================================
    # 3. COMPARER POUR CHAQUE IMAGE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("3. COMPARING PIPELINES")
    print("=" * 80)

    results = []

    for i, image in enumerate(images):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{len(images)}")
        print(f"{'='*60}")
        print(f"  Image: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")

        # ======================================================================
        # PIPELINE IHM (via CellVitEngine.analyze)
        # ======================================================================
        print("\n  [IHM] CellVitEngine.analyze():")

        result_ihm = engine.analyze(image, compute_morphometry=False, compute_uncertainty=False)
        n_ihm = result_ihm.n_nuclei

        print(f"      → Nuclei detected: {n_ihm}")

        # ======================================================================
        # VÉRIFICATION: Les résultats sont-ils identiques?
        # ======================================================================
        print(f"\n  ════════════════════════════════════════")
        print(f"  IHM (CellVitEngine): {n_ihm} noyaux")
        print(f"  ════════════════════════════════════════")

        results.append({
            'sample': i,
            'n_ihm': n_ihm,
        })

    # ==========================================================================
    # RÉSUMÉ
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n  Sample | IHM (CellVitEngine)")
    print("  " + "-" * 30)
    for r in results:
        print(f"  {r['sample']:6} | {r['n_ihm']:4}")

    print("""
\nCe script utilise EXACTEMENT le même moteur que l'IHM Gradio.
Si les résultats diffèrent de ce que vous voyez dans l'IHM web:

1. Vérifiez que l'IHM a été redémarrée après les modifications de code
2. Vérifiez que l'image uploadée est exactement 224×224 pixels
3. Vérifiez que l'image est au format RGB (pas BGR)
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
