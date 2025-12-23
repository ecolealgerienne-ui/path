#!/usr/bin/env python3
"""
Debug prédictions HoVer-Net pour comprendre pourquoi AJI est si faible.

Visualise:
- Image originale
- NP prediction (avant/après seuillage)
- HV gradients
- Instances watershed
- Ground truth

Usage:
    python scripts/evaluation/debug_predictions.py \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
        --family epidermal \
        --n_samples 3
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE
from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform


def load_one_sample(data_dir: Path, family: str, fold: int = 2):
    """Charge UN échantillon pour debug."""
    from src.models.organ_families import ORGAN_TO_FAMILY

    family_organs = [organ for organ, fam in ORGAN_TO_FAMILY.items() if fam == family]

    fold_dir = data_dir / f"fold{fold}"
    images = np.load(fold_dir / "images.npy", mmap_mode='r')
    masks = np.load(fold_dir / "masks.npy", mmap_mode='r')
    types = np.load(fold_dir / "types.npy")

    for i, organ in enumerate(types):
        if organ in family_organs:
            return {
                'image': images[i].copy(),
                'mask': masks[i].copy(),
                'organ': organ,
                'index': i
            }

    raise ValueError(f"Aucun échantillon {family} trouvé dans fold {fold}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--family', type=str, default='epidermal')
    parser.add_argument('--data_dir', type=Path, default=Path('/home/amar/data/PanNuke'))
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=Path, default=Path('results/debug'))

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"DEBUG PRÉDICTIONS - {args.family.upper()}")
    print(f"{'='*70}\n")

    # Load model
    print("📥 Chargement modèle...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    backbone = ModelLoader.load_hoptimus0(device=args.device)

    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet = hovernet.to(args.device)
    hovernet.eval()
    print(f"   ✅ Epoch {checkpoint.get('epoch', '?')}\n")

    # Load sample
    print("📥 Chargement échantillon...")
    sample = load_one_sample(args.data_dir, args.family, args.fold)
    image = sample['image']  # (256, 256, 3) uint8
    mask = sample['mask']    # (256, 256, 6) int32
    print(f"   ✅ {sample['organ']} (index {sample['index']})\n")

    # Ground truth
    np_gt = mask[:, :, 1:].sum(axis=-1) > 0
    n_instances_gt = len(np.unique(mask[:, :, 1:].flatten())) - 1
    print(f"Ground Truth:")
    print(f"  Noyaux: {np_gt.sum()} pixels")
    print(f"  Instances: {n_instances_gt}")

    # Inference
    print(f"\n🔍 Inférence...")
    transform = create_hoptimus_transform()

    with torch.no_grad():
        tensor = transform(image).unsqueeze(0).to(args.device)
        features = backbone.forward_features(tensor)
        np_out, hv_out, nt_out = hovernet(features)

    # Raw predictions (224×224)
    np_pred_224 = torch.sigmoid(np_out).cpu().numpy()[0, 0]
    hv_pred_224 = hv_out.cpu().numpy()[0]

    print(f"\nPrédictions BRUTES (224×224):")
    print(f"  NP min/max: [{np_pred_224.min():.3f}, {np_pred_224.max():.3f}]")
    print(f"  NP >0.5: {(np_pred_224 > 0.5).sum()} pixels")
    print(f"  NP >0.3: {(np_pred_224 > 0.3).sum()} pixels")
    print(f"  NP >0.1: {(np_pred_224 > 0.1).sum()} pixels")
    print(f"  HV range: [{hv_pred_224.min():.3f}, {hv_pred_224.max():.3f}]")

    # Resize to 256×256
    from src.utils.image_utils import prepare_predictions_for_evaluation
    np_pred, hv_pred, _ = prepare_predictions_for_evaluation(
        np_pred_224, hv_pred_224, np.zeros((5, 224, 224)),
        target_size=PANNUKE_IMAGE_SIZE
    )

    print(f"\nAprès resize (256×256):")
    print(f"  NP min/max: [{np_pred.min():.3f}, {np_pred.max():.3f}]")
    print(f"  NP >0.5: {(np_pred > 0.5).sum()} pixels")
    print(f"  NP >0.3: {(np_pred > 0.3).sum()} pixels")

    # Compute HV gradients
    from scripts.evaluation.quick_aji_test import get_gradient_hv
    gradient_mag = get_gradient_hv(hv_pred)
    print(f"  HV gradient range: [{gradient_mag.min():.3f}, {gradient_mag.max():.3f}]")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Image, GT, NP pred
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Image Originale\n{sample['organ']}")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np_gt, cmap='gray')
    axes[0, 1].set_title(f"GT NP\n{np_gt.sum()} pixels")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np_pred, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f"NP Pred (raw)\nmax={np_pred.max():.3f}")
    axes[0, 2].axis('off')

    # Row 2: NP thresholded, HV gradients, Comparison
    np_binary = np_pred > 0.5
    axes[1, 0].imshow(np_binary, cmap='gray')
    axes[1, 0].set_title(f"NP Pred >0.5\n{np_binary.sum()} pixels")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gradient_mag, cmap='hot')
    axes[1, 1].set_title("HV Gradient Magnitude")
    axes[1, 1].axis('off')

    # Overlay
    overlay = image.copy()
    overlay[np_binary] = [255, 0, 0]  # Rouge pour prédictions
    overlay[np_gt & ~np_binary] = [0, 255, 0]  # Vert pour GT manqués
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Overlay (Rouge=Pred, Vert=GT manqué)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = args.output_dir / f"debug_{sample['organ']}_{sample['index']}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualisation sauvegardée: {output_path}")

    # Diagnostic
    print(f"\n{'='*70}")
    print("DIAGNOSTIC")
    print(f"{'='*70}\n")

    coverage = (np_pred > 0.5).sum() / np_gt.sum() if np_gt.sum() > 0 else 0

    if coverage < 0.1:
        print("❌ PROBLÈME CRITIQUE: Modèle détecte <10% des noyaux GT")
        print("   Causes possibles:")
        print("   1. Checkpoint incorrect (mauvais epoch?)")
        print("   2. Preprocessing différent train vs test")
        print("   3. Seuil 0.5 trop élevé (tester 0.3 ou 0.1)")
        print(f"   → NP max = {np_pred.max():.3f} (devrait être proche de 1.0)")
    elif np_pred.max() < 0.8:
        print("⚠️  PROBLÈME: NP predictions trop faibles (max < 0.8)")
        print("   → Modèle pas assez confiant")
        print("   → Vérifier calibration / temperature scaling")
    elif coverage < 0.5:
        print("⚠️  PROBLÈME: Couverture partielle (<50%)")
        print("   → Modèle détecte certaines cellules mais pas toutes")
        print("   → Watershed peut être trop agressif")
    else:
        print("✅ NP predictions semblent correctes")
        print("   → Problème probablement dans watershed post-processing")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
