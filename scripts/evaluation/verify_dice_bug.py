#!/usr/bin/env python3
"""
Vérifie que le bug argmax vs sigmoid explique le Dice 0.96+ vs IoU 0.0366.

Compare:
- Dice calculé avec argmax (BUGGY - comme pendant training)
- Dice calculé avec sigmoid > 0.5 (CORRECT - comme pendant inference)
"""

import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import cv2

from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily
from src.inference.optimus_gate_multifamily import ORGAN_TO_FAMILY


def compute_dice_buggy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    Dice calculé avec argmax (BUGGY - comme pendant training).

    Args:
        pred_logits: (B, 2, H, W) - logits bruts
        target: (B, H, W) - masque binaire {0, 1}
    """
    pred_binary = (pred_logits.argmax(dim=1) == 1).float()
    target_float = target.float()

    intersection = (pred_binary * target_float).sum()
    union = pred_binary.sum() + target_float.sum()

    if union == 0:
        return 1.0

    return (2 * intersection / union).item()


def compute_dice_correct(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """
    Dice calculé avec sigmoid > 0.5 (CORRECT - comme pendant inference).

    Args:
        pred_logits: (B, 2, H, W) - logits bruts
        target: (B, H, W) - masque binaire {0, 1}
    """
    # Appliquer sigmoid sur canal 1 (nuclei)
    pred_probs = torch.sigmoid(pred_logits[:, 1, :, :])
    pred_binary = (pred_probs > 0.5).float()
    target_float = target.float()

    intersection = (pred_binary * target_float).sum()
    union = pred_binary.sum() + target_float.sum()

    if union == 0:
        return 1.0

    return (2 * intersection / union).item()


def verify_dice_bug(
    pannuke_dir: Path,
    checkpoint_dir: Path,
    fold: int = 0,
    image_idx: int = 0
):
    """Vérifie que le bug explique la différence Dice training vs inference."""

    print("=" * 70)
    print("VÉRIFICATION BUG ARGMAX vs SIGMOID")
    print("=" * 70)

    # 1. Charger image et GT
    images_path = pannuke_dir / f"fold{fold}" / "images.npy"
    masks_path = pannuke_dir / f"fold{fold}" / "masks.npy"

    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')

    image = images[image_idx]
    mask = masks[image_idx]

    # GT NP mask
    np_mask_gt = mask[:, :, 1:].sum(axis=-1) > 0
    np_mask_gt_224 = cv2.resize(np_mask_gt.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)

    print(f"\n📥 Image PanNuke {image_idx} (fold {fold}):")
    print(f"   GT NP coverage (256x256): {np_mask_gt.sum() / np_mask_gt.size * 100:.2f}%")
    print(f"   GT NP coverage (224x224): {np_mask_gt_224.sum() / np_mask_gt_224.size * 100:.2f}%")

    # 2. Charger modèle et prédire
    print(f"\n🤖 Loading model...")
    model = OptimusGateInferenceMultiFamily(checkpoint_dir=str(checkpoint_dir), device='cuda')

    # Extract features
    tensor = model.preprocess(image)
    features = model.extract_features(tensor)
    cls_token = features[:, 0, :]
    patch_tokens = features[:, 1:257, :]

    # Get organ and family
    pred_idx, probs = model.model.organ_head.predict(cls_token)
    organ_idx = pred_idx[0].item()
    organ_name = model.model.organ_head.organ_names[organ_idx]
    family = ORGAN_TO_FAMILY.get(organ_name, 'glandular')

    print(f"   Organ: {organ_name}, Family: {family}")

    # Get raw NP logits
    hovernet = model.model.hovernet_decoders[family]
    with torch.no_grad():
        np_logits, hv_pred, nt_logits = hovernet(patch_tokens)

    print(f"\n📊 NP Logits:")
    print(f"   Shape: {np_logits.shape}")
    print(f"   Range: [{np_logits.min().item():.3f}, {np_logits.max().item():.3f}]")
    print(f"   Canal 0 (BG) mean: {np_logits[:, 0, :, :].mean().item():.3f}")
    print(f"   Canal 1 (NP) mean: {np_logits[:, 1, :, :].mean().item():.3f}")

    # 3. Convertir GT en tensor
    np_target = torch.from_numpy(np_mask_gt_224).unsqueeze(0).cuda()

    # 4. Calculer Dice avec les deux méthodes
    print(f"\n{'='*70}")
    print("🔍 COMPARAISON DICE")
    print(f"{'='*70}")

    dice_buggy = compute_dice_buggy(np_logits, np_target)
    dice_correct = compute_dice_correct(np_logits, np_target)

    print(f"\n📊 Dice BUGGY (argmax sur logits):")
    print(f"   → {dice_buggy:.4f}")
    print(f"   ✓ C'est le Dice calculé pendant training!")

    print(f"\n📊 Dice CORRECT (sigmoid > 0.5):")
    print(f"   → {dice_correct:.4f}")
    print(f"   ✓ C'est le Dice réel pendant inference!")

    # 5. Analyser les prédictions pixel par pixel
    print(f"\n{'='*70}")
    print("🔬 ANALYSE PIXEL PAR PIXEL")
    print(f"{'='*70}")

    # Méthode BUGGY
    pred_binary_buggy = (np_logits.argmax(dim=1) == 1).cpu().numpy()[0]

    # Méthode CORRECT
    pred_probs = torch.sigmoid(np_logits[:, 1, :, :]).cpu().numpy()[0]
    pred_binary_correct = (pred_probs > 0.5).astype(np.float32)

    # GT
    gt = np_mask_gt_224.astype(np.float32)

    print(f"\n📊 Coverage Comparison:")
    print(f"   GT:               {gt.sum() / gt.size * 100:.2f}%")
    print(f"   BUGGY (argmax):   {pred_binary_buggy.sum() / pred_binary_buggy.size * 100:.2f}%")
    print(f"   CORRECT (sigmoid): {pred_binary_correct.sum() / pred_binary_correct.size * 100:.2f}%")

    # Pixels où argmax dit "nuclei" mais sigmoid < 0.5
    ambiguous = (pred_binary_buggy == 1) & (pred_binary_correct == 0)
    n_ambiguous = ambiguous.sum()

    print(f"\n⚠️  Pixels ambigus (argmax=1 MAIS sigmoid<0.5):")
    print(f"   Count: {n_ambiguous} / {224*224} ({n_ambiguous / (224*224) * 100:.2f}%)")

    if n_ambiguous > 0:
        # Analyser les logits de ces pixels ambigus
        ambiguous_mask = torch.from_numpy(ambiguous).cuda()
        logits_bg = np_logits[0, 0, :, :][ambiguous_mask]
        logits_np = np_logits[0, 1, :, :][ambiguous_mask]

        print(f"\n   Logits pour ces pixels ambigus:")
        print(f"      BG (canal 0) mean: {logits_bg.mean().item():.3f}")
        print(f"      NP (canal 1) mean: {logits_np.mean().item():.3f}")
        print(f"      → NP > BG donc argmax=1 ✓")

        probs_np = torch.sigmoid(logits_np)
        print(f"\n   Probabilités sigmoid:")
        print(f"      NP mean: {probs_np.mean().item():.3f}")
        print(f"      NP range: [{probs_np.min().item():.3f}, {probs_np.max().item():.3f}]")
        print(f"      → Tous < 0.5 donc sigmoid=0 ✓")

    # DIAGNOSTIC FINAL
    print(f"\n{'='*70}")
    print("🎯 DIAGNOSTIC")
    print(f"{'='*70}")

    print(f"\n✅ BUG CONFIRMÉ:")
    print(f"   Training Dice (BUGGY):  {dice_buggy:.4f}")
    print(f"   Inference Dice (CORRECT): {dice_correct:.4f}")
    print(f"   Différence: {abs(dice_buggy - dice_correct):.4f}")

    if dice_buggy > 0.9 and dice_correct < 0.1:
        print(f"\n❌ Le modèle a appris à maximiser argmax(logits), PAS sigmoid > 0.5!")
        print(f"   → Pendant training: Dice {dice_buggy:.4f} (basé sur argmax)")
        print(f"   → Pendant inference: Dice {dice_correct:.4f} (basé sur sigmoid)")
        print(f"\n💡 SOLUTION:")
        print(f"   1. Corriger compute_dice() pour utiliser sigmoid > 0.5")
        print(f"   2. Ré-entraîner tous les modèles avec métrique correcte")
        print(f"   3. Vérifier que les targets d'entraînement sont corrects (instances séparées)")
    elif dice_buggy > 0.9 and dice_correct > 0.9:
        print(f"\n✅ Le modèle fonctionne bien! Les deux méthodes donnent ~0.9+")
    else:
        print(f"\n⚠️  Résultats inattendus, investigation supplémentaire nécessaire")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pannuke_dir", type=Path, default=Path("/home/amar/data/PanNuke"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image_idx", type=int, default=0)

    args = parser.parse_args()

    verify_dice_bug(
        args.pannuke_dir,
        args.checkpoint_dir,
        args.fold,
        args.image_idx
    )


if __name__ == "__main__":
    main()
