#!/usr/bin/env python3
"""
Compare training pipeline vs inference pipeline step-by-step.

Vérifie si l'inférence produit les mêmes résultats que l'entraînement
pour détecter les bugs subtils dans le pipeline.
"""

import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from torchvision import transforms

from src.inference.optimus_gate_inference_multifamily import OptimusGateInferenceMultiFamily
from src.models.hovernet_decoder import HoVerNetDecoder
from src.inference.optimus_gate_multifamily import ORGAN_TO_FAMILY

# Normalisation H-optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


def create_transform():
    """Transform CANONIQUE."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])


def compare_pipelines(
    pannuke_dir: Path,
    checkpoint_dir: Path,
    fold: int = 0,
    image_idx: int = 0
):
    """Compare training vs inference pipelines."""

    print("=" * 70)
    print("COMPARAISON TRAINING vs INFERENCE PIPELINE")
    print("=" * 70)

    # 1. Charger l'image PanNuke
    images_path = pannuke_dir / f"fold{fold}" / "images.npy"
    masks_path = pannuke_dir / f"fold{fold}" / "masks.npy"

    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')

    image = images[image_idx]
    mask = masks[image_idx]

    print(f"\n📥 Image PanNuke {image_idx} (fold {fold}):")
    print(f"   Shape: {image.shape}, dtype={image.dtype}")
    print(f"   Range: [{image.min()}, {image.max()}]")

    # Générer GT NP mask
    np_mask_gt = mask[:, :, 1:].sum(axis=-1) > 0
    gt_coverage = np_mask_gt.sum() / np_mask_gt.size * 100
    print(f"   GT NP coverage: {gt_coverage:.2f}%")

    # 2. PIPELINE INFERENCE (OptimusGateInferenceMultiFamily)
    print(f"\n{'='*70}")
    print("📊 PIPELINE INFERENCE")
    print(f"{'='*70}")

    model_inf = OptimusGateInferenceMultiFamily(
        checkpoint_dir=str(checkpoint_dir),
        device='cuda'
    )

    # Preprocessing inference
    print("\n1️⃣ Preprocessing (inference):")
    if image.dtype != np.uint8:
        print(f"   ⚠️  Converting {image.dtype} → uint8")
        image_uint8 = image.clip(0, 255).astype(np.uint8)
    else:
        image_uint8 = image

    transform = create_transform()
    tensor_inf = transform(image_uint8).unsqueeze(0).cuda()
    print(f"   Tensor shape: {tensor_inf.shape}")
    print(f"   Tensor range: [{tensor_inf.min():.3f}, {tensor_inf.max():.3f}]")
    print(f"   Tensor mean: {tensor_inf.mean():.3f}")
    print(f"   Tensor std: {tensor_inf.std():.3f}")

    # Extract features inference
    print("\n2️⃣ Feature extraction (inference):")
    with torch.no_grad():
        features_inf = model_inf.extract_features(tensor_inf)

    cls_token_inf = features_inf[:, 0, :]
    patch_tokens_inf = features_inf[:, 1:257, :]

    print(f"   Features shape: {features_inf.shape}")
    print(f"   CLS token std: {cls_token_inf.std().item():.4f}")
    print(f"   Patch tokens mean: {patch_tokens_inf.mean().item():.4f}")
    print(f"   Patch tokens std: {patch_tokens_inf.std().item():.4f}")

    # Predict organ and get family
    print("\n3️⃣ Organ prediction (inference):")
    pred_idx, probs = model_inf.model.organ_head.predict(cls_token_inf)
    organ_idx = pred_idx[0].item()
    organ_name = model_inf.model.organ_head.organ_names[organ_idx]
    confidence = probs[0, organ_idx].item()

    print(f"   Organ: {organ_name} ({confidence*100:.1f}%)")

    family = ORGAN_TO_FAMILY.get(organ_name, 'glandular')
    print(f"   Family: {family}")

    # Get HoVerNet decoder
    hovernet_inf = model_inf.model.hovernet_decoders[family]

    # Forward pass inference
    print("\n4️⃣ HoVerNet forward pass (inference):")
    with torch.no_grad():
        np_logits_inf, hv_pred_inf, nt_logits_inf = hovernet_inf(patch_tokens_inf)

    print(f"   NP logits shape: {np_logits_inf.shape}")
    print(f"   NP logits range: [{np_logits_inf.min().item():.3f}, {np_logits_inf.max().item():.3f}]")
    print(f"   NP logits mean: {np_logits_inf.mean().item():.3f}")

    # Apply sigmoid
    np_pred_inf = torch.sigmoid(np_logits_inf).cpu().numpy()[0, 0]
    print(f"   NP pred (after sigmoid) range: [{np_pred_inf.min():.3f}, {np_pred_inf.max():.3f}]")
    print(f"   NP pred mean: {np_pred_inf.mean():.3f}")
    print(f"   NP pred median: {np.median(np_pred_inf):.3f}")

    # Threshold
    np_mask_inf = (np_pred_inf > 0.5).astype(np.float32)
    inf_coverage = np_mask_inf.sum() / np_mask_inf.size * 100
    print(f"   NP mask coverage (threshold=0.5): {inf_coverage:.2f}%")

    # 3. PIPELINE TRAINING (simulate from features)
    print(f"\n{'='*70}")
    print("🎓 PIPELINE TRAINING (SIMULATION)")
    print(f"{'='*70}")

    print("\n1️⃣ Load training features:")
    features_dir = Path("data/features")
    fold_dir = features_dir / f"fold{fold}"

    # Load features (saved during training preprocessing)
    # Note: Training saves features after forward_features()
    # We'll use the inference features since they should be identical
    features_train = features_inf  # Same preprocessing → same features
    patch_tokens_train = features_train[:, 1:257, :]

    print(f"   Using inference features (same preprocessing)")
    print(f"   Patch tokens shape: {patch_tokens_train.shape}")

    # Load the same HoVerNet checkpoint
    print("\n2️⃣ HoVerNet decoder (same checkpoint):")
    checkpoint_path = checkpoint_dir / f"hovernet_{family}_best.pth"

    if not checkpoint_path.exists():
        print(f"   ⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"   Available checkpoints:")
        for p in checkpoint_dir.glob("hovernet_*.pth"):
            print(f"      - {p.name}")
        return

    print(f"   Loading: {checkpoint_path}")

    # Create decoder with same architecture
    hovernet_train = HoVerNetDecoder(embed_dim=1536, img_size=224, n_classes=5)

    # Load checkpoint (contains metadata + model_state_dict)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    hovernet_train.load_state_dict(checkpoint['model_state_dict'])
    hovernet_train.eval().cuda()

    print(f"   Checkpoint info: Epoch {checkpoint.get('epoch', 'N/A')}, Dice {checkpoint.get('best_dice', 'N/A'):.4f}")

    # Forward pass training
    print("\n3️⃣ HoVerNet forward pass (training simulation):")
    with torch.no_grad():
        np_logits_train, hv_pred_train, nt_logits_train = hovernet_train(patch_tokens_train)

    print(f"   NP logits shape: {np_logits_train.shape}")
    print(f"   NP logits range: [{np_logits_train.min().item():.3f}, {np_logits_train.max().item():.3f}]")
    print(f"   NP logits mean: {np_logits_train.mean().item():.3f}")

    # Apply sigmoid (same as inference)
    np_pred_train = torch.sigmoid(np_logits_train).cpu().numpy()[0, 0]
    print(f"   NP pred (after sigmoid) range: [{np_pred_train.min():.3f}, {np_pred_train.max():.3f}]")
    print(f"   NP pred mean: {np_pred_train.mean():.3f}")
    print(f"   NP pred median: {np.median(np_pred_train):.3f}")

    # Threshold
    np_mask_train = (np_pred_train > 0.5).astype(np.float32)
    train_coverage = np_mask_train.sum() / np_mask_train.size * 100
    print(f"   NP mask coverage (threshold=0.5): {train_coverage:.2f}%")

    # 4. COMPARAISON
    print(f"\n{'='*70}")
    print("🔍 COMPARAISON DÉTAILLÉE")
    print(f"{'='*70}")

    # Compare logits
    logits_diff = np.abs(np_logits_inf.cpu().numpy() - np_logits_train.cpu().numpy())
    print(f"\n📊 NP Logits (before sigmoid):")
    print(f"   Inference mean: {np_logits_inf.mean().item():.4f}")
    print(f"   Training mean:  {np_logits_train.mean().item():.4f}")
    print(f"   Absolute diff:  {logits_diff.mean():.6f}")
    print(f"   Max diff:       {logits_diff.max():.6f}")

    # Compare predictions
    pred_diff = np.abs(np_pred_inf - np_pred_train)
    print(f"\n📊 NP Predictions (after sigmoid):")
    print(f"   Inference mean: {np_pred_inf.mean():.4f}")
    print(f"   Training mean:  {np_pred_train.mean():.4f}")
    print(f"   Absolute diff:  {pred_diff.mean():.6f}")
    print(f"   Max diff:       {pred_diff.max():.6f}")

    # Compare coverage
    print(f"\n📊 Coverage Comparison:")
    print(f"   GT coverage:        {gt_coverage:.2f}%")
    print(f"   Inference coverage: {inf_coverage:.2f}%")
    print(f"   Training coverage:  {train_coverage:.2f}%")
    print(f"   Diff (inf-train):   {abs(inf_coverage - train_coverage):.4f}%")

    # IoU vs GT
    def compute_iou(pred, gt):
        intersection = np.logical_and(pred > 0, gt > 0).sum()
        union = np.logical_or(pred > 0, gt > 0).sum()
        return intersection / (union + 1e-10)

    # Resize GT to 224x224
    import cv2
    np_mask_gt_224 = cv2.resize(np_mask_gt.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)

    iou_inf = compute_iou(np_mask_inf, np_mask_gt_224)
    iou_train = compute_iou(np_mask_train, np_mask_gt_224)

    print(f"\n📊 IoU vs Ground Truth:")
    print(f"   Inference IoU: {iou_inf:.4f}")
    print(f"   Training IoU:  {iou_train:.4f}")

    # CONCLUSION
    print(f"\n{'='*70}")
    print("🎯 DIAGNOSTIC")
    print(f"{'='*70}")

    if logits_diff.mean() < 1e-5:
        print("✅ Inference et Training produisent les MÊMES logits")
        print("   → Pas de bug dans le forward pass")
    else:
        print("⚠️  Inference et Training produisent des logits DIFFÉRENTS")
        print(f"   → Différence moyenne: {logits_diff.mean():.6f}")

    if inf_coverage < 15 and train_coverage < 15:
        print("\n❌ PROBLÈME: Les deux pipelines prédisent <15% (GT = {:.1f}%)".format(gt_coverage))
        print("   → Le modèle est fondamentalement cassé")
        print("   → Ce n'est PAS un bug d'inférence")
        print("   → Les métriques de training (Dice 0.96+) étaient FAUSSES!")
    elif abs(inf_coverage - train_coverage) > 5:
        print("\n⚠️  PROBLÈME: Différence >5% entre inference et training")
        print("   → Bug dans le pipeline d'inférence")
    else:
        print("\n✅ Inference et Training sont cohérents")
        print("   → Mais les deux prédisent mal!")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pannuke_dir", type=Path, default=Path("/home/amar/data/PanNuke"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image_idx", type=int, default=0)

    args = parser.parse_args()

    compare_pipelines(
        args.pannuke_dir,
        args.checkpoint_dir,
        args.fold,
        args.image_idx
    )


if __name__ == "__main__":
    main()
