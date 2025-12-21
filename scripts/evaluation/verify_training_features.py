#!/usr/bin/env python3
"""
Vérifie si les features utilisées pour training étaient corrompues (BUG ToPILImage).

Compare les features training vs features fraîches avec le fix uint8.
"""

import argparse
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import timm

# Normalisation H-optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

EXPECTED_CLS_STD_MIN = 0.70
EXPECTED_CLS_STD_MAX = 0.90


def create_transform_OLD_BUGGY():
    """Transform BUGGY (avant fix) - ToPILImage multiplie float64 par 255."""
    return transforms.Compose([
        transforms.ToPILImage(),  # BUG: multiplie float64 par 255!
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])


def create_transform_FIXED():
    """Transform FIXED (après fix) - Conversion uint8 AVANT ToPILImage."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])


def preprocess_FIXED(image: np.ndarray) -> torch.Tensor:
    """Prétraitement avec FIX uint8."""
    # CRITICAL FIX: Convertir en uint8 AVANT ToPILImage
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    transform = create_transform_FIXED()
    return transform(image).unsqueeze(0)


def verify_training_features(
    family_data_dir: Path,
    pannuke_dir: Path,
    family: str = "glandular",
    sample_idx: int = 0,
    device: str = "cuda"
):
    """Vérifie si les features training sont corrompues."""

    print("=" * 70)
    print("VÉRIFICATION FEATURES TRAINING (BUG ToPILImage?)")
    print("=" * 70)

    # 1. Charger les features training
    features_path = family_data_dir / f"{family}_features.npz"
    if not features_path.exists():
        print(f"❌ Features training non trouvées: {features_path}")
        return

    print(f"\n📂 Loading training features: {features_path}")
    data = np.load(features_path, mmap_mode='r')

    # Check available keys
    print(f"   Available keys: {list(data.keys())}")

    if 'features' in data:
        features_train = data['features']
    elif 'layer_24' in data:
        features_train = data['layer_24']
    else:
        print(f"❌ No features found in .npz file")
        return

    print(f"   Shape: {features_train.shape}")

    # Get one sample
    feature_train = features_train[sample_idx]  # (261, 1536)
    cls_train = feature_train[0]  # (1536,)

    print(f"\n🎯 TRAINING FEATURES (sample {sample_idx}):")
    print(f"   CLS token std: {cls_train.std():.4f}")
    print(f"   CLS token mean: {cls_train.mean():.4f}")
    print(f"   Expected std range: [{EXPECTED_CLS_STD_MIN}, {EXPECTED_CLS_STD_MAX}]")

    if cls_train.std() < EXPECTED_CLS_STD_MIN:
        print(f"   ⚠️  WARNING: CLS std too low! Likely CORRUPTED (no LayerNorm)")
        corrupted_layernorm = True
    else:
        print(f"   ✅ CLS std in expected range (has LayerNorm)")
        corrupted_layernorm = False

    # 2. Charger l'image correspondante depuis PanNuke
    # On doit trouver quel fold/index correspond à ce sample dans la famille
    # Pour simplifier, on va juste extraire des features sur fold2/image2 (qu'on utilise pour test)

    print(f"\n📥 Loading PanNuke fold 2, image 2...")
    images_path = pannuke_dir / "fold2" / "images.npy"
    images = np.load(images_path)
    image = images[2]  # Image #2

    print(f"   Image: {image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")

    # 3. Extraire features avec la méthode BUGGY (avant fix)
    print(f"\n🐛 Extracting features with BUGGY method (no uint8 conversion)...")

    # Load H-optimus-0
    print("   Loading H-optimus-0...")
    backbone = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    backbone.eval().to(device)

    # BUGGY extraction (ToPILImage on float64 directly)
    transform_buggy = create_transform_OLD_BUGGY()
    with torch.no_grad():
        tensor_buggy = transform_buggy(image).unsqueeze(0).to(device)
        features_buggy = backbone.forward_features(tensor_buggy)
        features_buggy = features_buggy.cpu().numpy()[0]

    cls_buggy = features_buggy[0]
    print(f"   CLS token std (BUGGY): {cls_buggy.std():.4f}")
    print(f"   CLS token mean (BUGGY): {cls_buggy.mean():.4f}")

    # 4. Extraire features avec la méthode FIXED (après fix)
    print(f"\n✅ Extracting features with FIXED method (uint8 conversion)...")

    with torch.no_grad():
        tensor_fixed = preprocess_FIXED(image).to(device)
        features_fixed = backbone.forward_features(tensor_fixed)
        features_fixed = features_fixed.cpu().numpy()[0]

    cls_fixed = features_fixed[0]
    print(f"   CLS token std (FIXED): {cls_fixed.std():.4f}")
    print(f"   CLS token mean (FIXED): {cls_fixed.mean():.4f}")

    # 5. Comparaison
    print(f"\n" + "=" * 70)
    print("🔍 DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(f"\nCLS Token Std Comparison:")
    print(f"   Training features: {cls_train.std():.4f}")
    print(f"   BUGGY extraction:  {cls_buggy.std():.4f}")
    print(f"   FIXED extraction:  {cls_fixed.std():.4f}")
    print(f"   Expected range:    [{EXPECTED_CLS_STD_MIN}, {EXPECTED_CLS_STD_MAX}]")

    # Distance entre training et buggy vs training et fixed
    dist_buggy = np.linalg.norm(cls_train - cls_buggy)
    dist_fixed = np.linalg.norm(cls_train - cls_fixed)

    print(f"\nL2 Distance from training features:")
    print(f"   To BUGGY:  {dist_buggy:.2f}")
    print(f"   To FIXED:  {dist_fixed:.2f}")

    print(f"\n" + "=" * 70)
    print("🎯 CONCLUSION")
    print("=" * 70)

    if corrupted_layernorm:
        print("❌ TRAINING FEATURES ARE CORRUPTED (no LayerNorm)")
        print("   → CLS std too low (<0.70)")
        print("   → Models trained on wrong features")
        print("   → MUST RETRAIN EVERYTHING")
    elif dist_buggy < dist_fixed:
        print("❌ TRAINING FEATURES MATCH BUGGY EXTRACTION")
        print("   → float64 was multiplied by 255 (ToPILImage bug)")
        print("   → Features are CORRUPTED")
        print("   → MUST RETRAIN with uint8 fix")
    else:
        print("✅ TRAINING FEATURES MATCH FIXED EXTRACTION")
        print("   → Features were extracted correctly")
        print("   → Problem is elsewhere (investigate model architecture)")
        print("   → Retraining may not be necessary")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family_data_dir", type=Path, default=Path("data/family"))
    parser.add_argument("--pannuke_dir", type=Path, default=Path("/home/amar/data/PanNuke"))
    parser.add_argument("--family", type=str, default="glandular")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    verify_training_features(
        args.family_data_dir,
        args.pannuke_dir,
        args.family,
        args.sample_idx,
        args.device
    )


if __name__ == "__main__":
    main()
