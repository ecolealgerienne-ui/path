#!/usr/bin/env python3
"""
SIPaKMeD Dataset Preprocessing with Proper Padding Strategy

CRITICAL RULE FOR CYTOLOGY:
========================
❌ NEVER resize small images (80×80) to 224×224 by stretching
   → Destroys chromatin texture (Haralick features)
   → Blurs nuclear details critical for malignancy detection

✅ ALWAYS use Zero-Padding with white background
   → Preserves native resolution 100%
   → Maintains texture information
   → Natural appearance (matches microscope slide background)

Why White Padding (value=255)?
==============================
- SIPaKMeD cells have bright/white background (microscope slide)
- Black padding (0) creates artificial dark borders
- White padding (255) or BORDER_REFLECT keeps image natural
- Model learns from real cell features, not padding artifacts

Author: CellViT-Optimus V14
Date: 2026-01-19
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

# Target size for classification network
TARGET_SIZE = 224

# =========================================================================
# TRANSFORMATION PIPELINE (TRAINING)
# =========================================================================

def get_train_transform():
    """
    Training pipeline with proper padding strategy

    Pipeline:
    1. PadIfNeeded: Pad small images to 224×224 with WHITE background
    2. Resize (only if > 224): Center crop if too large
    3. Augmentation: Rotate, flip, color jitter (optional)
    4. Normalize: ImageNet statistics
    5. ToTensor: Convert to PyTorch tensor
    """
    return A.Compose([
        # STEP 1: Pad small images (< 224×224) with WHITE background
        # CRITICAL: This preserves texture at native resolution!
        A.PadIfNeeded(
            min_height=TARGET_SIZE,
            min_width=TARGET_SIZE,
            border_mode=cv2.BORDER_CONSTANT,  # Solid fill
            value=255,  # WHITE padding (matches microscope background)
            p=1.0
        ),

        # STEP 2: Crop if image is larger than 224×224 (rare in SIPaKMeD)
        A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),

        # STEP 3: Data Augmentation (optional, can be disabled)
        A.OneOf([
            A.RandomRotate90(p=1.0),
            A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0),
        ], p=0.5),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Light color augmentation (preserve stain appearance)
        A.ColorJitter(
            brightness=0.1,  # Slight brightness variation
            contrast=0.1,    # Slight contrast variation
            saturation=0.1,  # Preserve stain colors
            hue=0.02,        # Minimal hue shift
            p=0.3
        ),

        # STEP 4: Normalize to ImageNet statistics
        # (Can be changed to SIPaKMeD-specific mean/std if computed)
        A.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet mean (RGB)
            std=(0.229, 0.224, 0.225),   # ImageNet std (RGB)
        ),

        # STEP 5: Convert to PyTorch tensor
        ToTensorV2(),
    ])


def get_val_transform():
    """
    Validation pipeline (NO augmentation)

    Only padding and normalization - no random transformations
    """
    return A.Compose([
        # Pad to 224×224 with white background
        A.PadIfNeeded(
            min_height=TARGET_SIZE,
            min_width=TARGET_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,  # WHITE padding
            p=1.0
        ),

        # Crop if needed (safety check)
        A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),

        # Normalize (same as training)
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        # Convert to tensor
        ToTensorV2(),
    ])


# =========================================================================
# ALTERNATIVE: BORDER_REFLECT (Mirror Padding)
# =========================================================================

def get_train_transform_reflect():
    """
    Alternative: Mirror padding instead of white padding

    Reflects pixels at borders instead of filling with solid color.
    Can be better for very small cells (avoids hard edges).
    """
    return A.Compose([
        A.PadIfNeeded(
            min_height=TARGET_SIZE,
            min_width=TARGET_SIZE,
            border_mode=cv2.BORDER_REFLECT_101,  # Mirror padding
            p=1.0
        ),
        A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# =========================================================================
# VISUALIZATION: Compare Padding Strategies
# =========================================================================

def visualize_padding_strategies(img_path, output_dir="visualizations"):
    """
    Compare different padding strategies visually

    Creates side-by-side comparison:
    - Original small image (e.g., 80×80)
    - ❌ BAD: Resized (stretched) to 224×224
    - ✅ GOOD: Padded with white to 224×224
    - ✅ GOOD: Padded with reflect to 224×224
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load original image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    print(f"Original size: {w}×{h}")

    # Strategy 1: ❌ BAD - Resize (stretch)
    img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_CUBIC)

    # Strategy 2: ✅ GOOD - White padding
    transform_white = A.Compose([
        A.PadIfNeeded(TARGET_SIZE, TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=255)
    ])
    img_padded_white = transform_white(image=img)['image']

    # Strategy 3: ✅ GOOD - Reflect padding
    transform_reflect = A.Compose([
        A.PadIfNeeded(TARGET_SIZE, TARGET_SIZE, border_mode=cv2.BORDER_REFLECT_101)
    ])
    img_padded_reflect = transform_reflect(image=img)['image']

    # Save comparisons
    Image.fromarray(img).save(output_dir / f"1_original_{w}x{h}.png")
    Image.fromarray(img_resized).save(output_dir / f"2_BAD_resized_224x224_BLURRY.png")
    Image.fromarray(img_padded_white).save(output_dir / f"3_GOOD_padded_white_224x224.png")
    Image.fromarray(img_padded_reflect).save(output_dir / f"4_GOOD_padded_reflect_224x224.png")

    print(f"✅ Visualizations saved to {output_dir}/")
    print(f"   → Compare 2_BAD (blurry) vs 3_GOOD/4_GOOD (sharp)")


# =========================================================================
# DATASET ANALYSIS: Check Image Sizes
# =========================================================================

def analyze_sipakmed_sizes(sipakmed_dir="data/raw/sipakmed/pictures"):
    """
    Analyze SIPaKMeD image dimensions to confirm padding is needed
    """
    sipakmed_dir = Path(sipakmed_dir)

    sizes = []
    small_count = 0  # Images < 224×224

    print("="*70)
    print("📊 ANALYZING SIPaKMeD IMAGE DIMENSIONS")
    print("="*70)

    for class_dir in sipakmed_dir.iterdir():
        if not class_dir.is_dir():
            continue

        for img_path in class_dir.glob("*.bmp"):
            try:
                img = Image.open(img_path)
                w, h = img.size
                sizes.append((w, h))

                if w < TARGET_SIZE or h < TARGET_SIZE:
                    small_count += 1
            except Exception as e:
                print(f"⚠️  Error reading {img_path}: {e}")

    if not sizes:
        print("❌ No images found")
        return

    sizes = np.array(sizes)

    print(f"\n📏 Image Size Statistics:")
    print(f"   Total images: {len(sizes)}")
    print(f"   Width:  min={sizes[:, 0].min()} | max={sizes[:, 0].max()} | mean={sizes[:, 0].mean():.0f}")
    print(f"   Height: min={sizes[:, 1].min()} | max={sizes[:, 1].max()} | mean={sizes[:, 1].mean():.0f}")

    print(f"\n⚠️  Images smaller than {TARGET_SIZE}×{TARGET_SIZE}: {small_count}/{len(sizes)} ({small_count/len(sizes)*100:.1f}%)")

    if small_count > 0:
        print(f"\n✅ PADDING IS REQUIRED")
        print(f"   → Use Albumentations PadIfNeeded with value=255 (white)")
        print(f"   → This preserves texture at native resolution")
    else:
        print(f"\n✅ All images ≥ {TARGET_SIZE}×{TARGET_SIZE}")
        print(f"   → Padding not strictly needed, but use for safety")

    return sizes


# =========================================================================
# EXAMPLE USAGE
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SIPaKMeD Padding Strategy Demo")
    parser.add_argument("--analyze", action="store_true", help="Analyze image sizes")
    parser.add_argument("--visualize", type=str, help="Visualize padding on sample image")
    parser.add_argument("--sipakmed_dir", default="data/raw/sipakmed/pictures", help="SIPaKMeD directory")

    args = parser.parse_args()

    if args.analyze:
        analyze_sipakmed_sizes(args.sipakmed_dir)

    if args.visualize:
        visualize_padding_strategies(args.visualize)

    if not args.analyze and not args.visualize:
        print("="*70)
        print("🔬 SIPaKMeD PADDING STRATEGY")
        print("="*70)
        print("\nUsage:")
        print("  python sipakmed_padding.py --analyze")
        print("  python sipakmed_padding.py --visualize data/raw/sipakmed/.../cell.bmp")
        print("\nTransformations available:")
        print("  - get_train_transform(): Training with white padding + augmentation")
        print("  - get_val_transform(): Validation with white padding only")
        print("  - get_train_transform_reflect(): Training with mirror padding")
        print("\n" + "="*70)
        print("\n📋 CRITICAL RULES:")
        print("="*70)
        print("❌ NEVER use cv2.resize() or Image.resize() for small cells")
        print("   → Destroys chromatin texture (Haralick features)")
        print("   → Blurs nuclear membrane irregularities")
        print("\n✅ ALWAYS use Albumentations.PadIfNeeded()")
        print("   → Preserves native resolution 100%")
        print("   → value=255 for white padding (matches microscope background)")
        print("   → border_mode=BORDER_REFLECT_101 for mirror padding (alternative)")
        print("\n🎯 Why This Matters:")
        print("   - Malignancy detection relies on nuclear texture granularity")
        print("   - Chromatin Density (Criterion 3 - ISBI 2014) requires sharp details")
        print("   - Haralick features (contrast, homogeneity) need original texture")
        print("   - Blurring = Loss of critical diagnostic information")
