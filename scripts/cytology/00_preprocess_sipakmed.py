"""
SIPaKMeD Preprocessing — Resize to 224×224 with White Padding

Ce script prépare les données SIPaKMeD pour le pipeline V14 Cytologie:
1. Load images BMP (dimensions variables: 69×70, etc.)
2. Padding blanc → 224×224 (préserve texture nucléaire)
3. Normalisation Macenko (optionnel)
4. Load masques ground truth (-d.bmp)
5. Resize masques → 224×224 (nearest neighbor)
6. Split train/val stratifié (80/20)
7. Sauvegarde PNG organized

Author: V14 Cytology Branch
Date: 2026-01-19
"""

import os
import glob
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

TARGET_SIZE = 224
CLASSES = [
    'normal_columnar',
    'normal_intermediate',
    'normal_superficiel',
    'light_dysplastic',
    'moderate_dysplastic',
    'severe_dysplastic',
    'carcinoma_in_situ'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


# ═════════════════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ═════════════════════════════════════════════════════════════════════════════

def get_preprocessing_transform(use_macenko: bool = False):
    """
    Transform pour preprocessing (padding blanc 224×224)

    Args:
        use_macenko: Appliquer Macenko normalization (recommandé pour cytologie)

    Returns:
        Albumentations Compose
    """
    transforms_list = [
        # CRITIQUE: Padding blanc (préserve texture)
        A.PadIfNeeded(
            min_height=TARGET_SIZE,
            min_width=TARGET_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,  # Blanc (fond microscope)
            mask_value=0,  # Fond = 0 pour masques
            p=1.0
        ),

        # Center crop si image plus grande (rare)
        A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    ]

    return A.Compose(transforms_list)


def apply_macenko_normalization(image: np.ndarray) -> np.ndarray:
    """
    Applique Macenko normalization (optionnel)

    Args:
        image: RGB image (H, W, 3)

    Returns:
        Normalized image

    Note:
        Nécessite torchstain ou custom implementation.
        Pour l'instant, retourne image inchangée (TODO).
    """
    # TODO: Implémenter Macenko avec torchstain
    # from torchstain import MacenkoNormalizer
    # normalizer.fit(reference_image)
    # normalized = normalizer.normalize(image)

    # Pour l'instant: passthrough
    return image


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_sipakmed_dataset(
    raw_dir: str,
    use_macenko: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[str]]:
    """
    Charge et preprocess tout le dataset SIPaKMeD

    Args:
        raw_dir: data/raw/sipakmed/pictures/
        use_macenko: Appliquer Macenko normalization

    Returns:
        images: List of preprocessed images (224, 224, 3)
        masks: List of preprocessed masks (224, 224) — classe noyau uniquement
        labels: List of class indices
        filenames: List of original filenames
    """
    transform = get_preprocessing_transform(use_macenko)

    images = []
    masks = []
    labels = []
    filenames = []

    print(f"📁 Loading SIPaKMeD from: {raw_dir}")

    for class_name in CLASSES:
        class_dir = os.path.join(raw_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠️  Skipping {class_name} (directory not found)")
            continue

        # Find all .BMP files (uppercase)
        image_files = sorted(glob.glob(os.path.join(class_dir, "*.BMP")))

        print(f"\n🔬 Processing {class_name}: {len(image_files)} images")

        for img_file in tqdm(image_files, desc=f"  {class_name}"):
            # Construire chemin masque
            mask_file = img_file.replace(".BMP", "-d.bmp")

            if not os.path.exists(mask_file):
                print(f"⚠️  Mask not found: {mask_file}")
                continue

            # Load image
            image_pil = Image.open(img_file).convert('RGB')
            image = np.array(image_pil)

            # Load mask (8-bit indexed)
            mask_pil = Image.open(mask_file)
            mask_indexed = np.array(mask_pil)

            # Extraire classe noyau (valeur 2 dans masque SIPaKMeD)
            # SIPaKMeD mask values: 0=artefact, 1=artefact, 2=NUCLEUS, 3=cytoplasm, 4=background
            mask_nucleus = (mask_indexed == 2).astype(np.uint8)

            # Macenko normalization (optionnel)
            if use_macenko:
                image = apply_macenko_normalization(image)

            # Apply padding/crop transform
            transformed = transform(image=image, mask=mask_nucleus)
            image_preprocessed = transformed['image']
            mask_preprocessed = transformed['mask']

            # Validation
            assert image_preprocessed.shape == (TARGET_SIZE, TARGET_SIZE, 3), \
                f"Image shape incorrect: {image_preprocessed.shape}"
            assert mask_preprocessed.shape == (TARGET_SIZE, TARGET_SIZE), \
                f"Mask shape incorrect: {mask_preprocessed.shape}"

            # Ajouter
            images.append(image_preprocessed)
            masks.append(mask_preprocessed)
            labels.append(CLASS_TO_IDX[class_name])
            filenames.append(os.path.basename(img_file))

    print(f"\n✅ Loaded {len(images)} samples total")

    return images, masks, labels, filenames


# ═════════════════════════════════════════════════════════════════════════════
#  TRAIN/VAL SPLIT
# ═════════════════════════════════════════════════════════════════════════════

def stratified_split(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    labels: List[int],
    filenames: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict, Dict]:
    """
    Split stratifié train/val (80/20)

    Args:
        images, masks, labels, filenames: Data from load_sipakmed_dataset
        test_size: Fraction validation (0.2 = 20%)
        random_state: Random seed

    Returns:
        train_data: Dict with 'images', 'masks', 'labels', 'filenames'
        val_data: Dict with same structure
    """
    # Créer indices
    indices = np.arange(len(images))

    # Stratified split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Construire dicts
    train_data = {
        'images': [images[i] for i in train_idx],
        'masks': [masks[i] for i in train_idx],
        'labels': [labels[i] for i in train_idx],
        'filenames': [filenames[i] for i in train_idx]
    }

    val_data = {
        'images': [images[i] for i in val_idx],
        'masks': [masks[i] for i in val_idx],
        'labels': [labels[i] for i in val_idx],
        'filenames': [filenames[i] for i in val_idx]
    }

    print(f"\n📊 Split Statistics:")
    print(f"  Train: {len(train_data['images'])} samples")
    print(f"  Val:   {len(val_data['images'])} samples")

    # Distribution par classe
    print(f"\n  Class Distribution (Train):")
    for cls_idx, cls_name in enumerate(CLASSES):
        count = sum(1 for lbl in train_data['labels'] if lbl == cls_idx)
        print(f"    {cls_name}: {count}")

    return train_data, val_data


# ═════════════════════════════════════════════════════════════════════════════
#  SAVE
# ═════════════════════════════════════════════════════════════════════════════

def save_preprocessed_dataset(
    data: Dict,
    output_dir: str,
    split_name: str
):
    """
    Sauvegarde dataset preprocessed

    Args:
        data: Dict with 'images', 'masks', 'labels', 'filenames'
        output_dir: data/processed/sipakmed/
        split_name: 'train' or 'val'
    """
    split_dir = os.path.join(output_dir, split_name)
    images_dir = os.path.join(split_dir, 'images')
    masks_dir = os.path.join(split_dir, 'masks')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"\n💾 Saving {split_name} split to: {split_dir}")

    metadata = []

    for idx, (image, mask, label, filename) in enumerate(tqdm(
        zip(data['images'], data['masks'], data['labels'], data['filenames']),
        total=len(data['images']),
        desc=f"  Saving {split_name}"
    )):
        # Nom fichier sans extension + index
        base_name = os.path.splitext(filename)[0]
        save_name = f"{idx:05d}_{base_name}"

        # Sauvegarder image (PNG)
        image_path = os.path.join(images_dir, f"{save_name}.png")
        Image.fromarray(image).save(image_path)

        # Sauvegarder masque (PNG binary)
        mask_path = os.path.join(masks_dir, f"{save_name}_mask.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

        # Metadata
        metadata.append({
            'id': idx,
            'filename': save_name,
            'original_filename': filename,
            'label': int(label),
            'class_name': CLASSES[label],
            'image_path': f"images/{save_name}.png",
            'mask_path': f"masks/{save_name}_mask.png"
        })

    # Sauvegarder metadata JSON
    metadata_path = os.path.join(split_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Saved {len(metadata)} samples")
    print(f"  📄 Metadata: {metadata_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SIPaKMeD dataset (224×224 with white padding)"
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw/sipakmed/pictures',
        help='Path to raw SIPaKMeD directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/sipakmed',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--use_macenko',
        action='store_true',
        help='Apply Macenko normalization (recommended for cytology)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Validation split size (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SIPAKMED PREPROCESSING — V14 Cytology")
    print("=" * 80)
    print(f"Raw directory:    {args.raw_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target size:      {TARGET_SIZE}×{TARGET_SIZE}")
    print(f"Macenko:          {'ON' if args.use_macenko else 'OFF'}")
    print(f"Val split:        {args.test_size * 100:.0f}%")
    print("=" * 80)

    # Load dataset
    images, masks, labels, filenames = load_sipakmed_dataset(
        raw_dir=args.raw_dir,
        use_macenko=args.use_macenko
    )

    # Split train/val
    train_data, val_data = stratified_split(
        images, masks, labels, filenames,
        test_size=args.test_size,
        random_state=args.seed
    )

    # Save
    save_preprocessed_dataset(train_data, args.output_dir, 'train')
    save_preprocessed_dataset(val_data, args.output_dir, 'val')

    # Summary
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total samples:    {len(images)}")
    print(f"Train samples:    {len(train_data['images'])}")
    print(f"Val samples:      {len(val_data['images'])}")
    print(f"Output directory: {args.output_dir}")
    print("\nNext step:")
    print("  python scripts/cytology/00b_validate_cellpose.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
