#!/usr/bin/env python3
"""
Script de validation visuelle des rotations HV.

Affiche une image PanNuke avec ses 5 rotations (0°, 90°, 180°, 270°, flip_h)
et les noyaux encerclés selon les masques tournés.

Usage:
    python scripts/validation/visualize_rotations.py
    python scripts/validation/visualize_rotations.py --image_idx 42
    python scripts/validation/visualize_rotations.py --save_path results/rotations.png
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.preprocessing.prepare_v13_smart_crops import apply_rotation


def load_test_image(
    pannuke_dir: Path = Path("/home/amar/data/PanNuke"),
    fold: int = 0,
    image_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Charge une image de test depuis PanNuke.

    Args:
        pannuke_dir: Répertoire racine PanNuke
        fold: Numéro du fold (0, 1, ou 2)
        image_idx: Index de l'image à charger

    Returns:
        image (256, 256, 3), mask (256, 256, 6), organ_name
    """
    fold_dir = pannuke_dir / f"fold{fold}"
    images_path = fold_dir / "images.npy"
    masks_path = fold_dir / "masks.npy"
    types_path = fold_dir / "types.npy"

    if not images_path.exists():
        raise FileNotFoundError(f"PanNuke fold{fold} not found at {fold_dir}")

    print(f"📂 Loading image {image_idx} from {fold_dir}")

    # Load avec mmap pour économiser RAM
    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')
    types = np.load(types_path)

    # Vérifier index
    if image_idx >= len(images):
        raise ValueError(f"Image index {image_idx} out of range (max: {len(images)-1})")

    # Copier en mémoire
    image = np.array(images[image_idx], dtype=np.uint8)
    mask = np.array(masks[image_idx])
    organ = types[image_idx].decode('utf-8') if isinstance(types[image_idx], bytes) else types[image_idx]

    print(f"✅ Loaded: {organ} (shape: {image.shape})")

    return image, mask, organ


def extract_nuclei_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    Extrait les contours des noyaux depuis un masque PanNuke.

    Args:
        mask: (H, W, 6) masque PanNuke (canal 0 = background, 1-5 = types)

    Returns:
        Liste de contours OpenCV
    """
    # Union des canaux 1-5 (tous les types de noyaux)
    nuclei_mask = mask[:, :, 1:6].sum(axis=-1) > 0
    nuclei_mask = nuclei_mask.astype(np.uint8) * 255

    # Trouver contours
    contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def draw_contours_on_image(image: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
    """
    Dessine les contours des noyaux sur une image.

    Args:
        image: (H, W, 3) image RGB
        contours: Liste de contours OpenCV

    Returns:
        Image avec contours dessinés
    """
    image_copy = image.copy()

    # Dessiner contours en vert (épais pour visibilité)
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)

    return image_copy


def center_crop(image: np.ndarray, crop_size: int = 224) -> Tuple[np.ndarray, int, int]:
    """
    Crop central 224×224 depuis 256×256.

    Args:
        image: (256, 256, ...) image ou masque
        crop_size: Taille du crop (default: 224)

    Returns:
        cropped_image, x_offset, y_offset
    """
    h, w = image.shape[:2]
    x_offset = (w - crop_size) // 2
    y_offset = (h - crop_size) // 2

    if image.ndim == 3:
        cropped = image[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size, :]
    else:
        cropped = image[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]

    return cropped, x_offset, y_offset


def visualize_rotations(
    image: np.ndarray,
    mask: np.ndarray,
    organ_name: str,
    save_path: Path = None
):
    """
    Visualise les 5 rotations avec noyaux encerclés.

    Args:
        image: (256, 256, 3) image RGB
        mask: (256, 256, 6) masque PanNuke
        organ_name: Nom de l'organe
        save_path: Chemin de sauvegarde (optionnel)
    """
    # Crop central 224×224
    print(f"\n📐 Crop central 224×224...")
    image_224, _, _ = center_crop(image, 224)
    mask_224, _, _ = center_crop(mask, 224)

    # Dummy targets (on s'en fout pour la visualisation)
    np_dummy = np.zeros((224, 224), dtype=np.float32)
    hv_dummy = np.zeros((2, 224, 224), dtype=np.float32)
    nt_dummy = np.zeros((224, 224), dtype=np.int64)

    # Les 5 rotations
    rotations = ['0', '90', '180', '270', 'flip_h']
    rotation_names = {
        '0': '0° (Original)',
        '90': '90° CW',
        '180': '180°',
        '270': '270° CW',
        'flip_h': 'Flip Horizontal'
    }

    # Créer figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Validation Rotations - {organ_name}', fontsize=16, fontweight='bold')

    # Flatten axes pour itération
    axes = axes.flatten()

    for i, rotation in enumerate(rotations):
        print(f"  🔄 Rotation {rotation_names[rotation]}...")

        # Appliquer rotation (utilise la même fonction que prepare_v13_smart_crops.py)
        img_rot, np_rot, hv_rot, nt_rot = apply_rotation(
            image_224, np_dummy, hv_dummy, nt_dummy, rotation
        )

        # Rotation du masque (même logique spatiale)
        if rotation == '0':
            mask_rot = mask_224.copy()
        elif rotation == '90':
            mask_rot = np.rot90(mask_224, k=-1, axes=(0, 1))  # 90° CW
        elif rotation == '180':
            mask_rot = np.rot90(mask_224, k=2, axes=(0, 1))
        elif rotation == '270':
            mask_rot = np.rot90(mask_224, k=1, axes=(0, 1))   # 270° CW = 90° CCW
        elif rotation == 'flip_h':
            mask_rot = np.flip(mask_224, axis=1)

        # Extraire contours depuis masque tourné
        contours = extract_nuclei_contours(mask_rot)

        # Dessiner contours sur image tournée
        img_with_contours = draw_contours_on_image(img_rot, contours)

        # Afficher
        ax = axes[i]
        ax.imshow(img_with_contours)
        ax.set_title(f'{rotation_names[rotation]}\n{len(contours)} noyaux', fontsize=12)
        ax.axis('off')

        # Ajouter texte avec info rotation
        textstr = f'Contours: {len(contours)}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    # Cacher le 6ème subplot (on a que 5 rotations)
    axes[5].axis('off')

    plt.tight_layout()

    # Sauvegarder ou afficher
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Sauvegardé: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description='Visualisation des rotations HV avec noyaux encerclés')
    parser.add_argument('--pannuke_dir', type=Path, default=Path('/home/amar/data/PanNuke'),
                        help='Répertoire PanNuke')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2],
                        help='Numéro du fold PanNuke')
    parser.add_argument('--image_idx', type=int, default=0,
                        help='Index de l\'image à visualiser')
    parser.add_argument('--save_path', type=Path, default=None,
                        help='Chemin de sauvegarde (optionnel, sinon affiche)')

    args = parser.parse_args()

    print("=" * 70)
    print("VALIDATION VISUELLE ROTATIONS HV")
    print("=" * 70)
    print()

    # Charger image
    try:
        image, mask, organ = load_test_image(
            pannuke_dir=args.pannuke_dir,
            fold=args.fold,
            image_idx=args.image_idx
        )
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return 1

    # Visualiser rotations
    print(f"\n🎨 Génération visualisation...")
    visualize_rotations(image, mask, organ, save_path=args.save_path)

    print("\n✅ TERMINÉ")

    return 0


if __name__ == "__main__":
    sys.exit(main())
