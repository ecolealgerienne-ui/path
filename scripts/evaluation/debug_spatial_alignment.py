#!/usr/bin/env python3
"""
Script de debug pour visualiser l'alignement spatial pred vs GT.

Usage:
    python scripts/evaluation/debug_spatial_alignment.py \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform


def compute_gt_instances(mask: np.ndarray) -> np.ndarray:
    """Compute GT instances depuis masque PanNuke."""
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: vraies instances annotées
    for c in range(1, 5):
        class_instances = mask[:, :, c]
        inst_ids = np.unique(class_instances)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = class_instances == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
    epithelial_binary = mask[:, :, 5] > 0
    if epithelial_binary.any():
        _, epithelial_labels = cv2.connectedComponents(epithelial_binary.astype(np.uint8))
        epithelial_ids = np.unique(epithelial_labels)
        epithelial_ids = epithelial_ids[epithelial_ids > 0]

        for epi_id in epithelial_ids:
            epi_mask = epithelial_labels == epi_id
            inst_map[epi_mask] = instance_counter
            instance_counter += 1

    return inst_map


def create_overlay(pred_binary: np.ndarray, gt_binary: np.ndarray, title: str) -> np.ndarray:
    """
    Crée une image overlay RGB:
    - Rouge: Prédictions
    - Vert: Ground Truth
    - Jaune: Intersection (rouge + vert)
    """
    h, w = pred_binary.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Rouge: Prédictions
    overlay[:, :, 0] = (pred_binary * 255).astype(np.uint8)

    # Vert: Ground Truth
    overlay[:, :, 1] = (gt_binary * 255).astype(np.uint8)

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Debug spatial alignment")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint HoVer-Net")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", default="results/debug_alignment", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔍 DEBUG SPATIAL ALIGNMENT")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")

    # Load model
    print("\n🔧 Chargement modèle...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    backbone.eval()

    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    transform = create_hoptimus_transform()

    # Load ONE epidermal test sample
    print("\n📦 Chargement données...")
    data_file = Path("data/family_FIXED/epidermal_data_FIXED.npz")
    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return

    data = np.load(data_file)
    images = data['images']

    # Load GT masks
    pannuke_dir = Path("/home/amar/data/PanNuke")
    if not pannuke_dir.exists():
        pannuke_dir = Path("data/PanNuke")

    masks = np.load(pannuke_dir / "fold0" / "masks.npy", mmap_mode='r')
    fold_ids = data.get('fold_ids', np.zeros(len(images), dtype=np.int32))
    image_ids = data.get('image_ids', np.arange(len(images)))

    # Take FIRST sample
    idx = 0
    image = images[idx]
    fold_id = fold_ids[idx]
    img_id = image_ids[idx]

    # Load GT
    if fold_id == 0:
        gt_mask = masks[img_id]
    else:
        fold_masks = np.load(pannuke_dir / f"fold{fold_id}" / "masks.npy", mmap_mode='r')
        gt_mask = fold_masks[img_id]

    print(f"\n🧪 Test sample: fold={fold_id}, img_id={img_id}")

    # Preprocess
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)

    tensor = transform(image).unsqueeze(0).to(args.device)

    # Extract features & predict
    print("🔬 Inférence...")
    with torch.no_grad():
        features = backbone.forward_features(tensor)
        patch_tokens = features[:, 1:257, :]
        np_out, hv_out, nt_out = hovernet(patch_tokens)

    # Convert to numpy with correct axes
    np_pred = torch.softmax(np_out, dim=1)[0].cpu().numpy().transpose(1, 2, 0)  # (224, 224, 2)
    hv_pred = hv_out[0].cpu().numpy().transpose(1, 2, 0)

    # Center padding 224→256
    diff = (256 - 224) // 2
    np_pred_256 = np.zeros((256, 256, 2), dtype=np_pred.dtype)
    np_pred_256[diff:diff+224, diff:diff+224, :] = np_pred

    # Extract prediction binary
    prob_map = np_pred_256[:, :, 1]
    pred_binary = (prob_map > 0.5).astype(np.uint8)

    # Extract GT binary
    gt_inst = compute_gt_instances(gt_mask)
    gt_binary = (gt_inst > 0).astype(np.uint8)

    print(f"\n📊 Stats:")
    print(f"  Pred pixels: {pred_binary.sum()}")
    print(f"  GT pixels: {gt_binary.sum()}")
    print(f"  Intersection: {(pred_binary & gt_binary).sum()}")
    print(f"  Union: {(pred_binary | gt_binary).sum()}")
    iou = (pred_binary & gt_binary).sum() / ((pred_binary | gt_binary).sum() + 1e-6)
    print(f"  IoU: {iou:.4f}")

    # Create overlays with different transformations
    print("\n🎨 Génération overlays...")

    transformations = [
        ("original", pred_binary, "Aucune transformation"),
        ("flipud", np.flipud(pred_binary), "Flip Vertical (haut↔bas)"),
        ("fliplr", np.fliplr(pred_binary), "Flip Horizontal (gauche↔droite)"),
        ("transpose", pred_binary.T, "Transpose (X↔Y)"),
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Image H&E Originale", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # GT only
    axes[1].imshow(gt_binary * 255, cmap='Greens')
    axes[1].set_title(f"Ground Truth ({gt_binary.sum()} pixels)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Test different transformations
    for i, (name, pred_transformed, desc) in enumerate(transformations):
        overlay = create_overlay(pred_transformed, gt_binary, desc)

        # Calculate IoU for this transformation
        intersection = (pred_transformed & gt_binary).sum()
        union = (pred_transformed | gt_binary).sum()
        iou_transform = intersection / (union + 1e-6)

        axes[i + 2].imshow(overlay)
        axes[i + 2].set_title(
            f"{desc}\nIoU: {iou_transform:.4f}",
            fontsize=12,
            fontweight='bold' if iou_transform == max([
                ((t & gt_binary).sum() / ((t | gt_binary).sum() + 1e-6))
                for _, t, _ in transformations
            ]) else 'normal'
        )
        axes[i + 2].axis('off')

        print(f"  {name:12s}: IoU = {iou_transform:.4f}")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Prédictions'),
        Patch(facecolor='green', label='Ground Truth'),
        Patch(facecolor='yellow', label='Intersection (accord)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    output_path = output_dir / "spatial_alignment_debug.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Image sauvegardée: {output_path}")

    # Find best transformation
    best_iou = max([
        ((t & gt_binary).sum() / ((t | gt_binary).sum() + 1e-6))
        for _, t, _ in transformations
    ])
    best_transform = [name for name, t, _ in transformations
                      if ((t & gt_binary).sum() / ((t | gt_binary).sum() + 1e-6)) == best_iou][0]

    print("\n" + "=" * 80)
    print("🎯 RÉSULTAT")
    print("=" * 80)
    print(f"\n✅ Meilleure transformation: {best_transform.upper()}")
    print(f"   IoU: {best_iou:.4f}")

    if best_transform == "original":
        print("\n💡 INTERPRÉTATION:")
        print("   L'alignement est déjà optimal (pas de transformation nécessaire).")
        print("   Le problème d'AJI faible vient probablement de:")
        print("   - Sur-segmentation (trop de petites instances)")
        print("   - Paramètres watershed à ajuster (min_size, dist_threshold)")
    else:
        print(f"\n💡 INTERPRÉTATION:")
        print(f"   Les prédictions nécessitent une transformation {best_transform}.")
        print(f"   Applique cette transformation dans test_epidermal_aji_FINAL.py:")
        print(f"   prob_map_256 = np.{best_transform}(prob_map_256)")
        print(f"   hv_map_256 = np.{best_transform}(hv_map_256)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
