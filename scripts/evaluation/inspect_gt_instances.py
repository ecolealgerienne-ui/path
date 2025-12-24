#!/usr/bin/env python3
"""
Diagnostic rapide: Inspecte le GT pour comprendre pourquoi PQ=0.

Usage:
    python scripts/evaluation/inspect_gt_instances.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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


def main():
    print("=" * 80)
    print("🔍 DIAGNOSTIC GT INSTANCES")
    print("=" * 80)

    # Load epidermal data
    data_file = Path("data/family_FIXED/epidermal_data_FIXED.npz")
    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return

    data = np.load(data_file)
    images = data['images']
    fold_ids = data.get('fold_ids', np.zeros(len(images), dtype=np.int32))
    image_ids = data.get('image_ids', np.arange(len(images)))

    print(f"\n📦 Données epidermal:")
    print(f"  Samples: {len(images)}")
    print(f"  Keys: {list(data.keys())}")

    # Load PanNuke masks
    pannuke_dir = Path("/home/amar/data/PanNuke")
    if not pannuke_dir.exists():
        pannuke_dir = Path("data/PanNuke")

    masks = np.load(pannuke_dir / "fold0" / "masks.npy", mmap_mode='r')

    # Test FIRST sample
    idx = 0
    fold_id = fold_ids[idx]
    img_id = image_ids[idx]

    print(f"\n🧪 Test sample: idx={idx}, fold={fold_id}, img_id={img_id}")

    # Load GT
    if fold_id == 0:
        gt_mask = masks[img_id]
    else:
        fold_masks = np.load(pannuke_dir / f"fold{fold_id}" / "masks.npy", mmap_mode='r')
        gt_mask = fold_masks[img_id]

    print(f"\n📊 GT MASK ANALYSIS:")
    print(f"  Shape: {gt_mask.shape}")
    print(f"  Dtype: {gt_mask.dtype}")
    print(f"  Min: {gt_mask.min()}, Max: {gt_mask.max()}")

    # Analyze each channel
    print(f"\n📋 CHANNEL BREAKDOWN:")
    channel_names = ["Type 0", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]

    for c in range(gt_mask.shape[2]):
        channel = gt_mask[:, :, c]
        unique_vals = np.unique(channel)
        nonzero_pixels = (channel > 0).sum()

        print(f"\n  Channel {c} ({channel_names[c]}):")
        print(f"    Unique values: {len(unique_vals)} ({unique_vals[:10].tolist()}...)" if len(unique_vals) > 10 else f"    Unique values: {unique_vals.tolist()}")
        print(f"    Nonzero pixels: {nonzero_pixels}")
        print(f"    Max value: {channel.max()}")

        if channel.max() > 1:
            print(f"    → INSTANCES DETECTED (IDs: {unique_vals[unique_vals > 0][:5].tolist()}...)")
        elif channel.max() == 1:
            print(f"    → BINARY MASK (pas d'instances séparées)")
        else:
            print(f"    → EMPTY")

    # Compute GT instances
    gt_inst = compute_gt_instances(gt_mask)
    n_inst = len(np.unique(gt_inst)) - 1

    print(f"\n🔬 COMPUTE_GT_INSTANCES OUTPUT:")
    print(f"  Instances détectées: {n_inst}")
    print(f"  IDs uniques: {np.unique(gt_inst)[:20].tolist()}")

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(images[idx])
    axes[0].set_title("Image H&E", fontweight='bold')
    axes[0].axis('off')

    # Channels 1-5
    for c in range(1, 6):
        axes[c].imshow(gt_mask[:, :, c], cmap='nipy_spectral')
        axes[c].set_title(f"{channel_names[c]}\n(max={gt_mask[:, :, c].max()})")
        axes[c].axis('off')

    # GT instances
    axes[6].imshow(gt_inst, cmap='nipy_spectral')
    axes[6].set_title(f"GT Instances\n({n_inst} instances)")
    axes[6].axis('off')

    # Binary union
    binary_union = (gt_mask[:, :, 1:].sum(axis=2) > 0).astype(np.uint8)
    axes[7].imshow(binary_union * 255, cmap='gray')
    axes[7].set_title(f"Binary Union\n({binary_union.sum()} pixels)")
    axes[7].axis('off')

    plt.tight_layout()
    output_path = Path("results/debug_alignment/gt_instances_diagnostic.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Image sauvegardée: {output_path}")

    # CONCLUSION
    print("\n" + "=" * 80)
    print("💡 DIAGNOSTIC")
    print("=" * 80)

    if n_inst == 0:
        print("\n❌ PROBLÈME DÉTECTÉ:")
        print("  compute_gt_instances() retourne 0 instances")
        print("  CAUSE: Tous les canaux 1-4 sont vides OU canal 5 vide")
        print("  SOLUTION: Vérifier filtrage famille epidermal")
    elif n_inst == 1:
        print("\n⚠️  PROBLÈME POTENTIEL:")
        print("  compute_gt_instances() retourne 1 seule instance")
        print("  CAUSE: Masques binaires fusionnés (pas de séparation)")
        print("  SOLUTION: Le GT est mal formé")
    else:
        print(f"\n✅ GT INSTANCES OK:")
        print(f"  {n_inst} instances séparées détectées")
        print(f"  Le problème d'AJI vient probablement d'autre chose")

        # Check if binary
        has_instances = False
        for c in range(1, 5):
            if gt_mask[:, :, c].max() > 1:
                has_instances = True
                break

        if not has_instances and gt_mask[:, :, 5].max() <= 1:
            print("\n⚠️  ATTENTION:")
            print("  Canaux 1-4: vides ou binaires")
            print("  Canal 5: binaire (connectedComponents utilisé)")
            print("  Les instances viennent de connectedComponents, pas des vraies annotations PanNuke")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
