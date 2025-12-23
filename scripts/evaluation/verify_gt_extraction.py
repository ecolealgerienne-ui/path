#!/usr/bin/env python3
"""
Vérification extraction GT: connectedComponents vs extract_pannuke_instances

Compare les deux méthodes pour montrer que connectedComponents fusionne les instances.

Usage:
    python scripts/evaluation/verify_gt_extraction.py \
        --family epidermal \
        --sample_idx 0 \
        --data_dir /home/amar/data/PanNuke
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE


def extract_gt_connectedcomponents(np_target: np.ndarray) -> np.ndarray:
    """
    Extraction GT méthode eval_aji_from_training_data.py

    PROBLÈME: Fusionne les cellules qui se touchent
    """
    np_binary = (np_target > 0.5).astype(np.uint8)
    _, inst_map = cv2.connectedComponents(np_binary)
    return inst_map.astype(np.int32)


def extract_gt_pannuke_native(mask: np.ndarray) -> np.ndarray:
    """
    Extraction GT méthode CORRECTE (prepare_family_data_FIXED.py)

    Utilise les IDs d'instances natifs PanNuke (canaux 1-4).
    """
    inst_map = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: IDs d'instances natifs PanNuke
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
    epithelial_binary = (mask[:, :, 5] > 0).astype(np.uint8)
    if epithelial_binary.sum() > 0:
        _, epithelial_labels = cv2.connectedComponents(epithelial_binary)
        epithelial_ids = np.unique(epithelial_labels)
        epithelial_ids = epithelial_ids[epithelial_ids > 0]

        for inst_id in epithelial_ids:
            inst_mask = epithelial_labels == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    return inst_map


def colorize_instances(inst_map: np.ndarray) -> np.ndarray:
    """Colorise les instances pour visualisation."""
    n_instances = inst_map.max()

    if n_instances == 0:
        return np.zeros((*inst_map.shape, 3), dtype=np.uint8)

    # Colormap aléatoire
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(n_instances + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background noir

    colored = colors[inst_map]
    return colored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--sample_idx', type=int, default=0,
                        help="Index de l'échantillon à vérifier")
    parser.add_argument('--data_dir', type=Path, required=True,
                        help="Répertoire PanNuke brut")

    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    cache_dir = PROJECT_ROOT / "data" / "cache" / "family_data"

    print(f"\n{'='*70}")
    print(f"VÉRIFICATION EXTRACTION GT - Famille {args.family.upper()}")
    print(f"{'='*70}\n")

    # ========================================================================
    # PARTIE 1: Charger données training (.npz)
    # ========================================================================

    print("📥 Chargement données training (.npz)...")

    targets_file = cache_dir / f"{args.family}_targets.npz"
    if not targets_file.exists():
        raise FileNotFoundError(f"Targets non trouvés: {targets_file}")

    targets_data = np.load(targets_file)

    np_target = targets_data['np_targets'][args.sample_idx]  # (256, 256)
    fold_id = targets_data['fold_ids'][args.sample_idx]
    image_id = targets_data['image_ids'][args.sample_idx]

    print(f"   Sample: idx={args.sample_idx}, fold={fold_id}, image_id={image_id}")

    # Extraction GT méthode connectedComponents
    inst_gt_cc = extract_gt_connectedcomponents(np_target)
    n_instances_cc = len(np.unique(inst_gt_cc)) - 1

    print(f"\n   Méthode connectedComponents:")
    print(f"      → {n_instances_cc} instances détectées")

    # ========================================================================
    # PARTIE 2: Charger image brute PanNuke correspondante
    # ========================================================================

    print(f"\n📥 Chargement PanNuke brut (fold {fold_id})...")

    fold_dir = args.data_dir / f"fold{fold_id}"
    images_path = fold_dir / "images.npy"
    masks_path = fold_dir / "masks.npy"

    if not images_path.exists():
        raise FileNotFoundError(f"Images non trouvées: {images_path}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks non trouvés: {masks_path}")

    # Charger UNIQUEMENT l'image demandée (memory-mapped)
    images_full = np.load(images_path, mmap_mode='r')
    masks_full = np.load(masks_path, mmap_mode='r')

    image = images_full[image_id].copy()
    mask = masks_full[image_id].copy()

    print(f"   Image shape: {image.shape}, Mask shape: {mask.shape}")

    # Extraction GT méthode native PanNuke
    inst_gt_native = extract_gt_pannuke_native(mask)
    n_instances_native = len(np.unique(inst_gt_native)) - 1

    print(f"\n   Méthode extract_pannuke_native:")
    print(f"      → {n_instances_native} instances détectées")

    # ========================================================================
    # PARTIE 3: Comparaison
    # ========================================================================

    print(f"\n{'='*70}")
    print("RÉSULTATS COMPARAISON")
    print(f"{'='*70}\n")

    print(f"connectedComponents:    {n_instances_cc:3d} instances")
    print(f"PanNuke Native:         {n_instances_native:3d} instances")
    print(f"Différence:             {n_instances_native - n_instances_cc:3d} instances perdues")

    if n_instances_cc > 0:
        pct_loss = 100 * (n_instances_native - n_instances_cc) / n_instances_native
        print(f"Perte:                  {pct_loss:.1f}%")

    print()

    # Analyse détaillée canaux
    print("Détails par canal PanNuke:")
    for c in range(1, 6):
        channel_mask = mask[:, :, c]
        n_unique = len(np.unique(channel_mask)) - 1
        if n_unique > 0:
            channel_names = ['Neo', 'Infl', 'Conn', 'Dead', 'Epit']
            print(f"   Canal {c} ({channel_names[c-1]:5s}): {n_unique:3d} instances")

    # ========================================================================
    # PARTIE 4: Visualisation
    # ========================================================================

    print(f"\n📊 Génération visualisation...")

    # Coloriser instances
    colored_cc = colorize_instances(inst_gt_cc)
    colored_native = colorize_instances(inst_gt_native)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title(f"Image H&E (fold {fold_id}, idx {image_id})", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(colored_cc)
    axes[1].set_title(f"connectedComponents\n{n_instances_cc} instances", fontsize=14, color='red')
    axes[1].axis('off')

    axes[2].imshow(colored_native)
    axes[2].set_title(f"PanNuke Native (CORRECT)\n{n_instances_native} instances", fontsize=14, color='green')
    axes[2].axis('off')

    plt.tight_layout()

    output_path = PROJECT_ROOT / "results" / f"verify_gt_{args.family}_sample{args.sample_idx}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f"   ✅ Sauvegardé: {output_path}")

    # ========================================================================
    # PARTIE 5: Conclusion
    # ========================================================================

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}\n")

    if n_instances_cc < n_instances_native:
        print("❌ PROBLÈME CONFIRMÉ:")
        print(f"   connectedComponents FUSIONNE les cellules qui se touchent")
        print(f"   {n_instances_native - n_instances_cc} instances perdues sur cette image")
        print()
        print("   Impact sur eval_aji_from_training_data.py:")
        print("   → Compare 'mauvaises instances' vs 'mauvaises instances'")
        print("   → AJI artificiellement élevé (0.94)")
        print()
        print("   Impact sur eval_aji_from_images.py:")
        print("   → Compare contre les VRAIES instances PanNuke")
        print("   → AJI faible (0.30) car watershed ne sépare pas assez")
    else:
        print("✅ Pas de différence détectée sur cette image")
        print("   Tester avec d'autres échantillons")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
