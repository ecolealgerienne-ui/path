#!/usr/bin/env python3
"""
Compare les vraies instances PanNuke avec celles générées par connectedComponents.

PanNuke structure:
- mask[:, :, 0]: Background
- mask[:, :, 1]: Neoplastic instance IDs [0, 88, 96, 107, ...]
- mask[:, :, 2]: Inflammatory instance IDs
- mask[:, :, 3]: Connective instance IDs
- mask[:, :, 4]: Dead instance IDs
- mask[:, :, 5]: Epithelial (binaire, pas d'IDs)

Vérifie si connectedComponents fusionne les cellules qui se touchent.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def compare_pannuke_instances(
    pannuke_dir: Path,
    fold: int = 0,
    image_idx: int = 2,  # Image avec plusieurs cellules
    output_dir: Path = Path("results/pannuke_instances")
):
    """Compare vraies instances PanNuke vs connectedComponents."""

    print("=" * 70)
    print("COMPARAISON INSTANCES PANNUKE")
    print("=" * 70)

    # 1. Charger PanNuke
    images_path = pannuke_dir / f"fold{fold}" / "images.npy"
    masks_path = pannuke_dir / f"fold{fold}" / "masks.npy"

    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')

    image = images[image_idx]
    mask = masks[image_idx]

    print(f"\n📥 PanNuke fold {fold}, image {image_idx}:")
    print(f"   Image: {image.shape}, dtype={image.dtype}")
    print(f"   Mask: {mask.shape}, dtype={mask.dtype}")

    # 2. Méthode PANNUKE (vraies instances)
    print(f"\n{'='*70}")
    print("🎯 MÉTHODE PANNUKE (VRAIES INSTANCES)")
    print(f"{'='*70}")

    # Extraire les IDs d'instances de PanNuke
    inst_map_pannuke = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    channel_names = ['BG', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

    for c in range(1, 6):
        channel_mask = mask[:, :, c]

        if c == 5:
            # Canal 5 (Epithelial) est binaire, pas d'IDs
            # Utiliser connectedComponents seulement pour ce canal
            _, epithelial_labels = cv2.connectedComponents(channel_mask.astype(np.uint8))
            epithelial_ids = np.unique(epithelial_labels)
            epithelial_ids = epithelial_ids[epithelial_ids > 0]
            n_epithelial = len(epithelial_ids)

            for inst_id in epithelial_ids:
                inst_mask = epithelial_labels == inst_id
                inst_map_pannuke[inst_mask] = instance_counter
                instance_counter += 1

            print(f"   Canal {c} ({channel_names[c]:15}): {n_epithelial:3} instances (binary → connectedComponents)")
        else:
            # Canaux 1-4: IDs d'instances déjà annotées
            inst_ids = np.unique(channel_mask)
            inst_ids = inst_ids[inst_ids > 0]  # Exclude 0 = background
            n_instances = len(inst_ids)

            for inst_id in inst_ids:
                inst_mask = channel_mask == inst_id
                inst_map_pannuke[inst_mask] = instance_counter
                instance_counter += 1

            print(f"   Canal {c} ({channel_names[c]:15}): {n_instances:3} instances (IDs natifs PanNuke)")

    n_total_pannuke = instance_counter - 1
    print(f"\n   ✅ Total instances PanNuke: {n_total_pannuke}")

    # Coverage
    coverage_pannuke = (inst_map_pannuke > 0).sum() / (256*256) * 100
    print(f"   Coverage: {coverage_pannuke:.2f}%")

    # 3. Méthode CONNECTEDCOMPONENTS (utilisée par prepare_family_data.py)
    print(f"\n{'='*70}")
    print("🔧 MÉTHODE CONNECTEDCOMPONENTS (PREPARE_FAMILY_DATA.PY)")
    print(f"{'='*70}")

    # Union binaire de tous les canaux 1-5
    np_mask_binary = mask[:, :, 1:].sum(axis=-1) > 0

    # ConnectedComponents sur l'union binaire
    _, inst_map_connected = cv2.connectedComponents(np_mask_binary.astype(np.uint8))

    inst_ids_connected = np.unique(inst_map_connected)
    inst_ids_connected = inst_ids_connected[inst_ids_connected > 0]
    n_total_connected = len(inst_ids_connected)

    print(f"   Union binaire (canaux 1-5) → connectedComponents")
    print(f"   ✅ Total instances: {n_total_connected}")

    # Coverage
    coverage_connected = (inst_map_connected > 0).sum() / (256*256) * 100
    print(f"   Coverage: {coverage_connected:.2f}%")

    # 4. Comparaison
    print(f"\n{'='*70}")
    print("🔍 COMPARAISON")
    print(f"{'='*70}")

    print(f"\n📊 Nombre d'instances:")
    print(f"   PanNuke (vraies instances):        {n_total_pannuke}")
    print(f"   ConnectedComponents (fusion):      {n_total_connected}")
    print(f"   Ratio (fusion/vraies):             {n_total_connected / n_total_pannuke:.2f}x")

    if n_total_connected < n_total_pannuke:
        fusion_rate = (n_total_pannuke - n_total_connected) / n_total_pannuke * 100
        print(f"\n⚠️  FUSION DÉTECTÉE: {fusion_rate:.1f}% des instances fusionnées!")
        print(f"   → {n_total_pannuke - n_total_connected} instances perdues par fusion")
    else:
        print(f"\n✅ Pas de fusion détectée")

    # 5. Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Image, PanNuke instances, ConnectedComponents
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image H&E", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(inst_map_pannuke, cmap='tab20')
    axes[0, 1].set_title(f"PanNuke (vraies instances)\n{n_total_pannuke} instances", fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(inst_map_connected, cmap='tab20')
    axes[0, 2].set_title(f"ConnectedComponents (fusion)\n{n_total_connected} instances", fontsize=14)
    axes[0, 2].axis('off')

    # Row 2: Breakdowns
    # Per-channel instances (PanNuke)
    channel_counts = []
    for c in range(1, 6):
        if c == 5:
            _, labels = cv2.connectedComponents(mask[:, :, c].astype(np.uint8))
            count = len(np.unique(labels)) - 1
        else:
            count = len(np.unique(mask[:, :, c])) - 1
        channel_counts.append(count)

    axes[1, 0].bar(range(1, 6), channel_counts, color=['red', 'green', 'blue', 'yellow', 'cyan'])
    axes[1, 0].set_xticks(range(1, 6))
    axes[1, 0].set_xticklabels([channel_names[i][:4] for i in range(1, 6)], rotation=45)
    axes[1, 0].set_title("Instances par Canal (PanNuke)", fontsize=14)
    axes[1, 0].set_ylabel("Nombre d'instances")
    axes[1, 0].grid(True, alpha=0.3)

    # Fusion map (différence)
    fusion_map = np.zeros_like(inst_map_pannuke)
    # Mark pixels where multiple PanNuke instances are merged into 1 connectedComponent
    for conn_id in inst_ids_connected:
        conn_mask = inst_map_connected == conn_id
        pannuke_ids_in_conn = np.unique(inst_map_pannuke[conn_mask])
        pannuke_ids_in_conn = pannuke_ids_in_conn[pannuke_ids_in_conn > 0]

        if len(pannuke_ids_in_conn) > 1:
            # Multiple PanNuke instances merged
            fusion_map[conn_mask] = len(pannuke_ids_in_conn)

    axes[1, 1].imshow(fusion_map, cmap='hot', vmin=0, vmax=fusion_map.max())
    axes[1, 1].set_title(f"Fusion Map\n(Rouge = plusieurs instances fusionnées)", fontsize=14)
    axes[1, 1].axis('off')

    # Overlay comparison
    overlay = np.zeros((256, 256, 3), dtype=np.uint8)
    overlay[inst_map_pannuke > 0, 0] = 255  # PanNuke in red
    overlay[inst_map_connected > 0, 1] = 255  # Connected in green (overlap = yellow)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Overlay: Red=PanNuke, Green=Connected", fontsize=14)
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"fold{fold}_image{image_idx}_instances_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")

    # 6. Analyse détaillée des fusions
    if n_total_connected < n_total_pannuke:
        print(f"\n{'='*70}")
        print("🔬 ANALYSE DES FUSIONS")
        print(f"{'='*70}")

        n_fusions = 0
        max_fusion_size = 0

        for conn_id in inst_ids_connected:
            conn_mask = inst_map_connected == conn_id
            pannuke_ids_in_conn = np.unique(inst_map_pannuke[conn_mask])
            pannuke_ids_in_conn = pannuke_ids_in_conn[pannuke_ids_in_conn > 0]

            if len(pannuke_ids_in_conn) > 1:
                n_fusions += 1
                max_fusion_size = max(max_fusion_size, len(pannuke_ids_in_conn))

        print(f"\n   Nombre de régions fusionnées: {n_fusions}")
        print(f"   Taille max de fusion: {max_fusion_size} instances PanNuke → 1 connectedComponent")

        # Example fusion
        for conn_id in inst_ids_connected:
            conn_mask = inst_map_connected == conn_id
            pannuke_ids_in_conn = np.unique(inst_map_pannuke[conn_mask])
            pannuke_ids_in_conn = pannuke_ids_in_conn[pannuke_ids_in_conn > 0]

            if len(pannuke_ids_in_conn) == max_fusion_size:
                print(f"\n   Exemple de fusion maximale:")
                print(f"      ConnectedComponent ID {conn_id} contient {len(pannuke_ids_in_conn)} instances PanNuke:")
                print(f"      IDs: {pannuke_ids_in_conn[:10]}{'...' if len(pannuke_ids_in_conn) > 10 else ''}")
                break

    # DIAGNOSTIC FINAL
    print(f"\n{'='*70}")
    print("🎯 DIAGNOSTIC")
    print(f"{'='*70}")

    if n_total_connected < n_total_pannuke * 0.9:
        fusion_rate = (n_total_pannuke - n_total_connected) / n_total_pannuke * 100
        print(f"\n❌ BUG CONFIRMÉ: ConnectedComponents fusionne {fusion_rate:.1f}% des instances!")
        print(f"\n   IMPACT:")
        print(f"      - Training HV maps calculées sur instances FUSIONNÉES")
        print(f"      - Gradients HV FAIBLES aux frontières réelles")
        print(f"      - Watershed ne peut PAS séparer les cellules")
        print(f"      - Modèle crée 1 INSTANCE GÉANTE au lieu de {n_total_pannuke}")
        print(f"\n   💡 SOLUTION:")
        print(f"      1. Modifier prepare_family_data.py pour utiliser IDs natifs PanNuke")
        print(f"      2. Ré-calculer HV targets avec vraies instances")
        print(f"      3. Ré-entraîner HoVer-Net HV branch (~10h pour 5 familles)")
    else:
        print(f"\n✅ Pas de fusion significative (<10%)")
        print(f"   → Le problème est ailleurs")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pannuke_dir", type=Path, default=Path("/home/amar/data/PanNuke"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image_idx", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("results/pannuke_instances"))

    args = parser.parse_args()

    compare_pannuke_instances(
        args.pannuke_dir,
        args.fold,
        args.image_idx,
        args.output_dir
    )


if __name__ == "__main__":
    main()
