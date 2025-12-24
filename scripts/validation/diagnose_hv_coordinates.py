#!/usr/bin/env python3
"""
DIAGNOSTIC VISUEL: Vérification Pixel-Perfect des Coordonnées HV

Ce script visualise EXACTEMENT ce qui se passe pour comprendre pourquoi
l'alignement spatial échoue (86px au lieu de <2px).

Questions à répondre:
1. Les vecteurs HV pointent-ils vers les bons centres?
2. Y a-t-il un swap (x↔y)?
3. Y a-t-il un flip (miroir vertical/horizontal)?
4. La normalisation donne-t-elle des valeurs [-1, +1]?
5. Normalisation bbox vs image - quelle différence?

Usage:
    python scripts/validation/diagnose_hv_coordinates.py --sample 70
    python scripts/validation/diagnose_hv_coordinates.py --sample 509 --family epidermal
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_instance_hv_original(inst_mask: np.ndarray) -> tuple:
    """
    Calcule HV pour UNE instance avec normalisation ACTUELLE (v4).

    Returns:
        (hv_map, centroid, stats)
    """
    h, w = inst_mask.shape
    y_coords, x_coords = np.where(inst_mask)

    if len(y_coords) == 0:
        return None, None, None

    # Centroid
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)

    # NORMALISATION V4 (par rapport à l'image complète)
    y_dist = (y_coords - center_y) / (h / 2)
    x_dist = (x_coords - center_x) / (w / 2)

    hv_map = np.zeros((2, h, w), dtype=np.float32)
    hv_map[0, y_coords, x_coords] = y_dist  # Vertical
    hv_map[1, y_coords, x_coords] = x_dist  # Horizontal

    stats = {
        'centroid': (center_y, center_x),
        'bbox': (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max()),
        'hv_range_v': (y_dist.min(), y_dist.max()),
        'hv_range_h': (x_dist.min(), x_dist.max()),
        'normalization': 'image_size (v4)',
    }

    return hv_map, (center_y, center_x), stats


def compute_instance_hv_bbox(inst_mask: np.ndarray) -> tuple:
    """
    Calcule HV pour UNE instance avec normalisation BBOX (HoVer-Net original).

    Returns:
        (hv_map, centroid, stats)
    """
    h, w = inst_mask.shape
    y_coords, x_coords = np.where(inst_mask)

    if len(y_coords) == 0:
        return None, None, None

    # Centroid
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)

    # Bounding box
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    # NORMALISATION BBOX (HoVer-Net original)
    bbox_h = y_max - y_min + 1e-6
    bbox_w = x_max - x_min + 1e-6

    y_dist = 2.0 * (y_coords - y_min) / bbox_h - 1.0
    x_dist = 2.0 * (x_coords - x_min) / bbox_w - 1.0

    hv_map = np.zeros((2, h, w), dtype=np.float32)
    hv_map[0, y_coords, x_coords] = y_dist  # Vertical
    hv_map[1, y_coords, x_coords] = x_dist  # Horizontal

    stats = {
        'centroid': (center_y, center_x),
        'bbox': (y_min, y_max, x_min, x_max),
        'hv_range_v': (y_dist.min(), y_dist.max()),
        'hv_range_h': (x_dist.min(), x_dist.max()),
        'normalization': 'bbox (HoVer-Net original)',
    }

    return hv_map, (center_y, center_x), stats


def extract_instances_from_mask(mask: np.ndarray) -> np.ndarray:
    """Extrait instance map depuis masque PanNuke (6 canaux)."""
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: instances annotées
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


def visualize_hv_vectors(
    image: np.ndarray,
    inst_map: np.ndarray,
    hv_map_v4: np.ndarray,
    hv_map_bbox: np.ndarray,
    sample_idx: int,
    output_dir: Path
):
    """
    Visualise les vecteurs HV pour TOUTES les instances.

    Compare:
    - Normalisation v4 (image size)
    - Normalisation bbox (HoVer-Net original)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identifier les instances
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    n_instances = len(inst_ids)
    print(f"\n{'='*80}")
    print(f"ANALYSE ÉCHANTILLON {sample_idx}")
    print(f"{'='*80}")
    print(f"Nombre d'instances: {n_instances}")

    # Créer figure avec 3 colonnes: Image | V4 | BBOX
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Colonne 1: Image H&E + instance contours
    axes[0].imshow(image)
    axes[0].set_title(f"Image H&E + Instance Contours\n({n_instances} instances)", fontsize=10)
    axes[0].axis('off')

    # Dessiner contours des instances
    for inst_id in inst_ids:
        inst_mask = (inst_map == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:
                axes[0].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1, alpha=0.7)

    # Colonnes 2 et 3: HV vectors (v4 vs bbox)
    for ax_idx, (hv_map, title, method) in enumerate([
        (hv_map_v4, "Normalisation V4 (Image Size)", "v4"),
        (hv_map_bbox, "Normalisation BBOX (HoVer-Net Original)", "bbox")
    ], start=1):
        ax = axes[ax_idx]
        ax.imshow(image, alpha=0.3)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        # Pour chaque instance, afficher quelques vecteurs
        for inst_id in inst_ids[:5]:  # Limiter à 5 instances pour lisibilité
            inst_mask = inst_map == inst_id
            y_coords, x_coords = np.where(inst_mask)

            if len(y_coords) == 0:
                continue

            # Centroid
            cy, cx = np.mean(y_coords), np.mean(x_coords)

            # Prendre échantillon de points (tous les 5 pixels)
            sample_indices = np.arange(0, len(y_coords), 5)

            for idx in sample_indices:
                y, x = y_coords[idx], x_coords[idx]
                hv_y = hv_map[0, y, x]
                hv_x = hv_map[1, y, x]

                # Vecteur HV (inversé car HV pointe VERS le centre)
                # On dessine le vecteur DEPUIS le pixel VERS le centre
                arrow = FancyArrowPatch(
                    (x, y),
                    (x - hv_x * 10, y - hv_y * 10),  # Échelle arbitraire pour visualisation
                    arrowstyle='->', mutation_scale=10,
                    color='lime', linewidth=0.5, alpha=0.6
                )
                ax.add_patch(arrow)

            # Marquer le centroid
            ax.plot(cx, cy, 'r*', markersize=8)

    plt.tight_layout()
    output_path = output_dir / f"hv_vectors_sample_{sample_idx:04d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Visualisation sauvée: {output_path}")

    # Analyser les statistiques par instance
    print(f"\n{'='*80}")
    print(f"STATISTIQUES PAR INSTANCE")
    print(f"{'='*80}")
    print(f"\n{'ID':>4s} | {'V4 Range V':>20s} | {'V4 Range H':>20s} | {'BBOX Range V':>20s} | {'BBOX Range H':>20s}")
    print(f"{'-'*4}|{'-'*22}|{'-'*22}|{'-'*22}|{'-'*22}")

    for inst_id in inst_ids[:10]:  # Afficher 10 premières instances
        inst_mask = inst_map == inst_id

        # Stats V4
        _, _, stats_v4 = compute_instance_hv_original(inst_mask)

        # Stats BBOX
        _, _, stats_bbox = compute_instance_hv_bbox(inst_mask)

        if stats_v4 and stats_bbox:
            v4_range_v = f"[{stats_v4['hv_range_v'][0]:+.3f}, {stats_v4['hv_range_v'][1]:+.3f}]"
            v4_range_h = f"[{stats_v4['hv_range_h'][0]:+.3f}, {stats_v4['hv_range_h'][1]:+.3f}]"
            bbox_range_v = f"[{stats_bbox['hv_range_v'][0]:+.3f}, {stats_bbox['hv_range_v'][1]:+.3f}]"
            bbox_range_h = f"[{stats_bbox['hv_range_h'][0]:+.3f}, {stats_bbox['hv_range_h'][1]:+.3f}]"

            print(f"{inst_id:4d} | {v4_range_v:>20s} | {v4_range_h:>20s} | {bbox_range_v:>20s} | {bbox_range_h:>20s}")

    # Résumé global
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ")
    print(f"{'='*80}")

    # Calculer range global pour v4 et bbox
    all_hv_v4 = hv_map_v4[hv_map_v4 != 0]
    all_hv_bbox = hv_map_bbox[hv_map_bbox != 0]

    print(f"\nV4 (Image Size Normalization):")
    print(f"  HV Range: [{all_hv_v4.min():.4f}, {all_hv_v4.max():.4f}]")
    print(f"  Median abs value: {np.median(np.abs(all_hv_v4)):.4f}")
    print(f"  ⚠️  Problème attendu: Valeurs très proches de 0 pour petites cellules")

    print(f"\nBBOX (HoVer-Net Original Normalization):")
    print(f"  HV Range: [{all_hv_bbox.min():.4f}, {all_hv_bbox.max():.4f}]")
    print(f"  Median abs value: {np.median(np.abs(all_hv_bbox)):.4f}")
    print(f"  ✅ Attendu: Proche de [-1, +1] pour toutes les cellules")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic visuel HV coordinates")
    parser.add_argument("--sample", type=int, default=70,
                        help="Index de l'échantillon à diagnostiquer (défaut: 70 = 18.30px)")
    parser.add_argument("--family", type=str, default="epidermal")
    parser.add_argument("--output_dir", type=str, default="results/hv_diagnosis")
    args = parser.parse_args()

    print("="*80)
    print("DIAGNOSTIC VISUEL: COORDONNÉES HV")
    print("="*80)
    print(f"\nFamille: {args.family}")
    print(f"Échantillon: {args.sample}")

    # 1. Charger les données
    data_file = Path(f"data/family_FIXED/{args.family}_data_FIXED.npz")
    if not data_file.exists():
        print(f"\n❌ Fichier non trouvé: {data_file}")
        return 1

    data = np.load(data_file)
    images = data['images']
    hv_targets_v4 = data['hv_targets']
    fold_ids = data['fold_ids']
    image_ids = data['image_ids']

    # 2. Charger GT mask depuis PanNuke
    pannuke_dir = Path("/home/amar/data/PanNuke")
    fold_id = fold_ids[args.sample]
    img_id = image_ids[args.sample]

    masks = np.load(pannuke_dir / f"fold{fold_id}" / "masks.npy", mmap_mode='r')
    gt_mask = masks[img_id]

    # 3. Extraire instance map
    inst_map = extract_instances_from_mask(gt_mask)

    # 4. Calculer HV avec BBOX normalization (pour comparaison)
    hv_targets_bbox = np.zeros_like(hv_targets_v4[args.sample])

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = inst_map == inst_id
        hv_map_bbox, _, _ = compute_instance_hv_bbox(inst_mask)
        if hv_map_bbox is not None:
            hv_targets_bbox += hv_map_bbox

    # 5. Visualiser
    visualize_hv_vectors(
        images[args.sample],
        inst_map,
        hv_targets_v4[args.sample],
        hv_targets_bbox,
        args.sample,
        Path(args.output_dir)
    )

    print("\n"+"="*80)
    print("✅ DIAGNOSTIC TERMINÉ")
    print("="*80)
    print(f"\nVoir visualisation: {args.output_dir}/hv_vectors_sample_{args.sample:04d}.png")
    print("\n📝 INTERPRÉTATION:")
    print("  1. Si vecteurs HV pointent VERS les centres → Orientation OK")
    print("  2. Si vecteurs HV pointent AILLEURS → Swap ou Flip")
    print("  3. Si V4 range proche de 0 → Normalisation bbox nécessaire")
    print("  4. Si BBOX range proche de [-1,+1] → Normalisation correcte")


if __name__ == "__main__":
    exit(main())
