#!/usr/bin/env python3
"""
Test d'Alignement Bit-à-Bit - Validation Visuelle V13.

Génère une image de debug montrant les 5 crops d'une même image source
avec overlay des targets pour vérifier l'alignement parfait.

⚠️ CRITIQUE: Ce test DOIT passer avant de lancer l'extraction de features.

Usage:
    python scripts/validation/test_crop_alignment.py \
        --input_file data/family_V13/epidermal_data_v13_crops.npz \
        --n_samples 5 \
        --output_dir results/v13_validation
"""

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_overlay(
    image: np.ndarray,
    np_mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Crée un overlay de l'image avec le masque NP superposé.

    Args:
        image: Image RGB (H, W, 3) uint8 [0-255]
        np_mask: Nuclear Presence (H, W) float32 [0-1]
        alpha: Transparence de l'overlay

    Returns:
        Image avec overlay (H, W, 3) uint8
    """
    # Convertir masque en colormap (rouge pour les noyaux)
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = (np_mask > 0.5) * 255  # Rouge pour noyaux

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

    return overlay


def visualize_crops_from_same_source(
    crops_data: dict,
    source_image_id: int,
    output_path: Path
):
    """
    Visualise les 5 crops provenant d'une même image source.

    Layout:
    ┌─────────────────────────────────────┐
    │  Center  │  Top-Left  │  Top-Right  │
    ├──────────┼────────────┼─────────────┤
    │ Bottom-L │ Bottom-R   │   Legend    │
    └─────────────────────────────────────┘

    Args:
        crops_data: Données V13 chargées
        source_image_id: ID de l'image source
        output_path: Chemin de sauvegarde
    """
    # Trouver tous les crops de cette image source
    mask = crops_data['source_image_ids'] == source_image_id
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"⚠️  Aucun crop trouvé pour source_image_id={source_image_id}")
        return

    # Organiser par position
    crops_by_position = {}
    for idx in indices:
        pos = crops_data['crop_positions'][idx]
        crops_by_position[pos] = idx

    print(f"\n📸 Image source {source_image_id}: {len(crops_by_position)} crops trouvés")
    for pos, idx in crops_by_position.items():
        print(f"  - {pos:15s}: index {idx}")

    # Créer figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f'V13 Multi-Crop Alignment Check - Source Image ID: {source_image_id}',
        fontsize=16,
        fontweight='bold'
    )

    # Positions dans la grille
    grid_positions = {
        'center':       (0, 0),
        'top_left':     (0, 1),
        'top_right':    (0, 2),
        'bottom_left':  (1, 0),
        'bottom_right': (1, 1),
    }

    for pos_name, (row, col) in grid_positions.items():
        ax = axes[row, col]

        if pos_name in crops_by_position:
            idx = crops_by_position[pos_name]

            # Charger crop
            image = crops_data['images'][idx]
            np_mask = crops_data['np_targets'][idx]
            hv_map = crops_data['hv_targets'][idx]
            nt_map = crops_data['nt_targets'][idx]

            # Créer overlay
            overlay = create_overlay(image, np_mask, alpha=0.4)

            # Afficher
            ax.imshow(overlay)
            ax.set_title(
                f'{pos_name.replace("_", " ").title()}\n'
                f'Nuclei: {(np_mask > 0.5).sum()} px | '
                f'HV range: [{hv_map.min():.2f}, {hv_map.max():.2f}]',
                fontsize=12
            )
            ax.axis('off')

            # Calculer statistiques
            num_nuclei_px = (np_mask > 0.5).sum()
            hv_min, hv_max = hv_map.min(), hv_map.max()

            # Vérifications
            checks = []
            if num_nuclei_px > 0:
                checks.append('✅ NP OK')
            else:
                checks.append('❌ NP vide')

            if -1.0 <= hv_min <= hv_max <= 1.0:
                checks.append('✅ HV OK')
            else:
                checks.append('❌ HV range invalide')

            # Afficher checks
            check_text = ' | '.join(checks)
            ax.text(
                0.5, -0.05,
                check_text,
                transform=ax.transAxes,
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        else:
            ax.text(
                0.5, 0.5,
                f'{pos_name}\n(filtered - GT vide)',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=14,
                color='gray'
            )
            ax.axis('off')

    # Légende dans le dernier subplot
    ax_legend = axes[1, 2]
    ax_legend.axis('off')

    legend_text = """
    ╔═══════════════════════════════╗
    ║    VALIDATION CHECKLIST       ║
    ╠═══════════════════════════════╣
    ║ ✅ Alignement Bit-à-Bit       ║
    ║    → Bords des noyaux nets    ║
    ║    → Pas de décalage spatial  ║
    ║                               ║
    ║ ✅ HV Range [-1, 1]           ║
    ║    → Gradients valides        ║
    ║                               ║
    ║ ✅ Intégrité Biologique       ║
    ║    → Noyaux non déformés      ║
    ║    → Morphologie préservée    ║
    ║                               ║
    ║ 🔴 Rouge = Noyaux (NP mask)   ║
    ╚═══════════════════════════════╝

    SOURCE IMAGE ID: {source_id}
    TOTAL CROPS: {n_crops}/5
    FILTERED: {n_filtered}/5
    """.format(
        source_id=source_image_id,
        n_crops=len(crops_by_position),
        n_filtered=5 - len(crops_by_position)
    )

    ax_legend.text(
        0.1, 0.5,
        legend_text,
        transform=ax_legend.transAxes,
        fontsize=11,
        family='monospace',
        verticalalignment='center'
    )

    plt.tight_layout()

    # Sauvegarder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")

    plt.close()


def test_crop_alignment(
    input_file: Path,
    output_dir: Path,
    n_samples: int = 5
):
    """
    Test d'alignement sur plusieurs échantillons.

    Args:
        input_file: Fichier .npz V13 (crops générés)
        output_dir: Répertoire de sortie pour images de debug
        n_samples: Nombre d'images sources à visualiser
    """
    print(f"\n{'='*70}")
    print(f"TEST D'ALIGNEMENT BIT-À-BIT V13")
    print(f"{'='*70}\n")

    # Charger données
    print(f"📂 Chargement: {input_file}")
    data = np.load(input_file)

    crops_images = data['images']
    crops_np = data['np_targets']
    crops_hv = data['hv_targets']
    crops_nt = data['nt_targets']
    source_ids = data['source_image_ids']
    crop_positions = data['crop_positions']
    fold_ids = data['fold_ids']

    n_crops = len(crops_images)
    print(f"✅ {n_crops} crops chargés")

    # Validation shapes
    print(f"\n📏 Validation des shapes:")
    print(f"  images:       {crops_images.shape} (attendu: (N, 224, 224, 3))")
    print(f"  np_targets:   {crops_np.shape} (attendu: (N, 224, 224))")
    print(f"  hv_targets:   {crops_hv.shape} (attendu: (N, 2, 224, 224))")
    print(f"  nt_targets:   {crops_nt.shape} (attendu: (N, 224, 224))")

    assert crops_images.shape[1:] == (224, 224, 3), f"Shape images invalide"
    assert crops_np.shape[1:] == (224, 224), f"Shape NP invalide"
    assert crops_hv.shape[1:] == (2, 224, 224), f"Shape HV invalide"
    assert crops_nt.shape[1:] == (224, 224), f"Shape NT invalide"

    print(f"✅ Toutes les shapes sont correctes")

    # Validation ranges
    print(f"\n📊 Validation des ranges:")
    print(f"  images:     [{crops_images.min()}, {crops_images.max()}] (attendu: [0, 255])")
    print(f"  np_targets: [{crops_np.min():.3f}, {crops_np.max():.3f}] (attendu: [0, 1])")
    print(f"  hv_targets: [{crops_hv.min():.3f}, {crops_hv.max():.3f}] (attendu: [-1, 1])")
    print(f"  nt_targets: [{crops_nt.min()}, {crops_nt.max()}] (attendu: [0, 1] binary)")

    # Statistiques par position
    print(f"\n📊 Répartition par position de crop:")
    unique_positions, counts = np.unique(crop_positions, return_counts=True)
    for pos, count in zip(unique_positions, counts):
        pct = 100 * count / n_crops
        print(f"  {pos:15s}: {count:4d} ({pct:5.1f}%)")

    # Sélectionner échantillons
    unique_source_ids = np.unique(source_ids)
    n_samples = min(n_samples, len(unique_source_ids))

    print(f"\n🎯 Génération de {n_samples} images de debug...")

    # Sélectionner échantillons variés (début, milieu, fin)
    sample_indices = np.linspace(0, len(unique_source_ids) - 1, n_samples, dtype=int)
    sample_source_ids = unique_source_ids[sample_indices]

    crops_data = {
        'images': crops_images,
        'np_targets': crops_np,
        'hv_targets': crops_hv,
        'nt_targets': crops_nt,
        'source_image_ids': source_ids,
        'crop_positions': crop_positions,
    }

    for i, source_id in enumerate(sample_source_ids):
        output_path = output_dir / f"crop_alignment_check_source_{source_id:04d}.png"
        visualize_crops_from_same_source(crops_data, source_id, output_path)

    print(f"\n{'='*70}")
    print(f"✅ TEST COMPLÉTÉ - {n_samples} images de debug générées")
    print(f"{'='*70}")
    print(f"\n📁 Images sauvegardées dans: {output_dir}")
    print(f"\n⚠️  VALIDATION MANUELLE REQUISE:")
    print(f"   1. Ouvrir les images de debug")
    print(f"   2. Vérifier que les bords des noyaux sont nets (pas de décalage)")
    print(f"   3. Vérifier que les HV ranges sont dans [-1, 1]")
    print(f"   4. Vérifier que les noyaux ne sont pas déformés")
    print(f"\n   Si tout est OK → GO pour extraction de features")
    print(f"   Si problème → Investiguer prepare_family_data_v13_multi_crop.py\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test d'Alignement Bit-à-Bit V13 (Validation Visuelle)"
    )
    parser.add_argument(
        '--input_file',
        type=Path,
        required=True,
        help="Fichier .npz V13 (crops générés)"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results/v13_validation'),
        help="Répertoire de sortie pour images de debug"
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=5,
        help="Nombre d'images sources à visualiser"
    )

    args = parser.parse_args()

    # Validation fichier d'entrée
    if not args.input_file.exists():
        print(f"❌ ERREUR: Fichier introuvable: {args.input_file}")
        sys.exit(1)

    # Test
    test_crop_alignment(
        input_file=args.input_file,
        output_dir=args.output_dir,
        n_samples=args.n_samples
    )


if __name__ == '__main__':
    main()
