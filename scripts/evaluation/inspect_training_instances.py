#!/usr/bin/env python3
"""
Inspecte les instances dans les fichiers de données d'entraînement.

Objectif: Déterminer si les données utilisent:
- IDs natifs PanNuke (FIXED) → 50-100 instances/image
- connectedComponents (OLD) → 5-15 instances/image fusionnées

Usage:
    python scripts/evaluation/inspect_training_instances.py \
        --data_file data/cache/family_data/glandular_data.npz

Output:
    - Nombre moyen d'instances par image
    - Distribution des tailles d'instances
    - Visualisation de quelques inst_map
    - Verdict: FIXED ou OLD
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_instance_map(inst_map: np.ndarray) -> dict:
    """
    Analyse une instance map.

    Args:
        inst_map: (H, W) avec IDs d'instances [0, 1, 2, ...]

    Returns:
        {
            'n_instances': int,
            'instance_sizes': list,
            'mean_size': float,
            'largest_instance_ratio': float (ratio de la plus grande instance)
        }
    """
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]  # Exclude background

    n_instances = len(inst_ids)
    instance_sizes = []

    total_pixels = np.sum(inst_map > 0)

    for inst_id in inst_ids:
        size = np.sum(inst_map == inst_id)
        instance_sizes.append(size)

    largest_size = max(instance_sizes) if instance_sizes else 0
    largest_ratio = largest_size / total_pixels if total_pixels > 0 else 0

    return {
        'n_instances': n_instances,
        'instance_sizes': instance_sizes,
        'mean_size': np.mean(instance_sizes) if instance_sizes else 0,
        'largest_instance_ratio': largest_ratio,
    }


def inspect_training_data(data_file: Path, n_samples: int = 50) -> None:
    """Inspecte les données d'entraînement."""
    print(f"\n{'='*70}")
    print(f"INSPECTION DES DONNÉES D'ENTRAÎNEMENT")
    print(f"{'='*70}\n")

    print(f"📂 Fichier: {data_file}")

    if not data_file.exists():
        print(f"❌ ERREUR: Fichier introuvable!")
        return

    # Load data
    print(f"\n📥 Chargement...")
    data = np.load(data_file)

    print(f"   Clés disponibles: {list(data.keys())}")

    # Check for expected keys
    required_keys = ['images', 'np_masks', 'hv_targets', 'nt_targets']
    missing_keys = [key for key in required_keys if key not in data.keys()]

    if missing_keys:
        print(f"⚠️  Clés manquantes: {missing_keys}")
        print(f"   Le fichier n'a peut-être pas le format attendu.")

    # Get images and masks
    images = data['images']
    np_masks = data['np_masks']

    print(f"\n📊 Statistiques globales:")
    print(f"   Nombre d'images: {len(images)}")
    print(f"   Shape images: {images[0].shape}")
    print(f"   Shape NP masks: {np_masks[0].shape}")

    # Analyze instances
    print(f"\n🔍 Analyse des instances (sur {n_samples} échantillons)...")

    n_instances_list = []
    mean_sizes_list = []
    largest_ratios_list = []

    for i in range(min(n_samples, len(np_masks))):
        np_mask = np_masks[i]

        # Create instance map from NP mask (simulate what training does)
        # This is the KEY test: does the data already have instances or just binary masks?

        # Option 1: Data already has instance IDs (FIXED)
        # In this case, np_mask would have values [0, 1, 2, 3, ...] for different instances

        # Option 2: Data has binary mask only (OLD)
        # In this case, np_mask would have values [0, 1] only

        unique_values = np.unique(np_mask)
        max_value = np_mask.max()

        if max_value <= 1:
            # Binary mask only - need to check if data has separate inst_map
            if 'inst_maps' in data.keys():
                inst_map = data['inst_maps'][i]
            else:
                print(f"⚠️  Image {i}: NP mask est binaire (max={max_value})")
                print(f"   Pas de inst_map trouvé - impossible d'analyser les instances")
                continue
        else:
            # Mask has instance IDs
            inst_map = np_mask

        stats = analyze_instance_map(inst_map)

        n_instances_list.append(stats['n_instances'])
        mean_sizes_list.append(stats['mean_size'])
        largest_ratios_list.append(stats['largest_instance_ratio'])

    # Compute statistics
    if not n_instances_list:
        print(f"❌ Impossible d'analyser les instances!")
        return

    mean_n_instances = np.mean(n_instances_list)
    std_n_instances = np.std(n_instances_list)
    mean_largest_ratio = np.mean(largest_ratios_list)

    print(f"\n📈 Résultats:")
    print(f"   Instances par image: {mean_n_instances:.1f} ± {std_n_instances:.1f}")
    print(f"   Taille moyenne instance: {np.mean(mean_sizes_list):.1f} pixels")
    print(f"   Ratio plus grande instance: {mean_largest_ratio:.2%}")

    # Verdict
    print(f"\n🎯 VERDICT:")

    if mean_n_instances > 40:
        print(f"   ✅ FIXED DATA (vraies instances PanNuke)")
        print(f"      → {mean_n_instances:.0f} instances/image est cohérent avec IDs natifs")
        print(f"      → Ratio max instance {mean_largest_ratio:.1%} indique bonne séparation")
    elif mean_n_instances < 20:
        print(f"   ❌ OLD DATA (connectedComponents)")
        print(f"      → {mean_n_instances:.0f} instances/image trop faible (fusionnées)")
        print(f"      → Ratio max instance {mean_largest_ratio:.1%} indique fusion")
        print(f"\n   ⚠️  PROBLÈME IDENTIFIÉ:")
        print(f"      Les cellules qui se touchent sont fusionnées en 1 instance géante")
        print(f"      → HV targets ont des gradients FAIBLES (pas de frontières réelles)")
        print(f"      → Watershed ne peut pas séparer les instances")
        print(f"\n   💡 SOLUTION:")
        print(f"      Régénérer les données avec prepare_family_data_FIXED.py")
    else:
        print(f"   ⚠️  INCERTAIN")
        print(f"      → {mean_n_instances:.0f} instances/image est dans la zone grise")
        print(f"      → Inspection visuelle nécessaire")

    # Visualize a few examples
    print(f"\n🖼️  Génération visualisations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Exemples Instance Maps ({data_file.stem})', fontsize=16)

    for idx, ax in enumerate(axes.flat):
        if idx >= min(6, len(images)):
            break

        if 'inst_maps' in data.keys():
            inst_map = data['inst_maps'][idx]
        elif np_masks[idx].max() > 1:
            inst_map = np_masks[idx]
        else:
            ax.text(0.5, 0.5, 'No instance map', ha='center', va='center')
            ax.axis('off')
            continue

        # Colorize instance map
        colored = np.zeros((*inst_map.shape, 3), dtype=np.uint8)
        inst_ids = np.unique(inst_map)
        inst_ids = inst_ids[inst_ids > 0]

        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(inst_ids), 3))

        for i, inst_id in enumerate(inst_ids):
            mask = inst_map == inst_id
            colored[mask] = colors[i]

        ax.imshow(colored)
        ax.set_title(f'Image {idx}: {len(inst_ids)} instances')
        ax.axis('off')

    plt.tight_layout()

    output_path = data_file.parent / f"{data_file.stem}_inspection.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Sauvegardé: {output_path}")

    plt.close()

    # Distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of instance counts
    axes[0].hist(n_instances_list, bins=20, edgecolor='black')
    axes[0].axvline(mean_n_instances, color='red', linestyle='--',
                     label=f'Moyenne: {mean_n_instances:.1f}')
    axes[0].set_xlabel('Nombre d\'instances par image')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution du nombre d\'instances')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of largest instance ratios
    axes[1].hist(largest_ratios_list, bins=20, edgecolor='black')
    axes[1].axvline(mean_largest_ratio, color='red', linestyle='--',
                     label=f'Moyenne: {mean_largest_ratio:.1%}')
    axes[1].set_xlabel('Ratio de la plus grande instance')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution des ratios max instance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = data_file.parent / f"{data_file.stem}_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Sauvegardé: {output_path}")

    print(f"\n✅ Inspection terminée!")


def main():
    parser = argparse.ArgumentParser(
        description="Inspecte les instances dans les données d'entraînement"
    )
    parser.add_argument(
        '--data_file',
        type=Path,
        required=True,
        help='Fichier .npz de données d\'entraînement'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=50,
        help='Nombre d\'échantillons à analyser (default: 50)'
    )

    args = parser.parse_args()

    inspect_training_data(args.data_file, args.n_samples)


if __name__ == '__main__':
    main()
