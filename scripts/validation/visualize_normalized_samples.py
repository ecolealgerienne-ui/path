#!/usr/bin/env python3
"""
Visualisation des échantillons normalisés Macenko.

Objectif: Vérifier que la normalisation n'a pas créé de pixels saturés
ou détruit les détails nucléaires (notamment après overflow warning).

Usage:
    python scripts/validation/visualize_normalized_samples.py --family digestive --n_samples 5
    python scripts/validation/visualize_normalized_samples.py --family digestive --indices 102 103 104 105
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def compute_image_statistics(image: np.ndarray) -> dict:
    """
    Calcule les statistiques détaillées d'une image.

    Args:
        image: Image RGB (H, W, 3) uint8

    Returns:
        Dictionnaire de statistiques
    """
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': int(image.min()),
        'max': int(image.max()),
        'mean': float(image.mean()),
        'std': float(image.std()),
    }

    # Pixels saturés (potentiellement corrompus par overflow)
    saturated_black = np.sum(image == 0)
    saturated_white = np.sum(image == 255)
    total_pixels = image.size

    stats['saturated_black'] = int(saturated_black)
    stats['saturated_white'] = int(saturated_white)
    stats['saturated_black_pct'] = float(saturated_black / total_pixels * 100)
    stats['saturated_white_pct'] = float(saturated_white / total_pixels * 100)

    # Statistiques par canal RGB
    for i, channel in enumerate(['R', 'G', 'B']):
        ch = image[:, :, i]
        stats[f'{channel}_min'] = int(ch.min())
        stats[f'{channel}_max'] = int(ch.max())
        stats[f'{channel}_mean'] = float(ch.mean())
        stats[f'{channel}_std'] = float(ch.std())

    # Intensité moyenne (approximation luminance)
    intensity = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    stats['intensity_mean'] = float(intensity.mean())
    stats['intensity_std'] = float(intensity.std())

    return stats


def detect_artifacts(image: np.ndarray) -> dict:
    """
    Détecte les artefacts potentiels causés par l'overflow.

    Args:
        image: Image RGB (H, W, 3) uint8

    Returns:
        Dictionnaire des artefacts détectés
    """
    artifacts = {
        'has_issues': False,
        'issues': []
    }

    # Seuils d'alerte
    SATURATED_THRESHOLD = 5.0  # % de pixels saturés acceptable

    # Vérifier pixels noirs (0, 0, 0)
    black_pixels = np.all(image == 0, axis=-1)
    black_pct = np.sum(black_pixels) / (image.shape[0] * image.shape[1]) * 100
    if black_pct > SATURATED_THRESHOLD:
        artifacts['has_issues'] = True
        artifacts['issues'].append(f"⚠️ {black_pct:.2f}% pixels noirs purs (overflow négatif?)")

    # Vérifier pixels blancs (255, 255, 255)
    white_pixels = np.all(image == 255, axis=-1)
    white_pct = np.sum(white_pixels) / (image.shape[0] * image.shape[1]) * 100
    if white_pct > SATURATED_THRESHOLD:
        artifacts['has_issues'] = True
        artifacts['issues'].append(f"⚠️ {white_pct:.2f}% pixels blancs purs (overflow positif?)")

    # Vérifier si l'image est trop sombre (perte de contraste)
    if image.mean() < 50:
        artifacts['has_issues'] = True
        artifacts['issues'].append(f"⚠️ Image très sombre (mean={image.mean():.1f})")

    # Vérifier si l'image est trop claire
    if image.mean() > 230:
        artifacts['has_issues'] = True
        artifacts['issues'].append(f"⚠️ Image très claire (mean={image.mean():.1f})")

    # Vérifier la variance (image "plate")
    if image.std() < 10:
        artifacts['has_issues'] = True
        artifacts['issues'].append(f"⚠️ Faible contraste (std={image.std():.1f})")

    if not artifacts['has_issues']:
        artifacts['issues'].append("✅ Aucun artefact détecté")

    return artifacts


def visualize_samples(
    data_path: Path,
    indices: list,
    output_dir: Path,
    family: str
) -> dict:
    """
    Visualise les échantillons spécifiés et calcule leurs statistiques.

    Args:
        data_path: Chemin vers le fichier .npz
        indices: Liste des indices à visualiser
        output_dir: Répertoire de sortie pour les images
        family: Nom de la famille

    Returns:
        Rapport complet des statistiques
    """
    print(f"\n{'='*60}")
    print(f"VISUALISATION ÉCHANTILLONS NORMALISÉS - {family.upper()}")
    print(f"{'='*60}")

    # Charger les données
    print(f"\nChargement: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    images = data['images']
    n_total = len(images)
    print(f"Total images: {n_total}")
    print(f"Shape: {images.shape}")
    print(f"Dtype: {images.dtype}")

    # Valider les indices
    valid_indices = [i for i in indices if 0 <= i < n_total]
    if len(valid_indices) < len(indices):
        invalid = [i for i in indices if i not in valid_indices]
        print(f"⚠️ Indices invalides ignorés: {invalid}")

    if not valid_indices:
        print("❌ Aucun indice valide à visualiser")
        return {}

    print(f"\nIndices à visualiser: {valid_indices}")

    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rapport global
    report = {
        'family': family,
        'data_path': str(data_path),
        'n_total': n_total,
        'n_visualized': len(valid_indices),
        'samples': {}
    }

    # Statistiques globales
    print(f"\n{'='*60}")
    print("STATISTIQUES GLOBALES")
    print(f"{'='*60}")

    global_stats = compute_image_statistics(images)
    report['global_stats'] = global_stats

    print(f"  Min:  {global_stats['min']}")
    print(f"  Max:  {global_stats['max']}")
    print(f"  Mean: {global_stats['mean']:.2f}")
    print(f"  Std:  {global_stats['std']:.2f}")
    print(f"  Pixels noirs saturés: {global_stats['saturated_black_pct']:.4f}%")
    print(f"  Pixels blancs saturés: {global_stats['saturated_white_pct']:.4f}%")

    # Visualiser chaque échantillon
    n_samples = len(valid_indices)
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))

    if n_samples == 1:
        axes = axes.reshape(2, 1)

    print(f"\n{'='*60}")
    print("STATISTIQUES PAR ÉCHANTILLON")
    print(f"{'='*60}")

    issues_detected = []

    for col, idx in enumerate(valid_indices):
        img = images[idx]

        # Statistiques
        stats = compute_image_statistics(img)
        artifacts = detect_artifacts(img)

        report['samples'][idx] = {
            'stats': stats,
            'artifacts': artifacts
        }

        print(f"\n--- Image {idx} ---")
        print(f"  Shape: {stats['shape']}")
        print(f"  Range: [{stats['min']}, {stats['max']}]")
        print(f"  Mean:  {stats['mean']:.2f}")
        print(f"  Std:   {stats['std']:.2f}")
        print(f"  Intensity Mean: {stats['intensity_mean']:.2f}")
        print(f"  Intensity Std:  {stats['intensity_std']:.2f}")
        print(f"  Canaux RGB:")
        print(f"    R: mean={stats['R_mean']:.1f}, std={stats['R_std']:.1f}, range=[{stats['R_min']}, {stats['R_max']}]")
        print(f"    G: mean={stats['G_mean']:.1f}, std={stats['G_std']:.1f}, range=[{stats['G_min']}, {stats['G_max']}]")
        print(f"    B: mean={stats['B_mean']:.1f}, std={stats['B_std']:.1f}, range=[{stats['B_min']}, {stats['B_max']}]")
        print(f"  Pixels saturés:")
        print(f"    Noirs (0):   {stats['saturated_black']} ({stats['saturated_black_pct']:.4f}%)")
        print(f"    Blancs (255): {stats['saturated_white']} ({stats['saturated_white_pct']:.4f}%)")
        print(f"  Artefacts:")
        for issue in artifacts['issues']:
            print(f"    {issue}")

        if artifacts['has_issues']:
            issues_detected.append(idx)

        # Afficher l'image
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"Image {idx}\nmean={stats['mean']:.1f}, std={stats['std']:.1f}")
        axes[0, col].axis('off')

        # Histogramme
        for i, (color, label) in enumerate(zip(['red', 'green', 'blue'], ['R', 'G', 'B'])):
            axes[1, col].hist(img[:, :, i].flatten(), bins=50, color=color, alpha=0.5, label=label)
        axes[1, col].set_xlabel('Intensité pixel')
        axes[1, col].set_ylabel('Fréquence')
        axes[1, col].set_title(f"Histogramme RGB")
        axes[1, col].legend()
        axes[1, col].set_xlim(0, 255)

    plt.suptitle(f"Famille {family.upper()} - Échantillons normalisés Macenko", fontsize=14)
    plt.tight_layout()

    # Sauvegarder
    output_file = output_dir / f"{family}_normalized_samples.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualisation sauvegardée: {output_file}")
    plt.close()

    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")

    if issues_detected:
        print(f"⚠️ ATTENTION: Artefacts détectés sur {len(issues_detected)} image(s): {issues_detected}")
        report['verdict'] = 'WARNING'
        report['issues_indices'] = issues_detected
    else:
        print("✅ Aucun artefact détecté sur les échantillons vérifiés")
        report['verdict'] = 'OK'
        report['issues_indices'] = []

    # Sauvegarder le rapport JSON
    import json
    report_file = output_dir / f"{family}_normalized_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Rapport JSON sauvegardé: {report_file}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Visualise les échantillons normalisés Macenko"
    )
    parser.add_argument(
        '--family',
        type=str,
        required=True,
        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
        help="Famille à visualiser"
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=5,
        help="Nombre d'échantillons aléatoires à visualiser (défaut: 5)"
    )
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        default=None,
        help="Indices spécifiques à visualiser (ex: --indices 102 103 104 105)"
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=Path('data/family_FIXED'),
        help="Répertoire des données normalisées"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results/normalization_check'),
        help="Répertoire de sortie"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Seed pour échantillonnage aléatoire"
    )

    args = parser.parse_args()

    # Chemin du fichier de données
    data_path = args.data_dir / f"{args.family}_data_FIXED.npz"

    if not data_path.exists():
        print(f"❌ Fichier non trouvé: {data_path}")
        sys.exit(1)

    # Déterminer les indices à visualiser
    if args.indices:
        indices = args.indices
    else:
        # Charger pour obtenir le nombre total
        data = np.load(data_path, allow_pickle=True)
        n_total = len(data['images'])

        # Sélection aléatoire
        np.random.seed(args.seed)
        indices = np.random.choice(n_total, min(args.n_samples, n_total), replace=False).tolist()

        # Si on vérifie Digestive, inclure les indices autour de 102-105 (zone overflow)
        if args.family == 'digestive':
            overflow_indices = [102, 103, 104, 105]
            valid_overflow = [i for i in overflow_indices if i < n_total]
            indices = list(set(indices + valid_overflow))[:args.n_samples + len(valid_overflow)]

    # Visualiser
    report = visualize_samples(
        data_path=data_path,
        indices=sorted(indices),
        output_dir=args.output_dir,
        family=args.family
    )

    # Code de retour basé sur le verdict
    if report.get('verdict') == 'WARNING':
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
