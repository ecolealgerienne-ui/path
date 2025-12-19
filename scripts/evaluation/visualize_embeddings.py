#!/usr/bin/env python3
"""
Visualisation des embeddings H-optimus-0 avec t-SNE.

Ce script charge les features extraites et les visualise en 2D
pour voir si les différents types de tissus se regroupent.

Usage:
    python scripts/evaluation/visualize_embeddings.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import argparse


def load_features(cache_dir: Path, fold: int = 1):
    """Charge les features et types depuis le cache."""
    features_path = cache_dir / f"pannuke_fold{fold}_features.npy"
    types_path = cache_dir / f"pannuke_fold{fold}_types.npy"

    features = np.load(features_path)
    types = np.load(types_path)

    print(f"Features chargées: {features.shape}")
    print(f"Types chargés: {types.shape}")
    print(f"Types uniques: {np.unique(types)}")

    return features, types


def compute_tsne(features: np.ndarray, perplexity: int = 30, random_state: int = 42):
    """Calcule la projection t-SNE."""
    print(f"\nCalcul t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=1
    )
    embeddings_2d = tsne.fit_transform(features)
    print(f"Projection terminée: {embeddings_2d.shape}")
    return embeddings_2d


def plot_embeddings(embeddings_2d: np.ndarray, types: np.ndarray, output_path: Path):
    """Crée le graphique des embeddings colorés par type."""

    unique_types = np.unique(types)
    n_types = len(unique_types)

    # Palette de couleurs
    cmap = plt.cm.get_cmap('tab20', n_types)

    fig, ax = plt.subplots(figsize=(14, 10))

    for i, organ_type in enumerate(unique_types):
        mask = types == organ_type
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(i)],
            label=organ_type,
            alpha=0.7,
            s=50
        )

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('Embeddings H-optimus-0 — PanNuke (colorés par organe)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: {output_path}")

    # Afficher aussi
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualisation des embeddings")
    parser.add_argument("--cache_dir", type=str, default="data/cache",
                        help="Répertoire des features")
    parser.add_argument("--output", type=str, default="data/cache/tsne_embeddings.png",
                        help="Chemin de sortie pour le graphique")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="Perplexité pour t-SNE")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold PanNuke")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)

    # Charger les features
    features, types = load_features(cache_dir, fold=args.fold)

    # Calculer t-SNE
    embeddings_2d = compute_tsne(features, perplexity=args.perplexity)

    # Visualiser
    plot_embeddings(embeddings_2d, types, output_path)

    print("\n✓ Visualisation terminée!")


if __name__ == "__main__":
    main()
