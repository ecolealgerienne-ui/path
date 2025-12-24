#!/usr/bin/env python3
"""
Sanity Check PanNuke Raw - Vérification Alignement Source

Vérifie que images[0] et masks[0] dans les fichiers .npy sources
sont PARFAITEMENT alignés AVANT tout preprocessing.

Si ce test échoue → Dataset PanNuke corrompu à la source
Si ce test passe → Bug dans prepare_family_data_FIXED_v2.py

Usage:
    python scripts/validation/sanity_check_pannuke_raw.py \
        --fold 0 --indices 0 1 2 512
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def visualize_raw_alignment(image, mask, idx, output_dir):
    """
    Visualise l'alignement RAW entre image et mask (AVANT preprocessing).

    Args:
        image: (256, 256, 3) float64 [0, 255] ou [0, 1]
        mask: (256, 256, 6) ou (6, 256, 256) - canaux PanNuke
        idx: Index de l'échantillon
        output_dir: Répertoire de sortie
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Détecter format mask
    if mask.shape == (6, 256, 256):
        print(f"   ⚠️  Index {idx}: Mask en format CHW (6, 256, 256) - TRANSPOSING")
        mask = mask.transpose(1, 2, 0)  # CHW → HWC
        format_str = "CHW→HWC"
    elif mask.shape == (256, 256, 6):
        print(f"   ✅ Index {idx}: Mask en format HWC (256, 256, 6) - OK")
        format_str = "HWC"
    else:
        raise ValueError(f"Format mask inattendu: {mask.shape}")

    # Normaliser image pour affichage
    if image.max() > 1.0:
        image_display = (image / 255.0).clip(0, 1)
    else:
        image_display = image.clip(0, 1)

    # Union des masques (canaux 1-5, exclure background)
    mask_union = mask[:, :, 1:].sum(axis=-1) > 0

    # Calculer overlap (comme dans test_pannuke_sources.py)
    # Assumer que les régions roses/violettes dans l'image sont du tissu
    image_gray = cv2.cvtColor((image_display * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, tissue_binary = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)
    tissue_mask = tissue_binary > 0

    overlap = np.logical_and(mask_union, tissue_mask).sum()
    mask_area = mask_union.sum()
    overlap_ratio = (overlap / mask_area * 100) if mask_area > 0 else 0

    # Trouver contours
    contours, _ = cv2.findContours(
        mask_union.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Image, Mask, Superposition
    axes[0, 0].imshow(image_display)
    axes[0, 0].set_title(f"Image RAW (Index {idx})")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask_union, cmap='gray')
    axes[0, 1].set_title(f"Mask Union (Canaux 1-5)\nFormat: {format_str}")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(image_display)
    for cnt in contours:
        axes[0, 2].plot(cnt[:, 0, 0], cnt[:, 0, 1], 'g-', linewidth=2)
    axes[0, 2].set_title(f"Superposition\nOverlap: {overlap_ratio:.1f}%")
    axes[0, 2].axis('off')

    # Row 2: Détail par canal
    channel_names = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']

    # Afficher canaux 1, 2, 5 (Neoplastic, Inflammatory, Epithelial)
    for i, c in enumerate([1, 2, 5]):
        channel_mask = mask[:, :, c]
        axes[1, i].imshow(channel_mask, cmap='viridis')
        axes[1, i].set_title(f"Canal {c}: {channel_names[c]}\nMax ID: {int(channel_mask.max())}")
        axes[1, i].axis('off')

    plt.suptitle(
        f"Sanity Check RAW - Index {idx}\n"
        f"Image shape: {image.shape}, Mask shape (original): {mask.shape}\n"
        f"Overlap: {overlap_ratio:.1f}% {'✅ ALIGNÉ' if overlap_ratio > 80 else '❌ DÉSALIGNÉ'}",
        fontsize=14, fontweight='bold'
    )

    output_file = output_dir / f"sanity_check_raw_idx{idx:04d}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"      💾 Visualisation: {output_file}")

    return overlap_ratio


def main():
    parser = argparse.ArgumentParser(description="Sanity Check PanNuke Raw Sources")
    parser.add_argument("--data_dir", type=Path, default=Path("/home/amar/data/PanNuke"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--indices", type=int, nargs='+', default=[0, 1, 2, 512],
                        help="Indices à tester")
    parser.add_argument("--output_dir", type=Path, default=Path("results/sanity_check_raw"))

    args = parser.parse_args()

    print("=" * 80)
    print("SANITY CHECK PANNUKE RAW (AVANT TOUT PREPROCESSING)")
    print("=" * 80)
    print(f"Fold: {args.fold}")
    print(f"Indices à tester: {args.indices}")
    print()

    # Charger fichiers sources
    fold_dir = args.data_dir / f"fold{args.fold}"
    images_path = fold_dir / "images.npy"
    masks_path = fold_dir / "masks.npy"
    types_path = fold_dir / "types.npy"

    if not images_path.exists() or not masks_path.exists():
        print(f"❌ ERREUR: Fichiers manquants dans {fold_dir}")
        return 1

    print(f"✅ Fichiers trouvés:")
    print(f"   Images: {images_path}")
    print(f"   Masks:  {masks_path}")
    print(f"   Types:  {types_path}")
    print()

    # Charger avec mmap (économie RAM)
    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')
    types = np.load(types_path) if types_path.exists() else None

    print(f"✅ Données chargées (mmap):")
    print(f"   Images shape: {images.shape}, dtype: {images.dtype}")
    print(f"   Masks shape:  {masks.shape}, dtype: {masks.dtype}")
    if types is not None:
        print(f"   Types shape:  {types.shape}, dtype: {types.dtype}")
    print()

    # Vérifier format global
    if masks.shape[1] == 6:
        print("⚠️  WARNING: Masks en format CHW (B, 6, H, W)")
        print("   → Nécessitera transposition dans preprocessing")
        mask_format_global = "CHW"
    elif masks.shape[3] == 6:
        print("✅ Masks en format HWC (B, H, W, 6) - CORRECT")
        mask_format_global = "HWC"
    else:
        print(f"❌ ERREUR: Format masks inconnu: {masks.shape}")
        return 1
    print()

    # Tester chaque index
    print("Tests d'alignement RAW:")
    print("-" * 40)

    results = []
    for idx in args.indices:
        if idx >= len(images):
            print(f"  ⚠️  Index {idx} hors limites (max: {len(images)-1}), skipping")
            continue

        organ = types[idx].decode('utf-8') if types is not None and isinstance(types[idx], bytes) else "Unknown"
        print(f"  Testing index {idx} (Organ: {organ})...")

        image = np.array(images[idx])
        mask = np.array(masks[idx])

        overlap = visualize_raw_alignment(image, mask, idx, args.output_dir)
        results.append((idx, organ, overlap))

    print()
    print("=" * 80)
    print("RÉSUMÉ SANITY CHECK")
    print("=" * 80)
    print(f"Format masques: {mask_format_global}")
    print(f"Indices testés: {len(results)}")
    print()

    all_aligned = all(overlap > 80 for _, _, overlap in results)

    for idx, organ, overlap in results:
        status = "✅ ALIGNÉ" if overlap > 80 else "❌ DÉSALIGNÉ"
        print(f"  Index {idx:4d} ({organ:20s}): overlap={overlap:5.1f}% {status}")

    print()
    if all_aligned:
        print("✅ VERDICT: TOUS LES INDICES SONT ALIGNÉS")
        print()
        print("   → Les fichiers sources PanNuke RAW sont SAINS")
        print("   → Le bug vient de prepare_family_data_FIXED_v2.py")
        print("   → Action: Appliquer le fix de l'expert (Transposition Fantôme)")
        return 0
    else:
        print("❌ VERDICT: DÉSALIGNEMENT DÉTECTÉ DANS LES SOURCES")
        print()
        print("   → Le dataset PanNuke est CORROMPU à la source")
        print("   → Action: Re-télécharger PanNuke officiel")
        print("   → URL: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke")
        return 1


if __name__ == "__main__":
    exit(main())
