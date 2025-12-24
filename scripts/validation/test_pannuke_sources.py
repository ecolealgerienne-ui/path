#!/usr/bin/env python3
"""
Test de Sanité des Sources PanNuke

Vérifie que images.npy, masks.npy et types.npy sont alignés correctement.

Usage:
    python scripts/validation/test_pannuke_sources.py \
        --fold 0 \
        --indices 0 10 512 \
        --output_dir results/pannuke_source_check
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def test_source_alignment(fold, indices, output_dir):
    """
    Teste l'alignement des sources PanNuke pour des indices donnés.

    Args:
        fold: Numéro du fold (0, 1 ou 2)
        indices: Liste d'indices à tester
        output_dir: Répertoire de sortie pour visualisations
    """
    print("="*80)
    print("TEST DE SANITÉ DES SOURCES PANNUKE")
    print("="*80)
    print(f"Fold: {fold}")
    print(f"Indices à tester: {indices}")
    print()

    # Chemins des fichiers sources
    base_path = Path(f"/home/amar/data/PanNuke/fold{fold}")

    images_path = base_path / "images.npy"
    masks_path = base_path / "masks.npy"
    types_path = base_path / "types.npy"

    # Vérifier existence
    for path in [images_path, masks_path, types_path]:
        if not path.exists():
            print(f"❌ ERREUR: Fichier manquant: {path}")
            sys.exit(1)

    print("✅ Fichiers sources trouvés:")
    print(f"   Images: {images_path}")
    print(f"   Masks:  {masks_path}")
    print(f"   Types:  {types_path}")
    print()

    # Charger avec mmap (économie RAM)
    print("Chargement des données (mmap)...")
    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')
    types = np.load(types_path)

    print(f"✅ Données chargées:")
    print(f"   Images shape: {images.shape}, dtype: {images.dtype}")
    print(f"   Masks shape:  {masks.shape}, dtype: {masks.dtype}")
    print(f"   Types shape:  {types.shape}, dtype: {types.dtype}")
    print()

    # ⚠️ DIAGNOSTIC CRITIQUE: Format HWC vs CHW
    if masks.ndim == 4:
        if masks.shape[1] == 6:
            print("⚠️ WARNING: Masks en format CHW (B, 6, H, W)")
            print("   → Conversion requise: mask = np.transpose(mask, (0, 2, 3, 1))")
            format_mask = "CHW"
        elif masks.shape[3] == 6:
            print("✅ Masks en format HWC (B, H, W, 6) - CORRECT")
            format_mask = "HWC"
        else:
            print(f"❌ ERREUR: Format masks inconnu: {masks.shape}")
            sys.exit(1)
    else:
        print(f"❌ ERREUR: Masks devrait être 4D, obtenu {masks.ndim}D")
        sys.exit(1)

    print()

    # Créer répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Tester chaque indice
    print("Tests d'alignement:")
    print("-" * 40)

    all_aligned = True

    for idx in indices:
        if idx >= len(images):
            print(f"  ⚠️ Index {idx} hors limites (max: {len(images)-1})")
            continue

        # Charger image et mask
        img = np.array(images[idx])
        mask = np.array(masks[idx])
        organ = types[idx].decode('utf-8') if hasattr(types[idx], 'decode') else str(types[idx])

        # Convertir mask si nécessaire
        if format_mask == "CHW":
            mask = np.transpose(mask, (1, 2, 0))  # (6, H, W) → (H, W, 6)

        # Calculer masque global (union de tous les canaux sauf background)
        mask_global = mask[:, :, 1:].sum(axis=-1) > 0  # Canaux 1-5 (pas 0 = background)

        # Vérifier alignement visuel
        # Critère: Au moins 50% des pixels du masque coïncident avec des pixels tissulaires dans l'image
        # (Image tissulaire = pixels pas complètement blancs)
        img_gray = img.mean(axis=-1)
        img_tissue = img_gray < 240  # Seuil pour détecter tissu vs background blanc

        overlap = (mask_global & img_tissue).sum()
        mask_area = mask_global.sum()

        overlap_ratio = overlap / mask_area if mask_area > 0 else 0

        aligned = overlap_ratio > 0.5
        icon = "✅" if aligned else "❌"

        print(f"  {icon} Index {idx:4d} ({organ:20s}): overlap={overlap_ratio:.1%}", end="")

        if not aligned:
            print(" ← DÉSALIGNÉ")
            all_aligned = False
        else:
            print(" ← OK")

        # Générer visualisation
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Subplot 1: Image originale
        axes[0].imshow(img)
        axes[0].set_title(f"Image (Index {idx})\nOrgane: {organ}")
        axes[0].axis('off')

        # Subplot 2: Masque global
        axes[1].imshow(mask_global, cmap='gray')
        axes[1].set_title(f"Masque Global (Canaux 1-5)\n{mask_area} pixels")
        axes[1].axis('off')

        # Subplot 3: Superposition
        axes[2].imshow(img)
        axes[2].contour(mask_global, colors='lime', linewidths=2, levels=[0.5])
        axes[2].set_title(f"Image + Contours\nOverlap: {overlap_ratio:.1%}")
        axes[2].axis('off')

        # Ajouter verdict
        verdict_text = "✅ ALIGNÉ" if aligned else "❌ DÉSALIGNÉ"
        fig.suptitle(f"Test Source PanNuke - {verdict_text}", fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_file = output_path / f"source_test_idx{idx:04d}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"     💾 Visualisation: {output_file}")

    print()
    print("="*80)
    print("RÉSUMÉ")
    print("="*80)
    print(f"Format masques: {format_mask}")
    print(f"Indices testés: {len(indices)}")

    if all_aligned:
        print("✅ VERDICT: TOUS LES INDICES SONT ALIGNÉS")
        print("\n   → Les fichiers sources PanNuke sont SAINS")
        print("   → Le problème vient de prepare_family_data_FIXED.py")
        print("   → Action: Débugger le script de préparation")
        return 0
    else:
        print("❌ VERDICT: DÉSALIGNEMENT DÉTECTÉ")
        print("\n   → Les fichiers sources PanNuke sont CORROMPUS")
        print("   → Action: Re-télécharger PanNuke depuis la source officielle")
        print("   → URL: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Test de sanité des sources PanNuke")
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2],
                        help="Numéro du fold à tester")
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 10, 100, 512],
                        help="Indices à tester (ex: 0 10 512)")
    parser.add_argument('--output_dir', type=str, default="results/pannuke_source_check",
                        help="Répertoire de sortie pour visualisations")

    args = parser.parse_args()

    try:
        exit_code = test_source_alignment(args.fold, args.indices, args.output_dir)
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
