#!/usr/bin/env python3
"""
Test du pipeline WSI avec une lame réelle (.svs, .ndpi, etc.)

Usage:
    python scripts/test_wsi_pipeline.py --input data/wsi_test/CMU-1.svs --max_tiles 10

Ce script:
1. Lit les métadonnées de la lame via OpenSlide
2. Utilise l'InputRouter pour extraire des tiles 224×224
3. Affiche les statistiques d'extraction
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ajouter le chemin racine
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Test WSI pipeline")
    parser.add_argument("--input", type=str, required=True,
                        help="Chemin vers la lame WSI (.svs, .ndpi, etc.)")
    parser.add_argument("--max_tiles", type=int, default=10,
                        help="Nombre maximum de tiles à extraire (défaut: 10)")
    parser.add_argument("--save_tiles", action="store_true",
                        help="Sauvegarder les tiles extraits")
    parser.add_argument("--output_dir", type=str, default="data/wsi_test/tiles",
                        help="Dossier de sortie pour les tiles")
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ Fichier non trouvé: {input_path}")
        sys.exit(1)

    # ===========================================================================
    # 1. Métadonnées OpenSlide
    # ===========================================================================
    print("=" * 60)
    print("1. MÉTADONNÉES OPENSLIDE")
    print("=" * 60)

    try:
        import openslide
        slide = openslide.OpenSlide(str(input_path))

        print(f"Fichier: {input_path.name}")
        print(f"Taille: {input_path.stat().st_size / 1e6:.1f} MB")
        print(f"Dimensions (W×H): {slide.dimensions[0]:,} × {slide.dimensions[1]:,} pixels")
        print(f"Niveaux: {slide.level_count}")
        print(f"Dimensions par niveau: {slide.level_dimensions}")
        print(f"Downsamples: {[round(d, 2) for d in slide.level_downsamples]}")

        mpp_x = slide.properties.get('openslide.mpp-x')
        mpp_y = slide.properties.get('openslide.mpp-y')
        print(f"MPP (microns/pixel): x={mpp_x}, y={mpp_y}")

        vendor = slide.properties.get('openslide.vendor', 'unknown')
        print(f"Vendor: {vendor}")

        objective = slide.properties.get('openslide.objective-power')
        print(f"Objectif: {objective}x" if objective else "Objectif: N/A")

        # Estimation tiles
        w, h = slide.dimensions
        tiles_x = w // 224
        tiles_y = h // 224
        total_tiles = tiles_x * tiles_y
        print(f"\nEstimation tiles 224×224 (niveau 0):")
        print(f"  Grille: {tiles_x} × {tiles_y} = {total_tiles:,} tiles")
        print(f"  Avec ~30% tissu: ~{int(total_tiles * 0.3):,} tiles utiles")

        slide.close()

    except ImportError:
        print("❌ OpenSlide non installé.")
        print("   Installer avec: pip install openslide-python openslide-bin")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur OpenSlide: {e}")
        sys.exit(1)

    # ===========================================================================
    # 2. Test InputRouter
    # ===========================================================================
    print("\n" + "=" * 60)
    print("2. TEST INPUT ROUTER")
    print("=" * 60)

    from src.wsi.input_router import InputRouter, get_input_metadata

    # Métadonnées via InputRouter
    metadata = get_input_metadata(input_path)
    print(f"Type détecté: {metadata.input_type.value}")
    print(f"Dimensions: {metadata.dimensions}")
    print(f"Format: {metadata.format}")
    print(f"MPP: {metadata.mpp}")
    print(f"Tiles estimés: {metadata.estimated_tiles:,}")

    # ===========================================================================
    # 3. Extraction de tiles
    # ===========================================================================
    print("\n" + "=" * 60)
    print(f"3. EXTRACTION DE {args.max_tiles} TILES")
    print("=" * 60)

    router = InputRouter(filter_tiles=True)

    tiles = []
    for tile in router.process(input_path, max_tiles=args.max_tiles):
        tiles.append(tile)
        print(f"  Tile {len(tiles)}: position ({tile.x:,}, {tile.y:,}), "
              f"shape {tile.image.shape}, dtype {tile.image.dtype}")

    print(f"\n✅ {len(tiles)} tiles extraits avec succès")

    # Statistiques
    if tiles:
        all_images = np.stack([t.image for t in tiles])
        print(f"\nStatistiques des tiles:")
        print(f"  Shape: {all_images.shape}")
        print(f"  Mean RGB: {all_images.mean(axis=(0,1,2)).round(1)}")
        print(f"  Std RGB: {all_images.std(axis=(0,1,2)).round(1)}")

    # ===========================================================================
    # 4. Sauvegarde optionnelle
    # ===========================================================================
    if args.save_tiles and tiles:
        import cv2

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n" + "=" * 60)
        print(f"4. SAUVEGARDE DANS {output_dir}")
        print("=" * 60)

        for i, tile in enumerate(tiles):
            tile_path = output_dir / f"tile_{i:03d}_x{tile.x}_y{tile.y}.png"
            cv2.imwrite(str(tile_path), cv2.cvtColor(tile.image, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {tile_path.name}")

        print(f"\n✅ {len(tiles)} tiles sauvegardés dans {output_dir}")

    print("\n" + "=" * 60)
    print("PIPELINE WSI FONCTIONNEL ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
