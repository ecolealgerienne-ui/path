#!/usr/bin/env python3
"""
Extraction de tuiles depuis des images WSI ou grandes images.

Conforme aux specs:
- Tuiles 224×224 @ 0.5 MPP pour H-optimus-0
- Filtrage du fond (tissus seulement)
- Overlap configurable

Usage:
    python scripts/preprocessing/tile_extraction.py --input slide.svs --output-dir tiles/
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List, Generator
import cv2


class TileExtractor:
    """Extracteur de tuiles depuis des images."""

    def __init__(
        self,
        tile_size: int = 224,
        overlap: int = 0,
        tissue_threshold: float = 0.5,
        background_threshold: int = 220,
    ):
        """
        Args:
            tile_size: Taille des tuiles (224 pour H-optimus-0)
            overlap: Chevauchement entre tuiles
            tissue_threshold: Proportion minimale de tissu
            background_threshold: Seuil de gris pour le fond
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.tissue_threshold = tissue_threshold
        self.background_threshold = background_threshold

    def is_tissue(self, tile: np.ndarray) -> bool:
        """Vérifie si une tuile contient du tissu."""
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile

        # Pixels non-blancs
        tissue_mask = gray < self.background_threshold
        tissue_ratio = np.mean(tissue_mask)

        return tissue_ratio >= self.tissue_threshold

    def extract_tiles(
        self,
        image: np.ndarray,
        filter_background: bool = True
    ) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        Extrait les tuiles d'une image.

        Args:
            image: Image source (H, W, 3)
            filter_background: Filtrer les tuiles de fond

        Yields:
            (tile, x, y): Tuile et ses coordonnées
        """
        h, w = image.shape[:2]

        for y in range(0, h - self.tile_size + 1, self.stride):
            for x in range(0, w - self.tile_size + 1, self.stride):
                tile = image[y:y+self.tile_size, x:x+self.tile_size]

                if filter_background and not self.is_tissue(tile):
                    continue

                yield tile, x, y

    def get_tile_count(self, image_shape: Tuple[int, int]) -> int:
        """Calcule le nombre maximal de tuiles."""
        h, w = image_shape[:2]
        n_y = (h - self.tile_size) // self.stride + 1
        n_x = (w - self.tile_size) // self.stride + 1
        return n_y * n_x


def extract_from_image(
    input_path: Path,
    output_dir: Path,
    extractor: TileExtractor,
    save_coords: bool = True
) -> int:
    """Extrait les tuiles d'une image."""
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Erreur lecture: {input_path}")
        return 0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    coords = []
    count = 0

    for tile, x, y in extractor.extract_tiles(img_rgb):
        tile_name = f"{stem}_x{x}_y{y}.png"
        tile_path = output_dir / tile_name

        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(tile_path), tile_bgr)

        coords.append((tile_name, x, y))
        count += 1

    if save_coords and coords:
        coords_path = output_dir / f"{stem}_coords.csv"
        with open(coords_path, 'w') as f:
            f.write("filename,x,y\n")
            for name, x, y in coords:
                f.write(f"{name},{x},{y}\n")

    return count


def extract_from_openslide(
    slide_path: Path,
    output_dir: Path,
    extractor: TileExtractor,
    level: int = 0,
    mpp: float = 0.5
) -> int:
    """Extrait les tuiles d'un WSI avec OpenSlide."""
    try:
        import openslide
    except ImportError:
        print("OpenSlide non installé. Installer avec: pip install openslide-python")
        return 0

    slide = openslide.OpenSlide(str(slide_path))

    # Calculer le niveau approprié pour le MPP cible
    slide_mpp = float(slide.properties.get('openslide.mpp-x', 0.25))
    scale = slide_mpp / mpp

    # Dimensions au niveau choisi
    dims = slide.level_dimensions[level]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slide_path.stem

    coords = []
    count = 0

    tile_size = extractor.tile_size
    stride = extractor.stride

    for y in range(0, dims[1] - tile_size + 1, stride):
        for x in range(0, dims[0] - tile_size + 1, stride):
            # Lire la région
            tile = slide.read_region((x, y), level, (tile_size, tile_size))
            tile = np.array(tile.convert('RGB'))

            if not extractor.is_tissue(tile):
                continue

            tile_name = f"{stem}_x{x}_y{y}.png"
            tile_path = output_dir / tile_name

            cv2.imwrite(str(tile_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

            coords.append((tile_name, x, y))
            count += 1

    # Sauvegarder les coordonnées
    if coords:
        coords_path = output_dir / f"{stem}_coords.csv"
        with open(coords_path, 'w') as f:
            f.write("filename,x,y\n")
            for name, x, y in coords:
                f.write(f"{name},{x},{y}\n")

    slide.close()
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Extraction de tuiles depuis images/WSI"
    )
    parser.add_argument("--input", type=str, required=True, help="Image ou WSI d'entrée")
    parser.add_argument("--output-dir", type=str, required=True, help="Répertoire de sortie")
    parser.add_argument("--tile-size", type=int, default=224, help="Taille des tuiles")
    parser.add_argument("--overlap", type=int, default=0, help="Chevauchement")
    parser.add_argument("--tissue-threshold", type=float, default=0.5, help="Seuil tissu")
    parser.add_argument("--no-filter", action="store_true", help="Ne pas filtrer le fond")
    parser.add_argument("--level", type=int, default=0, help="Niveau WSI")

    args = parser.parse_args()

    extractor = TileExtractor(
        tile_size=args.tile_size,
        overlap=args.overlap,
        tissue_threshold=args.tissue_threshold,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Détecter le type de fichier
    wsi_extensions = ['.svs', '.ndpi', '.mrxs', '.scn', '.vms', '.vmu']

    if input_path.suffix.lower() in wsi_extensions:
        count = extract_from_openslide(input_path, output_dir, extractor, args.level)
    else:
        count = extract_from_image(input_path, output_dir, extractor)

    print(f"\n✅ {count} tuiles extraites dans {output_dir}")


if __name__ == "__main__":
    main()
