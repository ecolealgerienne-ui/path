"""
Tile APCData Images for Native Resolution YOLO Training

Ce script découpe les images APCData en tuiles 672×672 (3×224) avec overlap 25%.
Cela permet d'entraîner YOLO à résolution native sans perte d'information.

Avantages:
- Préserve les détails cellulaires (pas de resize destructif)
- Augmente le dataset (~12× plus d'images)
- Taille 672 = 3×224, prêt pour H-Optimus-0

Gestion des cellules coupées:
- Overlap 25% (168 px) garantit que chaque cellule apparaît complète dans au moins une tuile
- Labels filtrés: seules les cellules avec >50% dans la tuile sont gardées
- NMS à l'inférence fusionne les détections dupliquées

Usage:
    python scripts/cytology/05_tile_apcdata.py \
        --input_dir data/raw/apcdata/APCData_YOLO_Detection \
        --output_dir data/raw/apcdata/APCData_YOLO_Tiled_672 \
        --tile_size 672 \
        --overlap 0.25

Author: V15 Cytology Branch
Date: 2026-01-23
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from dataclasses import dataclass


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_info(message: str):
    print(f"  [INFO] {message}")


def print_success(message: str):
    print(f"  [OK] {message}")


def print_warning(message: str):
    print(f"  [WARN] {message}")


@dataclass
class BBox:
    """YOLO format bounding box (normalized coordinates)."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_absolute(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to absolute pixel coordinates (x1, y1, x2, y2)."""
        x_center_abs = self.x_center * img_width
        y_center_abs = self.y_center * img_height
        w_abs = self.width * img_width
        h_abs = self.height * img_height

        x1 = int(x_center_abs - w_abs / 2)
        y1 = int(y_center_abs - h_abs / 2)
        x2 = int(x_center_abs + w_abs / 2)
        y2 = int(y_center_abs + h_abs / 2)

        return x1, y1, x2, y2

    @classmethod
    def from_absolute(cls, class_id: int, x1: int, y1: int, x2: int, y2: int,
                      img_width: int, img_height: int) -> 'BBox':
        """Create from absolute pixel coordinates."""
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        return cls(class_id, x_center, y_center, width, height)

    def to_yolo_line(self) -> str:
        """Convert to YOLO label format string."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}\n"


def read_yolo_labels(label_path: Path) -> List[BBox]:
    """Read YOLO format label file."""
    bboxes = []

    if not label_path.exists():
        return bboxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                bbox = BBox(
                    class_id=int(parts[0]),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4])
                )
                bboxes.append(bbox)

    return bboxes


def compute_bbox_overlap_ratio(
    bbox_abs: Tuple[int, int, int, int],
    tile_x: int,
    tile_y: int,
    tile_size: int
) -> float:
    """
    Compute what fraction of the bbox is inside the tile.

    Args:
        bbox_abs: (x1, y1, x2, y2) in absolute coordinates
        tile_x, tile_y: Top-left corner of tile
        tile_size: Size of tile

    Returns:
        Ratio of bbox area inside tile (0.0 to 1.0)
    """
    bx1, by1, bx2, by2 = bbox_abs
    tx1, ty1 = tile_x, tile_y
    tx2, ty2 = tile_x + tile_size, tile_y + tile_size

    # Compute intersection
    ix1 = max(bx1, tx1)
    iy1 = max(by1, ty1)
    ix2 = min(bx2, tx2)
    iy2 = min(by2, ty2)

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    bbox_area = (bx2 - bx1) * (by2 - by1)

    if bbox_area <= 0:
        return 0.0

    return intersection_area / bbox_area


def clip_bbox_to_tile(
    bbox_abs: Tuple[int, int, int, int],
    tile_x: int,
    tile_y: int,
    tile_size: int
) -> Tuple[int, int, int, int]:
    """
    Clip bbox to tile boundaries and convert to tile-relative coordinates.

    Args:
        bbox_abs: (x1, y1, x2, y2) in image absolute coordinates
        tile_x, tile_y: Top-left corner of tile in image coordinates
        tile_size: Size of tile

    Returns:
        (x1, y1, x2, y2) in tile-relative coordinates, clipped to tile
    """
    bx1, by1, bx2, by2 = bbox_abs

    # Convert to tile-relative coordinates
    rx1 = bx1 - tile_x
    ry1 = by1 - tile_y
    rx2 = bx2 - tile_x
    ry2 = by2 - tile_y

    # Clip to tile boundaries
    rx1 = max(0, min(tile_size, rx1))
    ry1 = max(0, min(tile_size, ry1))
    rx2 = max(0, min(tile_size, rx2))
    ry2 = max(0, min(tile_size, ry2))

    return rx1, ry1, rx2, ry2


def generate_tiles(
    img_width: int,
    img_height: int,
    tile_size: int,
    overlap: float
) -> List[Tuple[int, int]]:
    """
    Generate tile positions with overlap.

    Args:
        img_width, img_height: Image dimensions
        tile_size: Size of each tile
        overlap: Overlap ratio (0.0 to 0.5)

    Returns:
        List of (x, y) positions for top-left corner of each tile
    """
    stride = int(tile_size * (1 - overlap))
    tiles = []

    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            # Adjust last tile to not exceed image bounds
            actual_x = min(x, max(0, img_width - tile_size))
            actual_y = min(y, max(0, img_height - tile_size))

            # Avoid duplicates at edges
            if (actual_x, actual_y) not in tiles:
                tiles.append((actual_x, actual_y))

            x += stride
            if x >= img_width and x - stride + tile_size < img_width:
                # Add edge tile
                x = img_width - tile_size
                if (x, actual_y) not in tiles:
                    tiles.append((x, actual_y))
                break

        y += stride
        if y >= img_height and y - stride + tile_size < img_height:
            # Add edge row
            y = img_height - tile_size

    return tiles


def tile_image(
    image: np.ndarray,
    bboxes: List[BBox],
    tile_size: int,
    overlap: float,
    min_bbox_ratio: float = 0.5
) -> List[Tuple[np.ndarray, List[BBox], Tuple[int, int]]]:
    """
    Tile an image and its labels.

    Args:
        image: Input image (H, W, C)
        bboxes: List of bounding boxes in YOLO format
        tile_size: Size of each tile
        overlap: Overlap ratio
        min_bbox_ratio: Minimum ratio of bbox that must be inside tile to keep it

    Returns:
        List of (tile_image, tile_bboxes, (tile_x, tile_y))
    """
    img_height, img_width = image.shape[:2]

    # Handle small images
    if img_width < tile_size or img_height < tile_size:
        # Pad image to tile_size
        padded = np.zeros((max(img_height, tile_size), max(img_width, tile_size), 3), dtype=np.uint8)
        padded[:img_height, :img_width] = image
        image = padded
        img_height, img_width = image.shape[:2]

    tile_positions = generate_tiles(img_width, img_height, tile_size, overlap)
    results = []

    for tile_x, tile_y in tile_positions:
        # Extract tile
        tile_img = image[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size].copy()

        # Process bboxes for this tile
        tile_bboxes = []

        for bbox in bboxes:
            # Convert to absolute coordinates
            bbox_abs = bbox.to_absolute(img_width, img_height)

            # Check overlap ratio
            overlap_ratio = compute_bbox_overlap_ratio(bbox_abs, tile_x, tile_y, tile_size)

            if overlap_ratio >= min_bbox_ratio:
                # Clip and convert to tile coordinates
                clipped = clip_bbox_to_tile(bbox_abs, tile_x, tile_y, tile_size)
                rx1, ry1, rx2, ry2 = clipped

                # Skip degenerate boxes
                if rx2 - rx1 < 2 or ry2 - ry1 < 2:
                    continue

                # Convert back to YOLO format (normalized to tile)
                tile_bbox = BBox.from_absolute(
                    class_id=bbox.class_id,
                    x1=rx1, y1=ry1, x2=rx2, y2=ry2,
                    img_width=tile_size, img_height=tile_size
                )
                tile_bboxes.append(tile_bbox)

        results.append((tile_img, tile_bboxes, (tile_x, tile_y)))

    return results


def process_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    tile_size: int,
    overlap: float,
    min_bbox_ratio: float
) -> Dict[str, int]:
    """Process a single split (train or val)."""

    input_images = input_dir / split / "images"
    input_labels = input_dir / split / "labels"
    output_images = output_dir / split / "images"
    output_labels = output_dir / split / "labels"

    if not input_images.exists():
        return None

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    stats = {
        "original_images": 0,
        "total_tiles": 0,
        "tiles_with_cells": 0,
        "total_cells_original": 0,
        "total_cells_tiled": 0
    }

    # Get all images
    image_files = list(input_images.glob("*.jpg")) + \
                  list(input_images.glob("*.png")) + \
                  list(input_images.glob("*.jpeg"))

    for img_path in image_files:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print_warning(f"Could not read: {img_path.name}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read labels
        label_path = input_labels / f"{img_path.stem}.txt"
        bboxes = read_yolo_labels(label_path)

        stats["original_images"] += 1
        stats["total_cells_original"] += len(bboxes)

        # Generate tiles
        tiles = tile_image(image, bboxes, tile_size, overlap, min_bbox_ratio)

        for idx, (tile_img, tile_bboxes, (tx, ty)) in enumerate(tiles):
            # Generate tile filename
            tile_name = f"{img_path.stem}_tile_{idx:03d}_x{tx}_y{ty}"

            # Save tile image
            tile_img_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
            tile_img_path = output_images / f"{tile_name}.jpg"
            cv2.imwrite(str(tile_img_path), tile_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save tile labels
            tile_label_path = output_labels / f"{tile_name}.txt"
            with open(tile_label_path, 'w') as f:
                for bbox in tile_bboxes:
                    f.write(bbox.to_yolo_line())

            stats["total_tiles"] += 1
            if len(tile_bboxes) > 0:
                stats["tiles_with_cells"] += 1
            stats["total_cells_tiled"] += len(tile_bboxes)

    return stats


def create_data_yaml(output_dir: Path, tile_size: int) -> None:
    """Create YOLO data.yaml for tiled dataset."""

    yaml_content = f"""# APCData YOLO Tiled Dataset Configuration
# V15 Cytology Pipeline — Native Resolution Tiling
# Tile size: {tile_size}×{tile_size} (multiple of 224 for H-Optimus-0)

path: {output_dir.absolute()}
train: train/images
val: val/images

# Single class for detection
names:
  0: cell

nc: 1

# Note: Images tiled at native resolution for better cell detection
# Overlap ensures cells appear complete in at least one tile
# Use NMS at inference to handle duplicate detections
"""

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print_success(f"Created {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tile APCData images for native resolution YOLO training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/apcdata/APCData_YOLO_Detection",
        help="Path to input APCData directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/apcdata/APCData_YOLO_Tiled_672",
        help="Path to output tiled directory"
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=672,
        help="Tile size (default: 672 = 3×224)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap ratio (default: 0.25 = 25%%)"
    )
    parser.add_argument(
        "--min_bbox_ratio",
        type=float,
        default=0.5,
        help="Minimum bbox overlap ratio to keep (default: 0.5 = 50%%)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if exists"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("\n" + "=" * 80)
    print("  APCDATA TILING FOR NATIVE RESOLUTION TRAINING")
    print("  V15 Cytology Pipeline")
    print("=" * 80)

    print_header("CONFIGURATION")
    print_info(f"Input: {input_dir}")
    print_info(f"Output: {output_dir}")
    print_info(f"Tile size: {args.tile_size}×{args.tile_size}")
    print_info(f"Overlap: {args.overlap*100:.0f}%")
    print_info(f"Min bbox ratio: {args.min_bbox_ratio*100:.0f}%")

    # Verify input
    print_header("STEP 1: VERIFY INPUT")

    if not input_dir.exists():
        print(f"  [ERROR] Input directory not found: {input_dir}")
        return 1

    print_success(f"Input directory exists")

    # Handle output directory
    print_header("STEP 2: PREPARE OUTPUT")

    if output_dir.exists():
        if args.force:
            print_warning(f"Removing existing output: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print_warning(f"Output directory exists: {output_dir}")
            print_info("Use --force to overwrite")
            return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"Output directory created")

    # Process splits
    print_header("STEP 3: TILE IMAGES")

    all_stats = {}

    for split in ["train", "val"]:
        print_info(f"Processing {split}...")
        stats = process_split(
            input_dir, output_dir, split,
            args.tile_size, args.overlap, args.min_bbox_ratio
        )

        if stats:
            all_stats[split] = stats
            print_success(f"  {split}: {stats['original_images']} images → {stats['total_tiles']} tiles")
            print_info(f"    Tiles with cells: {stats['tiles_with_cells']}")
            print_info(f"    Cells: {stats['total_cells_original']} original → {stats['total_cells_tiled']} in tiles")

    # Create data.yaml
    print_header("STEP 4: CREATE DATA.YAML")
    create_data_yaml(output_dir, args.tile_size)

    # Summary
    print_header("TILING SUMMARY")

    total_original = sum(s['original_images'] for s in all_stats.values())
    total_tiles = sum(s['total_tiles'] for s in all_stats.values())
    total_tiles_with_cells = sum(s['tiles_with_cells'] for s in all_stats.values())

    print_info(f"Original images: {total_original}")
    print_info(f"Generated tiles: {total_tiles} ({total_tiles/total_original:.1f}× augmentation)")
    print_info(f"Tiles with cells: {total_tiles_with_cells} ({total_tiles_with_cells/total_tiles*100:.1f}%)")

    print("\n" + "=" * 80)
    print("  TILING COMPLETED")
    print("=" * 80)
    print(f"\n  Output: {output_dir}")
    print(f"\n  Next step:")
    print(f"    python scripts/cytology/03_train_yolo26_apcdata.py \\")
    print(f"        --data {output_dir}/data.yaml \\")
    print(f"        --model yolo26s.pt \\")
    print(f"        --epochs 300 \\")
    print(f"        --imgsz {args.tile_size} \\")
    print(f"        --batch 8 \\")
    print(f"        --name apcdata_tiled_yolo26s")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
