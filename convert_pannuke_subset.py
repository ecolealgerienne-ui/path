#!/usr/bin/env python3
"""
Convert PanNuke fold2 subset to .npz format for evaluation.

PanNuke format:
- images.npy: (N, 256, 256, 3) uint8
- masks.npy: (N, 256, 256, 6) int32
  - masks[:,:,:,0] = unused
  - masks[:,:,:,1] = Neoplastic instance IDs
  - masks[:,:,:,2] = Inflammatory instance IDs
  - masks[:,:,:,3] = Connective instance IDs
  - masks[:,:,:,4] = Dead instance IDs
  - masks[:,:,:,5] = Epithelial instance IDs
- types.npy: (N,) object - organ names

Output format (.npz):
- image: (256, 256, 3) uint8
- inst_map: (256, 256) int32 - unified instance map
- type_map: (256, 256) uint8 - type per instance
- organ: str - organ name
"""
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# PanNuke class mapping
PANNUKE_CLASSES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory", 
    3: "Connective",
    4: "Dead",
    5: "Epithelial"
}

def convert_pannuke_to_npz(
    images: np.ndarray,
    masks: np.ndarray,
    types: np.ndarray,
    indices: list,
    output_dir: Path
):
    """Convert PanNuke samples to individual .npz files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, idx in enumerate(tqdm(indices, desc="Converting")):
        image = images[idx]
        mask = masks[idx]
        organ = str(types[idx])
        
        # Create unified instance map from separate channels
        inst_map = np.zeros((256, 256), dtype=np.int32)
        type_map = np.zeros((256, 256), dtype=np.uint8)
        
        instance_counter = 1
        
        # Process each class channel (1-5)
        for class_id in range(1, 6):
            class_mask = mask[:, :, class_id]
            unique_ids = np.unique(class_mask)
            unique_ids = unique_ids[unique_ids > 0]  # Skip background
            
            for inst_id in unique_ids:
                inst_pixels = class_mask == inst_id
                inst_map[inst_pixels] = instance_counter
                type_map[inst_pixels] = class_id
                instance_counter += 1
        
        # Save as .npz
        filename = f"{organ.lower().replace(' ', '_')}_{idx:04d}.npz"
        output_file = output_dir / filename
        
        np.savez(
            output_file,
            image=image,
            inst_map=inst_map,
            type_map=type_map,
            organ=organ
        )
    
    print(f"\n✅ Converted {len(indices)} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold_dir",
        type=Path,
        default=Path("/home/amar/data/PanNuke/fold2"),
        help="PanNuke fold directory"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/evaluation/pannuke_fold2_test"),
        help="Output directory for .npz files"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to convert"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CONVERTING PANNUKE FOLD2 TO .NPZ FORMAT")
    print("="*70)
    print(f"Input:  {args.fold_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {args.num_samples}")
    print()
    
    # Load data (memory-mapped for efficiency)
    print("📁 Loading PanNuke data...")
    images = np.load(args.fold_dir / "images.npy", mmap_mode='r')
    masks = np.load(args.fold_dir / "masks.npy", mmap_mode='r')
    types = np.load(args.fold_dir / "types.npy")
    
    print(f"   Images: {images.shape}")
    print(f"   Masks:  {masks.shape}")
    print(f"   Types:  {types.shape}")
    print()
    
    # Select random subset
    np.random.seed(args.seed)
    n_total = len(images)
    indices = np.random.choice(n_total, size=args.num_samples, replace=False)
    indices = sorted(indices)
    
    print(f"📊 Selected indices: {indices[:10]}... (showing first 10)")
    print()
    
    # Convert
    convert_pannuke_to_npz(images, masks, types, indices, args.output_dir)
    
    # Verify one file
    print("\n🔍 Verifying first converted file...")
    npz_files = sorted(args.output_dir.glob("*.npz"))
    if npz_files:
        data = np.load(npz_files[0])
        print(f"   File: {npz_files[0].name}")
        print(f"   image: {data['image'].shape} {data['image'].dtype}")
        print(f"   inst_map: {data['inst_map'].shape} {data['inst_map'].dtype}")
        print(f"   type_map: {data['type_map'].shape} {data['type_map'].dtype}")
        print(f"   organ: {data['organ']}")
        print(f"   Instances: {len(np.unique(data['inst_map'])) - 1}")
    
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE")
    print("="*70)
    print(f"\nNext step:")
    print(f"  python scripts/evaluation/evaluate_ground_truth.py \\")
    print(f"      --dataset_dir {args.output_dir} \\")
    print(f"      --output_dir results/baseline_watershed_real")


if __name__ == "__main__":
    main()
