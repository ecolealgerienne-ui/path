#!/usr/bin/env python3
"""
Crée des données de test synthétiques avec info organ pour tester
le pipeline d'évaluation par famille.
"""
import numpy as np
from pathlib import Path
import argparse

# Distribution typique PanNuke
ORGANS_BY_FAMILY = {
    "glandular": ["Breast", "Prostate", "Thyroid"],
    "digestive": ["Colon", "Stomach"],
    "urologic": ["Kidney", "Bladder"],
    "respiratory": ["Lung", "Liver"],
    "epidermal": ["Skin", "HeadNeck"],
}


def create_synthetic_image(organ: str, idx: int, output_dir: Path):
    """Create a synthetic NPZ file with organ info."""
    # Create synthetic data (256x256)
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Random instance map (5-15 instances)
    n_instances = np.random.randint(5, 16)
    inst_map = np.zeros((256, 256), dtype=np.int32)

    # Create random circular instances
    for i in range(1, n_instances + 1):
        cx = np.random.randint(30, 226)
        cy = np.random.randint(30, 226)
        radius = np.random.randint(10, 20)

        y, x = np.ogrid[:256, :256]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        inst_map[mask] = i

    # Random type map (weighted by typical tissue)
    type_map = np.zeros_like(inst_map, dtype=np.uint8)

    # Assign random types to each instance
    for i in range(1, n_instances + 1):
        inst_mask = inst_map == i
        # Random type (1-5)
        type_idx = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.2, 0.1, 0.1, 0.2])
        type_map[inst_mask] = type_idx

    # Save as NPZ with organ info
    filename = f"{organ.lower()}_{idx:04d}.npz"
    output_file = output_dir / filename

    np.savez(
        output_file,
        image=image,
        inst_map=inst_map,
        type_map=type_map,
        organ=organ,  # ← KEY INFO FOR FAMILY ROUTING
    )

    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/evaluation/synthetic_test"),
        help="Output directory for synthetic data"
    )
    parser.add_argument(
        "--images_per_organ",
        type=int,
        default=10,
        help="Number of images per organ"
    )

    args = parser.parse_args()

    print("="*70)
    print("CREATING SYNTHETIC TEST DATA")
    print("="*70)
    print(f"Output: {args.output_dir}")
    print(f"Images per organ: {args.images_per_organ}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_created = 0

    for family, organs in ORGANS_BY_FAMILY.items():
        print(f"\n{family.upper()}")
        print("-"*40)

        for organ in organs:
            for i in range(args.images_per_organ):
                output_file = create_synthetic_image(organ, i, args.output_dir)
                total_created += 1

            print(f"  ✓ {organ:15}: {args.images_per_organ} images")

    print("\n" + "="*70)
    print(f"✅ Created {total_created} synthetic test images")
    print(f"   Saved to: {args.output_dir}")
    print("="*70)

    print("\nNext steps:")
    print(f"  1. Organize by family:")
    print(f"     python scripts/evaluation/organize_test_by_family.py \\")
    print(f"         --input_dir {args.output_dir} \\")
    print(f"         --output_dir data/evaluation/synthetic_by_family")
    print(f"")
    print(f"  2. Evaluate:")
    print(f"     python scripts/evaluation/evaluate_by_family.py \\")
    print(f"         --dataset_dir data/evaluation/synthetic_by_family \\")
    print(f"         --checkpoint_dir models/checkpoints_FIXED \\")
    print(f"         --output_dir results/synthetic_test")


if __name__ == "__main__":
    main()
