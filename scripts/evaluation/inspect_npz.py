#!/usr/bin/env python3
"""Inspect NPZ file structure to understand what data is available."""
import numpy as np
from pathlib import Path
import sys

def inspect_npz(npz_file: Path):
    """Inspect a single NPZ file."""
    print(f"\n{'='*70}")
    print(f"FILE: {npz_file.name}")
    print('='*70)

    data = np.load(npz_file, allow_pickle=True)

    print("\nKeys in file:")
    for key in sorted(data.keys()):
        val = data[key]
        if isinstance(val, np.ndarray):
            print(f"  {key:15} : shape={val.shape}, dtype={val.dtype}")
            # Show unique values for small arrays
            if val.size < 20:
                print(f"                   values={val}")
            elif key in ['inst_map', 'type_map']:
                unique = np.unique(val)
                print(f"                   unique values: {unique[:10]}...")
        else:
            print(f"  {key:15} : {type(val).__name__} = {val}")

    # Check if there's tissue type info
    if 'type_map' in data:
        type_map = data['type_map']
        unique_types = np.unique(type_map)
        print(f"\nType map unique values: {unique_types}")
        print("  (0=background, 1=Neoplastic, 2=Inflammatory, 3=Connective, 4=Dead, 5=Epithelial)")

if __name__ == "__main__":
    # Check a few files
    data_dir = Path("data/evaluation/pannuke_fold2_converted")

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        sys.exit(1)

    npz_files = sorted(data_dir.glob("*.npz"))[:3]  # First 3 files

    print(f"Found {len(list(data_dir.glob('*.npz')))} total NPZ files")
    print(f"Inspecting first {len(npz_files)} files...\n")

    for npz_file in npz_files:
        inspect_npz(npz_file)
