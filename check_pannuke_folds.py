import numpy as np
from pathlib import Path

folds_to_check = [
    '/home/amar/data/PanNuke/fold0',
    '/home/amar/data/PanNuke/fold1', 
    '/home/amar/data/PanNuke/fold2',
    '/home/amar/data/PanNuke/Fold 2',
]

print("🔍 Checking PanNuke folds...\n")

for fold_path in folds_to_check:
    path = Path(fold_path)
    if path.exists():
        npy_files = list(path.glob('*.npy'))
        print(f"✅ {path.name:10s}: {len(npy_files)} files")
        
        # Check file sizes
        for f in npy_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"   - {f.name:15s}: {size_mb:6.1f} MB")
        
        # Load and check shapes
        if (path / 'images.npy').exists():
            images = np.load(path / 'images.npy', mmap_mode='r')
            print(f"   → images.shape: {images.shape}")
        
        print()
    else:
        print(f"❌ {fold_path}: not found")

print("\n🎯 Recommendation:")
print("   Use fold2 for baseline (not used in training)")
print("   Or fold0/1 just to test pipeline works")
