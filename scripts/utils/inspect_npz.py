#!/usr/bin/env python3
"""Inspect .npz file keys."""
import sys
from pathlib import Path
import numpy as np

file_path = Path(sys.argv[1])
data = np.load(file_path)
print(f"Keys in {file_path.name}:")
for key in data.keys():
    print(f"  - {key}: shape {data[key].shape}, dtype {data[key].dtype}")
data.close()
