#!/usr/bin/env python3
"""Debug: Vérifier la sortie NT du modèle."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from src.ui.inference_engine import CellVitEngine

# Charger une image de test
val_data = np.load("data/family_data_v13_smart_crops/respiratory_val_v13_smart_crops.npz")
image = val_data['images'][0]  # Première image

print("=== DEBUG TYPE MAP ===")
print(f"Image shape: {image.shape}")

# Charger le moteur
engine = CellVitEngine(device="cuda", organ="Lung")

# Analyser
result = engine.analyze(image, compute_morphometry=False, compute_uncertainty=False)

print(f"\n=== RÉSULTATS ===")
print(f"n_nuclei: {result.n_nuclei}")
print(f"type_map shape: {result.type_map.shape}")
print(f"type_map unique values: {np.unique(result.type_map)}")
print(f"type_map value counts:")
for val in np.unique(result.type_map):
    count = np.sum(result.type_map == val)
    print(f"  Type {val}: {count} pixels")

# Vérifier les types par noyau
print(f"\n=== TYPES PAR NOYAU ===")
inst_map = result.instance_map
for inst_id in np.unique(inst_map):
    if inst_id == 0:
        continue
    mask = inst_map == inst_id
    types_in_mask = result.type_map[mask]
    if len(types_in_mask) > 0:
        dominant_type = int(np.bincount(types_in_mask.astype(int)).argmax())
        print(f"  Noyau {inst_id}: Type {dominant_type}")
