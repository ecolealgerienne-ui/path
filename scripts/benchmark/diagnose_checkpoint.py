#!/usr/bin/env python3
"""Diagnostic du checkpoint pour comprendre les différences."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.ui.organ_config import FAMILY_CHECKPOINTS

family = "respiratory"
checkpoint_path = FAMILY_CHECKPOINTS[family]

print(f"Checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("\n=== CHECKPOINT KEYS ===")
for k in checkpoint.keys():
    if k != "model_state_dict":
        print(f"  {k}: {checkpoint[k]}")

print("\n=== FLAGS EXPLICITES ===")
print(f"  use_hybrid: {checkpoint.get('use_hybrid', 'NOT FOUND')}")
print(f"  use_fpn_chimique: {checkpoint.get('use_fpn_chimique', 'NOT FOUND')}")
print(f"  use_h_alpha: {checkpoint.get('use_h_alpha', 'NOT FOUND')}")

print("\n=== DÉTECTION PAR CLÉS ===")
state_dict = checkpoint.get("model_state_dict", checkpoint)
has_h_alphas = any('h_alphas' in k for k in state_dict.keys())
has_fpn = any('fpn' in k.lower() or 'h_channel' in k for k in state_dict.keys())
print(f"  h_alphas keys exist: {has_h_alphas}")
print(f"  FPN/h_channel keys exist: {has_fpn}")

print("\n=== CLÉS H_ALPHAS ===")
for k in state_dict.keys():
    if 'h_alpha' in k.lower():
        print(f"  {k}: shape={state_dict[k].shape}")
