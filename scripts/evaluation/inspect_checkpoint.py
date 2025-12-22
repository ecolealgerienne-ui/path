#!/usr/bin/env python3
"""
Vérifie que le checkpoint HoVer-Net est bien chargé.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader

def main():
    checkpoint_path = "models/checkpoints/hovernet_glandular_best.pth"

    print("=" * 80)
    print(f"INSPECTION CHECKPOINT: {checkpoint_path}")
    print("=" * 80)
    print("")

    # Charger checkpoint brut
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("Clés checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    print("")

    # État du modèle
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    print(f"Nombre de paramètres: {len(state_dict)}")
    print("")

    # Vérifier les branches NP/HV/NT
    np_params = [k for k in state_dict.keys() if 'np_head' in k]
    hv_params = [k for k in state_dict.keys() if 'hv_head' in k]
    nt_params = [k for k in state_dict.keys() if 'nt_head' in k]

    print(f"NP head params: {len(np_params)}")
    for p in np_params:
        tensor = state_dict[p]
        print(f"  {p:50} {tensor.shape} [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    print("")

    print(f"HV head params: {len(hv_params)}")
    for p in hv_params:
        tensor = state_dict[p]
        print(f"  {p:50} {tensor.shape} [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    print("")

    print(f"NT head params: {len(nt_params)}")
    for p in nt_params:
        tensor = state_dict[p]
        print(f"  {p:50} {tensor.shape} [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    print("")

    # Charger modèle
    print("=" * 80)
    print("CHARGEMENT MODÈLE")
    print("=" * 80)
    print("")

    hovernet = ModelLoader.load_hovernet(checkpoint_path, device="cpu")
    hovernet.eval()

    # Vérifier les poids du modèle chargé
    print("Poids HV head dans le modèle chargé:")
    for name, param in hovernet.named_parameters():
        if 'hv_head' in name:
            print(f"  {name:50} {param.shape} [{param.min().item():.3f}, {param.max().item():.3f}]")
    print("")

    # Test forward
    print("=" * 80)
    print("TEST FORWARD")
    print("=" * 80)
    print("")

    # Features aléatoires
    dummy_features = torch.randn(1, 256, 1536)
    with torch.no_grad():
        np_out, hv_out, nt_out = hovernet(dummy_features)

    print(f"NP output: {np_out.shape} [{np_out.min().item():.3f}, {np_out.max().item():.3f}]")
    print(f"HV output: {hv_out.shape} [{hv_out.min().item():.3f}, {hv_out.max().item():.3f}]")
    print(f"NT output: {nt_out.shape} [{nt_out.min().item():.3f}, {nt_out.max().item():.3f}]")
    print("")

    # DIAGNOSTIC
    print("=" * 80)
    print("DIAGNOSTIC")
    print("=" * 80)
    print("")

    if abs(hv_out.max().item()) < 0.5 and abs(hv_out.min().item()) < 0.5:
        print("❌ HV outputs sont COMPRESSÉS autour de 0 (range < 0.5)")
        print("   → Problème probable:")
        print("     1. Modèle pas assez entraîné")
        print("     2. Learning rate trop faible")
        print("     3. Mauvaise initialisation")
        print("")
        print("   → Solution: Ajouter tanh() activation ou ré-entraîner")
    else:
        print("✅ HV outputs ont une range correcte")


if __name__ == "__main__":
    main()
