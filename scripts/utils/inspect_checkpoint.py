#!/usr/bin/env python3
"""
Inspecte la structure d'un checkpoint PyTorch.

Usage:
    python scripts/utils/inspect_checkpoint.py models/pretrained/CellViT-256.pth
"""

import argparse
import torch
from pathlib import Path
from collections import OrderedDict


def inspect_checkpoint(path: str, max_keys: int = 50):
    """Inspecte un checkpoint PyTorch."""
    path = Path(path)

    if not path.exists():
        print(f"❌ Fichier non trouvé: {path}")
        return

    print(f"\n📦 Inspection: {path}")
    print(f"   Taille: {path.stat().st_size / 1e6:.1f} MB")
    print("=" * 60)

    # Charger le checkpoint
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return

    # Afficher la structure
    if isinstance(checkpoint, dict):
        print(f"\n🔑 Clés principales ({len(checkpoint)}):")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"   • {key}: dict ({len(value)} clés)")
            elif isinstance(value, (list, tuple)):
                print(f"   • {key}: {type(value).__name__} ({len(value)} éléments)")
            elif isinstance(value, torch.Tensor):
                print(f"   • {key}: Tensor {value.shape}")
            elif isinstance(value, (OrderedDict,)):
                print(f"   • {key}: OrderedDict ({len(value)} clés)")
            else:
                print(f"   • {key}: {type(value).__name__} = {repr(value)[:50]}")

        # Inspecter les state_dicts
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in checkpoint:
                print(f"\n📋 Structure de '{key}':")
                inspect_state_dict(checkpoint[key], max_keys)
                break
        else:
            # Peut-être que c'est directement un state_dict
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                print("\n📋 Checkpoint est un state_dict direct:")
                inspect_state_dict(checkpoint, max_keys)

    elif isinstance(checkpoint, torch.nn.Module):
        print("\n📋 Checkpoint est un modèle complet")
        print(checkpoint)
    else:
        print(f"\n⚠️ Type inattendu: {type(checkpoint)}")


def inspect_state_dict(state_dict: dict, max_keys: int = 50):
    """Inspecte un state_dict."""
    keys = list(state_dict.keys())
    n_keys = len(keys)

    print(f"   Total: {n_keys} paramètres")

    # Calculer le nombre total de paramètres
    total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
    print(f"   Paramètres: {total_params:,} ({total_params/1e6:.1f}M)")

    # Grouper par préfixe
    prefixes = {}
    for key in keys:
        prefix = key.split(".")[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)

    print(f"\n   Modules ({len(prefixes)}):")
    for prefix, pkeys in sorted(prefixes.items()):
        params = sum(
            state_dict[k].numel() for k in pkeys
            if isinstance(state_dict[k], torch.Tensor)
        )
        print(f"      • {prefix}: {len(pkeys)} clés, {params:,} params")

    # Afficher quelques clés
    print(f"\n   Premières clés ({min(max_keys, n_keys)}/{n_keys}):")
    for key in keys[:max_keys]:
        value = state_dict[key]
        if isinstance(value, torch.Tensor):
            print(f"      {key}: {list(value.shape)}")
        else:
            print(f"      {key}: {type(value).__name__}")

    if n_keys > max_keys:
        print(f"      ... ({n_keys - max_keys} clés supplémentaires)")


def main():
    parser = argparse.ArgumentParser(description="Inspecte un checkpoint PyTorch")
    parser.add_argument("path", type=str, help="Chemin vers le checkpoint")
    parser.add_argument("--max-keys", type=int, default=50,
                        help="Nombre max de clés à afficher")

    args = parser.parse_args()
    inspect_checkpoint(args.path, args.max_keys)


if __name__ == "__main__":
    main()
