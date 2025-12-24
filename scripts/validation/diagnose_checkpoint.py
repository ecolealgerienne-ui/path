#!/usr/bin/env python3
"""
Diagnostic checkpoint HoVer-Net - Identifier pourquoi HV energy = 0.000.

Vérifie:
1. Clés du checkpoint (préfixes module./model.?)
2. Valeurs des poids (tous zeros? random?)
3. Architecture match (shapes compatibles?)
4. Test forward pass minimal
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder

def main():
    checkpoint_path = Path("models/checkpoints/hovernet_epidermal_best.pth")

    print("=" * 80)
    print("DIAGNOSTIC CHECKPOINT HOVERNET")
    print("=" * 80)
    print("")

    # Charger checkpoint
    print(f"Chargement: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print(f"\nClés checkpoint:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: dict avec {len(checkpoint[key])} clés")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: Tensor {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")

    # Inspecter state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"\nmodel_state_dict: {len(state_dict)} clés")

        # Lister premières clés
        print(f"\nPremières 10 clés:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            tensor = state_dict[key]
            print(f"  {i+1}. {key}")
            print(f"      Shape: {tensor.shape}")

            # Skip stats pour tensors non-float (ex: num_batches_tracked)
            if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
                print(f"      Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
                print(f"      Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}")
            else:
                print(f"      Dtype: {tensor.dtype} (skipping stats)")

        # Vérifier préfixes
        has_module = any(k.startswith("module.") for k in state_dict.keys())
        has_model = any(k.startswith("model.") for k in state_dict.keys())

        print(f"\nPréfixes détectés:")
        print(f"  'module.' prefix: {has_module}")
        print(f"  'model.' prefix: {has_model}")

        # Vérifier si poids sont tous zeros ou random
        all_zeros = all(torch.all(v == 0).item() for v in state_dict.values())
        print(f"\nTous les poids à zéro: {all_zeros}")

        if all_zeros:
            print("  ❌ PROBLÈME: Checkpoint contient uniquement des zéros!")
            return 1

        # Test de chargement
        print(f"\nTest chargement dans modèle:")
        model = HoVerNetDecoder(embed_dim=1536, n_classes=5, dropout=0.1)

        # Nettoyage clés
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "").replace("model.", "")
            new_state_dict[name] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        print(f"  Missing keys: {len(missing)}")
        if missing:
            print(f"    Premiers 5: {missing[:5]}")

        print(f"  Unexpected keys: {len(unexpected)}")
        if unexpected:
            print(f"    Premiers 5: {unexpected[:5]}")

        if not missing and not unexpected:
            print("  ✅ Toutes les clés matchent parfaitement!")

        # Test forward pass
        print(f"\nTest forward pass:")
        model.eval()

        # Créer input dummy (256 patches × 1536)
        dummy_input = torch.randn(1, 256, 1536)

        with torch.no_grad():
            try:
                np_out, hv_out, nt_out = model(dummy_input)

                print(f"  ✅ Forward pass OK")
                print(f"    NP output: {np_out.shape}")
                print(f"    HV output: {hv_out.shape}")
                print(f"    NT output: {nt_out.shape}")

                # Vérifier si sortie est non-nulle
                hv_mean = hv_out.abs().mean().item()
                np_mean = np_out.abs().mean().item()

                print(f"\n  Sorties moyennes:")
                print(f"    NP mean: {np_mean:.6f}")
                print(f"    HV mean: {hv_mean:.6f}")

                if hv_mean < 1e-6:
                    print(f"\n  ❌ PROBLÈME: HV output quasi-nul ({hv_mean:.2e})")
                    print(f"     → Modèle prédit uniquement des zéros")
                    print(f"     → Soit poids mal chargés, soit checkpoint corrompu")
                else:
                    print(f"\n  ✅ HV output normal ({hv_mean:.6f})")

            except Exception as e:
                print(f"  ❌ Forward pass échoué: {e}")
                return 1

    else:
        print("❌ ERREUR: Pas de 'model_state_dict' dans checkpoint!")
        return 1

    # Vérifier métriques
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print(f"\nMétriques du checkpoint:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("")
    print("=" * 80)
    print("DIAGNOSTIC TERMINÉ")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
