#!/usr/bin/env python3
"""
Vérifie que l'environnement est correctement configuré pour CellViT-Optimus.
"""

import sys

def check_python():
    """Vérifie la version Python."""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}", end=" ")
    if version.major == 3 and version.minor >= 10:
        print("✓")
        return True
    print("✗ (Python 3.10+ requis)")
    return False

def check_torch():
    """Vérifie PyTorch et CUDA."""
    try:
        import torch
        print(f"PyTorch: {torch.__version__}", end=" ")
        if torch.cuda.is_available():
            print("✓")
            print(f"  └─ CUDA: {torch.version.cuda}")
            print(f"  └─ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  └─ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        print("✗ (CUDA non disponible)")
        return False
    except ImportError:
        print("PyTorch: ✗ (non installé)")
        return False

def check_timm():
    """Vérifie timm (pour H-optimus-0)."""
    try:
        import timm
        print(f"timm: {timm.__version__} ✓")
        return True
    except ImportError:
        print("timm: ✗ (non installé)")
        return False

def check_huggingface():
    """Vérifie HuggingFace Hub et authentification."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"HuggingFace: connecté en tant que '{user['name']}' ✓")
        return True
    except Exception as e:
        print(f"HuggingFace: ✗ ({e})")
        return False

def check_hoptimus():
    """Teste le chargement de H-optimus-0."""
    try:
        import torch
        import timm
        print("H-optimus-0: chargement...", end=" ", flush=True)
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False
        )
        model = model.eval().cuda().half()

        # Test inférence
        x = torch.randn(1, 3, 224, 224, device='cuda', dtype=torch.float16)
        with torch.no_grad():
            out = model(x)

        params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"✓ ({params:.2f}B params, output: {out.shape})")

        # Libérer mémoire
        del model, x, out
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False

def main():
    print("=" * 60)
    print("CellViT-Optimus — Vérification Environnement")
    print("=" * 60)
    print()

    checks = [
        ("Python", check_python),
        ("PyTorch + CUDA", check_torch),
        ("timm", check_timm),
        ("HuggingFace", check_huggingface),
        ("H-optimus-0", check_hoptimus),
    ]

    results = []
    for name, check_fn in checks:
        results.append(check_fn())
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"Résultat: {passed}/{total} — Environnement OK ✓")
    else:
        print(f"Résultat: {passed}/{total} — Problèmes détectés ✗")
    print("=" * 60)

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
