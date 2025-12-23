#!/usr/bin/env python3
"""
Script de validation rapide des 5 checkpoints de famille.

Vérifie que tous les modèles se chargent correctement et extrait les métriques d'entraînement.

Usage:
    python scripts/evaluation/validate_all_checkpoints.py --checkpoints_dir models/checkpoints
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
from src.models.hovernet_decoder import HoVerNetDecoder

FAMILIES = ["glandular", "digestive", "urologic", "epidermal", "respiratory"]


def validate_checkpoint(checkpoint_path: Path, family: str):
    """Valide un checkpoint et extrait les métriques."""
    if not checkpoint_path.exists():
        return {
            "status": "❌ MISSING",
            "family": family,
            "path": str(checkpoint_path),
        }

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Vérifier clés obligatoires
        required_keys = ['model_state_dict', 'epoch', 'best_loss']
        missing_keys = [k for k in required_keys if k not in checkpoint]

        if missing_keys:
            return {
                "status": "⚠️  INCOMPLETE",
                "family": family,
                "missing_keys": missing_keys,
            }

        # Extraire métriques
        metrics = checkpoint.get('best_metrics', {})

        result = {
            "status": "✅ OK",
            "family": family,
            "epoch": checkpoint['epoch'],
            "best_loss": checkpoint['best_loss'],
            "np_dice": metrics.get('np_dice', 'N/A'),
            "hv_mse": metrics.get('hv_mse', 'N/A'),
            "nt_acc": metrics.get('nt_acc', 'N/A'),
            "size_mb": checkpoint_path.stat().st_size / 1e6,
        }

        # Tester chargement du modèle
        model = HoVerNetDecoder(embed_dim=1536, n_classes=5)
        model.load_state_dict(checkpoint['model_state_dict'])

        result["model_params"] = sum(p.numel() for p in model.parameters())

        return result

    except Exception as e:
        return {
            "status": "❌ ERROR",
            "family": family,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Valide tous les checkpoints de famille")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints", help="Répertoire checkpoints")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)

    print("="*80)
    print("VALIDATION DES CHECKPOINTS DE FAMILLE")
    print("="*80)

    results = []

    for family in FAMILIES:
        checkpoint_path = checkpoints_dir / f"hovernet_{family}_best.pth"
        print(f"\n📦 Validation {family}...")

        result = validate_checkpoint(checkpoint_path, family)
        results.append(result)

        # Affichage
        print(f"  Status: {result['status']}")

        if result['status'] == "✅ OK":
            print(f"  Epoch: {result['epoch']}")
            print(f"  Val Loss: {result['best_loss']:.4f}")
            print(f"  NP Dice: {result['np_dice']}")
            print(f"  HV MSE: {result['hv_mse']}")
            print(f"  NT Acc: {result['nt_acc']}")
            print(f"  Size: {result['size_mb']:.1f} MB")
            print(f"  Params: {result['model_params']:,}")

        elif result['status'] == "⚠️  INCOMPLETE":
            print(f"  Missing keys: {', '.join(result['missing_keys'])}")

        elif result['status'] == "❌ ERROR":
            print(f"  Error: {result['error']}")

    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)

    ok_count = sum(1 for r in results if r['status'] == "✅ OK")
    print(f"✅ {ok_count}/{len(FAMILIES)} checkpoints valides")

    if ok_count == len(FAMILIES):
        print("\n🎉 TOUS LES CHECKPOINTS SONT VALIDES!")

        print("\n📊 Tableau récapitulatif:")
        print(f"{'Famille':<15} {'Epoch':<8} {'NP Dice':<10} {'HV MSE':<10} {'NT Acc':<10}")
        print("-" * 63)

        for r in results:
            if r['status'] == "✅ OK":
                print(f"{r['family'].capitalize():<15} {r['epoch']:<8} {r['np_dice']:<10} {r['hv_mse']:<10} {r['nt_acc']:<10}")

    else:
        print("\n⚠️  Certains checkpoints sont manquants ou invalides.")

    return ok_count == len(FAMILIES)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
