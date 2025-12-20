#!/usr/bin/env python3
"""
Test complet de l'architecture Optimus-Gate.

Charge les checkpoints pré-entraînés et teste sur des données réelles.

Usage:
    python scripts/validation/test_optimus_gate.py \
        --data_dir /home/amar/data/PanNuke \
        --fold 0 \
        --n_samples 10
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.optimus_gate import OptimusGate
from src.models.organ_head import PANNUKE_ORGANS


def load_test_data(data_dir: Path, fold: int, n_samples: int = 10):
    """Charge quelques échantillons de test."""
    # Features pré-extraites
    features_path = PROJECT_ROOT / "data" / "cache" / "pannuke_features" / f"fold{fold}_features.npz"

    if not features_path.exists():
        raise FileNotFoundError(f"Features non trouvées: {features_path}")

    data = np.load(features_path)
    features = data['layer_24'] if 'layer_24' in data else data['layer_23']

    # Types (organes)
    types_path = data_dir / f"fold{fold}" / "types.npy"
    types = np.load(types_path) if types_path.exists() else None

    # Limiter le nombre d'échantillons
    indices = np.random.choice(len(features), min(n_samples, len(features)), replace=False)

    return features[indices], types[indices] if types is not None else None, indices


def main():
    parser = argparse.ArgumentParser(description="Test Optimus-Gate")
    parser.add_argument("--data_dir", type=str, default="/home/amar/data/PanNuke")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--hovernet_ckpt", type=str,
                       default="models/checkpoints/hovernet_best.pth")
    parser.add_argument("--organ_head_ckpt", type=str,
                       default="models/checkpoints/organ_head_best.pth")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Vérifier les checkpoints
    hovernet_path = PROJECT_ROOT / args.hovernet_ckpt
    organ_path = PROJECT_ROOT / args.organ_head_ckpt

    if not hovernet_path.exists():
        print(f"❌ HoVer-Net checkpoint non trouvé: {hovernet_path}")
        return
    if not organ_path.exists():
        print(f"❌ OrganHead checkpoint non trouvé: {organ_path}")
        return

    # Charger le modèle
    print("\n" + "=" * 60)
    print("🚀 CHARGEMENT OPTIMUS-GATE")
    print("=" * 60)

    model = OptimusGate.from_pretrained(
        hovernet_path=str(hovernet_path),
        organ_head_path=str(organ_path),
        device=device,
    )

    # Charger les données de test
    print("\n" + "=" * 60)
    print("📦 CHARGEMENT DONNÉES DE TEST")
    print("=" * 60)

    features, types, indices = load_test_data(
        Path(args.data_dir), args.fold, args.n_samples
    )
    print(f"  {len(features)} échantillons chargés")

    # Tester
    print("\n" + "=" * 60)
    print("🔬 TEST INFÉRENCE")
    print("=" * 60)

    correct_organs = 0
    total_cells = 0

    for i, (feat, true_organ) in enumerate(zip(features, types)):
        feat_tensor = torch.from_numpy(feat).float().to(device)

        with torch.no_grad():
            result = model.predict(feat_tensor)

        # Vérifier la prédiction d'organe
        pred_organ = result.organ.organ_name
        true_organ_str = str(true_organ).strip()
        is_correct = pred_organ.lower() in true_organ_str.lower() or true_organ_str.lower() in pred_organ.lower()

        if is_correct:
            correct_organs += 1

        total_cells += result.n_cells

        # Afficher
        status = "✓" if is_correct else "✗"
        ood_status = "🚫 OOD" if result.is_ood else "✅"

        print(f"\n  [{i+1}/{len(features)}] {status}")
        print(f"    Organe prédit: {pred_organ} ({result.organ.confidence:.1%})")
        print(f"    Organe réel:   {true_organ_str}")
        print(f"    Cellules:      {result.n_cells}")
        print(f"    Confiance:     {result.confidence_level.value} {ood_status}")

    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"  Accuracy Organe: {correct_organs}/{len(features)} ({correct_organs/len(features):.1%})")
    print(f"  Cellules totales: {total_cells}")
    print(f"  Cellules moyennes/image: {total_cells/len(features):.1f}")

    # Test avec un rapport complet
    print("\n" + "=" * 60)
    print("📋 EXEMPLE DE RAPPORT COMPLET")
    print("=" * 60)

    feat_tensor = torch.from_numpy(features[0]).float().to(device)
    with torch.no_grad():
        result = model.predict(feat_tensor)

    print(model.generate_report(result))

    print("\n✅ Test terminé!")


if __name__ == "__main__":
    main()
