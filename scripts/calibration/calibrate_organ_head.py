#!/usr/bin/env python3
"""
Calibration Temperature Scaling pour OrganHead.

Usage:
    python scripts/calibration/calibrate_organ_head.py
    python scripts/calibration/calibrate_organ_head.py --folds 0 1 2
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.calibration.temperature_scaling import (
    TemperatureScaler,
    compute_calibration_metrics,
)
from src.models.organ_head import OrganHead, PANNUKE_ORGANS


def main():
    parser = argparse.ArgumentParser(description="Calibrate OrganHead")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2],
                        help="Folds à utiliser")
    parser.add_argument("--features_dir", type=str,
                        default="data/cache/pannuke_features",
                        help="Répertoire des features")
    parser.add_argument("--data_dir", type=str, default="/home/amar/data/PanNuke",
                        help="Répertoire PanNuke")
    parser.add_argument("--checkpoint", type=str,
                        default="models/checkpoints/organ_head_best.pth",
                        help="Checkpoint OrganHead")
    parser.add_argument("--output", type=str,
                        default="models/checkpoints/organ_head_calibrated.pth",
                        help="Checkpoint calibré en sortie")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_dir = Path(args.features_dir)
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("CALIBRATION TEMPERATURE SCALING - OrganHead")
    print("=" * 60)

    # Charger OrganHead
    print("\n⏳ Chargement OrganHead...")
    organ_head = OrganHead(embed_dim=1536, n_organs=len(PANNUKE_ORGANS))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    organ_head.load_state_dict(checkpoint['model_state_dict'], strict=False)
    organ_head.eval().to(device)
    print(f"  ✓ OrganHead chargé")

    # Charger les données de validation (20% de chaque fold)
    print("\n⏳ Chargement des données de validation...")
    all_cls_tokens = []
    all_labels = []

    organ_to_idx = {name: idx for idx, name in enumerate(PANNUKE_ORGANS)}

    for fold in args.folds:
        # Features
        features_path = features_dir / f"fold{fold}_features.npz"
        data = np.load(features_path)
        if 'features' in data:
            features = data['features']
        elif 'layer_24' in data:
            features = data['layer_24']
        else:
            raise KeyError(f"Features non trouvées dans {features_path}")

        cls_tokens = features[:, 0, :]  # (N, 1536)

        # Labels
        types_path = data_dir / f"fold{fold}" / "types.npy"
        types = np.load(types_path)
        labels = np.array([organ_to_idx.get(str(t).strip(), 0) for t in types])

        # Split validation (20%)
        n = len(cls_tokens)
        val_size = int(n * 0.2)
        val_indices = np.arange(n - val_size, n)

        all_cls_tokens.append(cls_tokens[val_indices])
        all_labels.append(labels[val_indices])

        print(f"  Fold {fold}: {len(val_indices)} samples validation")

    cls_tokens = np.concatenate(all_cls_tokens, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"\n  Total: {len(labels)} samples de validation")

    # Extraire les logits
    print("\n⏳ Extraction des logits...")
    with torch.no_grad():
        cls_tensor = torch.from_numpy(cls_tokens).float().to(device)
        logits = organ_head(cls_tensor).cpu().numpy()

    print(f"  Logits shape: {logits.shape}")

    # Métriques avant calibration
    probs_before = softmax(logits, axis=-1)
    metrics_before = compute_calibration_metrics(probs_before, labels)

    print("\n" + "=" * 60)
    print("AVANT CALIBRATION")
    print("=" * 60)
    print(f"  Accuracy:        {metrics_before['accuracy']:.4f}")
    print(f"  Mean Confidence: {metrics_before['mean_confidence']:.4f}")
    print(f"  ECE:             {metrics_before['ECE']:.4f}")
    print(f"  MCE:             {metrics_before['MCE']:.4f}")
    print(f"  Brier Score:     {metrics_before['Brier']:.4f}")

    # Calibration
    print("\n⏳ Optimisation de la température...")
    scaler = TemperatureScaler()
    T = scaler.fit(logits, labels, method="nll")
    print(f"  ✓ Température optimale: {T:.4f}")

    # Métriques après calibration
    probs_after = scaler.predict_proba(logits)
    metrics_after = compute_calibration_metrics(probs_after, labels)

    print("\n" + "=" * 60)
    print("APRÈS CALIBRATION")
    print("=" * 60)
    print(f"  Accuracy:        {metrics_after['accuracy']:.4f}")
    print(f"  Mean Confidence: {metrics_after['mean_confidence']:.4f}")
    print(f"  ECE:             {metrics_after['ECE']:.4f} (↓ {metrics_before['ECE'] - metrics_after['ECE']:.4f})")
    print(f"  MCE:             {metrics_after['MCE']:.4f}")
    print(f"  Brier Score:     {metrics_after['Brier']:.4f}")

    # Comparaison confiance par organe
    print("\n" + "=" * 60)
    print("CONFIANCE PAR ORGANE")
    print("=" * 60)

    for organ_idx, organ_name in enumerate(PANNUKE_ORGANS):
        mask = labels == organ_idx
        if mask.sum() == 0:
            continue

        conf_before = probs_before[mask].max(axis=1).mean()
        conf_after = probs_after[mask].max(axis=1).mean()

        print(f"  {organ_name:15}: {conf_before:.1%} → {conf_after:.1%}")

    # Sauvegarder le checkpoint calibré
    print(f"\n💾 Sauvegarde checkpoint calibré...")
    calibrated_checkpoint = checkpoint.copy()
    calibrated_checkpoint['temperature'] = T
    calibrated_checkpoint['calibration_metrics'] = {
        'before': metrics_before,
        'after': metrics_after,
    }

    output_path = Path(args.output)
    torch.save(calibrated_checkpoint, output_path)
    print(f"  ✓ Sauvegardé: {output_path}")

    print("\n" + "=" * 60)
    print("✅ CALIBRATION TERMINÉE")
    print("=" * 60)
    print(f"\nPour utiliser la température en inférence:")
    print(f"  T = {T:.4f}")
    print(f"  calibrated_probs = softmax(logits / T)")


if __name__ == "__main__":
    main()
