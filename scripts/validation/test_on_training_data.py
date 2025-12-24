#!/usr/bin/env python3
"""
Test modèle sur SES PROPRES données d'entraînement.

Si le modèle échoue sur ses données d'entraînement (Dice << 0.95),
c'est que les poids ne sont PAS chargés ou que les features sont corrompues.

Expected results:
- NP Dice: ~0.95 (même que training)
- HV MSE: ~0.16 (même que training)
- NT Acc: ~0.90 (même que training)

If results are catastrophic (Dice < 0.5, HV energy = 0), then:
- Either checkpoint not loaded correctly
- Or features corrupted/wrong format
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader
from src.constants import DEFAULT_FAMILY_DATA_DIR

def main():
    parser = argparse.ArgumentParser(description="Test modèle sur données training")
    parser.add_argument("--family", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=10, help="Nombre échantillons à tester")
    parser.add_argument("--data_dir", default=DEFAULT_FAMILY_DATA_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 80)
    print(f"TEST SUR DONNÉES TRAINING - {args.family.upper()}")
    print("=" * 80)
    print("")

    # Charger features et targets
    print("Chargement données...")
    features_file = data_dir / f"{args.family}_features.npz"
    targets_file = data_dir / f"{args.family}_targets.npz"

    if not features_file.exists() or not targets_file.exists():
        print(f"❌ ERREUR: Fichiers manquants")
        return 1

    features_data = np.load(features_file)
    targets_data = np.load(targets_file)

    features = features_data['features']
    np_targets = targets_data['np_targets']
    hv_targets = targets_data['hv_targets']
    nt_targets = targets_data['nt_targets']

    print(f"  Features shape: {features.shape}")
    print(f"  Features dtype: {features.dtype}")
    print(f"  Features range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Features mean: {features.mean():.3f}")
    print("")

    # Charger modèle
    print(f"Chargement modèle: {args.checkpoint}")
    hovernet = ModelLoader.load_hovernet(args.checkpoint, device=args.device)
    hovernet.eval()
    print("")

    # Test sur échantillons
    print(f"Test sur {args.n_samples} échantillons...")

    dice_scores = []
    hv_mse_scores = []
    nt_acc_scores = []

    for i in range(min(args.n_samples, len(features))):
        # Features (B, 256, 1536)
        feat = torch.from_numpy(features[i:i+1]).to(args.device).float()

        print(f"\nÉchantillon {i}:")
        print(f"  Input shape: {feat.shape}")
        print(f"  Input dtype: {feat.dtype}")
        print(f"  Input range: [{feat.min().item():.3f}, {feat.max().item():.3f}]")

        # Prédiction
        with torch.no_grad():
            np_out, hv_out, nt_out = hovernet(feat)

        print(f"  Output NP shape: {np_out.shape}")
        print(f"  Output HV shape: {hv_out.shape}")
        print(f"  Output NT shape: {nt_out.shape}")

        # Convertir
        # NP utilise CrossEntropyLoss (2 canaux: background/foreground)
        # On prend le canal 1 (foreground) après softmax
        np_pred = torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]  # (224, 224) - foreground prob
        hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)
        nt_pred = torch.softmax(nt_out, dim=1).cpu().numpy()[0]  # (5, 224, 224)

        print(f"  NP pred range: [{np_pred.min():.3f}, {np_pred.max():.3f}]")
        print(f"  HV pred range: [{hv_pred.min():.3f}, {hv_pred.max():.3f}]")
        print(f"  HV magnitude: {np.sqrt(hv_pred[0]**2 + hv_pred[1]**2).max():.3f}")

        # GT (resize 256 → 224)
        from cv2 import resize, INTER_NEAREST
        np_gt = resize(np_targets[i], (224, 224), interpolation=INTER_NEAREST)
        hv_gt = np.stack([
            resize(hv_targets[i, 0], (224, 224)),
            resize(hv_targets[i, 1], (224, 224))
        ])
        nt_gt = resize(nt_targets[i], (224, 224), interpolation=INTER_NEAREST)

        # Calculer métriques
        np_pred_binary = (np_pred > 0.5).astype(np.float32)
        np_gt_binary = (np_gt > 0.5).astype(np.float32)

        intersection = (np_pred_binary * np_gt_binary).sum()
        union = np_pred_binary.sum() + np_gt_binary.sum()
        dice = 2 * intersection / union if union > 0 else 0

        # HV MSE (sur pixels de noyaux seulement)
        mask = np_gt_binary > 0.5
        if mask.sum() > 0:
            hv_mse = ((hv_pred - hv_gt)**2)[:, mask].mean()
        else:
            hv_mse = 0

        # NT Acc
        nt_pred_class = nt_pred.argmax(axis=0)
        nt_acc = (nt_pred_class == nt_gt).sum() / nt_gt.size

        dice_scores.append(dice)
        hv_mse_scores.append(hv_mse)
        nt_acc_scores.append(nt_acc)

        print(f"  → NP Dice: {dice:.4f}")
        print(f"  → HV MSE: {hv_mse:.4f}")
        print(f"  → NT Acc: {nt_acc:.4f}")

    # Résumé
    print("")
    print("=" * 80)
    print("RÉSULTATS MOYENS")
    print("=" * 80)
    print(f"NP Dice:  {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"HV MSE:   {np.mean(hv_mse_scores):.4f} ± {np.std(hv_mse_scores):.4f}")
    print(f"NT Acc:   {np.mean(nt_acc_scores):.4f} ± {np.std(nt_acc_scores):.4f}")
    print("")

    # Verdict
    dice_mean = np.mean(dice_scores)
    hv_mean = np.mean(hv_mse_scores)

    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    if dice_mean < 0.50:
        print("❌ CATASTROPHIQUE: Dice < 0.50")
        print("   → Modèle ne fonctionne PAS DU TOUT")
        print("   → Soit checkpoint mal chargé")
        print("   → Soit features corrompues")
    elif dice_mean < 0.85:
        print("⚠️  DÉGRADÉ: Dice < 0.85 (attendu ~0.95)")
        print("   → Modèle partiellement fonctionnel")
        print("   → Vérifier: features, normalisation, architecture")
    else:
        print("✅ BON: Dice ~0.95 comme attendu")
        if hv_mean > 0.25:
            print("⚠️  MAIS: HV MSE élevé (attendu ~0.16)")
        else:
            print("✅ HV MSE OK également")

    return 0

if __name__ == "__main__":
    sys.exit(main())
