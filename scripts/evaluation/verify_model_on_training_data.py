#!/usr/bin/env python3
"""
Vérification DIRECTE: Le modèle fonctionne-t-il sur ses propres données d'entraînement?

Ce script charge les features EXACTES utilisées pour l'entraînement
et vérifie que le modèle prédit correctement.

⚠️ DIAGNOSTIC: Si le modèle prédit 100% foreground même sur ses propres données,
   cela indique un problème avec le checkpoint ou les features.

Usage:
    python scripts/evaluation/verify_model_on_training_data.py \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import (
    get_family_features_path,
    get_family_targets_path,
    CURRENT_DATA_VERSION,
)
from src.models.hovernet_decoder import HoVerNetDecoder


def main():
    parser = argparse.ArgumentParser(description="Vérifie modèle sur données training")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint HoVer-Net")
    parser.add_argument("--family", default="epidermal", help="Famille")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--n_samples", type=int, default=10, help="Nombre d'échantillons")
    parser.add_argument("--verbose", action="store_true", help="Affiche détails logits")
    args = parser.parse_args()

    print("=" * 80)
    print("🔍 VÉRIFICATION: Modèle sur données d'entraînement")
    print("=" * 80)
    print(f"   Version données: {CURRENT_DATA_VERSION}")

    # ========================================================================
    # 1. Charger les mêmes features que l'entraînement
    # ========================================================================
    features_path = get_family_features_path(args.family)
    targets_path = get_family_targets_path(args.family)

    print(f"\n📂 Chargement features: {features_path}")
    features_data = np.load(features_path)

    if 'features' in features_data:
        features = features_data['features']
    elif 'layer_24' in features_data:
        features = features_data['layer_24']
    else:
        print(f"❌ Clés inattendues: {list(features_data.keys())}")
        return

    print(f"   Features shape: {features.shape}")

    print(f"\n📂 Chargement targets: {targets_path}")
    targets_data = np.load(targets_path)
    np_targets = targets_data['np_targets']
    print(f"   NP targets shape: {np_targets.shape}")

    # ========================================================================
    # 2. Charger le modèle
    # ========================================================================
    print(f"\n🔧 Chargement modèle: {args.checkpoint}")
    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Dice (sauvegardé): {checkpoint.get('val_dice', 'N/A')}")

    # ========================================================================
    # 3. Tester sur quelques échantillons
    # ========================================================================
    print(f"\n🧪 Test sur {args.n_samples} échantillons...")
    print(f"   Features shape: {features.shape}")
    print(f"   Targets shape: {np_targets.shape}")

    all_dice = []
    all_pred_fg = []
    all_target_fg = []
    all_logit_stats = []

    for i in range(min(args.n_samples, len(features))):
        # Features COMPLÈTES (261, 1536) - le décodeur gère l'extraction des patch tokens
        feat = torch.tensor(features[i:i+1]).to(args.device).float()

        # Target NP (256x256 ou 224x224?)
        np_target = np_targets[i]
        target_fg = (np_target > 0).sum()

        # Prédiction
        with torch.no_grad():
            np_out, hv_out, nt_out = hovernet(feat)

        # ========================================================================
        # DIAGNOSTIC: Analyse des logits bruts
        # ========================================================================
        np_logits = np_out[0].cpu().numpy()  # (2, H, W)
        logit_bg = np_logits[0]  # Logit background
        logit_fg = np_logits[1]  # Logit foreground

        logit_stats = {
            'bg_mean': logit_bg.mean(),
            'bg_min': logit_bg.min(),
            'bg_max': logit_bg.max(),
            'fg_mean': logit_fg.mean(),
            'fg_min': logit_fg.min(),
            'fg_max': logit_fg.max(),
            'diff_mean': (logit_fg - logit_bg).mean(),  # Si > 0 partout → prédit tout FG
        }
        all_logit_stats.append(logit_stats)

        # Conversion - méthode identique au training (argmax)
        pred_class = np_out.argmax(dim=1)[0].cpu().numpy()  # (H, W) - 0=bg, 1=fg
        pred_fg = (pred_class == 1).sum()

        # Alternative: softmax > 0.5 (devrait donner le même résultat)
        np_probs = torch.softmax(np_out, dim=1)  # (1, 2, H, W)
        pred_binary_soft = (np_probs[0, 1] > 0.5).cpu().numpy()

        # Vérifier cohérence argmax vs softmax
        if not np.array_equal(pred_class, pred_binary_soft.astype(int)):
            print(f"   ⚠️ INCOHÉRENCE argmax vs softmax!")

        # Dice avec resize si nécessaire
        import cv2
        if np_target.shape != pred_class.shape:
            np_target_resized = cv2.resize(
                np_target.astype(np.float32),
                (pred_class.shape[1], pred_class.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            target_binary = np_target_resized > 0
            target_fg_resized = target_binary.sum()
        else:
            target_binary = np_target > 0
            target_fg_resized = target_fg

        pred_binary = (pred_class == 1)
        intersection = (pred_binary & target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        dice = 2 * intersection / union if union > 0 else 1.0

        all_dice.append(dice)
        all_pred_fg.append(pred_fg)
        all_target_fg.append(target_fg_resized)

        if i < 3 or args.verbose:  # Print first 3 or all if verbose
            print(f"\n   Sample {i}:")
            print(f"     Target shape: {np_target.shape}, Pred shape: {pred_class.shape}")
            print(f"     Pred FG: {pred_fg} pixels ({100*pred_fg/pred_class.size:.1f}%)")
            print(f"     Target FG: {target_fg_resized} pixels ({100*target_fg_resized/target_binary.size:.1f}%)")
            print(f"     Dice: {dice:.4f}")
            print(f"     Logits BG: mean={logit_stats['bg_mean']:.2f}, min={logit_stats['bg_min']:.2f}, max={logit_stats['bg_max']:.2f}")
            print(f"     Logits FG: mean={logit_stats['fg_mean']:.2f}, min={logit_stats['fg_min']:.2f}, max={logit_stats['fg_max']:.2f}")
            print(f"     Diff (FG-BG): mean={logit_stats['diff_mean']:.2f}")

    # ========================================================================
    # 4. Résumé
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ")
    print("=" * 80)

    mean_dice = np.mean(all_dice)
    mean_pred_fg = np.mean(all_pred_fg)
    mean_target_fg = np.mean(all_target_fg)

    # Statistiques des logits agrégées
    mean_bg_logit = np.mean([s['bg_mean'] for s in all_logit_stats])
    mean_fg_logit = np.mean([s['fg_mean'] for s in all_logit_stats])
    mean_diff_logit = np.mean([s['diff_mean'] for s in all_logit_stats])

    print(f"""
    Dice moyen:           {mean_dice:.4f}
    Pred FG moyen:        {mean_pred_fg:.0f} pixels ({100*mean_pred_fg/50176:.1f}%)
    Target FG moyen:      {mean_target_fg:.0f} pixels ({100*mean_target_fg/50176:.1f}%)
    Ratio Pred/Target:    {mean_pred_fg/max(mean_target_fg, 1):.2f}x

    📈 ANALYSE LOGITS:
    Logit BG moyen:       {mean_bg_logit:.2f}
    Logit FG moyen:       {mean_fg_logit:.2f}
    Diff (FG-BG) moyen:   {mean_diff_logit:.2f}
    """)

    # Diagnostic basé sur les logits
    if mean_diff_logit > 5:
        print("🔴 DIAGNOSTIC: Logits FG >> BG partout!")
        print("   Le modèle a un BIAIS FORT vers foreground.")
        print("   Causes possibles:")
        print("   1. Checkpoint corrompu ou d'un entraînement raté")
        print("   2. Les features pendant training étaient différentes")
        print("   3. Le modèle n'a jamais appris correctement")
    elif mean_diff_logit < -5:
        print("🔴 DIAGNOSTIC: Logits BG >> FG partout!")
        print("   Le modèle prédit tout comme background.")
    elif abs(mean_diff_logit) < 0.1 and mean_dice < 0.5:
        print("🟡 DIAGNOSTIC: Logits équilibrés mais Dice faible")
        print("   Le modèle génère des prédictions semi-aléatoires.")

    if mean_dice > 0.90:
        print("✅ Le modèle fonctionne correctement sur ses données d'entraînement!")
    elif mean_dice > 0.50:
        print("⚠️ Le modèle a des performances moyennes - vérifier les features")
    else:
        print("❌ Le modèle ne fonctionne PAS sur ses propres données!")
        print("   → Les features utilisées pour l'inférence sont probablement")
        print("     DIFFÉRENTES de celles utilisées pour l'entraînement.")

    if mean_pred_fg / max(mean_target_fg, 1) > 5:
        print("\n🔴 ALERTE: Le modèle prédit BEAUCOUP trop de pixels comme foreground!")
        print("   Possible causes:")
        print("   1. Features corrompues pendant extraction")
        print("   2. Modèle entraîné sur des features différentes")
        print("   3. Bug dans le preprocessing")
        print("\n   💡 SOLUTION RECOMMANDÉE:")
        print("   1. Vérifier que le checkpoint correspond à la bonne famille")
        print("   2. Ré-extraire les features avec extract_features_from_v9.py")
        print("   3. Ré-entraîner avec train_hovernet_family.py")
        print("   4. Utiliser la config de test: fold0, 20 epochs")


if __name__ == "__main__":
    main()
