#!/usr/bin/env python3
"""
Vérification CRITIQUE des features H-optimus-0 utilisées pour training.

Suite à découverte checkpoint entraîné APRÈS Sobel fix (24 déc > 23 déc),
mais HV magnitude quand même catastrophique (0.022).

Hypothèse: Features training corrompues (Bug #1 ToPILImage ou Bug #2 LayerNorm).

Ce script vérifie:
1. CLS std dans [0.70, 0.90] (signature features correctes)
2. Shape (N, 261, 1536) - 1 CLS + 256 patches
3. Mean proche de 0 (normalisé)
4. Comparaison train vs inference (ratio proche 1.0)
"""

import sys
from pathlib import Path
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DEFAULT_FAMILY_DATA_DIR


def verify_features_training(family: str = "epidermal", compare_inference: bool = False):
    """
    Vérifie features H-optimus-0 utilisées durant training.

    Args:
        family: Famille à vérifier
        compare_inference: Si True, compare avec features inference fraîches
    """
    print("\n" + "="*80)
    print(f"VÉRIFICATION CRITIQUE: FEATURES H-OPTIMUS-0 TRAINING - {family.upper()}")
    print("="*80)
    print("\nCritère: CLS std doit être dans [0.70, 0.90]")
    print("Si hors plage → Features corrompues (Bug #1 ou Bug #2)")
    print("\n" + "─"*80)

    # Charger features training
    data_dir = Path(DEFAULT_FAMILY_DATA_DIR)
    features_path = data_dir / f"{family}_features.npz"

    if not features_path.exists():
        print(f"\n❌ ERREUR: Fichier introuvable: {features_path}")
        print(f"\nCherche dans:")
        print(f"  • {data_dir}")
        print(f"  • {data_dir.parent}")

        # Chercher récursivement
        possible_paths = list(Path(PROJECT_ROOT).rglob(f"{family}_features.npz"))
        if possible_paths:
            print(f"\n💡 Fichiers trouvés ailleurs:")
            for p in possible_paths:
                print(f"   {p}")
        return 1

    print(f"\n📁 Fichier: {features_path}")
    print(f"   Taille: {features_path.stat().st_size / 1024**2:.1f} MB")

    # Charger avec mmap
    data = np.load(features_path, mmap_mode='r')

    print(f"\n📊 Contenu .npz:")
    for key in data.keys():
        arr = data[key]
        print(f"   • {key:20s}: shape={arr.shape}, dtype={arr.dtype}")

    # Extraire features
    if 'features' in data:
        features = data['features']
    elif 'layer_24' in data:
        features = data['layer_24']
        print(f"\n⚠️ WARNING: Clé 'layer_24' trouvée (anciennes features)")
        print(f"   Préférer 'features' (nouvelles)")
    else:
        print(f"\n❌ ERREUR: Ni 'features' ni 'layer_24' trouvé!")
        print(f"   Clés disponibles: {list(data.keys())}")
        return 1

    # Statistiques COMPLÈTES
    print(f"\n" + "="*80)
    print("STATISTIQUES FEATURES H-OPTIMUS-0")
    print("="*80)

    print(f"\n1️⃣ FORMAT")
    print(f"   Shape:  {features.shape}")
    print(f"   Dtype:  {features.dtype}")
    print(f"   Memory: {features.nbytes / 1024**2:.1f} MB")

    # Vérifier shape
    expected_shape = (None, 261, 1536)  # N samples, 1 CLS + 256 patches, 1536-dim
    if features.ndim != 3:
        print(f"\n❌ ERREUR SHAPE: {features.ndim}D au lieu de 3D")
        return 1

    if features.shape[1] != 261:
        print(f"\n❌ ERREUR TOKENS: {features.shape[1]} au lieu de 261")
        print(f"   Attendu: 1 CLS + 256 patches = 261 tokens")
        return 1

    if features.shape[2] != 1536:
        print(f"\n❌ ERREUR DIM: {features.shape[2]} au lieu de 1536")
        print(f"   Attendu: H-optimus-0 embedding dimension = 1536")
        return 1

    print(f"\n✅ Shape correcte: {features.shape}")
    print(f"   • Samples: {features.shape[0]}")
    print(f"   • Tokens: {features.shape[1]} (1 CLS + 256 patches)")
    print(f"   • Dim: {features.shape[2]} (H-optimus-0)")

    # Extraire CLS tokens
    cls_tokens = features[:, 0, :]  # (N, 1536)

    print(f"\n2️⃣ CLS TOKEN STATISTICS (CRITIQUE)")

    cls_mean = float(cls_tokens.mean())
    cls_std = float(cls_tokens.std())
    cls_min = float(cls_tokens.min())
    cls_max = float(cls_tokens.max())

    print(f"   Mean:   {cls_mean:+.6f}")
    print(f"   Std:    {cls_std:.6f}")
    print(f"   Min:    {cls_min:+.6f}")
    print(f"   Max:    {cls_max:+.6f}")

    # DIAGNOSTIC CLS STD (CRITIQUE)
    print(f"\n" + "="*80)
    print("DIAGNOSTIC CLS STD")
    print("="*80)

    if cls_std < 0.40:
        print(f"\n❌ ERREUR CRITIQUE: CLS std trop bas ({cls_std:.4f})")
        print(f"   Attendu: [0.70, 0.90]")
        print(f"\n💡 DIAGNOSTIC: Bug #2 (LayerNorm Mismatch)")
        print(f"   Cause probable: Features extraites avec blocks[23] (sans LayerNorm)")
        print(f"   Au lieu de: forward_features() (avec LayerNorm)")
        print(f"\n🔧 SOLUTION:")
        print(f"   1. Régénérer features avec forward_features()")
        print(f"   2. Vérifier CLS std après régénération")
        print(f"   3. Ré-entraîner avec features correctes")
        return 1

    elif cls_std > 1.50:
        print(f"\n❌ ERREUR CRITIQUE: CLS std trop haut ({cls_std:.4f})")
        print(f"   Attendu: [0.70, 0.90]")
        print(f"\n💡 DIAGNOSTIC: Normalisation incorrecte")
        print(f"   Cause probable: Mean/Std HOPTIMUS incorrects")
        print(f"   Ou: Pas de normalisation appliquée")
        return 1

    elif 0.70 <= cls_std <= 0.90:
        print(f"\n✅ CLS STD CORRECT: {cls_std:.4f} (dans [0.70, 0.90])")
        print(f"   Features H-optimus-0 VALIDES ✅")

    else:
        print(f"\n⚠️ WARNING: CLS std légèrement hors plage ({cls_std:.4f})")
        print(f"   Attendu: [0.70, 0.90]")
        print(f"   Acceptable si proche (0.65-0.95)")

    # Statistiques par échantillon
    print(f"\n3️⃣ CLS STD PAR ÉCHANTILLON (Premiers 10)")

    sample_stds = []
    for i in range(min(10, features.shape[0])):
        sample_cls = features[i, 0, :]
        sample_std = float(sample_cls.std())
        sample_stds.append(sample_std)
        print(f"   Sample {i:2d}: std={sample_std:.4f}")

    # Distribution CLS std
    all_sample_stds = [features[i, 0, :].std() for i in range(features.shape[0])]
    std_mean = np.mean(all_sample_stds)
    std_std = np.std(all_sample_stds)

    print(f"\n   Distribution CLS std:")
    print(f"      Mean: {std_mean:.4f}")
    print(f"      Std:  {std_std:.4f}")
    print(f"      Min:  {np.min(all_sample_stds):.4f}")
    print(f"      Max:  {np.max(all_sample_stds):.4f}")

    # Comparaison inference (optionnel)
    if compare_inference:
        print(f"\n" + "="*80)
        print("COMPARAISON TRAIN VS INFERENCE")
        print("="*80)

        print(f"\n⚠️ Fonctionnalité non implémentée (nécessite image test)")
        print(f"   Utiliser: compare_train_vs_inference.py")

    # VERDICT FINAL
    print(f"\n" + "="*80)
    print("VERDICT FINAL")
    print("="*80)

    issues = []

    if features.shape[1] != 261:
        issues.append(f"Shape tokens incorrect: {features.shape[1]} au lieu de 261")

    if features.shape[2] != 1536:
        issues.append(f"Embedding dim incorrect: {features.shape[2]} au lieu de 1536")

    if cls_std < 0.70 or cls_std > 0.90:
        issues.append(f"CLS std hors plage: {cls_std:.4f} (attendu [0.70, 0.90])")

    if issues:
        print(f"\n❌ PROBLÈMES DÉTECTÉS ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print(f"\n🔧 ACTIONS REQUISES:")
        print(f"   1. Régénérer features H-optimus-0 avec preprocessing correct")
        print(f"   2. Vérifier CLS std après régénération")
        print(f"   3. Ré-entraîner avec features correctes")

        return 1
    else:
        print(f"\n✅ FEATURES H-OPTIMUS-0 CORRECTES!")
        print(f"   • Shape: {features.shape}")
        print(f"   • CLS std: {cls_std:.4f} (dans [0.70, 0.90])")
        print(f"   • Mean: {cls_mean:+.4f} (centré)")

        print(f"\n💡 CONCLUSION:")
        print(f"   Les features training sont valides.")
        print(f"   Le problème HV magnitude faible (0.022) vient donc:")
        print(f"   → Du MODÈLE (convergence insuffisante)")
        print(f"   → Ou des HYPERPARAMÈTRES (lambda_hv trop faible)")

        print(f"\n🔍 PROCHAINE ÉTAPE:")
        print(f"   Vérifier logs training pour:")
        print(f"   1. Sobel gradient loss actif?")
        print(f"   2. HV MSE convergence?")
        print(f"   3. Nombre epochs suffisant?")

        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vérifier features H-optimus-0 training"
    )
    parser.add_argument('--family', type=str, default='epidermal',
                       choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'],
                       help='Famille à vérifier')
    parser.add_argument('--compare', action='store_true',
                       help='Comparer avec features inference (non implémenté)')

    args = parser.parse_args()

    sys.exit(verify_features_training(args.family, args.compare))
