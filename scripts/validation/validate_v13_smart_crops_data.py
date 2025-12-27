#!/usr/bin/env python3
"""
Validation complète des données V13 Smart Crops avant entraînement.

Vérifie:
1. Features RGB (train + val)
2. Targets (train + val)
3. Cohérence splits (pas de data leakage)
4. Shapes, dtypes, ranges
5. Métadonnées (source_image_ids, crop_positions)

Usage:
    python scripts/validation/validate_v13_smart_crops_data.py --family epidermal
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    HOPTIMUS_CLS_STD_MIN,
    HOPTIMUS_CLS_STD_MAX,
    PANNUKE_IMAGE_SIZE,
    HOVERNET_OUTPUT_SIZE,
)


def validate_features(features_file: Path, split: str, family: str) -> bool:
    """Valide un fichier de features RGB."""

    print(f"\n{'='*80}")
    print(f"  VALIDATION FEATURES RGB — {split.upper()}")
    print(f"{'='*80}")
    print(f"Fichier: {features_file.name}")

    if not features_file.exists():
        print(f"❌ ERREUR: Fichier introuvable")
        return False

    # Taille fichier
    file_size_gb = features_file.stat().st_size / 1e9
    print(f"Taille: {file_size_gb:.2f} GB")
    print()

    # Charger données
    data = np.load(features_file)

    # Vérifier clés obligatoires
    required_keys = ['features', 'source_image_ids', 'crop_positions', 'fold_ids']
    missing_keys = [k for k in required_keys if k not in data.files]

    if missing_keys:
        print(f"❌ ERREUR: Clés manquantes: {missing_keys}")
        print(f"   Clés disponibles: {list(data.files)}")
        return False

    # Extraire arrays
    features = data['features']
    source_ids = data['source_image_ids']
    crop_positions = data['crop_positions']
    fold_ids = data['fold_ids']

    print(f"📊 Dimensions:")
    print(f"  Features:        {features.shape} ({features.dtype})")
    print(f"  Source IDs:      {source_ids.shape}")
    print(f"  Crop positions:  {crop_positions.shape}")
    print(f"  Fold IDs:        {fold_ids.shape}")
    print()

    # Vérifications shapes
    issues = []

    n_crops = len(features)

    if features.ndim != 3:
        issues.append(f"Features ndim={features.ndim} (attendu: 3)")

    if features.shape[1] != 261:
        issues.append(f"Features shape[1]={features.shape[1]} (attendu: 261 = CLS + 4 Registers + 256 Patches)")

    if features.shape[2] != 1536:
        issues.append(f"Features shape[2]={features.shape[2]} (attendu: 1536 = H-optimus-0 embed_dim)")

    if len(source_ids) != n_crops:
        issues.append(f"len(source_ids)={len(source_ids)} != n_crops={n_crops}")

    if len(crop_positions) != n_crops:
        issues.append(f"len(crop_positions)={len(crop_positions)} != n_crops={n_crops}")

    if len(fold_ids) != n_crops:
        issues.append(f"len(fold_ids)={len(fold_ids)} != n_crops={n_crops}")

    # Vérification CLS std
    cls_tokens = features[:, 0, :]  # (N_crops, 1536)
    cls_std = cls_tokens.std()
    cls_mean = cls_tokens.mean()

    print(f"📈 CLS Token Statistiques:")
    print(f"  Std:  {cls_std:.4f} (attendu: [{HOPTIMUS_CLS_STD_MIN}, {HOPTIMUS_CLS_STD_MAX}])")
    print(f"  Mean: {cls_mean:.4f}")

    if cls_std < 0.40:
        issues.append(f"❌ CLS std={cls_std:.4f} < 0.40 → Features CORROMPUES (LayerNorm manquant)!")
    elif cls_std < HOPTIMUS_CLS_STD_MIN:
        issues.append(f"⚠️  CLS std={cls_std:.4f} < {HOPTIMUS_CLS_STD_MIN} → Features suspectes")
    elif cls_std > HOPTIMUS_CLS_STD_MAX:
        issues.append(f"⚠️  CLS std={cls_std:.4f} > {HOPTIMUS_CLS_STD_MAX} → Features anormalement élevées")
    else:
        print(f"  ✅ CLS std dans la plage attendue")

    print()

    # Vérification NaN/Inf
    if np.isnan(features).any():
        issues.append("Features contiennent des NaN!")

    if np.isinf(features).any():
        issues.append("Features contiennent des Inf!")

    # Statistiques crops
    unique_sources = np.unique(source_ids)
    unique_crops = np.unique(crop_positions)

    print(f"📊 Statistiques Crops:")
    print(f"  Total crops:           {n_crops}")
    print(f"  Images sources uniques: {len(unique_sources)}")
    print(f"  Positions crops:        {list(unique_crops)}")
    print(f"  Amplification moyenne:  {n_crops / len(unique_sources):.1f}× (attendu: ~5×)")
    print()

    # Vérification amplification attendue (5 crops par image)
    expected_amplification = 5.0
    actual_amplification = n_crops / len(unique_sources)

    if abs(actual_amplification - expected_amplification) > 0.5:
        issues.append(f"Amplification {actual_amplification:.1f}× != attendu {expected_amplification}×")

    # Verdict
    if issues:
        print(f"❌ INVALIDE ({len(issues)} problèmes):")
        for issue in issues:
            print(f"  → {issue}")
        return False
    else:
        print(f"✅ VALIDE")
        return True


def validate_targets(targets_file: Path, split: str, family: str) -> bool:
    """Valide un fichier de targets."""

    print(f"\n{'='*80}")
    print(f"  VALIDATION TARGETS — {split.upper()}")
    print(f"{'='*80}")
    print(f"Fichier: {targets_file.name}")

    if not targets_file.exists():
        print(f"❌ ERREUR: Fichier introuvable")
        return False

    # Charger données
    data = np.load(targets_file)

    # Vérifier clés obligatoires
    required_keys = ['images', 'np_targets', 'hv_targets', 'nt_targets', 'source_image_ids']
    missing_keys = [k for k in required_keys if k not in data.files]

    if missing_keys:
        print(f"❌ ERREUR: Clés manquantes: {missing_keys}")
        print(f"   Clés disponibles: {list(data.files)}")
        return False

    # Extraire arrays
    images = data['images']
    np_targets = data['np_targets']
    hv_targets = data['hv_targets']
    nt_targets = data['nt_targets']
    source_ids = data['source_image_ids']

    print(f"📊 Dimensions:")
    print(f"  Images:      {images.shape} ({images.dtype})")
    print(f"  NP targets:  {np_targets.shape} ({np_targets.dtype})")
    print(f"  HV targets:  {hv_targets.shape} ({hv_targets.dtype})")
    print(f"  NT targets:  {nt_targets.shape} ({nt_targets.dtype})")
    print()

    # Vérifications
    issues = []

    n_crops = len(images)

    # Shapes
    if images.shape[1:] != (224, 224, 3):
        issues.append(f"Images shape={images.shape[1:]} (attendu: (224, 224, 3))")

    if np_targets.shape[1:] != (224, 224):
        issues.append(f"NP shape={np_targets.shape[1:]} (attendu: (224, 224))")

    if hv_targets.shape[1:] != (2, 224, 224):
        issues.append(f"HV shape={hv_targets.shape[1:]} (attendu: (2, 224, 224))")

    if nt_targets.shape[1:] != (224, 224):
        issues.append(f"NT shape={nt_targets.shape[1:]} (attendu: (224, 224))")

    # Dtypes
    if images.dtype != np.uint8:
        issues.append(f"Images dtype={images.dtype} (attendu: uint8)")

    if np_targets.dtype != np.float32:
        issues.append(f"NP dtype={np_targets.dtype} (attendu: float32)")

    if hv_targets.dtype != np.float32:
        issues.append(f"HV dtype={hv_targets.dtype} (attendu: float32) ← CRITIQUE Bug #3")

    if nt_targets.dtype != np.int64:
        issues.append(f"NT dtype={nt_targets.dtype} (attendu: int64)")

    # Ranges
    print(f"📈 Ranges:")
    print(f"  Images:  [{images.min()}, {images.max()}] (attendu: [0, 255])")
    print(f"  NP:      [{np_targets.min():.4f}, {np_targets.max():.4f}] (attendu: [0, 1])")
    print(f"  HV:      [{hv_targets.min():.4f}, {hv_targets.max():.4f}] (attendu: [-1, 1])")
    print(f"  NT:      [{nt_targets.min()}, {nt_targets.max()}] (attendu: [0, 4])")
    print()

    # Vérification ranges HV (CRITIQUE Bug #3)
    hv_min, hv_max = hv_targets.min(), hv_targets.max()

    if not (-1.01 <= hv_min <= -0.99 and 0.99 <= hv_max <= 1.01):
        if hv_min < -10 or hv_max > 10:
            issues.append(f"❌ HV range [{hv_min:.1f}, {hv_max:.1f}] → BUG #3 (int8 au lieu de float32)!")
        else:
            issues.append(f"⚠️  HV range [{hv_min:.4f}, {hv_max:.4f}] pas exactement [-1, 1]")
    else:
        print(f"  ✅ HV range correct (float32 [-1, 1])")

    # Cohérence nombre de samples
    if not (len(np_targets) == len(hv_targets) == len(nt_targets) == len(source_ids) == n_crops):
        issues.append(f"Nombre de samples incohérent entre arrays")

    # Verdict
    if issues:
        print(f"❌ INVALIDE ({len(issues)} problèmes):")
        for issue in issues:
            print(f"  → {issue}")
        return False
    else:
        print(f"✅ VALIDE")
        return True


def check_data_leakage(
    train_targets_file: Path,
    val_targets_file: Path,
    train_features_file: Path,
    val_features_file: Path
) -> bool:
    """Vérifie qu'il n'y a pas de data leakage entre train et val."""

    print(f"\n{'='*80}")
    print(f"  VÉRIFICATION DATA LEAKAGE")
    print(f"{'='*80}")

    # Charger source_image_ids
    train_targets = np.load(train_targets_file)
    val_targets = np.load(val_targets_file)
    train_features = np.load(train_features_file)
    val_features = np.load(val_features_file)

    train_sources_targets = set(train_targets['source_image_ids'])
    val_sources_targets = set(val_targets['source_image_ids'])
    train_sources_features = set(train_features['source_image_ids'])
    val_sources_features = set(val_features['source_image_ids'])

    print(f"📊 Images sources:")
    print(f"  Train targets:   {len(train_sources_targets)} uniques")
    print(f"  Val targets:     {len(val_sources_targets)} uniques")
    print(f"  Train features:  {len(train_sources_features)} uniques")
    print(f"  Val features:    {len(val_sources_features)} uniques")
    print()

    issues = []

    # Vérification cohérence targets vs features
    if train_sources_targets != train_sources_features:
        issues.append("Source IDs train différents entre targets et features!")

    if val_sources_targets != val_sources_features:
        issues.append("Source IDs val différents entre targets et features!")

    # Vérification data leakage
    overlap = train_sources_targets & val_sources_targets

    if overlap:
        issues.append(f"❌ DATA LEAKAGE: {len(overlap)} images partagées entre train et val!")
        print(f"  Images en commun: {sorted(list(overlap))[:10]}{'...' if len(overlap) > 10 else ''}")
    else:
        print(f"  ✅ Aucune image partagée entre train et val (split-first-then-rotate OK)")

    # Vérification ratio split
    total_sources = len(train_sources_targets) + len(val_sources_targets)
    train_ratio = len(train_sources_targets) / total_sources
    val_ratio = len(val_sources_targets) / total_sources

    print(f"\n📊 Ratio split:")
    print(f"  Train: {train_ratio*100:.1f}%")
    print(f"  Val:   {val_ratio*100:.1f}%")

    if not (0.75 <= train_ratio <= 0.85):
        issues.append(f"Train ratio {train_ratio*100:.1f}% hors plage attendue [75%, 85%]")

    print()

    # Verdict
    if issues:
        print(f"❌ INVALIDE ({len(issues)} problèmes):")
        for issue in issues:
            print(f"  → {issue}")
        return False
    else:
        print(f"✅ VALIDE - Aucun data leakage détecté")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Validation complète données V13 Smart Crops avant entraînement"
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"],
        help="Famille à valider"
    )
    parser.add_argument(
        "--data_dir",
        default="data/family_data_v13_smart_crops",
        help="Répertoire des targets"
    )
    parser.add_argument(
        "--features_dir",
        default="data/cache/family_data",
        help="Répertoire des features"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    features_dir = Path(args.features_dir)

    print(f"\n{'='*80}")
    print(f"  VALIDATION V13 SMART CROPS — {args.family.upper()}")
    print(f"{'='*80}")
    print()

    # Chemins fichiers
    train_targets = data_dir / f"{args.family}_train_v13_smart_crops.npz"
    val_targets = data_dir / f"{args.family}_val_v13_smart_crops.npz"
    train_features = features_dir / f"{args.family}_rgb_features_v13_smart_crops_train.npz"
    val_features = features_dir / f"{args.family}_rgb_features_v13_smart_crops_val.npz"

    # Validation par étape
    results = {}

    # 1. Features train
    results['train_features'] = validate_features(train_features, "train", args.family)

    # 2. Features val (si train OK)
    if val_features.exists():
        results['val_features'] = validate_features(val_features, "val", args.family)
    else:
        print(f"\n⚠️  Val features pas encore extraites: {val_features.name}")
        print(f"   Commande: python scripts/preprocessing/extract_features_v13_smart_crops.py --family {args.family} --split val")
        results['val_features'] = None

    # 3. Targets train
    results['train_targets'] = validate_targets(train_targets, "train", args.family)

    # 4. Targets val
    results['val_targets'] = validate_targets(val_targets, "val", args.family)

    # 5. Data leakage (si tout existe)
    if all([train_targets.exists(), val_targets.exists(), train_features.exists(), val_features.exists()]):
        results['data_leakage'] = check_data_leakage(
            train_targets, val_targets, train_features, val_features
        )
    else:
        print(f"\n⚠️  Vérification data leakage sautée (fichiers manquants)")
        results['data_leakage'] = None

    # Verdict final
    print(f"\n{'='*80}")
    print(f"  VERDICT FINAL")
    print(f"{'='*80}")
    print()

    all_valid = all(r for r in results.values() if r is not None)

    if all_valid:
        print(f"✅ TOUTES LES VALIDATIONS PASSENT")
        print()
        print(f"Données prêtes pour l'entraînement:")
        print(f"  → Train features: {train_features.name}")
        if results['val_features']:
            print(f"  → Val features:   {val_features.name}")
        print(f"  → Train targets:  {train_targets.name}")
        print(f"  → Val targets:    {val_targets.name}")
        print()
        print(f"Prochaine étape:")
        if not results['val_features']:
            print(f"  1. Extraire val features:")
            print(f"     python scripts/preprocessing/extract_features_v13_smart_crops.py \\")
            print(f"         --family {args.family} --split val")
            print()
            print(f"  2. Puis lancer entraînement:")
        else:
            print(f"  Lancer entraînement:")
        print(f"     python scripts/training/train_hovernet_family_v13_smart_crops.py \\")
        print(f"         --family {args.family} --epochs 30")
        print()
        return 0
    else:
        print(f"❌ CERTAINES VALIDATIONS ONT ÉCHOUÉ")
        print()
        print(f"Résumé:")
        for check, result in results.items():
            if result is None:
                status = "⏭️  Sauté"
            elif result:
                status = "✅ OK"
            else:
                status = "❌ ÉCHEC"
            print(f"  {check:20s} {status}")
        print()
        print(f"⚠️  Corriger les problèmes avant de lancer l'entraînement")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
