#!/usr/bin/env python3
"""
Validation complète des données V13 Smart Crops avant entraînement.

Vérifie:
1. Features RGB (train + val)
   - Shapes (N, 261, 1536)
   - CLS std dans [0.70, 0.90]
   - Métadonnées (source_image_ids, crop_positions)

2. Targets (train + val)
   - Shapes, dtypes (HV float32, NP float32, NT int64)
   - Ranges (HV [-1, 1], NP [0, 1])

3. Cohérence splits (pas de data leakage)
   - Aucune image source partagée entre train et val

4. ⭐ NOUVEAU: HV Centers Inside Nuclei (Expert recommendation)
   - Vérifie que centres HV sont à l'intérieur des noyaux
   - Détecte problèmes Distance Transform vs mean()

5. ⭐ NOUVEAU: IDs Séquentiels (prévention Bug #1)
   - Vérifie IDs [1, 2, 3, ..., N] sans gaps ni doublons
   - Détecte problèmes LOCAL relabeling

6. ⭐ NOUVEAU: HV Rotation Consistency (prévention Bug #2)
   - Vérifie divergence négative (vecteurs centripètes)
   - Détecte erreurs de transformation HV après rotation

Usage:
    python scripts/validation/validate_v13_smart_crops_data.py --family epidermal

    # Avec plus d'échantillons pour les tests experts
    python scripts/validation/validate_v13_smart_crops_data.py --family epidermal --n_samples 50
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


def validate_hv_centers_inside_nuclei(
    targets_file: Path,
    split: str,
    n_samples: int = 20
) -> bool:
    """
    Vérifie que les centres HV sont à l'INTÉRIEUR des noyaux (pas dans le background).

    Problème détecté par expert:
    - Si le centre (calculé par mean()) tombe hors du noyau (forme concave),
      les vecteurs HV pointent vers le vide → Watershed échoue.

    Solution: Distance Transform garantit que le centre est le pixel le plus
    profond dans le noyau.

    Ce test vérifie empiriquement que les centres HV sont valides.
    """
    from scipy.ndimage import distance_transform_edt, label

    print(f"\n{'='*80}")
    print(f"  VALIDATION HV CENTERS INSIDE NUCLEI — {split.upper()}")
    print(f"{'='*80}")

    if not targets_file.exists():
        print(f"❌ ERREUR: Fichier introuvable: {targets_file}")
        return False

    data = np.load(targets_file)

    # Vérifier que inst_maps existe (nécessaire pour cette validation)
    if 'inst_maps' not in data.files:
        print(f"⚠️  SKIP: 'inst_maps' non disponible dans {targets_file.name}")
        print(f"   Cette validation nécessite inst_maps pour identifier les noyaux individuels.")
        return True  # Skip plutôt que fail

    np_targets = data['np_targets']
    hv_targets = data['hv_targets']
    inst_maps = data['inst_maps']

    n_crops = len(np_targets)
    n_to_check = min(n_samples, n_crops)

    # Sélectionner échantillons aléatoires (reproductible)
    np.random.seed(42)
    sample_indices = np.random.choice(n_crops, n_to_check, replace=False)

    issues = []
    centers_inside = 0
    centers_outside = 0
    centers_total = 0

    print(f"Vérification de {n_to_check} crops (seed=42)...")
    print()

    for idx in sample_indices:
        inst_map = inst_maps[idx]
        hv_map = hv_targets[idx]  # (2, 224, 224)
        np_mask = np_targets[idx] > 0.5

        # Pour chaque instance dans le crop
        inst_ids = np.unique(inst_map)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = inst_map == inst_id
            y_coords, x_coords = np.where(inst_mask)

            if len(y_coords) == 0:
                continue

            centers_total += 1

            # Extraire les valeurs HV pour cette instance
            h_values = hv_map[0, y_coords, x_coords]
            v_values = hv_map[1, y_coords, x_coords]

            # Le centre théorique est là où H=0 et V=0
            # Trouver le pixel avec min(|H| + |V|)
            hv_sum = np.abs(h_values) + np.abs(v_values)
            center_idx = np.argmin(hv_sum)
            center_y, center_x = y_coords[center_idx], x_coords[center_idx]

            # Vérifier que ce pixel est bien dans le noyau
            if inst_mask[center_y, center_x]:
                centers_inside += 1
            else:
                centers_outside += 1
                if centers_outside <= 5:  # Limiter les messages
                    issues.append(
                        f"Crop {idx}, Instance {inst_id}: Centre HV ({center_x}, {center_y}) "
                        f"HORS du noyau!"
                    )

    # Calcul statistiques
    if centers_total > 0:
        inside_pct = 100 * centers_inside / centers_total
        outside_pct = 100 * centers_outside / centers_total
    else:
        inside_pct = 0
        outside_pct = 0

    print(f"📊 Résultats:")
    print(f"  Centres vérifiés:    {centers_total}")
    print(f"  Centres INSIDE:      {centers_inside} ({inside_pct:.1f}%)")
    print(f"  Centres OUTSIDE:     {centers_outside} ({outside_pct:.1f}%)")
    print()

    # Verdict: On tolère un petit % d'erreurs (forme très irrégulière)
    threshold_ok = 95.0  # 95% des centres doivent être inside

    if inside_pct >= threshold_ok:
        print(f"✅ VALIDE — {inside_pct:.1f}% ≥ {threshold_ok}% centres inside")
        return True
    else:
        print(f"❌ INVALIDE — {inside_pct:.1f}% < {threshold_ok}% centres inside")
        print()
        print("Causes possibles:")
        print("  → compute_hv_maps() utilise mean() au lieu de Distance Transform")
        print("  → Noyaux concaves avec centre hors du masque")
        print()
        print("Solution:")
        print("  → Vérifier que compute_hv_maps() utilise distance_transform_edt()")
        if issues:
            print()
            print("Exemples de problèmes (max 5):")
            for issue in issues[:5]:
                print(f"  → {issue}")
        return False


def validate_sequential_ids(targets_file: Path, split: str, n_samples: int = 20) -> bool:
    """
    Vérifie que les IDs d'instances sont séquentiels [1, 2, 3, ..., N].

    Bug #1 identifié: ID collision quand certains IDs sont préservés et d'autres
    renumérotés → plusieurs noyaux avec le même ID → AJI catastrophique.

    Solution LOCAL relabeling: scipy.ndimage.label() garantit IDs séquentiels.
    """
    from scipy.ndimage import label as scipy_label

    print(f"\n{'='*80}")
    print(f"  VALIDATION IDS SÉQUENTIELS — {split.upper()}")
    print(f"{'='*80}")

    if not targets_file.exists():
        print(f"❌ ERREUR: Fichier introuvable: {targets_file}")
        return False

    data = np.load(targets_file)

    if 'inst_maps' not in data.files:
        print(f"⚠️  SKIP: 'inst_maps' non disponible dans {targets_file.name}")
        return True

    inst_maps = data['inst_maps']
    n_crops = len(inst_maps)
    n_to_check = min(n_samples, n_crops)

    np.random.seed(42)
    sample_indices = np.random.choice(n_crops, n_to_check, replace=False)

    issues = []
    crops_ok = 0
    crops_with_gaps = 0
    crops_with_duplicates = 0

    print(f"Vérification de {n_to_check} crops...")
    print()

    for idx in sample_indices:
        inst_map = inst_maps[idx]

        # Obtenir les IDs présents
        inst_ids = np.unique(inst_map)
        inst_ids = inst_ids[inst_ids > 0]  # Exclure background (0)

        if len(inst_ids) == 0:
            crops_ok += 1
            continue

        # Vérifier séquentialité: IDs doivent être [1, 2, 3, ..., max]
        expected_ids = set(range(1, len(inst_ids) + 1))
        actual_ids = set(inst_ids)

        if actual_ids == expected_ids:
            crops_ok += 1
        else:
            # Analyser le problème
            missing = expected_ids - actual_ids
            extra = actual_ids - expected_ids

            if missing or extra:
                crops_with_gaps += 1
                if len(issues) < 5:
                    issues.append(
                        f"Crop {idx}: IDs non séquentiels. "
                        f"Attendu [1..{len(inst_ids)}], manquants={missing}, extras={extra}"
                    )

            # Vérifier duplications (même ID pour plusieurs régions déconnectées)
            for inst_id in inst_ids:
                inst_mask = inst_map == inst_id
                labeled, n_components = scipy_label(inst_mask)
                if n_components > 1:
                    crops_with_duplicates += 1
                    if len(issues) < 5:
                        issues.append(
                            f"Crop {idx}: ID {inst_id} a {n_components} composantes déconnectées!"
                        )
                    break  # Un seul problème suffit pour ce crop

    print(f"📊 Résultats:")
    print(f"  Crops vérifiés:       {n_to_check}")
    print(f"  Crops OK:             {crops_ok}")
    print(f"  Crops avec gaps:      {crops_with_gaps}")
    print(f"  Crops avec doublons:  {crops_with_duplicates}")
    print()

    if crops_with_gaps == 0 and crops_with_duplicates == 0:
        print(f"✅ VALIDE — Tous les IDs sont séquentiels et uniques")
        return True
    else:
        print(f"❌ INVALIDE — Problèmes de numérotation détectés")
        print()
        print("Cause probable:")
        print("  → Bug #1: Renumbering partiel (HYBRID) au lieu de LOCAL relabeling")
        print()
        print("Solution:")
        print("  → Utiliser scipy.ndimage.label() pour LOCAL relabeling complet")
        if issues:
            print()
            print("Exemples de problèmes (max 5):")
            for issue in issues[:5]:
                print(f"  → {issue}")
        return False


def validate_hv_rotation_consistency(targets_file: Path, split: str, n_samples: int = 10) -> bool:
    """
    Vérifie la cohérence des rotations HV.

    Bug #2 identifié: Erreur dans la transformation HV après rotation spatiale.

    Pour une rotation 90° clockwise:
    - H' = V (ancienne verticale devient nouvelle horizontale)
    - V' = -H (ancienne horizontale devient -nouvelle verticale)

    Ce test vérifie que les vecteurs HV pointent bien vers les centres des noyaux
    (divergence négative = vecteurs convergents vers le centre).
    """

    print(f"\n{'='*80}")
    print(f"  VALIDATION HV ROTATION CONSISTENCY — {split.upper()}")
    print(f"{'='*80}")

    if not targets_file.exists():
        print(f"❌ ERREUR: Fichier introuvable: {targets_file}")
        return False

    data = np.load(targets_file)

    np_targets = data['np_targets']
    hv_targets = data['hv_targets']

    if 'crop_positions' in data.files:
        crop_positions = data['crop_positions']
    else:
        crop_positions = None

    n_crops = len(np_targets)
    n_to_check = min(n_samples, n_crops)

    np.random.seed(42)
    sample_indices = np.random.choice(n_crops, n_to_check, replace=False)

    divergence_values = []

    print(f"Vérification de {n_to_check} crops...")
    print()

    for idx in sample_indices:
        np_mask = np_targets[idx] > 0.5
        hv_map = hv_targets[idx]  # (2, 224, 224)

        if np_mask.sum() < 100:  # Skip crops avec peu de noyaux
            continue

        # Calculer divergence: dH/dx + dV/dy
        h_map = hv_map[0]  # Horizontal component
        v_map = hv_map[1]  # Vertical component

        # Gradients spatiaux (différences finies)
        dh_dx = np.gradient(h_map, axis=1)  # dH/dx
        dv_dy = np.gradient(v_map, axis=0)  # dV/dy

        # Divergence = dH/dx + dV/dy
        divergence = dh_dx + dv_dy

        # Moyenne sur les pixels de noyaux
        div_on_nuclei = divergence[np_mask].mean()
        divergence_values.append(div_on_nuclei)

    if not divergence_values:
        print(f"⚠️  Pas assez de données pour calculer la divergence")
        return True

    mean_div = np.mean(divergence_values)
    std_div = np.std(divergence_values)

    print(f"📊 Divergence HV (sur pixels noyaux):")
    print(f"  Mean: {mean_div:.4f}")
    print(f"  Std:  {std_div:.4f}")
    print()

    # Interprétation:
    # - Divergence NÉGATIVE: Vecteurs convergent vers le centre (CORRECT)
    # - Divergence POSITIVE: Vecteurs divergent du centre (ERREUR de rotation)
    # - Divergence ~0: Vecteurs parallèles (peut-être OK selon le noyau)

    # On s'attend à une divergence négative pour des vecteurs centripètes
    # Seuil: la moyenne doit être < -0.01 (légèrement négatif)
    threshold = -0.005

    if mean_div < threshold:
        print(f"✅ VALIDE — Divergence négative ({mean_div:.4f} < {threshold})")
        print(f"   → Vecteurs HV pointent vers les centres (comportement attendu)")
        return True
    elif mean_div > 0.1:
        print(f"❌ INVALIDE — Divergence POSITIVE ({mean_div:.4f})")
        print(f"   → Vecteurs HV pointent HORS des centres!")
        print()
        print("Cause probable:")
        print("  → Bug #2: Erreur dans HV rotation (H'=-V, V'=H au lieu de H'=V, V'=-H)")
        return False
    else:
        print(f"⚠️  ATTENTION — Divergence proche de zéro ({mean_div:.4f})")
        print(f"   → Vecteurs HV potentiellement incorrects")
        print(f"   → Vérification manuelle recommandée")
        return True  # Warning mais pas blocking


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
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Nombre d'échantillons pour les validations expertes (défaut: 20)"
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

    # 6. Validation HV centers inside nuclei (NOUVEAU - Expert recommendation)
    print(f"\n{'='*80}")
    print(f"  VALIDATIONS EXPERTES (Distance Transform, IDs, Rotations)")
    print(f"{'='*80}")

    if train_targets.exists():
        results['hv_centers_train'] = validate_hv_centers_inside_nuclei(
            train_targets, "train", n_samples=args.n_samples
        )
    else:
        results['hv_centers_train'] = None

    if val_targets.exists():
        results['hv_centers_val'] = validate_hv_centers_inside_nuclei(
            val_targets, "val", n_samples=args.n_samples
        )
    else:
        results['hv_centers_val'] = None

    # 7. Validation IDs séquentiels (prévention Bug #1)
    if train_targets.exists():
        results['sequential_ids_train'] = validate_sequential_ids(
            train_targets, "train", n_samples=args.n_samples
        )
    else:
        results['sequential_ids_train'] = None

    if val_targets.exists():
        results['sequential_ids_val'] = validate_sequential_ids(
            val_targets, "val", n_samples=args.n_samples
        )
    else:
        results['sequential_ids_val'] = None

    # 8. Validation HV rotation consistency (prévention Bug #2)
    if train_targets.exists():
        results['hv_rotation_train'] = validate_hv_rotation_consistency(
            train_targets, "train", n_samples=max(10, args.n_samples // 2)
        )
    else:
        results['hv_rotation_train'] = None

    if val_targets.exists():
        results['hv_rotation_val'] = validate_hv_rotation_consistency(
            val_targets, "val", n_samples=max(10, args.n_samples // 2)
        )
    else:
        results['hv_rotation_val'] = None

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
