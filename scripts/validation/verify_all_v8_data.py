#!/usr/bin/env python3
"""
Script: verify_all_v8_data.py
Description: Vérification GLOBALE des données v8 avant re-training

Tests effectués:
1. Vérification NPZ (structure, dtype, range, timestamp)
2. Vérification alignement spatial (distance <2px)
3. Vérification direction HV (centripète vs centrifuge)
4. Cohérence inter-famille

Verdict: GO/NO-GO par famille + verdict global

Usage:
    python scripts/validation/verify_all_v8_data.py \
        --data_dir data/family_FIXED \
        --n_samples 10
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.constants import DEFAULT_FAMILY_FIXED_DIR

# ============================================================================
# CONSTANTES
# ============================================================================

FAMILIES = ["glandular", "digestive", "urologic", "epidermal", "respiratory"]

V8_TIMESTAMP_MIN = datetime(2025, 12, 26, 0, 0, 0)  # v8 généré après cette date

EXPECTED_KEYS = ['images', 'np_targets', 'hv_targets', 'nt_targets', 'inst_maps', 'fold_ids', 'image_ids']

# Critères GO/NO-GO
THRESHOLDS = {
    'distance_mean': 2.0,      # pixels
    'precision': 90.0,         # %
    'recall': 90.0,            # %
    'hv_range_min': -1.0,
    'hv_range_max': 1.0,
    'v8_votes_ratio': 0.90,    # 90%+ des instances doivent voter v8
    'angular_error_max': 10.0, # degrés
}

# ============================================================================
# VERIFICATION NPZ
# ============================================================================

def verify_npz_structure(npz_path: Path) -> Dict:
    """Vérifie structure, dtype, range, timestamp du NPZ."""

    result = {
        'path': str(npz_path),
        'exists': npz_path.exists(),
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    if not npz_path.exists():
        result['errors'].append(f"Fichier n'existe pas: {npz_path}")
        return result

    try:
        data = np.load(npz_path)

        # 1. Vérifier clés présentes
        missing_keys = set(EXPECTED_KEYS) - set(data.keys())
        if missing_keys:
            result['errors'].append(f"Clés manquantes: {missing_keys}")

        # 2. Vérifier inst_maps (signature v8)
        if 'inst_maps' not in data:
            result['errors'].append("CRITIQUE: 'inst_maps' manquant (NPZ v7 ou antérieur!)")
            return result

        # 3. Vérifier shapes cohérentes
        n_samples = data['images'].shape[0]
        for key in ['np_targets', 'hv_targets', 'nt_targets', 'inst_maps']:
            if key in data and data[key].shape[0] != n_samples:
                result['errors'].append(f"Shape incohérente {key}: {data[key].shape[0]} vs {n_samples}")

        # 4. Vérifier dtypes
        dtype_checks = {
            'images': np.uint8,
            'np_targets': np.float32,
            'hv_targets': np.float32,
            'nt_targets': np.int64,
            'inst_maps': np.int32,
        }

        for key, expected_dtype in dtype_checks.items():
            if key in data:
                actual_dtype = data[key].dtype
                if actual_dtype != expected_dtype:
                    result['warnings'].append(f"{key} dtype: {actual_dtype} (attendu: {expected_dtype})")

        # 5. Vérifier range HV
        if 'hv_targets' in data:
            hv = data['hv_targets']
            hv_min, hv_max = hv.min(), hv.max()
            result['stats']['hv_min'] = float(hv_min)
            result['stats']['hv_max'] = float(hv_max)

            if hv_min < THRESHOLDS['hv_range_min'] - 0.1:
                result['errors'].append(f"HV min={hv_min:.3f} < {THRESHOLDS['hv_range_min']}")

            if hv_max > THRESHOLDS['hv_range_max'] + 0.1:
                result['errors'].append(f"HV max={hv_max:.3f} > {THRESHOLDS['hv_range_max']}")

            # Vérifier si int8 (signature v7)
            if hv_min < -10 or hv_max > 10:
                result['errors'].append(f"CRITIQUE: HV range [{hv_min:.1f}, {hv_max:.1f}] - données v7 (int8)!")

        # 6. Timestamp (approximatif via metadata si disponible)
        file_mtime = datetime.fromtimestamp(npz_path.stat().st_mtime)
        result['stats']['timestamp'] = file_mtime.strftime('%Y-%m-%d %H:%M:%S')

        if file_mtime < V8_TIMESTAMP_MIN:
            result['warnings'].append(f"Timestamp {file_mtime} < {V8_TIMESTAMP_MIN} (possiblement v7)")

        # 7. Stats générales
        result['stats']['n_samples'] = n_samples
        result['stats']['n_folds'] = len(np.unique(data['fold_ids']))

        data.close()

    except Exception as e:
        result['errors'].append(f"Erreur lecture NPZ: {e}")

    return result

# ============================================================================
# VERIFICATION ALIGNEMENT SPATIAL
# ============================================================================

def extract_gt_centers(inst_map: np.ndarray) -> np.ndarray:
    """Extrait centroïdes GT depuis inst_map."""
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    centers = []
    for inst_id in inst_ids:
        y_coords, x_coords = np.where(inst_map == inst_id)
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)
        centers.append([center_y, center_x])

    return np.array(centers) if len(centers) > 0 else np.array([]).reshape(0, 2)

def extract_centers_from_hv(hv_map: np.ndarray, inst_map: np.ndarray) -> np.ndarray:
    """Extrait centroïdes prédits depuis HV (méthode v8 centripète)."""
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    # Méthode v8: HV magnitude minimale au centre
    hv_magnitude = np.sqrt(hv_map[0]**2 + hv_map[1]**2)

    centers = []
    for inst_id in inst_ids:
        y_coords, x_coords = np.where(inst_map == inst_id)

        if len(y_coords) == 0:
            continue

        mags_in_inst = hv_magnitude[y_coords, x_coords]
        min_idx = np.argmin(mags_in_inst)

        center_y = y_coords[min_idx]
        center_x = x_coords[min_idx]
        centers.append([center_y, center_x])

    return np.array(centers) if len(centers) > 0 else np.array([]).reshape(0, 2)

def compute_alignment_metrics(gt_centers: np.ndarray, pred_centers: np.ndarray, threshold: float = 10.0) -> Dict:
    """Calcule métriques d'alignement (distance, precision, recall)."""

    n_gt = len(gt_centers)
    n_pred = len(pred_centers)

    if n_gt == 0 or n_pred == 0:
        return {
            'distance_mean': 0.0,
            'distance_max': 0.0,
            'tp': 0,
            'fp': n_pred,
            'fn': n_gt,
            'precision': 0.0,
            'recall': 0.0,
        }

    # Hungarian matching
    from scipy.spatial.distance import cdist
    cost_matrix = cdist(gt_centers, pred_centers)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    distances = cost_matrix[row_ind, col_ind]

    # TP: matched avec distance < threshold
    tp = np.sum(distances < threshold)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = 100.0 * tp / n_pred if n_pred > 0 else 0.0
    recall = 100.0 * tp / n_gt if n_gt > 0 else 0.0

    return {
        'distance_mean': float(np.mean(distances)),
        'distance_max': float(np.max(distances)),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
    }

def verify_spatial_alignment(npz_path: Path, n_samples: int = 10) -> Dict:
    """Vérifie alignement spatial HV sur N échantillons."""

    result = {
        'distances': [],
        'precisions': [],
        'recalls': [],
        'errors': [],
    }

    try:
        data = np.load(npz_path)

        if 'inst_maps' not in data:
            result['errors'].append("inst_maps manquant - impossible de tester alignement")
            return result

        n_total = len(data['images'])
        indices = np.random.RandomState(42).choice(n_total, min(n_samples, n_total), replace=False)

        for idx in indices:
            inst_map = data['inst_maps'][idx]
            hv_target = data['hv_targets'][idx]

            gt_centers = extract_gt_centers(inst_map)
            pred_centers = extract_centers_from_hv(hv_target, inst_map)

            if len(gt_centers) == 0:
                continue

            metrics = compute_alignment_metrics(gt_centers, pred_centers)

            result['distances'].append(metrics['distance_mean'])
            result['precisions'].append(metrics['precision'])
            result['recalls'].append(metrics['recall'])

        data.close()

    except Exception as e:
        result['errors'].append(f"Erreur vérification alignement: {e}")

    return result

# ============================================================================
# VERIFICATION DIRECTION HV (v7 vs v8)
# ============================================================================

def verify_hv_direction(npz_path: Path, n_samples: int = 10) -> Dict:
    """Vérifie direction HV (centripète v8 vs centrifuge v7)."""

    result = {
        'v7_votes': 0,
        'v8_votes': 0,
        'angular_errors': [],
        'errors': [],
    }

    try:
        data = np.load(npz_path)

        if 'inst_maps' not in data or 'hv_targets' not in data:
            result['errors'].append("inst_maps ou hv_targets manquant")
            return result

        n_total = len(data['images'])
        indices = np.random.RandomState(42).choice(n_total, min(n_samples, n_total), replace=False)

        for idx in indices:
            inst_map = data['inst_maps'][idx]
            hv_map = data['hv_targets'][idx]

            inst_ids = np.unique(inst_map)
            inst_ids = inst_ids[inst_ids > 0]

            for inst_id in inst_ids:
                y_coords, x_coords = np.where(inst_map == inst_id)

                if len(y_coords) < 10:
                    continue

                # Centroïde
                center_y = np.mean(y_coords)
                center_x = np.mean(x_coords)

                # Test direction: pixels au DESSUS du centre (y < center_y)
                top_mask = y_coords < center_y
                if not np.any(top_mask):
                    continue

                y_top = y_coords[top_mask]
                x_top = x_coords[top_mask]
                v_vals = hv_map[0, y_top, x_top]

                mean_v = np.mean(v_vals)

                # v8 (centripète): pixels au dessus ont v > 0 (pointent vers BAS = centre)
                # v7 (centrifuge): pixels au dessus ont v < 0 (pointent vers HAUT = extérieur)
                if mean_v < 0:
                    result['v7_votes'] += 1
                else:
                    result['v8_votes'] += 1

                # Angular error (échantillonner quelques pixels)
                sample_indices = np.random.choice(len(y_coords), min(5, len(y_coords)), replace=False)

                for i in sample_indices:
                    py, px = y_coords[i], x_coords[i]

                    # Vecteur attendu (v8 centripète)
                    expected_dy = center_y - py
                    expected_dx = center_x - px
                    expected_norm = np.sqrt(expected_dy**2 + expected_dx**2)

                    if expected_norm < 1e-6:
                        continue

                    expected_dy /= expected_norm
                    expected_dx /= expected_norm

                    # Vecteur stocké
                    v_val = hv_map[0, py, px]
                    h_val = hv_map[1, py, px]
                    stored_norm = np.sqrt(v_val**2 + h_val**2)

                    if stored_norm < 1e-6:
                        continue

                    v_norm = v_val / stored_norm
                    h_norm = h_val / stored_norm

                    # Cosine similarity
                    dot = expected_dy * v_norm + expected_dx * h_norm
                    dot = np.clip(dot, -1.0, 1.0)
                    angle_error = np.rad2deg(np.arccos(dot))

                    result['angular_errors'].append(angle_error)

        data.close()

    except Exception as e:
        result['errors'].append(f"Erreur vérification direction HV: {e}")

    return result

# ============================================================================
# RAPPORT GLOBAL
# ============================================================================

def print_family_report(family: str, npz_result: Dict, alignment_result: Dict, direction_result: Dict) -> str:
    """Génère rapport pour une famille."""

    verdict = "GO"
    issues = []

    # 1. Vérifications NPZ
    if npz_result['errors']:
        verdict = "NO-GO"
        issues.extend(npz_result['errors'])

    # 2. Vérifications alignement
    if alignment_result['errors']:
        verdict = "NO-GO"
        issues.extend(alignment_result['errors'])
    else:
        if alignment_result['distances']:
            dist_mean = np.mean(alignment_result['distances'])
            if dist_mean > THRESHOLDS['distance_mean']:
                verdict = "NO-GO"
                issues.append(f"Distance moyenne {dist_mean:.2f}px > {THRESHOLDS['distance_mean']}px")

        if alignment_result['precisions']:
            prec_mean = np.mean(alignment_result['precisions'])
            if prec_mean < THRESHOLDS['precision']:
                verdict = "NO-GO"
                issues.append(f"Precision {prec_mean:.1f}% < {THRESHOLDS['precision']}%")

        if alignment_result['recalls']:
            rec_mean = np.mean(alignment_result['recalls'])
            if rec_mean < THRESHOLDS['recall']:
                verdict = "NO-GO"
                issues.append(f"Recall {rec_mean:.1f}% < {THRESHOLDS['recall']}%")

    # 3. Vérifications direction HV
    if direction_result['errors']:
        verdict = "NO-GO"
        issues.extend(direction_result['errors'])
    else:
        total_votes = direction_result['v7_votes'] + direction_result['v8_votes']
        if total_votes > 0:
            v8_ratio = direction_result['v8_votes'] / total_votes
            if v8_ratio < THRESHOLDS['v8_votes_ratio']:
                verdict = "NO-GO"
                issues.append(f"v8 votes {v8_ratio*100:.1f}% < {THRESHOLDS['v8_votes_ratio']*100}%")

        if direction_result['angular_errors']:
            ang_err_mean = np.mean(direction_result['angular_errors'])
            if ang_err_mean > THRESHOLDS['angular_error_max']:
                verdict = "NO-GO"
                issues.append(f"Angular error {ang_err_mean:.1f}° > {THRESHOLDS['angular_error_max']}°")

    # Construire rapport
    lines = []
    lines.append("="*80)
    lines.append(f"FAMILLE: {family.upper()}")
    lines.append("="*80)

    # NPZ
    lines.append("\n1. STRUCTURE NPZ")
    lines.append("-"*80)
    if npz_result['exists']:
        lines.append(f"  ✅ Fichier existe: {npz_result['path']}")
        lines.append(f"  📊 Samples: {npz_result['stats'].get('n_samples', 'N/A')}")
        lines.append(f"  📅 Timestamp: {npz_result['stats'].get('timestamp', 'N/A')}")
        lines.append(f"  📏 HV range: [{npz_result['stats'].get('hv_min', 'N/A'):.3f}, {npz_result['stats'].get('hv_max', 'N/A'):.3f}]")
    else:
        lines.append(f"  ❌ Fichier n'existe pas")

    if npz_result['errors']:
        lines.append("  ❌ ERREURS:")
        for err in npz_result['errors']:
            lines.append(f"     - {err}")

    if npz_result['warnings']:
        lines.append("  ⚠️  WARNINGS:")
        for warn in npz_result['warnings']:
            lines.append(f"     - {warn}")

    # Alignement
    lines.append("\n2. ALIGNEMENT SPATIAL")
    lines.append("-"*80)
    if alignment_result['distances']:
        dist_mean = np.mean(alignment_result['distances'])
        dist_max = np.max(alignment_result['distances'])
        prec_mean = np.mean(alignment_result['precisions'])
        rec_mean = np.mean(alignment_result['recalls'])

        lines.append(f"  Distance moyenne: {dist_mean:.2f}px (seuil: <{THRESHOLDS['distance_mean']}px)")
        lines.append(f"  Distance max:     {dist_max:.2f}px")
        lines.append(f"  Precision:        {prec_mean:.1f}% (seuil: >{THRESHOLDS['precision']}%)")
        lines.append(f"  Recall:           {rec_mean:.1f}% (seuil: >{THRESHOLDS['recall']}%)")

        status = "✅" if dist_mean < THRESHOLDS['distance_mean'] else "❌"
        lines.append(f"  {status} Alignement: {'PARFAIT' if dist_mean < 1.0 else 'BON' if dist_mean < 2.0 else 'DÉGRADÉ'}")
    else:
        lines.append("  ⚠️  Pas de données d'alignement")

    if alignment_result['errors']:
        for err in alignment_result['errors']:
            lines.append(f"  ❌ {err}")

    # Direction HV
    lines.append("\n3. DIRECTION HV (v7 vs v8)")
    lines.append("-"*80)
    total_votes = direction_result['v7_votes'] + direction_result['v8_votes']
    if total_votes > 0:
        v8_ratio = direction_result['v8_votes'] / total_votes
        lines.append(f"  v7 (centrifuge): {direction_result['v7_votes']} votes")
        lines.append(f"  v8 (centripète): {direction_result['v8_votes']} votes")
        lines.append(f"  Ratio v8: {v8_ratio*100:.1f}% (seuil: >{THRESHOLDS['v8_votes_ratio']*100}%)")

        status = "✅" if v8_ratio >= THRESHOLDS['v8_votes_ratio'] else "❌"
        lines.append(f"  {status} Verdict: {'v8 CONFIRMÉ' if v8_ratio > 0.9 else 'MIXTE' if v8_ratio > 0.5 else 'v7 DÉTECTÉ!'}")

    if direction_result['angular_errors']:
        ang_err_mean = np.mean(direction_result['angular_errors'])
        ang_err_std = np.std(direction_result['angular_errors'])
        lines.append(f"  Erreur angulaire: {ang_err_mean:.1f}° ± {ang_err_std:.1f}° (seuil: <{THRESHOLDS['angular_error_max']}°)")

        status = "✅" if ang_err_mean < THRESHOLDS['angular_error_max'] else "❌"
        lines.append(f"  {status} Précision directionnelle")

    if direction_result['errors']:
        for err in direction_result['errors']:
            lines.append(f"  ❌ {err}")

    # Verdict final
    lines.append("\n" + "="*80)
    if verdict == "GO":
        lines.append(f"✅ VERDICT: {verdict} - Famille {family} validée pour production")
    else:
        lines.append(f"❌ VERDICT: {verdict} - Famille {family} BLOQUÉE")
        lines.append("\nProblèmes identifiés:")
        for issue in issues:
            lines.append(f"  - {issue}")
    lines.append("="*80)

    return "\n".join(lines), verdict

def main():
    parser = argparse.ArgumentParser(description="Vérification globale données v8")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_FAMILY_FIXED_DIR,
                        help='Répertoire contenant les NPZ')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Nombre échantillons à tester par famille')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("="*80)
    print("VÉRIFICATION GLOBALE DONNÉES v8")
    print("="*80)
    print(f"Répertoire: {data_dir}")
    print(f"Échantillons par famille: {args.n_samples}")
    print()

    results = {}
    verdicts = {}

    for family in FAMILIES:
        print(f"\n🔍 Test famille: {family}...")

        npz_path = data_dir / f"{family}_data_FIXED.npz"

        # 1. Vérification NPZ
        npz_result = verify_npz_structure(npz_path)

        # 2. Vérification alignement
        alignment_result = verify_spatial_alignment(npz_path, n_samples=args.n_samples)

        # 3. Vérification direction HV
        direction_result = verify_hv_direction(npz_path, n_samples=args.n_samples)

        # Générer rapport
        report, verdict = print_family_report(family, npz_result, alignment_result, direction_result)

        print(report)

        results[family] = {
            'npz': npz_result,
            'alignment': alignment_result,
            'direction': direction_result,
        }
        verdicts[family] = verdict

    # RAPPORT GLOBAL
    print("\n" + "="*80)
    print("RAPPORT GLOBAL")
    print("="*80)

    print("\nRécapitulatif par famille:")
    print("-"*80)

    go_count = sum(1 for v in verdicts.values() if v == "GO")

    for family in FAMILIES:
        verdict = verdicts[family]
        symbol = "✅" if verdict == "GO" else "❌"
        print(f"  {symbol} {family.capitalize():12} : {verdict}")

    print()
    print(f"Total: {go_count}/{len(FAMILIES)} familles validées")
    print()

    # Verdict final
    if go_count == len(FAMILIES):
        print("="*80)
        print("🎉 VERDICT FINAL: GO - TOUTES LES FAMILLES VALIDÉES")
        print("="*80)
        print()
        print("✅ Données v8 prêtes pour re-training")
        print()
        print("Prochaines étapes:")
        print("  1. Re-training 5 familles (temps estimé: ~10h)")
        print("     bash scripts/training/train_all_families_v8.sh")
        print()
        print("  2. Évaluation AJI finale (objectif: 0.06 → 0.60+)")
        print("     python scripts/evaluation/evaluate_ground_truth.py")
        print()
        return 0
    else:
        print("="*80)
        print(f"❌ VERDICT FINAL: NO-GO - {len(FAMILIES) - go_count}/{len(FAMILIES)} FAMILLES BLOQUÉES")
        print("="*80)
        print()
        print("Familles problématiques:")
        for family in FAMILIES:
            if verdicts[family] == "NO-GO":
                print(f"  ❌ {family}")
        print()
        print("⚠️  NE PAS LANCER LE RE-TRAINING AVANT CORRECTION")
        print()
        print("Actions recommandées:")
        print("  1. Vérifier logs détaillés ci-dessus")
        print("  2. Régénérer famille(s) problématique(s)")
        print("  3. Re-lancer cette vérification")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())
