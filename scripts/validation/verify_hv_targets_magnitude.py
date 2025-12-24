#!/usr/bin/env python3
"""
Vérifie la magnitude des targets HV stockés.

Ce script charge les targets HV et calcule leur magnitude pour déterminer
si le problème de magnitude faible vient:
- Des DONNÉES (targets déjà faibles → Gaussian smoothing trop agressif)
- Du MODÈLE (targets forts mais prédictions faibles → loss function inadéquate)

Usage:
    python scripts/validation/verify_hv_targets_magnitude.py \
        --family epidermal \
        --n_samples 50
"""

import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import DEFAULT_FAMILY_DATA_DIR


def verify_hv_targets_magnitude(
    family: str,
    data_dir: str = None,
    n_samples: int = 50
):
    """
    Vérifie la magnitude des targets HV stockés.

    Args:
        family: Nom de la famille (epidermal, glandular, etc.)
        data_dir: Répertoire contenant les données par famille
        n_samples: Nombre d'échantillons à analyser

    Returns:
        dict avec statistiques de magnitude
    """
    if data_dir is None:
        data_dir = DEFAULT_FAMILY_DATA_DIR

    data_dir = Path(data_dir)

    print("\n" + "="*80)
    print(f"VÉRIFICATION MAGNITUDE TARGETS HV — {family.upper()}")
    print("="*80)
    print(f"\nData dir: {data_dir}")
    print(f"Échantillons: {n_samples}")
    print("")

    # Charger targets
    targets_path = data_dir / f"{family}_targets.npz"

    if not targets_path.exists():
        print(f"❌ ERREUR: Targets introuvables: {targets_path}")
        return None

    print(f"Chargement targets: {targets_path}")
    data = np.load(targets_path)

    # Extraire HV targets
    hv_targets = data['hv_targets']  # (N, 2, H, W)
    np_targets = data['np_targets']  # (N, H, W)

    n_total = hv_targets.shape[0]
    n_samples = min(n_samples, n_total)

    print(f"Targets shape: {hv_targets.shape}")
    print(f"Échantillons à analyser: {n_samples}/{n_total}")
    print("")

    # Statistiques globales
    print("="*80)
    print("STATISTIQUES GLOBALES HV TARGETS")
    print("="*80)
    print(f"Dtype:  {hv_targets.dtype}")
    print(f"Min:    {hv_targets.min():.4f}")
    print(f"Max:    {hv_targets.max():.4f}")
    print(f"Mean:   {hv_targets.mean():.4f}")
    print(f"Std:    {hv_targets.std():.4f}")
    print("")

    # Calcul magnitude par échantillon
    print("="*80)
    print("MAGNITUDE PAR ÉCHANTILLON")
    print("="*80)

    magnitudes_global = []
    magnitudes_masked = []

    for i in range(n_samples):
        hv = hv_targets[i]  # (2, H, W)
        np_mask = np_targets[i]  # (H, W)

        # Magnitude globale (sur toute l'image)
        mag_h_global = np.abs(hv[0]).max()
        mag_v_global = np.abs(hv[1]).max()
        mag_global = max(mag_h_global, mag_v_global)
        magnitudes_global.append(mag_global)

        # Magnitude masquée (uniquement sur pixels de noyaux)
        hv_masked = hv * np_mask[np.newaxis, :, :]  # Broadcast mask
        mag_h_masked = np.abs(hv_masked[0]).max()
        mag_v_masked = np.abs(hv_masked[1]).max()
        mag_masked = max(mag_h_masked, mag_v_masked)
        magnitudes_masked.append(mag_masked)

        if i < 10:  # Afficher détails pour premiers 10
            print(f"Sample {i:2d}: Global={mag_global:.4f}, Masked={mag_masked:.4f} "
                  f"(H_g={mag_h_global:.4f}, V_g={mag_v_global:.4f}, "
                  f"H_m={mag_h_masked:.4f}, V_m={mag_v_masked:.4f})")

    magnitudes_global = np.array(magnitudes_global)
    magnitudes_masked = np.array(magnitudes_masked)

    # Statistiques
    print("\n" + "="*80)
    print("STATISTIQUES MAGNITUDE TARGETS")
    print("="*80)

    print("\n📊 MAGNITUDE GLOBALE (toute l'image):")
    print(f"Mean:   {magnitudes_global.mean():.4f}")
    print(f"Std:    {magnitudes_global.std():.4f}")
    print(f"Min:    {magnitudes_global.min():.4f}")
    print(f"Max:    {magnitudes_global.max():.4f}")

    print("\n📊 MAGNITUDE MASQUÉE (pixels de noyaux uniquement):")
    print(f"Mean:   {magnitudes_masked.mean():.4f}")
    print(f"Std:    {magnitudes_masked.std():.4f}")
    print(f"Min:    {magnitudes_masked.min():.4f}")
    print(f"Max:    {magnitudes_masked.max():.4f}")

    # Diagnostic
    print("\n" + "="*80)
    print("DIAGNOSTIC")
    print("="*80)

    mean_mag_global = magnitudes_global.mean()
    mean_mag_masked = magnitudes_masked.mean()

    print(f"\nMagnitude targets (globale): {mean_mag_global:.4f}")
    print(f"Magnitude targets (masquée): {mean_mag_masked:.4f}")

    if mean_mag_masked < 0.10:
        print("\n❌ PROBLÈME DONNÉES: Magnitude targets trop faible (<0.10)")
        print("   Les targets HV ont été TROP LISSÉS (Gaussian smoothing agressif)")
        print("\n💡 SOLUTION:")
        print("   Ré-générer targets avec moins de smoothing:")
        print("   - Réduire sigma Gaussian (ex: 3.0 → 1.5)")
        print("   - Ou utiliser HV maps sans smoothing (peaks bruts)")
        status = "DATA_ISSUE"

    elif mean_mag_masked < 0.30:
        print("\n⚠️ MAGNITUDE TARGETS MODÉRÉE (0.10-0.30)")
        print("   Targets ont magnitude acceptable mais pas optimale")
        print(f"   Magnitude prédite: 0.04 vs targets: {mean_mag_masked:.4f}")
        print("\n💡 ANALYSE:")
        print("   - Si ratio pred/target < 0.2: MODÈLE ne muscle pas assez")
        print("   - Essayer Solution A (magnitude loss) pour forcer le modèle")
        status = "PARTIAL"

    else:
        print("\n✅ MAGNITUDE TARGETS FORTE (>0.30)")
        print("   Les targets sont corrects (magnitude élevée)")
        print(f"   Magnitude prédite: 0.04 vs targets: {mean_mag_masked:.4f}")
        print(f"   Ratio pred/target: {0.04/mean_mag_masked:.2f} (8-10× trop faible!)")
        print("\n💡 DIAGNOSTIC:")
        print("   Le problème vient du MODÈLE, pas des données")
        print("   La loss function ne force pas le modèle à prédire des gradients forts")
        print("\n💡 SOLUTION RECOMMANDÉE:")
        print("   Implémenter Solution A (magnitude loss) dans hovernet_decoder.py")
        status = "MODEL_ISSUE"

    # Distribution des magnitudes
    print("\n" + "="*80)
    print("DISTRIBUTION MAGNITUDE (masquée)")
    print("="*80)

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(magnitudes_masked, bins=bins)

    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / n_samples * 100
        bar = "█" * int(pct / 2)
        print(f"[{bins[i]:.1f}-{bins[i+1]:.1f}]: {count:3d} ({pct:5.1f}%) {bar}")

    print("\n" + "="*80)

    return {
        'mean_global': mean_mag_global,
        'mean_masked': mean_mag_masked,
        'std_global': magnitudes_global.std(),
        'std_masked': magnitudes_masked.std(),
        'min_masked': magnitudes_masked.min(),
        'max_masked': magnitudes_masked.max(),
        'status': status,
        'n_samples': n_samples
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérification magnitude targets HV")
    parser.add_argument('--family', type=str, default='epidermal',
                       help="Famille à analyser")
    parser.add_argument('--data_dir', type=str, default=None,
                       help=f"Répertoire données (défaut: {DEFAULT_FAMILY_DATA_DIR})")
    parser.add_argument('--n_samples', type=int, default=50,
                       help="Nombre d'échantillons à analyser")

    args = parser.parse_args()

    result = verify_hv_targets_magnitude(
        args.family,
        args.data_dir,
        args.n_samples
    )

    if result:
        print(f"\n✅ Analyse terminée")
        print(f"   Status: {result['status']}")
        print(f"   Magnitude moyenne (masquée): {result['mean_masked']:.4f}")
