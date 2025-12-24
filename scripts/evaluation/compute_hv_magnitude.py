#!/usr/bin/env python3
"""
Calcul RAPIDE de la HV magnitude pour vérifier si lambda_hv=3.0 a fonctionné.

Ce script charge le checkpoint, fait une inférence sur quelques échantillons,
et calcule max(abs(HV)) pour déterminer si les gradients sont FORTS.

Objectif:
- AVANT (lambda_hv=2.0): HV magnitude ~0.022
- APRÈS (lambda_hv=3.0): HV magnitude >0.15 (succès)
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader
from src.constants import DEFAULT_FAMILY_DATA_DIR

def compute_hv_magnitude(
    family: str,
    checkpoint_path: str,
    n_samples: int = 10,
    device: str = "cuda"
):
    """
    Calcule la magnitude HV (max(abs(HV))) sur échantillons d'entraînement.

    Returns:
        dict avec statistics HV magnitude
    """
    print("\n" + "="*80)
    print(f"CALCUL HV MAGNITUDE - {family.upper()}")
    print("="*80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Échantillons: {n_samples}")
    print("")

    # Charger features training
    data_dir = Path(DEFAULT_FAMILY_DATA_DIR)
    features_path = data_dir / f"{family}_features.npz"

    if not features_path.exists():
        print(f"❌ ERREUR: Features introuvables: {features_path}")
        return None

    print(f"Chargement features: {features_path}")
    features_data = np.load(features_path)

    if 'features' in features_data:
        features = features_data['features']
    elif 'layer_24' in features_data:
        features = features_data['layer_24']
    else:
        print(f"❌ ERREUR: Clés disponibles: {list(features_data.keys())}")
        return None

    n_total = features.shape[0]
    n_samples = min(n_samples, n_total)

    print(f"Features: {features.shape}")
    print(f"Échantillons à tester: {n_samples}/{n_total}")
    print("")

    # Charger modèle
    print(f"Chargement modèle...")
    model = ModelLoader.load_hovernet(checkpoint_path, device=device)
    model.eval()
    print(f"✅ Modèle chargé\n")

    # Inférence sur échantillons
    print("="*80)
    print("INFÉRENCE & CALCUL MAGNITUDE")
    print("="*80)

    magnitudes = []

    for i in range(n_samples):
        # Extraire features sample (B, 261, 1536) - CLS + 256 patches + 4 registers
        sample_features = torch.from_numpy(features[i:i+1]).float().to(device)

        # Le modèle HoVerNet attend (B, N, D) et fait le reshape lui-même
        # Pas besoin de reshaper manuellement
        with torch.no_grad():
            np_out, hv_out, nt_out = model(sample_features)

        # Extraire HV predictions
        hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

        # Calculer magnitude (max(abs(HV)))
        magnitude_h = np.abs(hv_pred[0]).max()
        magnitude_v = np.abs(hv_pred[1]).max()
        magnitude_max = max(magnitude_h, magnitude_v)

        magnitudes.append(magnitude_max)

        print(f"Sample {i:2d}: HV magnitude = {magnitude_max:.4f} (H={magnitude_h:.4f}, V={magnitude_v:.4f})")

    # Statistiques
    magnitudes = np.array(magnitudes)

    print("\n" + "="*80)
    print("STATISTIQUES HV MAGNITUDE")
    print("="*80)

    mean_mag = magnitudes.mean()
    std_mag = magnitudes.std()
    min_mag = magnitudes.min()
    max_mag = magnitudes.max()

    print(f"\nMean:   {mean_mag:.4f}")
    print(f"Std:    {std_mag:.4f}")
    print(f"Min:    {min_mag:.4f}")
    print(f"Max:    {max_mag:.4f}")

    # Diagnostic
    print("\n" + "="*80)
    print("DIAGNOSTIC")
    print("="*80)

    if mean_mag < 0.05:
        print("\n❌ ÉCHEC: Magnitude trop faible (<0.05)")
        print("   Lambda_hv=3.0 n'a PAS augmenté les gradients")
        print("\n💡 RECOMMANDATION: Tester lambda_hv=5.0 (Option B)")
        status = "FAIL"

    elif mean_mag < 0.15:
        print("\n⚠️  PROGRÈS: Magnitude améliorée mais insuffisante")
        print(f"   Amélioration: 0.022 → {mean_mag:.4f} (+{(mean_mag/0.022 - 1)*100:.0f}%)")
        print("\n💡 RECOMMANDATION: Tester lambda_hv=5.0 pour atteindre >0.15")
        status = "PARTIAL"

    else:
        print("\n✅ SUCCÈS: Magnitude élevée (>0.15)")
        print(f"   Amélioration: 0.022 → {mean_mag:.4f} (+{(mean_mag/0.022 - 1)*100:.0f}%)")
        print("\n🎯 Lambda_hv=3.0 a fonctionné! Gradients forts créés.")
        status = "SUCCESS"

    # Comparaison avec objectif
    print(f"\nObjectif magnitude: >0.15")
    print(f"Magnitude obtenue:  {mean_mag:.4f}")

    if mean_mag >= 0.15:
        print(f"Ratio: {mean_mag/0.15:.2f}× objectif ✅")
    else:
        print(f"Manque: {(0.15 - mean_mag)/0.15*100:.1f}% pour atteindre objectif")

    print("\n" + "="*80)

    return {
        'mean': mean_mag,
        'std': std_mag,
        'min': min_mag,
        'max': max_mag,
        'status': status,
        'samples': n_samples
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calcul rapide HV magnitude")
    parser.add_argument('--family', type=str, default='epidermal')
    parser.add_argument('--checkpoint', type=str,
                       default='models/checkpoints/hovernet_epidermal_best.pth')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    result = compute_hv_magnitude(
        args.family,
        args.checkpoint,
        args.n_samples,
        args.device
    )

    if result:
        print(f"\n✅ Test terminé")
        print(f"   Status: {result['status']}")
        print(f"   Mean magnitude: {result['mean']:.4f}")
