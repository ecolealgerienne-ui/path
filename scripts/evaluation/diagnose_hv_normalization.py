#!/usr/bin/env python3
"""
Diagnostic: Comprendre la normalisation HV dans OLD vs NEW data.

Vérifie:
1. Dtype des HV maps
2. Range des valeurs
3. Distribution des valeurs
4. Si normalisation est appliquée correctement
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def diagnose_hv_data(data_path: Path, label: str):
    """Diagnostique les HV maps."""

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: {label}")
    print(f"{'='*70}")

    data = np.load(data_path)

    # Charger HV targets
    if 'hv_targets' in data:
        hv_targets = data['hv_targets']
    else:
        print(f"❌ Pas de clé 'hv_targets' dans {data_path.name}")
        print(f"   Clés disponibles: {list(data.keys())}")
        return

    print(f"\n📊 PROPRIÉTÉS GLOBALES")
    print(f"   Shape: {hv_targets.shape}")
    print(f"   Dtype: {hv_targets.dtype}")
    print(f"   Size: {hv_targets.nbytes / 1024**2:.2f} MB")

    print(f"\n📏 STATISTIQUES PAR CANAL")

    # Canal 0: Horizontal
    h_channel = hv_targets[:, 0, :, :]
    print(f"\n   Canal H (Horizontal):")
    print(f"      Min: {h_channel.min():.6f}")
    print(f"      Max: {h_channel.max():.6f}")
    print(f"      Mean: {h_channel.mean():.6f}")
    print(f"      Std: {h_channel.std():.6f}")
    print(f"      Median: {np.median(h_channel):.6f}")

    # Canal 1: Vertical
    v_channel = hv_targets[:, 1, :, :]
    print(f"\n   Canal V (Vertical):")
    print(f"      Min: {v_channel.min():.6f}")
    print(f"      Max: {v_channel.max():.6f}")
    print(f"      Mean: {v_channel.mean():.6f}")
    print(f"      Std: {v_channel.std():.6f}")
    print(f"      Median: {np.median(v_channel):.6f}")

    print(f"\n🔍 DIAGNOSTIC NORMALISATION")

    # Vérifier si normalisé à [-1, 1]
    is_normalized = (h_channel.min() >= -1.1) and (h_channel.max() <= 1.1)

    if is_normalized:
        print(f"   ✅ NORMALISÉ: Range ≈ [-1, 1]")
    else:
        print(f"   ❌ NON NORMALISÉ: Range = [{h_channel.min():.2f}, {h_channel.max():.2f}]")

        # Estimer le facteur de normalisation
        max_abs = max(abs(h_channel.min()), abs(h_channel.max()))
        print(f"   💡 Facteur de normalisation estimé: {max_abs:.2f}")
        print(f"      (Pour normaliser: diviser par {max_abs:.2f})")

    # Vérifier symétrie (doit être centré autour de 0)
    h_mean_abs = abs(h_channel.mean())
    v_mean_abs = abs(v_channel.mean())

    if h_mean_abs < 0.1 and v_mean_abs < 0.1:
        print(f"   ✅ CENTRÉ: Mean proche de 0")
    else:
        print(f"   ⚠️  DÉCENTRÉ: Mean H={h_channel.mean():.4f}, V={v_channel.mean():.4f}")

    print(f"\n📈 DISTRIBUTION DES VALEURS (échantillon)")

    # Prendre un échantillon pour analyse
    sample_hv = hv_targets[0]  # Premier échantillon

    # Histogramme
    h_values = sample_hv[0].flatten()
    v_values = sample_hv[1].flatten()

    # Percentiles
    h_percentiles = np.percentile(h_values, [0, 25, 50, 75, 100])
    v_percentiles = np.percentile(v_values, [0, 25, 50, 75, 100])

    print(f"\n   Canal H percentiles [0, 25, 50, 75, 100]:")
    print(f"      {h_percentiles}")

    print(f"\n   Canal V percentiles [0, 25, 50, 75, 100]:")
    print(f"      {v_percentiles}")

    # Valeurs non-nulles (dans les noyaux uniquement)
    h_nonzero = h_values[h_values != 0]
    v_nonzero = v_values[v_values != 0]

    if len(h_nonzero) > 0:
        print(f"\n   Valeurs NON-NULLES (pixels de noyaux):")
        print(f"      H: {len(h_nonzero)} pixels ({len(h_nonzero)/len(h_values)*100:.2f}%)")
        print(f"         Range: [{h_nonzero.min():.4f}, {h_nonzero.max():.4f}]")
        print(f"      V: {len(v_nonzero)} pixels ({len(v_nonzero)/len(v_values)*100:.2f}%)")
        print(f"         Range: [{v_nonzero.min():.4f}, {v_nonzero.max():.4f}]")

    return {
        'is_normalized': is_normalized,
        'range': (h_channel.min(), h_channel.max()),
        'dtype': hv_targets.dtype,
        'shape': hv_targets.shape,
    }


def compare_normalization(old_path: Path, new_path: Path):
    """Compare OLD vs NEW normalization."""

    print("\n" + "="*70)
    print("COMPARAISON NORMALISATION OLD vs NEW")
    print("="*70)

    old_info = diagnose_hv_data(old_path, "OLD DATA")
    new_info = diagnose_hv_data(new_path, "NEW DATA")

    if old_info and new_info:
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)

        if old_info['is_normalized'] and new_info['is_normalized']:
            print("\n✅ COMPATIBLES: Les deux datasets sont normalisés [-1, 1]")
            print("   → Comparaison directe possible")

        elif not old_info['is_normalized'] and new_info['is_normalized']:
            print("\n⚠️  INCOMPATIBLES:")
            print(f"   OLD: NON normalisé (range {old_info['range']})")
            print(f"   NEW: Normalisé (range {new_info['range']})")
            print("\n   🔧 SOLUTIONS POSSIBLES:")
            print("   1. Normaliser OLD pour comparaison:")
            print("      hv_old_normalized = hv_old / max(abs(hv_old))")
            print("   2. Ré-entraîner avec NEW (données correctes)")
            print("      ⚠️  Modèle actuel incompatible avec NEW")

        elif old_info['is_normalized'] and not new_info['is_normalized']:
            print("\n❌ ERREUR: NEW devrait être normalisé mais ne l'est pas!")
            print("   → Bug dans prepare_family_data_FIXED.py")

        else:
            print("\n⚠️  Les deux datasets sont NON normalisés")
            print("   → Comparaison possible mais non conforme à HoVer-Net")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnostic normalisation HV maps")
    parser.add_argument("--old_data", type=str, required=True, help="Path to OLD data")
    parser.add_argument("--new_data", type=str, required=True, help="Path to NEW data")

    args = parser.parse_args()

    old_path = Path(args.old_data)
    new_path = Path(args.new_data)

    compare_normalization(old_path, new_path)

    print("\n" + "="*70)
    print("RECOMMANDATIONS")
    print("="*70)
    print("""
Selon la littérature HoVer-Net (Graham et al., 2019):

    HV maps doivent être normalisés à [-1, 1] où:
    - H[x,y] = (x - cx) / max_dist  ∈ [-1, 1]
    - V[x,y] = (y - cy) / max_dist  ∈ [-1, 1]

    cx, cy = centre de l'instance
    max_dist = rayon de l'instance (ou dimension max)

Si OLD n'est pas normalisé:
    → C'est un BUG dans l'ancien prepare_family_data.py
    → Le modèle actuel a appris sur données incorrectes
    → Ré-entraînement avec NEW est NÉCESSAIRE

Si NEW n'est pas normalisé:
    → BUG dans prepare_family_data_FIXED.py
    → Corriger avant entraînement
    """)


if __name__ == "__main__":
    main()
