#!/usr/bin/env python3
"""
INSPECTEUR NPZ - Détecte si les HV maps sont v7 (centrifuges) ou v8 (centripètes)

Méthode:
Pour un pixel en HAUT d'un noyau (y < center_y):
- v7 (centrifuge): y_dist = y - center_y < 0 → v_dist < 0 (vecteur vers HAUT = s'éloigne)
- v8 (centripète): y_dist = center_y - y > 0 → v_dist > 0 (vecteur vers BAS = vers centre)

On peut détecter v7 vs v8 en analysant la corrélation entre position et signe HV.

Usage:
    python scripts/validation/inspect_npz_hv_direction.py --family epidermal
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def inspect_hv_direction(family: str, n_samples: int = 10):
    """
    Inspecte la direction des vecteurs HV dans le NPZ.

    Args:
        family: Nom de la famille
        n_samples: Nombre d'échantillons à inspecter
    """
    print("=" * 80)
    print("INSPECTEUR NPZ - DÉTECTION v7 vs v8")
    print("=" * 80)
    print()

    # Charger NPZ
    data_file = Path(f"data/family_FIXED/{family}_data_FIXED.npz")
    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return 1

    data = np.load(data_file)

    # Timestamp
    import datetime
    timestamp = datetime.datetime.fromtimestamp(data_file.stat().st_mtime)
    print(f"NPZ: {data_file}")
    print(f"Créé: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Échantillonner
    total_samples = len(data['images'])
    n_samples = min(n_samples, total_samples)
    sample_indices = np.random.choice(total_samples, n_samples, replace=False)

    v7_votes = 0  # Vecteurs centrifuges
    v8_votes = 0  # Vecteurs centripètes

    print("Analyse des vecteurs HV:")
    print("-" * 80)

    for idx in sample_indices:
        hv_map = data['hv_targets'][idx]  # (2, 256, 256)
        np_target = data['np_targets'][idx]  # (256, 256)

        # Créer instance map
        labeled_mask, n_instances = label(np_target > 0)
        inst_ids = np.unique(labeled_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = labeled_mask == inst_id
            y_coords, x_coords = np.where(inst_mask)

            if len(y_coords) < 10:
                continue

            # Centroïde
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)

            # Sélectionner pixels au-DESSUS du centre (y < center_y)
            top_pixels = y_coords < center_y
            if np.sum(top_pixels) < 5:
                continue

            y_top = y_coords[top_pixels]
            x_top = x_coords[top_pixels]

            # Valeurs HV pour ces pixels
            v_vals = hv_map[0, y_top, x_top]  # Vertical
            h_vals = hv_map[1, y_top, x_top]  # Horizontal

            # Pour pixels au-DESSUS du centre:
            # - v7 (centrifuge): vecteur pointe vers HAUT → v_vals < 0 (majorité)
            # - v8 (centripète): vecteur pointe vers BAS (vers centre) → v_vals > 0 (majorité)

            mean_v = np.mean(v_vals)

            if mean_v < 0:
                v7_votes += 1  # Centrifuge
            else:
                v8_votes += 1  # Centripète

    # Verdict
    print()
    print("=" * 80)
    print("RÉSULTATS:")
    print("=" * 80)
    print(f"Votes v7 (centrifuge): {v7_votes}")
    print(f"Votes v8 (centripète): {v8_votes}")
    print()

    if v8_votes > v7_votes:
        print("✅ VERDICT: NPZ contient données v8 (CENTRIPÈTE)")
        print("   → Les vecteurs HV pointent VERS les centroïdes")
        print()
        print("   Si alignement toujours NO-GO, le problème est ailleurs:")
        print("   - Vérifier script de vérification")
        print("   - Vérifier Hungarian matching")
        print("   - Vérifier extraction des centroïdes")
    else:
        print("❌ VERDICT: NPZ contient données v7 (CENTRIFUGE)")
        print("   → Les vecteurs HV pointent LOIN des centroïdes")
        print()
        print("   CAUSES POSSIBLES:")
        print("   1. Mauvais script utilisé pour régénération")
        print("      Vérifier: prepare_family_data_FIXED_v8.py (PAS v7!)")
        print()
        print("   2. Script v8 a un bug dans compute_hv_maps()")
        print("      Vérifier lignes 70-75:")
        print("      y_dist = center_y - y_coords  # ✅ Correct")
        print("      x_dist = center_x - x_coords  # ✅ Correct")
        print()
        print("   COMMANDE CORRECTE:")
        print("   python scripts/preprocessing/prepare_family_data_FIXED_v8.py \\")
        print("       --family epidermal")

    print("=" * 80)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Inspecte direction HV dans NPZ")
    parser.add_argument("--family", type=str, required=True, help="Famille à inspecter")
    parser.add_argument("--n_samples", type=int, default=10, help="Nombre d'échantillons")

    args = parser.parse_args()

    return inspect_hv_direction(args.family, args.n_samples)


if __name__ == "__main__":
    exit(main())
