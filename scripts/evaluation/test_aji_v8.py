#!/usr/bin/env python3
"""
Test rapide AJI sur données v8 pour valider amélioration 0.06 → 0.60+
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader
from src.metrics.ground_truth_metrics import compute_aji, compute_panoptic_quality
from src.inference.optimus_gate_inference_multifamily import post_process_hovernet

def main():
    parser = argparse.ArgumentParser(description="Test AJI sur données v8")
    parser.add_argument("--family", required=True, choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"])
    parser.add_argument("--checkpoint", required=True, help="Chemin checkpoint HoVer-Net")
    parser.add_argument("--data_dir", default="data/family_data", help="Répertoire données features")
    parser.add_argument("--n_samples", type=int, default=50, help="Nombre échantillons à tester")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Charger features et targets
    features_file = data_dir / f"{args.family}_features.npz"
    targets_file = data_dir / f"{args.family}_targets.npz"

    if not features_file.exists() or not targets_file.exists():
        print(f"❌ ERREUR: Fichiers manquants dans {data_dir}")
        return 1

    print("=" * 80)
    print(f"TEST AJI v8 - FAMILLE {args.family.upper()}")
    print("=" * 80)
    print("")

    # Charger données
    print(f"Chargement données...")
    features_data = np.load(features_file)
    targets_data = np.load(targets_file)

    features = features_data['features']
    np_targets = targets_data['np_targets']
    hv_targets = targets_data['hv_targets']

    n_total = len(features)
    n_test = min(args.n_samples, n_total)

    print(f"  → {n_total} échantillons disponibles")
    print(f"  → Test sur {n_test} échantillons")
    print("")

    # Charger modèle
    print(f"Chargement modèle: {args.checkpoint}")
    hovernet = ModelLoader.load_hovernet(args.checkpoint, device=args.device)
    hovernet.eval()
    print("")

    # Test sur échantillons
    print(f"Évaluation AJI...")
    aji_scores = []
    pq_scores = []

    for i in tqdm(range(n_test)):
        # Features (B, 261, 1536)
        feat = torch.from_numpy(features[i:i+1]).to(args.device).float()

        # Prédiction
        with torch.no_grad():
            np_out, hv_out, nt_out = hovernet(feat)

        # Convertir en numpy
        np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]  # (224, 224)
        hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

        # Ground truth (256, 256) → resize à (224, 224)
        from cv2 import resize, INTER_NEAREST
        np_gt = resize(np_targets[i], (224, 224), interpolation=INTER_NEAREST)
        hv_gt = np.stack([
            resize(hv_targets[i, 0], (224, 224)),
            resize(hv_targets[i, 1], (224, 224))
        ])

        # Post-processing pour instances
        try:
            inst_pred = post_process_hovernet(np_pred, hv_pred)

            # GT instances (connectedComponents sur NP mask)
            import cv2
            np_gt_binary = (np_gt > 0.5).astype(np.uint8)
            _, inst_gt = cv2.connectedComponents(np_gt_binary)

            # Calculer AJI
            aji = compute_aji(inst_pred, inst_gt)
            aji_scores.append(aji)

            # Calculer PQ
            pq_result = compute_panoptic_quality(inst_pred, inst_gt)
            pq_scores.append(pq_result['pq'])

        except Exception as e:
            print(f"\n⚠️  Échantillon {i}: {e}")
            continue

    # Résultats
    print("")
    print("=" * 80)
    print("RÉSULTATS")
    print("=" * 80)
    print("")

    if len(aji_scores) > 0:
        aji_mean = np.mean(aji_scores)
        aji_std = np.std(aji_scores)
        pq_mean = np.mean(pq_scores)
        pq_std = np.std(pq_scores)

        print(f"AJI:  {aji_mean:.4f} ± {aji_std:.4f}")
        print(f"PQ:   {pq_mean:.4f} ± {pq_std:.4f}")
        print("")

        # Verdict
        print("=" * 80)
        print("VERDICT")
        print("=" * 80)
        print("")

        if aji_mean >= 0.60:
            print("✅ SUCCÈS: AJI >0.60 - Objectif atteint!")
            print(f"   Amélioration v7→v8: 0.06 → {aji_mean:.2f} (gain +{(aji_mean/0.06 - 1)*100:.0f}%)")
            print("")
            print("👉 RECOMMANDATION: Extraire features + entraîner les 4 autres familles")
        elif aji_mean >= 0.50:
            print("🟡 BON: AJI >0.50 - Amélioration significative")
            print(f"   Amélioration v7→v8: 0.06 → {aji_mean:.2f} (gain +{(aji_mean/0.06 - 1)*100:.0f}%)")
            print("")
            print("👉 RECOMMANDATION: Continuer les autres familles, viser optimisations post-processing")
        else:
            print(f"❌ ÉCHEC: AJI {aji_mean:.4f} <0.50")
            print(f"   Objectif non atteint (cible: 0.06 → 0.60+)")
            print("")
            print("👉 RECOMMANDATION: Investiguer pourquoi v8 ne donne pas les gains attendus")
            print("   - Vérifier post-processing (watershed params)")
            print("   - Vérifier que inst_maps sont bien utilisés")
            print("   - Comparer visuellement prédictions vs GT")

        print("")
    else:
        print("❌ ERREUR: Aucun échantillon traité avec succès")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
