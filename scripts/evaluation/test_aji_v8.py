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
from src.constants import DEFAULT_FAMILY_DATA_DIR
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def post_process_hv(np_pred: np.ndarray, hv_pred: np.ndarray, np_threshold: float = 0.5) -> np.ndarray:
    """
    Watershed sur HV maps pour séparer instances (v8 compatible).

    Args:
        np_pred: Nuclear presence mask (H, W) float [0, 1]
        hv_pred: HV maps (2, H, W) float [-1, 1]
        np_threshold: Seuil binarisation NP (défaut: 0.5)

    Returns:
        instance_map: (H, W) int32 avec IDs instances
    """
    # Binary mask
    binary_mask = (np_pred > np_threshold).astype(np.uint8)

    if not binary_mask.any():
        return np.zeros_like(np_pred, dtype=np.int32)

    # HV energy (magnitude) - Original HoVer-Net
    energy = np.sqrt(hv_pred[0]**2 + hv_pred[1]**2)

    # Find local maxima as markers
    dist_threshold = 2  # CONSERVATIVE
    local_max = peak_local_max(
        energy,
        min_distance=dist_threshold,
        labels=binary_mask.astype(int),
        exclude_border=False,
    )

    # Create markers
    markers = np.zeros_like(binary_mask, dtype=int)
    if len(local_max) > 0:
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

    # Watershed
    if markers.max() > 0:
        instance_map = watershed(-energy, markers, mask=binary_mask)
    else:
        instance_map = ndimage.label(binary_mask)[0]

    # Remove small instances
    min_size = 10
    for inst_id in range(1, instance_map.max() + 1):
        if (instance_map == inst_id).sum() < min_size:
            instance_map[instance_map == inst_id] = 0

    # Re-label
    instance_map, _ = ndimage.label(instance_map > 0)

    return instance_map

def main():
    parser = argparse.ArgumentParser(description="Test AJI sur données v8")
    parser.add_argument("--family", required=True, choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"])
    parser.add_argument("--checkpoint", required=True, help="Chemin checkpoint HoVer-Net")
    parser.add_argument("--data_dir", default=DEFAULT_FAMILY_DATA_DIR, help="Répertoire données features")
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

    # ✅ inst_maps DOIVENT être dans targets.npz (Solution B)
    if 'inst_maps' not in targets_data:
        print(f"❌ ERREUR: inst_maps manquants dans targets.npz!")
        print(f"   Ré-exécutez: python scripts/preprocessing/extract_features_from_fixed.py --family {args.family}")
        return 1

    inst_maps = targets_data['inst_maps']  # ✅ Instances NATIVES PanNuke

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
        # NP utilise CrossEntropyLoss (2 canaux: background/foreground)
        np_pred = torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]  # (224, 224) - foreground prob
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
            inst_pred = post_process_hv(np_pred, hv_pred)

            # ✅ GT instances NATIVES PanNuke (pas connectedComponents!)
            from cv2 import resize, INTER_NEAREST
            inst_gt = resize(inst_maps[i], (224, 224), interpolation=INTER_NEAREST)

            # Calculer AJI
            aji = compute_aji(inst_pred, inst_gt)
            aji_scores.append(aji)

            # Calculer PQ (retourne tuple: (PQ, DQ, SQ, PQ_per_class))
            pq_result = compute_panoptic_quality(inst_pred, inst_gt)
            pq_scores.append(pq_result[0])  # PQ est le premier élément

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
