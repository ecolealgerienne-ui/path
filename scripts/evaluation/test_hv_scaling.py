#!/usr/bin/env python3
"""
Test de scaling HV pour améliorer AJI.

Suit la recommandation expert: "Multiplie tes prédictions HV par un facteur (ex: 10 ou 50)
avant le Watershed pour voir si l'AJI remonte."
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

# Import model
from src.models.loader import ModelLoader
from src.constants import DEFAULT_FAMILY_DATA_DIR


def post_process_hv_scaled(np_pred: np.ndarray, hv_pred: np.ndarray,
                           hv_scale: float = 1.0, np_threshold: float = 0.5) -> np.ndarray:
    """
    Watershed avec scaling HV.

    Args:
        np_pred: Nuclear presence mask (H, W) float [0, 1]
        hv_pred: HV maps (2, H, W) float [-1, 1]
        hv_scale: Facteur de multiplication HV (TEST 1.0, 10.0, 50.0)
        np_threshold: Seuil binarisation NP

    Returns:
        instance_map: (H, W) int32
    """
    # Binary mask
    binary_mask = (np_pred > np_threshold).astype(np.uint8)

    if not binary_mask.any():
        return np.zeros_like(np_pred, dtype=np.int32)

    # SCALE HV MAPS (TEST)
    hv_scaled = hv_pred * hv_scale

    # HV energy (magnitude)
    energy = np.sqrt(hv_scaled[0]**2 + hv_scaled[1]**2)

    print(f"  HV scale: {hv_scale:5.1f}x - Energy range: [{energy.min():.4f}, {energy.max():.4f}], mean: {energy.mean():.4f}")

    # Find local maxima
    dist_threshold = 2
    local_max = peak_local_max(
        energy,
        min_distance=dist_threshold,
        labels=binary_mask.astype(int),
        exclude_border=False,
    )

    print(f"                    Peaks found: {len(local_max)}")

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


def compute_aji(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    """Calcule AJI (Aggregated Jaccard Index)."""
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]

    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0

    # Matched pairs
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    for i, gt_id in enumerate(gt_ids):
        gt_mask = gt_inst == gt_id
        for j, pred_id in enumerate(pred_ids):
            pred_mask = pred_inst == pred_id
            intersection = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0

    # Match greedy
    matched_gt = set()
    matched_pred = set()
    used_sum = 0.0

    while True:
        max_iou = iou_matrix.max()
        if max_iou == 0:
            break

        max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        i, j = max_idx

        if max_iou >= 0.5:
            used_sum += max_iou
            matched_gt.add(i)
            matched_pred.add(j)

        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    # Unused
    unused_gt = len(gt_ids) - len(matched_gt)
    unused_pred = len(pred_ids) - len(matched_pred)

    aji = used_sum / (used_sum + unused_gt + unused_pred)
    return aji


def main():
    """Test HV scaling sur epidermal."""

    print("\n" + "="*80)
    print("TEST HV SCALING - Amélioration AJI")
    print("="*80)

    # Load model
    checkpoint = Path("models/checkpoints/hovernet_epidermal_best.pth")
    if not checkpoint.exists():
        print(f"❌ Checkpoint introuvable: {checkpoint}")
        return 1

    print(f"\n📦 Chargement modèle: {checkpoint.name}")
    hovernet = ModelLoader.load_hovernet(checkpoint, device="cuda")
    hovernet.eval()

    # Load data
    data_dir = Path(DEFAULT_FAMILY_DATA_DIR)
    targets_path = data_dir / "epidermal_targets.npz"
    features_path = data_dir / "epidermal_features.npz"

    if not targets_path.exists() or not features_path.exists():
        print(f"❌ Données introuvables dans {data_dir}")
        return 1

    print(f"📁 Chargement données: {data_dir.name}")

    targets_data = np.load(targets_path)
    features_data = np.load(features_path)

    # Test sur 10 échantillons
    n_samples = min(10, len(targets_data['np_targets']))

    # Test différents scaling factors
    scale_factors = [1.0, 5.0, 10.0, 20.0, 50.0]

    results = {scale: [] for scale in scale_factors}

    print(f"\n🧪 Test sur {n_samples} échantillons avec {len(scale_factors)} facteurs de scaling")
    print("─"*80)

    for idx in range(n_samples):
        print(f"\n📊 Échantillon {idx + 1}/{n_samples}")

        # Get data
        features = torch.from_numpy(features_data['features'][idx:idx+1]).cuda().float()

        np_target = targets_data['np_targets'][idx]
        hv_target = targets_data['hv_targets'][idx]
        inst_target = targets_data['inst_maps'][idx]

        # Inférence
        with torch.no_grad():
            np_out, hv_out, nt_out = hovernet(features)

        # Convertir
        np_pred = torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]  # (224, 224)
        hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

        # Resize 224 → 256
        from skimage.transform import resize
        np_pred_256 = resize(np_pred, (256, 256), order=1, preserve_range=True, anti_aliasing=False)
        hv_pred_256 = np.stack([
            resize(hv_pred[0], (256, 256), order=1, preserve_range=True, anti_aliasing=False),
            resize(hv_pred[1], (256, 256), order=1, preserve_range=True, anti_aliasing=False)
        ])

        # Test chaque scaling factor
        for scale in scale_factors:
            inst_pred = post_process_hv_scaled(np_pred_256, hv_pred_256, hv_scale=scale)
            aji = compute_aji(inst_pred, inst_target)
            results[scale].append(aji)

    # Résultats
    print("\n" + "="*80)
    print("RÉSULTATS")
    print("="*80)
    print(f"\n{'Scale':>8s} | {'AJI Mean':>10s} | {'AJI Std':>10s} | {'Amélioration':>15s}")
    print("─"*80)

    baseline_aji = np.mean(results[1.0])

    for scale in scale_factors:
        ajis = results[scale]
        mean_aji = np.mean(ajis)
        std_aji = np.std(ajis)
        improvement = ((mean_aji - baseline_aji) / baseline_aji * 100) if baseline_aji > 0 else 0

        status = ""
        if scale == 1.0:
            status = " (baseline)"
        elif mean_aji > baseline_aji:
            status = f" (✅ +{improvement:.1f}%)"
        else:
            status = f" (❌ {improvement:.1f}%)"

        print(f"{scale:8.1f} | {mean_aji:10.4f} | {std_aji:10.4f} | {status}")

    # Recommandation
    print("\n" + "="*80)
    best_scale = max(scale_factors, key=lambda s: np.mean(results[s]))
    best_aji = np.mean(results[best_scale])
    improvement = ((best_aji - baseline_aji) / baseline_aji * 100) if baseline_aji > 0 else 0

    print(f"🎯 RECOMMANDATION:")
    print(f"   Meilleur scaling: {best_scale}x")
    print(f"   AJI: {baseline_aji:.4f} → {best_aji:.4f} (+{improvement:.1f}%)")

    if best_aji >= 0.60:
        print(f"\n✅ OBJECTIF ATTEINT: AJI {best_aji:.4f} >= 0.60")
    else:
        print(f"\n⚠️ OBJECTIF NON ATTEINT: AJI {best_aji:.4f} < 0.60")
        print(f"   Le scaling améliore mais ne suffit pas seul.")
        print(f"   → Investiguer Bug #3 (instance mismatch)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
