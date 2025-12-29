#!/usr/bin/env python3
"""
Visualisation des Erreurs de Frontières (Option D).

Ce script affiche côte à côte:
- Image RGB originale
- GT instances (colorées)
- Prédictions instances (colorées)
- Overlay des erreurs (rouge = FP, bleu = FN)

Objectif: Identifier si les erreurs sont:
- Décalage systématique (shift de quelques pixels)
- Noyaux trop petits/grands (problème de seuil)
- Erreurs aléatoires (bruit)

Usage:
    python scripts/evaluation/visualize_boundary_errors.py \
        --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_best.pth \
        --family epidermal \
        --n_samples 10 \
        --output_dir results/boundary_analysis
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from scipy.ndimage import label, distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import regionprops
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hovernet_decoder import HoVerNetDecoder


def load_model(checkpoint_path: str, device: str = "cuda"):
    """
    Charge le modèle HoVer-Net avec auto-détection de l'architecture.

    Détecte automatiquement:
    - Mode hybride (présence de ruifrok/h_projection keys)
    - Version V2 (1 H-channel, 65 input) vs V3 (16 H-channels, 80 input)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Auto-detect architecture from checkpoint keys
    has_ruifrok = any('ruifrok' in k for k in state_dict.keys())
    has_h_projection = any('h_projection' in k for k in state_dict.keys())

    # Check head input dimension to determine hybrid version
    # np_head.head.0.weight shape: (out_ch, in_ch, k, k)
    if 'np_head.head.0.weight' in state_dict:
        head_in_channels = state_dict['np_head.head.0.weight'].shape[1]
    else:
        head_in_channels = 64  # Default non-hybrid

    # Determine configuration
    if head_in_channels == 80:
        # V3 hybrid: 64 base + 16 H-channels
        use_hybrid = True
        print(f"  Auto-detected: V3 Hybrid (16 H-channels, {head_in_channels} input)")
    elif head_in_channels == 65:
        # V2 hybrid: 64 base + 1 H-channel
        use_hybrid = True
        print(f"  Auto-detected: V2 Hybrid (1 H-channel, {head_in_channels} input)")
    elif has_ruifrok or has_h_projection:
        # Has hybrid keys but different dimension
        use_hybrid = True
        print(f"  Auto-detected: Hybrid mode ({head_in_channels} input)")
    else:
        # Non-hybrid
        use_hybrid = False
        print(f"  Auto-detected: Non-hybrid ({head_in_channels} input)")

    # Create model with detected configuration
    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=5,
        use_hybrid=use_hybrid,
    )

    # Load state dict
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model, use_hybrid


def hv_to_instances(np_pred: np.ndarray, hv_pred: np.ndarray,
                    beta: float = 1.5, min_size: int = 40) -> np.ndarray:
    """
    Convertit NP + HV maps en carte d'instances via watershed.

    Args:
        np_pred: (H, W) probabilités NP [0, 1]
        hv_pred: (2, H, W) gradients HV [-1, 1]
        beta: Poids des gradients HV dans l'énergie
        min_size: Taille minimum d'une instance

    Returns:
        inst_map: (H, W) carte d'instances (0 = background)
    """
    # Binariser NP
    binary = (np_pred > 0.5).astype(np.uint8)

    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.int32)

    # Magnitude des gradients HV
    h_mag = np.abs(hv_pred[0])
    v_mag = np.abs(hv_pred[1])
    hv_magnitude = np.sqrt(h_mag**2 + v_mag**2)

    # Distance transform
    dist = distance_transform_edt(binary)
    dist = dist / (dist.max() + 1e-6)

    # Énergie combinée: distance - beta * magnitude_HV
    # Les frontières HV ont haute magnitude → énergie basse → barrières
    energy = dist - beta * hv_magnitude
    energy = energy * binary  # Masquer le background

    # Trouver les marqueurs (maxima locaux de l'énergie)
    from scipy.ndimage import maximum_filter
    local_max = (energy == maximum_filter(energy, size=7)) & (energy > 0.1)
    markers, n_markers = label(local_max)

    if n_markers == 0:
        # Fallback: utiliser connected components
        inst_map, _ = label(binary)
        return inst_map.astype(np.int32)

    # Watershed
    inst_map = watershed(-energy, markers, mask=binary)

    # Filtrer les petites instances
    for region in regionprops(inst_map):
        if region.area < min_size:
            inst_map[inst_map == region.label] = 0

    # Relabeler séquentiellement
    unique_labels = np.unique(inst_map)
    unique_labels = unique_labels[unique_labels > 0]
    new_inst_map = np.zeros_like(inst_map)
    for new_id, old_id in enumerate(unique_labels, start=1):
        new_inst_map[inst_map == old_id] = new_id

    return new_inst_map.astype(np.int32)


def create_instance_colormap(n_instances: int):
    """Crée une colormap pour les instances."""
    np.random.seed(42)
    colors = np.random.rand(n_instances + 1, 3)
    colors[0] = [0, 0, 0]  # Background = noir
    return ListedColormap(colors)


def compute_boundary_errors(gt_inst: np.ndarray, pred_inst: np.ndarray):
    """
    Calcule les erreurs de frontières.

    Returns:
        error_map: (H, W, 3) RGB avec:
            - Vert = Correct (TP)
            - Rouge = Faux Positif (prédit mais pas GT)
            - Bleu = Faux Négatif (GT mais pas prédit)
    """
    gt_binary = (gt_inst > 0).astype(np.uint8)
    pred_binary = (pred_inst > 0).astype(np.uint8)

    # TP, FP, FN
    tp = gt_binary & pred_binary
    fp = pred_binary & ~gt_binary
    fn = gt_binary & ~pred_binary

    # Créer image RGB
    error_map = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
    error_map[tp > 0] = [0, 255, 0]    # Vert = correct
    error_map[fp > 0] = [255, 0, 0]    # Rouge = FP
    error_map[fn > 0] = [0, 0, 255]    # Bleu = FN

    return error_map, tp.sum(), fp.sum(), fn.sum()


def compute_boundary_shift(gt_inst: np.ndarray, pred_inst: np.ndarray):
    """
    Analyse le décalage des frontières.

    Calcule pour chaque noyau GT:
    - Le décalage du centroïde (dx, dy)
    - La différence de taille (ratio)

    Returns:
        shifts: Liste de (dx, dy, size_ratio) pour chaque noyau
    """
    shifts = []

    gt_props = regionprops(gt_inst)
    pred_props = regionprops(pred_inst)

    # Créer dict des centroïdes prédits
    pred_centroids = {p.label: p.centroid for p in pred_props}
    pred_areas = {p.label: p.area for p in pred_props}

    for gt_prop in gt_props:
        gt_centroid = np.array(gt_prop.centroid)
        gt_area = gt_prop.area

        # Trouver le noyau prédit qui correspond (overlap maximal)
        gt_mask = (gt_inst == gt_prop.label)
        overlapping_labels = pred_inst[gt_mask]
        overlapping_labels = overlapping_labels[overlapping_labels > 0]

        if len(overlapping_labels) == 0:
            # Noyau non détecté
            shifts.append({
                'dx': None, 'dy': None, 'size_ratio': 0,
                'status': 'missed'
            })
            continue

        # Label le plus fréquent
        best_label = np.bincount(overlapping_labels).argmax()
        if best_label == 0:
            best_label = overlapping_labels[overlapping_labels > 0][0] if len(overlapping_labels[overlapping_labels > 0]) > 0 else 0

        if best_label > 0 and best_label in pred_centroids:
            pred_centroid = np.array(pred_centroids[best_label])
            pred_area = pred_areas[best_label]

            dx = pred_centroid[1] - gt_centroid[1]  # x = col
            dy = pred_centroid[0] - gt_centroid[0]  # y = row
            size_ratio = pred_area / gt_area

            shifts.append({
                'dx': dx, 'dy': dy, 'size_ratio': size_ratio,
                'status': 'matched'
            })
        else:
            shifts.append({
                'dx': None, 'dy': None, 'size_ratio': 0,
                'status': 'missed'
            })

    return shifts


def visualize_sample(image: np.ndarray, gt_inst: np.ndarray, pred_inst: np.ndarray,
                     output_path: Path, sample_idx: int):
    """
    Crée une visualisation complète pour un échantillon.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Image RGB originale
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image RGB")
    axes[0, 0].axis('off')

    # 2. GT instances
    n_gt = gt_inst.max()
    cmap_gt = create_instance_colormap(n_gt)
    axes[0, 1].imshow(gt_inst, cmap=cmap_gt, interpolation='nearest')
    axes[0, 1].set_title(f"GT Instances (n={n_gt})")
    axes[0, 1].axis('off')

    # 3. Prédiction instances
    n_pred = pred_inst.max()
    cmap_pred = create_instance_colormap(n_pred)
    axes[0, 2].imshow(pred_inst, cmap=cmap_pred, interpolation='nearest')
    axes[0, 2].set_title(f"Pred Instances (n={n_pred})")
    axes[0, 2].axis('off')

    # 4. Overlay erreurs
    error_map, tp, fp, fn = compute_boundary_errors(gt_inst, pred_inst)
    axes[1, 0].imshow(error_map)
    axes[1, 0].set_title(f"Erreurs: TP(vert)={tp}, FP(rouge)={fp}, FN(bleu)={fn}")
    axes[1, 0].axis('off')

    # 5. Overlay sur image
    overlay = image.copy()
    overlay[error_map[:,:,0] > 0] = [255, 100, 100]  # FP = rouge clair
    overlay[error_map[:,:,2] > 0] = [100, 100, 255]  # FN = bleu clair
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Erreurs sur Image")
    axes[1, 1].axis('off')

    # 6. Analyse des shifts
    shifts = compute_boundary_shift(gt_inst, pred_inst)
    matched = [s for s in shifts if s['status'] == 'matched']

    if matched:
        dx_vals = [s['dx'] for s in matched]
        dy_vals = [s['dy'] for s in matched]
        size_ratios = [s['size_ratio'] for s in matched]

        # Scatter plot des décalages
        axes[1, 2].scatter(dx_vals, dy_vals, c=size_ratios, cmap='RdYlGn',
                          vmin=0.5, vmax=1.5, s=50, alpha=0.7)
        axes[1, 2].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].axvline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Δx (pixels)')
        axes[1, 2].set_ylabel('Δy (pixels)')
        axes[1, 2].set_title(f"Décalage Centroïdes (n={len(matched)})\n"
                            f"Δx={np.mean(dx_vals):.2f}±{np.std(dx_vals):.2f}, "
                            f"Δy={np.mean(dy_vals):.2f}±{np.std(dy_vals):.2f}")
        axes[1, 2].set_xlim(-10, 10)
        axes[1, 2].set_ylim(-10, 10)
        axes[1, 2].set_aspect('equal')

        # Colorbar pour size_ratio
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0.5, 1.5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[1, 2], label='Size Ratio (pred/GT)')
    else:
        axes[1, 2].text(0.5, 0.5, "Pas de correspondance", ha='center', va='center')
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / f"sample_{sample_idx:03d}_analysis.png", dpi=150)
    plt.close()

    return {
        'n_gt': n_gt,
        'n_pred': n_pred,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'shifts': shifts
    }


def main():
    parser = argparse.ArgumentParser(description="Visualisation des erreurs de frontières")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Chemin vers le checkpoint du modèle")
    parser.add_argument("--family", type=str, default="epidermal",
                        help="Famille à analyser")
    parser.add_argument("--data_dir", type=str,
                        default="data/family_data_v13_smart_crops",
                        help="Répertoire des données V13")
    parser.add_argument("--features_dir", type=str,
                        default="data/cache/family_data",
                        help="Répertoire des features RGB")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Nombre d'échantillons à visualiser")
    parser.add_argument("--output_dir", type=str, default="results/boundary_analysis",
                        help="Répertoire de sortie")
    parser.add_argument("--beta", type=float, default=1.5,
                        help="Paramètre beta du watershed")
    parser.add_argument("--min_size", type=int, default=40,
                        help="Taille minimum d'instance")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda ou cpu)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger le modèle (auto-détection de l'architecture)
    print(f"Chargement du modèle: {args.checkpoint}")
    model, use_hybrid = load_model(args.checkpoint, args.device)

    # Charger les données de validation
    data_path = Path(args.data_dir) / f"{args.family}_val_v13_smart_crops.npz"
    features_path = Path(args.features_dir) / f"{args.family}_rgb_features_v13_smart_crops_val.npz"

    print(f"Chargement données: {data_path}")
    data = np.load(data_path)
    images = data['images']
    np_targets = data['np_targets']
    hv_targets = data['hv_targets']
    inst_maps = data['inst_maps']

    print(f"Chargement features: {features_path}")
    features_data = np.load(features_path)
    features = features_data['features']

    n_samples = min(args.n_samples, len(images))
    print(f"\nAnalyse de {n_samples} échantillons...")

    all_stats = []
    all_shifts = []

    for i in range(n_samples):
        print(f"\n[{i+1}/{n_samples}] Traitement échantillon {i}...")

        # Préparer l'input
        image = images[i]  # (224, 224, 3)
        gt_inst = inst_maps[i]  # (224, 224)
        feat = features[i]  # (261, 1536)

        # Inférence
        feat_tensor = torch.from_numpy(feat).unsqueeze(0).float().to(args.device)

        if use_hybrid:
            # Préparer image RGB pour hybrid
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(args.device)

            with torch.no_grad():
                np_out, hv_out, nt_out = model(feat_tensor, images_rgb=image_tensor)
        else:
            with torch.no_grad():
                np_out, hv_out, nt_out = model(feat_tensor)

        # Post-traitement
        np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 1]  # Proba classe "noyau"
        hv_pred = hv_out.cpu().numpy()[0]  # (2, H, W)

        # Convertir en instances
        pred_inst = hv_to_instances(np_pred, hv_pred, beta=args.beta, min_size=args.min_size)

        # Visualiser
        stats = visualize_sample(image, gt_inst, pred_inst, output_dir, i)
        all_stats.append(stats)
        all_shifts.extend([s for s in stats['shifts'] if s['status'] == 'matched'])

        print(f"   GT: {stats['n_gt']} instances, Pred: {stats['n_pred']} instances")
        print(f"   TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}")

    # Résumé global
    print("\n" + "="*70)
    print("RÉSUMÉ GLOBAL")
    print("="*70)

    total_gt = sum(s['n_gt'] for s in all_stats)
    total_pred = sum(s['n_pred'] for s in all_stats)
    total_tp = sum(s['tp'] for s in all_stats)
    total_fp = sum(s['fp'] for s in all_stats)
    total_fn = sum(s['fn'] for s in all_stats)

    print(f"\nInstances: GT={total_gt}, Pred={total_pred}, Ratio={total_pred/total_gt:.2f}×")
    print(f"Pixels: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Dice pixel-wise: {2*total_tp / (2*total_tp + total_fp + total_fn):.4f}")

    if all_shifts:
        dx_all = [s['dx'] for s in all_shifts]
        dy_all = [s['dy'] for s in all_shifts]
        size_all = [s['size_ratio'] for s in all_shifts]

        print(f"\n📊 ANALYSE DES DÉCALAGES ({len(all_shifts)} noyaux matchés):")
        print(f"   Δx: {np.mean(dx_all):.2f} ± {np.std(dx_all):.2f} pixels")
        print(f"   Δy: {np.mean(dy_all):.2f} ± {np.std(dy_all):.2f} pixels")
        print(f"   Size ratio: {np.mean(size_all):.3f} ± {np.std(size_all):.3f}")

        # Diagnostic
        print(f"\n🔍 DIAGNOSTIC:")

        mean_dx, mean_dy = np.mean(dx_all), np.mean(dy_all)
        std_dx, std_dy = np.std(dx_all), np.std(dy_all)
        mean_size = np.mean(size_all)

        if abs(mean_dx) > 2 or abs(mean_dy) > 2:
            print(f"   ⚠️  DÉCALAGE SYSTÉMATIQUE détecté!")
            print(f"       Les prédictions sont décalées de ({mean_dx:.1f}, {mean_dy:.1f}) pixels")
            print(f"       → Vérifier l'alignement features/targets")
        elif std_dx > 3 or std_dy > 3:
            print(f"   ⚠️  VARIANCE ÉLEVÉE dans les décalages")
            print(f"       → Frontières inconsistantes, ajuster beta watershed")
        else:
            print(f"   ✅ Décalages faibles et cohérents")

        if mean_size < 0.8:
            print(f"   ⚠️  NOYAUX PRÉDITS TROP PETITS (ratio={mean_size:.2f})")
            print(f"       → Augmenter le seuil NP ou réduire min_size")
        elif mean_size > 1.2:
            print(f"   ⚠️  NOYAUX PRÉDITS TROP GRANDS (ratio={mean_size:.2f})")
            print(f"       → Réduire le seuil NP ou augmenter min_size")
        else:
            print(f"   ✅ Taille des noyaux correcte (ratio={mean_size:.2f})")

    print(f"\n📁 Visualisations sauvegardées dans: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
