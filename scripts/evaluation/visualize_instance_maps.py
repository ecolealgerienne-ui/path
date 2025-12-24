#!/usr/bin/env python3
"""
Diagnostic visuel - Instance Maps

Vérifie si le problème est:
1. Giant Blob (toutes cellules fusionnées en 1 instance)
2. ID Mismatch (inst_maps GT décalés ou corrompus)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

from src.models.loader import ModelLoader
from src.constants import DEFAULT_FAMILY_DATA_DIR


def post_process_hv(np_pred: np.ndarray, hv_pred: np.ndarray, np_threshold: float = 0.5) -> np.ndarray:
    """Watershed sur HV maps."""
    binary_mask = (np_pred > np_threshold).astype(np.uint8)

    if not binary_mask.any():
        return np.zeros_like(np_pred, dtype=np.int32)

    # HV energy (magnitude)
    energy = np.sqrt(hv_pred[0]**2 + hv_pred[1]**2)

    # Find local maxima
    dist_threshold = 2
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


def create_colormap(n_colors):
    """Crée colormap avec couleurs distinctes."""
    np.random.seed(42)
    colors = np.random.rand(n_colors, 3)
    colors[0] = [0, 0, 0]  # Background = noir
    return ListedColormap(colors)


def main():
    """Visualise échantillon 9."""

    print("\n" + "="*80)
    print("DIAGNOSTIC VISUEL - ÉCHANTILLON 9")
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

    # Échantillon 9 (index 8)
    idx = 8

    print(f"\n🔍 Analyse échantillon {idx + 1}")
    print("─"*80)

    # Get data
    features = torch.from_numpy(features_data['features'][idx:idx+1]).cuda().float()

    np_target = targets_data['np_targets'][idx]
    hv_target = targets_data['hv_targets'][idx]
    inst_target = targets_data['inst_maps'][idx]

    print(f"Features shape: {features.shape}")
    print(f"NP target shape: {np_target.shape}")
    print(f"HV target shape: {hv_target.shape}")
    print(f"Inst target shape: {inst_target.shape}")

    # Inférence
    with torch.no_grad():
        np_out, hv_out, nt_out = hovernet(features)

    # Convertir
    np_pred = torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]  # (224, 224)
    hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

    print(f"\nNP pred shape: {np_pred.shape}")
    print(f"HV pred shape: {hv_pred.shape}")

    # ⚠️ VÉRIFICATION CRITIQUE: Comment resize?
    print("\n⚠️ VÉRIFICATION RESIZE")
    print("─"*80)

    # Resize 224 → 256 avec INTER_NEAREST (comme recommandé expert)
    import cv2
    np_pred_256 = cv2.resize(np_pred, (256, 256), interpolation=cv2.INTER_NEAREST)
    hv_pred_256 = np.stack([
        cv2.resize(hv_pred[0], (256, 256), interpolation=cv2.INTER_NEAREST),
        cv2.resize(hv_pred[1], (256, 256), interpolation=cv2.INTER_NEAREST)
    ])

    print(f"✅ Resize prédictions 224 → 256 avec INTER_NEAREST")
    print(f"   NP pred 256: {np_pred_256.shape}")
    print(f"   HV pred 256: {hv_pred_256.shape}")

    # Post-processing
    inst_pred = post_process_hv(np_pred_256, hv_pred_256)

    # Statistiques
    print(f"\n📊 STATISTIQUES INSTANCES")
    print("─"*80)

    n_pred_instances = inst_pred.max()
    n_gt_instances = inst_target.max()

    print(f"Instances prédites: {n_pred_instances}")
    print(f"Instances GT:       {n_gt_instances}")

    if n_pred_instances == 1:
        print("\n❌ GIANT BLOB DÉTECTÉ!")
        print("   Toutes les cellules sont fusionnées en 1 instance.")
        print("   → Problème: Watershed ne sépare pas (gradients HV trop faibles)")
    elif n_pred_instances > 50:
        print(f"\n✅ INSTANCES MULTIPLES DÉTECTÉES ({n_pred_instances})")
        print("   Le Watershed fonctionne correctement.")
        print("   → Problème probable: ID Mismatch (GT décalé ou corrompu)")
    else:
        print(f"\n⚠️ SOUS-SEGMENTATION ({n_pred_instances} instances)")
        print("   Moins d'instances que prévu.")
        print("   → Vérifier paramètres watershed (dist_threshold, min_size)")

    # Vérification inst_target
    print(f"\n🔍 VÉRIFICATION INST_TARGET")
    print("─"*80)
    print(f"Dtype: {inst_target.dtype}")
    print(f"Range: [{inst_target.min()}, {inst_target.max()}]")
    print(f"Shape: {inst_target.shape}")
    print(f"Valeurs uniques (premiers 10): {np.unique(inst_target)[:10]}")

    # VISUALISATION
    print(f"\n🎨 CRÉATION VISUALISATIONS")
    print("─"*80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Diagnostic Échantillon {idx + 1} - Instance Maps", fontsize=16, fontweight='bold')

    # Row 1: Prédictions
    # NP binary
    axes[0, 0].imshow(np_pred_256, cmap='gray')
    axes[0, 0].set_title(f"NP Pred (binary)\nRange: [{np_pred_256.min():.3f}, {np_pred_256.max():.3f}]")
    axes[0, 0].axis('off')

    # HV magnitude
    hv_mag = np.sqrt(hv_pred_256[0]**2 + hv_pred_256[1]**2)
    axes[0, 1].imshow(hv_mag, cmap='viridis')
    axes[0, 1].set_title(f"HV Magnitude\nRange: [{hv_mag.min():.4f}, {hv_mag.max():.4f}]")
    axes[0, 1].axis('off')

    # Instances prédites
    cmap_pred = create_colormap(n_pred_instances + 1)
    axes[0, 2].imshow(inst_pred, cmap=cmap_pred, vmin=0, vmax=n_pred_instances)
    axes[0, 2].set_title(f"Instances PRED\n{n_pred_instances} instances")
    axes[0, 2].axis('off')

    # Row 2: Ground Truth
    # NP GT
    axes[1, 0].imshow(np_target, cmap='gray')
    axes[1, 0].set_title(f"NP GT (binary)\nRange: [{np_target.min():.3f}, {np_target.max():.3f}]")
    axes[1, 0].axis('off')

    # HV GT magnitude
    hv_gt_mag = np.sqrt(hv_target[0]**2 + hv_target[1]**2)
    axes[1, 1].imshow(hv_gt_mag, cmap='viridis')
    axes[1, 1].set_title(f"HV GT Magnitude\nRange: [{hv_gt_mag.min():.4f}, {hv_gt_mag.max():.4f}]")
    axes[1, 1].axis('off')

    # Instances GT
    cmap_gt = create_colormap(n_gt_instances + 1)
    axes[1, 2].imshow(inst_target, cmap=cmap_gt, vmin=0, vmax=n_gt_instances)
    axes[1, 2].set_title(f"Instances GT\n{n_gt_instances} instances")
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Sauvegarder
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "diagnostic_instance_maps_sample9.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")

    # Diagnostic final
    print(f"\n" + "="*80)
    print("DIAGNOSTIC FINAL")
    print("="*80)

    if n_pred_instances == 1:
        print("\n❌ PROBLÈME: GIANT BLOB (1 instance au lieu de {})".format(n_gt_instances))
        print("\n🔧 SOLUTION:")
        print("   1. Vérifier que les HV targets sont bien normalisés [-1, 1]")
        print("   2. Vérifier que le décodeur a bien un Tanh() sur HV")
        print("   3. Bug #3 probable: instances fusionnées durant training")
        print("   → Lire CLAUDE.md lignes 745-819")

    elif n_pred_instances > 50 and abs(n_pred_instances - n_gt_instances) > 20:
        print(f"\n⚠️ PROBLÈME: ID MISMATCH")
        print(f"   Pred: {n_pred_instances} instances")
        print(f"   GT:   {n_gt_instances} instances")
        print(f"   Écart: {abs(n_pred_instances - n_gt_instances)}")
        print("\n🔧 SOLUTION:")
        print("   1. Vérifier que inst_target utilise VRAIES instances PanNuke")
        print("   2. Vérifier pas de connectedComponents sur union binaire")
        print("   3. Vérifier resize avec INTER_NEAREST (pas LINEAR)")

    else:
        print(f"\n✅ INSTANCES DÉTECTÉES CORRECTEMENT")
        print(f"   Pred: {n_pred_instances} instances")
        print(f"   GT:   {n_gt_instances} instances")
        print(f"\n   → Le problème est probablement dans le calcul de l'AJI")
        print(f"   → Vérifier la fonction compute_aji()")

    print(f"\n📊 Voir visualisation: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
