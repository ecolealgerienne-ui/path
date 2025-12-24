#!/usr/bin/env python3
"""
Visualisation diagnostic pour comprendre pourquoi AJI est à 0.0002.

Affiche pour un échantillon:
1. Image H&E originale  
2. GT instances (colorisé)
3. Prédiction NP (binaire)
4. Prédiction HV magnitude
5. Prédiction instances (après watershed)
6. Statistiques détaillées

Permet de diagnostiquer:
- Watershed échoue à séparer les instances?
- HV magnitude trop faible?
- Resize détruit les instances?
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader
from src.metrics.ground_truth_metrics import compute_aji
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from cv2 import resize, INTER_NEAREST

def post_process_hv(np_pred: np.ndarray, hv_pred: np.ndarray, np_threshold: float = 0.5) -> tuple:
    """
    Watershed sur HV maps avec retour de tous les intermédiaires.
    
    Returns:
        (instance_map, energy, markers, n_instances)
    """
    # Binary mask
    binary_mask = (np_pred > np_threshold).astype(np.uint8)
    
    if not binary_mask.any():
        return np.zeros_like(np_pred, dtype=np.int32), np.zeros_like(np_pred), np.zeros_like(np_pred), 0
    
    # HV energy (magnitude)
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
    instance_map, n_instances = ndimage.label(instance_map > 0)
    
    return instance_map, energy, markers, n_instances

def colorize_instances(inst_map: np.ndarray) -> np.ndarray:
    """Colorise instance map avec couleurs aléatoires."""
    if inst_map.max() == 0:
        return np.zeros((*inst_map.shape, 3), dtype=np.uint8)
    
    np.random.seed(42)
    colors = np.random.randint(0, 255, (inst_map.max() + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background noir
    
    colored = colors[inst_map]
    return colored

def main():
    parser = argparse.ArgumentParser(description="Diagnostic visuel AJI")
    parser.add_argument("--family", required=True, choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"])
    parser.add_argument("--checkpoint", required=True, help="Chemin checkpoint HoVer-Net")
    parser.add_argument("--data_dir", default="data/family_data", help="Répertoire données features")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index échantillon à visualiser")
    parser.add_argument("--output", default="results/diagnostic_visual.png", help="Fichier de sortie")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Charger données
    print("Chargement données...")
    features_data = np.load(data_dir / f"{args.family}_features.npz")
    targets_data = np.load(data_dir / f"{args.family}_targets.npz")
    fixed_data = np.load(Path("data/family_FIXED") / f"{args.family}_data_FIXED.npz")
    
    features = features_data['features']
    np_targets = targets_data['np_targets']
    hv_targets = targets_data['hv_targets']
    inst_maps = targets_data.get('inst_maps', fixed_data['inst_maps'])
    images = fixed_data['images']
    
    idx = args.sample_idx
    
    print(f"\nÉchantillon {idx}/{len(features)}")
    print("=" * 80)
    
    # Charger modèle
    print("Chargement modèle...")
    hovernet = ModelLoader.load_hovernet(args.checkpoint, device=args.device)
    hovernet.eval()
    
    # Prédiction
    feat = torch.from_numpy(features[idx:idx+1]).to(args.device).float()
    
    with torch.no_grad():
        np_out, hv_out, nt_out = hovernet(feat)
    
    np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]  # (224, 224)
    hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)
    
    # Post-processing
    inst_pred, energy, markers, n_pred = post_process_hv(np_pred, hv_pred)
    
    # GT (resize 256 → 224)
    inst_gt = resize(inst_maps[idx], (224, 224), interpolation=INTER_NEAREST)
    image = resize(images[idx], (224, 224))
    
    # Statistiques
    n_gt = len(np.unique(inst_gt)) - 1  # Hors background
    aji = compute_aji(inst_pred, inst_gt)
    
    print(f"\nSTATISTIQUES:")
    print(f"  GT instances:   {n_gt}")
    print(f"  Pred instances: {n_pred}")
    print(f"  AJI:            {aji:.4f}")
    print(f"  NP Dice:        {2 * ((np_pred > 0.5) & (np_targets[idx] > 0.5)).sum() / ((np_pred > 0.5).sum() + (np_targets[idx] > 0.5).sum()):.4f}")
    print(f"  HV energy:      min={energy.min():.3f}, max={energy.max():.3f}, mean={energy.mean():.3f}")
    print(f"  Markers trouvés: {markers.max()}")
    print("")
    
    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Image H&E originale\n(Sample {idx})")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(colorize_instances(inst_gt))
    axes[0, 1].set_title(f"GT Instances\n({n_gt} instances)")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np_pred, cmap='gray')
    axes[0, 2].set_title(f"Prédiction NP\n(Dice={2 * ((np_pred > 0.5) & (np_targets[idx] > 0.5)).sum() / ((np_pred > 0.5).sum() + (np_targets[idx] > 0.5).sum()):.3f})")
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(energy, cmap='hot')
    axes[1, 0].set_title(f"HV Magnitude (Energy)\nRange: [{energy.min():.2f}, {energy.max():.2f}]")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(markers > 0, cmap='gray')
    axes[1, 1].set_title(f"Watershed Markers\n({markers.max()} markers)")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(colorize_instances(inst_pred))
    axes[1, 2].set_title(f"Prédiction Instances\n({n_pred} instances, AJI={aji:.3f})")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    
    # Diagnostic
    print("\nDIAGNOSTIC:")
    if n_pred == 0:
        print("  ❌ PROBLÈME: Aucune instance prédite!")
        print("     → Watershed n'a trouvé aucun marqueur")
        print("     → Vérifier: HV magnitude trop faible?")
    elif n_pred == 1 and n_gt > 1:
        print("  ❌ PROBLÈME: 1 instance géante prédite!")
        print("     → Watershed n'a pas séparé les cellules")
        print("     → Vérifier: dist_threshold trop élevé? (actuel: 2)")
    elif n_pred < 0.5 * n_gt:
        print("  ⚠️  SOUS-SEGMENTATION sévère!")
        print(f"     → Pred {n_pred} vs GT {n_gt} (ratio {n_pred/n_gt:.2f})")
        print("     → Essayer: dist_threshold=1, min_size=5")
    elif n_pred > 2 * n_gt:
        print("  ⚠️  SUR-SEGMENTATION sévère!")
        print(f"     → Pred {n_pred} vs GT {n_gt} (ratio {n_pred/n_gt:.2f})")
        print("     → Essayer: dist_threshold=3, min_size=20")
    else:
        print(f"  ✅ Nombre instances OK (ratio {n_pred/n_gt:.2f})")
        print(f"     → Mais AJI faible ({aji:.4f}) indique mauvais alignement")
        print("     → Vérifier: resize détruit les instances?")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
