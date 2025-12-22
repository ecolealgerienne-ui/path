#!/usr/bin/env python3
"""
Diagnostic: Inspecter la distribution des prédictions
pour comprendre pourquoi Dice = 0.08 au lieu de 0.95
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader
from src.preprocessing import preprocess_image, validate_features
from src.constants import PANNUKE_IMAGE_SIZE

def diagnose_predictions(
    sample_path: str,
    checkpoint_path: str,
    device: str = "cuda"
):
    """Inspecte les prédictions brutes d'un échantillon."""

    print("=" * 80)
    print("DIAGNOSTIC: Distribution des Prédictions")
    print("=" * 80)
    print("")

    # Charger échantillon
    data = np.load(sample_path)
    images = data['images']
    masks = data['masks']

    sample_idx = 0
    image = images[sample_idx]
    mask = masks[sample_idx]

    print(f"Échantillon: {sample_path}")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print("")

    # Préparer ground truth
    np_gt = mask[:, :, 1:].sum(axis=-1) > 0
    n_pixels_gt = np_gt.sum()
    print(f"Ground Truth: {n_pixels_gt} pixels de noyaux ({n_pixels_gt / (256*256) * 100:.1f}%)")
    print("")

    # Charger modèle
    print(f"Chargement modèle: {checkpoint_path}")
    hovernet = ModelLoader.load_hovernet(checkpoint_path, device=device)
    hovernet.eval()

    print("Chargement backbone H-optimus-0...")
    backbone = ModelLoader.load_hoptimus0(device=device)
    print("")

    # Inférence
    print("Inférence...")
    tensor = preprocess_image(image, device=device)

    with torch.no_grad():
        features = backbone.forward_features(tensor)
        validation = validate_features(features)
        if not validation["valid"]:
            print(f"⚠️  ATTENTION: {validation['message']}")
            print("")

        patch_tokens = features[:, 1:257, :]
        np_out, hv_out, nt_out = hovernet(patch_tokens)

    # Analyser NP (Nuclear Presence)
    np_logits = np_out.cpu().numpy()[0, 0]  # Avant sigmoid
    np_probs = torch.sigmoid(np_out).cpu().numpy()[0, 0]  # Après sigmoid

    print("=" * 80)
    print("ANALYSE NP (Nuclear Presence)")
    print("=" * 80)
    print("")
    print(f"Shape: {np_probs.shape}")
    print("")
    print(f"Logits bruts (avant sigmoid):")
    print(f"  min   = {np_logits.min():.3f}")
    print(f"  max   = {np_logits.max():.3f}")
    print(f"  mean  = {np_logits.mean():.3f}")
    print(f"  std   = {np_logits.std():.3f}")
    print("")
    print(f"Probabilités (après sigmoid):")
    print(f"  min   = {np_probs.min():.3f}")
    print(f"  max   = {np_probs.max():.3f}")
    print(f"  mean  = {np_probs.mean():.3f}")
    print(f"  std   = {np_probs.std():.3f}")
    print("")

    # Analyse seuil
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("Pixels > seuil:")
    for thresh in thresholds:
        n_pixels = (np_probs > thresh).sum()
        pct = n_pixels / (224*224) * 100
        print(f"  > {thresh:.1f}: {n_pixels:6d} pixels ({pct:5.1f}%)")
    print("")

    # Intersection avec GT (après resize)
    from src.utils.image_utils import prepare_predictions_for_evaluation

    np_pred_256, _, _ = prepare_predictions_for_evaluation(
        np_probs,
        hv_out.cpu().numpy()[0],
        torch.softmax(nt_out, dim=1).cpu().numpy()[0],
        target_size=PANNUKE_IMAGE_SIZE
    )

    pred_binary = np_pred_256 > 0.5
    n_pixels_pred = pred_binary.sum()
    intersection = (pred_binary & np_gt).sum()

    print(f"Après resize 224→256 et seuil 0.5:")
    print(f"  Pixels prédits : {n_pixels_pred} ({n_pixels_pred / (256*256) * 100:.1f}%)")
    print(f"  Pixels GT      : {n_pixels_gt} ({n_pixels_gt / (256*256) * 100:.1f}%)")
    print(f"  Intersection   : {intersection} ({intersection / n_pixels_gt * 100:.1f}% du GT)")
    print("")

    dice = 2 * intersection / (n_pixels_pred + n_pixels_gt + 1e-8)
    print(f"Dice Score: {dice:.4f}")
    print("")

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print("")

    if np_probs.mean() < 0.1:
        print("❌ PROBLÈME CRITIQUE: Probabilités très faibles (mean < 0.1)")
        print("   → Le modèle prédit 'pas de noyau' presque partout")
        print("   → Causes possibles:")
        print("     1. Modèle mal entraîné (loss non convergée)")
        print("     2. Mismatch preprocessing train vs eval")
        print("     3. Checkpoint corrompu")
        print("")
    elif n_pixels_pred < n_pixels_gt * 0.1:
        print("❌ PROBLÈME: Très peu de pixels prédits")
        print(f"   → Seulement {n_pixels_pred} prédits vs {n_pixels_gt} attendus")
        print("   → Le modèle est trop conservateur (seuil 0.5 trop élevé ?)")
        print("")
    elif dice < 0.5:
        print("⚠️  PROBLÈME: Dice faible malgré nombre de pixels correct")
        print("   → Les pixels prédits ne sont PAS au bon endroit")
        print("   → Causes possibles:")
        print("     1. Spatial mismatch (décalage de grille)")
        print("     2. Resize interpolation incorrecte")
        print("")
    else:
        print("✅ Prédictions semblent correctes")
        print(f"   → Dice = {dice:.4f}")
        print("")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_predictions_distribution.py <sample.npz> <checkpoint.pth>")
        print("")
        print("Exemple:")
        print("  python diagnose_predictions_distribution.py \\")
        print("    results/family_validation_20251222_153551/test_samples/glandular/test_samples.npz \\")
        print("    models/checkpoints/hovernet_glandular_best.pth")
        sys.exit(1)

    sample_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    diagnose_predictions(sample_path, checkpoint_path)
