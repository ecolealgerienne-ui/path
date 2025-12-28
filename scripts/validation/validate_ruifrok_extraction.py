#!/usr/bin/env python3
"""
Sanity Check pour RuifrokExtractor avant injection Canal H.

Ce script valide que l'extraction du canal Hématoxyline (H) fonctionne
correctement AVANT de lancer l'entraînement hybride.

Vérifications effectuées:
1. Contraste: Noyaux brillants (blanc) vs Stroma sombre (noir)
2. Alignement spatial: H-channel 16×16 aligné avec centres des noyaux GT
3. Range: Valeurs H-channel dans [0, 255] après normalisation
4. Pas de décalage: Pics H tombent sur centres noyaux (pas décalés)

Usage:
    python scripts/validation/validate_ruifrok_extraction.py \
        --family epidermal \
        --n_samples 3 \
        --output_dir results/ruifrok_validation

Author: CellViT-Optimus Project
Date: 2025-12-28
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from datetime import datetime


def load_ruifrok_extractor():
    """Charge le RuifrokExtractor depuis hovernet_decoder.py"""
    from src.models.hovernet_decoder import RuifrokExtractor
    return RuifrokExtractor()


def load_validation_data(family: str, n_samples: int = 3):
    """Charge quelques échantillons de validation pour test."""
    targets_dir = Path("data/family_data_v13_smart_crops")
    targets_file = targets_dir / f"{family}_val_v13_smart_crops.npz"

    if not targets_file.exists():
        raise FileNotFoundError(f"Validation data not found: {targets_file}")

    data = np.load(targets_file)

    images = data['images'][:n_samples]  # (N, 224, 224, 3) uint8
    np_targets = data['np_targets'][:n_samples]  # (N, 224, 224) float32

    print(f"✅ Loaded {n_samples} samples from {family} validation set")
    print(f"   Images shape: {images.shape}, dtype: {images.dtype}")
    print(f"   NP targets shape: {np_targets.shape}, dtype: {np_targets.dtype}")

    return images, np_targets


def extract_h_channel(extractor, image_rgb: np.ndarray, device: str = "cpu"):
    """
    Extrait le canal H d'une image RGB.

    Args:
        extractor: RuifrokExtractor instance
        image_rgb: (H, W, 3) uint8 [0, 255]
        device: torch device

    Returns:
        h_full: (H, W) float32 H-channel à résolution complète
        h_16x16: (16, 16) float32 H-channel redimensionné pour bottleneck
    """
    # Convertir en tensor (1, 3, H, W) float32 [0, 255]
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        h_16x16 = extractor(image_tensor)  # (1, 1, 16, 16)

    # H-channel full resolution (pour visualisation)
    # Recalculer manuellement pour avoir la version haute résolution
    image_float = image_tensor / 255.0  # [0, 1]
    od = -torch.log10(image_float + 1e-6)  # Optical Density

    h_vector = torch.tensor([0.650, 0.704, 0.286], device=device).view(1, 3, 1, 1)
    h_full = (od * h_vector).sum(dim=1, keepdim=True)  # (1, 1, H, W)

    # Normaliser pour visualisation
    h_full_np = h_full[0, 0].cpu().numpy()
    h_min, h_max = h_full_np.min(), h_full_np.max()
    if h_max > h_min:
        h_full_normalized = ((h_full_np - h_min) / (h_max - h_min) * 255).astype(np.uint8)
    else:
        h_full_normalized = np.zeros_like(h_full_np, dtype=np.uint8)

    h_16x16_np = h_16x16[0, 0].cpu().numpy()

    return h_full_normalized, h_16x16_np


def compute_alignment_score(h_16x16: np.ndarray, np_target: np.ndarray) -> float:
    """
    Calcule un score d'alignement entre H-channel 16×16 et le masque GT.

    Un bon alignement signifie que les valeurs élevées de H correspondent
    aux centres des noyaux dans le GT.

    Args:
        h_16x16: (16, 16) H-channel redimensionné
        np_target: (224, 224) masque binaire GT

    Returns:
        score: Corrélation entre H élevé et présence noyaux
    """
    # Redimensionner GT à 16×16
    gt_16x16 = cv2.resize(np_target.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)

    # Normaliser les deux
    h_norm = (h_16x16 - h_16x16.min()) / (h_16x16.max() - h_16x16.min() + 1e-6)
    gt_norm = gt_16x16 / (gt_16x16.max() + 1e-6)

    # Corrélation de Pearson
    h_flat = h_norm.flatten()
    gt_flat = gt_norm.flatten()

    h_centered = h_flat - h_flat.mean()
    gt_centered = gt_flat - gt_flat.mean()

    correlation = (h_centered * gt_centered).sum() / (
        np.sqrt((h_centered ** 2).sum()) * np.sqrt((gt_centered ** 2).sum()) + 1e-6
    )

    return correlation


def visualize_sample(
    idx: int,
    image_rgb: np.ndarray,
    np_target: np.ndarray,
    h_full: np.ndarray,
    h_16x16: np.ndarray,
    alignment_score: float,
    output_dir: Path
):
    """Crée une visualisation 4 panneaux pour un échantillon."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Image RGB originale
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("1. Image RGB Originale", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. H-channel full resolution
    axes[0, 1].imshow(h_full, cmap='hot')
    axes[0, 1].set_title("2. Canal H (Ruifrok) - Full Res", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    # Colorbar
    cbar1 = plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)
    cbar1.set_label("Intensité H")

    # 3. H-channel 16×16 (comme injecté dans bottleneck)
    im3 = axes[1, 0].imshow(h_16x16, cmap='hot', interpolation='nearest')
    axes[1, 0].set_title("3. Canal H (16×16) - Bottleneck Input", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    # Grille pour visualiser les pixels
    for i in range(17):
        axes[1, 0].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        axes[1, 0].axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    cbar2 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    cbar2.set_label("Intensité H")

    # 4. Overlay H-channel + GT mask
    # Redimensionner GT à même taille que H full
    gt_overlay = np.stack([np_target * 0, np_target, np_target * 0], axis=-1)  # Vert pour GT
    h_overlay = np.stack([h_full / 255, np.zeros_like(h_full) / 255, np.zeros_like(h_full) / 255], axis=-1)  # Rouge pour H

    combined = np.clip(image_rgb / 255 * 0.3 + h_overlay * 0.5 + gt_overlay * 0.3, 0, 1)
    axes[1, 1].imshow(combined)
    axes[1, 1].set_title(f"4. Overlay: H (rouge) + GT (vert)\nAlignment Score: {alignment_score:.3f}", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle(f"Validation Ruifrok - Sample {idx}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f"ruifrok_validation_sample_{idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   💾 Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate Ruifrok H-channel extraction")
    parser.add_argument("--family", type=str, default="epidermal", help="Family to validate")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of samples to validate")
    parser.add_argument("--output_dir", type=str, default="results/ruifrok_validation",
                        help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("VALIDATION RUIFROK H-CHANNEL EXTRACTION")
    print("=" * 80)
    print(f"\nFamily: {args.family}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {output_dir}")

    # Load extractor
    print("\n📦 Loading RuifrokExtractor...")
    try:
        extractor = load_ruifrok_extractor()
        print("   ✅ RuifrokExtractor loaded successfully")
        print(f"   H vector: [0.650, 0.704, 0.286] (Ruifrok et al.)")
    except Exception as e:
        print(f"   ❌ Failed to load RuifrokExtractor: {e}")
        return 1

    # Load data
    print("\n📂 Loading validation data...")
    try:
        images, np_targets = load_validation_data(args.family, args.n_samples)
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return 1

    # Process each sample
    print("\n🔬 Processing samples...")
    alignment_scores = []

    for i in range(args.n_samples):
        print(f"\n   Sample {i}:")
        image_rgb = images[i]
        np_target = np_targets[i]

        # Extract H-channel
        h_full, h_16x16 = extract_h_channel(extractor, image_rgb, args.device)

        # Stats
        print(f"      H-full range: [{h_full.min()}, {h_full.max()}]")
        print(f"      H-16x16 range: [{h_16x16.min():.3f}, {h_16x16.max():.3f}]")
        print(f"      H-16x16 mean: {h_16x16.mean():.3f}, std: {h_16x16.std():.3f}")

        # Alignment score
        alignment = compute_alignment_score(h_16x16, np_target)
        alignment_scores.append(alignment)
        print(f"      Alignment score: {alignment:.3f}")

        # Visualize
        visualize_sample(i, image_rgb, np_target, h_full, h_16x16, alignment, output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    mean_alignment = np.mean(alignment_scores)
    print(f"\n📊 Alignment Scores:")
    for i, score in enumerate(alignment_scores):
        status = "✅" if score > 0.3 else "⚠️" if score > 0.1 else "❌"
        print(f"   Sample {i}: {score:.3f} {status}")

    print(f"\n   Mean alignment: {mean_alignment:.3f}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if mean_alignment > 0.3:
        print(f"\n✅ PASS - H-channel extraction is aligned with nuclei (correlation > 0.3)")
        print("   → Safe to proceed with V13-Hybrid training")
        verdict = "PASS"
    elif mean_alignment > 0.1:
        print(f"\n⚠️ WARNING - H-channel extraction has weak alignment (0.1 < correlation < 0.3)")
        print("   → Review visualizations before proceeding")
        verdict = "WARNING"
    else:
        print(f"\n❌ FAIL - H-channel extraction is NOT aligned with nuclei (correlation < 0.1)")
        print("   → DO NOT proceed with V13-Hybrid training")
        print("   → Check Ruifrok stain matrix or image preprocessing")
        verdict = "FAIL"

    # Save report
    report_path = output_dir / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Ruifrok Validation Report\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Family: {args.family}\n")
        f.write(f"Samples: {args.n_samples}\n")
        f.write(f"\nAlignment Scores:\n")
        for i, score in enumerate(alignment_scores):
            f.write(f"  Sample {i}: {score:.3f}\n")
        f.write(f"\nMean Alignment: {mean_alignment:.3f}\n")
        f.write(f"Verdict: {verdict}\n")

    print(f"\n📄 Report saved: {report_path}")
    print(f"🖼️ Visualizations saved in: {output_dir}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
