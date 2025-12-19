#!/usr/bin/env python3
"""
Évaluation CellViT-256 sur PanNuke Fold-3 (test set).

Usage:
    python scripts/validation/evaluate_pannuke.py \
        --checkpoint models/pretrained/CellViT-256.pth \
        --data_path ~/data/PanNuke_processed \
        --fold 2  # fold2 = test set dans la convention CellViT

Critère POC: Dice > 0.7
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from tabulate import tabulate

# Ajouter les chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CellViT"))

# Utiliser nos propres métriques
from scripts.evaluation.metrics_segmentation import dice_score, iou_score


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Charge CellViT-256 depuis le checkpoint."""
    from models.segmentation.cell_segmentation.cellvit import CellViT256

    print(f"Chargement du modèle depuis {checkpoint_path}...")

    # Créer le modèle
    model = CellViT256(
        model256_path=None,
        num_nuclei_classes=6,  # Background + 5 types
        num_tissue_classes=19,  # 19 organes PanNuke
    )

    # Charger les poids
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"✅ Modèle chargé: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def load_pannuke_fold(data_path: str, fold: int):
    """Charge un fold PanNuke."""
    fold_path = Path(data_path) / f"fold{fold}"
    images_path = fold_path / "images"
    labels_path = fold_path / "labels"

    if not images_path.exists():
        raise FileNotFoundError(f"Dossier images non trouvé: {images_path}")

    image_files = sorted(images_path.glob("*.png"))
    print(f"Trouvé {len(image_files)} images dans fold{fold}")

    return image_files, labels_path


def evaluate(
    model,
    image_files: list,
    labels_path: Path,
    device: str = "cuda",
    batch_size: int = 16,
):
    """Évalue le modèle sur les images."""

    # Normalisation (comme dans le training CellViT)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    binary_dice_scores = []
    binary_jaccard_scores = []
    tissue_results = {}

    # Mapping des tissus
    tissue_names = [
        "adrenal_gland", "bile-duct", "bladder", "breast", "cervix",
        "colon", "esophagus", "headneck", "kidney", "liver",
        "lung", "ovarian", "pancreatic", "prostate", "skin",
        "stomach", "testis", "thyroid", "uterus"
    ]

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Évaluation"):
            # Charger l'image
            img = np.array(Image.open(img_file).convert("RGB"))
            img_normalized = (img / 255.0 - mean) / std
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float().unsqueeze(0)
            img_tensor = img_tensor.to(device)

            # Charger le label
            label_file = labels_path / f"{img_file.stem}.npy"
            if not label_file.exists():
                continue
            label_data = np.load(label_file, allow_pickle=True).item()
            inst_map_gt = label_data["inst_map"]
            type_map_gt = label_data["type_map"]

            # Inférence
            predictions = model.forward(img_tensor)

            # Binary map prediction
            nuclei_binary_map = F.softmax(predictions["nuclei_binary_map"], dim=1)
            pred_binary = torch.argmax(nuclei_binary_map[0], dim=0).cpu().numpy()

            # Ground truth binary
            gt_binary = (inst_map_gt > 0).astype(np.uint8)

            # Calcul Dice (utilise notre fonction numpy)
            cell_dice = dice_score(pred_binary, gt_binary)
            binary_dice_scores.append(float(cell_dice))

            # Calcul Jaccard/IoU
            cell_jaccard = iou_score(pred_binary, gt_binary)
            binary_jaccard_scores.append(float(cell_jaccard))

            # Identifier le tissu depuis le nom du fichier
            # Format: fold_idx.png -> on utilise les métadonnées si disponibles

    # Calcul des moyennes
    results = {
        "Binary-Cell-Dice-Mean": np.mean(binary_dice_scores),
        "Binary-Cell-Dice-Std": np.std(binary_dice_scores),
        "Binary-Cell-Jaccard-Mean": np.mean(binary_jaccard_scores),
        "Binary-Cell-Jaccard-Std": np.std(binary_jaccard_scores),
        "Num-Images": len(binary_dice_scores),
    }

    return results, binary_dice_scores


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation CellViT-256 sur PanNuke"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Chemin vers CellViT-256.pth"
    )
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        required=True,
        help="Chemin vers PanNuke_processed"
    )
    parser.add_argument(
        "--fold", "-f",
        type=int,
        default=2,
        help="Fold à évaluer (0, 1, ou 2). Default: 2 (test set)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU à utiliser"
    )

    args = parser.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  ÉVALUATION CELLVIT-256 SUR PANNUKE")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data path:  {args.data_path}")
    print(f"Fold:       {args.fold}")
    print(f"Device:     {device}")
    print("=" * 60)

    # Charger le modèle
    model = load_model(args.checkpoint, device)

    # Charger les données
    image_files, labels_path = load_pannuke_fold(args.data_path, args.fold)

    # Évaluer
    results, dice_scores = evaluate(model, image_files, labels_path, device)

    # Afficher les résultats
    print("\n" + "=" * 60)
    print("  RÉSULTATS")
    print("=" * 60)

    dice_mean = results["Binary-Cell-Dice-Mean"]
    dice_std = results["Binary-Cell-Dice-Std"]
    jaccard_mean = results["Binary-Cell-Jaccard-Mean"]

    print(f"\nBinary-Cell-Dice:    {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"Binary-Cell-Jaccard: {jaccard_mean:.4f}")
    print(f"Images évaluées:     {results['Num-Images']}")

    # Critère POC
    print("\n" + "-" * 60)
    if dice_mean > 0.7:
        print(f"✅ CRITÈRE POC VALIDÉ: Dice {dice_mean:.4f} > 0.7")
    else:
        print(f"❌ CRITÈRE POC NON VALIDÉ: Dice {dice_mean:.4f} <= 0.7")
    print("-" * 60)

    # Distribution des scores
    print("\nDistribution des scores Dice:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(dice_scores, p)
        print(f"  P{p}: {val:.4f}")


if __name__ == "__main__":
    main()
