#!/usr/bin/env python3
"""
Teste chaque modèle HoVer-Net par famille de manière isolée.

Ce script charge chaque modèle de famille et le teste UNIQUEMENT sur ses propres données,
permettant d'isoler les problèmes potentiels:
- Modèle mal entraîné → Mauvaises métriques sur ses propres données
- Routage incorrect → Bonnes métriques isolées, mauvaises en multi-famille
- Problème d'intégration → Bonnes métriques partout sauf en production

Usage:
    python scripts/evaluation/test_family_models_isolated.py \
        --test_samples_dir data/test_samples_by_family \
        --checkpoint_dir models/checkpoints \
        --output_dir results/family_validation
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json
from typing import Dict, List, Tuple

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.organ_families import FAMILIES
from src.models.loader import ModelLoader
from src.preprocessing import preprocess_image, validate_features


def compute_hv_maps_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Compute HV targets from binary mask (même méthode que training).

    Returns:
        hv_targets: (2, H, W) float32 dans [-1, 1]
    """
    import cv2

    hv = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)

    if not mask.any():
        return hv

    binary_uint8 = (mask * 255).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary_uint8)

    for label_id in range(1, n_labels):
        instance_mask = labels == label_id
        coords = np.where(instance_mask)

        if len(coords[0]) == 0:
            continue

        cy = coords[0].mean()
        cx = coords[1].mean()

        for y, x in zip(coords[0], coords[1]):
            h_dist = (x - cx)
            v_dist = (y - cy)
            radius = max(np.sqrt(len(coords[0]) / np.pi), 1)
            hv[0, y, x] = h_dist / radius
            hv[1, y, x] = v_dist / radius

    hv = np.clip(hv, -1, 1)
    return hv


def compute_metrics(
    pred: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Calcule les métriques HoVer-Net.

    Args:
        pred: {"np": (H,W), "hv": (2,H,W), "nt": (H,W)}
        gt: {"np": (H,W), "hv": (2,H,W), "nt": (H,W)}

    Returns:
        {"dice": float, "hv_mse": float, "nt_acc": float}
    """
    # NP Dice
    pred_np = pred["np"] > 0.5
    gt_np = gt["np"] > 0

    intersection = (pred_np & gt_np).sum()
    union = pred_np.sum() + gt_np.sum()
    dice = (2 * intersection / (union + 1e-8))

    # HV MSE (uniquement sur les pixels de noyaux)
    mask = gt_np
    if mask.sum() > 0:
        hv_mse = ((pred["hv"][:, mask] - gt["hv"][:, mask]) ** 2).mean()
    else:
        hv_mse = 0.0

    # NT Accuracy (uniquement sur les pixels de noyaux)
    if mask.sum() > 0:
        pred_nt = pred["nt"].argmax(axis=0)
        nt_acc = (pred_nt[mask] == gt["nt"][mask]).mean()
    else:
        nt_acc = 0.0

    return {
        "dice": float(dice),
        "hv_mse": float(hv_mse),
        "nt_acc": float(nt_acc)
    }


def test_family(
    family: str,
    test_samples_dir: Path,
    checkpoint_dir: Path,
    device: str = "cuda"
) -> Dict:
    """
    Teste un modèle de famille sur ses propres données.

    Returns:
        {
            "family": str,
            "n_samples": int,
            "metrics": {"dice": float, "hv_mse": float, "nt_acc": float},
            "per_sample_metrics": List[Dict]
        }
    """
    print(f"\n{'=' * 70}")
    print(f"TEST ISOLÉ: {family.upper()}")
    print(f"{'=' * 70}")

    # Charger échantillons de test
    family_dir = test_samples_dir / family
    samples_file = family_dir / "test_samples.npz"

    if not samples_file.exists():
        print(f"⚠️  Aucun échantillon de test pour {family}")
        return None

    data = np.load(samples_file)
    images = data["images"]
    masks = data["masks"]
    organs = data["organs"]

    n_samples = len(images)
    print(f"Échantillons: {n_samples}")
    print(f"Organes: {list(set(organs))}")

    # Charger le modèle HoVer-Net de cette famille
    checkpoint_path = checkpoint_dir / f"hovernet_{family}_best.pth"

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint manquant: {checkpoint_path}")
        print(f"   Le modèle pour {family} n'a pas été entraîné.")
        return None

    print(f"\nChargement: {checkpoint_path.name}")

    try:
        hovernet = ModelLoader.load_hovernet(
            checkpoint_path=checkpoint_path,
            device=device
        )
        print("✅ Modèle chargé")
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return None

    # Charger backbone
    print("\nChargement H-optimus-0...")
    try:
        backbone = ModelLoader.load_hoptimus0(device=device)
        print("✅ Backbone chargé")
    except Exception as e:
        print(f"❌ Erreur chargement backbone: {e}")
        return None

    # Inférence sur les échantillons
    print(f"\nInférence sur {n_samples} échantillons...")

    all_metrics = []

    for i in tqdm(range(n_samples)):
        image = images[i]
        mask = masks[i]
        organ = str(organs[i])

        # Preprocessing
        tensor = preprocess_image(image, device=device)

        # Extraction features
        with torch.no_grad():
            features = backbone.forward_features(tensor)

            # Validation
            validation = validate_features(features)
            if not validation["valid"]:
                print(f"\n⚠️  Sample {i}: {validation['message']}")
                continue

            # Extraction patch tokens
            patch_tokens = features[:, 1:257, :]  # (1, 256, 1536)

            # HoVer-Net prediction (retourne tuple: np_out, hv_out, nt_out)
            np_out, hv_out, nt_out = hovernet(patch_tokens)

        # Convertir en numpy
        np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]  # (256, 256)
        hv_pred = hv_out.cpu().numpy()[0]  # (2, 256, 256)
        nt_pred = torch.softmax(nt_out, dim=1).cpu().numpy()[0]  # (n_classes, 256, 256)

        # Préparer ground truth
        np_gt = mask[:, :, 1:].sum(axis=-1) > 0  # Binary union
        hv_gt = compute_hv_maps_from_mask(np_gt)

        # NT ground truth (argmax sur canaux 1-5)
        nt_gt = np.zeros((256, 256), dtype=np.int64)
        for c in range(5):
            type_mask = mask[:, :, c + 1] > 0
            nt_gt[type_mask] = c

        # Calculer métriques
        pred = {"np": np_pred, "hv": hv_pred, "nt": nt_pred}
        gt = {"np": np_gt.astype(np.float32), "hv": hv_gt, "nt": nt_gt}

        metrics = compute_metrics(pred, gt)
        metrics["organ"] = organ
        metrics["sample_idx"] = i

        all_metrics.append(metrics)

    # Agréger métriques
    if not all_metrics:
        print("❌ Aucune métrique calculée")
        return None

    avg_dice = np.mean([m["dice"] for m in all_metrics])
    avg_hv_mse = np.mean([m["hv_mse"] for m in all_metrics])
    avg_nt_acc = np.mean([m["nt_acc"] for m in all_metrics])

    std_dice = np.std([m["dice"] for m in all_metrics])
    std_hv_mse = np.std([m["hv_mse"] for m in all_metrics])
    std_nt_acc = np.std([m["nt_acc"] for m in all_metrics])

    print(f"\n{'=' * 70}")
    print(f"RÉSULTATS: {family.upper()}")
    print(f"{'=' * 70}")
    print(f"NP Dice:  {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"HV MSE:   {avg_hv_mse:.4f} ± {std_hv_mse:.4f}")
    print(f"NT Acc:   {avg_nt_acc:.4f} ± {std_nt_acc:.4f}")

    return {
        "family": family,
        "n_samples": n_samples,
        "organs": list(set(organs)),
        "metrics": {
            "dice": {"mean": float(avg_dice), "std": float(std_dice)},
            "hv_mse": {"mean": float(avg_hv_mse), "std": float(std_hv_mse)},
            "nt_acc": {"mean": float(avg_nt_acc), "std": float(std_nt_acc)},
        },
        "per_sample_metrics": all_metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description="Teste chaque modèle de famille isolément"
    )
    parser.add_argument(
        "--test_samples_dir",
        type=Path,
        required=True,
        help="Répertoire avec échantillons de test par famille"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Répertoire avec les checkpoints HoVer-Net"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/family_validation"),
        help="Répertoire de sortie pour les résultats"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device PyTorch"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=FAMILIES,
        help="Familles à tester (défaut: toutes)"
    )

    args = parser.parse_args()

    # Créer output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TEST ISOLÉ DES MODÈLES PAR FAMILLE")
    print("=" * 70)
    print(f"Test samples: {args.test_samples_dir}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Device: {args.device}")
    print(f"Familles: {', '.join(args.families)}")
    print()

    # Tester chaque famille
    results = {}

    for family in args.families:
        result = test_family(
            family=family,
            test_samples_dir=args.test_samples_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device
        )

        if result:
            results[family] = result

            # Sauvegarder résultat individuel
            with open(args.output_dir / f"{family}_results.json", "w") as f:
                json.dump(result, f, indent=2)

    # Rapport comparatif
    print("\n" + "=" * 70)
    print("RAPPORT COMPARATIF")
    print("=" * 70)
    print()
    print(f"{'Famille':<15} {'N':<5} {'Dice':<12} {'HV MSE':<12} {'NT Acc':<12}")
    print("-" * 70)

    for family in FAMILIES:
        if family in results:
            r = results[family]
            dice = r["metrics"]["dice"]
            hv = r["metrics"]["hv_mse"]
            nt = r["metrics"]["nt_acc"]

            print(
                f"{family:<15} "
                f"{r['n_samples']:<5} "
                f"{dice['mean']:.4f}±{dice['std']:.3f}  "
                f"{hv['mean']:.4f}±{hv['std']:.3f}  "
                f"{nt['mean']:.4f}±{nt['std']:.3f}"
            )
        else:
            print(f"{family:<15} {'N/A':<5} {'—':<12} {'—':<12} {'—':<12}")

    # Sauvegarder rapport global
    global_report = {
        "test_samples_dir": str(args.test_samples_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "device": args.device,
        "families_tested": list(results.keys()),
        "results": results,
    }

    with open(args.output_dir / "global_report.json", "w") as f:
        json.dump(global_report, f, indent=2)

    print()
    print(f"✅ Résultats sauvegardés: {args.output_dir}")
    print()

    # Recommandations
    print("=" * 70)
    print("ANALYSE & RECOMMANDATIONS")
    print("=" * 70)
    print()

    for family in FAMILIES:
        if family not in results:
            continue

        r = results[family]
        dice = r["metrics"]["dice"]["mean"]
        hv_mse = r["metrics"]["hv_mse"]["mean"]
        nt_acc = r["metrics"]["nt_acc"]["mean"]

        status = []

        if dice < 0.90:
            status.append(f"⚠️  Dice faible ({dice:.3f})")
        if hv_mse > 0.10:
            status.append(f"⚠️  HV MSE élevé ({hv_mse:.3f})")
        if nt_acc < 0.85:
            status.append(f"⚠️  NT Acc faible ({nt_acc:.3f})")

        if status:
            print(f"{family}:")
            for s in status:
                print(f"  {s}")
            print(f"  → Recommandation: Ré-entraîner avec plus de données ou augmentation")
            print()
        else:
            print(f"✅ {family}: Performances bonnes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
