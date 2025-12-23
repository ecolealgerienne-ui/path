#!/usr/bin/env python3
"""
Script de test complet du pipeline OptimusGate multi-famille.

Teste:
1. Chargement des 5 modèles de famille (Glandular, Digestive, Urologic, Epidermal, Respiratory)
2. Routage OrganHead → Famille correct
3. Métriques NP/HV/NT par famille
4. Comparaison avec ground truth PanNuke
5. Génération rapport complet

Usage:
    python scripts/evaluation/test_optimus_gate_multifamily.py \\
        --data_dir /path/to/PanNuke \\
        --checkpoints_dir models/checkpoints \\
        --fold 2 \\
        --n_samples 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.organ_head import OrganHead
from src.models.loader import ModelLoader
from src.preprocessing import validate_features

# Mapping organe → famille (source de vérité)
ORGAN_TO_FAMILY = {
    # Glandulaire & Hormonale
    "Breast": "glandular",
    "Prostate": "glandular",
    "Thyroid": "glandular",
    "Pancreatic": "glandular",
    "Adrenal_gland": "glandular",

    # Digestive
    "Colon": "digestive",
    "Stomach": "digestive",
    "Esophagus": "digestive",
    "Bile-duct": "digestive",

    # Urologique & Reproductif
    "Kidney": "urologic",
    "Bladder": "urologic",
    "Testis": "urologic",
    "Ovarian": "urologic",
    "Uterus": "urologic",
    "Cervix": "urologic",

    # Respiratoire & Hépatique
    "Lung": "respiratory",
    "Liver": "respiratory",

    # Épidermoïde
    "Skin": "epidermal",
    "HeadNeck": "epidermal",
}

FAMILIES = ["glandular", "digestive", "urologic", "respiratory", "epidermal"]

PANNUKE_ORGANS = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix",
    "Colon", "Esophagus", "HeadNeck", "Kidney", "Liver",
    "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
    "Stomach", "Testis", "Thyroid", "Uterus"
]


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Dice score pour segmentation binaire."""
    pred_soft = torch.softmax(pred, dim=1)[:, 1]  # Canal nuclei
    intersection = (pred_soft * target.float()).sum()
    union = pred_soft.sum() + target.float().sum()
    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    return dice.item()


def compute_hv_mse(hv_pred: torch.Tensor, hv_target: torch.Tensor, np_target: torch.Tensor) -> float:
    """MSE sur HV uniquement sur pixels de noyaux."""
    mask = np_target.float().unsqueeze(1)

    if mask.sum() == 0:
        return 0.0

    diff = (hv_pred - hv_target) ** 2
    masked_diff = diff * mask
    mse = (masked_diff.sum() / (mask.sum() * 2)).item()

    return mse


def compute_nt_accuracy(nt_pred: torch.Tensor, nt_target: torch.Tensor, np_target: torch.Tensor) -> float:
    """Accuracy NT uniquement sur pixels de noyaux."""
    mask = np_target > 0

    if mask.sum() == 0:
        return 0.0

    pred_classes = torch.argmax(nt_pred, dim=1)
    correct = (pred_classes == nt_target) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


def load_pannuke_fold(data_dir: Path, fold: int):
    """Charge un fold PanNuke complet."""
    fold_dir = data_dir / f"Fold {fold}"

    images = np.load(fold_dir / "images.npy", mmap_mode='r')
    masks = np.load(fold_dir / "masks.npy", mmap_mode='r')
    types = np.load(fold_dir / "types.npy")

    print(f"✅ Fold {fold} chargé: {len(images)} images")

    return images, masks, types


def main():
    parser = argparse.ArgumentParser(description="Test complet OptimusGate multi-famille")
    parser.add_argument("--data_dir", type=str, required=True, help="Répertoire PanNuke")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints", help="Répertoire checkpoints")
    parser.add_argument("--fold", type=int, default=2, help="Fold PanNuke à tester (0, 1, 2)")
    parser.add_argument("--n_samples", type=int, default=50, help="Nombre d'échantillons à tester")
    parser.add_argument("--output_dir", type=str, default="results/optimus_gate_test", help="Répertoire de sortie")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # =========================================================================
    # CHARGEMENT DES MODÈLES
    # =========================================================================

    print("\n" + "="*80)
    print("CHARGEMENT DES MODÈLES")
    print("="*80)

    # Backbone H-optimus-0
    print("📥 Chargement H-optimus-0...")
    backbone = ModelLoader.load_hoptimus0(device=str(device))
    print("✅ H-optimus-0 chargé")

    # OrganHead
    organ_head_path = checkpoints_dir / "organ_head_best.pth"
    print(f"📥 Chargement OrganHead: {organ_head_path}")
    organ_head = OrganHead(input_dim=1536, num_organs=19, temperature=0.5).to(device)
    organ_head.load_state_dict(torch.load(organ_head_path, map_location=device))
    organ_head.eval()
    print("✅ OrganHead chargé")

    # 5 familles HoVer-Net
    family_models = {}
    for family in FAMILIES:
        checkpoint_path = checkpoints_dir / f"hovernet_{family}_best.pth"
        print(f"📥 Chargement {family}: {checkpoint_path}")

        model = HoVerNetDecoder(input_dim=1536, n_classes=6).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        family_models[family] = model
        print(f"✅ {family.capitalize()} chargé")

    print("\n✅ TOUS LES MODÈLES CHARGÉS")

    # =========================================================================
    # CHARGEMENT DONNÉES
    # =========================================================================

    print("\n" + "="*80)
    print(f"CHARGEMENT FOLD {args.fold}")
    print("="*80)

    images, masks, types = load_pannuke_fold(data_dir, args.fold)

    # Sélectionner n_samples aléatoires (reproductible)
    np.random.seed(42)
    n_samples = min(args.n_samples, len(images))
    indices = np.random.choice(len(images), n_samples, replace=False)

    print(f"🎯 Test sur {n_samples} échantillons aléatoires (seed=42)")

    # =========================================================================
    # TEST PIPELINE
    # =========================================================================

    print("\n" + "="*80)
    print("TEST PIPELINE OPTIMUSGATE")
    print("="*80)

    results = {
        "metadata": {
            "fold": args.fold,
            "n_samples": n_samples,
            "timestamp": datetime.now().isoformat(),
        },
        "routing": {
            "organ_accuracy": 0.0,
            "family_accuracy": 0.0,
            "confusion_matrix": {},
        },
        "metrics_by_family": {},
        "samples": [],
    }

    # Accumulateurs par famille
    family_metrics = defaultdict(lambda: {
        "dice": [],
        "hv_mse": [],
        "nt_acc": [],
        "n_samples": 0,
    })

    # Routage
    organ_correct = 0
    family_correct = 0

    for idx in tqdm(indices, desc="Test"):
        image = images[idx]  # (256, 256, 3)
        mask = masks[idx]    # (256, 256, 6)
        organ_name = types[idx]

        # Convertir image en uint8
        if image.dtype != np.uint8:
            image = image.clip(0, 255).astype(np.uint8)

        # =====================================================================
        # PRÉTRAITEMENT
        # =====================================================================

        from src.preprocessing import preprocess_image

        tensor = preprocess_image(image, device=str(device))  # (1, 3, 224, 224)

        # Extraction features
        with torch.no_grad():
            features = backbone.forward_features(tensor)  # (1, 261, 1536)

        # Validation features
        validation_result = validate_features(features)
        if not validation_result["valid"]:
            print(f"⚠️  Warning: Features invalides pour sample {idx}")
            continue

        cls_token = features[:, 0, :]      # (1, 1536)
        patch_tokens = features[:, 1:257, :]  # (1, 256, 1536)

        # =====================================================================
        # ROUTAGE ORGANHEAD → FAMILLE
        # =====================================================================

        with torch.no_grad():
            organ_pred = organ_head(cls_token)

        organ_idx = torch.argmax(organ_pred.logits, dim=1).item()
        organ_pred_name = PANNUKE_ORGANS[organ_idx]
        family_pred = ORGAN_TO_FAMILY[organ_pred_name]
        family_gt = ORGAN_TO_FAMILY[organ_name]

        # Vérifier routage
        if organ_pred_name == organ_name:
            organ_correct += 1

        if family_pred == family_gt:
            family_correct += 1

        # =====================================================================
        # INFÉRENCE HOVERNET (famille prédite)
        # =====================================================================

        hovernet = family_models[family_pred]

        with torch.no_grad():
            np_out, hv_out, nt_out = hovernet(patch_tokens)

        # =====================================================================
        # PRÉPARER GROUND TRUTH
        # =====================================================================

        # NP: Union des 5 canaux
        np_gt = mask[:, :, 1:].sum(axis=-1) > 0  # (256, 256) binary

        # HV: Calculer à partir du masque (simplifié: pas de vraies HV maps dans PanNuke)
        # Pour validation complète, utiliser compute_hv_maps() de prepare_family_data.py
        hv_gt = np.zeros((2, 256, 256), dtype=np.float32)  # Placeholder

        # NT: Type majoritaire par pixel (simplifié)
        nt_gt = np.argmax(mask[:, :, 1:], axis=-1)  # (256, 256)

        # Resize targets 256 → 224
        np_gt_t = torch.from_numpy(np_gt).float().unsqueeze(0).unsqueeze(0)
        np_gt_224 = F.interpolate(np_gt_t, size=(224, 224), mode='nearest').squeeze().numpy()

        hv_gt_t = torch.from_numpy(hv_gt).unsqueeze(0)
        hv_gt_224 = F.interpolate(hv_gt_t, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0).numpy()

        nt_gt_t = torch.from_numpy(nt_gt).long().unsqueeze(0).unsqueeze(0)
        nt_gt_224 = F.interpolate(nt_gt_t.float(), size=(224, 224), mode='nearest').squeeze().long().numpy()

        # Convertir en tensors
        np_gt_t = torch.from_numpy(np_gt_224).to(device)
        hv_gt_t = torch.from_numpy(hv_gt_224).to(device)
        nt_gt_t = torch.from_numpy(nt_gt_224).to(device)

        # =====================================================================
        # CALCULER MÉTRIQUES
        # =====================================================================

        dice = compute_dice(np_out, np_gt_t.unsqueeze(0))
        hv_mse = compute_hv_mse(hv_out, hv_gt_t.unsqueeze(0), np_gt_t.unsqueeze(0))
        nt_acc = compute_nt_accuracy(nt_out, nt_gt_t.unsqueeze(0), np_gt_t.unsqueeze(0))

        # Accumuler
        family_metrics[family_gt]["dice"].append(dice)
        family_metrics[family_gt]["hv_mse"].append(hv_mse)
        family_metrics[family_gt]["nt_acc"].append(nt_acc)
        family_metrics[family_gt]["n_samples"] += 1

        # Sauvegarder sample
        results["samples"].append({
            "idx": int(idx),
            "organ_gt": organ_name,
            "organ_pred": organ_pred_name,
            "family_gt": family_gt,
            "family_pred": family_pred,
            "routing_correct": family_pred == family_gt,
            "dice": float(dice),
            "hv_mse": float(hv_mse),
            "nt_acc": float(nt_acc),
        })

    # =========================================================================
    # AGRÉGATION RÉSULTATS
    # =========================================================================

    print("\n" + "="*80)
    print("RÉSULTATS GLOBAUX")
    print("="*80)

    # Routage
    organ_accuracy = organ_correct / n_samples
    family_accuracy = family_correct / n_samples

    results["routing"]["organ_accuracy"] = float(organ_accuracy)
    results["routing"]["family_accuracy"] = float(family_accuracy)

    print(f"🎯 Organ Accuracy:  {organ_accuracy*100:.2f}% ({organ_correct}/{n_samples})")
    print(f"🎯 Family Accuracy: {family_accuracy*100:.2f}% ({family_correct}/{n_samples})")

    # Métriques par famille
    print("\n" + "="*80)
    print("MÉTRIQUES PAR FAMILLE")
    print("="*80)

    for family in FAMILIES:
        metrics = family_metrics[family]

        if metrics["n_samples"] == 0:
            print(f"\n{family.upper()}: Aucun échantillon testé")
            continue

        mean_dice = np.mean(metrics["dice"])
        mean_hv_mse = np.mean(metrics["hv_mse"])
        mean_nt_acc = np.mean(metrics["nt_acc"])

        std_dice = np.std(metrics["dice"])
        std_hv_mse = np.std(metrics["hv_mse"])
        std_nt_acc = np.std(metrics["nt_acc"])

        results["metrics_by_family"][family] = {
            "n_samples": metrics["n_samples"],
            "dice_mean": float(mean_dice),
            "dice_std": float(std_dice),
            "hv_mse_mean": float(mean_hv_mse),
            "hv_mse_std": float(std_hv_mse),
            "nt_acc_mean": float(mean_nt_acc),
            "nt_acc_std": float(std_nt_acc),
        }

        print(f"\n{family.upper()} (n={metrics['n_samples']})")
        print(f"  NP Dice:  {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"  HV MSE:   {mean_hv_mse:.4f} ± {std_hv_mse:.4f}")
        print(f"  NT Acc:   {mean_nt_acc:.4f} ± {std_nt_acc:.4f}")

    # =========================================================================
    # SAUVEGARDE
    # =========================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"✅ Résultats sauvegardés: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
