#!/usr/bin/env python3
"""
Test du modèle sur les VRAIES données d'entraînement
pour comparer avec les résultats d'entraînement documentés.

Cela permet d'isoler si le problème vient de :
1. La préparation des données (train vs eval)
2. Les prédictions du modèle lui-même
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader
from src.constants import HOPTIMUS_INPUT_SIZE

def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Calcule Dice score."""
    pred_binary = pred > 0.5
    target_binary = target > 0

    intersection = (pred_binary & target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()

    return 2 * intersection / (union + 1e-8)

def compute_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Calcule MSE sur les pixels masqués."""
    if mask.sum() == 0:
        return 0.0
    return ((pred[:, mask] - target[:, mask]) ** 2).mean()

def compute_accuracy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Calcule accuracy sur les pixels masqués."""
    if mask.sum() == 0:
        return 0.0
    pred_class = pred.argmax(axis=0)
    return (pred_class[mask] == target[mask]).mean()

def test_on_training_data(
    family: str,
    checkpoint_path: str,
    data_dir: str = DEFAULT_FAMILY_FIXED_DIR,  # ⚠️ FIX: Source de vérité unique
    n_samples: int = 100,
    device: str = "cuda"
):
    """
    Teste le modèle sur les VRAIES données d'entraînement.

    Cette fonction reproduit EXACTEMENT le pipeline d'entraînement :
    1. Charge features pré-extraites (mêmes que train)
    2. Charge targets pré-calculés (mêmes que train)
    3. Applique resize 256→224 (même que train)
    4. Inférence modèle
    5. Compare avec targets (PAS de resize inverse)
    """

    print("=" * 80)
    print(f"TEST SUR DONNÉES D'ENTRAÎNEMENT: {family}")
    print("=" * 80)
    print("")

    data_dir = Path(data_dir)

    # Support pour les deux formats:
    # Format PRÉFÉRÉ: {family}_features.npz + {family}_targets.npz (séparés)
    # Format LEGACY: {family}_data_FIXED.npz (tout en un SAUF features)

    features_path = data_dir / f"{family}_features.npz"
    targets_path = data_dir / f"{family}_targets.npz"
    data_fixed_path = data_dir / f"{family}_data_FIXED.npz"

    # Essayer format FEATURES + TARGETS d'abord (format actuel)
    if features_path.exists() and targets_path.exists():
        print(f"✅ Format features+targets détecté")
        print(f"   Features: {features_path}")
        print(f"   Targets:  {targets_path}")
        print("")

    # Sinon, vérifier si format FIXED existe (legacy - features non extraites)
    elif data_fixed_path.exists():
        print(f"⚠️  Format FIXED détecté: {data_fixed_path}")
        print("   Ce fichier contient images + targets, mais PAS les features.")
        print("")
        print("   Pour tester le modèle, vous devez d'abord extraire les features:")
        print(f"   python scripts/preprocessing/extract_all_family_features.py")
        print("")
        return

    # Aucun format trouvé
    else:
        print(f"❌ ERREUR: Données introuvables dans {data_dir}")
        print(f"   Attendu: {features_path} + {targets_path}")
        print(f"   Ou: {data_fixed_path}")
        return

    print(f"Chargement features: {features_path}")
    features_data = np.load(features_path)

    # Support des deux formats de clés (features ou layer_24)
    if 'features' in features_data:
        features = features_data['features']  # (N, 261, 1536)
    elif 'layer_24' in features_data:
        features = features_data['layer_24']  # Ancien format
    else:
        print(f"❌ ERREUR: Clés disponibles: {list(features_data.keys())}")
        print("   Attendu: 'features' ou 'layer_24'")
        return

    print(f"Chargement targets: {targets_path}")
    targets_data = np.load(targets_path)
    np_targets = targets_data['np_targets']  # (N, 256, 256)
    hv_targets = targets_data['hv_targets']  # (N, 2, 256, 256)
    nt_targets = targets_data['nt_targets']  # (N, 256, 256)

    n_total = len(features)
    n_samples = min(n_samples, n_total)

    print(f"Dataset: {n_total} samples (test sur {n_samples})")
    print("")

    # Charger modèle
    print(f"Chargement modèle: {checkpoint_path}")
    hovernet = ModelLoader.load_hovernet(checkpoint_path, device=device)
    hovernet.eval()
    print("")

    # Test sur n_samples
    print(f"Inférence sur {n_samples} échantillons...")
    print("")

    all_dice = []
    all_hv_mse = []
    all_nt_acc = []

    for i in tqdm(range(n_samples)):
        # Extraire features (déjà pré-calculées)
        feat = features[i]  # (261, 1536)

        # Extraire targets à 256×256
        np_target_256 = np_targets[i]  # (256, 256)
        hv_target_256 = hv_targets[i]  # (2, 256, 256)
        nt_target_256 = nt_targets[i]  # (256, 256)

        # RESIZE TARGETS 256→224 (EXACTEMENT comme le DataLoader)
        import torch.nn.functional as F

        # Convertir en tenseurs avec le bon type (float pour interpolation bilinear)
        np_target_t = torch.from_numpy(np_target_256).float().unsqueeze(0).unsqueeze(0)
        hv_target_t = torch.from_numpy(hv_target_256).float().unsqueeze(0)
        nt_target_t = torch.from_numpy(nt_target_256).float().unsqueeze(0).unsqueeze(0)

        np_target_t = F.interpolate(np_target_t, size=(224, 224), mode='nearest').squeeze()
        hv_target_t = F.interpolate(hv_target_t, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        nt_target_t = F.interpolate(nt_target_t, size=(224, 224), mode='nearest').squeeze().long()

        np_target_224 = np_target_t.numpy()
        hv_target_224 = hv_target_t.numpy()
        nt_target_224 = nt_target_t.numpy()

        # Inférence (features → prédictions à 224×224)
        feat_tensor = torch.from_numpy(feat).unsqueeze(0).to(device)  # (1, 261, 1536)
        patch_tokens = feat_tensor[:, 1:257, :]  # (1, 256, 1536)

        with torch.no_grad():
            np_out, hv_out, nt_out = hovernet(patch_tokens)

        # Convertir en numpy (sorties à 224×224)
        # NP: Si 2 canaux (softmax), prendre canal 1 (nuclei), sinon sigmoid
        if np_out.shape[1] == 2:
            # Softmax 2-classes: [background, nuclei]
            np_pred = torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]  # (224, 224) - canal nuclei
        else:
            # Sigmoid 1-classe
            np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]  # (224, 224)

        hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)
        nt_pred = torch.softmax(nt_out, dim=1).cpu().numpy()[0]  # (n_classes, 224, 224)

        # Calculer métriques (TOUT à 224×224 - comme pendant l'entraînement)
        dice = compute_dice(np_pred, np_target_224)
        all_dice.append(dice)

        # HV MSE sur pixels de noyaux
        mask = np_target_224 > 0
        if mask.sum() > 0:
            hv_mse = compute_mse(hv_pred, hv_target_224, mask)
            all_hv_mse.append(hv_mse)

            # NT Accuracy sur pixels de noyaux
            nt_acc = compute_accuracy(nt_pred, nt_target_224, mask)
            all_nt_acc.append(nt_acc)

    # Résultats
    print("")
    print("=" * 80)
    print("RÉSULTATS SUR DONNÉES D'ENTRAÎNEMENT")
    print("=" * 80)
    print("")

    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    mean_hv = np.mean(all_hv_mse)
    std_hv = np.std(all_hv_mse)
    mean_nt = np.mean(all_nt_acc)
    std_nt = np.std(all_nt_acc)

    print(f"NP Dice:  {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"HV MSE:   {mean_hv:.4f} ± {std_hv:.4f}")
    print(f"NT Acc:   {mean_nt:.4f} ± {std_nt:.4f}")
    print("")

    # Comparaison avec résultats d'entraînement documentés
    expected_results = {
        "glandular": {"dice": 0.9648, "hv_mse": 0.0106, "nt_acc": 0.9111},
        "digestive": {"dice": 0.9634, "hv_mse": 0.0163, "nt_acc": 0.8824},
        "urologic": {"dice": 0.9318, "hv_mse": 0.2812, "nt_acc": 0.9139},
        "respiratory": {"dice": 0.9409, "hv_mse": 0.0500, "nt_acc": 0.9183},
        "epidermal": {"dice": 0.9542, "hv_mse": 0.2653, "nt_acc": 0.8857},
    }

    if family in expected_results:
        expected = expected_results[family]

        print("=" * 80)
        print("COMPARAISON AVEC RÉSULTATS D'ENTRAÎNEMENT DOCUMENTÉS")
        print("=" * 80)
        print("")

        dice_diff = (mean_dice - expected["dice"]) / expected["dice"] * 100
        hv_diff = (mean_hv - expected["hv_mse"]) / expected["hv_mse"] * 100
        nt_diff = (mean_nt - expected["nt_acc"]) / expected["nt_acc"] * 100

        print(f"NP Dice:  {mean_dice:.4f} vs {expected['dice']:.4f} (Δ {dice_diff:+.1f}%)")
        print(f"HV MSE:   {mean_hv:.4f} vs {expected['hv_mse']:.4f} (Δ {hv_diff:+.1f}%)")
        print(f"NT Acc:   {mean_nt:.4f} vs {expected['nt_acc']:.4f} (Δ {nt_diff:+.1f}%)")
        print("")

        # Verdict
        print("=" * 80)
        print("VERDICT")
        print("=" * 80)
        print("")

        if abs(dice_diff) < 5:
            print("✅ RÉSULTATS COHÉRENTS avec l'entraînement (Δ < 5%)")
            print("")
            print("   → Le modèle fonctionne correctement sur les données d'entraînement")
            print("   → Le problème vient de la PRÉPARATION DES DONNÉES d'évaluation")
            print("")
            print("   CAUSE PROBABLE:")
            print("   • TRAIN: Targets 256→224 (resize GT vers taille modèle)")
            print("   • EVAL:  Prédictions 224→256 (resize prédictions vers taille GT)")
            print("")
            print("   → Le sens du resize est INVERSÉ !")
            print("   → Cela introduit des artifacts différents")
            print("")
            print("   SOLUTION:")
            print("   Modifier test_family_models_isolated.py pour reproduire")
            print("   EXACTEMENT le pipeline d'entraînement:")
            print("   1. Resize GT 256→224 (au lieu de resize prédictions 224→256)")
            print("   2. Utiliser interpolation identique (nearest pour NP/NT, bilinear pour HV)")
            print("")
        else:
            print("⚠️  DIVERGENCE SIGNIFICATIVE avec l'entraînement (Δ > 5%)")
            print("")
            print("   → Le modèle ne reproduit PAS les performances d'entraînement")
            print("")
            print("   CAUSES POSSIBLES:")
            print("   1. Features d'entraînement différentes de celles utilisées ici")
            print("   2. Bug dans le chargement du checkpoint")
            print("   3. Targets pré-calculés corrompus")
            print("")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test modèle sur données d'entraînement")
    parser.add_argument("--family", type=str, required=True, choices=["glandular", "digestive", "urologic", "respiratory", "epidermal"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Chemin vers hovernet_*_best.pth")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_FAMILY_FIXED_DIR,
                        help="Répertoire des données (source de vérité unique)")
    parser.add_argument("--n_samples", type=int, default=100, help="Nombre d'échantillons à tester")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    test_on_training_data(
        family=args.family,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        device=args.device
    )
