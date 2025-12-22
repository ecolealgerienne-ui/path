#!/usr/bin/env python3
"""
Script de diagnostic pour le problème de prédiction d'organe.

Ce script trace étape par étape le pipeline pour comprendre pourquoi
Breast est prédit comme Prostate.

Usage:
    python scripts/validation/diagnose_organ_prediction.py --image path/to/breast_01.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
import cv2

# Ajouter le chemin du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports des modules centralisés (Phase 1 Refactoring)
from src.preprocessing import create_hoptimus_transform, HOPTIMUS_MEAN, HOPTIMUS_STD
from src.models.loader import ModelLoader

# Liste des organes PanNuke (ordre CRITIQUE pour les indices)
PANNUKE_ORGANS = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix",
    "Colon", "Esophagus", "HeadNeck", "Kidney", "Liver",
    "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin",
    "Stomach", "Testis", "Thyroid", "Uterus"
]


def diagnose_preprocessing(image_path: str, expected_organ: str = "Breast"):
    """
    Diagnostic complet du preprocessing.

    Affiche toutes les valeurs intermédiaires pour identifier le problème.
    """
    print("=" * 70)
    print("DIAGNOSTIC PRÉDICTION ORGANE")
    print("=" * 70)

    # 1. Charger l'image
    print("\n📥 ÉTAPE 1: Chargement de l'image")
    print("-" * 50)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ ERREUR: Impossible de charger {image_path}")
        return

    # OpenCV charge en BGR, convertir en RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"  Chemin: {image_path}")
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Range: [{image.min()}, {image.max()}]")
    print(f"  Mean RGB: {image.mean(axis=(0, 1))}")

    # Vérifier si l'image ressemble à du tissu H&E
    mean_r, mean_g, mean_b = image.mean(axis=(0, 1))
    if mean_r > 150 and mean_b > 150 and mean_g < mean_r:
        print(f"  ✓ Signature H&E probable (rose/violet)")
    else:
        print(f"  ⚠️ Signature H&E atypique - vérifier l'image")

    # 2. Conversion uint8
    print("\n🔄 ÉTAPE 2: Conversion uint8")
    print("-" * 50)

    if image.dtype != np.uint8:
        print(f"  ⚠️ Image n'est pas uint8, conversion...")
        if image.max() <= 1.0:
            image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
            print(f"  → float [0,1] → uint8 [0,255]")
        else:
            image_uint8 = image.clip(0, 255).astype(np.uint8)
            print(f"  → float [0,255] → uint8 [0,255]")
    else:
        image_uint8 = image
        print(f"  ✓ Déjà uint8")

    print(f"  Shape: {image_uint8.shape}")
    print(f"  Dtype: {image_uint8.dtype}")
    print(f"  Range: [{image_uint8.min()}, {image_uint8.max()}]")

    # 3. Application du transform
    print("\n🎨 ÉTAPE 3: Transform torchvision")
    print("-" * 50)

    transform = create_hoptimus_transform()

    # Étape par étape pour debug
    from PIL import Image as PILImage

    # ToPILImage
    pil_transform = transforms.ToPILImage()
    pil_img = pil_transform(image_uint8)
    print(f"  ToPILImage: mode={pil_img.mode}, size={pil_img.size}")

    # Resize
    resize_transform = transforms.Resize((224, 224))
    pil_resized = resize_transform(pil_img)
    print(f"  Resize: size={pil_resized.size}")

    # ToTensor
    tensor_transform = transforms.ToTensor()
    tensor = tensor_transform(pil_resized)
    print(f"  ToTensor: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"  ToTensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
    print(f"  ToTensor mean: {tensor.mean(dim=(1,2)).tolist()}")

    # Normalize
    norm_transform = transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD)
    tensor_norm = norm_transform(tensor)
    print(f"  Normalize range: [{tensor_norm.min():.4f}, {tensor_norm.max():.4f}]")
    print(f"  Normalize mean: {tensor_norm.mean(dim=(1,2)).tolist()}")

    # 4. Charger H-optimus-0 et extraire features
    print("\n🧠 ÉTAPE 4: Extraction features H-optimus-0")
    print("-" * 50)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        print("  Chargement H-optimus-0...")
        # Utiliser le chargeur centralisé (Phase 1 Refactoring)
        backbone = ModelLoader.load_hoptimus0(device=device)
        print("  ✓ H-optimus-0 chargé")

        # Forward via forward_features() qui inclut le LayerNorm final
        # Ceci est cohérent avec extract_features.py utilisé pour l'entraînement
        with torch.no_grad():
            input_tensor = tensor_norm.unsqueeze(0).to(device)
            print(f"  Input shape: {input_tensor.shape}")

            # forward_features() inclut le LayerNorm final
            features = backbone.forward_features(input_tensor).float()
            print(f"  Features shape: {features.shape}")

            cls_token = features[:, 0, :]
            print(f"  CLS token shape: {cls_token.shape}")
            print(f"  CLS token range: [{cls_token.min():.4f}, {cls_token.max():.4f}]")
            print(f"  CLS token mean: {cls_token.mean():.4f}")
            print(f"  CLS token std: {cls_token.std():.4f}")

    except Exception as e:
        print(f"  ❌ Erreur H-optimus-0: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Prédiction OrganHead
    print("\n🏥 ÉTAPE 5: Prédiction OrganHead")
    print("-" * 50)

    organ_head_path = PROJECT_ROOT / "models" / "checkpoints" / "organ_head_best.pth"

    if not organ_head_path.exists():
        print(f"  ❌ OrganHead non trouvé: {organ_head_path}")
        return

    try:
        from src.models.organ_head import OrganHead

        # Charger le modèle
        checkpoint = torch.load(organ_head_path, map_location=device)
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")

        organ_head = OrganHead(embed_dim=1536, n_organs=19)

        # Charger les poids
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Filtrer les buffers OOD
            filtered = {k: v for k, v in state_dict.items()
                       if k not in ['cls_mean', 'cls_cov_inv']}
            organ_head.load_state_dict(filtered, strict=False)
        else:
            organ_head.load_state_dict(checkpoint, strict=False)

        organ_head.eval()
        organ_head.to(device)
        print("  ✓ OrganHead chargé")

        # Forward
        with torch.no_grad():
            logits = organ_head(cls_token)
            probs = torch.softmax(logits, dim=-1)

            pred_idx = probs.argmax(dim=-1).item()
            pred_organ = PANNUKE_ORGANS[pred_idx]
            pred_conf = probs[0, pred_idx].item()

            print(f"\n  📊 RÉSULTAT:")
            print(f"  Prédiction: {pred_organ} (index {pred_idx})")
            print(f"  Confiance: {pred_conf:.2%}")

            # Top-5 prédictions
            print(f"\n  Top-5 prédictions:")
            top5_probs, top5_indices = probs.topk(5)
            for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
                organ = PANNUKE_ORGANS[idx.item()]
                marker = "👉" if organ == expected_organ else "  "
                print(f"  {marker} {i+1}. {organ}: {prob.item():.2%}")

            # Vérification
            expected_idx = PANNUKE_ORGANS.index(expected_organ)
            print(f"\n  🎯 COMPARAISON:")
            print(f"  Attendu: {expected_organ} (index {expected_idx})")
            print(f"  Prédit:  {pred_organ} (index {pred_idx})")

            if pred_organ == expected_organ:
                print(f"  ✅ CORRECT!")
            else:
                print(f"  ❌ ERREUR!")
                print(f"\n  🔍 ANALYSE DE L'ERREUR:")

                # Où se situe l'organe attendu dans le ranking?
                sorted_probs, sorted_indices = probs.sort(descending=True)
                for rank, idx in enumerate(sorted_indices[0]):
                    if idx.item() == expected_idx:
                        print(f"    → {expected_organ} est classé #{rank+1} avec {probs[0, expected_idx]:.2%}")
                        break

    except Exception as e:
        print(f"  ❌ Erreur OrganHead: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Comparer avec les features d'entraînement
    print("\n📈 ÉTAPE 6: Comparaison avec features d'entraînement")
    print("-" * 50)

    cache_dir = PROJECT_ROOT / "data" / "cache" / "pannuke_features"
    feature_files = list(cache_dir.glob("fold*_features.npz"))

    if not feature_files:
        print(f"  ⚠️ Pas de features cachées trouvées dans {cache_dir}")
        print(f"  Impossible de comparer avec les features d'entraînement")
    else:
        print(f"  Fichiers de features trouvés: {[f.name for f in feature_files]}")

        # Charger les labels
        pannuke_dir = Path("/home/amar/data/PanNuke")
        if pannuke_dir.exists():
            # Calculer la distance moyenne aux features de Breast et Prostate
            print("\n  Calcul des distances aux centroïdes par organe...")

            all_cls_tokens = []
            all_labels = []

            for fold in range(3):
                fold_dir = pannuke_dir / f"fold{fold}"
                types_path = fold_dir / "types.npy"
                features_path = cache_dir / f"fold{fold}_features.npz"

                if types_path.exists() and features_path.exists():
                    types = np.load(types_path)
                    features = np.load(features_path)

                    # CLS token = première position de layer_24
                    if 'layer_24' in features:
                        cls = features['layer_24'][:, 0, :]  # (N, 1536)
                        all_cls_tokens.append(cls)
                        all_labels.extend(types.tolist())

            if all_cls_tokens:
                all_cls = np.vstack(all_cls_tokens)
                all_labels = np.array(all_labels)

                # Calculer le centroïde par organe
                centroids = {}
                for organ in PANNUKE_ORGANS:
                    mask = all_labels == organ
                    if mask.sum() > 0:
                        centroids[organ] = all_cls[mask].mean(axis=0)
                        print(f"    {organ}: {mask.sum()} samples")

                # Distance de notre CLS token aux centroïdes
                query_cls = cls_token.cpu().numpy()[0]

                print(f"\n  🎯 Distances aux centroïdes:")
                distances = {}
                for organ, centroid in centroids.items():
                    dist = np.linalg.norm(query_cls - centroid)
                    distances[organ] = dist

                # Trier par distance
                sorted_dists = sorted(distances.items(), key=lambda x: x[1])
                for i, (organ, dist) in enumerate(sorted_dists[:10]):
                    marker = "👉" if organ == expected_organ else ("❌" if organ == pred_organ else "  ")
                    print(f"  {marker} {i+1}. {organ}: distance = {dist:.4f}")

    print("\n" + "=" * 70)
    print("FIN DU DIAGNOSTIC")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Diagnostic prédiction organe")
    parser.add_argument("--image", type=str, required=True,
                       help="Chemin vers l'image à diagnostiquer")
    parser.add_argument("--expected", type=str, default="Breast",
                       help="Organe attendu (default: Breast)")
    args = parser.parse_args()

    diagnose_preprocessing(args.image, args.expected)


if __name__ == "__main__":
    main()
