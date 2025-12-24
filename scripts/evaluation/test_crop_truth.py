#!/usr/bin/env python3
"""
TEST DE VÉRITÉ GÉOMÉTRIQUE (Expert 2025-12-23)

Question: Le modèle est-il corrompu ou juste mal utilisé?

Méthode: Inférence sur CROP CENTRAL 224x224 (sans resize) pour éliminer
         tout décalage géométrique.

Interprétation:
  - AJI > 0.45 → Modèle OK, problème = tuyauterie resize/crop
  - AJI < 0.10 → Modèle corrompu, re-training nécessaire
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
from tqdm import tqdm
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from src.models.loader import ModelLoader
from src.metrics.ground_truth_metrics import compute_aji, compute_panoptic_quality
from src.preprocessing import create_hoptimus_transform, validate_features


def center_crop(img, size=224):
    """Crop central size×size."""
    h, w = img.shape[:2]
    start_y = (h - size) // 2
    start_x = (w - size) // 2

    if img.ndim == 2:
        return img[start_y:start_y+size, start_x:start_x+size]
    else:
        return img[start_y:start_y+size, start_x:start_x+size, :]


def extract_instances_hv_magnitude(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    min_size: int = 20,
    dist_threshold: int = 5  # Plus conservateur pour test de vérité
) -> np.ndarray:
    """Extrait instances avec HV MAGNITUDE."""
    # 1. Binariser
    binary_mask = (np_pred > 0.5).astype(np.uint8)

    # 2. Remove small noise
    labeled, num = ndimage.label(binary_mask)
    if num == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    sizes = ndimage.sum(binary_mask, labeled, range(1, num + 1))
    mask_size = sizes >= min_size
    remove_small = mask_size[labeled - 1]
    remove_small[labeled == 0] = 0
    binary_mask = remove_small.astype(np.uint8)

    # 3. HV MAGNITUDE
    energy = np.sqrt(hv_pred[0]**2 + hv_pred[1]**2)

    # 4. Find peaks
    local_max = peak_local_max(
        energy,
        min_distance=dist_threshold,
        labels=binary_mask.astype(int),
        exclude_border=False,
    )

    # 5. Create markers
    markers = np.zeros_like(binary_mask, dtype=int)
    if len(local_max) > 0:
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

    # 6. Watershed
    if markers.max() > 0:
        inst_map = watershed(-energy, markers, mask=binary_mask)
    else:
        inst_map = ndimage.label(binary_mask)[0]

    return inst_map.astype(np.int32)


def extract_gt_instances_from_pannuke(mask: np.ndarray) -> np.ndarray:
    """Extrait instances GT depuis masque PanNuke (vraies instances séparées)."""
    inst_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-5: instances déjà annotées
    for c in range(mask.shape[2]):
        class_instances = mask[:, :, c]
        inst_ids = np.unique(class_instances)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = class_instances == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    return inst_map


def main():
    parser = argparse.ArgumentParser(description="Test de Vérité Géométrique (Crop 224)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("🔧 TEST DE VÉRITÉ GÉOMÉTRIQUE (Crop Central 224x224)")
    print("=" * 80)

    # 1. Load model
    print("\n📦 Chargement modèle...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    from src.models.hovernet_decoder import HoVerNetDecoder
    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=5).to(args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    # 2. Load data
    print("📦 Chargement données epidermal...")
    # ⚠️ FIX GHOST PATH BUG: Chercher UN SEUL endroit (source de vérité)
    # AVANT: Cherchait dans data/cache/family_data/ (ancien cache, peut être corrompu)
    # APRÈS: Cherche UNIQUEMENT dans data/family_FIXED/ (dernière version v4)
    data_file = Path("data/family_FIXED/epidermal_data_FIXED.npz")
    if not data_file.exists():
        raise FileNotFoundError(
            f"❌ Fichier non trouvé: {data_file}\n"
            f"   Régénérer avec: python scripts/preprocessing/prepare_family_data_FIXED_v4.py --family epidermal"
        )
    data = np.load(data_file)

    images = data['images']
    fold_ids = data['fold_ids']
    image_ids = data['image_ids']

    # Load GT masks
    pannuke_dir = Path("/home/amar/data/PanNuke")
    if not pannuke_dir.exists():
        pannuke_dir = Path("data/PanNuke")

    masks = np.load(pannuke_dir / "fold0" / "masks.npy", mmap_mode='r')

    # 3. Select test samples
    n_samples = min(args.n_samples, len(images))
    np.random.seed(42)
    test_indices = np.random.choice(len(images), n_samples, replace=False)

    print(f"  → {n_samples} échantillons sélectionnés\n")

    # 4. Transform (mais on va crop AVANT transform)
    transform = create_hoptimus_transform()

    # 5. Test loop
    print("🧪 Évaluation (CROP CENTRAL 224x224)...")

    all_dice = []
    all_aji = []
    all_pq = []

    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Testing Crop 224"):
            # Get image and GT
            image = images[idx]  # (256, 256, 3)
            fold_id = fold_ids[idx]
            img_id = image_ids[idx]

            # Load correct GT mask
            if fold_id == 0:
                gt_mask = masks[img_id]  # (256, 256, 6)
            else:
                fold_masks = np.load(pannuke_dir / f"fold{fold_id}" / "masks.npy", mmap_mode='r')
                gt_mask = fold_masks[img_id]

            # === CROP CENTRAL 224x224 (PAS de resize) ===
            img_224 = center_crop(image, 224)  # (224, 224, 3)
            gt_224 = center_crop(gt_mask, 224)  # (224, 224, 6)

            # Preprocess cropped image
            if img_224.dtype != np.uint8:
                img_224 = img_224.clip(0, 255).astype(np.uint8)

            tensor = transform(img_224).unsqueeze(0).to(args.device)

            # Extract features
            features = backbone.forward_features(tensor)

            # Validate features
            if idx == test_indices[0]:
                val_result = validate_features(features)
                print(f"\n🔍 VALIDATION FEATURES (first sample):")
                print(f"  CLS std: {val_result['cls_std']:.4f} (attendu: 0.70-0.90)")
                print(f"  Valid: {val_result['valid']}")

            patch_tokens = features[:, 1:257, :]  # (1, 256, 1536)

            # Predict
            np_out, hv_out, nt_out = hovernet(patch_tokens)

            # Convert to numpy
            np_pred_sigmoid = torch.sigmoid(np_out).cpu().numpy()[0]  # (2, 224, 224)
            hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)

            # DEBUG first sample
            if idx == test_indices[0]:
                print(f"  NP channel 0 max: {np_pred_sigmoid[0].max():.4f}")
                print(f"  NP channel 1 max: {np_pred_sigmoid[1].max():.4f}")
                print(f"  HV max: {hv_pred.max():.4f}")

            # Take channel 1 (nuclei)
            np_pred = np_pred_sigmoid[1]  # (224, 224)

            # === Extract instances sur 224x224 NATIF (pas de resize) ===
            pred_inst_224 = extract_instances_hv_magnitude(np_pred, hv_pred)

            # === Extract GT instances sur 224x224 CROP ===
            gt_inst_224 = extract_gt_instances_from_pannuke(gt_224)

            # DEBUG first sample
            if idx == test_indices[0]:
                n_pred = len(np.unique(pred_inst_224)) - 1
                n_gt = len(np.unique(gt_inst_224)) - 1
                print(f"  Instances Pred: {n_pred} | GT: {n_gt}")
                print()

            # === Compute metrics sur 224x224 ALIGNÉ ===
            # Dice
            pred_binary = (pred_inst_224 > 0).astype(np.float32)
            gt_binary = (gt_inst_224 > 0).astype(np.float32)

            intersection = (pred_binary * gt_binary).sum()
            union = pred_binary.sum() + gt_binary.sum()
            dice = 2 * intersection / union if union > 0 else 0.0

            # AJI
            aji = compute_aji(pred_inst_224, gt_inst_224)

            # PQ
            pq, dq, sq, _ = compute_panoptic_quality(pred_inst_224, gt_inst_224)

            all_dice.append(dice)
            all_aji.append(aji)
            all_pq.append(pq)

    # 6. Results
    print("\n" + "=" * 80)
    print("📊 RÉSULTATS TEST DE VÉRITÉ (Crop 224x224)")
    print("=" * 80)
    print()
    print(f"✅ Dice:  {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"✅ AJI:   {np.mean(all_aji):.4f} ± {np.std(all_aji):.4f}")
    print(f"✅ PQ:    {np.mean(all_pq):.4f} ± {np.std(all_pq):.4f}")
    print()
    print("🎯 INTERPRÉTATION:")

    mean_aji = np.mean(all_aji)
    if mean_aji > 0.45:
        print("  ✅ MODÈLE SAUVABLE! (AJI > 0.45)")
        print("  → Le problème était le resize/crop mismatch")
        print("  → Solution: Fixer le pipeline d'inférence (crop central au lieu de resize)")
    elif mean_aji > 0.20:
        print("  ⚠️  MODÈLE PARTIELLEMENT CORROMPU (AJI 0.20-0.45)")
        print("  → Amélioration possible avec fine-tuning")
    else:
        print("  ❌ MODÈLE CORROMPU (AJI < 0.20)")
        print("  → Re-training OBLIGATOIRE avec features FIXED")
    print()


if __name__ == "__main__":
    main()
