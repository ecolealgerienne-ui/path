#!/usr/bin/env python3
"""
Test AJI FINAL pour epidermal avec HV magnitude (pas Sobel).

Usage:
    python scripts/evaluation/test_epidermal_aji_FINAL.py \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
        --n_samples 50
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import (
    PANNUKE_IMAGE_SIZE,
    DEFAULT_PANNUKE_DIR,
    get_family_data_path,
    CURRENT_DATA_VERSION,
)
from src.metrics.ground_truth_metrics import compute_aji, compute_dice, compute_panoptic_quality
from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform


def extract_instances_hv_magnitude(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    min_size: int = 50,
    dist_threshold: int = 8
) -> np.ndarray:
    """
    Extrait instances avec HV MAGNITUDE (PAS Sobel).

    Méthode: HoVer-Net original (Graham et al. 2019)

    ===================================================================
    FIX ANTI-CONFETTIS (Expert 2025-12-24):
    ===================================================================
    PROBLÈME IDENTIFIÉ (IoU 0.71 mais AJI 0.15):
    - Sur-segmentation: 28 instances prédites vs 13 GT
    - Le modèle voit les bonnes formes mais les découpe en morceaux
    - Cause: Gradients HV bruités → trop de pics → watershed fracasse

    SOLUTION APPLIQUÉE:
    1. Lissage gaussien (sigma=1.0) sur HV gradients
       → Élimine micro-oscillations qui créent faux pics
    2. dist_threshold: 4 → 8 pixels
       → Force distance minimale entre centres (évite découpe)
    3. min_size: 20 → 50 pixels
       → Noyaux PanNuke font 50-200 pixels, élimine débris

    ATTENDU: Instances 28 → ~13, AJI 0.15 → >0.60 (+300%)
    ===================================================================
    """
    # 1. Binariser NP avec seuil strict
    binary_mask = (np_pred > 0.5).astype(np.uint8)

    # 2. HV MAGNITUDE
    energy = np.sqrt(hv_pred[0]**2 + hv_pred[1]**2)

    # 3. ⭐ LISSAGE CRITIQUE: Supprimer bruit dans gradients
    #    Sans ceci, chaque micro-oscillation = nouveau pic = sur-segmentation
    energy = ndimage.gaussian_filter(energy, sigma=1.0)

    # 4. Find peaks avec dist_threshold ÉLARGI
    #    Plus il est haut, moins il y a de sur-segmentation
    local_max = peak_local_max(
        energy,
        min_distance=dist_threshold,  # 8 pixels au lieu de 4
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

    # 7. ⭐ NETTOYAGE FINAL: Supprimer petits objets
    #    Noyaux PanNuke font 50-200 pixels
    #    À min_size=50, on garde seulement vrais noyaux
    inst_map_labeled = inst_map.astype(np.int32)
    for region_id in np.unique(inst_map_labeled):
        if region_id == 0:  # Skip background
            continue
        region_size = (inst_map_labeled == region_id).sum()
        if region_size < min_size:
            inst_map_labeled[inst_map_labeled == region_id] = 0

    # Re-label pour combler les trous d'ID
    inst_map_clean = np.zeros_like(inst_map_labeled)
    unique_ids = np.unique(inst_map_labeled)
    unique_ids = unique_ids[unique_ids > 0]
    for new_id, old_id in enumerate(unique_ids, start=1):
        inst_map_clean[inst_map_labeled == old_id] = new_id

    return inst_map_clean.astype(np.int32)


def get_correct_gt_instances(gt_mask: np.ndarray) -> np.ndarray:
    """
    FIX DÉFINITIF (Expert 2025-12-24): Utilise CANAL 0 de PanNuke.

    ===================================================================
    PROBLÈME IDENTIFIÉ (Diagnostic inspect_gt_instances.py):
    ===================================================================
    - Canal 0: 15 instances avec IDs [3, 4, 12, 16...68] ✅ VRAIES INSTANCES
    - Canaux 1-4: VIDES (pour epidermal) ❌
    - Canal 5: Masque binaire géant (56k pixels) ❌
    - compute_gt_instances() ne trouvait que 3 instances au lieu de 15+

    CAUSE RACINE:
    - Canal 0 contient les instances multi-types PanNuke (GOLD STANDARD)
    - compute_gt_instances() ignorait canal 0 et ne traitait que 1-5
    - Canal 5 binaire → connectedComponents → 2-3 grosses taches
    - Résultat: 28 instances prédites vs 3 GT → AJI 0.08 ❌

    SOLUTION:
    - Utiliser CANAL 0 directement (contient vraies instances)
    - Ajouter instances canaux 1-4 si non vides (rare pour epidermal)
    - Canal 5: Seulement si canal 0 quasi vide (fallback)

    ATTENDU: GT 3 → 15+, AJI 0.08 → >0.60 (+650%)
    ===================================================================
    """
    # 1. ⭐ CANAL 0: Les vraies instances PanNuke (IDs multi-types)
    inst_map = gt_mask[:, :, 0].astype(np.int32)

    # 2. Ajouter instances des canaux 1-4 (si non vides)
    #    En s'assurant que les IDs ne se chevauchent pas
    current_max_id = inst_map.max()
    for c in range(1, 5):
        channel_map = gt_mask[:, :, c].astype(np.int32)
        if channel_map.max() > 0:
            # Prendre seulement pixels non encore assignés
            mask = (inst_map == 0) & (channel_map > 0)
            # Renommer IDs pour éviter conflits
            unique_ids = np.unique(channel_map[mask])
            unique_ids = unique_ids[unique_ids > 0]
            for new_id, old_id in enumerate(unique_ids, start=current_max_id + 1):
                inst_map[(inst_map == 0) & (channel_map == old_id)] = new_id
            current_max_id = inst_map.max()

    # 3. Fallback: Canal 5 (Epithelial) SEULEMENT si quasi rien dans 0-4
    if inst_map.max() < 5:  # Si on a presque rien
        epi_mask = (gt_mask[:, :, 5] > 0).astype(np.uint8)
        epi_inst, n = ndimage.label(epi_mask)
        # Fusionner avec précaution (seulement pixels vides)
        mask = (inst_map == 0) & (epi_inst > 0)
        unique_epi_ids = np.unique(epi_inst[mask])
        unique_epi_ids = unique_epi_ids[unique_epi_ids > 0]
        for new_id, old_id in enumerate(unique_epi_ids, start=current_max_id + 1):
            inst_map[(inst_map == 0) & (epi_inst == old_id)] = new_id

    return inst_map


def main():
    parser = argparse.ArgumentParser(description="Test AJI FINAL epidermal")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint HoVer-Net")
    parser.add_argument("--n_samples", type=int, default=50, help="Nombre échantillons test")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Fichier de données (défaut: cherche v12 puis v11 puis FIXED.npz)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("🎯 TEST AJI FINAL - EPIDERMAL")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Samples: {args.n_samples}")
    print(f"Device: {args.device}")

    # Load model
    print("\n🔧 Chargement modèle...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    backbone.eval()

    # n_classes=2 pour matcher le checkpoint entraîné sur données v12 (binaire: 0=bg, 1=nucleus)
    hovernet = HoVerNetDecoder(embed_dim=1536, n_classes=2).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hovernet.load_state_dict(checkpoint['model_state_dict'])
    hovernet.eval()

    transform = create_hoptimus_transform()

    # Load epidermal test data
    print("\n📦 Chargement données epidermal...")
    print(f"   Version courante: {CURRENT_DATA_VERSION}")

    # ⚠️ UTILISE LA VERSION COURANTE (centralisée dans constants.py)
    if args.data_file:
        data_file = Path(args.data_file)
        print(f"   → Fichier spécifié: {data_file}")
    else:
        data_file = Path(get_family_data_path("epidermal"))
        print(f"   → Fichier auto: {data_file}")

    if not data_file.exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        print(f"   Lancez d'abord: python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py --family epidermal")
        return

    data = np.load(data_file)
    images = data['images']

    # =========================================================================
    # FIX EXPERT 2025-12-25: Utiliser np_targets du NPZ (alignement garanti)
    # =========================================================================
    # PROBLÈME: L'indexation fold_id/img_id pour charger les masques PanNuke
    #           était décalée → 40% de GT vides → Dice 0.22
    # SOLUTION: Utiliser np_targets du même fichier (aligné par définition)
    #           et créer instances via connected components
    # =========================================================================
    np_targets = data['np_targets']  # Binary masks (N, 256, 256) float32 [0, 1]
    print(f"  → np_targets shape: {np_targets.shape}")

    n_test = min(args.n_samples, len(images))
    test_indices = np.random.choice(len(images), n_test, replace=False)

    print(f"  → {n_test} échantillons sélectionnés")

    # Evaluate
    print("\n🧪 Évaluation...")
    all_aji = []
    all_dice = []
    all_pq = []
    n_skipped = 0  # Track empty GT samples

    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Testing"):
            # Get image and GT from NPZ (ALIGNÉ par définition)
            image = images[idx]
            np_target = np_targets[idx]  # Binary mask (256, 256) float32

            # Créer GT instances via connected components (depuis np_target aligné)
            from scipy import ndimage
            gt_binary = (np_target > 0.5).astype(np.uint8)
            gt_inst_256, n_gt_instances = ndimage.label(gt_binary)
            gt_inst_256 = gt_inst_256.astype(np.int32)

            # Preprocess
            if image.dtype != np.uint8:
                image = image.clip(0, 255).astype(np.uint8)

            tensor = transform(image).unsqueeze(0).to(args.device)

            # Extract features
            features = backbone.forward_features(tensor)
            patch_tokens = features[:, 1:257, :]  # (1, 256, 1536)

            # Predict
            np_out, hv_out, nt_out = hovernet(patch_tokens)

            # ===================================================================
            # STRATÉGIE EXPERT 2025-12-25: Évaluation en 224×224 (résolution native)
            # ===================================================================
            # Le modèle a été entraîné en 224×224. Pour une évaluation correcte:
            # - Prédictions: rester en 224×224 (natif, pas d'interpolation)
            # - GT: resize 256→224 avec INTER_NEAREST (préserve IDs instances)
            # ===================================================================

            # 1. Prédictions NATIVES (224×224) - PAS de resize
            prob_map = torch.softmax(np_out, dim=1)[0, 1].cpu().numpy()  # (224, 224)
            hv_map = hv_out[0].cpu().numpy()  # (2, 224, 224)

            # DEBUG
            if idx == test_indices[0]:
                print(f"\n🔍 DEBUG (first sample):")
                print(f"  prob_map shape: {prob_map.shape}, max: {prob_map.max():.4f}")
                print(f"  hv_map shape: {hv_map.shape}, max: {hv_map.max():.4f}")

            # 2. GT adapté: 256→224 avec INTER_NEAREST (préserve les IDs)
            # gt_inst_256 déjà créé via connected components ci-dessus
            gt_inst = cv2.resize(gt_inst_256, (224, 224), interpolation=cv2.INTER_NEAREST)

            # 3. Extraction instances (tout en 224×224)
            pred_inst = extract_instances_hv_magnitude(prob_map, hv_map)

            # gt_inst déjà calculé ci-dessus (256→224 avec INTER_NEAREST)

            # Count instances (needed for skip logic)
            n_pred = len(np.unique(pred_inst)) - 1  # -1 for background
            n_gt = len(np.unique(gt_inst)) - 1

            # DEBUG VISUEL (Expert demande)
            if idx == test_indices[0]:  # Print once
                print(f"  Instances Pred: {n_pred} | GT: {n_gt}")

                # Sauvegarde image debug
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 5))

                # 1. Image originale
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title("Image Originale")

                # 2. Prédiction 224×224 (Prob map > 0.5)
                plt.subplot(1, 3, 2)
                plt.imshow(prob_map > 0.5, cmap='gray')
                plt.title(f"Pred 224×224 (n={n_pred})")

                # 3. GT Canal 0 (resized to 224×224)
                plt.subplot(1, 3, 3)
                plt.imshow(gt_inst > 0, cmap='gray')
                plt.title(f"GT 224×224 (n={n_gt})")

                plt.tight_layout()
                import os
                os.makedirs("results", exist_ok=True)
                plt.savefig("results/DEBUG_CRASH_TEST.png", dpi=150)
                plt.close()
                print(f"\n📸 DEBUG: Image sauvée → results/DEBUG_CRASH_TEST.png")
                print("   REGARDEZ cette image pour diagnostiquer le problème!")

                # DEBUG GT (from np_targets NPZ - aligné par définition)
                print(f"\n🔍 DEBUG GT (NPZ np_targets):")
                print(f"  np_target shape: {np_target.shape}")
                print(f"  np_target dtype: {np_target.dtype}")
                print(f"  np_target range: [{np_target.min():.3f}, {np_target.max():.3f}]")
                print(f"  gt_binary nonzero: {gt_binary.sum()} pixels")
                print(f"  gt_inst_256 instances: {n_gt_instances}")
                print(f"  gt_inst (224×224) unique IDs: {np.unique(gt_inst)}")

            # ⚠️ CRITICAL: Skip empty GT (évite division par zéro dans AJI)
            if n_gt == 0:
                n_skipped += 1
                if idx == test_indices[0]:
                    print(f"  ⚠️  SKIPPING: GT vide (pas de cellules)")
                continue

            # Compute metrics
            aji = compute_aji(pred_inst, gt_inst)

            # Dice calculation (tout en 224×224 - résolution native du modèle)
            dice = compute_dice((prob_map > 0.5).astype(np.uint8), (gt_inst > 0).astype(np.uint8))

            pq, dq, sq, _ = compute_panoptic_quality(pred_inst, gt_inst)

            all_aji.append(aji)
            all_dice.append(dice)
            all_pq.append(pq)

    # Results
    print("\n" + "=" * 80)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 80)
    print(f"\n📈 Échantillons:")
    print(f"  Total testé: {len(test_indices)}")
    print(f"  Valides (GT non-vide): {len(all_aji)}")
    print(f"  Skippés (GT vide): {n_skipped}")
    print(f"\n✅ Dice:  {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"✅ AJI:   {np.mean(all_aji):.4f} ± {np.std(all_aji):.4f}")
    print(f"✅ PQ:    {np.mean(all_pq):.4f} ± {np.std(all_pq):.4f}")

    print("\n🎯 Objectifs:")
    print(f"  Dice: >0.90  → {'✅ ATTEINT' if np.mean(all_dice) > 0.90 else '❌ PAS ATTEINT'}")
    print(f"  AJI:  >0.60  → {'✅ ATTEINT' if np.mean(all_aji) > 0.60 else '❌ PAS ATTEINT'}")
    print(f"  PQ:   >0.65  → {'✅ ATTEINT' if np.mean(all_pq) > 0.65 else '❌ PAS ATTEINT'}")


if __name__ == "__main__":
    main()
