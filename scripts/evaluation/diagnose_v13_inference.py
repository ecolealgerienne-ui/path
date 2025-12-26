#!/usr/bin/env python3
"""
DIAGNOSTIC FORENSIQUE V13 - Crash Test sur 1 échantillon.

Analyse en profondeur:
1. INPUT: Image range, tensor normalisé
2. OUTPUT: Prob map range, pixels > 0.5
3. GT: Instances extraites, surface
4. ALIGNMENT: Centre crop 10×10 (pred vs GT)

Usage:
    python scripts/evaluation/diagnose_v13_inference.py \
        --checkpoint models/checkpoints_v13/hovernet_epidermal_v13_best.pth \
        --family epidermal
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE
from src.models.organ_families import ORGAN_TO_FAMILY
from src.models.hovernet_decoder import HoVerNetDecoder
from src.models.loader import ModelLoader
from src.preprocessing import create_hoptimus_transform


def extract_ground_truth_instances_fixed(mask: np.ndarray) -> np.ndarray:
    """Extraction GT avec TOUS les canaux 0-4."""
    inst_map = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int32)
    max_id = 0

    for c in range(5):
        channel = mask[:, :, c]
        if channel.max() > 0:
            inst_ids = np.unique(channel)
            inst_ids = inst_ids[inst_ids > 0]

            for inst_id in inst_ids:
                inst_mask = channel == inst_id
                max_id += 1
                inst_map[inst_mask] = max_id

    return inst_map


def main():
    parser = argparse.ArgumentParser(description="Diagnostic V13 Forensique")
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--pannuke_dir', type=Path, default=Path('/home/amar/data/PanNuke'))
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("="*70)
    print("🚨 DIAGNOSTIC FORENSIQUE V13 - 1 ÉCHANTILLON")
    print("="*70)

    # 1. Charger PanNuke fold 2
    fold_dir = args.pannuke_dir / "fold2"
    images = np.load(fold_dir / "images.npy", mmap_mode='r')
    masks = np.load(fold_dir / "masks.npy", mmap_mode='r')
    types = np.load(fold_dir / "types.npy")

    # Filtrer par famille
    organs = [org for org, fam in ORGAN_TO_FAMILY.items() if fam == args.family]
    family_indices = [i for i, org in enumerate(types) if org in organs]

    print(f"\n📂 Dataset:")
    print(f"   Famille: {args.family}")
    print(f"   Organes: {', '.join(organs)}")
    print(f"   Échantillons disponibles: {len(family_indices)}")

    # Prendre le PREMIER échantillon
    idx = family_indices[0]
    image = np.array(images[idx], dtype=np.uint8)
    mask = np.array(masks[idx])

    print(f"   Index sélectionné: {idx}")
    print(f"   Organe: {types[idx]}")

    # 2. Charger modèles
    print(f"\n🔧 Chargement modèles...")
    backbone = ModelLoader.load_hoptimus0(device=args.device)
    model = HoVerNetDecoder(embed_dim=1536, n_classes=2, dropout=0.0)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"   ✅ Checkpoint: {args.checkpoint.name}")
    print(f"   ✅ Epoch: {checkpoint.get('epoch', 'N/A')}")

    # ==============================================================================
    # 🕵️ ANALYSE DÉTAILLÉE
    # ==============================================================================

    print("\n" + "🚨" * 35)
    print("RAPPORT FORENSIQUE V13")
    print("🚨" * 35)

    # 1. ANALYSE INPUT (Image)
    print(f"\n{'='*70}")
    print("1. INPUT IMAGE STATS")
    print(f"{'='*70}")
    print(f"   Shape:       {image.shape}")
    print(f"   Dtype:       {image.dtype}")
    print(f"   Range:       [{image.min()}, {image.max()}]")

    # Crop central (16:240)
    y1, y2 = 16, 240
    x1, x2 = 16, 240
    crop = image[y1:y2, x1:x2]

    print(f"\n   Crop central (16:240):")
    print(f"   Shape:       {crop.shape}")
    print(f"   Range:       [{crop.min()}, {crop.max()}]")

    # Transform
    transform = create_hoptimus_transform()
    if crop.dtype != np.uint8:
        crop = crop.clip(0, 255).astype(np.uint8)

    tensor = transform(crop).unsqueeze(0).to(args.device)
    input_tensor = tensor.cpu().numpy()

    print(f"\n   Après transform (tensor pour modèle):")
    print(f"   Shape:       {tensor.shape}")
    print(f"   Range:       [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    print(f"   Mean:        {input_tensor.mean():.3f}")
    print(f"   Std:         {input_tensor.std():.3f}")
    print(f"   ⚠️  Attendu:  Range approx [-2, 2], mean ~0, std ~1")

    # 2. ANALYSE OUTPUT (Prob Map)
    print(f"\n{'='*70}")
    print("2. MODEL OUTPUT (Prob Map)")
    print(f"{'='*70}")

    with torch.no_grad():
        features = backbone.forward_features(tensor)
        np_out, hv_out, nt_out = model(features.float())

    # Activations
    prob_map = torch.sigmoid(np_out).cpu().numpy()[0, 0]

    print(f"   Shape:       {prob_map.shape}")
    print(f"   Range:       [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    print(f"   Mean:        {prob_map.mean():.4f}")
    print(f"   Std:         {prob_map.std():.4f}")

    pixels_over_threshold = (prob_map > 0.5).sum()
    print(f"\n   Pixels > 0.5: {pixels_over_threshold:,} / {prob_map.size:,} ({pixels_over_threshold/prob_map.size*100:.2f}%)")

    if pixels_over_threshold == 0:
        print(f"   🚨 ALERTE: Modèle AVEUGLE - aucun pixel > 0.5!")
    elif prob_map.max() < 0.5:
        print(f"   🚨 ALERTE: Modèle TIMIDE - max prob {prob_map.max():.4f} < 0.5!")
    else:
        print(f"   ✅ Modèle prédit des noyaux")

    # 3. ANALYSE GT
    print(f"\n{'='*70}")
    print("3. GROUND TRUTH (Target)")
    print(f"{'='*70}")

    gt_inst_full = extract_ground_truth_instances_fixed(mask)
    gt_inst_crop = gt_inst_full[y1:y2, x1:x2]

    print(f"   GT Full (256×256):")
    print(f"   Instances uniques: {np.unique(gt_inst_full)[:10]}... ({len(np.unique(gt_inst_full))} total)")
    print(f"   Surface noyaux:    {(gt_inst_full > 0).sum():,} pixels")

    print(f"\n   GT Crop (224×224, central):")
    print(f"   Instances uniques: {np.unique(gt_inst_crop)[:10]}... ({len(np.unique(gt_inst_crop))} total)")
    print(f"   Surface noyaux:    {(gt_inst_crop > 0).sum():,} pixels")

    if (gt_inst_crop > 0).sum() == 0:
        print(f"   🚨 ALERTE: GT VIDE - aucun noyau dans le crop central!")
    else:
        print(f"   ✅ GT contient des noyaux")

    # 4. ALIGNMENT CHECK (Centre 10×10)
    print(f"\n{'='*70}")
    print("4. CENTER CROP CHECK (10×10 pixels)")
    print(f"{'='*70}")

    cy, cx = 112, 112  # Centre du crop 224×224
    slice_pred = (prob_map > 0.5).astype(int)[cy:cy+10, cx:cx+10]
    slice_gt = (gt_inst_crop > 0).astype(int)[cy:cy+10, cx:cx+10]

    print(f"\n   PRED (centre 10×10):")
    for row in slice_pred:
        print("   " + "".join("█" if x else "·" for x in row))

    print(f"\n   GT (centre 10×10):")
    for row in slice_gt:
        print("   " + "".join("█" if x else "·" for x in row))

    # Overlap
    overlap = (slice_pred * slice_gt).sum()
    pred_area = slice_pred.sum()
    gt_area = slice_gt.sum()

    print(f"\n   Overlap:  {overlap} pixels")
    print(f"   Pred:     {pred_area} pixels")
    print(f"   GT:       {gt_area} pixels")

    if overlap > 0:
        print(f"   ✅ Alignement OK (overlap {overlap}/{min(pred_area, gt_area)} pixels)")
    elif pred_area > 0 and gt_area > 0:
        print(f"   ⚠️  ALERTE: Pred et GT non-nuls MAIS pas d'overlap → DÉCALAGE SPATIAL!")
    elif pred_area == 0:
        print(f"   🚨 ALERTE: Pred vide (modèle aveugle)")
    elif gt_area == 0:
        print(f"   🚨 ALERTE: GT vide (extraction incorrecte)")

    # 5. DIAGNOSTIC FINAL
    print(f"\n{'='*70}")
    print("5. DIAGNOSTIC FINAL")
    print(f"{'='*70}")

    # Scénario A: Double Scaling
    if input_tensor.max() < 0.1 or input_tensor.min() > -0.1:
        print(f"\n   🚨 SCÉNARIO A: DOUBLE SCALING DÉTECTÉ")
        print(f"      Tensor range [{input_tensor.min():.3f}, {input_tensor.max():.3f}] est anormal")
        print(f"      Attendu: approx [-2, 2]")
        print(f"      → L'image n'est pas correctement normalisée")

    # Scénario B: GT Fantôme
    if prob_map.max() > 0.7 and (gt_inst_crop > 0).sum() < 1000:
        print(f"\n   🚨 SCÉNARIO B: GT FANTÔME DÉTECTÉ")
        print(f"      Modèle prédit des noyaux (max prob {prob_map.max():.3f})")
        print(f"      MAIS GT a seulement {(gt_inst_crop > 0).sum()} pixels")
        print(f"      → Extraction GT incorrecte (canaux PanNuke?)")

    # Scénario C: Logits vs Prob
    if np_out.min() < -1 or np_out.max() > 5:
        print(f"\n   🚨 SCÉNARIO C: LOGITS VS PROB DÉTECTÉ")
        print(f"      Sortie brute range [{np_out.min():.3f}, {np_out.max():.3f}]")
        print(f"      → Manque sigmoid()?")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if pixels_over_threshold > 0 and (gt_inst_crop > 0).sum() > 0 and overlap > 0:
        print(f"\n   ✅ PIPELINE SAIN")
        print(f"      - Input: correctement normalisé")
        print(f"      - Output: modèle prédit des noyaux")
        print(f"      - GT: extrait correctement")
        print(f"      - Alignement: OK")
        print(f"\n   → Si Dice reste faible, problème dans post-processing (watershed)")
    else:
        print(f"\n   ❌ PIPELINE CORROMPU")
        print(f"      → Voir alertes ci-dessus pour identifier le problème")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
