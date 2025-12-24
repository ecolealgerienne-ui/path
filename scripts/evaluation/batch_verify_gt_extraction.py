#!/usr/bin/env python3
"""
Batch verification: connectedComponents vs Native PanNuke extraction.

Tests multiple samples and generates statistical report.

Usage:
    python scripts/evaluation/batch_verify_gt_extraction.py \
        --family epidermal \
        --n_samples 20 \
        --data_dir /home/amar/data/PanNuke
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import PANNUKE_IMAGE_SIZE


def extract_gt_connectedcomponents(np_target: np.ndarray) -> np.ndarray:
    """Extraction GT méthode connectedComponents (BUGGY)."""
    np_binary = (np_target > 0.5).astype(np.uint8)
    _, inst_map = cv2.connectedComponents(np_binary)
    return inst_map.astype(np.int32)


def extract_gt_pannuke_native(mask: np.ndarray) -> np.ndarray:
    """Extraction GT méthode CORRECTE (native PanNuke IDs)."""
    inst_map = np.zeros((PANNUKE_IMAGE_SIZE, PANNUKE_IMAGE_SIZE), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: IDs d'instances natifs PanNuke
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
    epithelial_binary = (mask[:, :, 5] > 0).astype(np.uint8)
    if epithelial_binary.sum() > 0:
        _, epithelial_labels = cv2.connectedComponents(epithelial_binary)
        epithelial_ids = np.unique(epithelial_labels)
        epithelial_ids = epithelial_ids[epithelial_ids > 0]

        for inst_id in epithelial_ids:
            inst_mask = epithelial_labels == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    return inst_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', type=str, required=True,
                        choices=['glandular', 'digestive', 'urologic', 'epidermal', 'respiratory'])
    parser.add_argument('--n_samples', type=int, default=20,
                        help="Nombre d'échantillons à tester")
    parser.add_argument('--data_dir', type=Path, required=True,
                        help="Répertoire PanNuke brut")

    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent.parent

    print(f"\n{'='*70}")
    print(f"BATCH VERIFICATION GT EXTRACTION - Famille {args.family.upper()}")
    print(f"{'='*70}\n")

    # ========================================================================
    # PARTIE 1: Charger données FIXED
    # ========================================================================

    print("📥 Chargement données FIXED...")

    fixed_dir = PROJECT_ROOT / "data" / "family_FIXED"
    fixed_file = fixed_dir / f"{args.family}_data_FIXED.npz"

    if not fixed_file.exists():
        raise FileNotFoundError(
            f"FIXED data not found: {fixed_file}\n"
            f"Generate using:\n"
            f"  python scripts/preprocessing/prepare_family_data_FIXED.py \\\n"
            f"      --data_dir {args.data_dir} \\\n"
            f"      --family {args.family}"
        )

    data = np.load(fixed_file)

    total_samples = len(data['np_targets'])
    n_test = min(args.n_samples, total_samples)

    print(f"   Total samples: {total_samples}")
    print(f"   Testing: {n_test} samples\n")

    # ========================================================================
    # PARTIE 2: Charger PanNuke brut (par fold)
    # ========================================================================

    print("📥 Chargement PanNuke brut...")

    # Organiser par fold pour chargement efficace
    fold_samples = {0: [], 1: [], 2: []}

    for idx in range(n_test):
        fold_id = int(data['fold_ids'][idx])
        image_id = int(data['image_ids'][idx])
        fold_samples[fold_id].append((idx, image_id))

    # Charger masks par fold (memory-mapped)
    fold_masks = {}
    for fold in [0, 1, 2]:
        if len(fold_samples[fold]) > 0:
            fold_dir = args.data_dir / f"fold{fold}"
            masks_path = fold_dir / "masks.npy"
            if masks_path.exists():
                fold_masks[fold] = np.load(masks_path, mmap_mode='r')
                print(f"   Fold {fold}: {len(fold_samples[fold])} samples")

    print()

    # ========================================================================
    # PARTIE 3: Comparaison batch
    # ========================================================================

    print("🔄 Comparaison connectedComponents vs Native PanNuke...\n")

    results = []

    for idx in tqdm(range(n_test), desc="Processing"):
        np_target = data['np_targets'][idx]
        fold_id = int(data['fold_ids'][idx])
        image_id = int(data['image_ids'][idx])

        # Extraction connectedComponents
        inst_gt_cc = extract_gt_connectedcomponents(np_target)
        n_cc = len(np.unique(inst_gt_cc)) - 1

        # Extraction native PanNuke
        mask = np.array(fold_masks[fold_id][image_id])
        inst_gt_native = extract_gt_pannuke_native(mask)
        n_native = len(np.unique(inst_gt_native)) - 1

        # Détails par canal
        channel_counts = {}
        for c in range(1, 6):
            channel_mask = mask[:, :, c]
            n_unique = len(np.unique(channel_mask)) - 1
            if n_unique > 0:
                channel_names = ['Neo', 'Infl', 'Conn', 'Dead', 'Epit']
                channel_counts[channel_names[c-1]] = n_unique

        results.append({
            'idx': idx,
            'fold': fold_id,
            'image_id': image_id,
            'n_cc': n_cc,
            'n_native': n_native,
            'lost': n_native - n_cc,
            'loss_pct': 100 * (n_native - n_cc) / n_native if n_native > 0 else 0,
            'channels': channel_counts
        })

    # ========================================================================
    # PARTIE 4: Rapport statistique
    # ========================================================================

    print(f"\n{'='*70}")
    print("RÉSULTATS STATISTIQUES")
    print(f"{'='*70}\n")

    # Filtrer images non-vides
    non_empty = [r for r in results if r['n_native'] > 0]
    n_non_empty = len(non_empty)

    if n_non_empty == 0:
        print("⚠️  Aucune image avec cellules détectées dans les échantillons testés")
        return

    # Statistiques globales
    total_cc = sum(r['n_cc'] for r in non_empty)
    total_native = sum(r['n_native'] for r in non_empty)
    total_lost = total_native - total_cc
    avg_loss_pct = np.mean([r['loss_pct'] for r in non_empty])

    print(f"Images testées:           {n_test}")
    print(f"Images avec cellules:     {n_non_empty}")
    print(f"Images background:        {n_test - n_non_empty}\n")

    print(f"Instances connectedComponents:  {total_cc:4d}")
    print(f"Instances PanNuke Native:       {total_native:4d}")
    print(f"Instances perdues:              {total_lost:4d} ({avg_loss_pct:.1f}%)\n")

    # Distribution
    loss_pcts = [r['loss_pct'] for r in non_empty]
    print("Distribution perte par image:")
    print(f"  Min:     {min(loss_pcts):.1f}%")
    print(f"  Q25:     {np.percentile(loss_pcts, 25):.1f}%")
    print(f"  Médiane: {np.median(loss_pcts):.1f}%")
    print(f"  Q75:     {np.percentile(loss_pcts, 75):.1f}%")
    print(f"  Max:     {max(loss_pcts):.1f}%\n")

    # Exemples extrêmes
    print("Cas extrêmes:")

    # Pire cas
    worst = max(non_empty, key=lambda r: r['loss_pct'])
    print(f"\n  🔴 Pire cas (idx {worst['idx']}):")
    print(f"     connectedComponents: {worst['n_cc']} instances")
    print(f"     PanNuke Native:      {worst['n_native']} instances")
    print(f"     Perte:               {worst['lost']} instances ({worst['loss_pct']:.1f}%)")
    if worst['channels']:
        print(f"     Canaux: {worst['channels']}")

    # Meilleur cas (parmi ceux avec perte)
    with_loss = [r for r in non_empty if r['lost'] > 0]
    if with_loss:
        best_with_loss = min(with_loss, key=lambda r: r['loss_pct'])
        print(f"\n  🟡 Meilleur cas avec perte (idx {best_with_loss['idx']}):")
        print(f"     connectedComponents: {best_with_loss['n_cc']} instances")
        print(f"     PanNuke Native:      {best_with_loss['n_native']} instances")
        print(f"     Perte:               {best_with_loss['lost']} instances ({best_with_loss['loss_pct']:.1f}%)")

    # Cas sans perte
    no_loss = [r for r in non_empty if r['lost'] == 0]
    if no_loss:
        print(f"\n  🟢 Images sans perte: {len(no_loss)}/{n_non_empty}")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}\n")

    if avg_loss_pct > 10:
        print("❌ PROBLÈME CONFIRMÉ:")
        print(f"   Perte moyenne: {avg_loss_pct:.1f}%")
        print(f"   {total_lost} instances fusionnées sur {total_native} ({n_non_empty} images)")
        print()
        print("   Impact:")
        print("   → eval_aji_from_training_data.py utilise instances fusionnées")
        print("   → AJI 0.94 est une FAUSSE métrique (bad vs bad)")
        print("   → Le modèle a appris des gradients HV FAIBLES")
        print("   → Watershed échoue à séparer les cellules touchantes")
    else:
        print("✅ Impact limité:")
        print(f"   Perte moyenne: {avg_loss_pct:.1f}%")

    print(f"\n{'='*70}\n")

    # Sauvegarder rapport détaillé
    output_file = PROJECT_ROOT / "results" / f"batch_verify_{args.family}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"BATCH VERIFICATION GT EXTRACTION - {args.family.upper()}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Total tested: {n_test}\n")
        f.write(f"Non-empty: {n_non_empty}\n")
        f.write(f"Total instances CC: {total_cc}\n")
        f.write(f"Total instances Native: {total_native}\n")
        f.write(f"Total lost: {total_lost} ({avg_loss_pct:.1f}%)\n\n")

        f.write("Per-image results:\n")
        f.write("Idx\tFold\tImageID\tCC\tNative\tLost\tLoss%\tChannels\n")
        for r in results:
            channels_str = ','.join(f"{k}:{v}" for k, v in r['channels'].items())
            f.write(f"{r['idx']}\t{r['fold']}\t{r['image_id']}\t"
                   f"{r['n_cc']}\t{r['n_native']}\t{r['lost']}\t"
                   f"{r['loss_pct']:.1f}\t{channels_str}\n")

    print(f"📄 Rapport détaillé: {output_file}")


if __name__ == '__main__':
    main()
