#!/usr/bin/env python3
"""
Comparaison FIXED vs OLD modèles sur Ground Truth.

Compare les performances des modèles entraînés avec:
- OLD data: HV maps int8 [-127, 127]
- NEW FIXED data: HV maps float32 [-1, 1]

Génère un rapport comparatif montrant l'amélioration.

Usage:
    python scripts/evaluation/compare_fixed_vs_old.py \
        --dataset_dir data/evaluation/pannuke_fold2_converted \
        --num_samples 50
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.metrics.ground_truth_metrics import evaluate_batch
from src.inference.optimus_gate_inference_multifamily import (
    OptimusGateInferenceMultiFamily,
)


def load_npz_files(dataset_dir: Path, num_samples: int) -> List[Path]:
    """Charge les fichiers .npz du dataset."""
    npz_files = sorted(dataset_dir.glob("*.npz"))

    if not npz_files:
        raise ValueError(f"Aucun fichier .npz trouvé dans {dataset_dir}")

    # Limiter le nombre d'échantillons
    if num_samples > 0:
        npz_files = npz_files[:num_samples]

    return npz_files


def evaluate_checkpoints(
    checkpoint_dir: Path,
    npz_files: List[Path],
    device: str = "cuda",
) -> Dict:
    """
    Évalue un ensemble de checkpoints.

    Returns:
        Dict avec métriques agrégées
    """
    print(f"\n📂 Chargement modèle depuis: {checkpoint_dir}")

    model = OptimusGateInferenceMultiFamily(
        checkpoint_dir=str(checkpoint_dir),
        device=device,
    )

    print(f"   ✅ Modèle chargé ({len(npz_files)} échantillons à évaluer)")

    # Listes pour métriques
    all_dices = []
    all_ajis = []
    all_pqs = []
    all_tp = []
    all_fp = []
    all_fn = []

    for i, npz_file in enumerate(npz_files, 1):
        # Load GT
        data = np.load(npz_file)
        image = data['image']
        gt_inst = data['inst_map']
        gt_type = data['type_map']

        # Prédiction
        from PIL import Image
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray((image * 255).clip(0, 255).astype(np.uint8))

        result = model.predict(pil_image)

        # Métriques
        from src.metrics.ground_truth_metrics import evaluate_predictions

        metrics = evaluate_predictions(
            pred_inst=result['instance_map'],
            pred_type=result['type_map'],
            gt_inst=gt_inst,
            gt_type=gt_type,
            iou_threshold=0.5,
        )

        all_dices.append(metrics.dice)
        all_ajis.append(metrics.aji)
        all_pqs.append(metrics.pq)
        all_tp.append(metrics.tp)
        all_fp.append(metrics.fp)
        all_fn.append(metrics.fn)

        if i % 10 == 0:
            print(f"   [{i}/{len(npz_files)}] Dice: {metrics.dice:.4f}, AJI: {metrics.aji:.4f}")

    # Agrégation
    return {
        'dice_mean': float(np.mean(all_dices)),
        'dice_std': float(np.std(all_dices)),
        'aji_mean': float(np.mean(all_ajis)),
        'aji_std': float(np.std(all_ajis)),
        'pq_mean': float(np.mean(all_pqs)),
        'pq_std': float(np.std(all_pqs)),
        'tp_total': int(np.sum(all_tp)),
        'fp_total': int(np.sum(all_fp)),
        'fn_total': int(np.sum(all_fn)),
        'precision': float(np.sum(all_tp) / (np.sum(all_tp) + np.sum(all_fp) + 1e-8)),
        'recall': float(np.sum(all_tp) / (np.sum(all_tp) + np.sum(all_fn) + 1e-8)),
    }


def generate_comparison_report(
    fixed_metrics: Dict,
    old_metrics: Dict,
    output_path: Path,
):
    """Génère un rapport comparatif formaté."""

    report = []
    report.append("="*70)
    report.append("RAPPORT COMPARATIF: FIXED vs OLD")
    report.append("="*70)
    report.append("")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Dice Score
    report.append("📊 DICE SCORE (Segmentation Binaire)")
    report.append("-" * 70)
    report.append(f"  FIXED: {fixed_metrics['dice_mean']:.4f} ± {fixed_metrics['dice_std']:.4f}")
    report.append(f"  OLD:   {old_metrics['dice_mean']:.4f} ± {old_metrics['dice_std']:.4f}")

    dice_improvement = (fixed_metrics['dice_mean'] - old_metrics['dice_mean']) / old_metrics['dice_mean'] * 100
    if dice_improvement > 0:
        report.append(f"  ✅ Amélioration: +{dice_improvement:.2f}%")
    else:
        report.append(f"  ⚠️  Régression: {dice_improvement:.2f}%")
    report.append("")

    # AJI (Aggregated Jaccard Index)
    report.append("📊 AJI (Instance Segmentation Quality)")
    report.append("-" * 70)
    report.append(f"  FIXED: {fixed_metrics['aji_mean']:.4f} ± {fixed_metrics['aji_std']:.4f}")
    report.append(f"  OLD:   {old_metrics['aji_mean']:.4f} ± {old_metrics['aji_std']:.4f}")

    aji_improvement = (fixed_metrics['aji_mean'] - old_metrics['aji_mean']) / old_metrics['aji_mean'] * 100
    if aji_improvement > 0:
        report.append(f"  ✅ Amélioration: +{aji_improvement:.2f}%")
    else:
        report.append(f"  ⚠️  Régression: {aji_improvement:.2f}%")
    report.append("")

    # PQ (Panoptic Quality)
    report.append("📊 PQ (Panoptic Quality = DQ × SQ)")
    report.append("-" * 70)
    report.append(f"  FIXED: {fixed_metrics['pq_mean']:.4f} ± {fixed_metrics['pq_std']:.4f}")
    report.append(f"  OLD:   {old_metrics['pq_mean']:.4f} ± {old_metrics['pq_std']:.4f}")

    pq_improvement = (fixed_metrics['pq_mean'] - old_metrics['pq_mean']) / old_metrics['pq_mean'] * 100
    if pq_improvement > 0:
        report.append(f"  ✅ Amélioration: +{pq_improvement:.2f}%")
    else:
        report.append(f"  ⚠️  Régression: {pq_improvement:.2f}%")
    report.append("")

    # Détection
    report.append("📊 DÉTECTION (TP/FP/FN)")
    report.append("-" * 70)
    report.append(f"  FIXED: TP={fixed_metrics['tp_total']}, FP={fixed_metrics['fp_total']}, FN={fixed_metrics['fn_total']}")
    report.append(f"         Precision={fixed_metrics['precision']:.4f}, Recall={fixed_metrics['recall']:.4f}")
    report.append(f"  OLD:   TP={old_metrics['tp_total']}, FP={old_metrics['fp_total']}, FN={old_metrics['fn_total']}")
    report.append(f"         Precision={old_metrics['precision']:.4f}, Recall={old_metrics['recall']:.4f}")
    report.append("")

    # Bilan
    report.append("="*70)
    report.append("BILAN")
    report.append("="*70)

    improvements = [dice_improvement, aji_improvement, pq_improvement]
    avg_improvement = np.mean(improvements)

    if avg_improvement > 2.0:
        report.append(f"\n🎉 AMÉLIORATION SIGNIFICATIVE: +{avg_improvement:.2f}% en moyenne")
        report.append("\n✅ RECOMMANDATION: Déployer les modèles FIXED")
    elif avg_improvement > 0:
        report.append(f"\n✅ AMÉLIORATION LÉGÈRE: +{avg_improvement:.2f}% en moyenne")
        report.append("\n⚠️  RECOMMANDATION: Analyser les cas de régression")
    else:
        report.append(f"\n⚠️  RÉGRESSION DÉTECTÉE: {avg_improvement:.2f}% en moyenne")
        report.append("\n❌ RECOMMANDATION: Ne PAS déployer, investiguer la cause")

    report.append("")
    report.append("="*70)

    # Sauvegarder
    output_path.write_text("\n".join(report))
    print(f"\n✅ Rapport sauvegardé: {output_path}")

    # Afficher aussi dans console
    print("\n" + "\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="Comparaison FIXED vs OLD")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory with converted .npz annotations"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--fixed_dir",
        type=str,
        default="models/checkpoints_FIXED",
        help="Directory with FIXED checkpoints"
    )
    parser.add_argument(
        "--old_dir",
        type=str,
        default="models/checkpoints",
        help="Directory with OLD checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparison_FIXED_vs_OLD",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    fixed_dir = Path(args.fixed_dir)
    old_dir = Path(args.old_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("COMPARAISON FIXED vs OLD")
    print("="*70)
    print(f"\nDataset:      {dataset_dir}")
    print(f"FIXED models: {fixed_dir}")
    print(f"OLD models:   {old_dir}")
    print(f"Samples:      {args.num_samples}")
    print(f"Output:       {output_dir}")
    print("")

    # Charger dataset
    npz_files = load_npz_files(dataset_dir, args.num_samples)
    print(f"✅ {len(npz_files)} fichiers chargés")

    # Évaluer FIXED
    print("\n" + "="*70)
    print("ÉVALUATION MODÈLES FIXED")
    print("="*70)
    fixed_metrics = evaluate_checkpoints(fixed_dir, npz_files, args.device)

    # Évaluer OLD
    print("\n" + "="*70)
    print("ÉVALUATION MODÈLES OLD")
    print("="*70)
    old_metrics = evaluate_checkpoints(old_dir, npz_files, args.device)

    # Sauvegarder métriques JSON
    metrics_json = {
        'fixed': fixed_metrics,
        'old': old_metrics,
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(npz_files),
    }

    json_path = output_dir / f"metrics_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path.write_text(json.dumps(metrics_json, indent=2))
    print(f"\n✅ Métriques JSON: {json_path}")

    # Générer rapport
    report_path = output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    generate_comparison_report(fixed_metrics, old_metrics, report_path)


if __name__ == "__main__":
    main()
