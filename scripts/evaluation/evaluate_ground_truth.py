#!/usr/bin/env python3
"""
Évaluation des prédictions Optimus-Gate contre Ground Truth.

Compare les prédictions "aveugles" du modèle avec les annotations expertes
et génère un rapport de fidélité clinique complet.

Usage:
    # Évaluer sur PanNuke Fold 2
    python scripts/evaluation/evaluate_ground_truth.py \
        --dataset_dir data/evaluation/pannuke_fold2_converted \
        --output_dir results/pannuke_fold2 \
        --num_samples 100

    # Évaluer sur CoNSeP
    python scripts/evaluation/evaluate_ground_truth.py \
        --dataset_dir data/evaluation/consep_converted \
        --output_dir results/consep \
        --dataset consep

    # Évaluer sur une seule image
    python scripts/evaluation/evaluate_ground_truth.py \
        --image path/to/image.npz \
        --output_dir results/single
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.metrics.ground_truth_metrics import (
    evaluate_predictions,
    evaluate_batch,
    EvaluationResult,
    PANNUKE_CLASSES,
)
from src.inference.optimus_gate_inference_multifamily import (
    OptimusGateInferenceMultiFamily,
)


def load_ground_truth(npz_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ground truth from converted .npz file.

    Args:
        npz_file: Path to .npz file

    Returns:
        Tuple of (image, inst_map, type_map)
    """
    data = np.load(npz_file)

    # Extract data
    image = data.get('image', None)
    inst_map = data['inst_map']
    type_map = data['type_map']

    # If image not in npz, try to load from corresponding image file
    if image is None:
        # Try to find image with same stem
        image_patterns = [
            npz_file.with_suffix('.png'),
            npz_file.with_suffix('.jpg'),
            npz_file.parent / 'images' / npz_file.with_suffix('.png').name,
        ]
        for img_path in image_patterns:
            if img_path.exists():
                from PIL import Image
                image = np.array(Image.open(img_path))
                break

    if image is None:
        raise ValueError(f"No image found for {npz_file}")

    return image, inst_map, type_map


def collect_dataset_files(
    dataset_dir: Path,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
) -> List[Path]:
    """
    Collect all .npz files from dataset directory.

    Args:
        dataset_dir: Directory with .npz files
        max_samples: Maximum number of samples (None = all)
        shuffle: Shuffle the files

    Returns:
        List of .npz file paths
    """
    npz_files = sorted(dataset_dir.glob("*.npz"))

    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {dataset_dir}")

    if shuffle:
        np.random.shuffle(npz_files)

    if max_samples is not None:
        npz_files = npz_files[:max_samples]

    return npz_files


def evaluate_single_image(
    model: OptimusGateInferenceMultiFamily,
    image: np.ndarray,
    gt_inst: np.ndarray,
    gt_type: np.ndarray,
    iou_threshold: float = 0.5,
    verbose: bool = False,
) -> EvaluationResult:
    """
    Evaluate a single image.

    Args:
        model: Optimus-Gate model
        image: Input image (H, W, 3)
        gt_inst: Ground truth instance map (H, W)
        gt_type: Ground truth type map (H, W)
        iou_threshold: IoU threshold for matching
        verbose: Print detailed info

    Returns:
        EvaluationResult for this image
    """
    # Blind prediction
    pred_result = model.predict(image)

    # Extract predictions
    pred_inst = pred_result['instance_map']
    pred_type_raw = pred_result['nt_mask']

    # Convert nt_mask to type_map format (instance-based)
    # nt_mask uses -1 for background, 0-4 for types
    # type_map uses 0 for background, 1-5 for types
    pred_type = np.zeros_like(pred_inst, dtype=np.uint8)

    for inst_id in range(1, pred_inst.max() + 1):
        inst_mask = pred_inst == inst_id
        if inst_mask.sum() == 0:
            continue

        # Get majority type for this instance
        types_in_inst = pred_type_raw[inst_mask]
        types_valid = types_in_inst[types_in_inst >= 0]

        if len(types_valid) > 0:
            inst_type = int(np.bincount(types_valid).argmax())
            # Convert from 0-indexed to 1-indexed (PanNuke format)
            pred_type[inst_mask] = inst_type + 1

    # Resize predictions if needed
    if pred_inst.shape != gt_inst.shape:
        import cv2
        h, w = gt_inst.shape
        pred_inst = cv2.resize(
            pred_inst.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        pred_type = cv2.resize(
            pred_type.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

    # Evaluate
    result = evaluate_predictions(
        pred_inst=pred_inst,
        gt_inst=gt_inst,
        pred_type=pred_type,
        gt_type=gt_type,
        iou_threshold=iou_threshold,
        num_classes=6  # Including background
    )

    if verbose:
        print(f"  Detected organ: {pred_result['organ'].organ_name}")
        print(f"  Family: {pred_result['family']}")
        print(f"  Instances: GT={result.n_gt}, Pred={result.n_pred}")
        print(f"  Dice: {result.dice:.4f}, AJI: {result.aji:.4f}, PQ: {result.pq:.4f}")

    return result


def save_results(
    result: EvaluationResult,
    output_dir: Path,
    dataset_name: str = "evaluation",
) -> None:
    """
    Save evaluation results to files.

    Args:
        result: Evaluation results
        output_dir: Output directory
        dataset_name: Dataset name for filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Clinical report (text)
    report_file = output_dir / f"clinical_report_{dataset_name}_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(result.format_clinical_report())

    print(f"\n✅ Clinical report saved: {report_file}")

    # 2. Metrics JSON
    metrics = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "global_metrics": {
            "dice": float(result.dice),
            "aji": float(result.aji),
            "pq": float(result.pq),
            "dq": float(result.dq),
            "sq": float(result.sq),
        },
        "detection": {
            "n_gt": result.n_gt,
            "n_pred": result.n_pred,
            "n_tp": result.n_tp,
            "n_fp": result.n_fp,
            "n_fn": result.n_fn,
        },
        "classification": {
            "accuracy": float(result.classification_accuracy),
        },
        "per_class": {},
    }

    for class_id in sorted(result.f1_per_class.keys()):
        class_name = PANNUKE_CLASSES.get(class_id, f"Class {class_id}")
        metrics["per_class"][class_name] = {
            "f1": float(result.f1_per_class[class_id]),
            "precision": float(result.precision_per_class[class_id]),
            "recall": float(result.recall_per_class[class_id]),
            "pq": float(result.pq_per_class.get(class_id, 0.0)),
            "gt_count": result.gt_counts_per_class.get(class_id, 0),
            "pred_count": result.pred_counts_per_class.get(class_id, 0),
        }

    metrics_file = output_dir / f"metrics_{dataset_name}_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Metrics JSON saved: {metrics_file}")

    # 3. Confusion matrix (numpy)
    if result.confusion_matrix is not None:
        cm_file = output_dir / f"confusion_matrix_{dataset_name}_{timestamp}.npy"
        np.save(cm_file, result.confusion_matrix)
        print(f"✅ Confusion matrix saved: {cm_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Optimus-Gate predictions against Ground Truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on PanNuke Fold 2 (100 images)
  python scripts/evaluation/evaluate_ground_truth.py \\
      --dataset_dir data/evaluation/pannuke_fold2_converted \\
      --output_dir results/pannuke_fold2 \\
      --num_samples 100

  # Evaluate on full CoNSeP dataset
  python scripts/evaluation/evaluate_ground_truth.py \\
      --dataset_dir data/evaluation/consep_converted \\
      --output_dir results/consep \\
      --dataset consep

  # Evaluate single image
  python scripts/evaluation/evaluate_ground_truth.py \\
      --image data/evaluation/consep_converted/test_001.npz \\
      --output_dir results/single
        """
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        help="Directory with converted .npz files"
    )

    parser.add_argument(
        "--image",
        type=Path,
        help="Single .npz file to evaluate"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for results"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation",
        help="Dataset name (for filenames)"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )

    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for instance matching (default: 0.5)"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models/checkpoints",
        help="Directory with model checkpoints"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-image results"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load model
    print("\n" + "="*70)
    print("LOADING OPTIMUS-GATE MODEL")
    print("="*70)

    model = OptimusGateInferenceMultiFamily(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    print(f"\n✅ Model loaded on {args.device}")

    # Collect files
    print("\n" + "="*70)
    print("COLLECTING GROUND TRUTH FILES")
    print("="*70)

    if args.image:
        # Single image mode
        npz_files = [args.image]
        print(f"Evaluating single image: {args.image}")
    elif args.dataset_dir:
        # Batch mode
        npz_files = collect_dataset_files(
            args.dataset_dir,
            max_samples=args.num_samples,
            shuffle=True
        )
        print(f"Found {len(npz_files)} images")
    else:
        print("❌ Error: Either --dataset_dir or --image is required")
        return

    # Evaluate
    print("\n" + "="*70)
    print("RUNNING EVALUATION (BLIND PREDICTIONS)")
    print("="*70)

    predictions = []
    ground_truths = []

    for npz_file in tqdm(npz_files, desc="Evaluating"):
        try:
            # Load GT
            image, gt_inst, gt_type = load_ground_truth(npz_file)

            # Predict
            pred_result = model.predict(image)
            pred_inst = pred_result['instance_map']
            pred_type_raw = pred_result['nt_mask']

            # Convert nt_mask to type_map
            pred_type = np.zeros_like(pred_inst, dtype=np.uint8)
            for inst_id in range(1, pred_inst.max() + 1):
                inst_mask = pred_inst == inst_id
                if inst_mask.sum() == 0:
                    continue
                types_in_inst = pred_type_raw[inst_mask]
                types_valid = types_in_inst[types_in_inst >= 0]
                if len(types_valid) > 0:
                    inst_type = int(np.bincount(types_valid).argmax())
                    pred_type[inst_mask] = inst_type + 1

            # Resize if needed
            if pred_inst.shape != gt_inst.shape:
                import cv2
                h, w = gt_inst.shape
                pred_inst = cv2.resize(
                    pred_inst.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
                pred_type = cv2.resize(
                    pred_type.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.uint8)

            predictions.append((pred_inst, pred_type))
            ground_truths.append((gt_inst, gt_type))

        except Exception as e:
            print(f"\n⚠️ Error processing {npz_file.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    if len(predictions) == 0:
        print("❌ No successful predictions")
        return

    print(f"\n✅ Evaluated {len(predictions)}/{len(npz_files)} images successfully")

    # Compute aggregated metrics
    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70)

    result = evaluate_batch(
        predictions=predictions,
        ground_truths=ground_truths,
        iou_threshold=args.iou_threshold,
        num_classes=6
    )

    # Display results
    print("\n" + result.format_clinical_report())

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    save_results(result, args.output_dir, args.dataset)

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir.absolute()}")

    # Summary
    print("\n📊 SUMMARY")
    print(f"  Dice: {result.dice:.4f}")
    print(f"  AJI:  {result.aji:.4f}")
    print(f"  PQ:   {result.pq:.4f}")
    print(f"  Classification Accuracy: {result.classification_accuracy:.2%}")

    # Check against targets
    print("\n🎯 TARGETS")
    targets = {
        "Dice": (result.dice, 0.95, 0.90, 0.85),
        "AJI": (result.aji, 0.80, 0.70, 0.60),
        "PQ": (result.pq, 0.70, 0.60, 0.50),
    }

    for metric_name, (value, target, acceptable, critical) in targets.items():
        if value >= target:
            status = "✅ EXCELLENT"
        elif value >= acceptable:
            status = "🟡 ACCEPTABLE"
        elif value >= critical:
            status = "🟠 BELOW TARGET"
        else:
            status = "🔴 CRITICAL"

        print(f"  {metric_name}: {value:.4f} (target: {target:.2f}) {status}")


if __name__ == "__main__":
    main()
