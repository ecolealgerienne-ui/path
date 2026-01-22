"""
YOLO26 Training on APCData — Cell Detection

Ce script entraîne YOLO26 pour la détection de cellules sur APCData.

Prérequis:
    1. Exécuter 02_prepare_apcdata_split.py pour créer train/val split
    2. Le fichier data.yaml doit exister

Usage:
    python scripts/cytology/03_train_yolo26_apcdata.py \
        --data data/raw/apcdata/APCData_YOLO/data.yaml \
        --model yolo26n.pt \
        --epochs 100 \
        --imgsz 640

Modèles disponibles:
    - yolo26n.pt: Nano (rapide, moins précis)
    - yolo26s.pt: Small (bon équilibre)
    - yolo26m.pt: Medium (plus précis)
    - yolo26l.pt: Large (haute précision)
    - yolo26x.pt: Extra-large (maximum précision)

Author: V15 Cytology Branch
Date: 2026-01-22
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_info(message: str):
    print(f"  ℹ️  {message}")


def print_success(message: str):
    print(f"  ✅ {message}")


def print_error(message: str):
    print(f"  ❌ {message}")


def verify_prerequisites(data_yaml: str) -> bool:
    """Verify training prerequisites."""

    data_path = Path(data_yaml)

    if not data_path.exists():
        print_error(f"data.yaml not found: {data_yaml}")
        print_info("Run 02_prepare_apcdata_split.py first")
        return False

    # Check train/val directories
    data_dir = data_path.parent
    train_images = data_dir / "train" / "images"
    val_images = data_dir / "val" / "images"

    if not train_images.exists():
        print_error(f"train/images not found: {train_images}")
        return False

    if not val_images.exists():
        print_error(f"val/images not found: {val_images}")
        return False

    n_train = len(list(train_images.glob("*")))
    n_val = len(list(val_images.glob("*")))

    print_success(f"data.yaml: {data_yaml}")
    print_success(f"Train images: {n_train}")
    print_success(f"Val images: {n_val}")

    return True


def train_yolo(
    data_yaml: str,
    model: str = "yolo26n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/cytology",
    name: str = None,
    resume: bool = False,
    patience: int = 50,
    save_period: int = 10
) -> str:
    """
    Train YOLO26 on APCData.

    Returns:
        Path to best weights
    """
    from ultralytics import YOLO

    # Generate run name
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model).stem
        name = f"apcdata_{model_name}_{timestamp}"

    print_header("TRAINING CONFIGURATION")
    print_info(f"Model: {model}")
    print_info(f"Data: {data_yaml}")
    print_info(f"Epochs: {epochs}")
    print_info(f"Image size: {imgsz}")
    print_info(f"Batch size: {batch}")
    print_info(f"Device: {device}")
    print_info(f"Project: {project}")
    print_info(f"Name: {name}")
    print_info(f"Patience (early stopping): {patience}")

    # Load model
    print_header("LOADING MODEL")

    if resume and Path(f"{project}/{name}/weights/last.pt").exists():
        print_info("Resuming from last checkpoint...")
        yolo_model = YOLO(f"{project}/{name}/weights/last.pt")
    else:
        print_info(f"Loading pretrained {model}...")
        yolo_model = YOLO(model)

    print_success("Model loaded")

    # Train
    print_header("STARTING TRAINING")
    print_info("This may take a while depending on your GPU...")

    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        plots=True,
        verbose=True,
        # Augmentation settings optimized for cytology
        hsv_h=0.015,  # Hue augmentation (subtle for staining variation)
        hsv_s=0.4,    # Saturation
        hsv_v=0.4,    # Value
        degrees=180,  # Full rotation (cells can be at any angle)
        translate=0.1,
        scale=0.5,
        flipud=0.5,   # Vertical flip
        fliplr=0.5,   # Horizontal flip
        mosaic=0.5,   # Mosaic augmentation (reduced for medical)
        mixup=0.0,    # No mixup (preserve cell integrity)
    )

    # Results
    print_header("TRAINING COMPLETED")

    best_weights = Path(project) / name / "weights" / "best.pt"
    last_weights = Path(project) / name / "weights" / "last.pt"

    if best_weights.exists():
        print_success(f"Best weights: {best_weights}")
    if last_weights.exists():
        print_success(f"Last weights: {last_weights}")

    # Print metrics
    if hasattr(results, 'results_dict'):
        print_header("FINAL METRICS")
        for key, value in results.results_dict.items():
            if isinstance(value, float):
                print_info(f"{key}: {value:.4f}")

    return str(best_weights)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 on APCData")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/apcdata/APCData_YOLO/data.yaml",
        help="Path to data.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        choices=["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
        help="YOLO26 model variant"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device (0, 1, 2, etc.) or 'cpu'"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/cytology",
        help="Project directory for results"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (auto-generated if not specified)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  YOLO26 TRAINING ON APCDATA")
    print("  V15 Cytology Pipeline")
    print("=" * 80)

    # Verify prerequisites
    print_header("VERIFYING PREREQUISITES")

    if not verify_prerequisites(args.data):
        return 1

    # Check ultralytics
    try:
        import ultralytics
        print_success(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print_error("Ultralytics not installed. Run: pip install ultralytics")
        return 1

    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print_info("CUDA not available, training on CPU (slow)")
        args.device = "cpu"

    # Train
    best_weights = train_yolo(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience
    )

    # Summary
    print("\n" + "=" * 80)
    print("  TRAINING SUMMARY")
    print("=" * 80)
    print(f"\n  Best weights saved to: {best_weights}")
    print(f"\n  Next steps:")
    print(f"    1. Evaluate: yolo val model={best_weights} data={args.data}")
    print(f"    2. Test: python scripts/cytology/04_test_yolo26_trained.py --weights {best_weights}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
