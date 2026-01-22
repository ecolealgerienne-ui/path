"""
YOLO26 Test Script for APCData — Step-by-Step Validation

Ce script teste YOLO26 sur APCData de manière progressive:
1. Vérifier l'installation ultralytics
2. Charger le modèle YOLO26 pré-entraîné
3. Tester l'inférence sur quelques images
4. Comparer avec les annotations GT
5. Afficher les métriques de base

IMPORTANT: Ce script teste le modèle GÉNÉRALISTE (COCO pretrained).
           Il NE détectera PAS les cellules directement (classes différentes).
           L'objectif est de valider que le pipeline fonctionne.

Usage:
    python scripts/cytology/01_test_yolo26_apcdata.py --data_dir data/raw/apcdata/APCData_YOLO

Author: V15 Cytology Branch
Date: 2026-01-22
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json


def print_step(step_num: int, description: str):
    """Print formatted step header"""
    print("\n" + "=" * 80)
    print(f"  STEP {step_num}: {description}")
    print("=" * 80)


def print_substep(description: str):
    """Print substep"""
    print(f"\n  → {description}")


def print_success(message: str):
    """Print success message"""
    print(f"  ✅ {message}")


def print_error(message: str):
    """Print error message"""
    print(f"  ❌ {message}")


def print_warning(message: str):
    """Print warning message"""
    print(f"  ⚠️  {message}")


def print_info(message: str):
    """Print info message"""
    print(f"  ℹ️  {message}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1: VERIFY INSTALLATION
# ═════════════════════════════════════════════════════════════════════════════

def verify_installation() -> bool:
    """Verify ultralytics is installed and check version"""
    print_step(1, "VERIFY ULTRALYTICS INSTALLATION")

    try:
        import ultralytics
        print_success(f"ultralytics version: {ultralytics.__version__}")

        # Check if YOLO26 is available
        from ultralytics import YOLO
        print_success("YOLO class imported successfully")

        # Check torch
        import torch
        print_success(f"PyTorch version: {torch.__version__}")
        print_info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        return True

    except ImportError as e:
        print_error(f"Import error: {e}")
        print_info("Install with: pip install ultralytics")
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2: VERIFY DATASET STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

def verify_dataset(data_dir: str) -> Dict:
    """Verify APCData YOLO dataset structure"""
    print_step(2, "VERIFY DATASET STRUCTURE")

    data_path = Path(data_dir)

    # Check root exists
    if not data_path.exists():
        print_error(f"Dataset directory not found: {data_dir}")
        return None

    print_success(f"Dataset root found: {data_dir}")

    # Check subdirectories
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    stats = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "num_images": 0,
        "num_labels": 0,
        "matched": 0,
        "sample_images": []
    }

    # Count images
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        stats["num_images"] = len(image_files)
        stats["sample_images"] = [f.name for f in image_files[:5]]
        print_success(f"Images directory: {len(image_files)} images found")
    else:
        print_error(f"Images directory not found: {images_dir}")
        return None

    # Count labels
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        # Filter out Zone.Identifier files (Windows)
        label_files = [f for f in label_files if "Zone.Identifier" not in f.name]
        stats["num_labels"] = len(label_files)
        print_success(f"Labels directory: {len(label_files)} label files found")
    else:
        print_error(f"Labels directory not found: {labels_dir}")
        return None

    # Check matching
    image_basenames = {f.stem for f in (images_dir.glob("*.jpg"))}
    image_basenames.update({f.stem for f in images_dir.glob("*.png")})
    label_basenames = {f.stem for f in label_files}

    matched = image_basenames & label_basenames
    stats["matched"] = len(matched)

    if len(matched) == len(image_basenames):
        print_success(f"All {len(matched)} images have matching labels")
    else:
        print_warning(f"Matched: {len(matched)} / {len(image_basenames)} images")
        missing = image_basenames - label_basenames
        if missing:
            print_info(f"Images without labels: {list(missing)[:5]}...")

    # Check classes.txt
    classes_file = data_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file) as f:
            classes = [line.strip() for line in f.readlines()]
        stats["classes"] = classes
        print_success(f"Classes: {classes}")
    else:
        print_warning("classes.txt not found (optional)")
        stats["classes"] = ["NILM", "ASCUS", "ASCH", "LSIL", "HSIL", "SCC"]

    return stats


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3: LOAD SAMPLE ANNOTATIONS
# ═════════════════════════════════════════════════════════════════════════════

def load_sample_annotations(data_dir: str, n_samples: int = 3) -> List[Dict]:
    """Load and display sample YOLO annotations"""
    print_step(3, "LOAD SAMPLE ANNOTATIONS")

    labels_dir = Path(data_dir) / "labels"
    label_files = sorted(labels_dir.glob("*.txt"))
    label_files = [f for f in label_files if "Zone.Identifier" not in f.name]

    samples = []

    for label_file in label_files[:n_samples]:
        print_substep(f"Loading: {label_file.name}")

        with open(label_file) as f:
            lines = f.readlines()

        annotations = []
        class_counts = {}

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                annotations.append({
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                })

                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        sample = {
            "file": label_file.name,
            "num_cells": len(annotations),
            "class_counts": class_counts,
            "annotations": annotations[:3]  # First 3 for display
        }
        samples.append(sample)

        print_info(f"  Cells: {len(annotations)}")
        print_info(f"  Class distribution: {class_counts}")

    return samples


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4: LOAD YOLO26 MODEL
# ═════════════════════════════════════════════════════════════════════════════

def load_yolo26_model(model_name: str = "yolo26n.pt") -> Optional[object]:
    """Load YOLO26 pretrained model"""
    print_step(4, f"LOAD YOLO26 MODEL ({model_name})")

    try:
        from ultralytics import YOLO

        print_substep(f"Loading {model_name}...")
        model = YOLO(model_name)

        print_success(f"Model loaded: {model_name}")
        print_info(f"Model type: {type(model).__name__}")

        # Model info
        if hasattr(model, 'names'):
            print_info(f"Pretrained classes: {len(model.names)} (COCO)")
            print_info(f"First 5 classes: {list(model.names.values())[:5]}")

        return model

    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5: TEST INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def test_inference(model, data_dir: str, n_samples: int = 2) -> List[Dict]:
    """Test inference on sample images"""
    print_step(5, "TEST INFERENCE ON SAMPLE IMAGES")

    images_dir = Path(data_dir) / "images"
    image_files = sorted(images_dir.glob("*.jpg"))[:n_samples]

    if not image_files:
        image_files = sorted(images_dir.glob("*.png"))[:n_samples]

    results_summary = []

    for img_path in image_files:
        print_substep(f"Running inference on: {img_path.name}")

        try:
            # Run inference
            results = model(str(img_path), verbose=False)

            if results and len(results) > 0:
                result = results[0]

                # Get detections
                boxes = result.boxes
                num_detections = len(boxes) if boxes is not None else 0

                summary = {
                    "image": img_path.name,
                    "num_detections": num_detections,
                    "detections": []
                }

                print_info(f"  Detections: {num_detections}")

                if num_detections > 0 and boxes is not None:
                    # Show top 5 detections
                    for i, box in enumerate(boxes[:5]):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id]

                        det = {
                            "class": class_name,
                            "class_id": cls_id,
                            "confidence": conf
                        }
                        summary["detections"].append(det)
                        print_info(f"    [{i+1}] {class_name}: {conf:.2%}")

                results_summary.append(summary)

        except Exception as e:
            print_error(f"Inference failed: {e}")
            results_summary.append({
                "image": img_path.name,
                "error": str(e)
            })

    return results_summary


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 6: SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(dataset_stats: Dict, inference_results: List[Dict]):
    """Print final summary"""
    print_step(6, "SUMMARY")

    print_substep("Dataset:")
    print_info(f"  Images: {dataset_stats.get('num_images', 'N/A')}")
    print_info(f"  Labels: {dataset_stats.get('num_labels', 'N/A')}")
    print_info(f"  Matched: {dataset_stats.get('matched', 'N/A')}")

    print_substep("YOLO26 Inference Test:")
    for result in inference_results:
        if "error" in result:
            print_error(f"  {result['image']}: {result['error']}")
        else:
            print_info(f"  {result['image']}: {result['num_detections']} detections")

    print_substep("Next Steps:")
    print_info("1. Le modèle COCO ne détecte pas les cellules (classes différentes)")
    print_info("2. Pour détecter les cellules, il faut ENTRAÎNER sur APCData")
    print_info("3. Commande: model.train(data='configs/cytology/apcdata_yolo.yaml', epochs=100)")

    print("\n" + "=" * 80)
    print("  TEST COMPLETED")
    print("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test YOLO26 on APCData")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/apcdata/APCData_YOLO",
        help="Path to APCData_YOLO directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        choices=["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo11n.pt", "yolov8n.pt"],
        help="YOLO model to use"
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference test (useful if no GPU)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  YOLO26 TEST SCRIPT FOR APCDATA")
    print("  V15 Cytology Pipeline Validation")
    print("=" * 80)

    # Step 1: Verify installation
    if not verify_installation():
        print_error("Installation verification failed. Exiting.")
        return 1

    # Step 2: Verify dataset
    dataset_stats = verify_dataset(args.data_dir)
    if dataset_stats is None:
        print_error("Dataset verification failed. Exiting.")
        return 1

    # Step 3: Load sample annotations
    samples = load_sample_annotations(args.data_dir, n_samples=3)

    # Step 4: Load model
    model = load_yolo26_model(args.model)
    if model is None:
        print_error("Model loading failed. Exiting.")
        return 1

    # Step 5: Test inference
    inference_results = []
    if not args.skip_inference:
        inference_results = test_inference(model, args.data_dir, n_samples=2)
    else:
        print_step(5, "INFERENCE SKIPPED (--skip_inference)")

    # Step 6: Summary
    print_summary(dataset_stats, inference_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
