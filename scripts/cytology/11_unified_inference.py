"""
Pipeline d'Inférence Unifié — Cell Triage + MultiHead Bethesda

Ce script intègre les deux modèles entraînés pour une inférence complète:
1. Cell Triage: Filtre les patches sans cellules (96.28% recall)
2. MultiHead Bethesda: Classifie les cellules (96.88% binary, 85.48% severity)

Pipeline:
    Image LBC
        ↓
    Sliding Window 224×224, stride 112 (50% overlap)
        ↓
    H-Optimus-0 → Features (1536-dim)
        ↓
    Cell Triage → Filtre patches vides
        ↓
    MultiHead Bethesda → Binary + Severity + Fine-grained
        ↓
    Rapport diagnostique

Usage:
    python scripts/cytology/11_unified_inference.py \
        --image path/to/image.jpg \
        --cell_triage models/cytology/cell_triage.pt \
        --bethesda models/cytology/multihead_bethesda_combined.pt \
        --output results/

Author: V15.2 Cytology Branch
Date: 2026-01-23
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
#  CONFIGURATION
# =============================================================================

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224

BETHESDA_CLASSES = {
    0: "NILM",
    1: "ASCUS",
    2: "ASCH",
    3: "LSIL",
    4: "HSIL",
    5: "SCC"
}

SEVERITY_MAPPING = {
    0: "Normal",
    1: "Low-grade",
    2: "High-grade",
    3: "Low-grade",
    4: "High-grade",
    5: "High-grade"
}


# =============================================================================
#  DATA CLASSES
# =============================================================================

@dataclass
class PatchResult:
    """Résultat pour un patch"""
    patch_idx: int
    x: int
    y: int
    has_cells: bool
    cell_confidence: float
    predicted_class: int
    class_name: str
    is_abnormal: bool
    is_high_grade: bool
    severity: str
    confidence: float


@dataclass
class ImageDiagnosis:
    """Diagnostic agrégé pour une image"""
    image_path: str
    total_patches: int
    patches_with_cells: int
    abnormal_patches: int
    high_grade_patches: int
    binary_result: str  # "NORMAL" ou "ABNORMAL"
    severity_result: str  # "Normal", "Low-grade", "High-grade"
    recommendation: str
    class_distribution: Dict[str, int]
    patch_results: List[PatchResult]


# =============================================================================
#  MODELS
# =============================================================================

class CellTriageClassifier(nn.Module):
    """Cell Triage Model (from 07_train_cell_triage.py)"""

    def __init__(self, input_dim: int = 1536, hidden_dims: Tuple[int, int] = (256, 64)):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dims[1], 2)
        )

    def forward(self, x):
        return self.classifier(x)

    def predict(self, x, threshold: float = 0.01):
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            predictions = (probs[:, 1] >= threshold).long()
            return predictions, probs


class MultiHeadBethesdaClassifier(nn.Module):
    """MultiHead Bethesda Model (from 08_train_multihead_bethesda.py)"""

    def __init__(self, input_dim: int = 1536, hidden_dims: Tuple[int, int] = (512, 256)):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.finegrained_head = nn.Sequential(
            nn.Linear(hidden_dims[1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        shared = self.shared(x)
        return {
            'binary': self.binary_head(shared),
            'severity': self.severity_head(shared),
            'finegrained': self.finegrained_head(shared)
        }

    def predict(self, x, binary_threshold: float = 0.3, severity_threshold: float = 0.4):
        with torch.no_grad():
            outputs = self.forward(x)

            binary_probs = F.softmax(outputs['binary'], dim=1)
            severity_probs = F.softmax(outputs['severity'], dim=1)
            finegrained_probs = F.softmax(outputs['finegrained'], dim=1)

            binary_preds = (binary_probs[:, 1] >= binary_threshold).long()
            severity_preds = (severity_probs[:, 1] >= severity_threshold).long()
            finegrained_preds = finegrained_probs.argmax(dim=1)

            return {
                'binary': binary_preds,
                'binary_prob': binary_probs[:, 1],
                'severity': severity_preds,
                'severity_prob': severity_probs[:, 1],
                'finegrained': finegrained_preds,
                'finegrained_prob': finegrained_probs
            }


# =============================================================================
#  PIPELINE
# =============================================================================

class UnifiedInferencePipeline:
    """Pipeline unifié Cell Triage + MultiHead Bethesda"""

    def __init__(
        self,
        cell_triage_path: str,
        bethesda_path: str,
        device: str = "cuda",
        triage_threshold: float = 0.01,
        binary_threshold: float = 0.3,
        severity_threshold: float = 0.4
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.triage_threshold = triage_threshold
        self.binary_threshold = binary_threshold
        self.severity_threshold = severity_threshold

        self.hoptimus = None
        self.cell_triage = None
        self.bethesda = None

        self._load_models(cell_triage_path, bethesda_path)

    def _load_models(self, cell_triage_path: str, bethesda_path: str):
        """Charge tous les modèles"""

        print("  [INFO] Loading H-Optimus-0...")
        try:
            import timm
            self.hoptimus = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False
            )
            self.hoptimus = self.hoptimus.to(self.device)
            self.hoptimus.eval()
            for param in self.hoptimus.parameters():
                param.requires_grad = False
            print(f"  [OK] H-Optimus-0 loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load H-Optimus-0: {e}")

        print("  [INFO] Loading Cell Triage model...")
        checkpoint = torch.load(cell_triage_path, map_location=self.device, weights_only=False)
        self.cell_triage = CellTriageClassifier(
            input_dim=checkpoint.get('input_dim', 1536),
            hidden_dims=checkpoint.get('hidden_dims', (256, 64))
        )
        self.cell_triage.load_state_dict(checkpoint['model_state_dict'])
        self.cell_triage = self.cell_triage.to(self.device)
        self.cell_triage.eval()
        print(f"  [OK] Cell Triage loaded")

        print("  [INFO] Loading MultiHead Bethesda model...")
        checkpoint = torch.load(bethesda_path, map_location=self.device, weights_only=False)
        self.bethesda = MultiHeadBethesdaClassifier(
            input_dim=checkpoint.get('input_dim', 1536),
            hidden_dims=checkpoint.get('hidden_dims', (512, 256))
        )
        self.bethesda.load_state_dict(checkpoint['model_state_dict'])
        self.bethesda = self.bethesda.to(self.device)
        self.bethesda.eval()
        print(f"  [OK] MultiHead Bethesda loaded")

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Préprocess image pour H-Optimus"""
        if image.shape[0] != HOPTIMUS_INPUT_SIZE or image.shape[1] != HOPTIMUS_INPUT_SIZE:
            image = cv2.resize(image, (HOPTIMUS_INPUT_SIZE, HOPTIMUS_INPUT_SIZE))

        image = image.astype(np.float32) / 255.0
        mean = np.array(HOPTIMUS_MEAN, dtype=np.float32)
        std = np.array(HOPTIMUS_STD, dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _extract_features(self, images: List[np.ndarray], batch_size: int = 16) -> torch.Tensor:
        """Extrait features H-Optimus pour un batch"""
        all_features = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            tensors = torch.cat([self._preprocess(img) for img in batch], dim=0)

            with torch.no_grad():
                features = self.hoptimus.forward_features(tensors)
                if len(features.shape) == 3:
                    cls_tokens = features[:, 0, :]
                else:
                    cls_tokens = features
                all_features.append(cls_tokens)

        return torch.cat(all_features, dim=0)

    def _generate_patches(
        self,
        image: np.ndarray,
        tile_size: int = 224,
        stride: int = 112
    ) -> List[Tuple[np.ndarray, int, int]]:
        """Génère patches sliding window"""
        h, w = image.shape[:2]
        patches = []

        # Pad if needed
        pad_h = max(0, tile_size - h)
        pad_w = max(0, tile_size - w)
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
            h, w = image.shape[:2]

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                patch = image[y:y+tile_size, x:x+tile_size]
                patches.append((patch, x, y))

        return patches

    def run(
        self,
        image_path: str,
        tile_size: int = 224,
        stride: int = 112,
        batch_size: int = 16
    ) -> ImageDiagnosis:
        """
        Exécute le pipeline complet sur une image.

        Returns:
            ImageDiagnosis avec tous les résultats
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate patches
        patches = self._generate_patches(image, tile_size, stride)
        patch_images = [p[0] for p in patches]

        # Extract features
        features = self._extract_features(patch_images, batch_size)

        # Step 1: Cell Triage
        triage_preds, triage_probs = self.cell_triage.predict(features, threshold=self.triage_threshold)
        cell_mask = triage_preds == 1  # Has cells

        # Step 2: Bethesda classification (only for patches with cells)
        patch_results = []
        class_distribution = {name: 0 for name in BETHESDA_CLASSES.values()}

        abnormal_count = 0
        high_grade_count = 0

        for idx, ((patch, x, y), has_cells, cell_prob) in enumerate(zip(patches, triage_preds, triage_probs[:, 1])):
            has_cells_bool = has_cells.item() == 1
            cell_conf = cell_prob.item()

            if has_cells_bool:
                # Classify this patch
                patch_features = features[idx:idx+1]
                beth_preds = self.bethesda.predict(
                    patch_features,
                    binary_threshold=self.binary_threshold,
                    severity_threshold=self.severity_threshold
                )

                pred_class = beth_preds['finegrained'][0].item()
                class_name = BETHESDA_CLASSES[pred_class]
                is_abnormal = beth_preds['binary'][0].item() == 1
                is_high_grade = beth_preds['severity'][0].item() == 1
                severity = SEVERITY_MAPPING[pred_class]
                confidence = beth_preds['finegrained_prob'][0, pred_class].item()

                class_distribution[class_name] += 1
                if is_abnormal:
                    abnormal_count += 1
                if is_high_grade:
                    high_grade_count += 1
            else:
                # No cells - mark as empty
                pred_class = -1
                class_name = "EMPTY"
                is_abnormal = False
                is_high_grade = False
                severity = "Empty"
                confidence = 0.0

            patch_results.append(PatchResult(
                patch_idx=idx,
                x=x,
                y=y,
                has_cells=has_cells_bool,
                cell_confidence=cell_conf,
                predicted_class=pred_class,
                class_name=class_name,
                is_abnormal=is_abnormal,
                is_high_grade=is_high_grade,
                severity=severity,
                confidence=confidence
            ))

        # Aggregate diagnosis
        patches_with_cells = cell_mask.sum().item()

        if abnormal_count == 0:
            binary_result = "NORMAL"
            severity_result = "Normal"
            recommendation = "Routine screening in 3 years"
        elif high_grade_count > 0:
            binary_result = "ABNORMAL"
            severity_result = "High-grade"
            recommendation = "URGENT: Colposcopy recommended"
        else:
            binary_result = "ABNORMAL"
            severity_result = "Low-grade"
            recommendation = "Follow-up in 6-12 months"

        return ImageDiagnosis(
            image_path=image_path,
            total_patches=len(patches),
            patches_with_cells=patches_with_cells,
            abnormal_patches=abnormal_count,
            high_grade_patches=high_grade_count,
            binary_result=binary_result,
            severity_result=severity_result,
            recommendation=recommendation,
            class_distribution=class_distribution,
            patch_results=patch_results
        )


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Cytology Inference Pipeline")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--input_dir", type=str, help="Path to directory of images")
    parser.add_argument("--cell_triage", type=str, default="models/cytology/cell_triage.pt",
                        help="Path to Cell Triage model")
    parser.add_argument("--bethesda", type=str, default="models/cytology/multihead_bethesda_combined.pt",
                        help="Path to MultiHead Bethesda model")
    parser.add_argument("--output", type=str, default="results/unified_inference",
                        help="Output directory")
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--triage_threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  UNIFIED CYTOLOGY INFERENCE PIPELINE")
    print("  Cell Triage + MultiHead Bethesda")
    print("=" * 80)

    # Validate inputs
    if args.image is None and args.input_dir is None:
        print("  [ERROR] Must specify --image or --input_dir")
        return 1

    # Get images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        input_dir = Path(args.input_dir)
        image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    print(f"  [INFO] Found {len(image_paths)} images")

    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    print("\n" + "=" * 80)
    print("  LOADING MODELS")
    print("=" * 80)

    try:
        pipeline = UnifiedInferencePipeline(
            cell_triage_path=args.cell_triage,
            bethesda_path=args.bethesda,
            device=args.device,
            triage_threshold=args.triage_threshold
        )
    except Exception as e:
        print(f"  [ERROR] Failed to initialize pipeline: {e}")
        return 1

    # Run inference
    print("\n" + "=" * 80)
    print("  RUNNING INFERENCE")
    print("=" * 80)

    all_results = []
    for image_path in tqdm(image_paths, desc="Processing"):
        try:
            result = pipeline.run(
                str(image_path),
                tile_size=args.tile_size,
                stride=args.stride,
                batch_size=args.batch_size
            )
            all_results.append(result)

            print(f"\n  {image_path.name}:")
            print(f"    Patches: {result.total_patches} ({result.patches_with_cells} with cells)")
            print(f"    Result: {result.binary_result} - {result.severity_result}")
            print(f"    Abnormal: {result.abnormal_patches}, High-grade: {result.high_grade_patches}")
            print(f"    → {result.recommendation}")

        except Exception as e:
            print(f"  [ERROR] {image_path.name}: {e}")

    # Save results
    print("\n" + "=" * 80)
    print("  SAVING RESULTS")
    print("=" * 80)

    summary = []
    for result in all_results:
        summary.append({
            "image": result.image_path,
            "binary_result": result.binary_result,
            "severity_result": result.severity_result,
            "recommendation": result.recommendation,
            "total_patches": result.total_patches,
            "patches_with_cells": result.patches_with_cells,
            "abnormal_patches": result.abnormal_patches,
            "high_grade_patches": result.high_grade_patches,
            "class_distribution": result.class_distribution
        })

    summary_path = output_dir / "diagnosis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Summary saved to {summary_path}")

    # Print final summary
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)

    normal_count = sum(1 for r in all_results if r.binary_result == "NORMAL")
    abnormal_count = sum(1 for r in all_results if r.binary_result == "ABNORMAL")
    high_grade_count = sum(1 for r in all_results if r.severity_result == "High-grade")

    print(f"  Total images: {len(all_results)}")
    print(f"  Normal: {normal_count}")
    print(f"  Abnormal: {abnormal_count}")
    print(f"    Low-grade: {abnormal_count - high_grade_count}")
    print(f"    High-grade: {high_grade_count}")

    print("\n" + "=" * 80)
    print("  INFERENCE COMPLETED")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
