"""
Ground Truth Metrics for Nuclei Segmentation Evaluation.

Implements standard metrics for comparing predictions against expert annotations:
- Dice Score (binary overlap)
- AJI (Aggregated Jaccard Index) - instance quality
- PQ (Panoptic Quality) = DQ × SQ
- F1d (Detection F1 per class) - clinical fidelity
- Confusion Matrix for classification analysis

Reference: HoVer-Net (Graham et al., 2019), CoNIC Challenge 2022
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


# PanNuke class mapping (0 = background)
PANNUKE_CLASSES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial"
}

# MoNuSAC class mapping
MONUSAC_CLASSES = {
    0: "Background",
    1: "Epithelial",
    2: "Lymphocyte",
    3: "Neutrophil",
    4: "Macrophage"
}


@dataclass
class InstanceMatch:
    """Represents a matched pair of GT and predicted instances."""
    gt_id: int
    pred_id: int
    iou: float
    gt_type: int
    pred_type: int


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    # Global metrics
    dice: float = 0.0
    aji: float = 0.0

    # Panoptic Quality
    pq: float = 0.0
    dq: float = 0.0  # Detection Quality
    sq: float = 0.0  # Segmentation Quality

    # Per-class metrics
    pq_per_class: Dict[int, float] = field(default_factory=dict)
    f1_per_class: Dict[int, float] = field(default_factory=dict)
    precision_per_class: Dict[int, float] = field(default_factory=dict)
    recall_per_class: Dict[int, float] = field(default_factory=dict)

    # Counts
    n_gt: int = 0
    n_pred: int = 0
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0

    # Classification
    confusion_matrix: np.ndarray = None
    classification_accuracy: float = 0.0

    # Per-class counts (for clinical report)
    gt_counts_per_class: Dict[int, int] = field(default_factory=dict)
    pred_counts_per_class: Dict[int, int] = field(default_factory=dict)

    def clinical_fidelity(self, class_id: int) -> float:
        """
        Calculate clinical fidelity for a specific class.

        Returns the percentage of correctly detected instances of that class.
        Example: "Expert: 20 néoplasiques → Modèle: 19 → 95% fidélité"
        """
        gt_count = self.gt_counts_per_class.get(class_id, 0)
        if gt_count == 0:
            return 1.0  # No GT instances, perfect fidelity

        # Use recall (sensitivity) as clinical fidelity
        return self.recall_per_class.get(class_id, 0.0)

    def format_clinical_report(self, class_names: Dict[int, str] = None) -> str:
        """Generate a clinical fidelity report."""
        if class_names is None:
            class_names = PANNUKE_CLASSES

        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║               RAPPORT DE FIDÉLITÉ CLINIQUE                   ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ Dice Global: {self.dice:.4f}  |  AJI: {self.aji:.4f}  |  PQ: {self.pq:.4f}   ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ DÉTECTION                                                    ║",
            f"║   TP: {self.n_tp:4d}  |  FP: {self.n_fp:4d}  |  FN: {self.n_fn:4d}              ║",
            f"║   Précision: {self.n_tp/(self.n_tp+self.n_fp) if (self.n_tp+self.n_fp)>0 else 0:.2%}  |  Rappel: {self.n_tp/(self.n_tp+self.n_fn) if (self.n_tp+self.n_fn)>0 else 0:.2%}               ║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ FIDÉLITÉ PAR TYPE CELLULAIRE                                 ║",
        ]

        for class_id in sorted(self.gt_counts_per_class.keys()):
            if class_id == 0:  # Skip background
                continue
            name = class_names.get(class_id, f"Class {class_id}")
            gt_count = self.gt_counts_per_class.get(class_id, 0)
            pred_count = self.pred_counts_per_class.get(class_id, 0)
            f1 = self.f1_per_class.get(class_id, 0.0)
            fidelity = self.clinical_fidelity(class_id)

            emoji = "🔴" if name == "Neoplastic" else "🟢" if name == "Inflammatory" else "🔵" if name == "Connective" else "🟡" if name == "Dead" else "🩵"
            lines.append(f"║   {emoji} {name:12s}: Expert={gt_count:3d} → Modèle={pred_count:3d} → {fidelity:.1%} ║")

        lines.extend([
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ CLASSIFICATION ACCURACY: {self.classification_accuracy:.2%}                        ║",
            "╚══════════════════════════════════════════════════════════════╝"
        ])

        return "\n".join(lines)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_dice(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """
    Compute Dice score between binary masks.

    Dice = 2 × |P ∩ GT| / (|P| + |GT|)
    """
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    total = pred_binary.sum() + gt_binary.sum()
    if total == 0:
        return 1.0  # Both empty
    return 2 * intersection / total


def compute_aji(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    """
    Compute Aggregated Jaccard Index (AJI).

    AJI measures instance segmentation quality by:
    1. Matching each GT instance to the best overlapping prediction
    2. Aggregating IoU scores

    Reference: Kumar et al., "A Dataset and a Technique for Generalized
    Nuclear Segmentation for Computational Pathology"

    Args:
        pred_inst: Predicted instance map (0=bg, 1..N=instances)
        gt_inst: Ground truth instance map (0=bg, 1..M=instances)

    Returns:
        AJI score in [0, 1]
    """
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]  # Remove background (0)

    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0  # Both empty
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0  # One empty

    # Track which predictions have been matched
    used_pred = set()

    total_intersection = 0.0
    total_union = 0.0

    for gt_id in gt_ids:
        gt_mask = gt_inst == gt_id

        best_iou = 0.0
        best_pred_id = None
        best_intersection = 0
        best_union = 0

        for pred_id in pred_ids:
            if pred_id in used_pred:
                continue

            pred_mask = pred_inst == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()

            if intersection > 0:
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou = intersection / union

                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_id
                    best_intersection = intersection
                    best_union = union

        if best_pred_id is not None:
            used_pred.add(best_pred_id)
            total_intersection += best_intersection
            total_union += best_union
        else:
            # Unmatched GT - add its area to union
            total_union += gt_mask.sum()

    # Add unmatched predictions to union
    for pred_id in pred_ids:
        if pred_id not in used_pred:
            total_union += (pred_inst == pred_id).sum()

    if total_union == 0:
        return 0.0

    return total_intersection / total_union


def match_instances(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
    pred_type: np.ndarray = None,
    gt_type: np.ndarray = None,
    iou_threshold: float = 0.5
) -> Tuple[List[InstanceMatch], List[int], List[int]]:
    """
    Match predicted instances to ground truth using IoU > threshold.

    Uses Hungarian algorithm for optimal assignment.

    Args:
        pred_inst: Predicted instance map
        gt_inst: Ground truth instance map
        pred_type: Predicted type map (optional)
        gt_type: Ground truth type map (optional)
        iou_threshold: Minimum IoU for a valid match (default 0.5)

    Returns:
        Tuple of (matched_pairs, unmatched_gt_ids, unmatched_pred_ids)
    """
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return [], list(gt_ids), list(pred_ids)

    # Build IoU matrix
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))

    for i, gt_id in enumerate(gt_ids):
        gt_mask = gt_inst == gt_id
        for j, pred_id in enumerate(pred_ids):
            pred_mask = pred_inst == pred_id
            iou_matrix[i, j] = compute_iou(gt_mask, pred_mask)

    # Hungarian algorithm (maximize IoU = minimize negative IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches = []
    matched_gt = set()
    matched_pred = set()

    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            gt_id = gt_ids[i]
            pred_id = pred_ids[j]

            # Get types if available
            gt_t = 0
            pred_t = 0
            if gt_type is not None:
                gt_mask = gt_inst == gt_id
                types_in_mask = gt_type[gt_mask]
                types_valid = types_in_mask[types_in_mask > 0]
                if len(types_valid) > 0:
                    gt_t = int(np.bincount(types_valid).argmax())

            if pred_type is not None:
                pred_mask = pred_inst == pred_id
                types_in_mask = pred_type[pred_mask]
                types_valid = types_in_mask[types_in_mask >= 0]
                if len(types_valid) > 0:
                    gt_t_counts = np.bincount(types_valid.astype(int), minlength=6)
                    if gt_t_counts[1:].sum() > 0:  # Exclude background
                        pred_t = int(gt_t_counts[1:].argmax() + 1)

            matches.append(InstanceMatch(
                gt_id=gt_id,
                pred_id=pred_id,
                iou=iou_matrix[i, j],
                gt_type=gt_t,
                pred_type=pred_t
            ))
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)

    unmatched_gt = [gid for gid in gt_ids if gid not in matched_gt]
    unmatched_pred = [pid for pid in pred_ids if pid not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_panoptic_quality(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
    pred_type: np.ndarray = None,
    gt_type: np.ndarray = None,
    iou_threshold: float = 0.5,
    num_classes: int = 6
) -> Tuple[float, float, float, Dict[int, float]]:
    """
    Compute Panoptic Quality (PQ) = DQ × SQ.

    - DQ (Detection Quality) = TP / (TP + 0.5×FP + 0.5×FN)
    - SQ (Segmentation Quality) = mean(IoU of matched pairs)

    Args:
        pred_inst: Predicted instance map
        gt_inst: Ground truth instance map
        pred_type: Predicted type map (optional, for mPQ)
        gt_type: Ground truth type map (optional, for mPQ)
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes (including background)

    Returns:
        Tuple of (PQ, DQ, SQ, PQ_per_class)
    """
    matches, unmatched_gt, unmatched_pred = match_instances(
        pred_inst, gt_inst, pred_type, gt_type, iou_threshold
    )

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    # Global PQ
    if tp == 0:
        pq, dq, sq = 0.0, 0.0, 0.0
    else:
        sq = sum(m.iou for m in matches) / tp
        dq = tp / (tp + 0.5 * fp + 0.5 * fn)
        pq = dq * sq

    # Per-class PQ (mPQ)
    pq_per_class = {}

    if pred_type is not None and gt_type is not None:
        for class_id in range(1, num_classes):  # Skip background
            # Get instances of this class
            class_matches = [m for m in matches if m.gt_type == class_id]
            class_fn = sum(1 for gid in unmatched_gt
                          if get_instance_type(gt_inst, gt_type, gid) == class_id)
            class_fp = sum(1 for pid in unmatched_pred
                          if get_instance_type(pred_inst, pred_type, pid) == class_id)

            class_tp = len(class_matches)

            if class_tp == 0 and class_fn == 0 and class_fp == 0:
                pq_per_class[class_id] = np.nan  # Class not present
            elif class_tp == 0:
                pq_per_class[class_id] = 0.0
            else:
                class_sq = sum(m.iou for m in class_matches) / class_tp
                class_dq = class_tp / (class_tp + 0.5 * class_fp + 0.5 * class_fn)
                pq_per_class[class_id] = class_dq * class_sq

    return pq, dq, sq, pq_per_class


def get_instance_type(inst_map: np.ndarray, type_map: np.ndarray, inst_id: int) -> int:
    """Get the majority type for an instance."""
    mask = inst_map == inst_id
    types = type_map[mask]
    types = types[types > 0]  # Exclude background
    if len(types) == 0:
        return 0
    return int(np.bincount(types).argmax())


def compute_f1_per_class(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
    pred_type: np.ndarray,
    gt_type: np.ndarray,
    iou_threshold: float = 0.5,
    num_classes: int = 6
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], np.ndarray]:
    """
    Compute F1 score per class (F1d - Detection F1).

    This is more clinically relevant than global PQ as it shows
    if the model confuses specific cell types.

    Returns:
        Tuple of (F1_per_class, Precision_per_class, Recall_per_class, ConfusionMatrix)
    """
    matches, unmatched_gt, unmatched_pred = match_instances(
        pred_inst, gt_inst, pred_type, gt_type, iou_threshold
    )

    # Build confusion matrix (rows=GT, cols=Pred)
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    # Matched instances
    for m in matches:
        confusion[m.gt_type, m.pred_type] += 1

    # Unmatched GT (False Negatives) - count as GT type, pred=0 (missed)
    for gt_id in unmatched_gt:
        gt_t = get_instance_type(gt_inst, gt_type, gt_id)
        confusion[gt_t, 0] += 1  # Missed detection

    # Unmatched Pred (False Positives) - count as pred type, gt=0 (spurious)
    for pred_id in unmatched_pred:
        pred_t = get_instance_type(pred_inst, pred_type, pred_id)
        confusion[0, pred_t] += 1  # False detection

    # Compute per-class metrics
    f1_per_class = {}
    precision_per_class = {}
    recall_per_class = {}

    for c in range(1, num_classes):  # Skip background
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp  # All predicted as c, except true c
        fn = confusion[c, :].sum() - tp  # All true c, except predicted as c

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_per_class[c] = f1
        precision_per_class[c] = precision
        recall_per_class[c] = recall

    return f1_per_class, precision_per_class, recall_per_class, confusion


def evaluate_predictions(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
    pred_type: np.ndarray = None,
    gt_type: np.ndarray = None,
    iou_threshold: float = 0.5,
    num_classes: int = 6
) -> EvaluationResult:
    """
    Complete evaluation of predictions against ground truth.

    Args:
        pred_inst: Predicted instance segmentation map (H, W)
                   Values: 0=background, 1..N=instance IDs
        gt_inst: Ground truth instance map (H, W)
                 Values: 0=background, 1..M=instance IDs
        pred_type: Predicted type map (H, W), optional
                   Values: 0=background, 1..K=class IDs
        gt_type: Ground truth type map (H, W), optional
        iou_threshold: Minimum IoU for matching (default 0.5)
        num_classes: Number of classes including background

    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult()

    # Binary masks for Dice
    pred_binary = pred_inst > 0
    gt_binary = gt_inst > 0

    # Dice score
    result.dice = compute_dice(pred_binary, gt_binary)

    # AJI
    result.aji = compute_aji(pred_inst, gt_inst)

    # Instance counts
    result.n_gt = len(np.unique(gt_inst)) - 1  # Exclude background
    result.n_pred = len(np.unique(pred_inst)) - 1

    # Matching
    matches, unmatched_gt, unmatched_pred = match_instances(
        pred_inst, gt_inst, pred_type, gt_type, iou_threshold
    )

    result.n_tp = len(matches)
    result.n_fp = len(unmatched_pred)
    result.n_fn = len(unmatched_gt)

    # Panoptic Quality
    result.pq, result.dq, result.sq, result.pq_per_class = compute_panoptic_quality(
        pred_inst, gt_inst, pred_type, gt_type, iou_threshold, num_classes
    )

    # Classification metrics
    if pred_type is not None and gt_type is not None:
        f1, prec, rec, conf = compute_f1_per_class(
            pred_inst, gt_inst, pred_type, gt_type, iou_threshold, num_classes
        )
        result.f1_per_class = f1
        result.precision_per_class = prec
        result.recall_per_class = rec
        result.confusion_matrix = conf

        # Classification accuracy (on matched instances)
        correct = sum(1 for m in matches if m.gt_type == m.pred_type)
        result.classification_accuracy = correct / len(matches) if matches else 0.0

        # Per-class counts
        for c in range(1, num_classes):
            result.gt_counts_per_class[c] = sum(
                1 for gid in np.unique(gt_inst) if gid > 0
                and get_instance_type(gt_inst, gt_type, gid) == c
            )
            result.pred_counts_per_class[c] = sum(
                1 for pid in np.unique(pred_inst) if pid > 0
                and get_instance_type(pred_inst, pred_type, pid) == c
            )

    return result


def evaluate_batch(
    predictions: List[Tuple[np.ndarray, np.ndarray]],
    ground_truths: List[Tuple[np.ndarray, np.ndarray]],
    iou_threshold: float = 0.5,
    num_classes: int = 6
) -> EvaluationResult:
    """
    Evaluate a batch of predictions against ground truths.

    Aggregates metrics across all images following CoNIC convention.

    Args:
        predictions: List of (inst_map, type_map) tuples
        ground_truths: List of (inst_map, type_map) tuples
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes

    Returns:
        Aggregated EvaluationResult
    """
    # Aggregate statistics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0

    # Per-class aggregation
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    class_iou_sum = defaultdict(float)

    all_dice = []
    all_aji = []

    confusion = np.zeros((num_classes, num_classes), dtype=int)

    gt_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    for (pred_inst, pred_type), (gt_inst, gt_type) in zip(predictions, ground_truths):
        # Individual metrics
        result = evaluate_predictions(
            pred_inst, gt_inst, pred_type, gt_type, iou_threshold, num_classes
        )

        all_dice.append(result.dice)
        all_aji.append(result.aji)

        total_tp += result.n_tp
        total_fp += result.n_fp
        total_fn += result.n_fn

        if result.confusion_matrix is not None:
            confusion += result.confusion_matrix

        for c, count in result.gt_counts_per_class.items():
            gt_counts[c] += count
        for c, count in result.pred_counts_per_class.items():
            pred_counts[c] += count

        # Aggregate IoU for SQ
        matches, _, _ = match_instances(pred_inst, gt_inst, pred_type, gt_type, iou_threshold)
        total_iou_sum += sum(m.iou for m in matches)

        for m in matches:
            class_tp[m.gt_type] += 1
            class_iou_sum[m.gt_type] += m.iou

    # Compute aggregated metrics
    result = EvaluationResult()
    result.dice = np.mean(all_dice) if all_dice else 0.0
    result.aji = np.mean(all_aji) if all_aji else 0.0

    result.n_tp = total_tp
    result.n_fp = total_fp
    result.n_fn = total_fn
    result.n_gt = sum(gt_counts.values())
    result.n_pred = sum(pred_counts.values())

    # Global PQ
    if total_tp > 0:
        result.sq = total_iou_sum / total_tp
        result.dq = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
        result.pq = result.dq * result.sq

    # Per-class PQ and F1
    for c in range(1, num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp

        if tp + fp + fn > 0:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            result.precision_per_class[c] = prec
            result.recall_per_class[c] = rec
            result.f1_per_class[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if class_tp[c] > 0:
                sq = class_iou_sum[c] / class_tp[c]
                dq = class_tp[c] / (class_tp[c] + 0.5 * fp + 0.5 * fn)
                result.pq_per_class[c] = dq * sq
            else:
                result.pq_per_class[c] = 0.0

    result.confusion_matrix = confusion
    result.gt_counts_per_class = dict(gt_counts)
    result.pred_counts_per_class = dict(pred_counts)

    # Classification accuracy
    correct = np.trace(confusion[1:, 1:])  # Diagonal excluding background
    total = confusion[1:, 1:].sum()
    result.classification_accuracy = correct / total if total > 0 else 0.0

    return result
