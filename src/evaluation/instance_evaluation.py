"""
Shared Instance Segmentation Evaluation Module.

SINGLE SOURCE OF TRUTH for evaluation logic.
Both test_v13_smart_crops_aji.py and optimize_watershed_aji.py MUST use this module.

This ensures:
1. Identical inference processing (softmax, not sigmoid)
2. Identical watershed application
3. Identical AJI computation
4. No divergence between evaluation and optimization

Author: CellViT-Optimus Project
Date: 2025-12-29
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional

from src.postprocessing import hv_guided_watershed
from src.metrics.ground_truth_metrics import compute_aji


def run_inference(
    model: torch.nn.Module,
    features: torch.Tensor,
    images_rgb: Optional[torch.Tensor] = None,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model inference and extract NP/HV predictions.

    CRITICAL: Uses SOFTMAX for NP (2-channel CrossEntropyLoss output).

    Args:
        model: HoVerNetDecoder model
        features: H-optimus-0 features (1, 261, 1536)
        images_rgb: RGB images for hybrid mode (1, 3, H, W) or None
        device: Device to use

    Returns:
        Tuple of (np_pred, hv_pred):
            - np_pred: Nuclear presence probability (H, W) in [0, 1]
            - hv_pred: HV maps (2, H, W) in [-1, 1]
    """
    model.eval()

    with torch.no_grad():
        outputs = model(features, images_rgb=images_rgb)

        # Handle both dict and tuple returns
        if isinstance(outputs, dict):
            np_out = outputs['np']
            hv_out = outputs['hv']
        else:
            np_out, hv_out, _ = outputs

        # CRITICAL: Use SOFTMAX (not sigmoid!) for CrossEntropyLoss
        # Channel 0 = Background, Channel 1 = Nuclei
        np_probs = torch.softmax(np_out, dim=1).cpu().numpy()[0]  # (2, H, W)
        np_pred = np_probs[1]  # Channel 1 = Nuclei (H, W)
        hv_pred = hv_out.cpu().numpy()[0]  # (2, H, W)

    return np_pred, hv_pred


def evaluate_sample(
    np_pred: np.ndarray,
    hv_pred: np.ndarray,
    gt_inst: np.ndarray,
    np_threshold: float = 0.40,
    beta: float = 0.50,
    min_size: int = 30,
    min_distance: int = 5
) -> Dict[str, float]:
    """
    Evaluate a single sample with given watershed parameters.

    Args:
        np_pred: Nuclear presence probability (H, W)
        hv_pred: HV maps (2, H, W)
        gt_inst: Ground truth instance map (H, W)
        np_threshold: NP binarization threshold
        beta: HV magnitude exponent
        min_size: Minimum instance size in pixels
        min_distance: Minimum distance between peaks

    Returns:
        Dict with metrics: aji, n_pred, n_gt
    """
    # Apply watershed with given parameters
    pred_inst = hv_guided_watershed(
        np_pred, hv_pred,
        np_threshold=np_threshold,
        beta=beta,
        min_size=min_size,
        min_distance=min_distance
    )

    # Compute AJI
    aji = compute_aji(pred_inst, gt_inst)

    # Count instances
    n_pred = len(np.unique(pred_inst)) - 1  # Exclude background
    n_gt = len(np.unique(gt_inst)) - 1

    return {
        'aji': aji,
        'n_pred': n_pred,
        'n_gt': n_gt,
        'pred_inst': pred_inst
    }


def evaluate_batch_with_params(
    predictions: List[Tuple[np.ndarray, np.ndarray]],
    gt_instances: List[np.ndarray],
    np_threshold: float = 0.40,
    beta: float = 0.50,
    min_size: int = 30,
    min_distance: int = 5
) -> Dict[str, float]:
    """
    Evaluate a batch of predictions with given watershed parameters.

    Args:
        predictions: List of (np_pred, hv_pred) tuples
        gt_instances: List of ground truth instance maps
        np_threshold: NP binarization threshold
        beta: HV magnitude exponent
        min_size: Minimum instance size in pixels
        min_distance: Minimum distance between peaks

    Returns:
        Dict with aggregated metrics: aji_mean, aji_std, n_pred_mean, n_gt_mean, over_seg_ratio
    """
    all_aji = []
    all_n_pred = []
    all_n_gt = []

    for (np_pred, hv_pred), gt_inst in zip(predictions, gt_instances):
        result = evaluate_sample(
            np_pred, hv_pred, gt_inst,
            np_threshold=np_threshold,
            beta=beta,
            min_size=min_size,
            min_distance=min_distance
        )
        all_aji.append(result['aji'])
        all_n_pred.append(result['n_pred'])
        all_n_gt.append(result['n_gt'])

    mean_n_pred = np.mean(all_n_pred)
    mean_n_gt = np.mean(all_n_gt)

    return {
        'aji_mean': np.mean(all_aji),
        'aji_std': np.std(all_aji),
        'aji_median': np.median(all_aji),
        'n_pred_mean': mean_n_pred,
        'n_gt_mean': mean_n_gt,
        'over_seg_ratio': mean_n_pred / mean_n_gt if mean_n_gt > 0 else 0.0
    }
