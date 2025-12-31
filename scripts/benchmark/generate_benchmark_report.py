#!/usr/bin/env python3
"""
CellViT-Optimus Benchmark Report Generator.

Génère un rapport HTML comparant les prédictions du système avec les ground truth PanNuke.

Structure de sortie:
    benchmark/
    └── {family}/
        ├── images/
        │   ├── {Organ}_{index}_comparison.png
        │   └── ...
        ├── rapport_{family}.html
        └── metrics_{family}.csv

Usage:
    # Test avec 2 images
    python scripts/benchmark/generate_benchmark_report.py \
        --family respiratory --n_samples 2 --test

    # Production: 30 images par famille
    python scripts/benchmark/generate_benchmark_report.py \
        --family respiratory --n_samples 30

Author: CellViT-Optimus Project
Date: 2025-12-30
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
import cv2
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import csv

# === IMPORTS EXISTANTS (AUCUNE DUPLICATION) ===
from src.metrics.ground_truth_metrics import evaluate_predictions, PANNUKE_CLASSES
from src.postprocessing import hv_guided_watershed
from src.evaluation import run_inference
from src.ui.visualizations import (
    create_segmentation_overlay,
    create_contour_overlay,
    CELL_COLORS_RGB,
    TYPE_NAMES,
)
from src.ui.organ_config import FAMILY_WATERSHED_PARAMS, FAMILY_CHECKPOINTS
from src.models.hovernet_decoder import HoVerNetDecoder


# =============================================================================
# FONCTIONS DE FILTRAGE
# =============================================================================

def count_types_in_sample(nt_target: np.ndarray, inst_map: np.ndarray) -> Dict[int, int]:
    """
    Compte le nombre de noyaux par type dans un sample.

    Args:
        nt_target: Type map (H, W) avec valeurs 0-4
        inst_map: Instance map (H, W)

    Returns:
        Dict {type_id: count}
    """
    type_counts = {}
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        mask = inst_map == inst_id
        types_in_mask = nt_target[mask]
        if len(types_in_mask) > 0:
            dominant_type = int(np.bincount(types_in_mask.astype(int)).argmax())
            type_counts[dominant_type] = type_counts.get(dominant_type, 0) + 1
    return type_counts


def filter_samples_with_min_types(
    nt_targets: np.ndarray,
    inst_maps: np.ndarray,
    min_types: int = 2
) -> List[int]:
    """
    Filtre les samples ayant au moins min_types types cellulaires différents.

    Args:
        nt_targets: (N, H, W) type maps
        inst_maps: (N, H, W) instance maps
        min_types: Nombre minimum de types requis

    Returns:
        Liste des indices valides
    """
    valid_indices = []
    for i in range(len(nt_targets)):
        type_counts = count_types_in_sample(nt_targets[i], inst_maps[i])
        # Compter les types avec au moins 1 noyau
        n_types = len([t for t, c in type_counts.items() if c > 0])
        if n_types >= min_types:
            valid_indices.append(i)
    return valid_indices


# =============================================================================
# FONCTIONS DE VISUALISATION (NOUVEAU)
# =============================================================================

def create_diff_overlay(
    image: np.ndarray,
    gt_inst: np.ndarray,
    pred_inst: np.ndarray,
) -> np.ndarray:
    """
    Crée un overlay montrant les différences GT vs Pred.

    Couleurs:
        - Vert: TP (match IoU > 0.5)
        - Rouge: FN (GT manqué)
        - Jaune: FP (fausse détection)

    Args:
        image: Image RGB originale (H, W, 3)
        gt_inst: Ground truth instance map
        pred_inst: Predicted instance map

    Returns:
        Image avec diff overlay
    """
    result = image.copy()
    h, w = gt_inst.shape

    # Couleurs
    TP_COLOR = (50, 255, 50)    # Vert
    FN_COLOR = (255, 50, 50)    # Rouge
    FP_COLOR = (255, 255, 50)   # Jaune

    gt_ids = set(np.unique(gt_inst)) - {0}
    pred_ids = set(np.unique(pred_inst)) - {0}

    matched_gt = set()
    matched_pred = set()

    # Trouver les matches (IoU > 0.5)
    for gt_id in gt_ids:
        gt_mask = gt_inst == gt_id
        best_iou = 0
        best_pred_id = None

        for pred_id in pred_ids:
            if pred_id in matched_pred:
                continue
            pred_mask = pred_inst == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                iou = intersection / union
                if iou > best_iou and iou >= 0.5:
                    best_iou = iou
                    best_pred_id = pred_id

        if best_pred_id is not None:
            matched_gt.add(gt_id)
            matched_pred.add(best_pred_id)

    # Dessiner TP (vert) - contours des GT matchés
    for gt_id in matched_gt:
        mask = (gt_inst == gt_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, TP_COLOR, 2)

    # Dessiner FN (rouge) - contours des GT non matchés
    for gt_id in gt_ids - matched_gt:
        mask = (gt_inst == gt_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, FN_COLOR, 2)
        # Ajouter "FN" au centre
        coords = np.where(gt_inst == gt_id)
        if len(coords[0]) > 0:
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            cv2.putText(result, "FN", (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, FN_COLOR, 1)

    # Dessiner FP (jaune) - contours des Pred non matchés
    for pred_id in pred_ids - matched_pred:
        mask = (pred_inst == pred_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, FP_COLOR, 2)
        # Ajouter "FP" au centre
        coords = np.where(pred_inst == pred_id)
        if len(coords[0]) > 0:
            cy, cx = int(coords[0].mean()), int(coords[1].mean())
            cv2.putText(result, "FP", (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, FP_COLOR, 1)

    return result


def assemble_comparison_panels(
    original: np.ndarray,
    gt_overlay: np.ndarray,
    pred_overlay: np.ndarray,
    diff_overlay: np.ndarray,
    sample_info: Dict,
) -> np.ndarray:
    """
    Assemble les 4 panels en une seule image avec légende.

    Layout:
        ┌──────────┬──────────┬──────────┬──────────┐
        │ Original │    GT    │   Pred   │   Diff   │
        └──────────┴──────────┴──────────┴──────────┘
        │  Infos + Métriques                        │
        └───────────────────────────────────────────┘

    Args:
        original: Image H&E originale
        gt_overlay: Overlay GT avec contours colorés
        pred_overlay: Overlay Pred avec contours colorés
        diff_overlay: Overlay des différences
        sample_info: Dict avec métriques

    Returns:
        Image assemblée
    """
    h, w = original.shape[:2]
    panel_w = w
    panel_h = h

    # Hauteur pour le texte d'info
    info_h = 80

    # Taille totale: 4 panels + barre d'info
    total_w = panel_w * 4
    total_h = panel_h + info_h

    # Créer image de sortie (fond blanc)
    output = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    # Placer les 4 panels
    output[0:panel_h, 0:panel_w] = original
    output[0:panel_h, panel_w:panel_w*2] = gt_overlay
    output[0:panel_h, panel_w*2:panel_w*3] = pred_overlay
    output[0:panel_h, panel_w*3:panel_w*4] = diff_overlay

    # Ajouter les titres sur chaque panel
    titles = ["Original", "GT (PanNuke)", "Prediction", "Diff (TP/FN/FP)"]
    for i, title in enumerate(titles):
        x = i * panel_w + 5
        cv2.putText(output, title, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Zone d'info (fond gris clair)
    output[panel_h:, :] = (240, 240, 240)

    # Ligne de séparation
    cv2.line(output, (0, panel_h), (total_w, panel_h), (200, 200, 200), 1)

    # Infos textuelles
    y_base = panel_h + 20

    # Ligne 1: Sample info
    info_line1 = f"{sample_info['organ']} #{sample_info['index']} | GT: {sample_info['n_gt']} nuclei | Pred: {sample_info['n_pred']} nuclei"
    cv2.putText(output, info_line1, (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Ligne 2: Métriques
    info_line2 = f"AJI: {sample_info['aji']:.4f} | Dice: {sample_info['dice']:.4f} | PQ: {sample_info['pq']:.4f}"
    cv2.putText(output, info_line2, (10, y_base + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Ligne 3: Types GT
    gt_types_str = " | ".join([f"{TYPE_NAMES[t]}: {c}" for t, c in sorted(sample_info['gt_type_counts'].items())])
    cv2.putText(output, f"GT Types: {gt_types_str}", (10, y_base + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # Légende Diff (côté droit)
    legend_x = total_w - 200
    cv2.putText(output, "Legende:", (legend_x, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.circle(output, (legend_x + 10, y_base + 15), 5, (50, 255, 50), -1)
    cv2.putText(output, "TP", (legend_x + 20, y_base + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.circle(output, (legend_x + 60, y_base + 15), 5, (255, 50, 50), -1)
    cv2.putText(output, "FN", (legend_x + 70, y_base + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.circle(output, (legend_x + 110, y_base + 15), 5, (255, 255, 50), -1)
    cv2.putText(output, "FP", (legend_x + 120, y_base + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    return output


# =============================================================================
# GÉNÉRATION HTML
# =============================================================================

def generate_html_report(
    family: str,
    samples_data: List[Dict],
    output_path: Path,
    images_dir: str = "images",
) -> None:
    """
    Génère le rapport HTML.

    Args:
        family: Nom de la famille
        samples_data: Liste des données par sample
        output_path: Chemin du fichier HTML
        images_dir: Nom du dossier images (relatif)
    """
    # Calculer les métriques agrégées
    mean_aji = np.mean([s['aji'] for s in samples_data])
    mean_dice = np.mean([s['dice'] for s in samples_data])
    mean_pq = np.mean([s['pq'] for s in samples_data])

    # Compter par type (agrégé)
    total_gt_by_type = {}
    total_pred_by_type = {}
    for s in samples_data:
        for t, c in s.get('gt_type_counts', {}).items():
            total_gt_by_type[t] = total_gt_by_type.get(t, 0) + c
        for t, c in s.get('pred_type_counts', {}).items():
            total_pred_by_type[t] = total_pred_by_type.get(t, 0) + c

    # Couleurs CSS pour les types
    type_colors_css = {
        0: "#ff3232",  # Neoplastic - Rouge
        1: "#32ff32",  # Inflammatory - Vert
        2: "#3232ff",  # Connective - Bleu
        3: "#ffff32",  # Dead - Jaune
        4: "#32ffff",  # Epithelial - Cyan
    }

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report - {family.capitalize()}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a5490, #2980b9);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .header .subtitle {{
            opacity: 0.9;
            margin-top: 10px;
        }}
        .metrics-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #1a5490;
        }}
        .metric-card .label {{
            color: #666;
            margin-top: 5px;
        }}
        .type-summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .type-row {{
            display: flex;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .type-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 15px;
        }}
        .type-name {{
            flex: 1;
            font-weight: 500;
        }}
        .type-count {{
            text-align: right;
            color: #666;
        }}
        .samples-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .sample-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .sample-card img {{
            width: 100%;
            display: block;
        }}
        .sample-info {{
            padding: 15px;
            background: #f9f9f9;
        }}
        .sample-title {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        .sample-metrics {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .sample-metric {{
            background: #e8f4fc;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f0f0f0;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Benchmark Report: {family.capitalize()}</h1>
        <div class="subtitle">
            CellViT-Optimus — Comparaison GT vs Prédictions<br>
            Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(samples_data)} échantillons
        </div>
    </div>

    <div class="metrics-summary">
        <div class="metric-card">
            <div class="value">{mean_aji:.4f}</div>
            <div class="label">AJI Moyen</div>
        </div>
        <div class="metric-card">
            <div class="value">{mean_dice:.4f}</div>
            <div class="label">Dice Moyen</div>
        </div>
        <div class="metric-card">
            <div class="value">{mean_pq:.4f}</div>
            <div class="label">PQ Moyen</div>
        </div>
        <div class="metric-card">
            <div class="value">{len(samples_data)}</div>
            <div class="label">Échantillons</div>
        </div>
    </div>

    <div class="type-summary">
        <h2>📊 Distribution par Type Cellulaire</h2>
        <table>
            <tr>
                <th>Type</th>
                <th>GT (Total)</th>
                <th>Pred (Total)</th>
                <th>Δ</th>
            </tr>
"""

    for t in range(5):
        gt_count = total_gt_by_type.get(t, 0)
        pred_count = total_pred_by_type.get(t, 0)
        delta = pred_count - gt_count
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        html += f"""
            <tr>
                <td><span class="type-color" style="background: {type_colors_css[t]}; display: inline-block; vertical-align: middle;"></span> {TYPE_NAMES[t]}</td>
                <td>{gt_count}</td>
                <td>{pred_count}</td>
                <td>{delta_str}</td>
            </tr>
"""

    html += """
        </table>
    </div>

    <div class="samples-section">
        <h2>🖼️ Détail par Échantillon</h2>
"""

    # Trier par organe puis par index
    samples_sorted = sorted(samples_data, key=lambda x: (x['organ'], x['index']))

    for s in samples_sorted:
        html += f"""
        <div class="sample-card">
            <img src="{images_dir}/{s['image_filename']}" alt="{s['organ']} #{s['index']}">
            <div class="sample-info">
                <div class="sample-title">{s['organ']} #{s['index']}</div>
                <div class="sample-metrics">
                    <span class="sample-metric">AJI: {s['aji']:.4f}</span>
                    <span class="sample-metric">Dice: {s['dice']:.4f}</span>
                    <span class="sample-metric">PQ: {s['pq']:.4f}</span>
                    <span class="sample-metric">GT: {s['n_gt']} nuclei</span>
                    <span class="sample-metric">Pred: {s['n_pred']} nuclei</span>
                </div>
            </div>
        </div>
"""

    html += f"""
    </div>

    <div class="footer">
        <p>CellViT-Optimus Benchmark Report v1.0</p>
        <p>⚠️ Document technique — Ne constitue pas un diagnostic médical</p>
    </div>
</body>
</html>
"""

    output_path.write_text(html, encoding='utf-8')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark report comparing predictions vs GT"
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=["glandular", "digestive", "urologic", "epidermal", "respiratory"],
        help="Family to benchmark"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=30,
        help="Number of samples per family (default: 30)"
    )
    parser.add_argument(
        "--min_types",
        type=int,
        default=2,
        help="Minimum number of cell types in GT (default: 2)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("benchmark"),
        help="Output directory (default: benchmark/)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: verbose output"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    family = args.family

    print("=" * 80)
    print("CELLVIT-OPTIMUS BENCHMARK REPORT GENERATOR")
    print("=" * 80)
    print(f"\nFamily: {family}")
    print(f"N samples: {args.n_samples}")
    print(f"Min types in GT: {args.min_types}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir / family}/")

    # ==========================================================================
    # 1. CHARGER LES DONNÉES VAL
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. LOADING VALIDATION DATA")
    print("=" * 80)

    val_data_path = Path(f"data/family_data_v13_smart_crops/{family}_val_v13_smart_crops.npz")
    features_path = Path(f"data/cache/family_data/{family}_rgb_features_v13_smart_crops_val.npz")

    if not val_data_path.exists():
        print(f"❌ ERROR: {val_data_path} not found")
        return 1
    if not features_path.exists():
        print(f"❌ ERROR: {features_path} not found")
        return 1

    print(f"Loading {val_data_path.name}...")
    val_data = np.load(val_data_path)

    images = val_data['images']           # (N, 224, 224, 3) uint8
    inst_maps = val_data['inst_maps']     # (N, 224, 224) int32
    nt_targets = val_data['nt_targets']   # (N, 224, 224) int64
    organ_names = val_data['organ_names'] # (N,) str

    print(f"Loading {features_path.name}...")
    features_data = np.load(features_path)
    all_features = features_data['features']  # (N, 261, 1536)

    print(f"  → {len(images)} samples loaded")
    print(f"  → Images: {images.shape}, dtype={images.dtype}")
    print(f"  → Organs: {np.unique(organ_names)}")

    # ==========================================================================
    # 2. FILTRER LES SAMPLES (≥ min_types)
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"2. FILTERING SAMPLES (≥ {args.min_types} cell types)")
    print("=" * 80)

    valid_indices = filter_samples_with_min_types(nt_targets, inst_maps, args.min_types)
    print(f"  → {len(valid_indices)} samples have ≥ {args.min_types} types")

    # Sélectionner les premiers n_samples
    selected_indices = valid_indices[:args.n_samples]
    print(f"  → Selected {len(selected_indices)} samples for benchmark")

    if len(selected_indices) == 0:
        print("❌ No valid samples found!")
        return 1

    # ==========================================================================
    # 3. CHARGER LE MODÈLE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("3. LOADING MODEL")
    print("=" * 80)

    checkpoint_path = FAMILY_CHECKPOINTS[family]
    print(f"Checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        print(f"❌ ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Détecter les flags du checkpoint
    use_hybrid = checkpoint.get('use_hybrid', False)
    use_fpn_chimique = checkpoint.get('use_fpn_chimique', False)
    use_h_alpha = any('h_alphas' in k for k in checkpoint['model_state_dict'].keys())

    print(f"  use_hybrid: {use_hybrid}")
    print(f"  use_fpn_chimique: {use_fpn_chimique}")
    print(f"  use_h_alpha: {use_h_alpha}")

    model = HoVerNetDecoder(
        embed_dim=1536,
        n_classes=5,
        dropout=0.1,
        use_hybrid=use_hybrid,
        use_fpn_chimique=use_fpn_chimique,
        use_h_alpha=use_h_alpha
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✅ Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")

    # Params watershed
    watershed_params = FAMILY_WATERSHED_PARAMS[family]
    print(f"  Watershed params: {watershed_params}")

    # ==========================================================================
    # 4. CRÉER LE DOSSIER DE SORTIE
    # ==========================================================================
    output_dir = args.output_dir / family
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output dir: {output_dir}")

    # ==========================================================================
    # 5. GÉNÉRER LES COMPARAISONS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("5. GENERATING COMPARISONS")
    print("=" * 80)

    samples_data = []

    for idx in tqdm(selected_indices, desc="Processing"):
        # Données GT
        image = images[idx]                  # (224, 224, 3) uint8
        gt_inst = inst_maps[idx]             # (224, 224) int32
        gt_type = nt_targets[idx]            # (224, 224) int64
        organ = str(organ_names[idx])

        # Features
        features = torch.from_numpy(all_features[idx]).unsqueeze(0).float().to(device)

        # Image RGB pour mode hybride
        if use_hybrid:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            image_tensor = None

        # === INFÉRENCE (utilise run_inference existant) ===
        np_pred, hv_pred = run_inference(model, features, image_tensor, device=str(device))

        # === WATERSHED (utilise hv_guided_watershed existant) ===
        pred_inst = hv_guided_watershed(np_pred, hv_pred, **watershed_params)

        # === TYPE MAP PRÉDIT ===
        # Extraire le type prédit par le modèle
        with torch.no_grad():
            outputs = model(features, images_rgb=image_tensor)
            if isinstance(outputs, dict):
                nt_out = outputs['nt']
            else:
                _, _, nt_out = outputs
            pred_type = torch.argmax(nt_out, dim=1).cpu().numpy()[0]  # (224, 224)

        # === ÉVALUATION (utilise evaluate_predictions existant) ===
        eval_result = evaluate_predictions(
            pred_inst, gt_inst,
            pred_type, gt_type.astype(np.int32),
            iou_threshold=0.5,
            num_classes=6
        )

        # === VISUALISATIONS (utilise create_*_overlay existants) ===
        gt_overlay = create_contour_overlay(image, gt_inst, gt_type, thickness=2)
        pred_overlay = create_contour_overlay(image, pred_inst, pred_type, thickness=2)
        diff_overlay = create_diff_overlay(image, gt_inst, pred_inst)

        # Compter les types GT
        gt_type_counts = count_types_in_sample(gt_type, gt_inst)
        pred_type_counts = count_types_in_sample(pred_type, pred_inst)

        # === ASSEMBLER LES PANELS ===
        sample_info = {
            'organ': organ,
            'index': idx,
            'n_gt': eval_result.n_gt,
            'n_pred': eval_result.n_pred,
            'aji': eval_result.aji,
            'dice': eval_result.dice,
            'pq': eval_result.pq,
            'gt_type_counts': gt_type_counts,
            'pred_type_counts': pred_type_counts,
        }

        comparison_img = assemble_comparison_panels(
            image, gt_overlay, pred_overlay, diff_overlay, sample_info
        )

        # === SAUVEGARDER L'IMAGE ===
        image_filename = f"{organ}_{idx:04d}_comparison.png"
        image_path = images_output_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))

        # Ajouter les données pour le rapport
        sample_info['image_filename'] = image_filename
        samples_data.append(sample_info)

        if args.test:
            print(f"  [{organ}_{idx}] AJI={eval_result.aji:.4f}, GT={eval_result.n_gt}, Pred={eval_result.n_pred}")

    # ==========================================================================
    # 6. GÉNÉRER LE RAPPORT HTML
    # ==========================================================================
    print("\n" + "=" * 80)
    print("6. GENERATING HTML REPORT")
    print("=" * 80)

    report_path = output_dir / f"rapport_{family}.html"
    generate_html_report(family, samples_data, report_path)
    print(f"  ✅ HTML report: {report_path}")

    # ==========================================================================
    # 7. GÉNÉRER LE CSV
    # ==========================================================================
    csv_path = output_dir / f"metrics_{family}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_filename', 'organ', 'index', 'n_gt', 'n_pred', 'aji', 'dice', 'pq'
        ])
        writer.writeheader()
        for s in samples_data:
            writer.writerow({k: s[k] for k in writer.fieldnames})
    print(f"  ✅ CSV metrics: {csv_path}")

    # ==========================================================================
    # RÉSUMÉ
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    mean_aji = np.mean([s['aji'] for s in samples_data])
    mean_dice = np.mean([s['dice'] for s in samples_data])
    print(f"\n  Family: {family}")
    print(f"  Samples: {len(samples_data)}")
    print(f"  Mean AJI: {mean_aji:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"\n  Output: {output_dir}/")
    print(f"    ├── images/ ({len(samples_data)} files)")
    print(f"    ├── rapport_{family}.html")
    print(f"    └── metrics_{family}.csv")

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
