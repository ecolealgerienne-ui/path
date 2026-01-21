"""
Formal Evaluation Script — V14 Cytology POC Validation

Ce script évalue formellement le modèle MLP entraîné et génère un rapport
de validation complet avec toutes les métriques Safety First.

Métriques évaluées:
1. Sensibilité Malin (>98%) — CRITIQUE
2. Cohen's Kappa (>0.80) — CRITIQUE
3. Spécificité
4. Matrice de confusion (analyse "Death Column")
5. Per-class metrics (precision, recall, F1)
6. Binary metrics (Normal vs Abnormal)

Author: V14 Cytology Branch
Date: 2026-01-21
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)
import seaborn as sns


# =============================================================================
#  CONSTANTS
# =============================================================================

SIPAKMED_CLASSES = [
    'normal_columnar',
    'normal_intermediate',
    'normal_superficiel',
    'light_dysplastic',
    'moderate_dysplastic',
    'severe_dysplastic',
    'carcinoma_in_situ'
]

# Short names for plots
SIPAKMED_SHORT_NAMES = [
    'N-Col',
    'N-Int',
    'N-Sup',
    'L-Dys',
    'M-Dys',
    'S-Dys',
    'CIS'
]

# Binary grouping for Safety First metrics
NORMAL_CLASSES = [0, 1, 2]  # normal_columnar, normal_intermediate, normal_superficiel
ABNORMAL_CLASSES = [3, 4, 5, 6]  # dysplastic + carcinoma

# Critical classes (should NEVER be classified as Normal)
CRITICAL_CLASSES = [5, 6]  # severe_dysplastic, carcinoma_in_situ

# KPI Targets
KPI_TARGETS = {
    'sensitivity_abnormal': 0.98,
    'cohen_kappa': 0.80,
    'specificity': 0.60,
    'death_column_max': 0.02,  # Max 2% Severe/Carcinoma → Normal
}


# =============================================================================
#  DATASET
# =============================================================================

class FeaturesDataset(Dataset):
    """Dataset for fused features"""

    def __init__(self, features_path: str):
        data = torch.load(features_path, weights_only=False)
        self.features = data['fused_features'].float()
        self.labels = data['labels'].long()
        self.class_names = data['class_names']
        self.filenames = data['filenames']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# =============================================================================
#  MLP MODEL (same as training)
# =============================================================================

class CytologyMLP(nn.Module):
    """MLP Classifier with BatchNorm"""

    def __init__(
        self,
        input_dim: int = 1556,
        hidden_dims: List[int] = [512, 256, 128],
        n_classes: int = 7,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# =============================================================================
#  METRICS COMPUTATION
# =============================================================================

def compute_binary_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute binary metrics (Normal vs Abnormal)

    Safety First: Sensitivity is the PRIMARY metric
    """
    # Convert to binary
    preds_binary = np.isin(preds, ABNORMAL_CLASSES).astype(int)
    labels_binary = np.isin(labels, ABNORMAL_CLASSES).astype(int)

    # Confusion matrix components
    tp = np.sum((preds_binary == 1) & (labels_binary == 1))
    fn = np.sum((preds_binary == 0) & (labels_binary == 1))
    fp = np.sum((preds_binary == 1) & (labels_binary == 0))
    tn = np.sum((preds_binary == 0) & (labels_binary == 0))

    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # F1 Score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'accuracy': accuracy,
        'f1': f1,
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
    }


def compute_death_column_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Analyze "Death Column" — Critical classes misclassified as Normal

    This is the most dangerous error: Severe Dysplastic or Carcinoma
    classified as Normal (patient sent home with cancer)
    """
    results = {}

    for class_idx in CRITICAL_CLASSES:
        class_name = SIPAKMED_CLASSES[class_idx]

        # Samples of this class
        class_mask = labels == class_idx
        n_samples = np.sum(class_mask)

        if n_samples == 0:
            continue

        # Predictions for this class
        class_preds = preds[class_mask]

        # Count misclassifications to Normal classes
        to_normal = np.sum(np.isin(class_preds, NORMAL_CLASSES))
        to_normal_pct = to_normal / n_samples

        # Count misclassifications to other abnormal (acceptable)
        to_other_abnormal = np.sum(np.isin(class_preds, ABNORMAL_CLASSES) & (class_preds != class_idx))

        # Correct classifications
        correct = np.sum(class_preds == class_idx)

        results[class_name] = {
            'total': int(n_samples),
            'correct': int(correct),
            'correct_pct': correct / n_samples,
            'to_normal': int(to_normal),
            'to_normal_pct': to_normal_pct,
            'to_other_abnormal': int(to_other_abnormal),
            'to_other_abnormal_pct': to_other_abnormal / n_samples,
            'is_safe': to_normal_pct <= KPI_TARGETS['death_column_max'],
        }

    # Overall Death Column rate
    critical_mask = np.isin(labels, CRITICAL_CLASSES)
    critical_preds = preds[critical_mask]
    total_critical = np.sum(critical_mask)
    total_to_normal = np.sum(np.isin(critical_preds, NORMAL_CLASSES))

    results['overall'] = {
        'total_critical': int(total_critical),
        'total_to_normal': int(total_to_normal),
        'death_column_rate': total_to_normal / total_critical if total_critical > 0 else 0.0,
        'is_safe': (total_to_normal / total_critical if total_critical > 0 else 0.0) <= KPI_TARGETS['death_column_max'],
    }

    return results


def compute_per_class_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute per-class precision, recall, F1"""
    results = {}

    for i, class_name in enumerate(SIPAKMED_CLASSES):
        class_mask = labels == i
        n_samples = np.sum(class_mask)

        if n_samples == 0:
            continue

        # True positives, false positives, false negatives
        tp = np.sum((preds == i) & (labels == i))
        fp = np.sum((preds == i) & (labels != i))
        fn = np.sum((preds != i) & (labels == i))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[class_name] = {
            'support': int(n_samples),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    return results


# =============================================================================
#  VISUALIZATIONS
# =============================================================================

def plot_confusion_matrix_detailed(
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: str,
    normalize: bool = True
):
    """
    Plot detailed confusion matrix with Death Column highlighted
    """
    cm = confusion_matrix(labels, preds)

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=SIPAKMED_SHORT_NAMES,
        yticklabels=SIPAKMED_SHORT_NAMES,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    # Highlight Death Column (Critical → Normal)
    # Draw red rectangles around critical errors
    for i in CRITICAL_CLASSES:
        for j in NORMAL_CLASSES:
            if cm_display[i, j] > 0.01:  # More than 1%
                rect = plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor='red',
                    linewidth=3
                )
                ax.add_patch(rect)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix — V14 Cytology POC\n(Red boxes = Critical errors: Cancer → Normal)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_binary_confusion(
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: str
):
    """Plot binary confusion matrix (Normal vs Abnormal)"""
    # Convert to binary
    preds_binary = np.isin(preds, ABNORMAL_CLASSES).astype(int)
    labels_binary = np.isin(labels, ABNORMAL_CLASSES).astype(int)

    cm = confusion_matrix(labels_binary, preds_binary)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='RdYlGn_r',
        xticklabels=['Normal', 'Abnormal'],
        yticklabels=['Normal', 'Abnormal'],
        ax=ax,
        annot_kws={'size': 16}
    )

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Binary Confusion Matrix — Safety First View', fontsize=14)

    # Add metrics as text
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    textstr = f'Sensitivity: {sensitivity:.1%}\nSpecificity: {specificity:.1%}\nFN (Missed): {fn}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_per_class_performance(
    per_class_metrics: Dict,
    output_path: str
):
    """Plot per-class recall (most important for Safety First)"""
    classes = list(per_class_metrics.keys())
    recalls = [per_class_metrics[c]['recall'] for c in classes]
    supports = [per_class_metrics[c]['support'] for c in classes]

    # Color by class type
    colors = []
    for c in classes:
        idx = SIPAKMED_CLASSES.index(c)
        if idx in CRITICAL_CLASSES:
            colors.append('#d62728')  # Red for critical
        elif idx in ABNORMAL_CLASSES:
            colors.append('#ff7f0e')  # Orange for abnormal
        else:
            colors.append('#2ca02c')  # Green for normal

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(classes))
    bars = ax.bar(x, recalls, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, support in zip(bars, supports):
        height = bar.get_height()
        ax.annotate(
            f'{height:.1%}\n(n={support})',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )

    # Add target line
    ax.axhline(y=0.98, color='red', linestyle='--', linewidth=2, label='Target (98%)')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Per-Class Recall — V14 Cytology POC\n(🔴 Critical  🟠 Abnormal  🟢 Normal)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in classes], rotation=0, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_kpi_summary(metrics: Dict, output_path: str):
    """Plot KPI summary with targets"""
    kpis = {
        'Sensitivity\n(Abnormal)': (metrics['binary']['sensitivity'], KPI_TARGETS['sensitivity_abnormal']),
        'Cohen\'s\nKappa': (metrics['cohen_kappa'], KPI_TARGETS['cohen_kappa']),
        'Specificity': (metrics['binary']['specificity'], KPI_TARGETS['specificity']),
        'Death Column\nSafety': (1 - metrics['death_column']['overall']['death_column_rate'], 1 - KPI_TARGETS['death_column_max']),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(kpis))
    width = 0.35

    values = [v[0] for v in kpis.values()]
    targets = [v[1] for v in kpis.values()]

    # Determine colors based on target achievement
    colors = ['#2ca02c' if v >= t else '#d62728' for v, t in zip(values, targets)]

    bars = ax.bar(x - width/2, values, width, label='Achieved', color=colors, edgecolor='black')
    ax.bar(x + width/2, targets, width, label='Target', color='lightgray', edgecolor='black', hatch='//')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.1%}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('KPI Summary — V14 Cytology POC Validation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(kpis.keys(), fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


# =============================================================================
#  REPORT GENERATION
# =============================================================================

def generate_validation_report(metrics: Dict, output_dir: str) -> str:
    """Generate markdown validation report"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine overall status
    sensitivity_ok = metrics['binary']['sensitivity'] >= KPI_TARGETS['sensitivity_abnormal']
    kappa_ok = metrics['cohen_kappa'] >= KPI_TARGETS['cohen_kappa']
    death_ok = metrics['death_column']['overall']['is_safe']

    overall_status = "✅ VALIDATED" if (sensitivity_ok and death_ok) else "⚠️ NEEDS REVIEW"

    report = f"""# V14 Cytology POC — Validation Report

**Generated:** {timestamp}
**Status:** {overall_status}

---

## 📊 Executive Summary

| KPI | Result | Target | Status |
|-----|--------|--------|--------|
| **Sensitivity (Abnormal)** | **{metrics['binary']['sensitivity']:.2%}** | >98% | {"✅" if sensitivity_ok else "❌"} |
| **Cohen's Kappa** | **{metrics['cohen_kappa']:.4f}** | >0.80 | {"✅" if kappa_ok else "⚠️"} |
| **Specificity** | {metrics['binary']['specificity']:.2%} | >60% | {"✅" if metrics['binary']['specificity'] >= 0.60 else "⚠️"} |
| **Death Column Rate** | {metrics['death_column']['overall']['death_column_rate']:.2%} | <2% | {"✅" if death_ok else "❌"} |

### Key Finding

> **Safety First Objective: {"ACHIEVED" if sensitivity_ok else "NOT ACHIEVED"}**
>
> With {metrics['binary']['sensitivity']:.2%} sensitivity, only **{metrics['binary']['fn']} out of {metrics['binary']['tp'] + metrics['binary']['fn']} abnormal cells**
> were missed (classified as Normal).

---

## 🎯 Binary Classification (Normal vs Abnormal)

```
                    Predicted
                Normal    Abnormal
Actual  Normal    {metrics['binary']['tn']:4d}      {metrics['binary']['fp']:4d}
      Abnormal    {metrics['binary']['fn']:4d}      {metrics['binary']['tp']:4d}
```

| Metric | Value |
|--------|-------|
| Sensitivity (Recall) | {metrics['binary']['sensitivity']:.4f} |
| Specificity | {metrics['binary']['specificity']:.4f} |
| Precision | {metrics['binary']['precision']:.4f} |
| NPV | {metrics['binary']['npv']:.4f} |
| F1 Score | {metrics['binary']['f1']:.4f} |
| Accuracy | {metrics['binary']['accuracy']:.4f} |

---

## ⚠️ Death Column Analysis (Critical Errors)

**Definition:** Severe Dysplastic or Carcinoma classified as Normal
**Target:** < 2% of critical cases

"""

    for class_name in ['severe_dysplastic', 'carcinoma_in_situ']:
        if class_name in metrics['death_column']:
            dc = metrics['death_column'][class_name]
            report += f"""
### {class_name.replace('_', ' ').title()}

- Total samples: {dc['total']}
- Correctly classified: {dc['correct']} ({dc['correct_pct']:.1%})
- **Misclassified as Normal: {dc['to_normal']} ({dc['to_normal_pct']:.1%})** {"✅" if dc['is_safe'] else "❌ CRITICAL"}
- Misclassified as other abnormal: {dc['to_other_abnormal']} ({dc['to_other_abnormal_pct']:.1%}) *(acceptable)*
"""

    report += f"""
### Overall Death Column

- Total critical samples: {metrics['death_column']['overall']['total_critical']}
- Total misclassified as Normal: {metrics['death_column']['overall']['total_to_normal']}
- **Death Column Rate: {metrics['death_column']['overall']['death_column_rate']:.2%}** {"✅ SAFE" if death_ok else "❌ UNSAFE"}

---

## 📈 Per-Class Performance

| Class | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
"""

    for class_name, m in metrics['per_class'].items():
        is_critical = SIPAKMED_CLASSES.index(class_name) in CRITICAL_CLASSES
        marker = "🔴" if is_critical else ""
        report += f"| {marker} {class_name} | {m['support']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |\n"

    report += f"""
---

## 📋 Multi-class Metrics

| Metric | Value |
|--------|-------|
| **Cohen's Kappa** | {metrics['cohen_kappa']:.4f} |
| Overall Accuracy | {metrics['accuracy']:.4f} |
| Macro F1 | {metrics['macro_f1']:.4f} |
| Weighted F1 | {metrics['weighted_f1']:.4f} |

### Interpretation

- **Cohen's Kappa {metrics['cohen_kappa']:.2f}:** {"Substantial agreement (0.61-0.80)" if 0.61 <= metrics['cohen_kappa'] < 0.80 else "Almost perfect agreement (>0.80)" if metrics['cohen_kappa'] >= 0.80 else "Moderate agreement (0.41-0.60)"}
- Kappa measures agreement beyond chance, accounting for class imbalance

---

## 🔬 Recommendations

"""

    if sensitivity_ok and death_ok:
        report += """
### ✅ POC Validated for Safety First

The model achieves the primary objective of >98% sensitivity for abnormal cell detection.
The Death Column rate is within acceptable limits.

**Next Steps:**
1. Proceed to Phase 2 (Production) with real slides
2. Test CellPose integration for multi-cell patches
3. Validate on external datasets (Herlev, other sources)
"""
    else:
        report += """
### ⚠️ Additional Work Required

"""
        if not sensitivity_ok:
            report += "- **Sensitivity below target:** Consider adjusting class weights or threshold\n"
        if not death_ok:
            report += "- **Death Column rate too high:** Increase penalty for Severe/Carcinoma → Normal errors\n"
        if not kappa_ok:
            report += "- **Cohen's Kappa below target:** Consider hierarchical classification or more training data\n"

    report += f"""
---

## 📁 Output Files

- `validation_report.md` — This report
- `confusion_matrix_detailed.png` — 7-class confusion matrix
- `confusion_matrix_binary.png` — Binary (Normal vs Abnormal) confusion matrix
- `per_class_recall.png` — Per-class recall visualization
- `kpi_summary.png` — KPI achievement summary
- `validation_metrics.json` — Raw metrics data

---

*Report generated by V14 Cytology POC Validation Script*
"""

    return report


# =============================================================================
#  MAIN EVALUATION
# =============================================================================

def evaluate_model(
    checkpoint_path: str,
    features_dir: str,
    output_dir: str,
    device: str = 'cuda'
):
    """Main evaluation function"""

    print("=" * 80)
    print("V14 CYTOLOGY POC — FORMAL VALIDATION")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Features:   {features_dir}")
    print(f"Output:     {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Load validation data
    val_path = os.path.join(features_dir, 'sipakmed_val_features.pt')
    if not os.path.exists(val_path):
        # Try alternative path
        val_path = os.path.join(features_dir, 'val_features.pt')

    print(f"\nLoading validation data from: {val_path}")
    val_dataset = FeaturesDataset(val_path)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"Validation samples: {len(val_dataset)}")

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model_config = checkpoint.get('model_config', {})
    model = CytologyMLP(
        input_dim=model_config.get('input_dim', 1556),
        hidden_dims=model_config.get('hidden_dims', [512, 256, 128]),
        n_classes=model_config.get('n_classes', 7),
        dropout=model_config.get('dropout', 0.3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")

    # Run inference
    print("\nRunning inference...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute all metrics
    print("\nComputing metrics...")

    # Binary metrics
    binary_metrics = compute_binary_metrics(all_preds, all_labels)

    # Cohen's Kappa
    cohen_kappa = cohen_kappa_score(all_labels, all_preds)
    cohen_kappa_weighted = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    # Per-class metrics
    per_class_metrics = compute_per_class_metrics(all_preds, all_labels)

    # Death Column analysis
    death_column_metrics = compute_death_column_metrics(all_preds, all_labels)

    # Overall accuracy and F1
    accuracy = np.mean(all_preds == all_labels)

    # Macro and weighted F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    # Compile all metrics
    metrics = {
        'binary': binary_metrics,
        'cohen_kappa': cohen_kappa,
        'cohen_kappa_weighted': cohen_kappa_weighted,
        'per_class': per_class_metrics,
        'death_column': death_column_metrics,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'n_samples': len(all_labels),
        'checkpoint_epoch': checkpoint.get('epoch', None),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    print(f"\n📊 Binary Classification (Normal vs Abnormal):")
    print(f"   Sensitivity:  {binary_metrics['sensitivity']:.4f} {'✅' if binary_metrics['sensitivity'] >= 0.98 else '⚠️'} (target: >0.98)")
    print(f"   Specificity:  {binary_metrics['specificity']:.4f}")
    print(f"   Precision:    {binary_metrics['precision']:.4f}")
    print(f"   F1 Score:     {binary_metrics['f1']:.4f}")

    print(f"\n📈 Multi-class Metrics:")
    print(f"   Cohen's Kappa: {cohen_kappa:.4f} {'✅' if cohen_kappa >= 0.80 else '⚠️'} (target: >0.80)")
    print(f"   Accuracy:      {accuracy:.4f}")
    print(f"   Macro F1:      {macro_f1:.4f}")

    print(f"\n⚠️ Death Column Analysis (Critical → Normal):")
    for class_name in ['severe_dysplastic', 'carcinoma_in_situ']:
        if class_name in death_column_metrics:
            dc = death_column_metrics[class_name]
            status = "✅ SAFE" if dc['is_safe'] else "❌ CRITICAL"
            print(f"   {class_name}: {dc['to_normal']}/{dc['total']} ({dc['to_normal_pct']:.1%}) {status}")

    overall_dc = death_column_metrics['overall']
    print(f"   OVERALL: {overall_dc['total_to_normal']}/{overall_dc['total_critical']} ({overall_dc['death_column_rate']:.1%})")

    # Generate visualizations
    print("\n📊 Generating visualizations...")

    plot_confusion_matrix_detailed(
        all_labels, all_preds,
        os.path.join(output_dir, 'confusion_matrix_detailed.png')
    )

    plot_binary_confusion(
        all_labels, all_preds,
        os.path.join(output_dir, 'confusion_matrix_binary.png')
    )

    plot_per_class_performance(
        per_class_metrics,
        os.path.join(output_dir, 'per_class_recall.png')
    )

    plot_kpi_summary(
        metrics,
        os.path.join(output_dir, 'kpi_summary.png')
    )

    # Generate report
    print("\n📝 Generating validation report...")
    report = generate_validation_report(metrics, output_dir)

    report_path = os.path.join(output_dir, 'validation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # Save raw metrics
    metrics_path = os.path.join(output_dir, 'validation_metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        json.dump(convert(metrics), f, indent=2)
    print(f"  Saved: {metrics_path}")

    # Final verdict
    print("\n" + "=" * 80)
    sensitivity_ok = binary_metrics['sensitivity'] >= KPI_TARGETS['sensitivity_abnormal']
    death_ok = death_column_metrics['overall']['is_safe']

    if sensitivity_ok and death_ok:
        print("✅ POC VALIDATION: PASSED (Safety First Objectives Met)")
    else:
        print("⚠️ POC VALIDATION: NEEDS REVIEW")
        if not sensitivity_ok:
            print(f"   - Sensitivity {binary_metrics['sensitivity']:.2%} < 98% target")
        if not death_ok:
            print(f"   - Death Column rate {death_column_metrics['overall']['death_column_rate']:.2%} > 2% target")

    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")

    return metrics


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V14 Cytology POC Formal Validation"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/checkpoints_v14_cytology/best_model.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='data/features/sipakmed',
        help='Directory containing validation features'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports/v14_cytology_validation',
        help='Output directory for validation report'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    evaluate_model(
        checkpoint_path=args.checkpoint,
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
