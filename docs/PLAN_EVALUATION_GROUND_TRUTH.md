# Plan : Pipeline d'Évaluation avec Ground Truth

> **Document de spécification pour l'implémentation du système d'évaluation**
>
> Ce document décrit le plan complet pour comparer les prédictions d'Optimus-Gate
> avec les annotations d'experts (Ground Truth) et calculer les métriques de fidélité clinique.

---

## 1. Objectif

Créer un pipeline automatisé qui :
1. Charge des images avec leurs annotations Ground Truth
2. Fait passer l'image dans Optimus-Gate (prédiction "aveugle")
3. Compare les prédictions aux annotations experts
4. Génère un rapport de fidélité clinique

**Exemple de sortie attendue :**
```
╔══════════════════════════════════════════════════════════════╗
║               RAPPORT DE FIDÉLITÉ CLINIQUE                   ║
╠══════════════════════════════════════════════════════════════╣
║ Dice Global: 0.9601  |  AJI: 0.8234  |  PQ: 0.7891           ║
╠══════════════════════════════════════════════════════════════╣
║ FIDÉLITÉ PAR TYPE CELLULAIRE                                 ║
║   🔴 Neoplastic  : Expert=20 → Modèle=19 → 95.0%             ║
║   🟢 Inflammatory: Expert=15 → Modèle=14 → 93.3%             ║
║   🔵 Connective  : Expert=8  → Modèle=8  → 100.0%            ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 2. Datasets de Référence

### 2.1 Hiérarchie des Datasets

| Priorité | Dataset | Usage | Téléchargement |
|----------|---------|-------|----------------|
| 🥇 **1** | **PanNuke** | Calibration native (même ontologie 5 classes) | [Warwick TIA](https://warwick.ac.uk/fac/cross_fac/tia/data/) |
| 🥈 **2** | **CoNSeP** | Validation famille Glandulaire | [Direct ZIP](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep_dataset.zip) |
| 🥉 **3** | **MoNuSAC** | SAV (types cellulaires détaillés) | [Hugging Face](https://huggingface.co/datasets/RationAI/MoNuSAC) |
| 4 | **Lizard** | Stress test (500k noyaux côlon) | [TIA Warwick](https://warwick.ac.uk/TIA) |

### 2.2 Format des Annotations

```python
# Format PanNuke (.npy)
images = np.load("images.npy")      # (N, 256, 256, 3) - RGB
masks = np.load("masks.npy")        # (N, 256, 256, 6) - 5 types + instances
# Canal 0: Neoplastic
# Canal 1: Inflammatory
# Canal 2: Connective
# Canal 3: Dead
# Canal 4: Epithelial
# Canal 5: Instance map

# Format CoNSeP/Lizard (.mat)
import scipy.io as sio
data = sio.loadmat("image.mat")
inst_map = data['inst_map']     # (H, W) - 0=fond, 1..N=instances
type_map = data['type_map']     # (H, W) - 0=fond, 1..K=classes
centroids = data['inst_centroid']  # (N, 2) - coordonnées [x, y]

# ⚠️ ATTENTION: L'indexation commence à 1, pas 0 !
# Le 0 est TOUJOURS le background
```

### 2.3 Mapping des Classes

```python
# PanNuke (5 classes) - NOTRE RÉFÉRENCE
PANNUKE_CLASSES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial"
}

# MoNuSAC (4 classes) - Nécessite mapping
MONUSAC_CLASSES = {
    0: "Background",
    1: "Epithelial",      # → 5 (Epithelial)
    2: "Lymphocyte",      # → 2 (Inflammatory)
    3: "Neutrophil",      # → 2 (Inflammatory)
    4: "Macrophage"       # → 2 (Inflammatory)
}

# Mapping MoNuSAC → PanNuke
MONUSAC_TO_PANNUKE = {
    1: 5,  # Epithelial → Epithelial
    2: 2,  # Lymphocyte → Inflammatory
    3: 2,  # Neutrophil → Inflammatory
    4: 2   # Macrophage → Inflammatory
}
```

---

## 3. Métriques d'Évaluation

### 3.1 Métriques Globales

| Métrique | Formule | Ce qu'elle mesure |
|----------|---------|-------------------|
| **Dice** | 2×\|P∩GT\| / (\|P\|+\|GT\|) | Chevauchement binaire |
| **AJI** | Σ IoU_matched / (TP + FP + FN) | Qualité des instances |
| **PQ** | DQ × SQ | Panoptic Quality |

### 3.2 Panoptic Quality (PQ)

```
PQ = DQ × SQ

Où:
- DQ (Detection Quality) = TP / (TP + 0.5×FP + 0.5×FN)
- SQ (Segmentation Quality) = moyenne(IoU des paires matchées)

Match valide si IoU > 0.5
```

### 3.3 F1 par Classe (F1d) — PLUS CLINIQUEMENT PERTINENT

> **Point de vigilance** : Le PQ peut être biaisé pour les petits objets.
> Le F1d par classe est plus parlant pour un pathologiste car il montre
> si le modèle confond un lymphocyte avec une cellule tumorale.

```python
# Pour chaque classe c:
TP_c = instances correctement détectées ET classifiées
FP_c = instances prédites comme c mais incorrectes
FN_c = instances de classe c non détectées

Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
```

### 3.4 Matrice de Confusion

```
                    PRÉDIT
              Neo  Inf  Con  Dead  Epi
        Neo   18    1    0    0     1    ← 18/20 = 90% correct
GT      Inf    0   14    1    0     0
        Con    0    0    8    0     0
        Dead   0    0    0    5     0
        Epi    2    0    0    0    10
              ↑
         2 Epithelial classées comme Neoplastic = ERREUR GRAVE
```

---

## 4. Pipeline d'Évaluation

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 PIPELINE ÉVALUATION GROUND TRUTH                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ÉTAPE 1: Charger GT                                           │
│  ┌─────────────────┐                                           │
│  │ .mat/.npy file  │ → inst_map, type_map, centroids           │
│  └─────────────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ÉTAPE 2: Prédiction "Aveugle"                                 │
│  ┌─────────────────┐                                           │
│  │  OPTIMUS-GATE   │ → pred_inst, pred_type, pred_centroids    │
│  └─────────────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ÉTAPE 3: Matching IoU > 0.5                                   │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Pour chaque noyau prédit:                           │       │
│  │   - Calculer IoU avec tous les GT                   │       │
│  │   - Match si IoU > 0.5 (algorithme Hongrois)        │       │
│  │   - Classer en TP, FP, FN                           │       │
│  └─────────────────────────────────────────────────────┘       │
│           │                                                     │
│           ▼                                                     │
│  ÉTAPE 4: Calcul Métriques                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ • Dice global                                       │       │
│  │ • AJI (instance quality)                            │       │
│  │ • PQ = DQ × SQ                                      │       │
│  │ • F1 par type cellulaire                            │       │
│  │ • Confusion Matrix                                  │       │
│  └─────────────────────────────────────────────────────┘       │
│           │                                                     │
│           ▼                                                     │
│  ÉTAPE 5: Rapport de Fidélité Clinique                         │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ "Expert: 20 néoplasiques → Modèle: 19 → 95% fidélité"│       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Algorithme de Matching (Détail)

```python
def match_instances(pred_inst, gt_inst, iou_threshold=0.5):
    """
    Matching optimal avec algorithme Hongrois.

    1. Construire matrice IoU (M_gt × N_pred)
    2. Résoudre l'assignation optimale
    3. Filtrer les matches avec IoU < seuil
    """
    # Construire matrice IoU
    iou_matrix = np.zeros((n_gt, n_pred))
    for i, gt_id in enumerate(gt_ids):
        for j, pred_id in enumerate(pred_ids):
            iou_matrix[i, j] = compute_iou(gt_inst == gt_id, pred_inst == pred_id)

    # Algorithme Hongrois (scipy.optimize.linear_sum_assignment)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximiser = minimiser le négatif

    # Filtrer par seuil
    matches = []
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            matches.append((gt_ids[i], pred_ids[j], iou_matrix[i, j]))

    return matches, unmatched_gt, unmatched_pred
```

---

## 5. Fichiers à Créer

### 5.1 Structure

```
scripts/
├── evaluation/
│   ├── download_evaluation_datasets.py   # Télécharge PanNuke, CoNSeP, MoNuSAC
│   ├── evaluate_ground_truth.py          # Pipeline principal
│   ├── convert_annotations.py            # Convertit .mat → format unifié
│   └── generate_clinical_report.py       # Génère rapport PDF/HTML
│
src/
├── metrics/
│   └── ground_truth_metrics.py           # ✅ DÉJÀ CRÉÉ
│       ├── compute_dice()
│       ├── compute_aji()
│       ├── compute_panoptic_quality()
│       ├── compute_f1_per_class()
│       ├── match_instances()
│       └── evaluate_predictions()
```

### 5.2 Script Principal: `evaluate_ground_truth.py`

```python
#!/usr/bin/env python3
"""
Évaluation des prédictions Optimus-Gate contre Ground Truth.

Usage:
    python scripts/evaluation/evaluate_ground_truth.py \
        --dataset pannuke \
        --fold 2 \
        --output_dir results/evaluation
"""

def main():
    # 1. Charger le dataset GT
    images, gt_inst, gt_type = load_ground_truth(args.dataset, args.fold)

    # 2. Charger Optimus-Gate
    model = OptimusGateInference.from_pretrained()

    # 3. Prédictions
    predictions = []
    for img in tqdm(images, desc="Prédiction aveugle"):
        result = model.predict(img)
        predictions.append((result['instance_map'], result['type_map']))

    # 4. Évaluation
    result = evaluate_batch(predictions, ground_truths)

    # 5. Rapport
    print(result.format_clinical_report())
    save_report(result, args.output_dir)
```

### 5.3 Script de Téléchargement: `download_evaluation_datasets.py`

```python
#!/usr/bin/env python3
"""
Télécharge les datasets d'évaluation.

Usage:
    python scripts/evaluation/download_evaluation_datasets.py --dataset all
    python scripts/evaluation/download_evaluation_datasets.py --dataset consep
"""

DATASETS = {
    "pannuke": {
        "url": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/",
        "format": "npy",
        "classes": 5
    },
    "consep": {
        "url": "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep_dataset.zip",
        "format": "mat",
        "classes": 7  # Nécessite mapping vers 5
    },
    "monusac": {
        "url": "https://huggingface.co/datasets/RationAI/MoNuSAC",
        "format": "huggingface",
        "classes": 4  # Nécessite mapping vers 5
    }
}
```

---

## 6. Points de Vigilance (SAV)

### 6.1 Indexation Off-by-One

```python
# ⚠️ ATTENTION: inst_map commence à 1, pas 0
# Le 0 est TOUJOURS le background

# ❌ FAUX
for inst_id in range(inst_map.max()):  # Manque le dernier ID
    ...

# ✅ CORRECT
for inst_id in range(1, inst_map.max() + 1):  # 1 à N inclus
    ...

# ✅ ENCORE MIEUX
inst_ids = np.unique(inst_map)
inst_ids = inst_ids[inst_ids > 0]  # Exclure le background
for inst_id in inst_ids:
    ...
```

### 6.2 Cohérence des Types

```python
# ⚠️ VÉRIFIER que type_map utilise la même indexation que inst_map

# Pour chaque instance, le type est déterminé par MAJORITÉ des pixels
def get_instance_type(inst_map, type_map, inst_id):
    mask = inst_map == inst_id
    types = type_map[mask]
    types = types[types > 0]  # Exclure background
    if len(types) == 0:
        return 0  # Background
    return int(np.bincount(types).argmax())  # Mode (valeur la plus fréquente)
```

### 6.3 Seuil IoU = 0.5

```python
# Le seuil de 0.5 est la NORME dans la communauté
# Un IoU de 0.5 signifie que la prédiction recouvre AU MOINS 50% du noyau réel

# IoU = Intersection / Union
# Si IoU = 0.5:
#   Intersection = 50% de la zone totale
#   Prédiction et GT se chevauchent significativement

# Ne PAS changer ce seuil sans bonne raison
IOU_THRESHOLD = 0.5
```

### 6.4 Gestion des Classes Absentes

```python
# Certaines images n'ont pas toutes les classes
# Ex: une image de prostate peut n'avoir que Neoplastic + Epithelial

# Le mPQ (multi-class PQ) doit ignorer les classes absentes
def compute_mpq(pq_per_class):
    valid_pq = [pq for pq in pq_per_class.values() if not np.isnan(pq)]
    return np.mean(valid_pq) if valid_pq else 0.0
```

---

## 7. Commandes d'Exécution

### 7.1 Téléchargement des Données

```bash
# Télécharger tous les datasets
python scripts/evaluation/download_evaluation_datasets.py --dataset all --output_dir data/evaluation

# Télécharger seulement CoNSeP (rapide, 41 images)
python scripts/evaluation/download_evaluation_datasets.py --dataset consep
```

### 7.2 Évaluation

```bash
# Évaluer sur PanNuke Fold 2 (non utilisé pour entraînement)
python scripts/evaluation/evaluate_ground_truth.py \
    --dataset pannuke \
    --fold 2 \
    --output_dir results/pannuke_fold2

# Évaluer sur CoNSeP (validation Glandular)
python scripts/evaluation/evaluate_ground_truth.py \
    --dataset consep \
    --output_dir results/consep

# Évaluer sur image unique
python scripts/evaluation/evaluate_ground_truth.py \
    --image path/to/image.png \
    --gt_inst path/to/inst_map.npy \
    --gt_type path/to/type_map.npy
```

### 7.3 Génération de Rapport

```bash
# Générer rapport HTML
python scripts/evaluation/generate_clinical_report.py \
    --results_dir results/pannuke_fold2 \
    --format html \
    --output rapport_fidelite.html
```

---

## 8. Résultats Attendus

### 8.1 Cibles de Performance

| Métrique | Cible | Acceptable | Critique |
|----------|-------|------------|----------|
| **Dice** | > 0.95 | > 0.90 | < 0.85 |
| **AJI** | > 0.80 | > 0.70 | < 0.60 |
| **mPQ** | > 0.70 | > 0.60 | < 0.50 |
| **F1 Neoplastic** | > 0.90 | > 0.85 | < 0.80 |
| **Classification Acc** | > 0.90 | > 0.85 | < 0.80 |

### 8.2 Validation par Famille

| Famille | Dataset de validation | Cible F1 |
|---------|----------------------|----------|
| Glandular | CoNSeP (côlon) | > 0.90 |
| Digestive | Lizard subset | > 0.88 |
| Urologique | PanNuke (kidney, bladder) | > 0.85 |
| Respiratoire | PanNuke (lung) | > 0.85 |
| Épidermoïde | PanNuke (skin) | > 0.85 |

---

## 9. Références

- [HoVer-Net GitHub](https://github.com/vqdang/hover_net)
- [CoNIC Challenge 2022](https://github.com/TissueImageAnalytics/CoNIC)
- [MoNuSAC Hugging Face](https://huggingface.co/datasets/RationAI/MoNuSAC)
- [PanNuke Paper](https://arxiv.org/abs/2003.10778)
- [PQ Limitations for Nuclei (Nature)](https://www.nature.com/articles/s41598-023-35605-7)
- [PyTorch-Metrics PQ](https://lightning.ai/docs/torchmetrics/stable/detection/panoptic_quality.html)

---

## 10. Checklist d'Implémentation

- [ ] Créer `scripts/evaluation/download_evaluation_datasets.py`
- [ ] Créer `scripts/evaluation/convert_annotations.py`
- [ ] Créer `scripts/evaluation/evaluate_ground_truth.py`
- [ ] Tester sur PanNuke Fold 2
- [ ] Tester sur CoNSeP
- [ ] Générer rapport de fidélité clinique
- [ ] Documenter dans CLAUDE.md
- [ ] Intégrer dans l'IHM (onglet "Évaluation GT")
