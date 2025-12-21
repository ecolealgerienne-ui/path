# Ground Truth Evaluation Scripts

Scripts pour évaluer la fidélité clinique d'Optimus-Gate en comparant ses prédictions avec des annotations expertes (Ground Truth).

## 📋 Vue d'ensemble

| Script | Description | Usage |
|--------|-------------|-------|
| `download_evaluation_datasets.py` | Télécharge les datasets d'évaluation | [Docs](#1-téléchargement-des-datasets) |
| `convert_annotations.py` | Convertit les annotations au format unifié | [Docs](#2-conversion-des-annotations) |
| `evaluate_ground_truth.py` | Évalue le modèle contre le GT | [Docs](#3-évaluation) |

## 🎯 Workflow complet

```bash
# 1. Télécharger les datasets
python scripts/evaluation/download_evaluation_datasets.py --dataset consep

# 2. Convertir au format unifié (.npz)
python scripts/evaluation/convert_annotations.py \
    --dataset consep \
    --input_dir data/evaluation/consep/Test \
    --output_dir data/evaluation/consep_converted

# 3. Évaluer le modèle
python scripts/evaluation/evaluate_ground_truth.py \
    --dataset_dir data/evaluation/consep_converted \
    --output_dir results/consep \
    --dataset consep

# 4. Consulter le rapport
cat results/consep/clinical_report_consep_*.txt
```

## 1. Téléchargement des Datasets

### Afficher les datasets disponibles

```bash
python scripts/evaluation/download_evaluation_datasets.py --info
```

### Télécharger CoNSeP (rapide, 70 MB)

```bash
python scripts/evaluation/download_evaluation_datasets.py --dataset consep
```

### Télécharger PanNuke (lent, ~1.5 GB)

```bash
# Tous les folds
python scripts/evaluation/download_evaluation_datasets.py --dataset pannuke

# Seulement Fold 2 (pour validation)
python scripts/evaluation/download_evaluation_datasets.py \
    --dataset pannuke \
    --folds 2
```

## 2. Conversion des Annotations

### CoNSeP (.mat → .npz)

```bash
python scripts/evaluation/convert_annotations.py \
    --dataset consep \
    --input_dir data/evaluation/consep/Test \
    --output_dir data/evaluation/consep_converted
```

### PanNuke (.npy → .npz)

```bash
python scripts/evaluation/convert_annotations.py \
    --dataset pannuke \
    --input_dir "data/evaluation/pannuke/Fold 2" \
    --output_dir data/evaluation/pannuke_fold2_converted
```

### Vérifier une conversion

```bash
python scripts/evaluation/convert_annotations.py \
    --verify data/evaluation/consep_converted/image_001.npz
```

## 3. Évaluation

### Évaluation complète sur un dataset

```bash
python scripts/evaluation/evaluate_ground_truth.py \
    --dataset_dir data/evaluation/consep_converted \
    --output_dir results/consep \
    --dataset consep
```

### Évaluation sur un sous-ensemble

```bash
# 100 images de PanNuke
python scripts/evaluation/evaluate_ground_truth.py \
    --dataset_dir data/evaluation/pannuke_fold2_converted \
    --output_dir results/pannuke_fold2 \
    --num_samples 100
```

### Évaluation d'une seule image

```bash
python scripts/evaluation/evaluate_ground_truth.py \
    --image data/evaluation/consep_converted/test_001.npz \
    --output_dir results/single \
    --verbose
```

## 📊 Résultats

### Fichiers générés

| Fichier | Description |
|---------|-------------|
| `clinical_report_*.txt` | Rapport de fidélité clinique (format texte) |
| `metrics_*.json` | Métriques détaillées (format JSON) |
| `confusion_matrix_*.npy` | Matrice de confusion (format NumPy) |

### Exemple de rapport clinique

```
╔══════════════════════════════════════════════════════════════╗
║               RAPPORT DE FIDÉLITÉ CLINIQUE                   ║
╠══════════════════════════════════════════════════════════════╣
║ Dice Global: 0.9601  |  AJI: 0.8234  |  PQ: 0.7891           ║
╠══════════════════════════════════════════════════════════════╣
║ DÉTECTION                                                    ║
║   TP:  180  |  FP:   12  |  FN:    8                        ║
║   Précision: 93.75%  |  Rappel: 95.74%                      ║
╠══════════════════════════════════════════════════════════════╣
║ FIDÉLITÉ PAR TYPE CELLULAIRE                                 ║
║   🔴 Neoplastic  : Expert= 20 → Modèle= 19 → 95.0%           ║
║   🟢 Inflammatory: Expert= 15 → Modèle= 14 → 93.3%           ║
║   🔵 Connective  : Expert=  8 → Modèle=  8 → 100.0%          ║
╠══════════════════════════════════════════════════════════════╣
║ CLASSIFICATION ACCURACY: 91.25%                              ║
╚══════════════════════════════════════════════════════════════╝
```

## 📏 Métriques Expliquées

| Métrique | Formule | Ce qu'elle mesure | Cible |
|----------|---------|-------------------|-------|
| **Dice** | 2×\|P∩GT\| / (\|P\|+\|GT\|) | Chevauchement binaire | > 0.95 |
| **AJI** | Σ IoU_matched / (TP+FP+FN) | Qualité des instances | > 0.80 |
| **PQ** | DQ × SQ | Panoptic Quality | > 0.70 |
| **F1** | 2×Precision×Recall / (Prec+Rec) | Équilibre Prec/Rec | > 0.90 |

### Seuils de Qualité

| Niveau | Dice | AJI | PQ | Statut |
|--------|------|-----|----|----|
| **Excellent** | ≥ 0.95 | ≥ 0.80 | ≥ 0.70 | ✅ Cible |
| **Acceptable** | ≥ 0.90 | ≥ 0.70 | ≥ 0.60 | 🟡 OK |
| **Sous-optimal** | ≥ 0.85 | ≥ 0.60 | ≥ 0.50 | 🟠 Améliorer |
| **Critique** | < 0.85 | < 0.60 | < 0.50 | 🔴 Problème |

## 🎓 Références

- **PanNuke**: Gamper et al. (2019) - [Paper](https://arxiv.org/abs/2003.10778)
- **CoNSeP**: Graham et al. (2019) - [HoVer-Net](https://github.com/vqdang/hover_net)
- **AJI**: Kumar et al. (2017) - [Paper](https://ieeexplore.ieee.org/document/7872382)
- **Panoptic Quality**: Kirillov et al. (2019) - [Paper](https://arxiv.org/abs/1801.00868)

## 🐛 Dépannage

### Erreur: "No .npz files found"

Vérifiez que la conversion a été effectuée :

```bash
ls -la data/evaluation/consep_converted/*.npz
```

### Erreur: "No 'inst_map' found in .mat"

Le fichier .mat doit contenir les clés `inst_map`, `type_map`, `inst_centroid`. Vérifiez le format :

```python
import scipy.io as sio
data = sio.loadmat("file.mat")
print(data.keys())
```

### Mémoire insuffisante

Réduisez le nombre d'images évaluées :

```bash
python scripts/evaluation/evaluate_ground_truth.py \
    --dataset_dir ... \
    --num_samples 10  # Évaluer seulement 10 images
```

## 📝 Notes

- **Seuil IoU**: Par défaut 0.5 (norme communauté). Ne pas changer sans raison.
- **Indexation**: `inst_map` commence à 1, pas 0 (0 = background).
- **Classes**: PanNuke utilise 1-5 pour les types (0 = background).
- **Mapping**: CoNSeP et MoNuSAC sont automatiquement mappés vers PanNuke.
