# Checklist Évaluation Ground Truth avec PanNuke

## ✅ Scripts d'Évaluation (COMPLET)

- [x] `download_evaluation_datasets.py` - Téléchargement datasets
- [x] `convert_annotations.py` - Conversion .npy → .npz
- [x] `evaluate_ground_truth.py` - Évaluation modèle vs GT
- [x] `src/metrics/ground_truth_metrics.py` - Métriques (Dice, AJI, PQ, F1)

## 📦 Dataset PanNuke

### Fold 2 (Test Set - NON utilisé pour entraînement)

**Statut:** À vérifier sur votre machine

```bash
# Vérifier si PanNuke Fold 2 existe
ls -lh /home/amar/data/PanNuke/Fold\ 2/

# Si non présent, télécharger (500 MB)
cd /home/amar/projects/cellvit-optimus
python scripts/evaluation/download_evaluation_datasets.py \
    --dataset pannuke \
    --folds 2 \
    --output_dir data/evaluation
```

**Fichiers attendus:**
```
data/evaluation/pannuke/Fold 2/
├── images.npy      # (N, 256, 256, 3) RGB images
├── masks.npy       # (N, 256, 256, 6) Masks (5 classes + instances)
└── types.npy       # (N,) Organ types
```

## 🤖 Modèle Optimus-Gate

### Checkpoints Nécessaires

**Statut:** À vérifier sur votre machine

```bash
# Vérifier les checkpoints
ls -lh /home/amar/projects/cellvit-optimus/models/checkpoints/

# Fichiers requis:
# - organ_head_best.pth          (OrganHead - 99.94% accuracy)
# - hovernet_glandular_best.pth  (HoVer-Net famille glandulaire)
# - hovernet_digestive_best.pth  (HoVer-Net famille digestive)
# - hovernet_urologic_best.pth   (HoVer-Net famille urologique)
# - hovernet_epidermal_best.pth  (HoVer-Net famille épidermoïde)
# - hovernet_respiratory_best.pth (HoVer-Net famille respiratoire)
```

**Si les checkpoints manquent:** Ils doivent être entraînés selon CLAUDE.md sections:
- OrganHead: Section "2025-12-20 — Entraînement Multi-Folds (3 folds)"
- HoVer-Net: Section "2025-12-20 — Entraînement 5 Familles Complété"

## 🐍 Dépendances Python

**Statut:** À vérifier dans l'environnement conda `cellvit`

```bash
# Activer l'environnement
conda activate cellvit

# Vérifier les dépendances
python -c "
import numpy
import torch
import scipy
import cv2
import timm
import sklearn
from skimage.segmentation import watershed
print('✅ Toutes les dépendances sont installées')
"
```

**Si des dépendances manquent:**
```bash
conda activate cellvit
pip install numpy scipy opencv-python scikit-image scikit-learn timm
```

## 🔍 Workflow Complet d'Évaluation

### Étape 1: Télécharger PanNuke Fold 2 (si nécessaire)

```bash
cd /home/amar/projects/cellvit-optimus

python scripts/evaluation/download_evaluation_datasets.py \
    --dataset pannuke \
    --folds 2 \
    --output_dir data/evaluation
```

**Temps estimé:** 10-15 min (500 MB)

### Étape 2: Convertir les Annotations

```bash
python scripts/evaluation/convert_annotations.py \
    --dataset pannuke \
    --input_dir data/evaluation/pannuke/Fold\ 2 \
    --output_dir data/evaluation/pannuke_fold2_converted
```

**Temps estimé:** 5-10 min (7,901 images → .npz)

**Vérification:**
```bash
ls -lh data/evaluation/pannuke_fold2_converted/*.npz | wc -l
# Devrait afficher le nombre d'images converties
```

### Étape 3: Évaluation Complète (ou Échantillon)

**Option A: Échantillon rapide (100 images, ~10 min)**
```bash
python scripts/evaluation/evaluate_ground_truth.py \
    --dataset_dir data/evaluation/pannuke_fold2_converted \
    --output_dir results/pannuke_fold2_sample \
    --num_samples 100 \
    --dataset pannuke_fold2
```

**Option B: Évaluation complète (toutes les images, ~2-3h)**
```bash
python scripts/evaluation/evaluate_ground_truth.py \
    --dataset_dir data/evaluation/pannuke_fold2_converted \
    --output_dir results/pannuke_fold2_full \
    --dataset pannuke_fold2
```

**Vérification:**
```bash
cat results/pannuke_fold2_sample/clinical_report_pannuke_fold2_*.txt
```

### Étape 4: Analyser les Résultats

Fichiers générés:
- `clinical_report_*.txt` - Rapport lisible
- `metrics_*.json` - Métriques détaillées
- `confusion_matrix_*.npy` - Matrice 6×6

**Exemple de commande d'analyse:**
```bash
python -c "
import json
with open('results/pannuke_fold2_sample/metrics_*.json') as f:
    metrics = json.load(f)
    print(f\"Dice: {metrics['global_metrics']['dice']:.4f}\")
    print(f\"AJI:  {metrics['global_metrics']['aji']:.4f}\")
    print(f\"PQ:   {metrics['global_metrics']['pq']:.4f}\")
"
```

## 🎯 Métriques Cibles

| Métrique | Cible | Acceptable | Critique |
|----------|-------|------------|----------|
| **Dice** | ≥ 0.95 | ≥ 0.90 | < 0.85 |
| **AJI** | ≥ 0.80 | ≥ 0.70 | < 0.60 |
| **PQ** | ≥ 0.70 | ≥ 0.60 | < 0.50 |
| **F1 Neoplastic** | ≥ 0.90 | ≥ 0.85 | < 0.80 |

## ⚠️ Points de Blocage Potentiels

### 1. VRAM Insuffisante (RTX 4070 SUPER - 12 GB)

**Solution:** Utiliser batch_size=1 (déjà par défaut dans le script)

### 2. Checkpoints Manquants

**Vérifier:**
```bash
ls -lh models/checkpoints/organ_head_best.pth
ls -lh models/checkpoints/hovernet_*_best.pth
```

**Si manquants:** Re-entraîner selon CLAUDE.md ou contacter l'équipe.

### 3. PanNuke Trop Volumineux

**Alternatives:**
- Utiliser `--num_samples 100` pour test rapide
- Utiliser seulement certains organes (filtrage manuel)
- Télécharger seulement Fold 2 au lieu des 3 folds

## 📊 Résultats Attendus

Basé sur les performances d'entraînement (CLAUDE.md):

| Composant | Métrique | Valeur Entraînement | Attendu sur Test |
|-----------|----------|---------------------|------------------|
| OrganHead | Accuracy | 99.94% | ~99% |
| Glandular | NP Dice | 0.9648 | ~0.96 |
| Digestive | NP Dice | 0.9634 | ~0.96 |
| Urologic | NP Dice | 0.9318 | ~0.92 |
| Epidermal | NP Dice | 0.9542 | ~0.95 |
| Respiratory | NP Dice | 0.9409 | ~0.93 |

**Global attendu:**
- Dice: 0.94-0.96 (Excellent)
- AJI: 0.75-0.85 (Bon à Excellent)
- PQ: 0.65-0.75 (Acceptable à Excellent)

## ✅ Checklist Finale

Avant de lancer l'évaluation, vérifier:

- [ ] PanNuke Fold 2 téléchargé (~500 MB)
- [ ] Conversion .npz effectuée
- [ ] Tous les checkpoints présents (6 fichiers .pth)
- [ ] Environnement conda `cellvit` activé
- [ ] GPU accessible (nvidia-smi fonctionne)
- [ ] ~30 GB d'espace disque libre (pour résultats + cache)

## 🚀 Commande Rapide de Validation

Pour tester rapidement que tout fonctionne (1 image):

```bash
# Créer un fichier de test
cd /home/amar/projects/cellvit-optimus

# Convertir une seule image
python scripts/evaluation/convert_annotations.py \
    --dataset pannuke \
    --input_dir data/evaluation/pannuke/Fold\ 2 \
    --output_dir data/evaluation/test_single

# Évaluer
python scripts/evaluation/evaluate_ground_truth.py \
    --image data/evaluation/test_single/image_00000.npz \
    --output_dir results/test_single \
    --verbose
```

Si cette commande fonctionne, le pipeline complet est opérationnel !

---

**Note:** Ce fichier est une checklist de travail. Une fois l'évaluation terminée, documenter les résultats dans CLAUDE.md section "Pipeline d'Évaluation Ground Truth".
