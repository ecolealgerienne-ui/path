# V14 Cytology Pipeline Scripts

Pipeline complet pour entraînement et évaluation du système cytologie V14.

## 📋 Vue d'Ensemble

```
PIPELINE V14 CYTOLOGIE — STRATÉGIE DUALE

┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DÉVELOPPEMENT (SIPaKMeD)      │  PHASE 2: PRODUCTION (Lames Réelles)
├─────────────────────────────────────────┼───────────────────────────────────┤
│  Segmentation: Masques GT               │  Segmentation: CellPose           │
│  Dataset: Cellules isolées              │  Dataset: Groupes cellulaires     │
│  But: Valider architecture              │  But: Déploiement clinique        │
└─────────────────────────────────────────┴───────────────────────────────────┘
```

## 🎯 Décision Stratégique (2026-01-20)

> **CellPose inadapté pour SIPaKMeD** — Utilisation des masques Ground Truth

### Pourquoi?

| Aspect | SIPaKMeD | Production (Lames réelles) |
|--------|----------|---------------------------|
| Format | 1 cellule isolée/image | 100+ cellules/patch |
| Fond | Blanc (padding) | Tissu/variable |
| CellPose | ❌ Sur-segmente (4 objets au lieu de 1) | ✅ Optimisé pour groupes |
| Solution | **Masques GT** | **CellPose Master/Slave** |

### Validation Expérimentale

```
CellPose sur SIPaKMeD (cellule isolée 168×156):
  Diameter=50: 4 objets détectés, 21.8% coverage (attendu: 1 objet, 22%)
  → Sur-segmentation systématique

CellPose sur tissu (groupes cellulaires):
  → Fonctionne correctement (usage prévu)
```

---

## 🚀 Pipeline Actuel (Phase 1: SIPaKMeD)

```
00_preprocess_sipakmed.py     → Prépare images 224×224 + masques GT
00b_validate_cellpose.py      → Validation CellPose (diagnostic uniquement)
01_extract_embeddings_gt.py   → Extrait H-Optimus avec masques GT ← NOUVEAU
02_compute_morphometry.py     → Calcule 20 features morphométriques
03_train_mlp_classifier.py    → Entraîne MLP fusion (1550D → classes)
04_evaluate_cytology.py       → Évalue (Sensibilité > 0.98)
```

### Exécution

```bash
# Étape 0: Prétraitement SIPaKMeD (images + masques GT)
python scripts/cytology/00_preprocess_sipakmed.py \
    --raw_dir data/raw/sipakmed/pictures \
    --output_dir data/processed/sipakmed

# Étape 0b: Validation CellPose (optionnel, diagnostic)
python scripts/cytology/00b_validate_cellpose.py \
    --data_dir data/processed/sipakmed \
    --split val \
    --n_samples 50

# Étape 1: Extraire embeddings H-Optimus (avec masques GT)
python scripts/cytology/01_extract_embeddings_gt.py \
    --data_dir data/processed/sipakmed \
    --output_dir data/embeddings/sipakmed \
    --split both \
    --batch_size 16

# Étape 2: Calculer features morphométriques
python scripts/cytology/02_compute_morphometry.py \
    --data_dir data/processed/sipakmed \
    --embeddings_dir data/embeddings/sipakmed \
    --output_dir data/features/sipakmed

# Étape 3: Entraîner MLP
python scripts/cytology/03_train_mlp_classifier.py \
    --features_dir data/features/sipakmed \
    --output_dir models/checkpoints_v14_cytology \
    --epochs 100 \
    --use_focal_loss

# Étape 4: Évaluer (Safety First) — Validation formelle POC
python scripts/cytology/04_evaluate_cytology.py \
    --checkpoint models/checkpoints_v14_cytology/best_model.pth \
    --features_dir data/features/sipakmed \
    --output_dir reports/v14_cytology_validation
```

**Outputs générés:**
- `validation_report.md` — Rapport complet avec KPIs
- `confusion_matrix_detailed.png` — Matrice 7 classes
- `confusion_matrix_binary.png` — Normal vs Abnormal
- `per_class_recall.png` — Recall par classe
- `kpi_summary.png` — Résumé KPIs vs targets
- `validation_metrics.json` — Métriques brutes

---

## 📊 Métriques Prioritaires

**Safety First (Cytologie):**

| Métrique | Seuil Cible | Priorité |
|----------|-------------|----------|
| **Sensibilité Malin** | **> 0.98** | 🔴 CRITIQUE |
| **FROC (FP/WSI @ 98% sens)** | **< 2.0** | 🔴 CRITIQUE |
| **Cohen's Kappa** | **> 0.80** | 🔴 CRITIQUE |
| IoU Noyau | > 0.85 | 🟡 Important |
| AP50 (COCO) | > 0.90 | 🟡 Important |
| Spécificité | > 0.60 | 🟢 Secondaire |

**Principe:** Ne JAMAIS rater un cancer (Sensibilité > Précision).

---

## 📁 Structure Données

```
data/
├── raw/
│   └── sipakmed/
│       └── pictures/
│           ├── carcinoma_in_situ/      # 813 images
│           ├── severe_dysplastic/      # 1,470 images
│           ├── moderate_dysplastic/    # 793 images
│           ├── light_dysplastic/       # 1,484 images
│           ├── normal_columnar/        # 787 images
│           ├── normal_intermediate/    # 518 images
│           └── normal_superficiel/     # 502 images
│
├── processed/
│   └── sipakmed/
│       ├── train/
│       │   ├── images/                 # 224×224 PNG
│       │   ├── masks/                  # Masques GT binaires
│       │   └── metadata.json
│       └── val/
│
├── embeddings/
│   └── sipakmed/
│       ├── sipakmed_train_embeddings.pt  # CLS + patch tokens
│       └── sipakmed_val_embeddings.pt
│
└── features/
    └── sipakmed/
        ├── train_features.csv          # 20 features morpho
        └── val_features.csv
```

---

## 🔧 Configuration

### H-Optimus Parameters

```python
HOPTIMUS_CONFIG = {
    "model_name": "bioptimus/H-optimus-0",
    "input_size": 224,
    "mean": (0.707223, 0.578729, 0.703617),
    "std": (0.211883, 0.230117, 0.177517),
    "cls_dim": 1536,
    "n_patches": 256,
}
```

### MLP Fusion Architecture

```python
MLP_CONFIG = {
    "input_dim": 1550,        # 1536 (CLS) + 14 (morpho) → 20 morpho bientôt
    "hidden_dims": [512, 256, 128],
    "n_classes": 7,           # SIPaKMeD classes
    "dropout": 0.3,
    "use_batchnorm": True,
}
```

### SIPaKMeD Mask Values

```python
# Masques indexés SIPaKMeD (-d.bmp files)
SIPAKMED_MASK_VALUES = {
    0: "artifact",
    1: "artifact",
    2: "NUCLEUS",      # ← Valeur utilisée
    3: "cytoplasm",
    4: "background",
}
```

---

## ⚠️ Points Critiques

### 1. Masques GT vs CellPose

```python
# ✅ Phase 1 (SIPaKMeD): Utiliser masques GT
mask = load_gt_mask(sample)  # Depuis data/processed/sipakmed/masks/

# ✅ Phase 2 (Production): Utiliser CellPose
mask = cellpose_model.eval(patch)  # Sur lames réelles
```

### 2. H-Optimus est un Extracteur, pas un Segmenteur

```
CellPose: Détecte/segmente les noyaux → sensible au domaine visuel
H-Optimus: Extrait features d'un patch → fonctionne sur tout patch propre

→ H-Optimus fonctionne sur SIPaKMeD même si CellPose échoue
```

### 3. SINGLE SOURCE OF TRUTH

```python
# ❌ INTERDICTION: Lire features pré-calculées
features = pd.read_csv("sipakmed_features_provided.csv")

# ✅ OBLIGATOIRE: Recalculer sur masques utilisés
features = compute_morpho_features(images, masks)
```

---

## 📚 Documentation Associée

| Document | Description |
|----------|-------------|
| [V14_PIPELINE_EXECUTION_ORDER.md](../../docs/cytology/V14_PIPELINE_EXECUTION_ORDER.md) | Ordre d'exécution complet |
| [V14_MACENKO_STRATEGY.md](../../docs/cytology/V14_MACENKO_STRATEGY.md) | Normalisation router-dependent |
| [V14_CYTOLOGY_BRANCH.md](../../docs/cytology/V14_CYTOLOGY_BRANCH.md) | Specs complètes V14 |
| [V14_MASTER_SLAVE_ARCHITECTURE.md](../../docs/cytology/V14_MASTER_SLAVE_ARCHITECTURE.md) | CellPose pour production |

---

## 🐛 Troubleshooting

### CellPose sur-segmente (SIPaKMeD)

```
Problème: CellPose détecte 4 objets au lieu de 1
Cause: CellPose optimisé pour tissus, pas cellules isolées
Solution: Utiliser masques GT (Phase 1)
```

### Erreur: "H-Optimus model not found"

```bash
# Installer dépendances
pip install timm huggingface_hub

# Se connecter à HuggingFace
huggingface-cli login
```

### Erreur: "CUDA out of memory"

```bash
# Réduire batch size
python scripts/cytology/01_extract_embeddings_gt.py --batch_size 8
```

---

**Auteur:** V14 Cytology Branch
**Date:** 2026-01-20
**Statut:** ✅ Phase 1 Ready (Masques GT)
