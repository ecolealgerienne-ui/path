# V14 Cytology Pipeline Scripts

Pipeline complet pour entraînement et évaluation du système cytologie V14.

## 📋 Vue d'Ensemble

```
PIPELINE V14 CYTOLOGIE (5 Étapes)

1. Generate Masks (CellPose)    → 01_generate_cellpose_masks.py
2. Extract Embeddings (H-Optimus) → 02_extract_h_optimus_embeddings.py
3. Compute Features (Morphométrie) → 03_compute_morpho_features.py
4. Train Classifier (MLP)        → 04_train_mlp_classifier.py
5. Evaluate (Metrics Safety First) → 05_evaluate_cytology.py
```

## 🚀 Quick Start

### Préparation Dataset (SIPaKMeD)

```bash
# Structure attendue:
data/raw/sipakmed/pictures/
├── carcinoma_in_situ/      # 813 images
├── severe_dysplastic/      # 1,470 images
├── moderate_dysplastic/    # 793 images
├── light_dysplastic/       # 1,484 images
├── normal_columnar/        # 787 images
├── normal_intermediate/    # 518 images
└── normal_superficiel/     # 502 images
```

### Exécution Pipeline Complet

```bash
# Étape 1: Générer masques CellPose (nuclei)
python scripts/cytology/01_generate_cellpose_masks.py \
    --sipakmed_dir data/raw/sipakmed/pictures \
    --output_dir data/processed/cellpose_masks \
    --model_type nuclei \
    --batch_size 8

# Étape 2: Extraire embeddings H-Optimus
python scripts/cytology/02_extract_h_optimus_embeddings.py \
    --images_dir data/raw/sipakmed/pictures \
    --masks_dir data/processed/cellpose_masks \
    --output_dir data/processed/h_optimus_embeddings \
    --use_macenko \
    --batch_size 32

# Étape 3: Calculer features morphométriques
python scripts/cytology/03_compute_morpho_features.py \
    --images_dir data/raw/sipakmed/pictures \
    --masks_dir data/processed/cellpose_masks \
    --output_csv data/processed/morpho_features/sipakmed_features.csv

# Étape 4: Entraîner classifier MLP
python scripts/cytology/04_train_mlp_classifier.py \
    --embeddings_dir data/processed/h_optimus_embeddings \
    --features_csv data/processed/morpho_features/sipakmed_features.csv \
    --output_dir models/checkpoints_v14_cytology \
    --epochs 100 \
    --batch_size 64 \
    --use_focal_loss

# Étape 5: Évaluer (Safety First)
python scripts/cytology/05_evaluate_cytology.py \
    --checkpoint models/checkpoints_v14_cytology/best_model.pth \
    --embeddings_dir data/processed/h_optimus_embeddings \
    --features_csv data/processed/morpho_features/sipakmed_features.csv \
    --split val \
    --sensitivity_threshold 0.98
```

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

## 🔧 Configuration

### CellPose Parameters

```python
# 01_generate_cellpose_masks.py
CELLPOSE_CONFIG = {
    "model_type": "nuclei",      # Master model (toujours)
    "diameter": 30,               # Taille noyau moyenne (pixels)
    "flow_threshold": 0.4,        # Sensibilité détection
    "cellprob_threshold": 0.0,    # Seuil probabilité cellule
    "channels": [0, 0],           # Grayscale
}
```

### H-Optimus Parameters

```python
# 02_extract_h_optimus_embeddings.py
HOPTIMUS_CONFIG = {
    "model_name": "bioptimus/H-optimus-0",
    "input_size": 224,
    "mean": (0.707223, 0.578729, 0.703617),
    "std": (0.211883, 0.230117, 0.177517),
    "use_macenko": True,          # ✅ ON pour Cytologie
}
```

### Training Parameters

```python
# 04_train_mlp_classifier.py
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "use_focal_loss": True,       # Déséquilibre classes
    "gamma": 2.0,                 # Focal loss gamma
    "early_stopping_patience": 15,
    "reduce_lr_patience": 5,
}
```

## 📁 Structure Données Générées

```
data/
├── processed/
│   ├── cellpose_masks/
│   │   ├── train/
│   │   │   ├── carcinoma_in_situ/
│   │   │   │   ├── img001_mask.npy
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── val/
│   ├── h_optimus_embeddings/
│   │   ├── train_embeddings.npy      # (N_train, 1536)
│   │   ├── val_embeddings.npy        # (N_val, 1536)
│   │   └── metadata.json             # IDs, labels
│   └── morpho_features/
│       ├── train_features.csv        # (N_train, 20)
│       └── val_features.csv          # (N_val, 20)
└── models/
    └── checkpoints_v14_cytology/
        ├── best_model.pth
        ├── training_log.json
        └── confusion_matrix.png
```

## ⚠️ Points Critiques

### 1. SINGLE SOURCE OF TRUTH

**❌ INTERDICTION:**
```python
# Ne JAMAIS lire features depuis Excel/CSV fourni
features = pd.read_csv("sipakmed_features_provided.csv")  # ❌
```

**✅ OBLIGATOIRE:**
```python
# TOUJOURS recalculer features sur masques CellPose
features = compute_morpho_features(images, masks)  # ✅
```

### 2. BatchNorm Training/Inference

```python
# Training: BatchNorm utilise batch statistics
model.train()
loss = criterion(model(emb, morpho), targets)

# Inference: BatchNorm utilise running stats
model.eval()
with torch.no_grad():
    probs = model.predict_proba(emb, morpho)
```

### 3. Macenko ON pour Cytologie

```python
# ✅ CORRECT (V14 Cytologie)
patch_normalized = macenko_normalize(patch)
embedding = h_optimus(patch_normalized)

# ❌ INCORRECT (Causerait régression V13 Histologie)
# Mais OK pour Cytologie car pas de FPN Chimique
```

## 📚 Documentation Associée

| Document | Description |
|----------|-------------|
| [V14_PIPELINE_EXECUTION_ORDER.md](../../docs/cytology/V14_PIPELINE_EXECUTION_ORDER.md) | Ordre d'exécution complet |
| [V14_MACENKO_STRATEGY.md](../../docs/cytology/V14_MACENKO_STRATEGY.md) | Normalisation router-dependent |
| [V14_CYTOLOGY_BRANCH.md](../../docs/cytology/V14_CYTOLOGY_BRANCH.md) | Specs complètes V14 |

## 🐛 Troubleshooting

### Erreur: "Empty nucleus mask"

```bash
# Vérifier paramètres CellPose
python scripts/cytology/01_generate_cellpose_masks.py \
    --diameter 30 \
    --flow_threshold 0.3  # Réduire pour plus de sensibilité
```

### Erreur: "Shape mismatch (1536) vs (1550)"

```bash
# Vérifier que morpho features = 20 dims (pas 14)
python -c "from src.cytology import get_feature_names; print(len(get_feature_names()))"
# Output attendu: 20
```

### Accuracy trop basse (< 0.80)

```bash
# Vérifier déséquilibre classes
python scripts/cytology/04_train_mlp_classifier.py \
    --use_focal_loss \
    --gamma 2.0 \
    --balance_classes
```

---

**Auteur:** V14 Cytology Branch
**Date:** 2026-01-19
**Statut:** ✅ Ready for Implementation
