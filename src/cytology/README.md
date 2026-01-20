# V14 Cytology — Source Code

> **Version:** 14.0 (Production Ready)
> **Date:** 2026-01-19
> **Statut:** ✅ Architecture Validée (Expert)

---

## 📋 Vue d'Ensemble

Ce dossier contient **tout le code source** du système V14 Cytologie.

**Architecture:** Fusion multimodale (H-Optimus 1536D + Morphométrie 20D → MLP)

---

## 📁 Structure Modules

```
src/cytology/
├── morphometry.py              # 20 features morphométriques (570 lignes)
├── models/
│   ├── cytology_classifier.py  # MLP + BatchNorm + Focal Loss (370 lignes)
│   └── __init__.py             # Exports models
├── __init__.py                 # Exports module complet
└── README.md                   # Ce fichier
```

---

## 🔬 Module: `morphometry.py`

**Fonction Principale:** Calcul des 20 features morphométriques à partir des masques CellPose

### Imports

```python
from src.cytology import (
    compute_single_cell_features,
    compute_batch_features,
    get_feature_names,
    validate_features,
    interpret_nc_ratio,
    interpret_chromatin_density,
)
```

### Usage

```python
import numpy as np
from src.cytology import compute_single_cell_features

# Pour UNE cellule
features = compute_single_cell_features(
    image_rgb=patch,              # (H, W, 3) RGB [0, 255]
    mask_nucleus=mask_nuclei,     # (H, W) binary
    mask_cytoplasm=mask_cyto,     # (H, W) binary (optionnel)
    pixel_size_um=0.25            # Résolution microscope
)
# Output: np.array shape (20,)

# Pour un BATCH de cellules
features_batch = compute_batch_features(
    images=patches,               # (N, H, W, 3)
    masks_nuclei=masks_nuclei,    # (N, H, W)
    masks_cytoplasm=masks_cyto,   # (N, H, W) ou None
    pixel_size_um=0.25
)
# Output: np.array shape (N, 20)
```

### Features Calculées (20)

| # | Feature | Importance Clinique |
|---|---------|---------------------|
| 1 | area_nucleus | Criterion 1 (Size of Nuclei) — ISBI 2014 |
| 2-10 | Géométrie | Forme, circularité, solidité, axes |
| 11-13 | Intensité + H-channel | **Criterion 3 (Chromatin Density)** 🔴 |
| 14-16 | Haralick texture | Granularité chromatine (GLCM) |
| 17-18 | **N/C ratio** | **Paris System (> 0.7 = High Grade)** 🔴 |
| 19-20 | Feret, roundness | Dimensions max, forme |

### ⚠️ CRITIQUE: SINGLE SOURCE OF TRUTH

**INTERDICTION:**
```python
# ❌ Ne JAMAIS lire features depuis CSV/Excel externe
features = pd.read_csv("sipakmed_features_provided.csv")
```

**OBLIGATOIRE:**
```python
# ✅ TOUJOURS recalculer features sur masques CellPose
features = compute_single_cell_features(image, mask_nucleus, mask_cyto)
```

**Raison:** Garantir cohérence 100% entre masques et features.

---

## 🧠 Module: `models/cytology_classifier.py`

**Fonction Principale:** MLP Classification Head avec fusion multimodale

### Imports

```python
from src.cytology.models import (
    CytologyClassifier,
    FocalLoss,
    compute_class_weights,
    count_parameters,
)
```

### Usage

```python
import torch
from src.cytology.models import CytologyClassifier, FocalLoss

# Créer modèle
model = CytologyClassifier(
    embedding_dim=1536,    # H-Optimus embeddings
    morpho_dim=20,         # Features morphométriques
    num_classes=7,         # SIPaKMeD (7 classes)
    use_batchnorm_morpho=True  # CRITIQUE
)

# Forward pass (training)
model.train()
logits = model(embeddings, morpho_features)

# Inference
model.eval()
probs = model.predict_proba(embeddings, morpho_features)

# Loss function (déséquilibre classes)
criterion = FocalLoss(gamma=2.0)
loss = criterion(logits, targets)
```

### Architecture

```
Input: embedding (1536D) + morpho (20D) = 1556D
    ↓
BatchNorm sur morpho (CRITIQUE pour équilibrage gradients)
    ↓
Concatenation → 1556D
    ↓
Dense(512) + ReLU + Dropout(0.3)
    ↓
Dense(256) + ReLU + Dropout(0.2)
    ↓
Dense(num_classes) + Softmax
    ↓
Output: Probabilités [0, 1] par classe
```

### Paramètres Totaux

```python
from src.cytology.models import count_parameters

total_params = count_parameters(model)
print(f"Total Parameters: {total_params:,}")
# Output: ~802,567 paramètres trainables
```

### ⚠️ CRITIQUE: BatchNorm sur Morpho

**Pourquoi OBLIGATOIRE:**

```python
# Problème déséquilibre dimensionnel:
embedding:    1536 dims, valeurs normalisées ~[-1, +1]
morpho:       20 dims, valeurs brutes (area=500, nc_ratio=0.7)

# Sans BatchNorm:
# → Gradient écrase features morpho (1536 >> 20)
# → MLP apprend UNIQUEMENT sur embedding
# → Morphométrie devient inutile ❌

# Avec BatchNorm:
# → Features morpho normalisées à même échelle qu'embedding
# → Gradient équilibré entre les deux branches
# → Fusion réellement multimodale ✅
```

---

## 🎯 Pipeline Production

### Workflow Complet

```python
import torch
from src.cytology import (
    compute_single_cell_features,
    CytologyClassifier,
)

# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE COMPLET (Pour UNE cellule)
# ═════════════════════════════════════════════════════════════════════════════

# Étape 1: Détection CellPose (externe)
bbox, mask_nucleus = cellpose_master.detect(tile)

# Étape 2: Crop + Padding (externe)
patch = crop_and_pad(tile, bbox, target_size=224)

# Étape 3A: H-Optimus embedding
with torch.no_grad():
    embedding = h_optimus_model(patch)  # (1536,)

# Étape 3B: Morphométrie (CE MODULE)
morpho_features = compute_single_cell_features(
    image_rgb=patch.numpy(),
    mask_nucleus=mask_nucleus,
    pixel_size_um=0.25
)  # (20,)

# Étape 4: Classification (CE MODULE)
model = CytologyClassifier(num_classes=7)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

embedding_tensor = torch.tensor(embedding).unsqueeze(0)  # (1, 1536)
morpho_tensor = torch.tensor(morpho_features).unsqueeze(0)  # (1, 20)

probs = model.predict_proba(embedding_tensor, morpho_tensor)  # (1, 7)
predicted_class = torch.argmax(probs, dim=1).item()

print(f"Predicted: {class_names[predicted_class]}")
print(f"Confidence: {probs[0, predicted_class]:.3f}")
```

---

## 📊 Tests & Validation

### Test Morphometry

```python
from src.cytology import compute_single_cell_features, validate_features

# Calculer features
features = compute_single_cell_features(image, mask)

# Valider
is_valid, message = validate_features(features)
assert is_valid, f"Features invalides: {message}"

# Vérifier noms
from src.cytology import get_feature_names
names = get_feature_names()
assert len(names) == 20
```

### Test Classifier

```python
from src.cytology.models import CytologyClassifier

# Créer modèle
model = CytologyClassifier(num_classes=7)

# Test forward pass
batch_size = 4
embedding = torch.randn(batch_size, 1536)
morpho = torch.randn(batch_size, 20)

# Training mode
model.train()
logits = model(embedding, morpho)
assert logits.shape == (batch_size, 7)

# Eval mode
model.eval()
probs = model.predict_proba(embedding, morpho)
assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))
```

---

## 🔗 Documentation Associée

**Specs Techniques:**
- `docs/cytology/V14_CYTOLOGY_BRANCH.md` — Architecture complète
- `docs/cytology/V14_PIPELINE_EXECUTION_ORDER.md` — Ordre exécution

**Scripts Pipeline:**
- `scripts/cytology/README.md` — Guide pratique

---

## 📝 Changelog

### Version 14.0 — 2026-01-19 (Production Ready)

**Nouveau:**
- ✅ `morphometry.py` — 20 features complètes (ISBI 2014 + Paris System)
- ✅ `models/cytology_classifier.py` — MLP avec BatchNorm + Focal Loss
- ✅ Clinical interpretation functions (Paris System, Bethesda)

**Validé:**
- ✅ Architecture expert (2026-01-19)
- ✅ SINGLE SOURCE OF TRUTH (features sur masques)
- ✅ BatchNorm critique pour fusion multimodale

---

**Auteur:** V14 Cytology Branch
**Validation:** Expert (2026-01-19)
**Statut:** ✅ Production Ready
