# V14 Cytology — Pipeline Production

> **Version:** 1.0
> **Date:** 2026-01-21
> **Statut:** Spécification validée
> **Objectif:** Documenter le pipeline end-to-end pour la production

---

## 📋 Vue d'Ensemble

### Différence POC vs Production

| Aspect | POC (SIPaKMeD) | Production |
|--------|----------------|------------|
| **Input** | Cellules pré-découpées | Images complètes (FOV) |
| **Détection** | Non requise (GT masks) | **CellPose** |
| **Noyaux** | Connus d'avance | Détectés automatiquement |
| **Validation** | MLP seul | Pipeline complet |

### Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE V14 CYTOLOGY — PRODUCTION                        │
└─────────────────────────────────────────────────────────────────────────────┘

                         IMAGE COMPLÈTE
                        (ex: 2048×1532 px)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1: DÉTECTION (CellPose)                                              │
│  ─────────────────────────────                                              │
│  • Input: Image RGB complète                                                │
│  • Modèle: CellPose "nuclei" (pré-entraîné)                                │
│  • Output: N masques de noyaux + N bounding boxes                           │
│  • Temps: ~300-500ms / image                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ N noyaux détectés
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2: EXTRACTION PATCHES (Pour chaque noyau)                            │
│  ───────────────────────────────────────────────                            │
│  • Calcul du centroïde du masque                                            │
│  • Crop 224×224 centré sur le centroïde                                     │
│  • Padding blanc si proche du bord                                          │
│  • Output: N patches 224×224 RGB                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ N patches 224×224
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 3: FEATURE EXTRACTION (H-Optimus-0)                                  │
│  ─────────────────────────────────────────                                  │
│  • Modèle: H-Optimus-0 (1.1B params, GELÉ)                                 │
│  • Input: Batch de patches 224×224                                          │
│  • Output: N embeddings CLS (1536 dims)                                     │
│  • Temps: ~100ms / batch de 16                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ N embeddings (1536D)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 4: MORPHOMÉTRIE (Optionnel, sur masques CellPose)                    │
│  ───────────────────────────────────────────────────────                    │
│  • 20 features: géométrie, intensité H-channel, texture                     │
│  • Calculées sur les masques CellPose (pas GT)                              │
│  • Output: N vecteurs (20 dims)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ N vecteurs fusionnés (1556D)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 5: CLASSIFICATION (MLP)                                              │
│  ─────────────────────────────                                              │
│  • Input: Vecteur fusionné (1556D)                                          │
│  • Architecture: 1556 → 512 → 256 → 128 → K classes                        │
│  • Output: Probabilités par classe + Confiance                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    RAPPORT DIAGNOSTIC
              "X cellules anormales sur N analysées"
```

---

## 🔬 Étape 1: Détection CellPose

### Rôle

CellPose localise et segmente les noyaux dans l'image complète.
**C'est la brique CRITIQUE** — sans détection correcte, pas de classification.

### Configuration

```python
from cellpose import models

# Modèle recommandé pour noyaux
model = models.CellposeModel(gpu=True, pretrained_model='nuclei')

# Paramètres optimaux (LBC - validés sur APCData 2026-01-21)
masks, flows, styles = model.eval(
    image,
    diameter=60,              # Diamètre optimal pour LBC (validé)
    flow_threshold=0.4,       # Seuil de confiance
    cellprob_threshold=0.0
)

# Post-filtrage par surface (élimine débris)
MIN_AREA = 500  # px² (élimine lymphocytes/débris)
MAX_AREA = 100000  # px² (élimine artefacts)
```

### Paramètres Validés (APCData - 2026-01-21)

| Paramètre | Valeur | Impact |
|-----------|--------|--------|
| `diameter` | **60** | Taille moyenne noyaux LBC |
| `flow_threshold` | **0.4** | Balance détection/précision |
| `cellprob_threshold` | **0.0** | Standard |
| `min_area` | **500 px²** | Filtre lymphocytes/débris |
| `max_distance` | **100 px** | Tolérance matching GT |

### Résultats de Validation (n=20 images)

| Métrique | Valeur | Cible | Status |
|----------|--------|-------|--------|
| **Abnormal Detection Rate** | **92.3%** | ≥98% | ⚠️ WARN |
| Detection Rate (All) | 85.9% | ≥90% | - |
| ASCUS | 100% | - | ✅ |
| ASCH | 100% | - | ✅ |
| HSIL | 100% | - | ✅ |
| LSIL | 90.9% | - | ✅ |
| SCC | 72.7% | - | ⚠️ |

> **Note:** La précision basse (~8%) est ATTENDUE car APCData n'annote qu'un
> sous-ensemble de cellules. CellPose détecte correctement les cellules
> non-annotées (normales) qui seront filtrées par le classifieur.

### Output

| Élément | Type | Description |
|---------|------|-------------|
| `masks` | np.array (H, W) | Image labelisée (0=fond, 1=noyau1, 2=noyau2, ...) |
| `flows` | list | Champs de flux (pour debug) |
| `styles` | np.array | Style embeddings (ignoré) |

### Métriques de Validation

| Métrique | Formule | Cible |
|----------|---------|-------|
| **Detection Rate** | TP / (TP + FN) | > 90% |
| **False Positive Rate** | FP / Total détections | < 10% |
| **IoU moyen** | Mean IoU avec GT | > 0.7 |

---

## 🔬 Étape 2: Extraction Patches

### Algorithme

```python
def extract_patch(image, mask, cell_id, patch_size=224):
    """
    Extrait un patch 224×224 centré sur un noyau détecté.

    Args:
        image: Image RGB complète (H, W, 3)
        mask: Masque CellPose (H, W) avec labels
        cell_id: ID du noyau dans le masque
        patch_size: Taille du patch (224)

    Returns:
        patch: Image 224×224 RGB
        cell_mask: Masque 224×224 binaire
    """
    # 1. Extraire le masque du noyau spécifique
    cell_mask = (mask == cell_id).astype(np.uint8)

    # 2. Calculer le centroïde
    props = regionprops(cell_mask)
    if len(props) == 0:
        return None, None

    cy, cx = props[0].centroid
    cx, cy = int(cx), int(cy)

    # 3. Calculer les coordonnées du crop
    half = patch_size // 2
    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half

    # 4. Gérer les bords (padding blanc)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - image.shape[1])
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - image.shape[0])

    x1, x2 = max(0, x1), min(image.shape[1], x2)
    y1, y2 = max(0, y1), min(image.shape[0], y2)

    # 5. Extraire et padder
    patch = image[y1:y2, x1:x2]

    if any([pad_left, pad_right, pad_top, pad_bottom]):
        patch = np.pad(
            patch,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=255  # Blanc
        )

    # 6. Même chose pour le masque
    cell_mask_crop = cell_mask[y1:y2, x1:x2]
    if any([pad_left, pad_right, pad_top, pad_bottom]):
        cell_mask_crop = np.pad(
            cell_mask_crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )

    return patch, cell_mask_crop
```

### Points Critiques

1. **Centroïde** — Utiliser le centre de masse du masque, pas le centre du bounding box
2. **Padding blanc** — Fond de microscope = blanc (255, 255, 255)
3. **Masque associé** — Garder le masque pour la morphométrie

---

## 🔬 Étape 3: Feature Extraction (H-Optimus-0)

### Configuration

```python
import torch
from transformers import AutoModel

# Charger H-Optimus-0
model = AutoModel.from_pretrained(
    "bioptimus/H-optimus-0",
    trust_remote_code=True
)
model.eval()
model.cuda()

# Normalisation spécifique H-Optimus
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD)
])
```

### Batch Processing

```python
def extract_embeddings_batch(patches, model, batch_size=16):
    """
    Extrait les embeddings CLS pour un batch de patches.

    Args:
        patches: List de N patches 224×224 RGB
        model: H-Optimus-0
        batch_size: Taille du batch GPU

    Returns:
        embeddings: Tensor (N, 1536)
    """
    embeddings = []

    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        batch_tensor = torch.stack([transform(p) for p in batch]).cuda()

        with torch.no_grad():
            outputs = model(batch_tensor)
            # CLS token = première position
            cls_tokens = outputs[:, 0, :]  # (B, 1536)

        embeddings.append(cls_tokens.cpu())

    return torch.cat(embeddings, dim=0)  # (N, 1536)
```

---

## 🔬 Étape 4: Morphométrie

### Features Calculées (20 dims)

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | area | regionprops | Aire du noyau (pixels²) |
| 2 | perimeter | regionprops | Périmètre |
| 3 | circularity | calculé | 4π × area / perimeter² |
| 4 | eccentricity | regionprops | 0=rond, 1=ligne |
| 5 | solidity | regionprops | area / convex_area |
| 6 | extent | regionprops | area / bbox_area |
| 7 | major_axis | regionprops | Longueur axe majeur |
| 8 | minor_axis | regionprops | Longueur axe mineur |
| 9 | aspect_ratio | calculé | major / minor |
| 10 | compactness | calculé | perimeter² / area |
| 11 | mean_intensity | H-channel | Intensité moyenne (Ruifrok) |
| 12 | std_intensity | H-channel | Écart-type intensité |
| 13 | max_intensity | H-channel | Intensité max |
| 14 | min_intensity | H-channel | Intensité min |
| 15 | integrated_od | H-channel | mean × area (proxy ploïdie) |
| 16 | contrast | GLCM | Texture Haralick |
| 17 | homogeneity | GLCM | Texture Haralick |
| 18 | energy | GLCM | Texture Haralick |
| 19 | correlation | GLCM | Texture Haralick |
| 20 | entropy | GLCM | Texture Haralick |

### Important

**En production, les features sont calculées sur les masques CellPose, PAS sur des masques GT.**

Cela signifie que la qualité de la segmentation CellPose impacte directement la morphométrie.

---

## 🔬 Étape 5: Classification MLP

### Architecture

```
Input (1556) → Linear(512) → BN → ReLU → Dropout(0.3)
            → Linear(256) → BN → ReLU → Dropout(0.3)
            → Linear(128) → BN → ReLU → Dropout(0.3)
            → Linear(K) → Softmax
```

### Classes de Sortie

**Option A: 6 classes Bethesda (APCData)**
```python
CLASSES_BETHESDA = ['NILM', 'ASCUS', 'ASCH', 'LSIL', 'HSIL', 'SCC']
```

**Option B: Binaire (Safety First)**
```python
CLASSES_BINARY = ['Normal', 'Abnormal']
# NILM → Normal
# ASCUS, ASCH, LSIL, HSIL, SCC → Abnormal
```

---

## 📊 Validation avec APCData

### Stratégie

APCData fournit des annotations GT (nucleus_x, nucleus_y, classe).
On les utilise pour **valider CellPose**, pas pour entraîner.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VALIDATION PIPELINE AVEC APCDATA                                           │
└─────────────────────────────────────────────────────────────────────────────┘

IMAGE APCDATA (2048×1532)
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│  CellPose Détection │              │  GT Annotations     │
│  (automatique)      │              │  (nucleus_x, y)     │
└─────────────────────┘              └─────────────────────┘
         │                                      │
         │ N détections                         │ M annotations
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  MATCHING           │
              │  (Distance < 50px)  │
              └─────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Matched │   │ Missed  │   │ False   │
    │ (TP)    │   │ (FN)    │   │ Pos (FP)│
    └─────────┘   └─────────┘   └─────────┘
         │
         │ Pour chaque match:
         │ - Utiliser label GT
         │ - Crop autour détection CellPose
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION (H-Optimus + MLP)                                           │
│  → Comparer prédiction vs GT label                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Métriques de Validation

| Étape | Métrique | Calcul | Cible |
|-------|----------|--------|-------|
| **CellPose** | Detection Rate | TP / (TP + FN) | > 90% |
| **CellPose** | Precision | TP / (TP + FP) | > 85% |
| **CellPose** | F1 Score | 2×P×R / (P+R) | > 87% |
| **Classification** | Sensitivity | Sur TP uniquement | > 98% |
| **Classification** | Cohen's Kappa | Sur TP uniquement | > 0.80 |
| **End-to-End** | Sensitivity | Détecté ET bien classé | > 88% |

---

## 🔧 Scripts du Pipeline

| Script | Rôle | Input | Output |
|--------|------|-------|--------|
| `05_validate_cellpose_apcdata.py` | Valider CellPose sur APCData | Images + GT | Detection metrics |
| `06_end_to_end_apcdata.py` | Pipeline complet | Images | Classifications |
| `07_compare_with_gt.py` | Comparer prédictions vs GT | Predictions + GT | Métriques finales |

### Commandes

```bash
# Étape 1: Valider CellPose
python scripts/cytology/05_validate_cellpose_apcdata.py \
    --data_dir data/raw/apcdata/APCData_points \
    --output_dir reports/cellpose_validation \
    --n_samples 50

# Étape 2: Pipeline End-to-End
python scripts/cytology/06_end_to_end_apcdata.py \
    --data_dir data/raw/apcdata/APCData_points \
    --checkpoint models/checkpoints_v14_cytology/best_model.pth \
    --output_dir reports/end_to_end_validation

# Étape 3: Métriques finales
python scripts/cytology/07_compare_with_gt.py \
    --predictions reports/end_to_end_validation/predictions.json \
    --gt_dir data/raw/apcdata/APCData_points/labels/json
```

---

## ⚠️ Points Critiques Production

### 1. Erreurs en Cascade

```
CellPose rate un noyau → Pas de crop → Pas de classification → FN
```

**Solution:** Optimiser d'abord CellPose (Detection Rate > 90%)

### 2. Faux Positifs CellPose

```
CellPose détecte un débris → Crop → Classification → Potentiel FP
```

**Solution:**
- Filtrer par taille (area < seuil → ignorer)
- Confiance CellPose (flow_threshold)

### 3. Qualité du Masque

```
Mauvais masque → Mauvaise morphométrie → Classification dégradée
```

**Solution:**
- Valider IoU masques CellPose vs GT
- Features robustes (moins dépendantes du contour exact)

---

## 📈 KPIs Production

| Catégorie | KPI | Seuil | Priorité |
|-----------|-----|-------|----------|
| **Détection** | Detection Rate | > 90% | 🔴 Critique |
| **Détection** | False Positive Rate | < 10% | 🔴 Critique |
| **Classification** | Sensitivity (Abnormal) | > 98% | 🔴 Critique |
| **Classification** | Specificity | > 60% | 🟡 Important |
| **End-to-End** | Sensitivity globale | > 88% | 🔴 Critique |
| **Performance** | Temps / image | < 2s | 🟡 Important |

---

## 🎯 Roadmap Validation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ROADMAP VALIDATION V14 PRODUCTION                                          │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: POC (SIPaKMeD) ────────────────────────────────────── ✅ DONE
  └── MLP seul, GT masks
  └── Sensitivity 99.26%, Kappa 0.72

PHASE 2: CellPose Validation (APCData) ─────────────────────── 🔄 EN COURS
  └── Sprint 2.1: Detection Rate sur APCData (cible > 90%)
  └── Sprint 2.2: IoU masques vs GT

PHASE 3: End-to-End (APCData) ──────────────────────────────── ⏳ PENDING
  └── Pipeline complet: CellPose → H-Optimus → MLP
  └── Sensitivity end-to-end > 88%

PHASE 4: Stress Test (CRIC) ────────────────────────────────── ⏳ PENDING
  └── Images difficiles (chevauchements)
  └── Valider robustesse CellPose

PHASE 5: Production Ready ──────────────────────────────────── ⏳ PENDING
  └── Intégration Router Histo/Cyto
  └── Tests multi-scanners
  └── Déploiement Dubai
```

---

*Documentation générée le 2026-01-21*
