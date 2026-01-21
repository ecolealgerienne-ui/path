# CRICVA Dataset — Documentation

> **Version:** 1.1
> **Date:** 2026-01-21
> **Source:** CRIC Cervix Database (Visual Attention subset)
> **URL Officiel:** https://sites.google.com/view/cricvadataset
> **Mendeley:** https://data.mendeley.com/datasets/bk45c9yxb9/1

---

## Vue d'Ensemble

| Attribut | Valeur |
|----------|--------|
| **Nom** | CRICVA (CRIC Visual Attention) |
| **Type** | Eye-tracking / Visual Attention |
| **Images** | 232 (8 trials) |
| **Résolution** | 1280 × 960 px (variable ~956-960) |
| **Format** | PNG RGB |
| **Classes** | 5 Bethesda (Negative, ASC-US, ASC-H, LSIL, ca) |
| **Annotations** | Labels image-level + heatmaps eye-tracking |
| **Équipement** | Eye Link 1000 (SR Research), 1000 Hz, œil droit |

---

## ⚠️ Limitation Critique pour V14

> **CRICVA ≠ Dataset de segmentation cellulaire**
>
> Ce dataset contient des **données d'eye-tracking** (où les pathologistes regardent),
> **PAS** des annotations de localisation des cellules.

### Verdict Définitif (Analyse 2026-01-21)

| Contenu | Type | Usage CellPose |
|---------|------|----------------|
| `fixLocs` (.mat) | Heatmap 960×1280 (eye-tracking) | ❌ **Non** |
| `fixation_maps/` | PNG grayscale des mêmes heatmaps | ❌ **Non** |
| `labels_*.txt` | Classes image-level uniquement | ❌ **Non** (pas de coordonnées) |

**Utilisation possible:**
- Validation classification image-level (232 images)
- Recherche sur l'attention visuelle des pathologistes
- Entraînement modèles attention-guided (R&D avancé)

**NON utilisable pour:**
- ❌ Validation CellPose (pas de coordonnées GT cellules)
- ❌ Entraînement segmentation
- ❌ Matching détections vs GT

---

## Structure

```
data/raw/CRICVA/
├── CRICVA/
│   ├── trial_01/               # 26 images
│   │   ├── images/             # PNG files
│   │   ├── fixation_locs/      # Eye-tracking coordinates
│   │   ├── fixation_maps/      # Heatmaps attention
│   │   └── labels_trial_01.txt # Image-level labels
│   ├── trial_02/               # 26 images
│   ├── trial_03/               # 25 images
│   ├── trial_04/               # 25 images
│   ├── trial_05/               # 25 images
│   ├── trial_06/               # 25 images
│   ├── trial_07/               # 40 images
│   └── trial_08/               # 40 images
└── preview/
```

### Distribution par Trial

| Trial | Images |
|-------|--------|
| trial_01 | 26 |
| trial_02 | 26 |
| trial_03 | 25 |
| trial_04 | 25 |
| trial_05 | 25 |
| trial_06 | 25 |
| trial_07 | 40 |
| trial_08 | 40 |
| **Total** | **232** |

---

## Format des Labels

**Fichier:** `labels_trial_XX.txt`

```
id,hash,class
1,011fda505d7e4af4b8cc57545343624d,ASC-US
2,02c7fb946ad5c5e5f9c1e1178c21fc92,ca
3,03f5d5ec88161b9365bea549d7ce92cd,LSIL
...
```

| Colonne | Description |
|---------|-------------|
| `id` | Index séquentiel (1, 2, 3, ...) |
| `hash` | Identifiant unique (MD5), correspond au nom de fichier image |
| `class` | Classe Bethesda |

### Classes Bethesda

| Classe | Description | Mapping Binaire |
|--------|-------------|-----------------|
| `Negative` | Normal (NILM) | Normal |
| `ASC-US` | Atypical Squamous Cells of Undetermined Significance | **Abnormal** |
| `ASC-H` | Atypical Squamous Cells, cannot exclude HSIL | **Abnormal** |
| `LSIL` | Low-grade Squamous Intraepithelial Lesion | **Abnormal** |
| `ca` | Carcinoma | **Abnormal** (Critical) |

> **Note:** Pas de HSIL ni SCC explicites dans ce subset.

---

## Données Eye-Tracking — Structure Détaillée

### fixation_locs/ (Fichiers MATLAB .mat)

**Format:** MATLAB v5 mat-file (little endian)

**Structure du fichier .mat:**

```python
import scipy.io as sio

data = sio.loadmat('fixation_locs/011fda505d7e4af4b8cc57545343624d.mat')

# Clés disponibles:
# - '__header__': Métadonnées MATLAB
# - '__version__': Version du format
# - '__globals__': Variables globales
# - 'fixLocs': DONNÉES PRINCIPALES

# Structure de fixLocs:
data['fixLocs'].shape  # → (960, 1280) = dimensions image
data['fixLocs'].dtype  # → uint8
```

**Interprétation de `fixLocs`:**

```
fixLocs[y, x] = 0  →  Pas de fixation oculaire à ce pixel
fixLocs[y, x] > 0  →  Fixation oculaire détectée (intensité = durée/fréquence)
```

> **Important:** C'est une matrice 2D de la même taille que l'image (960×1280),
> pas une liste de coordonnées. Chaque pixel indique si le pathologiste a regardé
> cette zone de l'image.

### fixation_maps/ (PNG Grayscale)

Visualisation des mêmes données sous forme d'images:

| Propriété | Valeur |
|-----------|--------|
| Format | PNG 8-bit grayscale |
| Dimensions | 1280 × 960 (même que images source) |
| Valeurs | 0-255 (intensité de fixation) |

**Exemple de lecture:**

```python
from PIL import Image
import numpy as np

# Charger la heatmap
heatmap = np.array(Image.open('fixation_maps/011fda505d7e4af4b8cc57545343624d.png'))
# → shape: (960, 1280), dtype: uint8

# Les zones blanches = forte attention
# Les zones noires = pas d'attention
```

### Protocole Expérimental (Source: Publication)

| Aspect | Détail |
|--------|--------|
| **Équipement** | Eye Link 1000 (SR Research Ltd., Canada) |
| **Fréquence** | 1000 Hz |
| **Œil enregistré** | Droit uniquement |
| **Participants** | 3 cytopathologistes certifiés |
| **Tâche** | Interpréter l'image + cliquer sur cellules anormales |
| **Temps** | Libre (pas de limite) |

### Usage Potentiel (R&D Avancé)

1. **Attention-Guided Training:**
   - Pondérer les régions "importantes" dans la loss function
   - Les zones à haute fixation = régions diagnostiques critiques

2. **Validation de Saillance:**
   - Comparer où le modèle "regarde" vs où l'expert regarde
   - Grad-CAM vs fixation_maps

3. **Augmentation Guidée:**
   - Cropper autour des zones à haute attention
   - Générer des données d'entraînement ciblées

---

## Comparaison avec Autres Datasets

| Aspect | CRICVA | APCData | SIPaKMeD |
|--------|--------|---------|----------|
| **Images** | 232 | 425 | 4,049 |
| **Cellules annotées** | ❌ Non | ✅ 3,619 | ✅ 1 par image |
| **Type annotation** | Eye-tracking | Points (x, y) | Masques complets |
| **Coordonnées cellules** | ❌ Non | ✅ Oui | ✅ Oui (masques) |
| **Classes** | 5 | 6 | 7 |
| **Résolution** | 1280×960 | 2048×1532 | ~150×150 (variable) |
| **Méthode préparation** | Pap conventionnel | LBC | Pap conventionnel |
| **Usage CellPose** | ❌ Non | ✅ **Recommandé** | ⚠️ Sur-segmente |
| **Usage Classification** | Image-level | Cell-level | Cell-level |
| **Multi-cellules/image** | ✅ Oui | ✅ Oui | ❌ Non (isolées) |

### Recommandation V14 Pipeline

| Phase | Dataset | Raison |
|-------|---------|--------|
| **POC (Phase 1)** | SIPaKMeD | Masques GT, cellules isolées |
| **CellPose Validation (Phase 2)** | **APCData** | Coordonnées cellules, multi-cellules |
| **R&D Attention** | CRICVA | Eye-tracking pathologistes |

---

## Utilisation dans V14 Pipeline

### Recommandation

| Phase | Usage CRICVA | Priorité |
|-------|--------------|----------|
| CellPose Validation | ❌ Impossible (pas de GT cellules) | - |
| Classification Image-Level | ✅ Possible | Basse |
| Attention-Guided Training | 🔬 R&D future | Optionnel |

### Script de Validation (Image-Level)

Si besoin de valider la classification au niveau image:

```bash
# Hypothétique - à créer si nécessaire
python scripts/cytology/validate_image_classification.py \
    --data_dir data/raw/CRICVA/CRICVA \
    --model_checkpoint models/checkpoints_v14_cytology/best_model.pth
```

---

## Conclusion

**CRICVA n'est PAS adapté pour valider CellPose** car il ne contient pas de coordonnées de cellules.

**Pour la validation CellPose, utiliser:**
1. **APCData** (3,619 cellules avec coordonnées) ← Recommandé
2. **CRIC Cervix complet** (si disponible avec annotations cellulaires)

**CRICVA peut être utilisé pour:**
- Validation classification image-level (232 images)
- Recherche sur l'attention visuelle (eye-tracking)

---

## Références

### Sources Officielles

- **Site officiel:** https://sites.google.com/view/cricvadataset
- **Mendeley Data:** https://data.mendeley.com/datasets/bk45c9yxb9/1
- **CRIC Database:** https://database.cric.com.br/

### Publication Associée

> **"Saliency-driven system models for cell analysis with deep learning"**
> DOI: https://doi.org/10.1016/j.cmpb.2019.105053
> Computer Methods and Programs in Biomedicine, 2019

### Contact

- Daniel Ferreira: daniels@ifce.edu.br

---

## Résumé Exécutif

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CRICVA — VERDICT FINAL                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ✅ CE QUE C'EST:                                                           │
│     • Dataset d'eye-tracking (attention visuelle)                           │
│     • 232 images avec labels Bethesda (image-level)                         │
│     • Heatmaps de fixation oculaire (3 pathologistes)                       │
│                                                                              │
│  ❌ CE QUE CE N'EST PAS:                                                    │
│     • PAS de coordonnées de cellules                                        │
│     • PAS de masques de segmentation                                        │
│     • PAS utilisable pour valider CellPose                                  │
│                                                                              │
│  🎯 POUR V14 CELLPOSE VALIDATION:                                           │
│     → Utiliser APCData (3,619 cellules avec coordonnées)                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Documentation mise à jour le 2026-01-21 après analyse complète des fichiers .mat*

