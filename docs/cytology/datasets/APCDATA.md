# APCData — Cervical Cytology Dataset

> **Source:** Mendeley Data
> **URL:** https://data.mendeley.com/datasets/ytd568rh3p/1
> **Téléchargé:** 2026-01-21
> **Usage V14:** Train principal (LBC = proxy Dubai)

---

## 📋 Vue d'Ensemble

| Attribut | Valeur |
|----------|--------|
| **Nom** | APCData (Anatomical Pathology and Cytology) |
| **Origine** | Laboratoire APC, Rivera, Uruguay |
| **Méthode** | **LBC (Liquid-Based Cytology)** par cytocentrifugation |
| **Période** | 2018-2021 |
| **Lames** | 73 études Pap smear diagnostiquées |
| **Images** | 425 images |
| **Résolution** | 2048 × 1532 pixels |
| **Cellules** | **3,619 cellules annotées** |
| **Magnification** | Non spécifié (estimé 20-40x) |

---

## 📁 Structure des Données

```
data/raw/apcdata/
├── APCData_YOLO/                    # Format YOLO (bounding boxes)
│   ├── classes.txt                  # 6 classes Bethesda
│   ├── images/                      # 425 images PNG
│   │   ├── 0a1b2c3d4e5f6789.png
│   │   └── ...
│   └── labels/                      # 425 fichiers .txt
│       ├── 0a1b2c3d4e5f6789.txt    # YOLO format
│       └── ...
│
└── APCData_points/                  # Format Points (coordonnées noyaux)
    ├── images/                      # 425 images PNG (identiques)
    └── labels/
        ├── csv/                     # 425 fichiers CSV
        │   └── {image_name}.csv
        └── json/                    # 420 fichiers JSON
            └── {image_name}.json
```

---

## 🏷️ Classes (Système Bethesda)

| Index | Code | Nom Complet | Catégorie | Nombre |
|-------|------|-------------|-----------|--------|
| 0 | NILM | Negative for Intraepithelial Lesion or Malignancy | **Normal** | 2,114 |
| 1 | ASCUS | Atypical Squamous Cells of Undetermined Significance | Atypique | 333 |
| 2 | ASCH | Atypical Squamous Cells, cannot exclude HSIL | Atypique | 182 |
| 3 | LSIL | Low-grade Squamous Intraepithelial Lesion | **Abnormal** | 444 |
| 4 | HSIL | High-grade Squamous Intraepithelial Lesion | **Abnormal** | 421 |
| 5 | SCC | Squamous Cell Carcinoma | **Malin** | 125 |

### Distribution des Classes

```
NILM   ████████████████████████████████████████  2,114 (58.4%)
LSIL   ████████                                    444 (12.3%)
HSIL   ███████                                     421 (11.6%)
ASCUS  █████                                       333 (9.2%)
ASCH   ███                                         182 (5.0%)
SCC    ██                                          125 (3.5%)
       ─────────────────────────────────────────
TOTAL                                            3,619 (100%)
```

---

## 📄 Format des Annotations

### Format CSV (APCData_points/labels/csv/)

```csv
image_id,image_filename,image_doi,cell_id,bethesda_system,nucleus_x,nucleus_y
425,49a2215c2453312c.png,null,11685,Negative,886,67
425,49a2215c2453312c.png,null,11686,Negative,510,376
425,49a2215c2453312c.png,null,11687,Negative,716,281
```

| Colonne | Description |
|---------|-------------|
| `image_id` | ID unique de l'image |
| `image_filename` | Nom du fichier PNG |
| `image_doi` | DOI (souvent null) |
| `cell_id` | ID unique de la cellule |
| `bethesda_system` | Classe Bethesda (Negative, ASCUS, LSIL, etc.) |
| `nucleus_x` | Coordonnée X du centre du noyau |
| `nucleus_y` | Coordonnée Y du centre du noyau |

### Format JSON (APCData_points/labels/json/)

```json
[
  {
    "image_id": 444,
    "image_doi": null,
    "image_name": "624a7d611524fe5e.png",
    "classifications": [
      {
        "cell_id": 11826,
        "bethesda_system": "Negative",
        "nucleus_x": 1047,
        "nucleus_y": 70
      },
      {
        "cell_id": 11832,
        "bethesda_system": "LSIL",
        "nucleus_x": 1567,
        "nucleus_y": 379
      }
    ]
  }
]
```

### Format YOLO (APCData_YOLO/labels/)

```
# Format: class_id x_center y_center width height (normalisé 0-1)
0 0.432617 0.043732 0.045898 0.065274
0 0.249023 0.245430 0.041016 0.058824
3 0.765137 0.247389 0.037109 0.052288
```

**classes.txt:**
```
NILM
ASCUS
ASCH
LSIL
HSIL
SCC
```

---

## 🔄 Mapping vers SIPaKMeD

| APCData | Code | SIPaKMeD Equivalent | Catégorie V14 |
|---------|------|---------------------|---------------|
| NILM | 0 | normal_* | Normal |
| ASCUS | 1 | light_dysplastic | Abnormal (Low) |
| LSIL | 3 | light_dysplastic | Abnormal (Low) |
| ASCH | 2 | moderate_dysplastic | Abnormal (Mid) |
| HSIL | 4 | severe_dysplastic | Abnormal (High) |
| SCC | 5 | carcinoma_in_situ | Abnormal (Malin) |

### Mapping Binaire (Safety First)

| APCData | Catégorie Binaire |
|---------|-------------------|
| NILM | **Normal** |
| ASCUS, ASCH, LSIL, HSIL, SCC | **Abnormal** |

---

## 🎯 Utilisation pour V14

### Avantages

1. **LBC (Liquid-Based Cytology)** — Fond propre, identique aux préparations Urine/Thyroïde modernes
2. **Annotations point** — Coordonnées exactes des noyaux pour crop 224×224
3. **Multi-cellules par image** — Simule conditions cliniques réelles
4. **6 classes Bethesda** — Classification standard internationale
5. **Volume** — 3,619 cellules annotées

### Pipeline d'Intégration

```
APCData Image (2048×1532)
    │
    ├── Charger annotations JSON/CSV
    │
    ├── Pour chaque cellule annotée:
    │   ├── Extraire patch 224×224 centré sur (nucleus_x, nucleus_y)
    │   ├── Padding blanc si bord d'image
    │   └── Sauvegarder avec label
    │
    └── Output: Patches 224×224 + labels (format SIPaKMeD-compatible)
```

### Script de Preprocessing

```bash
python scripts/cytology/05_preprocess_apcdata.py \
    --raw_dir data/raw/apcdata/APCData_points \
    --output_dir data/processed/apcdata \
    --patch_size 224
```

---

## ⚠️ Points d'Attention

### 1. Classe "Negative" vs "NILM"

Dans les annotations CSV/JSON, la classe est notée `"Negative"` (pas `"NILM"`).

```python
# Mapping à appliquer
CLASS_MAPPING = {
    "Negative": "NILM",
    "ASCUS": "ASCUS",
    "ASCH": "ASCH",
    "LSIL": "LSIL",
    "HSIL": "HSIL",
    "SCC": "SCC"
}
```

### 2. Cellules au Bord

Certaines cellules ont des coordonnées proches des bords de l'image:
- `nucleus_x < 112` ou `nucleus_x > 1936` (bord gauche/droit)
- `nucleus_y < 112` ou `nucleus_y > 1420` (bord haut/bas)

**Solution:** Padding blanc (comme SIPaKMeD)

### 3. Déséquilibre des Classes

- NILM domine (58.4%)
- SCC minoritaire (3.5%)

**Solution:** Focal Loss + Class Weights (déjà implémenté)

---

## 📚 Références

- **Paper:** "APCData: A benchmark dataset for cervical cytology cell analysis"
- **Mendeley:** https://data.mendeley.com/datasets/ytd568rh3p/1
- **Licence:** CC BY 4.0

---

## 📊 Comparaison avec SIPaKMeD

| Aspect | SIPaKMeD | APCData |
|--------|----------|---------|
| **Méthode** | Pap conventionnel | **LBC** |
| **Format** | Cellules isolées | Multi-cellules/image |
| **Cellules** | 917 | 3,619 |
| **Classes** | 7 (granulaire) | 6 (Bethesda) |
| **Annotations** | Masques | Points (x,y) |
| **Résolution** | Variable (petites) | 2048×1532 |
| **Fond** | Variable | **Propre (LBC)** |

**Complémentarité:**
- SIPaKMeD = Validation sur cellules isolées (Phase 1 POC) ✅
- APCData = Entraînement robuste avec LBC (Phase 2 Production)

---

*Documentation générée le 2026-01-21*
