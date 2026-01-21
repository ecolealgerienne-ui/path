# CRICVA Dataset — Documentation

> **Version:** 1.0
> **Date:** 2026-01-21
> **Source:** CRIC Cervix Database (Visual Attention subset)
> **URL:** https://database.cric.com.br/

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
| **Annotations** | Labels par image (pas de coordonnées cellule) |

---

## ⚠️ Limitation Critique

> **CRICVA ≠ Dataset de segmentation cellulaire**
>
> Ce dataset contient des **données d'eye-tracking** (où les observateurs regardent),
> PAS des annotations de localisation des cellules.
>
> **Utilisation possible:**
> - Validation classification (image-level labels)
> - Recherche sur l'attention visuelle des pathologistes
>
> **NON utilisable pour:**
> - Validation CellPose (pas de coordonnées GT)
> - Entraînement segmentation

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

## Données Eye-Tracking

### fixation_locs/

Coordonnées des points de fixation oculaire des observateurs humains.

### fixation_maps/

Heatmaps de densité d'attention visuelle (où les pathologistes regardent le plus).

**Usage potentiel (R&D avancé):**
- Entraîner un modèle d'attention guidée par l'expert
- Pondérer les régions "importantes" dans les images

---

## Comparaison avec APCData

| Aspect | CRICVA | APCData |
|--------|--------|---------|
| **Images** | 232 | 425 |
| **Cellules annotées** | ❌ Non | ✅ 3,619 |
| **Coordonnées** | ❌ Non | ✅ (nucleus_x, nucleus_y) |
| **Classes** | 5 | 6 |
| **Résolution** | 1280×960 | 2048×1532 |
| **Usage CellPose** | ❌ Non | ✅ Oui |
| **Usage Classification** | ✅ Image-level | ✅ Cell-level |

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

- CRIC Database: https://database.cric.com.br/
- Publication: "CRIC Searchable Image Database for Cervical Cytopathology Research"

---

*Documentation générée le 2026-01-21*
