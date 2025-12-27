# V13 Smart Crops Strategy — CTO Validated

## Contexte

Suite aux résultats V13-Hybrid POC (Dice 0.7066 vs V12 0.9542), le CTO a recommandé :
1. **Conserver H-optimus-0 + Crops 224×224** (architecture validée)
2. **Ajouter rotations déterministes** pour diversité maximale
3. **Split-first-then-rotate** pour prévenir data leakage

**Objectif:** Maximiser diversité sans volume explosion (5 crops par image).

---

## Architecture de Crops (CTO Validated)

### Stratégie : 5 Perspectives Complémentaires

Chaque image 256×256 génère **5 crops 224×224** avec rotations déterministes :

```
Image PanNuke 256×256
    │
    ├─ Crop CENTRE (16, 16) → Rotation 0° (référence)
    │
    ├─ Crop COIN Haut-Gauche (0, 0) → Rotation 90° clockwise
    │
    ├─ Crop COIN Haut-Droit (32, 0) → Rotation 180°
    │
    ├─ Crop COIN Bas-Gauche (0, 32) → Rotation 270° clockwise
    │
    └─ Crop COIN Bas-Droit (32, 32) → Flip horizontal
```

### Matrice de Correspondance Formelle

**Standard Image:** $(x, y)$ où $(0, 0)$ = coin haut-gauche

| Position du Crop | Coordonnées $(x_1, y_1)$ | Zone extraite 224×224 | Rotation à appliquer |
|------------------|--------------------------|----------------------|---------------------|
| **CENTRE** | $(16, 16)$ | $[16:240, 16:240]$ | **0°** (Original) |
| **HAUT-GAUCHE** | $(0, 0)$ | $[0:224, 0:224]$ | **90° CW** |
| **HAUT-DROIT** | $(32, 0)$ | $[32:256, 0:224]$ | **180°** |
| **BAS-GAUCHE** | $(0, 32)$ | $[0:224, 32:256]$ | **270° CW** |
| **BAS-DROIT** | $(32, 32)$ | $[32:256, 32:256]$ | **Flip Horizontal** |

**Propriétés géométriques:**
- Recouvrement intelligent: déplacement de 32 pixels couvre 100% des pixels du patch 256×256
- Diversité orientation: chaque coin a une signature géométrique unique
- Zero data leakage: split train/val AVANT extraction de crops

### Bénéfices Scientifiques

| Crop | Perspective | Bénéfice |
|------|-------------|----------|
| Centre 0° | Vue de référence | Cohérence avec V13 POC |
| TL 90° | Structures verticales → horizontales | Détection glandes/cryptes multidirectionnelles |
| TR 180° | Vue inversée | Invariance orientation noyaux |
| BL 270° | Structures horizontales → verticales | Robustesse angles |
| BR Flip | Symétrie gauche-droite | Invariance latérale |

**Comparaison avec V13 POC Multi-Crop:**
- V13 POC : 5 crops aléatoires (centre + 4 positions variées)
- **V13 Smart Crops : 5 crops déterministes + rotations stratégiques**
- Gain attendu : +10-15% diversité grâce aux rotations

---

## Prévention Data Leakage — CRITIQUE

### ⚠️ Problème si on rotate AVANT split

```
❌ MAUVAIS WORKFLOW (data leakage possible):

1. Load 2514 images PanNuke
2. Apply 5 crops rotation → 12,570 crops
3. Split train/val 80/20 → 10,056 train / 2,514 val

PROBLÈME: Une image source peut avoir crop A en train et crop B (rotation de A) en val
→ Le modèle "voit" indirectement les données de validation
→ Métriques gonflées artificiellement
```

### ✅ Solution : Split-First-Then-Rotate

```
✅ CORRECT WORKFLOW (zero leakage):

1. Load 2514 images PanNuke
2. SPLIT by patient (80/20) → 2011 train sources / 503 val sources
3. Apply 5 crops rotation to TRAIN → 10,055 train crops
4. Apply 5 crops rotation to VAL → 2,515 val crops

GARANTIE: Aucune image source partagée entre train et val
→ Les rotations sont appliquées APRÈS séparation
→ Validation 100% indépendante
```

**Citation CTO:**
> "Attention, pour moi on fait la séparation en 2 dataset, train et val, ensuite on applique la rotation sur chaque dataset, comme ça nous sommes sur de na pas avoir une image sur les 2 dataset, même avec une rotation différentes."

---

## HV Maps Rotation — Transformations Vectorielles

### Problématique

Les HV maps ne sont PAS de simples images — ce sont des **champs vectoriels** encodant (H, V) = distance normalisée au centre du noyau.

**Rotation spatiale ≠ Rotation vectorielle**

```
Exemple: Noyau à (100, 100) avec vecteur HV = (0.5, 0.3)

Après rotation 90° clockwise:
- Position spatiale: (100, 100) → (124, 100)  [rotation image]
- Vecteur HV: (0.5, 0.3) → (0.3, -0.5)  [swapping composantes!]
                H     V      V'    -H'
```

### Transformations Correctes

| Transform | Composantes HV | Formule |
|-----------|----------------|---------|
| **0° (identité)** | H' = H, V' = V | Aucun changement |
| **90° clockwise** | H' = V, V' = -H | Rotation horaire vecteur |
| **180°** | H' = -H, V' = -V | Inversion complète |
| **270° clockwise** | H' = -V, V' = H | Rotation anti-horaire vecteur |
| **Flip horizontal** | H' = -H, V' = V | Inversion axe X uniquement |

### Implémentation avec Albumentations

**Albumentations** (recommandé CTO) gère la rotation spatiale MAIS pas le swapping vectoriel automatiquement.

**Solution implémentée:**

```python
# Step 1: Albumentations rotate spatially (image + masks)
transform = A.Compose([
    A.Rotate(limit=(90, 90), p=1.0)  # Rotate 90° clockwise
], additional_targets={
    'mask_np': 'mask',
    'mask_hv': 'image',  # Traité comme image (preserve float32)
    'mask_nt': 'mask'
})

transformed = transform(
    image=image_crop,
    mask_np=np_crop,
    mask_hv=hv_crop,  # (224, 224, 2)
    mask_nt=nt_crop
)

# Step 2: Correct HV component swapping AFTER spatial rotation
hv_rotated = correct_hv_after_rotation(transformed['mask_hv'], angle=90)
# Applies: H' = V, V' = -H

# Step 3: Verify divergence is negative (vectors point inward)
div = compute_hv_divergence(hv_rotated, np_mask)
assert div < 0, "HV vectors should point INWARD to nucleus center"
```

**Fonction de correction:**
```python
def correct_hv_after_rotation(hv_map, rotation_angle):
    h_comp = hv_map[:, :, 0]
    v_comp = hv_map[:, :, 1]

    if rotation_angle == 90:
        new_h = v_comp
        new_v = -h_comp
    elif rotation_angle == 180:
        new_h = -h_comp
        new_v = -v_comp
    elif rotation_angle == 270:
        new_h = -v_comp
        new_v = h_comp

    return np.stack([new_h, new_v], axis=2)
```

---

## Bibliothèques de Référence (CTO Recommendation)

### 1. Albumentations ⭐ CHOISI

**Pourquoi (CTO):**
> "C'est la bibliothèque la plus rapide et la plus flexible. Elle permet d'appliquer la même rotation simultanément à l'image RGB, au masque de segmentation (NP) et aux cartes de gradients (HV)."

**Avantages:**
- ✅ Rotations 90°/180°/270° sans interpolation (pixel-perfect)
- ✅ `additional_targets` pour synchroniser image + NP + HV + NT
- ✅ Preserve float32 pour HV maps (pas de clipping)
- ✅ Standard industriel (HoVer-Net, CoNIC winners)

**Installation:**
```bash
pip install albumentations
```

### 2. MONAI (Alternative médical-spécifique)

**Pourquoi:**
> "Développée par NVIDIA et le King's College London, elle est spécifiquement conçue pour l'imagerie médicale."

**Usage:** Si besoin de transformations 3D ou formats DICOM/NIfTI.

### 3. Torchvision (Non recommandé ici)

**Limitation:**
> "Elle est parfois plus rigide pour synchroniser des rotations complexes sur plusieurs 'targets' (comme vos cartes HV qui sont des vecteurs)."

---

## Pipeline Complet — Étapes d'Exécution

### Prérequis

1. **Données sources FIXED** (HV float32 [-1, 1])
   ```bash
   ls data/family_FIXED/epidermal_data_FIXED.npz
   ```

2. **Albumentations installé**
   ```bash
   pip install albumentations
   ```

### Étape 1: Préparation Smart Crops (5 min)

```bash
python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal
```

**Outputs:**
```
data/family_data_v13_smart_crops/
├── epidermal_train_v13_smart_crops.npz  (~10,055 crops)
└── epidermal_val_v13_smart_crops.npz    (~2,515 crops)
```

**Logs attendus:**
```
Loading source data: data/family_FIXED/epidermal_data_FIXED.npz
Loaded 2514 samples for family 'epidermal'
✅ HV targets validated (float32, range [-1, 1])

Splitting by patient (80% train / 20% val)...
  Train: 2011 samples
  Val:   503 samples

Applying 5 strategic crops to TRAIN dataset...
  Processed 100/2011 samples...
  ...
  Train amplified: 10055 crops

Applying 5 strategic crops to VAL dataset...
  Val amplified: 2515 crops

✅ V13 SMART CROPS DATA PREPARATION COMPLETE
Family:       epidermal
Train:        10055 crops (from 2011 sources)
Val:          2515 crops (from 503 sources)
Amplification: 5× (centre + 4 corners with rotations)
Data leakage: PREVENTED (split-first-then-rotate)
```

### Étape 2: Validation HV Rotation (2 min)

```bash
python scripts/validation/validate_hv_rotation.py \
    --data_file data/family_data_v13_smart_crops/epidermal_train_v13_smart_crops.npz \
    --n_samples 5
```

**Critères de validation:**

| Métrique | Cible | Signification |
|----------|-------|---------------|
| Range valid | 100% | HV values ∈ [-1, 1] |
| Divergence mean | < 0 | Vecteurs pointent VERS centre (inward) |
| Divergence negative | ~100% | Cohérence sur tous les crops |

**Outputs:**
```
results/hv_validation/
├── hv_validation_sample_000.png  (5 crops visualisés)
├── hv_validation_sample_001.png
└── ...
```

**Exemple de visualisation:**
- Flèches jaunes = vecteurs HV
- Flèches doivent pointer VERS centres noyaux (pas vers l'extérieur)
- Centre 0° vs rotations 90°/180°/270° doivent être cohérents

### Étape 3: Extraction Features H-optimus-0 (10 min)

```bash
# Train features
python scripts/preprocessing/extract_features_from_fixed.py \
    --input_file data/family_data_v13_smart_crops/epidermal_train_v13_smart_crops.npz \
    --output_dir data/cache/family_data \
    --family epidermal \
    --split train

# Val features
python scripts/preprocessing/extract_features_from_fixed.py \
    --input_file data/family_data_v13_smart_crops/epidermal_val_v13_smart_crops.npz \
    --output_dir data/cache/family_data \
    --family epidermal \
    --split val
```

**Outputs:**
```
data/cache/family_data/
├── epidermal_rgb_features_v13_smart_crops_train.npz  (~20 GB)
└── epidermal_rgb_features_v13_smart_crops_val.npz    (~5 GB)
```

### Étape 4: Training V13 Smart Crops (40 min)

```bash
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal \
    --epochs 30 \
    --batch_size 16 \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0
```

**Métriques cibles (validation):**
- NP Dice: > 0.90 (segmentation binaire)
- HV MSE: < 0.05 (gradients sharp)
- NT Acc: > 0.85 (classification 5 types)

### Étape 5: Évaluation AJI (5 min)

```bash
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_best.pth \
    --family epidermal \
    --n_samples 50
```

**Objectif AJI:** ≥ 0.68 (+18% vs V13 POC baseline 0.57)

---

## Comparaison Architectures

| Version | Crops | Rotations | Split Strategy | Data Leakage | AJI (epidermal) |
|---------|-------|-----------|----------------|--------------|-----------------|
| V12 | Resize 256→224 | None | 80/20 patient | ✅ None | 0.57* (train data) |
| V13 POC Multi-Crop | 5 random | None | 80/20 patient | ✅ None | 0.57* (train data) |
| **V13 Smart Crops** | **5 strategic** | **90°/180°/270°/flip** | **Split-first** | ✅ **None** | **≥0.68** 🎯 |

*Note: AJI 0.57 était mesuré sur données d'entraînement (invalidé).

---

## Avantages Scientifiques

### 1. Diversité Maximale

Les 5 crops couvrent **toutes les perspectives spatiales** :
- Centre : Région principale
- Coins : Zones périphériques (souvent riches en structures glandulaires)
- Rotations : Invariance orientation (cryptes verticales/horizontales/obliques)

### 2. Volume Contrôlé

**V13 Smart Crops : 5× amplification** (même volume que V13 POC)

Comparaison avec approche naïve :
- ❌ 5 crops × 4 rotations = 20× amplification (explosion volume, overfitting)
- ✅ 5 crops avec rotations déterministes = 5× amplification (optimal)

### 3. Prévention Overfitting

**Split-first garantit** :
- Train et Val sont **patients différents**
- Aucune fuite d'information via rotations
- Métriques validation = vraie généralisation

### 4. Littérature Validée

**CoNIC Challenge Winners** (2022) utilisent :
- Rotations 90°/180°/270° déterministes
- HorizontalFlip / VerticalFlip
- Split patient-based

**HoVer-Net** (Graham et al. 2019) :
- RandomRotate90 pendant training
- Sign inversion pour HV maps lors des flips

---

## Troubleshooting

### Erreur: "HV values outside [-1, 1]"

**Cause:** Source data utilise HV int8 (Bug #3)

**Solution:**
```bash
# Re-générer données FIXED avec float32
python scripts/preprocessing/prepare_family_data_FIXED.py --family epidermal
```

### Erreur: "Divergence positive"

**Cause:** HV component swapping incorrect (vecteurs pointent vers l'extérieur)

**Diagnostic:**
```bash
python scripts/validation/validate_hv_rotation.py \
    --data_file data/family_data_v13_smart_crops/epidermal_train_v13_smart_crops.npz \
    --n_samples 5
```

**Solution:** Vérifier `correct_hv_after_rotation()` dans `prepare_v13_smart_crops.py`

### Erreur: "FileNotFoundError: epidermal_data_FIXED.npz"

**Cause:** Données sources manquantes

**Solution:**
```bash
# Générer données FIXED depuis PanNuke
python scripts/preprocessing/prepare_family_data_FIXED.py --family epidermal
```

---

## Références

### Littérature Scientifique

1. **HoVer-Net** (Graham et al., 2019)
   - Paper: "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images"
   - Rotation strategy: RandomRotate90 + sign inversion for HV maps

2. **CoNIC Challenge** (2022)
   - Winners utilisent rotations déterministes 90°/180°/270°
   - Patient-based split pour prévenir data leakage

3. **Albumentations** (Buslaev et al., 2020)
   - Paper: "Albumentations: Fast and Flexible Image Augmentations"
   - Standard industriel pour medical imaging

### Code Repositories

- HoVer-Net official: https://github.com/vqdang/hover_net
- Albumentations: https://github.com/albumentations-team/albumentations
- MONAI: https://github.com/Project-MONAI/MONAI

---

## Métriques Attendues

### Comparaison V13 POC vs V13 Smart Crops

| Métrique | V13 POC Multi-Crop | V13 Smart Crops (cible) | Amélioration |
|----------|-------------------|------------------------|--------------|
| **Dice** | 0.95 | >0.90 | Maintenu |
| **AJI** | 0.57* (train data) | **≥0.68** | **+18%** 🎯 |
| **HV MSE** | 0.03 | <0.05 | Maintenu/Amélioré |
| **NT Acc** | 0.88 | >0.85 | Maintenu |
| **Data leakage** | None | **None** ✅ | Garanti |

*Note: AJI 0.57 invalidé car mesuré sur données d'entraînement.

### Temps Estimé Pipeline Complet

| Étape | Durée | GPU |
|-------|-------|-----|
| Smart crops preparation | 5 min | No |
| HV validation | 2 min | No |
| Features extraction | 10 min | Yes |
| Training (30 epochs) | 40 min | Yes |
| AJI evaluation | 5 min | Yes |
| **Total** | **~1h** | - |

---

## Conclusion

La stratégie **V13 Smart Crops** combine :
- ✅ Architecture validée (H-optimus-0 + crops 224×224)
- ✅ Diversité maximale (5 perspectives + rotations déterministes)
- ✅ Prévention data leakage (split-first-then-rotate)
- ✅ Transformations HV correctes (component swapping via Albumentations)

**Objectif:** Atteindre **AJI ≥0.68** (+18% vs baseline) sur données de validation indépendantes.

**Validé par:** CTO + Littérature scientifique (HoVer-Net, CoNIC Challenge)
