# Session Continuation Prompt — V13 Smart Crops

## Pour Claude dans la prochaine session

Bonjour Claude,

Cette session continue le projet **CellViT-Optimus** (système d'assistance au triage histopathologique). Voici le contexte complet du travail effectué et ce qui reste à faire.

---

## 📋 Contexte Immédiat

### Session Précédente (2025-12-27)

Nous avons implémenté la **stratégie V13 Smart Crops** validée par le CTO, pour atteindre **AJI ≥0.68** (+18% vs baseline).

**Problème résolu:**
- V13-Hybrid a échoué (Dice 0.7066 vs V12 0.9542 -26% dégradation)
- Gated Fusion freeze (gate α=0.1192-0.1196, gradient vanishing)
- AJI 0.57 était mesuré sur données **d'entraînement** (invalidé)

**Solution implémentée:**
- Architecture validée : H-optimus-0 + crops 224×224
- **5 crops stratégiques** avec rotations déterministes (90°/180°/270°/flip)
- **Split-first-then-rotate** pour prévenir data leakage
- **Albumentations** pour transformations synchronisées (CTO recommandé)

---

## 🎯 Objectif Actuel

**Exécuter le pipeline V13 Smart Crops complet** pour atteindre AJI ≥0.68 sur données de validation indépendantes.

**Métriques cibles:**

| Métrique | V13 POC | V13 Smart Crops (cible) | Amélioration |
|----------|---------|-------------------------|--------------|
| Dice | 0.95 | >0.90 | Maintenu |
| **AJI** | 0.57* (train data) | **≥0.68** | **+18%** 🎯 |
| HV MSE | 0.03 | <0.05 | Maintenu |
| NT Acc | 0.88 | >0.85 | Maintenu |

*Note: AJI 0.57 invalidé car mesuré sur données d'entraînement.

---

## 📁 Fichiers Créés (Session Précédente)

### Scripts Implémentés

| Fichier | Rôle | Statut |
|---------|------|--------|
| `scripts/preprocessing/prepare_v13_smart_crops.py` | Génération 5 crops + rotations avec split-first (430 lignes) | ✅ Créé |
| `scripts/validation/validate_hv_rotation.py` | Validation divergence HV < 0 (280 lignes) | ✅ Créé |
| `docs/V13_SMART_CROPS_STRATEGY.md` | Documentation complète CTO-validée (600 lignes) | ✅ Créé |

### Localisation Code

```
cellvit-optimus/
├── scripts/
│   ├── preprocessing/
│   │   └── prepare_v13_smart_crops.py  ← Génère train/val séparés
│   └── validation/
│       └── validate_hv_rotation.py     ← Vérifie divergence HV
├── docs/
│   ├── V13_SMART_CROPS_STRATEGY.md    ← Guide complet
│   └── SESSION_CONTINUATION_PROMPT.md ← Ce fichier
└── CLAUDE.md                          ← Journal mis à jour
```

---

## 🔧 Détails Techniques Critiques

### Architecture 5 Crops (À NE PAS MODIFIER)

```
Image PanNuke 256×256
    ├─ Crop CENTRE (16, 16) → Rotation 0° (référence)
    ├─ Crop COIN Haut-Gauche (0, 0) → Rotation 90° clockwise
    ├─ Crop COIN Haut-Droit (0, 32) → Rotation 180°
    ├─ Crop COIN Bas-Gauche (32, 0) → Rotation 270° clockwise
    └─ Crop COIN Bas-Droit (32, 32) → Flip horizontal
```

### HV Maps Rotation (Transformations Vectorielles)

**⚠️ CRITIQUE:** HV maps sont des **champs vectoriels**, pas des images simples.

| Transform | Composantes HV | Formule |
|-----------|----------------|---------|
| 90° CW | H' = V, V' = -H | Rotation horaire vecteur |
| 180° | H' = -H, V' = -V | Inversion complète |
| 270° CW | H' = -V, V' = H | Rotation anti-horaire vecteur |
| Flip H | H' = -H, V' = V | Inversion axe X uniquement |

**Implémentation dans `prepare_v13_smart_crops.py`:**

```python
# Step 1: Albumentations rotate spatially
transform = A.Compose([
    A.Rotate(limit=(90, 90), p=1.0)
], additional_targets={'mask_hv': 'image'})

# Step 2: Correct HV component swapping
hv_corrected = correct_hv_after_rotation(transformed['mask_hv'], angle=90)

# Step 3: Verify divergence < 0 (vectors point inward)
div = compute_hv_divergence(hv_corrected, np_mask)
assert div < 0
```

### Split-First-Then-Rotate Workflow

**⚠️ CRITIQUE pour prévenir data leakage:**

```python
# 1. Split FIRST by patient (80/20)
train_data, val_data = split_by_patient(images, masks, source_ids, ratio=0.8, seed=42)

# 2. Apply 5 crops rotation to TRAIN separately
train_crops = amplify_with_crops(train_data)  # 2011 → 10,055 crops

# 3. Apply 5 crops rotation to VAL separately
val_crops = amplify_with_crops(val_data)  # 503 → 2,515 crops

# GARANTIE: Aucune image source partagée entre train et val
```

---

## 🚀 Pipeline Complet (À Exécuter)

### Prérequis

1. **Données sources FIXED** (HV float32 [-1, 1])
   ```bash
   ls data/family_FIXED/epidermal_data_FIXED.npz
   ```

2. **Albumentations installé**
   ```bash
   pip install albumentations
   ```

### Étapes d'Exécution

**Étape 1: Préparation Smart Crops (5 min)**

```bash
python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal
```

**Outputs attendus:**
```
data/family_data_v13_smart_crops/
├── epidermal_train_v13_smart_crops.npz  (10,055 crops)
└── epidermal_val_v13_smart_crops.npz    (2,515 crops)
```

**Logs critiques à vérifier:**
```
✅ HV targets validated (float32, range [-1, 1])
  Train: 2011 samples
  Val:   503 samples
  Train amplified: 10055 crops
  Val amplified: 2515 crops
Data leakage: PREVENTED (split-first-then-rotate)
```

**Étape 2: Validation HV Rotation (2 min)**

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

**Si échec:**
- Range invalid → Vérifier source data (HV dtype float32)
- Divergence positive → Bug dans `correct_hv_after_rotation()` (component swapping incorrect)

**Étape 3: Extraction Features H-optimus-0 (10 min)**

⚠️ **NOTE:** Ce script doit être créé ou adapté depuis `extract_features_from_fixed.py`

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

**Outputs attendus:**
```
data/cache/family_data/
├── epidermal_rgb_features_v13_smart_crops_train.npz  (~20 GB)
└── epidermal_rgb_features_v13_smart_crops_val.npz    (~5 GB)
```

**Étape 4: Training V13 Smart Crops (40 min)**

⚠️ **NOTE:** Ce script doit être créé ou adapté depuis `train_hovernet_family.py`

```bash
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal \
    --epochs 30 \
    --batch_size 16 \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0
```

**Métriques validation cibles:**
- NP Dice: > 0.90
- HV MSE: < 0.05
- NT Acc: > 0.85

**Étape 5: Évaluation AJI (5 min)**

⚠️ **NOTE:** Ce script doit être créé ou adapté depuis `test_v13_hybrid_aji.py`

```bash
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_best.pth \
    --family epidermal \
    --n_samples 50
```

**Objectif:** AJI ≥ 0.68

---

## ⚠️ Points de Vigilance

### 1. Données Sources Manquantes

**Symptôme:**
```
FileNotFoundError: data/family_FIXED/epidermal_data_FIXED.npz
```

**Solution:**
```bash
python scripts/preprocessing/prepare_family_data_FIXED.py --family epidermal
```

### 2. HV Divergence Positive

**Symptôme:**
```
Divergence mean: 0.15 (should be < 0)
```

**Cause:** Component swapping incorrect dans `correct_hv_after_rotation()`

**Vérification:**
```python
# Pour rotation 90° clockwise:
assert new_h == v_comp
assert new_v == -h_comp  # PAS h_comp !
```

### 3. Data Leakage Accidentel

**Symptôme:** Métriques validation trop élevées (Dice >0.98, AJI >0.75)

**Vérification:**
```python
# Train et val source_image_ids DOIVENT être disjoints
train_ids = set(train_source_ids)
val_ids = set(val_source_ids)
assert len(train_ids & val_ids) == 0, "Data leakage detected!"
```

---

## 📊 Scripts À Créer/Adapter

### Script 1: Features Extraction (Priorité Haute)

**Fichier:** `scripts/preprocessing/extract_features_from_fixed.py`

**Adaptations requises:**
- Support `--split train/val` parameter
- Load from `*_train_v13_smart_crops.npz` ou `*_val_v13_smart_crops.npz`
- Output naming: `{family}_rgb_features_v13_smart_crops_{split}.npz`

**Référence:** Voir `extract_features.py` existant pour logique H-optimus-0

### Script 2: Training (Priorité Haute)

**Fichier:** `scripts/training/train_hovernet_family_v13_smart_crops.py`

**Adaptations requises:**
- Load RGB features train + val séparés
- Load targets train + val séparés
- Dataset class supporte split explicite (pas 80/20 automatique)
- Checkpoint naming: `hovernet_{family}_v13_smart_crops_best.pth`

**Référence:** Voir `train_hovernet_family.py` existant

### Script 3: AJI Evaluation (Priorité Moyenne)

**Fichier:** `scripts/evaluation/test_v13_smart_crops_aji.py`

**Adaptations requises:**
- Load val split uniquement
- Watershed post-processing avec beta optimal (voir `test_v13_hybrid_aji.py`)

**Référence:** Voir `test_v13_hybrid_aji.py` pour logique AJI

---

## 📚 Documentation de Référence

### Fichiers à Lire

1. **`docs/V13_SMART_CROPS_STRATEGY.md`**
   - Guide complet (600 lignes)
   - Justifications scientifiques
   - Troubleshooting détaillé

2. **`CLAUDE.md` (Section Journal de Développement)**
   - Entry 2025-12-27: V13 Smart Crops Strategy
   - Leçons apprises
   - Comparaison architectures

3. **`scripts/preprocessing/prepare_v13_smart_crops.py`**
   - Code de référence pour split-first-then-rotate
   - Fonctions HV rotation à réutiliser

### Littérature Scientifique

- **HoVer-Net** (Graham et al., 2019): RandomRotate90 + HV sign inversion
- **CoNIC Challenge** (2022): Patient-based split + rotations déterministes
- **Albumentations** (Buslaev et al., 2020): Standard industriel medical imaging

---

## 🎯 Prochaines Actions Recommandées

### Action Immédiate (Si Données Sources OK)

**Scénario A: Données `epidermal_data_FIXED.npz` existantes**

```bash
# 1. Vérifier données sources
ls -lh data/family_FIXED/epidermal_data_FIXED.npz

# 2. Lancer pipeline complet
bash scripts/run_v13_smart_crops_pipeline.sh epidermal
```

**Scénario B: Données sources manquantes**

```bash
# 1. Générer données FIXED
python scripts/preprocessing/prepare_family_data_FIXED.py --family epidermal

# 2. Vérifier HV targets float32
python scripts/validation/diagnose_targets.py \
    --data_file data/family_FIXED/epidermal_data_FIXED.npz

# 3. Lancer pipeline V13 Smart Crops
bash scripts/run_v13_smart_crops_pipeline.sh epidermal
```

### Action Suivante (Après Validation Epidermal)

**Étendre aux 4 autres familles:**

```bash
for family in glandular digestive urologic respiratory; do
    python scripts/preprocessing/prepare_v13_smart_crops.py --family $family
    python scripts/validation/validate_hv_rotation.py \
        --data_file data/family_data_v13_smart_crops/${family}_train_v13_smart_crops.npz
    python scripts/preprocessing/extract_features_from_fixed.py --family $family --split train
    python scripts/preprocessing/extract_features_from_fixed.py --family $family --split val
    python scripts/training/train_hovernet_family_v13_smart_crops.py --family $family --epochs 30
    python scripts/evaluation/test_v13_smart_crops_aji.py --family $family --n_samples 50
done
```

---

## 📝 Questions pour l'Utilisateur

### Question 1: Localisation Données Sources

**Vérifier si ces fichiers existent:**
```bash
ls -lh data/family_FIXED/epidermal_data_FIXED.npz
ls -lh data/family_FIXED/glandular_data_FIXED.npz
ls -lh data/family_FIXED/digestive_data_FIXED.npz
ls -lh data/family_FIXED/urologic_data_FIXED.npz
ls -lh data/family_FIXED/respiratory_data_FIXED.npz
```

**Si manquant:** Lancer `prepare_family_data_FIXED.py` d'abord.

### Question 2: Scripts à Créer/Adapter

**Scripts manquants identifiés:**
- `extract_features_from_fixed.py` avec support `--split train/val`
- `train_hovernet_family_v13_smart_crops.py` avec split explicite
- `test_v13_smart_crops_aji.py` adapté pour V13 Smart Crops

**Dois-je créer ces scripts ou adapter les existants?**

### Question 3: Famille à Tester

**Recommandation:** Commencer par **epidermal** (574 samples, stress test).

**Raison:** Si fonctionne sur petite famille → fonctionne sur toutes.

---

## 🔑 Informations Clés à Retenir

### Architecture Validée

✅ **H-optimus-0 (gelé) + Crops 224×224** (pas de Gated Fusion)

### Stratégie Data Leakage

✅ **Split-first-then-rotate** (CTO validé)

### Bibliothèque Transformation

✅ **Albumentations** (standard industriel)

### HV Maps = Champs Vectoriels

✅ **Component swapping OBLIGATOIRE** après rotation spatiale

### Objectif Final

✅ **AJI ≥ 0.68** (+18% vs baseline 0.57 sur données validation indépendantes)

---

## 📌 Résumé en 3 Points

1. **Implémentation complète V13 Smart Crops** (5 crops stratégiques + rotations déterministes)
2. **3 scripts créés** (`prepare_v13_smart_crops.py`, `validate_hv_rotation.py`, documentation)
3. **Prochaine étape:** Exécuter pipeline complet (prep + validation + features + training + AJI)

---

**Temps estimé pipeline complet:** ~1h (5 min prep + 2 min validation + 10 min features + 40 min train + 5 min eval)

**Objectif:** AJI ≥ 0.68 pour publier résultats validés scientifiquement.

---

Bonne continuation !

— Session 2025-12-27
