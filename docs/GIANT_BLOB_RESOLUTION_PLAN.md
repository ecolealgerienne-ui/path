# Giant Blob - Plan de Résolution

**Date:** 2025-12-24
**Problème:** AJI 0.09 (objectif: 0.60+) - 1 instance au lieu de 8 détectées
**Diagnostic:** Giant Blob confirmé par visualisation

---

## 🔍 Résumé Diagnostic

### Ce Qui Fonctionne ✅

1. **Modèle détecte bien les cellules:**
   - NP Dice: 0.92 (excellent)
   - 137 peaks trouvés par `peak_local_max` (modèle "voit" les 137 cellules)

2. **Architecture correcte:**
   - ✅ Tanh() sur branche HV (ligne 118-121 hovernet_decoder.py)
   - ✅ Sobel gradient loss utilisé (ligne 347) avec poids 2.0×
   - ✅ HV targets normalisés float32 [-1, 1] (données v8)
   - ✅ Vraies instances PanNuke (pas connectedComponents)

3. **Données v8 correctes:**
   - Utilise vraies instances des canaux 1-4 PanNuke
   - HV maps bien normalisés [-1, 1]

### Ce Qui NE Fonctionne PAS ❌

1. **HV magnitude trop faible:**
   - GT range: [0.0000, 0.9992] ✅ NORMAL
   - PRED range: [0.0022, 0.0221] ❌ TRÈS FAIBLE (50× trop faible!)

2. **Watershed créé 1 instance au lieu de 8:**
   - Les 137 peaks sont détectés
   - Mais watershed ne sépare pas (gradients HV trop faibles)

3. **Scaling HV n'améliore pas l'AJI:**
   - Test ×1, ×5, ×10, ×20, ×50 → AJI stable à 0.09
   - Prouve que le problème n'est PAS juste une amplitude faible

---

## 📋 Documentation Consultée

### FIX_SOBEL_GRADIENT_LOSS.md (2025-12-23)

**Problème décrit:**
- AJI 0.07 vs cible 0.80
- HV MSE bon (0.05) mais gradients "doux" (pas nets)
- Watershed échoue car pas de frontières fermées

**Cause:**
- Gradient loss trop faible (signal 0.01 avec différences finies)
- Modèle apprend HV maps lisses au lieu de frontières nettes

**Solution implémentée:**
- Opérateur Sobel au lieu de différences finies simples
- Signal 4× plus fort

**Statut actuel:** ✅ DÉJÀ IMPLÉMENTÉ dans notre code

### ARCHITECTURE_HV_ACTIVATION.md (2025-12-21)

**Décision:** Conserver architecture SANS Tanh explicite

**Justification:** Tests empiriques montrent que modèle produit naturellement [-1, 1] via SmoothL1

**MAIS:** ⚠️ Cette décision a été CHANGÉE plus tard (ligne 118-121 hovernet_decoder.py AJOUTÉ Tanh)

---

## 🎯 Hypothèses Restantes

### Hypothèse #1: Modèle Entraîné AVANT Sobel Fix

**Possibilité:** Le checkpoint `hovernet_epidermal_best.pth` a été entraîné AVANT l'ajout de Sobel gradient loss.

**Vérification:**
```bash
# Voir date de création checkpoint
ls -l models/checkpoints/hovernet_epidermal_best.pth

# Comparer avec date de la session Sobel fix (2025-12-23)
```

**Test:**
```bash
# Ré-entraîner epidermal AVEC Sobel (déjà dans le code)
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0
```

**Attendu:** HV magnitude 0.022 → 0.5+ (gain ×20), AJI 0.09 → 0.60+ (gain ×6)

### Hypothèse #2: Gaussian Smoothing Trop Agressif

**Code v8 (ligne 135-136 prepare_family_data_FIXED_v8.py):**
```python
# Gaussian smoothing (sigma=0.5) pour réduire le bruit
hv_map[0] = gaussian_filter(hv_map[0], sigma=0.5)
hv_map[1] = gaussian_filter(hv_map[1], sigma=0.5)
```

**Impact:**
- Lisse les gradients HV dans les targets
- Modèle apprend à reproduire cette version lissée
- Watershed ne peut pas séparer instances

**Test:**
Régénérer données v8 SANS Gaussian smoothing:

```python
# prepare_family_data_FIXED_v8_nosmooth.py
# Commenter lignes 135-136

# PAS de smoothing
# hv_map[0] = gaussian_filter(hv_map[0], sigma=0.5)
# hv_map[1] = gaussian_filter(hv_map[1], sigma=0.5)
```

**Coût:** 30 min (régénération epidermal) + 2h (ré-entraînement)

### Hypothèse #3: Lambda_hv Trop Faible

**Code actuel (ligne 348):**
```python
hv_loss = hv_l1 + 2.0 * hv_gradient  # Équilibré: MSE + 2× gradient
```

**Expert recommandation (FIX_SOBEL_GRADIENT_LOSS.md ligne 198):**
```python
--lambda_hv 2.0
```

Mais ce lambda_hv s'applique à la LOSS TOTALE, pas au gradient:

```python
total = 1.0*np_loss + 2.0*hv_loss + 1.0*nt_loss
```

Donc: `total_hv_contribution = 2.0 * (hv_l1 + 2.0*hv_gradient) = 2.0*hv_l1 + 4.0*hv_gradient`

**Test:** Augmenter poids gradient de 2.0 → 5.0:

```python
hv_loss = hv_l1 + 5.0 * hv_gradient  # Plus de pression pour frontières nettes
```

**Risque:** Over-regularization (comme lambda_hv=10.0 qui a cassé le modèle)

---

## ⚡ Actions Immédiates (Ordre de Priorité)

### Action 1: Vérifier Date Checkpoint (2 min)

```bash
ls -l models/checkpoints/hovernet_epidermal_best.pth
```

**Si date < 2025-12-23:** Checkpoint entraîné AVANT Sobel fix → Ré-entraînement requis

**Si date ≥ 2025-12-23:** Checkpoint entraîné AVEC Sobel → Autre problème

### Action 2A: Si Checkpoint Ancien → Ré-entraîner (2h)

```bash
conda activate cellvit

python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0 \
    --batch_size 16
```

**Attendu:** HV magnitude 0.022 → 0.5+, AJI 0.09 → 0.60+

**Si échec:** Passer à Action 3

### Action 2B: Si Checkpoint Récent → Test Watershed Params (5 min)

Le modèle produit des gradients faibles même avec Sobel → Ajuster post-processing:

```python
# visualize_instance_maps.py ligne 44
# AVANT:
dist_threshold = 2  # CONSERVATIVE

# APRÈS:
dist_threshold = 1  # Moins conservateur
min_size = 5        # Au lieu de 10
```

**Attendu:** Plus d'instances détectées (1 → 5-8)

**Si échec:** Passer à Action 3

### Action 3: Si Échecs Persistants → Régénérer Sans Smoothing (3h)

1. Créer version v9 sans Gaussian smoothing:
   ```bash
   cp scripts/preprocessing/prepare_family_data_FIXED_v8.py \
      scripts/preprocessing/prepare_family_data_v9_nosmooth.py

   # Commenter lignes 135-136 (gaussian_filter)
   ```

2. Régénérer données epidermal:
   ```bash
   python scripts/preprocessing/prepare_family_data_v9_nosmooth.py \
       --family epidermal \
       --data_dir /home/amar/data/PanNuke \
       --output_dir data/cache/family_data_v9
   ```

3. Ré-entraîner:
   ```bash
   python scripts/training/train_hovernet_family.py \
       --family epidermal \
       --cache_dir data/cache/family_data_v9 \
       --epochs 50 \
       --augment
   ```

**Attendu:** HV magnitude plus forte (gradients non lissés), AJI amélioré

---

## 📊 Timeline Estimée

| Scénario | Temps | Probabilité |
|----------|-------|-------------|
| **A: Checkpoint ancien** | 2h (ré-entraînement) | 70% |
| **B: Watershed params** | 5 min (ajustement) | 15% |
| **C: Smoothing trop fort** | 3h (régénération + ré-entraînement) | 15% |

---

## 🔬 Vérifications Post-Fix

Après chaque fix, valider avec:

```bash
# 1. Test sur training data
python scripts/validation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10

# Attendu:
# NP Dice:  ~0.95
# HV magnitude: >0.5 (au lieu de 0.022)

# 2. Visualisation échantillon 9
python scripts/evaluation/visualize_instance_maps.py

# Attendu:
# Instances prédites: 5-8 (au lieu de 1)
# Instances GT: 8

# 3. AJI ground truth
python scripts/evaluation/test_aji_v8.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 50

# Attendu:
# AJI: >0.60 (au lieu de 0.09)
# PQ: >0.65 (au lieu de ~0.10)
```

---

## 📝 Recommandation Finale

**Scénario le plus probable:** Checkpoint entraîné AVANT l'implémentation du Sobel gradient loss.

**Action immédiate:**
1. Vérifier date checkpoint (2 min)
2. Si ancien: Ré-entraîner avec Sobel (2h)
3. Valider amélioration AJI 0.09 → 0.60+

**Si échec après ré-entraînement:** Investiguer Gaussian smoothing (Action 3)

---

**Prochaine action:** Vérifier date du checkpoint et lancer Action 1.
