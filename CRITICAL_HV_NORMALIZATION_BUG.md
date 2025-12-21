# DÉCOUVERTE MAJEURE: Bug de Normalisation HV Maps

**Date**: 2025-12-21
**Gravité**: 🔴 CRITIQUE
**Impact**: Ré-entraînement OBLIGATOIRE (pas optionnel)

---

## 🔍 Découverte

Le diagnostic a révélé un **bug fondamental** dans l'ancien pipeline de préparation des données:

### OLD DATA (INCORRECT)
```
Dtype: int8
Range: [-127, 127]
Size: 423.88 MB
❌ NON conforme à HoVer-Net (Graham et al., 2019)
```

### NEW DATA (CORRECT)
```
Dtype: float32
Range: [-1, 1]
Size: 1695.50 MB
✅ Conforme à HoVer-Net (Graham et al., 2019)
```

---

## 📊 Statistiques Comparatives

| Propriété | OLD (Bugué) | NEW (Fixé) | Impact |
|-----------|-------------|------------|--------|
| **Dtype** | int8 | float32 | Incompatibles |
| **Range H** | [-127, 127] | [-1, 1] | 127x différence |
| **Range V** | [-127, 127] | [-1, 1] | 127x différence |
| **Mean** | ~0 | ~0 | ✓ Centré OK |
| **Std** | 64.35 | 0.535 | 120x différence |
| **Taille** | 423.88 MB | 1695.50 MB | 4x plus gros (correct) |

---

## 🤔 Pourquoi OLD Utilisait int8?

L'ancien script avait **deux objectifs contradictoires**:

1. ✅ **Économiser l'espace disque**: int8 = 4x compression (423 MB vs 1695 MB)
2. ❌ **Normaliser à [-1, 1]**: OUBLIÉ!

Résultat:
- Les données étaient stockées en int8 [-127, 127]
- MAIS jamais converties à float32 [-1, 1] avant entraînement
- Le modèle a appris sur des valeurs **127x trop grandes**

---

## 🚨 Impact sur le Modèle Actuel (Dice 0.9645)

### ❌ Le Modèle Actuel est "Fonctionnel par Accident"

Le modèle a quand même convergé car:
- Les poids se sont adaptés à l'échelle [−127, 127]
- La forme relative des gradients HV est préservée
- La branche NP (binaire) n'est pas affectée

**MAIS:**
- Ce n'est **pas conforme** à HoVer-Net (Graham et al., 2019)
- Les gradients prédits sont mal calibrés
- Impossible de comparer avec d'autres implémentations
- Incompatible avec le nouveau modèle (NEW data)

### ✅ Ré-entraînement Obligatoire

Ce n'est **PAS optionnel**:
- NEW data est la **seule version correcte**
- Impossible de transférer les poids (échelles incompatibles)
- Le modèle actuel ne peut pas utiliser NEW data sans retraining

---

## 🛠️ Validation Corrigée

### Problème Initial dans `validate_fixed_data.py`

Le script comparait les gradients **sans normaliser OLD**:

```python
# ❌ BUGGY (avant):
grad_old = np.abs(np.gradient(hv_old, axis=(1, 2))).mean()  # int8 [-127, 127]
grad_new = np.abs(np.gradient(hv_new, axis=(1, 2))).mean()  # float32 [-1, 1]
ratio = grad_new / grad_old  # 0.0089 / 0.6926 = 0.013 ← FAUX!

# ✅ FIXED (maintenant):
if hv_old_raw.dtype == np.int8:
    hv_old = hv_old_raw.astype(np.float32) / 127.0  # Normaliser OLD
grad_old = np.abs(np.gradient(hv_old, axis=(1, 2))).mean()
grad_new = np.abs(np.gradient(hv_new, axis=(1, 2))).mean()
ratio = grad_new / grad_old  # Comparaison ÉQUITABLE
```

### Fix Appliqué (commit `ffbe2a5`)

1. Détecter dtype int8 dans OLD data
2. Normaliser à float32: `hv_old / 127.0`
3. Comparer les gradients normalisés
4. Ajout check dtype dans `count_instances_in_target()`

---

## 🎯 Prochaines Étapes

### 1. Pull le Fix (OBLIGATOIRE)

```bash
git pull origin claude/evaluation-ground-truth-zJB9O
```

### 2. Re-lancer la Validation avec Script Corrigé

```bash
python scripts/evaluation/validate_fixed_data.py \
    --old_data data/cache/family_data/glandular_targets.npz \
    --new_data data/family_FIXED/glandular_data_FIXED.npz \
    --family glandular \
    --sample_idx 0
```

**Sortie Attendue** (maintenant correcte):
```
📊 OLD DATA (BUGGY - int8 normalized for comparison):
   ⚠️  Original range: [-127, 127] (int8)
   ✓ Normalized to: [-1.000, 1.000] (float32)
   HV gradient magnitude: 0.XXXX

📊 NEW DATA (FIXED):
   HV gradient magnitude: 0.YYYY

📈 COMPARAISON OLD vs NEW
   HV Gradient Magnitude:
     OLD: 0.XXXX
     NEW: 0.YYYY
     Ratio: Z.ZZx  ← DOIT ÊTRE >= 1.0 (idéalement >= 1.2)
```

### 3. Critère de Validation

**Question clé**: Les instances sont-elles **mieux séparées** dans NEW vs OLD?

| Ratio | Interprétation | Action |
|-------|----------------|--------|
| **≥ 1.2x** | ✅ Amélioration significative | GO pour entraînement |
| **1.0 - 1.2x** | ⚠️ Amélioration faible | Discuter, probablement GO |
| **0.8 - 1.0x** | ~ Similaire | Pas d'amélioration instance separation, mais NEW correct |
| **< 0.8x** | ❌ Régression | Investiguer pourquoi |

**Note importante**: Même si ratio < 1.2x, le **ré-entraînement reste OBLIGATOIRE** car:
- NEW est conforme à HoVer-Net ([-1, 1] normalisé)
- OLD est incorrect (int8 non normalisé)
- La conformité à la littérature prime sur l'amélioration mesurable

### 4. Entraînement Glandular (~2.5h)

Si validation OK (ou même si ratio < 1.2x car ré-entraînement obligatoire):

```bash
python scripts/training/train_hovernet_family.py \
    --family glandular \
    --data_dir data/family_FIXED \
    --output_dir models/checkpoints_FIXED \
    --epochs 50 \
    --augment \
    --batch_size 32
```

**Résultats Attendus**:

| Métrique | Avant (OLD) | Cible (NEW) | Critique |
|----------|-------------|-------------|----------|
| NP Dice | 0.9645 | ≥ 0.96 | Maintenir |
| HV MSE | 0.0150 | **< 0.015** | Améliorer si ratio > 1.0 |
| NT Acc | 0.8800 | ≥ 0.88 | Maintenir |

**HV MSE** devrait s'améliorer SI le ratio gradient NEW/OLD > 1.0.

---

## 📖 Littérature HoVer-Net

**Source**: Graham et al., "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images", Medical Image Analysis 2019

**Spécification HV Maps** (section 3.2):

```
Pour chaque pixel (x, y) d'une instance i:

H[x,y] = (x - cx_i) / R_i  ∈ [-1, 1]
V[x,y] = (y - cy_i) / R_i  ∈ [-1, 1]

Où:
  (cx_i, cy_i) = centre de masse de l'instance i
  R_i = rayon de l'instance (max distance centre → bord)
```

**Propriétés**:
- Normalisé à [-1, 1] (OBLIGATOIRE)
- Gradient fort aux frontières entre instances
- Post-processing: Sobel(HV) → Watershed → Instances séparées

---

## 🎓 Leçons Apprises

### 1. Compression != Normalisation

```python
# ❌ PIÈGE: Stocker en int8 sans normaliser ensuite
hv_map_int8 = (hv_map * 127).astype(np.int8)  # Économie mémoire
np.savez(path, hv_targets=hv_map_int8)        # Sauvegarde

# À l'entraînement:
hv = data['hv_targets']  # int8 [-127, 127] ← INCORRECT!
loss_hv = mse_loss(pred_hv, hv)  # Échelle incorrecte

# ✅ CORRECT: Normaliser à [-1, 1] même si stocké en int8
hv = data['hv_targets'].astype(np.float32) / 127.0  # [-1, 1]
loss_hv = mse_loss(pred_hv, hv)  # Échelle correcte
```

### 2. Toujours Valider les Ranges

```python
# Vérification obligatoire avant entraînement:
assert hv_targets.min() >= -1.1 and hv_targets.max() <= 1.1, \
    f"HV maps non normalisés! Range: [{hv_targets.min()}, {hv_targets.max()}]"
```

### 3. Documentation des Formats

Créer un README pour chaque dataset:

```markdown
# Dataset: glandular_data_FIXED.npz

## Format
- images: (N, 256, 256, 3) float64 [0, 255]
- np_targets: (N, 256, 256) float32 [0, 1]
- hv_targets: (N, 2, 256, 256) float32 [-1, 1] ← NORMALIZED
- nt_targets: (N, 256, 256) int64 [0, 4]

## Normalisation HV
Conforme à HoVer-Net (Graham et al., 2019):
H[x,y] = (x - cx) / R ∈ [-1, 1]
V[x,y] = (y - cy) / R ∈ [-1, 1]
```

---

## ✅ Confirmation de Compréhension

Avant de procéder, confirmer:

- [ ] J'ai compris que OLD data est **incorrecte** (int8 non normalisé)
- [ ] J'ai compris que NEW data est **correcte** (float32 normalisé)
- [ ] J'ai compris que le ré-entraînement est **OBLIGATOIRE**
- [ ] J'ai pullé le fix (`git pull origin claude/evaluation-ground-truth-zJB9O`)
- [ ] Je vais re-lancer la validation avec le script corrigé
- [ ] Je suis prêt à lancer l'entraînement (~10h pour 5 familles)

---

**Créé le**: 2025-12-21
**Par**: Claude (Investigation Root Cause - HV normalization bug)
**Commit Fix**: `ffbe2a5`
