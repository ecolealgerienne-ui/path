# Checklist Élimination Méthodique — Giant Blob

**Date:** 2025-12-24
**Problème:** AJI 0.09 vs objectif 0.60+ (1 instance au lieu de 8)
**Approche:** Élimination systématique, point par point

---

## 📋 Méthode d'Élimination

Chaque test ÉLIMINE ou CONFIRME une hypothèse. On procède séquentiellement jusqu'à identifier la cause racine unique.

---

## ✅ ÉTAPE 1: Vérifier Architecture Code

**Hypothèse:** "Le code manque Tanh ou Sobel"

**Tests effectués:**
- [x] Vérifier Tanh dans HV branch (ligne 118-121 hovernet_decoder.py)
- [x] Vérifier Sobel gradient loss (ligne 244-280 hovernet_decoder.py)
- [x] Vérifier données v8 (vraies instances PanNuke)

**Résultats:**
```
✅ Tanh présent: nn.Tanh() ligne 119
✅ Sobel implémenté: gradient_loss() ligne 244-280
✅ Lambda_hv = 2.0 (poids Sobel)
✅ Données v8: vraies instances (pas connectedComponents)
```

**Conclusion:** ✅ Architecture code CORRECTE

**Hypothèse ÉLIMINÉE:** ❌ "Code incomplet"

---

## ✅ ÉTAPE 2: Vérifier Targets HV Stockés

**Hypothèse:** "Targets .npz incorrects (int8, pixels bruts, mal normalisés)"

**Test effectué:**
```bash
python scripts/validation/verify_hv_targets_npz.py --family epidermal
```

**Résultats:**
```
✅ Dtype: float32 (correct)
✅ Range: [-0.990, 0.977] (correct, dans [-1, 1])
✅ Symétrie: Mean=0.000000 (centré)
✅ Variance: Std=0.373928 (bonne dynamique [0.3, 0.7])

Échantillons vérifiés:
  Sample 0: Range [-0.830, +0.900], 92% pixels non-zero
  Sample 1: Range [-0.900, +0.926], 99% pixels non-zero
  Sample 2: Range [-0.942, +0.930], 100% pixels non-zero
```

**Conclusion:** ✅ Targets HV CORRECTS (données v8 intègres)

**Hypothèses ÉLIMINÉES:**
- ❌ "Targets en int8 [-127, 127]"
- ❌ "Targets en pixels bruts (non normalisés)"
- ❌ "Gaussian smoothing trop agressif (std=0.374 OK)"

---

## ✅ ÉTAPE 3: Vérifier Date Checkpoint vs Sobel Fix

**Hypothèse:** "Checkpoint entraîné AVANT Sobel fix (2025-12-23)"

**Test effectué:**
```bash
find models/checkpoints -name "hovernet_epidermal_best.pth" -exec ls -l {} \;
```

**Résultats:**
```
-rw-r--r-- 1 amar amar 13888090 Dec 24 17:09 models/checkpoints/hovernet_epidermal_best.pth
```

**Date checkpoint:** 24 décembre 2025, 17h09
**Date Sobel fix:** 23 décembre 2025

**Conclusion:** ❌ Hypothèse REJETÉE

Le checkpoint a été entraîné **APRÈS** le Sobel fix (24 déc > 23 déc), donc le modèle DEVRAIT avoir bénéficié du Sobel.

**Hypothèse ÉLIMINÉE:** ❌ "Checkpoint pré-Sobel"

**NOUVELLE QUESTION CRITIQUE:**
> Si le modèle a été entraîné AVEC Sobel (24 déc), pourquoi HV magnitude est-elle quand même catastrophique (0.022) ?

---

## ⏳ ÉTAPE 4: Vérifier Features H-optimus-0 (EN COURS)

**Hypothèse:** "Mismatch normalisation features H-optimus-0"

**Recommandation script verify_hv_targets_npz.py:**
> "Le problème vient soit du MODÈLE (poids mal entraînés) soit des FEATURES (mismatch normalisation H-optimus-0). Vérifier CLS std doit être dans [0.70, 0.90]"

**Test à effectuer:**

### 4.1. Vérifier Features Training (famille epidermal)

**Script à créer:** `scripts/validation/verify_features_training.py`

**Objectif:** Charger features utilisées DURANT training et vérifier CLS std

**Commande:**
```bash
python scripts/validation/verify_features_training.py \
    --family epidermal \
    --features_file data/cache/family_data_FIXED/epidermal_features.npz
```

**Checks:**
| Check | Valeur Attendue | Interprétation |
|-------|----------------|----------------|
| **CLS std** | **[0.70, 0.90]** | Features H-optimus-0 correctes |
| CLS mean | ~0.0 | Centré (après normalisation) |
| Shape | (N, 261, 1536) | 1 CLS + 256 patches, 1536-dim |

**Scénarios possibles:**

**A. ✅ CLS std dans [0.70, 0.90]:**
```
✅ CLS std: 0.768
✅ Shape: (571, 261, 1536)
✅ Mean: ~0.0
```
→ Features training CORRECTES
→ Problème vient du MODÈLE lui-même (poids mal convergés)
→ Passer à ÉTAPE 5 (logs training)

**B. ❌ CLS std hors plage:**
```
❌ CLS std: 0.28 (trop bas - Bug #2 LayerNorm mismatch)
ou
❌ CLS std: 1.50 (trop haut - normalisation incorrecte)
```
→ Features training CORROMPUES (Bug #1 ToPILImage ou Bug #2 LayerNorm)
→ STOP — Régénérer features AVANT ré-entraînement

---

### 4.2. Vérifier Features Inference (test actuel)

**Script à utiliser:** `scripts/validation/compare_train_vs_inference.py`

**Objectif:** Comparer CLS std entre training et inference

**Commande:**
```bash
python scripts/validation/compare_train_vs_inference.py \
    --family epidermal \
    --training_features data/cache/family_data_FIXED/epidermal_features.npz \
    --test_image data/test_samples_by_family/epidermal/test_samples.npz \
    --test_index 8
```

**Attendu:**
```
Training CLS std:  0.768
Inference CLS std: 0.771
Ratio (I/T):       1.004  ← Doit être proche de 1.0

✅ Cohérence train/inference OK (ratio < 1.05)
```

**Si ratio > 1.20 ou < 0.80:**
→ ❌ MISMATCH train/inference
→ Preprocessing différent entre training et inference
→ Cause possible: Bug #1 ou Bug #2 résolu APRÈS training

**Statut:** ⏳ NON EXÉCUTÉ (script à créer)

---

## ⏳ ÉTAPE 5: Vérifier Logs Training

**Hypothèse:** "Sobel présent dans code mais pas actif durant training"

**Test à effectuer:**

**Commande:**
```bash
find results -name "*epidermal*train*.log" -o -name "training_log_epidermal*"
```

**Si log trouvé:**
```bash
# Vérifier présence Sobel gradient
grep -i "hv_gradient" <log_file>
grep -i "sobel" <log_file>

# Vérifier convergence HV MSE
grep -E "Epoch.*HV MSE" <log_file> | tail -20
```

**Attendu si Sobel actif:**
```
Epoch 1:  hv_l1=0.45, hv_gradient=0.12, hv_loss=0.69
Epoch 10: hv_l1=0.28, hv_gradient=0.09, hv_loss=0.46
Epoch 50: hv_l1=0.15, hv_gradient=0.05, hv_loss=0.25
```

**Si Sobel ABSENT des logs:**
→ ❌ Sobel non actif (bug dans script training)
→ Vérifier train_hovernet_family.py ligne 347

**Si HV MSE ne descend PAS:**
```
Epoch 1:  HV MSE: 0.35
Epoch 10: HV MSE: 0.34
Epoch 50: HV MSE: 0.33  ← Stagnation (pas de convergence)
```
→ ❌ Problème convergence (learning rate? features corrompues?)

**Statut:** ⏳ NON EXÉCUTÉ

---

## ⏳ ÉTAPE 6: Test Lambda_hv Augmenté

**Hypothèse:** "Lambda_hv=2.0 insuffisant, augmenter à 3.0 ou 5.0"

**Condition:** SEULEMENT si Étapes 4 et 5 ✅ (features OK, Sobel actif, mais HV magnitude quand même faible)

**Test rapide (1 epoch):**
```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 1 \
    --augment \
    --lambda_hv 5.0 \
    --batch_size 16
```

**Vérifier après epoch 1:**
```bash
python scripts/evaluation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 5
```

**Attendu:**
| Métrique | Lambda=2.0 | Lambda=5.0 | Amélioration |
|----------|------------|------------|--------------|
| HV Magnitude | 0.022 | >0.10 | +350% |

**Si amélioration visible:**
→ ✅ Lambda_hv était insuffisant
→ Ré-entraîner complet avec lambda_hv=5.0 (50 epochs)

**Si pas d'amélioration:**
→ ❌ Problème plus profond (features ou architecture)

**Statut:** ⏳ NON EXÉCUTÉ

---

## ⏳ ÉTAPE 7: Vérifier Post-Processing Watershed

**Hypothèse:** "Watershed mal configuré (malgré gradients HV corrects)"

**Condition:** SEULEMENT si HV magnitude > 0.5 mais AJI quand même faible

**Test paramètres:**
```python
# Dans scripts/evaluation/visualize_instance_maps.py
# AVANT (actuel):
markers = peak_local_max(energy, min_distance=2, threshold_abs=0.05)

# TEST A (moins conservateur):
markers = peak_local_max(energy, min_distance=1, threshold_abs=0.02)

# TEST B (agressif):
markers = peak_local_max(energy, min_distance=1, threshold_abs=0.01)
```

**Attendu:**
| Paramètres | Instances PRED | AJI |
|------------|----------------|-----|
| Original (min_dist=2, thresh=0.05) | 1 | 0.09 |
| Test A (min_dist=1, thresh=0.02) | 5-8 | 0.40+ |
| Test B (min_dist=1, thresh=0.01) | 10-15 | 0.50+ (sur-segmentation) |

**Statut:** ⏳ NON EXÉCUTÉ (conditionnel à HV magnitude > 0.5)

---

## 🎯 Arbre de Décision (État Actuel)

```
ÉTAPE 1: Architecture Code ────────────────────────── ✅ ÉLIMINÉE (code correct)
         │
         ▼
ÉTAPE 2: Targets HV .npz ──────────────────────────── ✅ ÉLIMINÉE (targets corrects)
         │
         ▼
ÉTAPE 3: Date Checkpoint ──────────────────────────── ✅ ÉLIMINÉE (post-Sobel, 24 déc)
         │
         ▼
ÉTAPE 4: Features H-optimus-0 ◄───────────────────── ⏳ EN COURS
         │
         ├─ ✅ CLS std OK [0.70-0.90] ──────────────► ÉTAPE 5 (Logs training)
         │                                                   │
         │                                                   ├─ Sobel actif ─────► ÉTAPE 6 (Lambda_hv)
         │                                                   │
         │                                                   └─ Sobel absent ────► FIX script + re-train
         │
         └─ ❌ CLS std KO (<0.40 ou >1.0) ─────────► STOP ─► Régénérer features
                                                              │
                                                              ▼
                                                         Re-train avec features fixes
                                                              │
                                                              ▼
                                                         ÉTAPE 8 (Validation)
```

---

## 📊 Résumé État Actuel

**Tests complétés:** 3/7 (43%)

| Étape | Statut | Résultat | Hypothèse |
|-------|--------|----------|-----------|
| 1. Architecture | ✅ | Code correct | ❌ ÉLIMINÉE |
| 2. Targets HV | ✅ | Targets corrects | ❌ ÉLIMINÉE |
| 3. Date Checkpoint | ✅ | Post-Sobel (24 déc) | ❌ ÉLIMINÉE |
| **4. Features H-optimus-0** | **⏳** | **À vérifier** | **?** |
| 5. Logs Training | ⏳ | - | - |
| 6. Lambda_hv | ⏳ | - | - |
| 7. Watershed | ⏳ | - | - |

**Prochaine action CRITIQUE:**
→ **ÉTAPE 4: Vérifier CLS std features training**

---

## 🔑 Points Clés de l'Investigation

### Découverte Majeure (ÉTAPE 3)

Le checkpoint epidermal a été entraîné **AUJOURD'HUI (24 déc 17h09)**, APRÈS le Sobel fix (23 déc).

**Implication:**
- Le modèle DEVRAIT avoir Sobel actif
- Mais HV magnitude quand même catastrophique (0.022)
- **Nouvelle hypothèse:** Problème durant training (features? convergence? bug code?)

### Citation Script verify_hv_targets_npz.py

> "Les targets HV sont bien normalisés. Le problème de magnitude faible (0.022) vient donc:
> → Soit du MODÈLE (poids mal entraînés)
> → Soit des FEATURES (mismatch normalisation H-optimus-0)"

**Prochaine investigation:** Vérifier features training (CLS std)

---

## 📝 Scripts à Créer

| Script | Objectif | Priorité |
|--------|----------|----------|
| `verify_features_training.py` | Vérifier CLS std features epidermal | ⚠️ CRITIQUE |
| `compare_train_vs_inference.py` | Comparer features train/inference | Haute |
| `analyze_training_logs.py` | Parser logs et extraire convergence HV MSE | Moyenne |

---

## ✅ Checklist Prochaine Session

Avant de continuer:

- [ ] Créer `verify_features_training.py`
- [ ] Exécuter et vérifier CLS std features training
- [ ] Si CLS std OK → Chercher logs training
- [ ] Si CLS std KO → Régénérer features
- [ ] Documenter résultats dans HISTORIQUE_TESTS_GIANT_BLOB.md

---

**Dernière mise à jour:** 2025-12-24
**Prochaine action:** ÉTAPE 4 - Vérifier features H-optimus-0
