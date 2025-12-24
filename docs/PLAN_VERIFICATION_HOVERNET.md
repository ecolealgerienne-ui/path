# Plan de Vérification Méthodique — Résolution Giant Blob

**Date:** 2025-12-24
**Problème:** AJI 0.09 (objectif: 0.60+) - 1 instance au lieu de 8 détectées
**Diagnostic:** Giant Blob confirmé par visualisation

---

## 📋 Compilation des Analyses

### Analyse Claude (Initial)

**Découvertes confirmées:**

1. ✅ **Architecture correcte:**
   - Tanh présent (ligne 118-121 hovernet_decoder.py)
   - Sobel gradient loss implémenté (ligne 244-280)
   - Lambda_hv = 2.0 (équilibré MSE + 2× gradient)

2. ✅ **Données v8 correctes:**
   - Utilise vraies instances PanNuke (canaux 1-4)
   - Pas de connectedComponents fusion (Bug #3 écarté)

3. ✅ **Giant Blob confirmé:**
   - Instances prédites: **1** (au lieu de 8 GT)
   - HV magnitude: **0.0022-0.0221** (50× trop faible vs 0.0000-0.9992 GT)
   - 137 peaks détectés (modèle "voit" les cellules)
   - Scaling ×50 n'améliore PAS l'AJI (0.0905 constant)

**Hypothèse principale:** Checkpoint entraîné AVANT Sobel fix (2025-12-23)

---

### Analyse Expert (Validation + Recommandations)

**Verdict expert:**

> "Tu as identifié ce qu'on appelle un **'mismatch de version logique'**. Avoir le code du Sobel dans tes fichiers .py ne sert à rien si les poids du fichier .pth ont été figés à une époque où le gradient était encore 'mou'."

**Points clés:**

1. **Sobel Fix = Game Changer**
   - **Sans Sobel:** Modèle minimise MSE pixel-wise → prédit valeurs "floues" moyennes
   - **Avec Sobel:** Modèle forcé à respecter variations brusques (pentes) → crée "barrages" pour Watershed

2. **Magnitude 0.022 = Signature d'un modèle "peureux"**
   - Modèle reste proche de zéro pour minimiser perte L1/MSE
   - N'ose pas créer gradients forts (risque erreur élevée)
   - Sobel fix force modèle à "muscler" ses prédictions

3. **Gaussian Smoothing (sigma=0.5) n'est PAS le problème**
   - Sigma 0.5 très léger, sert à éviter aliasing (crénelage pixels)
   - Garde le smoothing, ne pas le supprimer
   - Vrai problème: absence Sobel au training

4. **Prédiction expert: AJI 0.60+ après ré-entraînement**
   - Dice 0.95 prouve que modèle sait OÙ sont les cellules
   - Sobel fix lui apprend COMMENT les séparer
   - Analogie: "Donner des lunettes à quelqu'un qui voyait des formes sans détails"

---

## 🔍 Plan de Vérification (5 Étapes)

### Étape 1: Vérification HV Targets (CRITIQUE - 30s)

**Script:** `verify_hv_targets_npz.py`

**Commande:**
```bash
conda activate cellvit
python scripts/validation/verify_hv_targets_npz.py --family epidermal
```

**Checks automatiques:**
| Check | Attendu | Impact si échec |
|-------|---------|-----------------|
| Dtype | float32 | ❌ BLOQUANT - Régénération v9 requise |
| Range | [-1.0, 1.0] | ❌ BLOQUANT - Régénération v9 requise |
| Symétrie | mean ≈ 0.0 | ⚠️ WARNING - Vérifier compute_hv_maps() |
| Variance | std [0.3, 0.7] | ⚠️ Si <0.3: Gaussian trop agressif |

**Scénarios:**

**A. ✅ Targets corrects (dtype float32, range [-1, 1]):**
→ Passer à Étape 2

**B. ❌ Targets incorrects (int8, pixels bruts, etc.):**
→ **STOP** — Régénérer données v9 AVANT ré-entraînement
```bash
# Créer v9 sans bug normalization
python scripts/preprocessing/prepare_family_data_v9.py --family epidermal
```

---

### Étape 2: Vérification Date Checkpoint (2 min)

**Objectif:** Confirmer que checkpoint est antérieur au Sobel fix (2025-12-23)

**Commande:**
```bash
# Trouver le checkpoint
find models/checkpoints -name "hovernet_epidermal_best.pth" -exec ls -l {} \;

# Comparer avec date Sobel fix
echo "Date Sobel fix: 2025-12-23"
```

**Scénarios:**

**A. Checkpoint date < 2025-12-23:**
→ ✅ Confirme hypothèse "mismatch version logique"
→ Ré-entraînement avec Sobel résoudra le problème

**B. Checkpoint date ≥ 2025-12-23:**
→ ⚠️ Checkpoint entraîné AVEC Sobel, mais performances catastrophiques
→ Autre problème (features corrompues? Bug code?)
→ Vérifier logs training epoch par epoch

---

### Étape 3: Vérification Logs Training (5 min)

**Objectif:** S'assurer que Sobel gradient loss était bien actif durant training

**Fichier:** `results/training_hovernet_epidermal.log` (ou équivalent)

**Chercher dans logs:**
```bash
grep -i "hv_gradient" results/training_hovernet_epidermal.log
grep -i "sobel" results/training_hovernet_epidermal.log
```

**Attendu:**
```
Epoch 1: hv_l1=0.45, hv_gradient=0.12, hv_loss=0.69 (hv_l1 + 2.0*hv_gradient)
```

**Si absent:**
→ ✅ Confirme que Sobel n'était PAS actif → Ré-entraînement requis

---

### Étape 4: Décision GO/NO-GO Ré-entraînement

**Arbre de décision:**

```
Étape 1 (HV targets):
├─ ✅ Targets corrects → Continuer
└─ ❌ Targets incorrects → STOP, régénérer v9

Étape 2 (Date checkpoint):
├─ Date < 2025-12-23 → ✅ GO ré-entraînement
└─ Date ≥ 2025-12-23 → ⚠️ Investiguer logs (Étape 3)

Étape 3 (Logs training):
├─ Sobel absent des logs → ✅ GO ré-entraînement
└─ Sobel présent dans logs → ❌ NO-GO, autre problème
```

**Critères GO ré-entraînement:**
- [x] HV targets float32 [-1, 1] ✅
- [x] Checkpoint date < 2025-12-23 ✅
- [x] Sobel absent des logs training ✅

**Si tous critères GO:**
→ Passer à Étape 5 (Ré-entraînement)

**Si un critère NO-GO:**
→ Investigation approfondie requise (autre bug caché)

---

### Étape 5: Ré-entraînement avec Sobel Fix

**Recommandation Expert (lambda_hv augmenté):**

```bash
conda activate cellvit

python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_np 1.0 \
    --lambda_hv 3.0 \
    --lambda_nt 1.0 \
    --batch_size 16
```

**Changement clé:** `lambda_hv 2.0 → 3.0`

**Justification expert:**
> "Augmenté légèrement (2.0 → 3.0) pour **vraiment pousser le gradient**. Le modèle doit comprendre que la netteté des frontières est AUSSI importante que la présence des noyaux."

**Métriques à surveiller durant training:**

| Epoch | HV MSE Attendu | Interprétation |
|-------|----------------|----------------|
| 1-5 | 0.30-0.40 | Normal (modèle apprend) |
| 10-20 | 0.15-0.25 | Convergence en cours |
| 30-50 | **0.05-0.10** | ✅ Sobel actif (descente lente mais stable) |

**Citation expert:**
> "Si [HV MSE] descend plus lentement ou reste plus haute qu'avant tout en étant stable, c'est bon signe : le modèle travaille plus dur sur les détails complexes du gradient."

**Durée estimée:** ~40 minutes (571 samples epidermal)

---

## 📊 Métriques de Validation Post-Training

**Test 1: Training Data (10 échantillons)**
```bash
python scripts/evaluation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10
```

**Attendu:**
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| NP Dice | 0.92 | ~0.95 | Stable/Légère hausse |
| **HV Magnitude** | **0.022** | **>0.50** | **+2200%** 🎯 |
| NT Acc | ~0.89 | ~0.90 | Stable |

---

**Test 2: Visualisation Instance Maps**
```bash
python scripts/evaluation/visualize_instance_maps.py
```

**Fichier généré:** `results/diagnostic_instance_maps_sample9.png`

**Attendu:**
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| **Instances PRED** | **1** | **5-8** | **+500-700%** 🎯 |
| Instances GT | 8 | 8 | (référence) |
| Couleurs visibles | 1 violette | 5-8 distinctes | ✅ Séparation |

---

**Test 3: AJI Ground Truth (50 échantillons)**
```bash
python scripts/evaluation/test_aji_v8.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 50
```

**Attendu:**
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| **AJI** | **0.09** | **>0.60** | **+567%** 🎯 |
| **PQ** | ~0.10 | **>0.65** | **+550%** 🎯 |
| Dice | 0.92 | ~0.95 | Stable |

---

## 🎯 Critères de Succès

**Niveau 1: Acceptable (objectif minimum)**
- AJI ≥ 0.50 (+455% vs 0.09)
- PQ ≥ 0.55 (+450% vs 0.10)
- Instances PRED ≥ 5 (vs 1 actuel)

**Niveau 2: Bon (objectif cible)**
- AJI ≥ 0.60 (+567%)
- PQ ≥ 0.65 (+550%)
- Instances PRED ≥ 6-7

**Niveau 3: Excellent (dépassement objectif)**
- AJI ≥ 0.70 (+678%)
- PQ ≥ 0.75 (+650%)
- Instances PRED = 8 (parfait match GT)

**Prédiction expert:** Niveau 2 (AJI 0.60+) fortement probable

---

## ⚠️ Plan de Contingence

**Si échec partiel (AJI 0.30-0.50):**

1. **Test lambda_hv = 5.0** (encore plus agressif)
   ```bash
   python scripts/training/train_hovernet_family.py \
       --family epidermal --epochs 50 --augment \
       --lambda_hv 5.0
   ```

2. **Vérifier Gaussian smoothing**
   - Régénérer v9 avec sigma=0.3 (au lieu de 0.5)
   - Ré-entraîner

**Si échec total (AJI <0.30):**

→ Investigation approfondie requise:
- Vérifier features H-optimus-0 (CLS std, corruption)
- Vérifier fonction compute_hv_maps()
- Vérifier post-processing Watershed

---

## 📝 Checklist Pré-Lancement

Avant de lancer le ré-entraînement, vérifier:

- [ ] **Étape 1:** HV targets vérifiés (dtype float32, range [-1, 1])
- [ ] **Étape 2:** Date checkpoint confirmée < 2025-12-23
- [ ] **Étape 3:** Logs training vérifiés (Sobel absent)
- [ ] **Décision:** GO ré-entraînement confirmé
- [ ] **Environnement:** `conda activate cellvit` activé
- [ ] **GPU:** VRAM disponible (~8-10 GB requis)
- [ ] **Durée:** 40 minutes disponibles (epidermal)

---

## 🔗 Références

- **GIANT_BLOB_RESOLUTION_PLAN.md:** Plan initial avec 3 hypothèses
- **FIX_SOBEL_GRADIENT_LOSS.md:** Documentation Sobel fix (2025-12-23)
- **ANALYSE_TEST_SCALING_NEGATIF.md:** Tests scaling ×1 à ×50
- **ARCHITECTURE_HV_ACTIVATION.md:** Décision Tanh (2025-12-21)
- **HoVer-Net (Graham et al., 2019):** Paper original
- **CellViT (Hägele et al., 2023):** ViT + HoVer-Net decoder

---

**Prochaine action:** Exécuter Étape 1 (vérification HV targets) → Décision GO/NO-GO
