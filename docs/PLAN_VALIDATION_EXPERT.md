# Plan de Validation Expert — Diagnostic AJI Catastrophique

**Date:** 2025-12-23
**Problème:** AJI 0.0524 vs cible 0.80 (écart +1427%)
**Hypothèse expert:** 1 blob géant au lieu de N noyaux séparés

---

## 📋 LES 3 KILLERS DE L'AJI (Expert Externe)

### Killer #1: Magnitude HV Trop Faible
**Hypothèse:** Gradients calculés sur pixels adjacents → magnitude <0.5 → watershed voit "plateau plat"

**Test à faire:**
```bash
# Script déjà créé: diagnose_predictions.py
python scripts/evaluation/diagnose_predictions.py \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --dataset_dir <CHEMIN_IMAGES_NPZ> \
    --num_samples 20 \
    --device cuda
```

**Critères de validation:**
- ✅ **SI pred_hv.max() > 0.5:** Magnitude OK → Killer #1 ÉLIMINÉ
- ❌ **SI pred_hv.max() < 0.3:** Tanh SATURE → Confirme Killer #1
- ⚠️ **SI 0.3 < pred_hv.max() < 0.5:** Tanh sous-utilisé → Killer #1 PARTIEL

**Action si confirmé:**
- Augmenter lambda_hv de 0.5 → 10.0 (recommandation expert)

---

### Killer #2: Binarisation Molle (Seuil NP)
**Hypothèse:** Seuil 0.5 trop bas → blobs "gonflent" et fusionnent

**Test à faire:**
```bash
# Modifier evaluate_ground_truth.py temporairement
# Ligne ~250: np_binary = (np_pred > 0.5).astype(np.uint8)
# Tester avec seuils: 0.3, 0.5, 0.7, 0.9
```

**Critères de validation:**
- ✅ **SI AJI(seuil=0.7) > AJI(seuil=0.5) +20%:** Confirme Killer #2
- ❌ **SI AJI stable ±5% entre seuils:** Killer #2 ÉLIMINÉ

**Action si confirmé:**
- Ajuster threshold optimal (probablement entre 0.6-0.8)

---

### Killer #3: Normalisation Incohérente
**Hypothèse:** Preprocessing train ≠ preprocessing inference → features corrompues

**Test à faire:**
```bash
# Vérifier CLS std sur échantillon d'évaluation
python scripts/validation/verify_features.py \
    --features_dir <CHEMIN_FEATURES_EVAL> \
    --expected_std_min 0.70 \
    --expected_std_max 0.90
```

**Alerte déjà observée:**
- Image 01889: "⚠️ Features SUSPECTES (CLS std=0.661, attendu 0.70-0.90)"

**Critères de validation:**
- ✅ **SI CLS std ∈ [0.70, 0.90]:** Normalisation OK → Killer #3 ÉLIMINÉ
- ❌ **SI CLS std < 0.65 OU > 0.95:** Features corrompues → Confirme Killer #3

**Action si confirmé:**
- Vérifier preprocessing dans evaluate_ground_truth.py
- Re-générer features avec normalisation correcte

---

## 🧪 PLAN DE SORTIE (Recommandation Expert)

### Étape 1: Diagnostic HV Brute ⬅️ **ON EST ICI**
**Objectif:** Vérifier si pred_hv.max() < 0.5

**Action:**
- Exécuter `diagnose_predictions.py` sur 20 images
- Visualiser 1 image avec `visualize_raw_predictions.py`

**Décision:**
- SI magnitude OK (>0.5) → Passer à Étape 2 (Killer #2)
- SI magnitude FAIBLE (<0.3) → Appliquer Étape 2b (Force la séparation)

---

### Étape 2: Test Binarisation
**Objectif:** Éliminer ou confirmer Killer #2

**Action:**
- Sweep threshold NP: [0.3, 0.5, 0.7, 0.9]
- Mesurer AJI/PQ pour chaque seuil

**Décision:**
- SI amélioration significative (+20% AJI) → Ajuster threshold
- SI pas d'amélioration → Killer #2 éliminé

---

### Étape 2b: Force la Séparation (SI Killer #1 confirmé)
**Objectif:** Forcer modèle à créer gradients nets

**Action:**
- Modifier `hovernet_decoder.py` ligne 343: `0.5 * hv_gradient` → `10.0 * hv_gradient`
- Ré-entraîner epidermal (50 epochs, ~1-2h)
- Évaluer sur ground truth

**Critères de succès:**
- AJI: 0.05 → >0.60 (+1100%)
- PQ: 0.08 → >0.70 (+775%)
- Rappel: 5.53% → >80% (+1347%)

---

### Étape 3: Nettoyage Features (SI Killer #3 confirmé)
**Objectif:** Garantir cohérence preprocessing

**Action:**
- Re-générer features avec `extract_features.py` (preprocessing unifié)
- Vérifier CLS std ∈ [0.70, 0.90]
- Ré-entraîner modèle sur features propres

---

## 📊 MATRICE DE DÉCISION

| Killer #1 (HV mag) | Killer #2 (Seuil) | Killer #3 (Norm) | Action Recommandée |
|--------------------|-------------------|------------------|-------------------|
| ❌ (<0.3) | - | - | lambda_hv → 10.0 + ré-entraîner |
| ⚠️ (0.3-0.5) | ✅ (+20% AJI) | - | Ajuster threshold NP |
| ⚠️ (0.3-0.5) | ❌ | ❌ (<0.65 std) | Re-générer features |
| ✅ (>0.5) | ✅ (+20% AJI) | ✅ | Ajuster threshold seul |
| ✅ (>0.5) | ❌ | ❌ | Re-générer features |
| ❌ | ❌ | ❌ | lambda_hv → 10.0 + features |

---

## 🎯 PRIORITÉ D'EXÉCUTION

**MAINTENANT (15 min):**
1. Trouver chemin vers images .npz (PanNuke fold2 ou family_data)
2. Exécuter `diagnose_predictions.py` (statistiques HV magnitude)
3. Exécuter `visualize_raw_predictions.py` (visualisation 1 image)

**SI Killer #1 confirmé (magnitude <0.3):**
- Modifier lambda_hv → 10.0
- Ré-entraîner (~1-2h)
- Évaluer ground truth (~5 min)

**SI Killer #2 suspect (magnitude OK mais AJI faible):**
- Sweep threshold NP [0.3, 0.5, 0.7, 0.9]
- Identifier seuil optimal

**SI Killer #3 confirmé (CLS std anormal):**
- Re-générer features avec preprocessing unifié
- Ré-entraîner modèle

---

## ⚠️ RÈGLES MÉTHODOLOGIQUES

1. **NE JAMAIS modifier le code sans avoir testé l'hypothèse**
2. **UN SEUL changement à la fois** (isolation des variables)
3. **Toujours mesurer AVANT/APRÈS** (baseline obligatoire)
4. **Documenter chaque test** (traçabilité scientifique)
5. **Si plusieurs killers confirmés:** Fixer dans l'ordre de priorité (HV mag > Norm > Seuil)

---

## 📝 CHECKLIST DE VALIDATION

- [ ] **Test Killer #1:** Exécuté diagnose_predictions.py → HV mag = ?
- [ ] **Test Killer #2:** Sweep threshold NP → AJI optimal à seuil = ?
- [ ] **Test Killer #3:** Vérifié CLS std → Features OK/KO ?
- [ ] **Décision:** Quel(s) killer(s) confirmé(s) ?
- [ ] **Action:** Modification appliquée = ?
- [ ] **Validation:** AJI après fix = ?

---

**STATUS:** ⏸️ En attente localisation données .npz pour tests
