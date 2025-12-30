# Session 2025-12-29 — Refactoring Modules Partagés & Résultats 60 Epochs

## Résumé

- **Découverte bug critique:** Divergence algorithme watershed entre scripts (-2.8% AJI)
- **Refactoring:** Création de 3 modules partagés (single source of truth)
- **Respiratory:** AJI **0.6872** ✅ **OBJECTIF ATTEINT** (101.1%)
- **Urologic:** AJI **0.6743** (99.2% de l'objectif)
- **Epidermal:** AJI 0.6203 (91.2% de l'objectif)
- **Digestive:** AJI 0.6160 (90.6% de l'objectif)
- **Découverte:** Notre implémentation calcule AJI+ (one-to-one), pas AJI original

---

## 1. Bug Critique Découvert

### Symptôme
- Test manuel avec np_threshold=0.45: AJI **0.6128**
- Grid search avec np_threshold=0.45: AJI **0.5955**
- Différence inexpliquée de **-2.8%**

### Cause Racine
Les deux scripts utilisaient des algorithmes **différents**:

| Aspect | `test_v13_smart_crops_aji.py` | `optimize_watershed_aji.py` |
|--------|-------------------------------|------------------------------|
| Fonction de labeling | `scipy.ndimage.label` | `skimage.measure.label` |
| Moment du labeling | AVANT watershed | APRÈS watershed |
| Matching instances | Différent | Différent |

### Impact
`scipy.ndimage.label` et `skimage.measure.label` produisent des résultats différents sur les frontières d'instances.

---

## 2. Solution: Modules Partagés

### Nouvelle Architecture

```
src/
├── postprocessing/
│   ├── __init__.py
│   └── watershed.py              # hv_guided_watershed()
├── metrics/
│   └── ground_truth_metrics.py   # compute_aji()
└── evaluation/
    ├── __init__.py
    └── instance_evaluation.py    # run_inference(), evaluate_sample(), evaluate_batch_with_params()
```

### Principe
**Single Source of Truth** — Un seul algorithme partagé par tous les scripts.

```python
# ✅ CORRECT
from src.postprocessing import hv_guided_watershed
from src.evaluation import run_inference, evaluate_batch_with_params

# ❌ INTERDIT
def hv_guided_watershed(...):  # Copie locale = divergence future
```

### Commits
- `8d30d2d` - fix(eval): Align hv_to_instances with test script watershed algorithm
- `980d99a` - refactor(postprocessing): Create shared hv_guided_watershed module
- `bf4281c` - docs: Add mandatory shared modules rule to CLAUDE.md
- `69be805` - refactor(evaluation): Create shared evaluation module

---

## 3. Découverte AJI vs AJI+

### Notre Implémentation
Notre `compute_aji()` utilise un **matching one-to-one**:
```python
if pred_id in used_pred:
    continue  # Chaque prédiction ne peut être utilisée qu'une fois
```

### Différence avec AJI Original (Kumar et al. 2017)
| Métrique | Matching | Sur-pénalisation |
|----------|----------|------------------|
| **AJI (original)** | Many-to-one | Oui |
| **AJI+ (notre implémentation)** | One-to-one | Non |

### Implications
- Nos résultats sont valides (AJI+ est reconnu)
- Non directement comparables aux benchmarks utilisant AJI original
- AJI+ donne généralement des scores légèrement plus élevés

### Référence
- [HoVer-Net Metrics](https://github.com/vqdang/hover_net/blob/master/metrics/stats_utils.py)
- Kumar et al., IEEE TMI 2017

---

## 4. Résultats Respiratory ✅ OBJECTIF ATTEINT

### Configuration
- **Architecture:** V13 Smart Crops + FPN Chimique + H-Alpha
- **Epochs:** 60
- **Dataset:** 408 samples (Lung, Liver)

### Métriques Finales

```
Dice:        0.8470 ± 0.0564
AJI:         0.6872 ± 0.1012  ✅ > 0.68
AJI Median:  0.6814
PQ:          0.6286 ± 0.1074

Instances pred: 22.6
Instances GT:   23.1
Over-seg ratio: 0.98×
```

### Évolution des Résultats

| Configuration | AJI | Progress |
|---------------|-----|----------|
| Baseline (sans FPN) | 0.6113 | 89.9% |
| FPN Chimique 30ep | 0.6527 | 96.0% |
| FPN + Watershed optimisé | 0.6734 | 99.0% |
| **60ep + H-Alpha** | **0.6872** | **101.1%** ✅ |

### Paramètres Watershed Optimaux

| Paramètre | Valeur |
|-----------|--------|
| np_threshold | 0.40 |
| min_size | 30 |
| beta | 0.50 |
| min_distance | 5 |

---

## 5. Résultats Epidermal

### Configuration
- **Architecture:** V13 Smart Crops + FPN Chimique + H-Alpha
- **Epochs:** 60
- **Dataset:** 574 samples (Skin, HeadNeck)

### Évolution des Résultats

| Configuration | AJI | Progress | Delta |
|---------------|-----|----------|-------|
| 30 epochs baseline | 0.5868 | 86.3% | — |
| 60 epochs baseline | 0.6025 | 88.6% | +2.7% |
| 30 epochs + H-Alpha | 0.6128 | 90.1% | +1.7% |
| **60 epochs + H-Alpha** | **0.6203** | **91.2%** | +1.2% |

### Métriques Détaillées (60 epochs + H-Alpha)

```
Dice:        0.7995 ± 0.1228
AJI:         0.6203 ± 0.1364
AJI Median:  0.6202
PQ:          0.5834 ± 0.1526

Instances pred: 17.7
Instances GT:   18.7
Over-seg ratio: 0.95×
```

### Paramètres Watershed Optimaux

| Paramètre | Valeur |
|-----------|--------|
| np_threshold | 0.45 |
| min_size | 20 |
| beta | 1.00 |
| min_distance | 3 |

---

## 6. Résultats Urologic

### Configuration
- **Architecture:** V13 Smart Crops + FPN Chimique + H-Alpha
- **Epochs:** 60
- **Dataset:** 1101 samples (Kidney, Bladder, Testis, Ovarian, Uterus, Cervix)

### Évolution des Résultats

| Configuration | AJI | Progress |
|---------------|-----|----------|
| 60 epochs + H-Alpha | 0.6534 | 96.1% |
| **+ Watershed optimisé** | **0.6743** | **99.2%** |

**Gain optimization:** +3.3%

### Métriques Détaillées

```
Dice:        0.8565 ± 0.1311
AJI:         0.6743 ± 0.1642
PQ:          0.6328 ± 0.1670

Instances pred: 23.9
Instances GT:   24.3
Over-seg ratio: 1.01×
```

### Paramètres Watershed Optimaux

| Paramètre | Valeur |
|-----------|--------|
| np_threshold | 0.45 |
| min_size | 30 |
| beta | 0.50 |
| min_distance | 2 |

---

## 7. Résultats Digestive

### Configuration
- **Architecture:** V13 Smart Crops + FPN Chimique + H-Alpha
- **Epochs:** 60
- **Dataset:** 2430 samples (Colon, Stomach, Esophagus, Bile-duct)

### Évolution des Résultats

| Configuration | AJI | Progress |
|---------------|-----|----------|
| 60 epochs + H-Alpha | 0.6065 | 89.2% |
| **+ Watershed optimisé** | **0.6160** | **90.6%** |

**Gain optimization:** +3.6%

### Métriques Détaillées

```
Dice:        0.8198 ± 0.0839
AJI:         0.6160 ± 0.1471
PQ:          0.5747 ± 0.1383

Instances pred: 17.9
Instances GT:   18.1
Over-seg ratio: 0.94×
```

### Paramètres Watershed Optimaux

| Paramètre | Valeur |
|-----------|--------|
| np_threshold | 0.45 |
| min_size | 60 |
| beta | 2.00 |
| min_distance | 5 |

**Note:** Digestive nécessite `min_size=60` (le plus élevé) et `beta=2.0`, suggérant des noyaux plus grands avec des gradients HV plus marqués.

---

## 8. Paramètres Optimaux par Famille

| Famille | np_threshold | min_size | beta | min_distance | AJI | Status |
|---------|--------------|----------|------|--------------|-----|--------|
| **Respiratory** | 0.40 | 30 | 0.50 | 5 | **0.6872** | ✅ Objectif atteint |
| **Urologic** | 0.45 | 30 | 0.50 | 2 | **0.6743** | 99.2% |
| **Epidermal** | 0.45 | 20 | 1.00 | 3 | 0.6203 | 91.2% |
| **Digestive** | 0.45 | 60 | 2.00 | 5 | 0.6160 | 90.6% |

---

## 9. Règle Ajoutée à CLAUDE.md

### Règle #2: Modules Partagés OBLIGATOIRES

> **🚫 JAMAIS de duplication de code critique**
>
> Les algorithmes critiques DOIVENT être dans `src/` et importés par tous les scripts.
> **NE JAMAIS copier-coller** une fonction entre scripts — créer un module partagé.

---

## 10. Prochaines Étapes

1. **Glandular** (3391 samples) — Plus grand dataset, attendu >0.68 AJI
2. Considérer ajout de l'AJI original pour comparaison littérature
