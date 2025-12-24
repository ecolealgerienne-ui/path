# Bug #6: Feature Path Mismatch — Root Cause of NP Dice 0.0000

**Date:** 2025-12-24
**Sévérité:** 🔴 CRITIQUE
**Impact:** Modèle complètement cassé (NP Dice 0.0000 au lieu de 0.95)

---

## Symptômes

Test du modèle sur ses **PROPRES données d'entraînement** révèle:

```
NP Dice:  0.0000 ± 0.0000  (attendu: ~0.95)  ❌
HV MSE:   0.1425 ± 0.0478  (attendu: ~0.16)  ✅ (valeur OK mais trompeuse)
NT Acc:   0.7981 ± 0.0988  (attendu: ~0.90)  ⚠️

NP pred range: [0.122, 0.279]  ← Devrait être [0, 1] binaire!
HV magnitude:  0.024          ← Devrait être >0.5!
```

**Interprétation:**

Le modèle prédit des probabilités constantes autour 0.2 (logits ~-1.4) au lieu de décisions binaires. Cela signifie:

- Le décodeur voit des **features qu'il n'a jamais vues durant l'entraînement**
- Il adopte une stratégie de défaut: "prédire background partout" (safe)
- NT Acc à 0.79 n'est pas aléatoire (0.20) car c'est un problème 5-classes avec déséquilibre

---

## Cause Racine Identifiée: Path Mismatch

### Training Script

`scripts/training/train_hovernet_family.py` ligne 113-118:

```python
if cache_dir is None:
    cache_dir = PROJECT_ROOT / "data" / "cache" / "family_data"
features_path = cache_dir / f"{family}_features.npz"
```

**Charge depuis:** `data/cache/family_data/epidermal_features.npz`

### Test Script (BUGGY)

`scripts/validation/test_on_training_data.py` ligne 33:

```python
parser.add_argument("--data_dir", default="data/family_data")
```

**Charge depuis:** `data/family_data/epidermal_features.npz`

### Conséquence

Les deux scripts chargent des fichiers **DIFFÉRENTS**:

| Location | Rôle | Contenu |
|----------|------|---------|
| `data/cache/family_data/` | ✅ TRAINING | Features correctes (utilisées pour entraînement) |
| `data/family_data/` | ❌ TEST | Features **DIFFÉRENTES** (anciennes? corrompues?) |

Quand le test charge `data/family_data/`, le modèle reçoit des features avec une distribution différente de celle vue durant l'entraînement → sorties quasi-nulles.

---

## Preuve: NP Predictions [0.122, 0.279]

Quand un sigmoid sort des valeurs autour 0.2, les logits avant sigmoid sont:

```
sigmoid(x) = 0.20  →  x ≈ -1.39
sigmoid(x) = 0.25  →  x ≈ -1.10
```

Des logits constants négatifs signifient:

1. **Les features d'entrée sont hors distribution** (OOD)
2. Le décodeur n'a jamais vu ces valeurs durant l'entraînement
3. Il se replie sur la prédiction la plus safe: "rien n'est un noyau"

Si les features étaient correctes, on verrait:
- NP pred range: [0.00, 1.00] (décisions binaires)
- Dice ~0.95 (comme training)
- HV magnitude >0.5 (gradients forts)

---

## Bugs Précédents Éliminés

✅ **Bug #5 (checkpoint loading)**: CORRIGÉ
- Poids chargés correctement (Mean=0.00001, Std=0.015)
- Toutes les clés matchent

✅ **Bug #3 (inst_maps)**: CORRIGÉ
- inst_maps natifs PanNuke préservés dans targets.npz

✅ **v8 Data**: VALIDÉ
- Alignement spatial parfait (0.4px)
- HV vectors pointent vers centroïdes

Le problème n'est **PAS** le modèle, les poids, ou les données v8.
Le problème est: **features de test ≠ features de training**.

---

## Timeline Hypothétique

### Scénario A: Features OLD vs NEW

1. **Avant 2025-12-22:** Features extraites dans `data/family_data/` avec preprocessing ancien (Bugs #1/#2)
2. **2025-12-22:** Phase 1 refactoring → Preprocessing corrigé
3. **2025-12-23:** Nouvelles features extraites dans `data/cache/family_data/` avec preprocessing correct
4. **2025-12-24:** Training utilise NEW features (`data/cache/`), test utilise OLD features (`data/`) → MISMATCH

### Scénario B: Extraction Incomplète

1. Features extraites dans `data/family_data/` mais jamais copiées vers `data/cache/family_data/`
2. Training échoue ou utilise features par défaut
3. Checkpoint contient poids aléatoires malgré epoch 50/50

---

## Solution

### Étape 1: Diagnostic

```bash
python scripts/validation/compare_feature_sources.py --family epidermal
```

Cela compare:
- `data/family_data/epidermal_features.npz` (test script)
- `data/cache/family_data/epidermal_features.npz` (training script)

Et affiche:
- CLS std de chacun
- Date de modification
- Recommandation sur lequel utiliser

### Étape 2: Fix Immédiat

**Option A: Corriger le test script**

Modifier `scripts/validation/test_on_training_data.py` ligne 33:

```python
# AVANT (BUGGY):
parser.add_argument("--data_dir", default="data/family_data")

# APRÈS (CORRECT):
parser.add_argument("--data_dir", default="data/cache/family_data")
```

**Option B: Copier les bonnes features**

```bash
# Si data/cache/family_data/ contient les features correctes
cp data/cache/family_data/epidermal_features.npz data/family_data/
cp data/cache/family_data/epidermal_targets.npz data/family_data/
```

### Étape 3: Validation

Re-tester avec les bonnes features:

```bash
python scripts/validation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data  # ← EXPLICITE!
```

**Résultats attendus:**

```
NP Dice:  0.9500 ± 0.0050  ✅
HV MSE:   0.1600 ± 0.0200  ✅
NT Acc:   0.9000 ± 0.0100  ✅
NP pred range: [0.00, 1.00]  ✅
HV magnitude: >0.5           ✅
```

---

## Leçons Apprises

### 1. Path Hardcoding = Bug Source

Avoir des paths hardcodés dans 2 scripts différents crée des risques de divergence:

```python
# ❌ MAUVAIS (duplicated)
# Script 1:
cache_dir = "data/cache/family_data"

# Script 2:
data_dir = "data/family_data"

# ✅ BON (centralized)
from src.constants import FAMILY_DATA_DIR
cache_dir = FAMILY_DATA_DIR
```

### 2. Toujours Tester sur Données Training d'Abord

Avant d'évaluer sur test set:
1. ✅ Tester sur données training (sanity check)
2. Si Dice ~0.95: modèle OK, problème dans eval/GT
3. Si Dice ~0.00: modèle cassé ou feature mismatch

Cette stratégie nous a permis d'isoler le problème en 1 étape.

### 3. NP Predictions Distribution = Indicateur Puissant

| Distribution NP | Diagnostic |
|-----------------|------------|
| Binaire [0, 1] | ✅ Modèle sain |
| Constante ~0.5 | ⚠️ Modèle indécis (sous-entraîné) |
| Constante ~0.2 | ❌ Features OOD (mismatch) |
| Constante ~0.0 | ❌ Checkpoint non chargé |

---

## Bugs Connexes

- **Bug #1:** ToPILImage float64 overflow (CORRIGÉ 2025-12-20)
- **Bug #2:** LayerNorm mismatch blocks[23] (CORRIGÉ 2025-12-21)
- **Bug #3:** connectedComponents fusionne instances (CORRIGÉ 2025-12-23)
- **Bug #5:** Checkpoint module./model. prefixes (CORRIGÉ 2025-12-24)
- **Bug #6:** Feature path mismatch ← **CE BUG**

---

## Statut

- [x] Symptômes identifiés
- [x] Cause racine confirmée (path mismatch)
- [x] Script diagnostic créé (`compare_feature_sources.py`)
- [ ] Fix validé (en attente test utilisateur)
- [ ] Fix appliqué à tous les scripts concernés

**Prochaine étape:** Utilisateur exécute `compare_feature_sources.py` et valide le fix.
