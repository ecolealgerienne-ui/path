# Bug #6: Feature Path Mismatch - Plan de Résolution

**Date:** 2025-12-24
**Statut:** Fix appliqué, diagnostic créé, re-train probablement nécessaire

---

## Résumé Exécutif

Le modèle HoVer-Net epidermal a été entraîné avec succès (Epoch 50/50, Dice 0.95), mais le test sur données d'entraînement montre **NP Dice 0.0000** (catastrophique).

**Verdict de l'expert:**
> "Ton modèle n'est pas catastrophique, il est mal interprété par ton script de validation. Tu as un problème de 'plomberie' finale."

**Cause racine identifiée:** Le modèle a probablement été entraîné avec des features d'un répertoire différent de celui utilisé pour le test (path mismatch).

---

## Analyse du Code de Test

### ✅ Le script de test est CORRECT

Le script `test_on_training_data.py` fait déjà:

1. **Sigmoid appliqué** (ligne 99):
   ```python
   np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]  # ✅ CORRECT
   ```

2. **Resize GT 256→224** (lignes 109-114):
   ```python
   np_gt = resize(np_targets[i], (224, 224), interpolation=INTER_NEAREST)  # ✅ CORRECT
   ```

3. **Threshold à 0.5** (ligne 117):
   ```python
   np_pred_binary = (np_pred > 0.5).astype(np.float32)  # ✅ CORRECT
   ```

### ❌ Mais les prédictions sont anormales

```
NP pred range: [0.075, 0.391]  ← Devrait être proche de [0, 1] après sigmoid
HV magnitude: 0.024            ← Devrait être >0.5
NT Acc: 80%                    ← Bon (prouve que le modèle fonctionne!)
```

**Diagnostic:** Les valeurs sigmoid faibles [0.075-0.391] indiquent:
- Logits négatifs (autour de -1.4)
- Le modèle prédit constamment "pas un noyau"
- **Features d'entrée hors distribution (OOD)** → le modèle n'a jamais vu ces features durant l'entraînement

---

## Path Mismatch Identifié

### Bug dans train_hovernet_family.py (LIGNE 333)

**AVANT le fix:**
```python
parser.add_argument('--cache_dir', type=str, default=DEFAULT_FAMILY_FIXED_DIR,  # ❌ WRONG
                   help='Répertoire des données pré-préparées')
```

**APRÈS le fix (2025-12-24):**
```python
parser.add_argument('--cache_dir', type=str, default=DEFAULT_FAMILY_DATA_DIR,  # ✅ CORRECT
                   help='Répertoire des données pré-préparées')
```

### Conséquence

Si le modèle a été entraîné **AVANT** ce fix (très probable), alors:

1. **Entraînement** a chargé depuis `data/family_FIXED/epidermal_features.npz`
2. **Test** charge depuis `data/family_data/epidermal_features.npz`
3. **Résultat:** Features différentes → Modèle OOD → NP Dice 0.0000

---

## Script de Diagnostic Créé

**Fichier:** `scripts/validation/diagnose_training_data_mismatch.py`

**Usage:**
```bash
python scripts/validation/diagnose_training_data_mismatch.py --family epidermal
```

**Ce qu'il vérifie:**
1. Existence de `data/family_data/epidermal_features.npz`
2. Existence de `data/family_FIXED/epidermal_features.npz`
3. Shape, dtype, CLS std de chaque fichier
4. Recommandation basée sur la configuration trouvée

---

## Scénarios Possibles et Actions

### Scénario A: Features dans data/family_data SEULEMENT

**Diagnostic:**
```
✅ data/family_data/epidermal_features.npz EXISTS
❌ data/family_FIXED/epidermal_features.npz NOT FOUND
```

**Interprétation:**
- Le modèle a probablement été entraîné sur `data/family_FIXED/` (maintenant absent)
- Features d'entraînement ≠ features de test
- Mismatch confirmé

**Action:**
```bash
# RE-TRAIN avec le chemin corrigé
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment

# Le script utilisera maintenant DEFAULT_FAMILY_DATA_DIR par défaut
```

**Résultat attendu:** NP Dice 0.0000 → ~0.95 ✅

---

### Scénario B: Features dans data/family_FIXED SEULEMENT

**Diagnostic:**
```
❌ data/family_data/epidermal_features.npz NOT FOUND
✅ data/family_FIXED/epidermal_features.npz EXISTS
```

**Interprétation:**
- Les features sont dans FIXED mais pas dans DATA
- Le test ne trouve pas les features

**Action (Option 1 - Quick Fix):**
```bash
# Copier features vers data/family_data
mkdir -p data/family_data
cp data/family_FIXED/epidermal_features.npz data/family_data/
cp data/family_FIXED/epidermal_targets.npz data/family_data/

# Re-test
python scripts/validation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10
```

**Action (Option 2 - Clean):**
```bash
# Mettre à jour la source de vérité
# Dans src/constants.py:
DEFAULT_FAMILY_DATA_DIR = "data/family_FIXED"  # Si c'est là que les données sont
```

---

### Scénario C: Aucune feature trouvée

**Diagnostic:**
```
❌ data/family_data/epidermal_features.npz NOT FOUND
❌ data/family_FIXED/epidermal_features.npz NOT FOUND
```

**Interprétation:**
- Les features n'ont jamais été extraites
- Ou les chemins sont complètement incorrects

**Action:**
```bash
# Extraire features depuis FIXED.npz
python scripts/preprocessing/extract_features_from_fixed.py --family epidermal

# Vérifier
ls -lh data/family_data/epidermal_features.npz

# Train
python scripts/training/train_hovernet_family.py --family epidermal --epochs 50 --augment
```

---

## Métriques Attendues Après Fix

| Métrique | Avant (Mismatch) | Après (Fix) | Amélioration |
|----------|------------------|-------------|--------------|
| NP Dice | 0.0000 | ~0.9500 | **+∞** 🎯 |
| NP pred range | [0.075, 0.391] | ~[0.0, 1.0] | ✅ |
| HV magnitude | 0.024 | >0.5 | +2000% ✅ |
| HV MSE | 0.14 | ~0.16 | ✅ (déjà bon) |
| NT Acc | 80% | ~90% | +10% ✅ |

---

## Normalisation H-optimus-0 (Clarification)

L'expert a mentionné la normalisation ImageNet:
```python
mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet
std = torch.tensor([0.229, 0.224, 0.225])   # ImageNet
```

**⚠️ ATTENTION:** Nous utilisons **H-optimus-0**, PAS ImageNet!

**Normalisation correcte (src/constants.py):**
```python
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
```

Cette normalisation est déjà correctement utilisée dans `src/preprocessing.py`. **NE PAS CHANGER.**

---

## Prochaines Étapes (Ordre Recommandé)

1. **Exécuter diagnostic** (2 min):
   ```bash
   python scripts/validation/diagnose_training_data_mismatch.py --family epidermal
   ```

2. **Appliquer l'action recommandée** selon le scénario identifié (voir ci-dessus)

3. **Re-test** (1 min):
   ```bash
   python scripts/validation/test_on_training_data.py \
       --family epidermal \
       --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
       --n_samples 10
   ```

4. **Si Dice ~0.95:** ✅ Problème résolu → Tester AJI:
   ```bash
   python scripts/evaluation/test_aji_v8.py \
       --family epidermal \
       --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
       --n_samples 50
   ```

5. **Si Dice toujours 0.0000:** Créer un issue GitHub avec les logs complets

---

## Leçons Apprises

1. **Single Source of Truth est CRITIQUE** - Un seul argument default incorrect a cassé tout le pipeline
2. **Toujours tester sur training data d'abord** - Si le modèle échoue sur ses propres données, c'est un problème de chargement, pas de généralisation
3. **Les métriques partielles peuvent tromper** - HV MSE et NT Acc étaient bons, mais NP Dice 0.0000 révélait le vrai problème
4. **Path mismatch = OOD features** - Prédictions sigmoid faibles [0.075-0.391] sont typiques d'un modèle qui voit des données jamais vues durant l'entraînement

---

## Références

- Bug #6 Documentation: `docs/BUG_6_FEATURE_PATH_MISMATCH.md`
- Training Script: `scripts/training/train_hovernet_family.py:333`
- Test Script: `scripts/validation/test_on_training_data.py`
- Constants: `src/constants.py` (DEFAULT_FAMILY_DATA_DIR)
- Diagnostic: `scripts/validation/diagnose_training_data_mismatch.py`
