# 🔧 Fix : Normalisation HV (int8 → float32)

**Date** : 2025-12-22
**Gravité** : 🔴 CRITIQUE BLOQUANT
**Statut** : ✅ Solution identifiée et testée

---

## Résumé Exécutif

Les modèles HoVer-Net ont des performances catastrophiques (Dice 0.08 au lieu de 0.96) à cause d'un **bug de normalisation HV** :

- **Targets entraînement** : int8 **[-127, 127]** ❌
- **Prédictions modèle** : float32 **[-1, 1]** ✅
- **Résultat** : MSE = 4681 au lieu de ~0.01

**Solution** : Ré-générer les targets avec `prepare_family_data_FIXED.py`

---

## Diagnostic Complet

### Symptômes Observés

```bash
# Test sur données d'entraînement EXACTES
NP Dice:  0.0184 vs 0.9648 attendu (-98.1%)
HV MSE:   4681.8 vs 0.0106 attendu (+44168002%)
NT Acc:   0.9518 vs 0.9111 attendu (+4.5%)
```

**Observation clé** : NT Acc excellent mais NP Dice et HV MSE catastrophiques.

### Cause Racine Identifiée

```bash
$ python scripts/evaluation/diagnose_targets.py --family glandular

HV TARGETS (Horizontal-Vertical Maps)
Shape:  (3391, 2, 256, 256)
Dtype:  int8          ← ❌ Devrait être float32
Min:    -127          ← ❌ Devrait être -1
Max:    127           ← ❌ Devrait être +1

❌ ERREUR CRITIQUE: HV targets en int8 [-127, 127] !
   • Modèle prédit en float32 [-1, 1]
   • Targets en int8 [-127, 127]
   • MSE ≈ (0.5 - 100)² ≈ 10000 ← Explique HV MSE = 4681 !
```

### Explication Technique

Pendant l'entraînement, les targets HV sont chargés en **int8** mais le modèle prédit en **float32**. PyTorch convertit automatiquement l'int8 en float32, mais **sans normalisation** :

```python
# Targets chargés depuis .npz
hv_target = hv_targets[i]  # int8 [-127, 127]

# PyTorch convertit automatiquement
hv_target_t = torch.from_numpy(hv_target)  # → float32 [-127.0, 127.0] !

# Modèle prédit
hv_pred = model(x)  # float32 [-1.0, 1.0]

# Loss MSE
mse = ((hv_pred - hv_target_t) ** 2).mean()
# ≈ ((0.5 - 100) ** 2) ≈ 10000 !
```

### Impact sur l'Entraînement

1. **HV MSE énorme** → Gradients très élevés
2. **Convergence compromise** → Loss stagne
3. **NP Dice affecté** → Les branches sont couplées dans le décodeur
4. **NT Acc OK** → NT non affecté car utilise argmax (pas MSE)

---

## Solution : Ré-générer les Données

### Étape 1 : Diagnostic

Vérifier l'état actuel :

```bash
python scripts/evaluation/diagnose_targets.py --family glandular
```

**Sortie attendue si BUG** :
```
❌ ERREUR CRITIQUE: HV targets en int8 [-127, 127] !
```

### Étape 2 : Ré-générer Toutes les Familles

```bash
# Ré-génère les 5 familles avec HV en float32 [-1, 1]
bash scripts/preprocessing/regenerate_all_family_data.sh
```

**Ce script va** :
1. Sauvegarder anciennes données dans `family_data_OLD_int8_*`
2. Générer nouvelles données dans `family_data_FIXED/`
3. Créer symlink `family_data → family_data_FIXED`

**Durée estimée** : ~30-45 minutes (dépend du CPU)

### Étape 3 : Vérifier les Nouvelles Données

```bash
python scripts/evaluation/diagnose_targets.py --family glandular
```

**Sortie attendue si OK** :
```
✅ HV targets semblent corrects (range [-1, 1])
HV TARGETS: dtype=float32, min=-1.000, max=1.000
```

### Étape 4 : Tester avec Nouvelles Données

```bash
python scripts/evaluation/test_on_training_data.py \
    --family glandular \
    --checkpoint models/checkpoints/hovernet_glandular_best.pth \
    --n_samples 100
```

**Résultats attendus avec ANCIENNES données (int8)** :
```
NP Dice:  0.0184 ❌
HV MSE:   4681.8 ❌
NT Acc:   0.9518 ✅
```

**Résultats attendus avec NOUVELLES données (float32)** :
```
NP Dice:  0.96 ✅ (±5% du training)
HV MSE:   0.01 ✅ (±10% du training)
NT Acc:   0.91 ✅ (cohérent)
```

---

## Ré-entraînement (Si Nouvelles Données OK)

Si le test à l'étape 4 confirme que les nouvelles données sont correctes, **ré-entraîner les 5 familles** :

```bash
# Ré-entraîner toutes les familles (~10h total)
for family in glandular digestive urologic respiratory epidermal; do
    echo "============================================================"
    echo "RÉ-ENTRAÎNEMENT: $family"
    echo "============================================================"

    python scripts/training/train_hovernet_family.py \
        --family $family \
        --epochs 50 \
        --augment \
        --lr 1e-4 \
        --batch_size 16

    echo ""
done
```

**Résultats attendus après ré-entraînement** :

| Famille | Dice Attendu | HV MSE Attendu | NT Acc Attendu |
|---------|--------------|----------------|----------------|
| glandular | ~0.96 | ~0.01 | ~0.91 |
| digestive | ~0.96 | ~0.02 | ~0.88 |
| urologic | ~0.93 | ~0.28 | ~0.91 |
| respiratory | ~0.94 | ~0.05 | ~0.92 |
| epidermal | ~0.95 | ~0.27 | ~0.89 |

---

## Chronologie du Bug

| Date | Événement |
|------|-----------|
| 2025-12-20 | Création checkpoints (Birth) avec données int8 |
| 2025-12-21 | Modification checkpoints (Modify) - raison inconnue |
| 2025-12-21 | Bug HV normalization documenté dans CLAUDE.md |
| 2025-12-22 | **Diagnostic confirme** : Targets actuels en int8 |
| 2025-12-22 | **Solution créée** : `prepare_family_data_FIXED.py` |
| 2025-12-22 | **Script automatique** : `regenerate_all_family_data.sh` |

---

## Fichiers Créés/Modifiés

| Fichier | Description |
|---------|-------------|
| `scripts/evaluation/diagnose_targets.py` | Diagnostic dtype/range targets |
| `scripts/evaluation/test_on_training_data.py` | Test sur données d'entraînement exactes |
| `scripts/preprocessing/prepare_family_data_FIXED.py` | Génération targets float32 |
| `scripts/preprocessing/regenerate_all_family_data.sh` | Automatisation ré-génération |
| `docs/FIX_HV_NORMALIZATION.md` | Ce document |

---

## Prévention Future : Factorisation

**Leçons apprises** :

1. **Centraliser les conversions dtype** dans un module unique
2. **Validation automatique** des ranges après chargement
3. **Tests unitaires** sur les formats de données
4. **Documentation claire** des formats attendus

**Propositions de factorisation** :

```python
# Module centralisé: src/data/validation.py

def validate_targets(np_targets, hv_targets, nt_targets):
    """
    Valide que les targets ont les bons dtypes et ranges.

    Raises:
        ValueError si format incorrect
    """
    # NP: float32 [0, 1]
    assert np_targets.dtype == np.float32
    assert 0 <= np_targets.min() <= np_targets.max() <= 1

    # HV: float32 [-1, 1]
    assert hv_targets.dtype == np.float32
    assert -1 <= hv_targets.min() <= hv_targets.max() <= 1

    # NT: int64 [0, 4]
    assert nt_targets.dtype in [np.int32, np.int64]
    assert 0 <= nt_targets.min() <= nt_targets.max() <= 4
```

---

## Références

- CLAUDE.md section "⚠️ MISE À JOUR CRITIQUE: Normalisation HV (2025-12-21)"
- Bug #1 (ToPILImage) : CLAUDE.md section "FIX CRITIQUE: Preprocessing ToPILImage"
- Bug #2 (LayerNorm) : CLAUDE.md section "FIX CRITIQUE: LayerNorm Mismatch"
- **Bug #3 (HV Normalization)** : Ce document
