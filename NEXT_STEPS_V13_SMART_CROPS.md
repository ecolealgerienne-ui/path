# V13 Smart Crops - Prochaines Étapes (TRUE Instance Evaluation)

## Contexte

✅ **Code modifié et committé** (commit fe223fb):
- `prepare_v13_smart_crops.py`: Sauvegarde inst_maps dans train/val splits
- `test_v13_smart_crops_aji.py`: Utilise inst_maps TRUE au lieu de reconstruction watershed

## Problème Résolu

**AVANT (BIAISÉ)**:
```python
# Evaluation comparait pseudo-instances vs prédictions
gt_inst = watershed(HV_GT_HYBRID)  # ❌ Pseudo-instances
aji = compute_aji(pred_inst, gt_inst)
```

**APRÈS (CORRECT)**:
```python
# Evaluation utilise VRAIES instances PanNuke
gt_inst = inst_maps[i]  # ✅ Instances réelles cropées avec HYBRID
aji = compute_aji(pred_inst, gt_inst)
```

## Étapes d'Exécution

### Étape 1: Régénérer Données VAL avec inst_maps (5 min)

```bash
# Activer environnement
conda activate cellvit

# Régénérer train + val splits avec inst_maps
python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal
```

**Sortie attendue**:
```
data/family_data_v13_smart_crops/
├── epidermal_train_v13_smart_crops.npz  (~800 MB)
│   ├── images: (N_train, 224, 224, 3)
│   ├── np_targets: (N_train, 224, 224)
│   ├── hv_targets: (N_train, 2, 224, 224)
│   ├── nt_targets: (N_train, 224, 224)
│   ├── inst_maps: (N_train, 224, 224) int32  ✅ NOUVEAU
│   └── metadata...
└── epidermal_val_v13_smart_crops.npz    (~200 MB)
    └── (même structure)
```

### Étape 2: Ré-évaluer avec TRUE Instances (5 min)

```bash
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_best.pth \
    --family epidermal \
    --n_samples 50
```

**Métriques attendues**:

| Métrique | Avant (BIAISÉ) | Après (TRUE) | Objectif |
|----------|---------------|--------------|----------|
| Dice | 0.7683 | ~0.76-0.80 | Maintenu |
| **AJI** | **0.5759** | **≥0.68** 🎯 | **+18%** |
| PQ | 0.5094 | ≥0.62 | +20% |
| Over-seg | 1.10× | ~0.95× | Optimal |

### Étape 3: Analyser Résultats

Si **AJI ≥0.68** ✅:
- HYBRID approach VALIDÉ
- Objectif atteint (+18% vs baseline 0.5529)
- Extension aux 4 autres familles

Si **0.60 ≤ AJI < 0.68** ⚠️:
- Proche objectif
- Tuning watershed parameters (beta, min_size)
- Possible avec `scripts/evaluation/optimize_watershed_params.py`

Si **AJI < 0.60** ❌:
- Diagnostic approfondi nécessaire
- Vérifier HV magnitude et gradients
- Possible problème HV targets HYBRID

## Validation Data Integrity

Avant évaluation, vérifier que inst_maps sont bien sauvegardés:

```bash
python -c "
import numpy as np
data = np.load('data/family_data_v13_smart_crops/epidermal_val_v13_smart_crops.npz')
print('Keys:', list(data.keys()))
print('inst_maps shape:', data['inst_maps'].shape)
print('inst_maps dtype:', data['inst_maps'].dtype)
print('Unique instances (sample 0):', len(np.unique(data['inst_maps'][0])) - 1)  # -1 for background
"
```

**Sortie attendue**:
```
Keys: ['images', 'np_targets', 'hv_targets', 'nt_targets', 'inst_maps', ...]
inst_maps shape: (N_val, 224, 224)
inst_maps dtype: int32
Unique instances (sample 0): 5-15  # Variable selon densité cellulaire
```

## Temps Total Estimé

- Régénération données: ~5 min
- Validation integrity: ~1 min
- Ré-évaluation AJI: ~5 min
- **Total: ~11 minutes**

## Fichiers Modifiés (Commit fe223fb)

| Fichier | Modifications |
|---------|--------------|
| `prepare_v13_smart_crops.py` | +inst_map return, rotation, saving |
| `test_v13_smart_crops_aji.py` | -watershed GT loop, +inst_maps loading |

## Raison du Fix

**Citation initiale**:
> "Pourquoi tu n'utilise pas les données de VAL, déjà calculer et enregistrer? Inutilie de repartir de 0 et refaire tout le calcul avec le risque d'erreur."

✅ **Solution pragmatique adoptée**: ENRICHIR les données VAL existantes avec inst_maps (déjà calculés lors du cropping HYBRID) au lieu de repartir from scratch.

## Documentation Mise à Jour

Après validation, mettre à jour `CLAUDE.md` section Journal de Développement avec:
- Date: 2025-12-27
- Résultats AJI TRUE vs BIAISÉ
- Décision sur extension multi-familles

---

**Status**: ⏳ En attente exécution par utilisateur avec environnement Python/GPU/données
