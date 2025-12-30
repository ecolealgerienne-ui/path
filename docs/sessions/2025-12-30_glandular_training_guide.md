# Session 2025-12-30 — Entraînement Glandular (Dernière Famille)

## Contexte

**Famille Glandular** est la dernière famille à entraîner pour compléter le projet CellViT-Optimus.

| Propriété | Valeur |
|-----------|--------|
| **Organes** | Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland |
| **Samples** | 3391 (le plus grand dataset) |
| **Objectif** | AJI ≥ 0.68 |

---

## État des Familles (4/5 Complétées)

| Famille | Samples | AJI | Status |
|---------|---------|-----|--------|
| **Respiratory** | 408 | **0.6872** | ✅ Objectif atteint |
| **Urologic** | 1101 | **0.6743** | 99.2% |
| Epidermal | 574 | 0.6203 | 91.2% |
| Digestive | 2430 | 0.6160 | 90.6% |
| **Glandular** | 3391 | **À TESTER** | — |

---

## Pipeline Complet pour Glandular

### Étape 1: Normalisation Macenko

```bash
python scripts/preprocessing/normalize_staining_source.py --family glandular
```

### Étape 2: Préparation Smart Crops

```bash
python scripts/preprocessing/prepare_v13_smart_crops.py \
    --family glandular \
    --max_samples 5000
```

### Étape 3: Vérification des Données

```bash
# Vérifier les fichiers générés
ls -la data/family_data_v13_smart_crops/

# Vérifier split train
python scripts/validation/verify_v13_smart_crops_data.py --family glandular --split train

# Vérifier split val
python scripts/validation/verify_v13_smart_crops_data.py --family glandular --split val
```

### Étape 4: Extraction Features H-optimus-0

```bash
# Train features
python scripts/preprocessing/extract_features_v13_smart_crops.py --family glandular --split train

# Val features
python scripts/preprocessing/extract_features_v13_smart_crops.py --family glandular --split val

# Vérifier les features
ls -la data/cache/family_data/
```

### Étape 5: Entraînement FPN Chimique + H-Alpha

```bash
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family glandular \
    --epochs 60 \
    --use_hybrid \
    --use_fpn_chimique \
    --use_h_alpha
```

**Paramètres:**
- `--use_hybrid`: Active le mode hybride (features + RGB)
- `--use_fpn_chimique`: Active l'injection multi-échelle H-channel
- `--use_h_alpha`: Active le paramètre α learnable

**Checkpoint généré:**
```
models/checkpoints_v13_smart_crops/hovernet_glandular_v13_smart_crops_hybrid_fpn_best.pth
```

### Étape 6: Optimisation Watershed

```bash
python scripts/evaluation/optimize_watershed_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_glandular_v13_smart_crops_hybrid_fpn_best.pth \
    --family glandular \
    --n_samples 50
```

Cette commande effectue un grid search sur:
- `np_threshold`: [0.35, 0.40, 0.45, 0.50]
- `min_size`: [20, 30, 40, 60]
- `beta`: [0.5, 1.0, 2.0]
- `min_distance`: [2, 3, 5]

### Étape 7: Évaluation Finale

Après l'optimisation watershed, utiliser les paramètres trouvés:

```bash
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_glandular_v13_smart_crops_hybrid_fpn_best.pth \
    --family glandular \
    --n_samples 50 \
    --use_hybrid \
    --use_fpn_chimique \
    --np_threshold <OPTIMIZED> \
    --min_size <OPTIMIZED> \
    --beta <OPTIMIZED> \
    --min_distance <OPTIMIZED>
```

---

## Paramètres de Référence (Autres Familles)

Pour référence, voici les paramètres watershed optimaux des autres familles:

| Famille | np_threshold | min_size | beta | min_distance |
|---------|--------------|----------|------|--------------|
| Respiratory | 0.40 | 30 | 0.50 | 5 |
| Urologic | 0.45 | 30 | 0.50 | 2 |
| Epidermal | 0.45 | 20 | 1.00 | 3 |
| Digestive | 0.45 | 60 | 2.00 | 5 |

**Note:** Glandular a des caractéristiques similaires à Digestive (tissus glandulaires avec acini), donc les paramètres initiaux à essayer pourraient être proches.

---

## Attentes

Avec 3391 samples (le plus grand dataset), on s'attend à:
- Meilleur apprentissage grâce au volume de données
- AJI potentiellement > 0.68 (objectif)
- Bonne généralisation sur les 5 organes glandulaires

---

## Points Importants

1. **Modules partagés OBLIGATOIRES:**
   ```python
   from src.postprocessing import hv_guided_watershed
   from src.evaluation import run_inference, evaluate_batch_with_params
   ```

2. **FPN Chimique nécessite toujours:**
   ```bash
   --use_hybrid --use_fpn_chimique
   ```

3. **Temps estimé:**
   - Extraction features: ~30-45 min (3391 samples × 2 splits)
   - Entraînement 60 epochs: ~2-3 heures
   - Optimisation watershed: ~15-20 min
