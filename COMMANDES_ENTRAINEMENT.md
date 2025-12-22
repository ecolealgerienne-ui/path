# Commandes d'Entraînement - CellViT-Optimus

**Date:** 2025-12-22
**Objectif:** Ré-entraîner 5 familles HoVer-Net avec données FIXED (float32)

---

## Phase 2: Extraction Features (~30 min)

### Fold 0
```bash
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 0 \
    --batch_size 8 \
    --chunk_size 300
```

### Fold 1
```bash
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 1 \
    --batch_size 8 \
    --chunk_size 300
```

### Fold 2
```bash
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 2 \
    --batch_size 8 \
    --chunk_size 300
```

### Validation Features
```bash
python scripts/validation/verify_features.py --features_dir data/cache/pannuke_features
```

**Attendu:** CLS std ~0.77 (entre 0.70-0.90)

---

## Phase 3: Backup Anciens Checkpoints (optionnel mais recommandé)

```bash
cp -r models/checkpoints models/checkpoints_OLD_20251222
```

---

## Phase 4: Entraînement par Famille (~2h total)

### 1. Famille Glandular (~25 min)
**Organes:** Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland
**Samples:** 3391

```bash
python scripts/training/train_hovernet_family.py \
    --family glandular \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Performances attendues:**
- NP Dice: ~0.96
- HV MSE: ~0.01
- NT Acc: ~0.91

---

### 2. Famille Digestive (~20 min)
**Organes:** Colon, Stomach, Esophagus, Bile-duct
**Samples:** 2430

```bash
python scripts/training/train_hovernet_family.py \
    --family digestive \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Performances attendues:**
- NP Dice: ~0.96
- HV MSE: ~0.02
- NT Acc: ~0.88

---

### 3. Famille Urologic (~15 min)
**Organes:** Kidney, Bladder, Testis, Ovarian, Uterus, Cervix
**Samples:** 1101

```bash
python scripts/training/train_hovernet_family.py \
    --family urologic \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Performances attendues:**
- NP Dice: ~0.93
- HV MSE: ~0.28 (peu de données)
- NT Acc: ~0.91

---

### 4. Famille Epidermal (~10 min)
**Organes:** Skin, HeadNeck
**Samples:** 571

```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Performances attendues:**
- NP Dice: ~0.95
- HV MSE: ~0.27 (peu de données)
- NT Acc: ~0.89

---

### 5. Famille Respiratory (~10 min)
**Organes:** Lung, Liver
**Samples:** 408

```bash
python scripts/training/train_hovernet_family.py \
    --family respiratory \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Performances attendues:**
- NP Dice: ~0.94
- HV MSE: ~0.05 (surprise positive!)
- NT Acc: ~0.92

---

## Phase 5: Validation (~10 min)

### Glandular
```bash
python scripts/evaluation/test_on_training_data.py \
    --family glandular \
    --checkpoint models/checkpoints/hovernet_glandular_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data_FIXED
```

### Digestive
```bash
python scripts/evaluation/test_on_training_data.py \
    --family digestive \
    --checkpoint models/checkpoints/hovernet_digestive_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data_FIXED
```

### Urologic
```bash
python scripts/evaluation/test_on_training_data.py \
    --family urologic \
    --checkpoint models/checkpoints/hovernet_urologic_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data_FIXED
```

### Epidermal
```bash
python scripts/evaluation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data_FIXED
```

### Respiratory
```bash
python scripts/evaluation/test_on_training_data.py \
    --family respiratory \
    --checkpoint models/checkpoints/hovernet_respiratory_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data_FIXED
```

**Critères de succès:**
- NP Dice proche du train (écart < 2%)
- HV MSE proche du train (écart < 20%)
- NT Acc proche du train (écart < 3%)

---

## Phase 6: Cleanup (APRÈS validation OK)

### Vérifier taille
```bash
du -sh data/cache/family_data_OLD_int8_*
```

### Supprimer
```bash
rm -rf data/cache/family_data_OLD_int8_*
```

**Libération:** ~10-15 GB

---

## Notes RAM

- **Extraction features:** Pic ~6 GB par fold (chunk_size=300)
- **Entraînement:** Pic ~11 GB par famille (données en RAM)
- **Validation:** Pic ~4 GB (10 samples)

**Total RAM requis:** 12 GB disponibles → ✅ OK

---

## Ordre d'Exécution

1. ✅ Phase 2: Extraction features (folds 0, 1, 2)
2. ✅ Validation features (CLS std ~0.77)
3. ✅ Backup checkpoints (optionnel)
4. ✅ Phase 4: Entraînement (5 familles séquentiellement)
5. ✅ Phase 5: Validation (test sur 10 samples)
6. ✅ Phase 6: Cleanup (après validation OK)

**Temps total:** ~2h30
