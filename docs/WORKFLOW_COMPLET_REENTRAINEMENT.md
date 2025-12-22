# Workflow Complet de Ré-entraînement CellViT-Optimus

**Date:** 2025-12-22
**Objectif:** Ré-entraîner tous les modèles avec données corrigées (uint8 + float32)
**Temps total estimé:** ~3h (GPU rapide)

---

## ✅ Phase 1: Préparation Données (COMPLÉTÉ)

### 1.1 Nettoyage Anciennes Données
```bash
# ✅ FAIT - Supprimé family_data_OLD_int8_*
# ✅ FAIT - Régénéré family_data_FIXED avec uint8
```

### 1.2 Cleanup pannuke_features (~12 GB)
```bash
# Vérifier taille avant suppression
du -sh data/cache/pannuke_features

# Supprimer (features corrompues - Bug #1 et #2)
rm -rf data/cache/pannuke_features

# Libération: ~12 GB
```

**Raison:** Ces features ont été extraites AVANT fix preprocessing → CLS std ~0.28 au lieu de ~0.77

---

## ⏳ Phase 2: Extraction Features Folds (~30 min)

### 2.1 Fold 0
```bash
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 0 \
    --batch_size 8 \
    --chunk_size 300
```

**Sortie:** `data/cache/pannuke_features/fold0_features.npz` (~5.8 GB)

### 2.2 Fold 1
```bash
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 1 \
    --batch_size 8 \
    --chunk_size 300
```

**Sortie:** `data/cache/pannuke_features/fold1_features.npz` (~5.8 GB)

### 2.3 Fold 2
```bash
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 2 \
    --batch_size 8 \
    --chunk_size 300
```

**Sortie:** `data/cache/pannuke_features/fold2_features.npz` (~5.8 GB)

### 2.4 Validation Features
```bash
python scripts/validation/verify_features.py --features_dir data/cache/pannuke_features
```

**Attendu:** CLS std ~0.77 (entre 0.70-0.90)

**Si CLS std < 0.40:** Features corrompues → Vérifier preprocessing

---

## ⏳ Phase 3: Ré-entraînement OrganHead (~10 min)

### 3.1 Backup Ancien Checkpoint (Optionnel)
```bash
cp -r models/checkpoints models/checkpoints_OLD_20251222
```

### 3.2 Entraînement
```bash
python scripts/training/train_organ_head.py --folds 0 1 2 --epochs 50
```

**Attendu:** Val Accuracy > 99% (ancien: 99.94%)

---

## ⏳ Phase 4: Extraction Features par Famille (~20 min)

### 4.1 Glandular
```bash
python scripts/preprocessing/extract_features_from_fixed.py --family glandular
```

### 4.2 Digestive
```bash
python scripts/preprocessing/extract_features_from_fixed.py --family digestive
```

### 4.3 Urologic
```bash
python scripts/preprocessing/extract_features_from_fixed.py --family urologic
```

### 4.4 Epidermal
```bash
python scripts/preprocessing/extract_features_from_fixed.py --family epidermal
```

### 4.5 Respiratory
```bash
python scripts/preprocessing/extract_features_from_fixed.py --family respiratory
```

**Sortie pour chaque famille:**
- `data/cache/family_data_FIXED/{family}_features.npz`
- `data/cache/family_data_FIXED/{family}_targets.npz`

---

## ⏳ Phase 5: Entraînement HoVer-Net par Famille (~2h total)

### 5.1 Glandular (~25 min)
```bash
python scripts/training/train_hovernet_family.py \
    --family glandular \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Attendu:** NP Dice ~0.96, HV MSE ~0.01, NT Acc ~0.91

### 5.2 Digestive (~20 min)
```bash
python scripts/training/train_hovernet_family.py \
    --family digestive \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Attendu:** NP Dice ~0.96, HV MSE ~0.02, NT Acc ~0.88

### 5.3 Urologic (~15 min)
```bash
python scripts/training/train_hovernet_family.py \
    --family urologic \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Attendu:** NP Dice ~0.93, HV MSE ~0.28, NT Acc ~0.91

### 5.4 Epidermal (~10 min)
```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Attendu:** NP Dice ~0.95, HV MSE ~0.27, NT Acc ~0.89

### 5.5 Respiratory (~10 min)
```bash
python scripts/training/train_hovernet_family.py \
    --family respiratory \
    --epochs 50 \
    --augment \
    --dropout 0.1 \
    --cache_dir data/cache/family_data_FIXED
```

**Attendu:** NP Dice ~0.94, HV MSE ~0.05, NT Acc ~0.92

---

## ⏳ Phase 6: Validation Finale (~10 min)

### 6.1 Glandular
```bash
python scripts/evaluation/test_on_training_data.py \
    --family glandular \
    --checkpoint models/checkpoints/hovernet_glandular_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data_FIXED
```

### 6.2 Digestive
```bash
python scripts/evaluation/test_on_training_data.py \
    --family digestive \
    --checkpoint models/checkpoints/hovernet_digestive_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data_FIXED
```

### 6.3 Urologic
```bash
python scripts/evaluation/test_on_training_data.py \
    --family urologic \
    --checkpoint models/checkpoints/hovernet_urologic_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data_FIXED
```

### 6.4 Epidermal
```bash
python scripts/evaluation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10 \
    --data_dir data/cache/family_data_FIXED
```

### 6.5 Respiratory
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

## ⏳ Phase 7: Cleanup Final (APRÈS validation OK)

### 7.1 Vérifier Taille
```bash
du -sh data/cache/family_data_OLD_int8_*
```

### 7.2 Supprimer Anciennes Données
```bash
rm -rf data/cache/family_data_OLD_int8_*
```

**Libération attendue:** ~10-15 GB

---

## 📊 Résumé Temps Estimés

| Phase | Temps | GPU |
|-------|-------|-----|
| Phase 1: Cleanup | < 1 min | Non |
| Phase 2: Features folds | ~30 min | Oui |
| Phase 3: OrganHead | ~10 min | Oui |
| Phase 4: Features familles | ~20 min | Oui |
| Phase 5: HoVer-Net (5 familles) | ~2h | Oui |
| Phase 6: Validation | ~10 min | Oui |
| Phase 7: Cleanup | < 1 min | Non |
| **TOTAL** | **~3h10** | |

---

## ⚠️ Points de Vigilance

### RAM Peak
- Extraction features: ~6 GB par fold (avec `--chunk_size 300`)
- Entraînement HoVer-Net: ~11 GB par famille (données en RAM)
- **Total RAM requis:** 12 GB disponibles → ✅ OK

### Vérifications Critiques
1. **Après Phase 2:** CLS std ~0.77 (détecte Bug #2)
2. **Après Phase 4:** HV dtype=float32 min=-1 max=1 (détecte Bug #3)
3. **Après Phase 5:** NP Dice ~0.96 (vs 0.02 avec int8)

### Checkpoints Sauvegardés
- OrganHead: `models/checkpoints/organ_head_best.pth`
- Glandular: `models/checkpoints/hovernet_glandular_best.pth`
- Digestive: `models/checkpoints/hovernet_digestive_best.pth`
- Urologic: `models/checkpoints/hovernet_urologic_best.pth`
- Epidermal: `models/checkpoints/hovernet_epidermal_best.pth`
- Respiratory: `models/checkpoints/hovernet_respiratory_best.pth`

---

## 🔧 Dépannage

### Problème: CLS std ~0.28 au lieu de ~0.77
**Cause:** Bug #2 LayerNorm mismatch
**Solution:** Vérifier que `extract_features.py` utilise `forward_features()` et non `blocks[X]`

### Problème: HV MSE ~4681 au lieu de ~0.01
**Cause:** Bug #3 HV int8 au lieu de float32
**Solution:** Vérifier que `prepare_family_data_FIXED.py` utilise `dtype=np.float32` pour HV

### Problème: ModuleNotFoundError: No module named 'src'
**Cause:** PYTHONPATH non configuré
**Solution:** Vérifier lignes 28-31 de `extract_features.py`:
```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

---

## 📚 Références

- Plan complet: `docs/PLAN_DECISION_DONNEES.md`
- Commandes détaillées: `COMMANDES_ENTRAINEMENT.md`
- Impact uint8: `docs/IMPACT_UINT8_CONVERSION.md`
- Journal développement: `CLAUDE.md` section "2025-12-22"
