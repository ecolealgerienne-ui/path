# Diagnostic Lambda_hv=10.0 - Analyse Post-Mortem

**Date:** 2025-12-23  
**Contexte:** Test lambda_hv=10.0 pour résoudre AJI catastrophique  
**Résultat:** Test de stress réussi révélant cause racine (features corrompues)

---

## 📊 RÉSULTATS

### Comparaison Avant/Après

| Métrique | Sobel (λ=0.5) | Lambda_hv=10.0 | Δ |
|----------|---------------|----------------|---|
| Dice | 0.9489 | 0.6916 | **-27%** 🔴 |
| AJI | 0.0524 | 0.0357 | **-32%** 🔴 |
| PQ | 0.0856 | 0.0638 | **-25%** 🔴 |
| Rappel | 5.53% | 4.00% | **-28%** 🔴 |
| Classification Acc | ? | **0.00%** | **CASSÉ** 🔴 |

**Alerte critique:**
```
⚠️ Features SUSPECTES (CLS std=0.661, attendu 0.70-0.90)
```

---

## 🎯 CAUSE RACINE IDENTIFIÉE

**Mismatch features training vs inference:**
- Training: CLS std ~0.82 (preprocessing corrompu, avant fix Bug #1/Bug #2)
- Inference: CLS std ~0.66 (preprocessing correct, après Phase 1 Refactoring)
- **Écart 20%** → Décodeur "voit flou" → AJI catastrophique

---

## 🚀 PLAN D'ACTION (4 Étapes)

### Étape 1: Régénérer Features (PRIORITÉ)
```bash
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 0 \
    --batch_size 8 \
    --chunk_size 300
```

### Étape 2: Fix Post-Processing
Remplacer `Sobel(HV)` par `HV magnitude` dans `optimus_gate_inference_multifamily.py:161`

### Étape 3: Lambda_hv=2.0
Modifier `hovernet_decoder.py:349` → `10.0` → `2.0`

### Étape 4: Ré-entraîner
```bash
python scripts/training/train_hovernet_family.py --family epidermal --epochs 50 --augment
```

**ETA Total:** 2-3 heures

---

Voir `docs/RESULTATS_VERIFICATION_ETAPE3.md` pour analyse complète.
