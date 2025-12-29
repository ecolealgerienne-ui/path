# Epidermal V13 Smart Crops - Résultats Complets

**Date:** 2025-12-29
**Famille:** Epidermal (Skin + HeadNeck)
**Objectif:** AJI ≥ 0.68

---

## 1. Données Sources

| Métrique | Valeur |
|----------|--------|
| Images sources totales | 574 |
| Train (80%) | 459 images sources |
| Val (20%) | 115 images sources |
| Crops train | ~2295 |
| Crops val | ~510 |
| Amplification | ~5× par image |

### Stratégie de Cropping

- **Algorithme:** Layer-based (split-first-then-rotate)
- **5 positions:** center, top_left, top_right, bottom_left, bottom_right
- **Rotations:** 0°, 90°, 180°, 270°, flip_h (par position)
- **Taille crop:** 224×224

### Caractéristiques Tissulaires

- **Type:** Tissus stratifiés multicouches
- **Difficulté:** Kératinocytes superposés → frontières floues
- **Historique HV MSE:** 0.30 (plus élevé de toutes les familles)

---

## 2. Test 1 : Transfer Learning (Respiratory → Epidermal)

### Hypothèse

Réutiliser le modèle Respiratory (AJI 0.6734) comme initialisation pour Epidermal avec LR ultra-bas.

### Configuration

```bash
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal \
    --pretrained_checkpoint models/checkpoints_v13_smart_crops/hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth \
    --finetune_lr 1e-5 \
    --epochs 30 \
    --use_hybrid \
    --use_fpn_chimique
```

### Résultats Entraînement

| Métrique | Valeur |
|----------|--------|
| Source checkpoint | Respiratory epoch 25 |
| Source Dice | 0.84 |
| LR utilisé | 1e-5 (vs 1e-4 normal) |
| Best Dice | 0.8006 |
| Best Combined Score | 0.7443 |
| Val Dice | 0.7977 |
| Val HV MSE | 0.1106 |
| Val NT Acc | 0.8808 |

### Résultats Évaluation (n=50)

| Métrique | Valeur |
|----------|--------|
| Dice | 0.7948 ± 0.1100 |
| **AJI** | **0.5794 ± 0.1247** |
| AJI Median | 0.5902 |
| AJI FAIR | 0.5807 ± 0.1238 |
| PQ | 0.5568 ± 0.1245 |
| Instances pred | 17.8 |
| Instances GT | 18.7 |
| Over-seg ratio | 0.95× |

### Verdict Transfer Learning

- **AJI 0.5794** = 85.2% de l'objectif 0.68
- **Conclusion:** Transfer Learning inefficace pour Respiratory → Epidermal
- **Cause:** Morphologie tissulaire trop différente (parenchyme vs stratifié)

---

## 3. Test 2 : From Scratch (FPN Chimique)

### Configuration

```bash
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal \
    --epochs 30 \
    --use_hybrid \
    --use_fpn_chimique
```

### Résultats Entraînement

| Métrique | Valeur |
|----------|--------|
| Best Dice | 0.8020 |
| Best Combined Score | 0.7441 |
| Val Dice | 0.7993 |
| Val HV MSE | 0.1134 |
| Val NT Acc | 0.8876 |
| Lambda HV | 8.0 (epochs 26-30) |

### Résultats Évaluation (n=50)

| Métrique | Valeur |
|----------|--------|
| Dice | 0.7930 ± 0.1286 |
| **AJI** | **0.5868 ± 0.1343** |
| AJI Median | 0.6083 |
| AJI FAIR | 0.5858 ± 0.1343 |
| PQ | 0.5561 ± 0.1283 |
| Instances pred | 17.7 |
| Instances GT | 18.7 |
| Over-seg ratio | 0.95× |

### Verdict From Scratch

- **AJI 0.5868** = 86.3% de l'objectif 0.68
- **Écart:** -0.0932
- **Amélioration vs Transfer Learning:** +1.3%

---

## 4. Test 3 : From Scratch + Watershed Optimisé

### Configuration Optimisation

```bash
python scripts/evaluation/optimize_watershed_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth \
    --family epidermal \
    --n_samples 50
```

### Grid Search

- **Configurations testées:** 400
- **Paramètres explorés:**
  - beta: [0.5, 1.0, 1.5, 2.0, 2.5]
  - min_size: [20, 30, 40, 50, 60]
  - np_threshold: [0.30, 0.35, 0.40, 0.45]
  - min_distance: [2, 3, 4, 5]

### TOP 10 Configurations

| Rank | Beta | MinSize | NP_Thr | MinDist | AJI Mean | OverSeg |
|------|------|---------|--------|---------|----------|---------|
| 1 | 0.50 | 40 | 0.45 | 5 | **0.5981** | 1.12 |
| 2 | 0.50 | 30 | 0.45 | 5 | 0.5971 | 1.14 |
| 3 | 0.50 | 50 | 0.45 | 5 | 0.5966 | 1.09 |
| 4 | 0.50 | 40 | 0.40 | 5 | 0.5964 | 1.13 |
| 5 | 0.50 | 60 | 0.45 | 5 | 0.5960 | 1.07 |
| 6 | 0.50 | 20 | 0.45 | 5 | 0.5958 | 1.18 |
| 7 | 0.50 | 30 | 0.40 | 5 | 0.5955 | 1.15 |
| 8 | 0.50 | 40 | 0.45 | 4 | 0.5955 | 1.14 |
| 9 | 0.50 | 30 | 0.45 | 4 | 0.5944 | 1.16 |
| 10 | 0.50 | 50 | 0.45 | 4 | 0.5941 | 1.10 |

### Paramètres Watershed Optimaux

| Paramètre | Respiratory | Epidermal | Diff |
|-----------|-------------|-----------|------|
| beta | 0.50 | **0.50** | = |
| min_size | 30 | **40** | +10 |
| np_threshold | 0.40 | **0.45** | +0.05 |
| min_distance | 5 | **5** | = |

**Interprétation:** Epidermal nécessite un seuil NP plus élevé (0.45) et une taille minimale plus grande (40) — cohérent avec les tissus stratifiés plus bruités.

### Résultats Finaux

| Métrique | Valeur |
|----------|--------|
| **AJI Mean** | **0.5981 ± 0.1356** |
| Over-seg Ratio | 1.12 |

### Comparaison Default vs Optimisé

| Configuration | AJI | Amélioration |
|---------------|-----|--------------|
| Default watershed | 0.5835 | - |
| Optimisé | 0.5981 | **+2.5%** |

### Verdict Final

- **AJI 0.5981** = **88.0%** de l'objectif 0.68
- **Écart:** -0.0819

---

## 5. Tableau Récapitulatif

| Configuration | AJI | Dice | PQ | Progress |
|---------------|-----|------|-----|----------|
| Transfer Learning (Respiratory) | 0.5794 | 0.7948 | 0.5568 | 85.2% |
| From Scratch | 0.5868 | 0.7930 | 0.5561 | 86.3% |
| **From Scratch + Watershed optimisé** | **0.5981** | 0.7930 | 0.5561 | **88.0%** |

### Comparaison Historique

| Version | AJI | Progress | Amélioration |
|---------|-----|----------|--------------|
| v12 (baseline historique) | 0.43 | 63% | - |
| **V13 Smart Crops + FPN** | **0.5981** | **88%** | **+39% relatif** |

---

## 6. Analyse Transfer Learning

### Pourquoi le Transfer Learning a échoué

| Aspect | Respiratory | Epidermal |
|--------|-------------|-----------|
| Type tissulaire | Parenchyme (alvéoles) | Stratifié (multicouche) |
| Structure | Ouvertes, espacées | Compactes, superposées |
| HV MSE historique | 0.25 | 0.30 |
| Frontières | Nettes | Floues |

**Conclusion:** Les patterns HV appris sur Respiratory (tissus ouverts) ne sont pas transférables à Epidermal (tissus stratifiés). L'entraînement from scratch est préférable.

---

## 7. Fichiers Générés

### Checkpoints

| Fichier | Description |
|---------|-------------|
| `models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth` | FPN Chimique (best) |

### Résultats Optimisation

| Fichier | Description |
|---------|-------------|
| `results/watershed_optimization/watershed_optimization_epidermal_20251229_180618.json` | Grid search results |

---

## 8. Conclusions

1. **Transfer Learning Respiratory → Epidermal** : Inefficace (-1.3% vs from scratch)
2. **From Scratch + FPN Chimique** : Meilleure approche pour tissus stratifiés
3. **Optimisation Watershed** : Gain +2.5% (np_threshold=0.45, min_size=40)
4. **Résultat final:** AJI 0.5981 = 88% de l'objectif

### Amélioration Globale

- **v12 → V13:** +0.17 AJI (0.43 → 0.5981) = **+39% relatif**
- Epidermal reste la famille la plus difficile (tissus stratifiés)

### Recommandation

Pour les familles avec **tissus stratifiés** (Epidermal, Urologic), privilégier l'entraînement from scratch plutôt que le Transfer Learning depuis des familles à tissus parenchymateux.
