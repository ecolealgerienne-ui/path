# Session 2025-12-30 — Résultats Glandular (5/5 Familles Complétées)

## Résumé

- **Glandular entraîné et évalué** — Dernière famille du projet
- **AJI: 0.6566** (96.6% de l'objectif 0.68)
- **5/5 familles complétées** — Projet V13 Smart Crops terminé

---

## ⚠️ IMPORTANT: Tests SANS Normalisation Macenko

> **Tous les résultats ci-dessous ont été obtenus SANS normalisation Macenko.**
>
> La normalisation Macenko a été implémentée (`normalize_staining_source.py`) mais
> n'était pas intégrée au pipeline `prepare_v13_smart_crops.py` jusqu'à cette session.
>
> Un test comparatif avec normalisation est prévu sur Respiratory.

---

## Résultats Glandular

### Configuration

| Paramètre | Valeur |
|-----------|--------|
| Architecture | V13 Smart Crops + FPN Chimique + H-Alpha |
| Epochs | 60 |
| Dataset | 3391 samples (Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland) |
| Normalisation | **NON** |

### Métriques Entraînement

```
Train - Loss: 4.5830, Dice: 0.8645
Val   - Loss: 5.3986, Dice: 0.8372, HV MSE: 0.1083, NT Acc: 0.8791

Best Dice: 0.8382
Best Combined Score: 0.7863
```

### Optimisation Watershed

```
TOP 3 CONFIGURATIONS:
Rank   Beta     MinSize    NP_Thr     MinDist    AJI Mean
1      0.50     50         0.40       3          0.6566
2      0.50     50         0.40       2          0.6558
3      0.50     40         0.40       3          0.6551
```

### Paramètres Optimaux

| Paramètre | Valeur |
|-----------|--------|
| beta | 0.50 |
| min_size | 50 |
| np_threshold | 0.40 |
| min_distance | 3 |

### Métriques Finales

```
AJI:             0.6566 ± 0.1386
Over-seg ratio:  0.96
Progress:        96.6% de l'objectif 0.68
```

---

## Récapitulatif Complet — 5/5 Familles (SANS Normalisation)

| Famille | Samples | AJI | Progress | Paramètres Watershed |
|---------|---------|-----|----------|----------------------|
| **Respiratory** | 408 | **0.6872** | **101.1%** ✅ | beta=0.50, min_size=30, np_thr=0.40, min_dist=5 |
| **Urologic** | 1101 | **0.6743** | **99.2%** | beta=0.50, min_size=30, np_thr=0.45, min_dist=2 |
| **Glandular** | 3391 | **0.6566** | **96.6%** | beta=0.50, min_size=50, np_thr=0.40, min_dist=3 |
| Epidermal | 574 | 0.6203 | 91.2% | beta=1.00, min_size=20, np_thr=0.45, min_dist=3 |
| Digestive | 2430 | 0.6160 | 90.6% | beta=2.00, min_size=60, np_thr=0.45, min_dist=5 |

### Statistiques

- **Objectif atteint:** 1/5 familles (Respiratory)
- **Proche objectif (>96%):** 3/5 familles (Respiratory, Urologic, Glandular)
- **AJI moyen:** 0.6509
- **Meilleur AJI:** 0.6872 (Respiratory)

---

## Observation: Volume ≠ Performance

| Famille | Samples | AJI | Observation |
|---------|---------|-----|-------------|
| Respiratory | 408 | 0.6872 | **Meilleur** malgré le plus petit dataset |
| Glandular | 3391 | 0.6566 | Plus grand dataset, pas le meilleur AJI |
| Digestive | 2430 | 0.6160 | 2ème plus grand, avant-dernier AJI |

**Hypothèses:**
1. **Homogénéité tissulaire** — Respiratory (Lung, Liver) = 2 organes similaires
2. **Diversité intra-famille** — Glandular = 5 organes très différents
3. **Déséquilibre des données** — Breast = 72% de Glandular
4. **Structures tissulaires** — Tissus respiratoires plus réguliers

---

## Prochaine Étape: Test Normalisation Macenko

### Objectif

Comparer les résultats AVEC vs SANS normalisation Macenko sur Respiratory.

### Commandes

```bash
# 1. Générer smart crops AVEC normalisation
python scripts/preprocessing/prepare_v13_smart_crops.py \
    --family respiratory \
    --use_normalized \
    --max_samples 5000

# 2. Extraire features
python scripts/preprocessing/extract_features_v13_smart_crops.py \
    --family respiratory --split train
python scripts/preprocessing/extract_features_v13_smart_crops.py \
    --family respiratory --split val

# 3. Entraîner (60 epochs)
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family respiratory \
    --epochs 60 \
    --use_hybrid \
    --use_fpn_chimique \
    --use_h_alpha

# 4. Évaluer avec paramètres optimaux connus
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth \
    --family respiratory \
    --n_samples 50 \
    --use_hybrid \
    --use_fpn_chimique \
    --np_threshold 0.40 \
    --min_size 30 \
    --beta 0.50 \
    --min_distance 5
```

### Résultat Attendu

| Condition | AJI Attendu |
|-----------|-------------|
| SANS normalisation | 0.6872 (baseline) |
| AVEC normalisation | ??? (à mesurer) |

---

## Modifications Apportées Cette Session

### Nouvelles Fonctionnalités

1. **`--use_normalized`** — Utiliser images Macenko-normalisées
2. **`--organ`** — Filtrer par organe spécifique
3. **`organ_names`** — Stocké dans les fichiers .npz générés

### Commits

```
f674bcf feat(preprocessing): Add --organ option to filter by specific organ
6ff4ef8 feat(preprocessing): Add organ_names to smart crops output
0e8442c feat(preprocessing): Add --use_normalized option to smart crops
c348d04 docs: Add PanNuke distribution by organ and family
30b10aa docs: Add Glandular training guide for final family
```

---

## Métadonnées

- **Date:** 2025-12-30
- **Durée entraînement:** ~45 min (60 epochs, 3391 samples)
- **GPU:** RTX 4070 SUPER
- **Checkpoint:** `hovernet_glandular_v13_smart_crops_hybrid_fpn_best.pth`
