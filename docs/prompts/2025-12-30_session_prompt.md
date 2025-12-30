# Prompt Session 2025-12-30 — CellViT-Optimus

## Contexte Projet

Tu travailles sur **CellViT-Optimus**, un système de segmentation de noyaux cellulaires pour l'histopathologie.

**Architecture:** V13 Smart Crops + FPN Chimique + H-Alpha (α learnable)
**Objectif:** AJI ≥ 0.68 sur les 5 familles PanNuke

## État Actuel (4/5 familles testées)

| Famille | Samples | AJI | Status |
|---------|---------|-----|--------|
| **Respiratory** | 408 | **0.6872** | ✅ Objectif atteint |
| **Urologic** | 1101 | **0.6743** | 99.2% |
| Epidermal | 574 | 0.6203 | 91.2% |
| Digestive | 2430 | 0.6160 | 90.6% |
| Glandular | 3391 | **À TESTER** | — |

## Travail Restant

1. **Glandular** (3391 samples) — Le plus grand dataset, à entraîner et évaluer
2. Considérer ajout de l'AJI original pour comparaison littérature (actuellement on calcule AJI+)

## Commandes pour Glandular

```bash
# 1. Préparation données (si pas déjà fait)
python scripts/preprocessing/normalize_staining_source.py --family glandular
python scripts/preprocessing/prepare_v13_smart_crops.py --family glandular --max_samples 5000

# 2. Extraction features
python scripts/preprocessing/extract_features_v13_smart_crops.py --family glandular --split train
python scripts/preprocessing/extract_features_v13_smart_crops.py --family glandular --split val

# 3. Entraînement
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family glandular \
    --epochs 60 \
    --use_hybrid \
    --use_fpn_chimique \
    --use_h_alpha

# 4. Optimisation watershed
python scripts/evaluation/optimize_watershed_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_glandular_v13_smart_crops_hybrid_fpn_best.pth \
    --family glandular \
    --n_samples 50

# 5. Évaluation finale (avec params optimisés du grid search)
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

## Documentation à Consulter

| Document | Description |
|----------|-------------|
| `CLAUDE.md` | Instructions projet, paramètres optimaux par famille |
| `docs/sessions/2025-12-29_shared_modules_refactoring.md` | Session complète avec résultats et refactoring |
| `src/postprocessing/watershed.py` | Algorithme watershed (single source of truth) |
| `src/evaluation/instance_evaluation.py` | Module évaluation partagé |

## Points Importants

1. **Modules partagés OBLIGATOIRES** — Ne jamais dupliquer le code critique
   - `from src.postprocessing import hv_guided_watershed`
   - `from src.evaluation import run_inference, evaluate_batch_with_params`

2. **FPN Chimique** nécessite toujours `--use_hybrid --use_fpn_chimique`

3. **AJI+ vs AJI** — Notre implémentation calcule AJI+ (one-to-one matching), pas AJI original (many-to-one)

4. **Paramètres watershed** varient par famille — Toujours utiliser les valeurs optimisées du grid search

## Accomplissements Session 2025-12-29

- ✅ Découvert et corrigé bug watershed (-2.8% AJI)
- ✅ Créé modules partagés (postprocessing, evaluation, metrics)
- ✅ Respiratory: AJI 0.6872 — OBJECTIF ATTEINT
- ✅ Urologic: AJI 0.6743 (99.2%)
- ✅ Epidermal: AJI 0.6203 (91.2%)
- ✅ Digestive: AJI 0.6160 (90.6%)
- ✅ Documenté tous les résultats et paramètres optimaux
