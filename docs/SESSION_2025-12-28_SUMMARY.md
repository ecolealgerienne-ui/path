# Session 2025-12-28 : FPN Chimique - Résumé et État Final

## 🎯 Objectif de la Session

Atteindre **AJI ≥ 0.68** pour la segmentation nucléaire sur la famille Epidermal.

## 📊 Résultats Obtenus

| Étape | AJI | Dice | Over-seg | Commit |
|-------|-----|------|----------|--------|
| Baseline (avant session) | 0.5444 | 0.80 | 1.00× | - |
| **FPN Chimique (30 epochs)** | **0.6020** | 0.82 | 1.01× | Checkpoint existant |
| + Watershed optimisé | 0.6074 | 0.82 | 1.29× | - |
| + Fine-tuning (10 epochs) | 0.6047 | 0.82 | 1.02× | - |

**Gain total : +10.6%** (0.5444 → 0.6047)

**Objectif atteint à 88.9%** (0.6047 / 0.68)

## 🔧 Bugs Corrigés Cette Session

### 1. Gradient Monitoring FPN (Commit `68f2176`)
- **Problème** : Le monitoring affichait gradient H-channel = 0 en mode FPN
- **Cause** : Code vérifiait `model.ruifrok` (None en mode FPN) au lieu de `model.h_pyramid`
- **Fix** : Vérifier `model.h_pyramid.projections['16'][0].weight.grad`

### 2. Checkpoint FPN Non Détecté (Commit `ef1fba4`)
- **Problème** : `optimize_watershed_aji.py` ne chargeait pas le mode FPN
- **Cause** : Paramètre `use_fpn_chimique` non lu depuis le checkpoint
- **Fix** : Ajout `use_fpn_chimique = checkpoint.get('use_fpn_chimique', False)`

### 3. AJI Direction Inversée (Commit `d64f35a`)
- **Problème** : AJI local utilisait Pred→GT au lieu de GT→Pred (standard)
- **Cause** : Fonction `compute_aji` locale différente de la version centralisée
- **Fix** : Import de `src.metrics.ground_truth_metrics.compute_aji`

### 4. Option --resume Manquante (Commit `75381da`)
- **Problème** : Impossible de faire du fine-tuning depuis un checkpoint
- **Fix** : Ajout de `--resume` au script de training

## 📁 Fichiers Modifiés

```
scripts/training/train_hovernet_family_v13_smart_crops.py
  - Fix gradient monitoring FPN
  - Ajout option --resume

scripts/evaluation/optimize_watershed_aji.py
  - Support use_fpn_chimique
  - Utilisation AJI centralisé (GT-centric)

scripts/evaluation/test_v13_smart_crops_aji.py
  - Ajout "fair AJI" pour diagnostic
```

## 🏆 Checkpoints Disponibles

```
models/checkpoints_v13_smart_crops/
├── hovernet_epidermal_v13_smart_crops_best.pth           # Baseline (AJI ~0.54)
├── hovernet_epidermal_v13_smart_crops_hybrid_best.pth    # Hybrid simple
└── hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth # FPN Chimique (AJI 0.60) ✅
```

## 📈 Analyse de Performance

### Pourquoi le plateau à 0.60 ?

1. **FPN Chimique = Gain principal (+10.6%)**
   - L'injection multi-échelle du H-channel a brisé la "cécité profonde"
   - Over-segmentation ratio optimal (1.01×)

2. **Watershed = Gain marginal (+0.9%)**
   - Beta=0.5 reste optimal même avec FPN
   - Les gradients HV ne sont pas assez nets pour beta plus élevé

3. **Fine-tuning = Plateau**
   - LR=1e-5 + lambda_hv=10.0 n'ont pas amélioré
   - Le modèle a convergé à sa capacité maximale

### Limites Identifiées

- **Famille Epidermal** : Seulement 571 samples (vs 3535 pour Glandular)
- **Tissus stratifiés** : Architecture 3D complexe → frontières ambiguës
- **HV MSE ~0.10** : Gradients pas assez nets pour séparation parfaite

## 🚀 Prochaines Étapes Possibles

### Option A : Tester sur Glandular (Recommandé)
```bash
# Plus de données (3535 samples) → potentiellement AJI > 0.68
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family glandular \
    --epochs 30 \
    --use_hybrid \
    --use_fpn_chimique \
    --augment
```

### Option B : Data Augmentation Agressive
- Rotations, flips, color jitter
- Mixup / CutMix pour régularisation

### Option C : Architecture Plus Complexe
- Attention mechanisms (CBAM, Self-Attention)
- Deeper decoder
- Boundary-aware loss functions

### Option D : Accepter AJI 0.60
- Bon résultat pour tissus complexes
- 88.9% de l'objectif atteint
- Déployer et itérer

## 📋 Commandes de Référence

### Évaluation FPN Chimique
```bash
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth \
    --family epidermal \
    --n_samples 50 \
    --use_hybrid
```

### Grid Search Watershed
```bash
python scripts/evaluation/optimize_watershed_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth \
    --family epidermal \
    --n_samples 50
```

### Training FPN Chimique (Nouvelle Famille)
```bash
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family glandular \
    --epochs 30 \
    --use_hybrid \
    --use_fpn_chimique \
    --augment
```

## 🔗 Commits de la Session

| Hash | Description |
|------|-------------|
| `68f2176` | fix(fpn-chimique): Correct gradient monitoring for FPN mode |
| `098e9fe` | feat(eval): Add fair AJI comparison using same watershed for GT and Pred |
| `ef1fba4` | fix(optimize_watershed): Add use_fpn_chimique support |
| `d64f35a` | fix(optimize_watershed): Use centralized GT-centric AJI |
| `75381da` | feat(training): Add --resume option for fine-tuning |

## 📊 Métriques Finales (Epidermal)

```
AJI:         0.6047 ± 0.1110
AJI Median:  0.6186
Dice:        0.8184 ± 0.0706
PQ:          0.5794 ± 0.1166
Over-seg:    1.02×
Instances:   19.1 pred vs 18.7 GT
```

---

*Document généré le 2025-12-28*
