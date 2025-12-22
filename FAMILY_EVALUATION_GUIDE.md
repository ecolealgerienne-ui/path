# 📊 Guide: Évaluation par Famille (Sans OrganHead)

**Date:** 2025-12-22
**Commit:** 4c39044
**Statut:** ✅ Scripts créés - Prêt pour évaluation

---

## 🎯 Objectif

Évaluer chaque décodeur HoVer-Net sur ses tissus appropriés **sans utiliser OrganHead** pour le routage.

**Approche:**
1. Organiser les images de test par famille (glandular, digestive, etc.)
2. Tester chaque famille HoVer-Net sur ses propres images
3. Obtenir des métriques fiables par famille

**Avantages:**
- ✅ Pas besoin d'OrganHead (checkpoint manquant)
- ✅ Évaluation contrôlée de chaque décodeur
- ✅ Métriques comparatives entre familles
- ✅ Identifie quelles familles performent bien

---

## 📋 Workflow Complet

### Étape 1: Organiser les Images par Famille

**Script:** `organize_test_by_family.py`

```bash
# D'abord: Preview (dry run)
python scripts/evaluation/organize_test_by_family.py \
    --input_dir data/evaluation/pannuke_fold2_converted \
    --output_dir data/evaluation/by_family \
    --dry_run

# Sortie attendue:
# DISTRIBUTION BY FAMILY
# ========================================
# Glandular  :   XX images
#   └─ Breast      :   XX
#   └─ Prostate    :   XX
#   └─ Thyroid     :   XX
# Digestive  :   XX images
#   └─ Colon       :   XX
#   └─ Stomach     :   XX
# ...

# Si tout semble bon, exécuter réellement:
python scripts/evaluation/organize_test_by_family.py \
    --input_dir data/evaluation/pannuke_fold2_converted \
    --output_dir data/evaluation/by_family
```

**Résultat:** Créée la structure:
```
data/evaluation/by_family/
├── glandular/
│   ├── image_00001.npz (Breast)
│   ├── image_00023.npz (Prostate)
│   └── ...
├── digestive/
│   ├── image_00012.npz (Colon)
│   └── ...
├── urologic/
├── respiratory/
└── epidermal/
```

### Étape 2: Évaluer Toutes les Familles

**Script:** `evaluate_by_family.py`

```bash
# Test rapide (10 images par famille)
python scripts/evaluation/evaluate_by_family.py \
    --dataset_dir data/evaluation/by_family \
    --checkpoint_dir models/checkpoints_FIXED \
    --output_dir results/by_family_test \
    --num_samples 10

# Évaluation complète
python scripts/evaluation/evaluate_by_family.py \
    --dataset_dir data/evaluation/by_family \
    --checkpoint_dir models/checkpoints_FIXED \
    --output_dir results/by_family_full
```

**Sortie attendue:**
```
======================================================================
SUMMARY REPORT
======================================================================

Family       Images   Dice     AJI      PQ       Recall
----------------------------------------------------------------------
Glandular    XXX      0.9XXX   0.XXX    0.XXX    XX.XX%
Digestive    XXX      0.9XXX   0.XXX    0.XXX    XX.XX%
Urologic     XXX      0.9XXX   0.XXX    0.XXX    XX.XX%
Respiratory  XXX      0.9XXX   0.XXX    0.XXX    XX.XX%
Epidermal    XXX      0.9XXX   0.XXX    0.XXX    XX.XX%
----------------------------------------------------------------------
AVERAGE      XXX      0.9XXX   0.XXX    0.XXX    XX.XX%

======================================================================
TARGETS
======================================================================
Dice:   > 0.95
AJI:    > 0.80
PQ:     > 0.70
Recall: > 90%

✅ Summary saved: results/by_family_test/summary_by_family.json
```

### Étape 3: Analyser les Résultats

```bash
# Voir le résumé JSON
cat results/by_family_test/summary_by_family.json

# Consulter les rapports détaillés par famille
ls results/by_family_test/

# Exemple pour Glandular:
cat results/by_family_test/glandular/clinical_report_*.txt
```

---

## 🔍 Mapping Organe → Famille

| Famille | Organes PanNuke |
|---------|-----------------|
| **Glandular** | Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland |
| **Digestive** | Colon, Stomach, Esophagus, Bile-duct |
| **Urologic** | Kidney, Bladder, Testis, Ovarian, Uterus, Cervix |
| **Respiratory** | Lung, Liver |
| **Epidermal** | Skin, HeadNeck |

---

## 📊 Métriques à Surveiller

### Par Famille

**Glandular & Digestive (>2000 samples train):**
- ✅ Dice: > 0.96 (attendu excellent)
- ✅ AJI:  > 0.80 (bonne instance segmentation)
- ✅ PQ:   > 0.70 (bonne qualité panoptique)
- ⚠️ HV MSE: < 0.02 (excellent gradients)

**Urologic, Respiratory, Epidermal (<600 samples train):**
- ✅ Dice: > 0.93 (attendu bon)
- ⚠️ AJI:  > 0.50 (possibly lower due to HV MSE ~0.27)
- ⚠️ PQ:   > 0.40
- ⚠️ HV MSE: ~0.27 (dégradé, voir CLAUDE.md section "Résultats par Famille")

### Globales (Moyenne des 5 familles)

- Dice:   > 0.95
- AJI:    > 0.70 (pondéré par nombre d'échantillons)
- PQ:     > 0.60
- Recall: > 85%

---

## 🚨 Troubleshooting

### Problème: Aucune image dans une famille

```
⚠️ Skipping glandular: no images found
```

**Cause:** Les fichiers NPZ n'ont pas d'info `organ` ou le mapping est incorrect.

**Solution:**
1. Vérifier le contenu d'un fichier NPZ:
```python
import numpy as np
data = np.load('data/evaluation/pannuke_fold2_converted/image_00000.npz', allow_pickle=True)
print(data.keys())
print(data.get('organ', 'NO ORGAN KEY'))
```

2. Si `organ` manque, il faut le recréer lors de la conversion (voir `scripts/evaluation/convert_annotations.py`)

### Problème: Métriques très basses

```
Dice: 0.30  AJI: 0.05
```

**Causes possibles:**
1. Mauvaise famille utilisée (force_family incorrect)
2. Paramètres watershed non optimaux pour cette famille
3. Python cache pas cleared

**Solution:**
```bash
# Clear cache
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null
find . -type f -name '*.pyc' -delete

# Vérifier que force_family matche la famille des images
python scripts/evaluation/organize_test_by_family.py --input_dir ... --dry_run
```

---

## 📝 Notes Importantes

1. **Cette approche ne teste PAS OrganHead** - elle suppose que le routage est correct
2. **Pour tester OrganHead**, il faudra:
   - Entraîner OrganHead (voir `scripts/training/train_organ_head.py`)
   - Ou copier le checkpoint existant vers `models/checkpoints_FIXED/organ_head_best.pth`
3. **Les résultats par famille sont valides** même sans OrganHead - ils testent les décodeurs HoVer-Net isolément

---

## 🚀 Après l'Évaluation

Si les métriques sont bonnes par famille:
1. ✅ Valide que les décodeurs HoVer-Net fonctionnent bien
2. ⏭️ Prochaine étape: Entraîner/copier OrganHead pour test end-to-end complet
3. 📊 Comparer avec baseline (CellViT-256, autres méthodes)

Si les métriques sont mauvaises:
1. 🔍 Identifier quelle famille performe mal
2. 🔧 Optimiser watershed params spécifiquement pour cette famille
3. 🎯 Ou considérer ré-entraînement avec plus de données

---

**Créé:** 2025-12-22
**Par:** Claude (Family Evaluation)
**Commits:** 070c8db (organize), 4c39044 (evaluate)
**Status:** ✅ Prêt pour utilisation
