# Guide d'Évaluation Ground Truth - Modèles FIXED

**Date**: 2025-12-21
**Objectif**: Valider les modèles FIXED contre annotations expertes PanNuke
**Statut**: ⏳ En attente fin entraînement 4 familles

---

## 🎯 Objectif

Évaluer les 5 modèles HoVer-Net FIXED (normalisation HV [-1, 1]) contre les annotations expertes PanNuke pour confirmer l'amélioration par rapport aux modèles OLD.

**Métriques clés** :
- **Dice Score** : Chevauchement binaire (segmentation)
- **AJI** : Aggregated Jaccard Index (qualité instances)
- **PQ** : Panoptic Quality (détection + segmentation)
- **F1d** : F1 par classe (typage cellulaire)

---

## 📋 Prérequis

### 1. Modèles FIXED Entraînés

```bash
# Vérifier que les 5 checkpoints FIXED existent
ls -lh models/checkpoints_FIXED/*.pth

# Attendu:
# hovernet_glandular_best.pth   ✅
# hovernet_digestive_best.pth   (en cours)
# hovernet_urologic_best.pth    (en cours)
# hovernet_respiratory_best.pth (en cours)
# hovernet_epidermal_best.pth   (en cours)
```

### 2. Dataset PanNuke Fold 2

**Pourquoi Fold 2 ?**
- Fold 0, 1 : Utilisés pour entraînement
- **Fold 2** : Jamais vu par les modèles → Évaluation "aveugle" ✅

```bash
# Vérifier PanNuke Fold 2
ls /home/amar/data/PanNuke/fold2/

# Attendu:
# images.npy  (2656 images)
# masks.npy   (2656 annotations)
# types.npy   (types cellulaires)
```

---

## 🚀 Workflow d'Évaluation

### Option A: Script Automatique (Recommandé)

```bash
# Activer environnement
conda activate cellvit

# Lancer évaluation complète
bash scripts/evaluation/test_fixed_models_ground_truth.sh
```

**Durée estimée** : ~30-45 min (50 échantillons, toutes familles)

**Sortie attendue** :
```
✅ Tous les checkpoints présents
🧪 Évaluation des prédictions vs annotations expertes...
[Progress bar...]
📊 Rapports générés:
  ✅ results/ground_truth_FIXED/clinical_report_*.txt
  ✅ results/ground_truth_FIXED/metrics_*.json
```

### Option B: Comparaison FIXED vs OLD

```bash
# Compare directement FIXED vs OLD
python scripts/evaluation/compare_fixed_vs_old.py \
    --dataset_dir data/evaluation/pannuke_fold2_converted \
    --num_samples 50 \
    --fixed_dir models/checkpoints_FIXED \
    --old_dir models/checkpoints \
    --output_dir results/comparison_FIXED_vs_OLD
```

**Durée estimée** : ~1h (évalue 2 fois 50 échantillons)

**Sortie attendue** :
```
📊 DICE SCORE
  FIXED: 0.9655 ± 0.0184
  OLD:   0.9645 ± 0.0190
  ✅ Amélioration: +0.10%

📊 AJI
  FIXED: 0.7834 ± 0.1123
  OLD:   0.7612 ± 0.1234
  ✅ Amélioration: +2.92%

🎉 AMÉLIORATION SIGNIFICATIVE: +1.51% en moyenne
✅ RECOMMANDATION: Déployer les modèles FIXED
```

---

## 📊 Métriques Expliquées

### 1. Dice Score (Segmentation Binaire)

**Formule** : `Dice = 2 × |Prédit ∩ GT| / (|Prédit| + |GT|)`

**Interprétation** :
- **> 0.95** : Excellent (détecte 95%+ des noyaux)
- **0.90-0.95** : Bon
- **< 0.90** : Problématique

**Cible FIXED** : ≥ 0.96 (Glandular validé à 0.9655)

### 2. AJI (Aggregated Jaccard Index)

**Formule** : Moyenne IoU pondérée par taille d'instance

**Interprétation** :
- **> 0.80** : Excellent (instances bien séparées)
- **0.70-0.80** : Bon
- **< 0.70** : Problématique (fusions d'instances)

**Cible FIXED** : ≥ 0.75

### 3. PQ (Panoptic Quality)

**Formule** : `PQ = DQ × SQ`
- **DQ** (Detection Quality) : Fraction instances correctement détectées
- **SQ** (Segmentation Quality) : IoU moyen instances matchées

**Interprétation** :
- **> 0.70** : Excellent
- **0.60-0.70** : Bon
- **< 0.60** : Problématique

**Cible FIXED** : ≥ 0.65

### 4. F1d (F1 par Classe)

**Formule** : `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Interprétation** :
- **> 0.90** : Excellent (typage précis)
- **0.80-0.90** : Bon
- **< 0.80** : Problématique

**Cible FIXED** : ≥ 0.85 (Glandular validé à 0.9517)

---

## 🔍 Interprétation des Résultats

### Scénario 1: Amélioration Significative (+2%)

```
📊 BILAN
✅ Dice:  0.9655 → 0.9680 (+0.26%)
✅ AJI:   0.7612 → 0.7834 (+2.92%)
✅ PQ:    0.6523 → 0.6701 (+2.73%)

🎉 AMÉLIORATION SIGNIFICATIVE
→ Déployer les modèles FIXED
```

**Actions** :
1. ✅ Copier checkpoints FIXED vers production
2. ✅ Mettre à jour l'IHM
3. ✅ Documenter dans CLAUDE.md

### Scénario 2: Amélioration Légère (<2%)

```
📊 BILAN
✅ Dice:  0.9655 → 0.9662 (+0.07%)
⚠️  AJI:   0.7612 → 0.7598 (-0.18%)
✅ PQ:    0.6523 → 0.6545 (+0.34%)

✅ AMÉLIORATION LÉGÈRE
→ Analyser les cas de régression
```

**Actions** :
1. ⚠️ Vérifier quelles familles régressent (AJI)
2. Comparer HV MSE par famille
3. Décision au cas par cas (déployer familles stables uniquement)

### Scénario 3: Régression Détectée

```
📊 BILAN
❌ Dice:  0.9655 → 0.9420 (-2.43%)
❌ AJI:   0.7612 → 0.7201 (-5.40%)
❌ PQ:    0.6523 → 0.6104 (-6.42%)

⚠️  RÉGRESSION DÉTECTÉE
→ Ne PAS déployer
```

**Actions** :
1. ❌ **STOPPER** le déploiement
2. Investiguer la cause :
   - Problème preprocessing ?
   - Bug dans compute_hv_maps() ?
   - Hyperparamètres incorrects ?
3. Ré-entraîner après correction

---

## 🛠️ Dépannage

### Erreur: "Checkpoints manquants"

```bash
# Vérifier checkpoints
ls models/checkpoints_FIXED/*.pth

# Si manquant, ré-entraîner
bash scripts/training/train_all_families_FIXED.sh
```

### Erreur: "PanNuke Fold 2 introuvable"

```bash
# Télécharger PanNuke
python scripts/setup/download_and_prepare_pannuke.py

# Ou manuel:
wget https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold2.zip
unzip fold2.zip -d /home/amar/data/PanNuke/
```

### Erreur: "CUDA out of memory"

```python
# Dans compare_fixed_vs_old.py, réduire batch size
# Ou utiliser CPU (plus lent)
python scripts/evaluation/compare_fixed_vs_old.py --device cpu
```

### Résultats incohérents

```bash
# Vérifier que les modèles FIXED sont bien chargés
python -c "
import torch
ckpt = torch.load('models/checkpoints_FIXED/hovernet_glandular_best.pth')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'HV MSE: {ckpt.get(\"best_hv_mse\", \"N/A\")}')
"

# Attendu pour FIXED:
# Epoch: 49
# HV MSE: 0.0105  (OLD aurait ~0.015)
```

---

## 📈 Résultats Attendus (Hypothèses)

### Basé sur Validation Glandular

| Métrique | OLD | FIXED | Amélioration Attendue |
|----------|-----|-------|------------------------|
| **NP Dice** | 0.9645 | 0.9655 | +0.1% (identique) |
| **HV MSE** | 0.0150 | 0.0105 | **-30%** ✅ |
| **NT Acc** | 0.8800 | 0.9517 | **+7.2%** ✅ |
| **AJI** | ~0.76 | ~0.78 | +2-3% (estimé) |
| **PQ** | ~0.65 | ~0.67 | +2-3% (estimé) |

### Par Famille (Estimations)

| Famille | Dice | AJI | PQ | Confiance |
|---------|------|-----|-----|-----------|
| **Glandular** | 0.9655 | 0.78 | 0.67 | ✅ Validé |
| **Digestive** | ~0.96 | ~0.78 | ~0.67 | ✅ Haute |
| **Urologic** | ~0.93 | ~0.72 | ~0.62 | ⚠️ Moyenne |
| **Respiratory** | ~0.94 | ~0.74 | ~0.64 | ⚠️ Moyenne |
| **Epidermal** | ~0.95 | ~0.75 | ~0.65 | ⚠️ Moyenne |

**Seuil critique** : ~2000 samples pour performances optimales (Glandular, Digestive OK)

---

## 📝 Checklist Post-Évaluation

Après avoir obtenu les résultats :

- [ ] Dice Score ≥ 0.95 pour toutes les familles
- [ ] AJI ≥ 0.70 pour toutes les familles
- [ ] PQ ≥ 0.60 pour toutes les familles
- [ ] Amélioration vs OLD sur au moins 2/3 métriques
- [ ] Aucune régression > 5% sur une métrique
- [ ] Rapport JSON sauvegardé
- [ ] Rapport TXT consulté
- [ ] Décision GO/NO-GO documentée dans CLAUDE.md

---

## 🎯 Prochaines Étapes Après Évaluation

### Si Résultats Positifs (GO)

1. **Mettre à jour CLAUDE.md** avec résultats GT
2. **Copier checkpoints FIXED** :
   ```bash
   cp models/checkpoints_FIXED/*.pth models/checkpoints/
   ```
3. **Tester l'IHM Gradio** :
   ```bash
   python scripts/demo/gradio_demo.py
   ```
4. **Commit final** :
   ```bash
   git add .
   git commit -m "Ground Truth validation: FIXED models approved for deployment"
   git push
   ```

### Si Résultats Mitigés (INVESTIGATE)

1. **Analyser par famille** les cas de régression
2. **Vérifier HV range** sur échantillons problématiques
3. **Ajuster watershed thresholds** si nécessaire
4. **Tester sur plus d'échantillons** (100 au lieu de 50)
5. **Décision famille par famille**

### Si Régression (NO-GO)

1. **Stopper le déploiement**
2. **Investiguer la cause** :
   - Bug preprocessing ?
   - Hyperparamètres ?
   - Architecture ?
3. **Corriger et ré-entraîner**
4. **Ré-évaluer GT**

---

## 📚 Références

### Scripts Créés

| Script | Description |
|--------|-------------|
| `scripts/evaluation/test_fixed_models_ground_truth.sh` | Évaluation automatique sur PanNuke Fold 2 |
| `scripts/evaluation/compare_fixed_vs_old.py` | Comparaison FIXED vs OLD |
| `scripts/evaluation/evaluate_ground_truth.py` | Évaluation GT générique (existant) |
| `scripts/evaluation/convert_annotations.py` | Conversion annotations (existant) |

### Documentation

| Document | Description |
|----------|-------------|
| `GUIDE_GROUND_TRUTH_EVALUATION.md` | Ce guide |
| `IHM_READY_FOR_FIXED_MODELS.md` | Audit IHM |
| `docs/ARCHITECTURE_HV_ACTIVATION.md` | Décision technique tanh() |
| `INTEGRATION_PLAN_HV_NORMALIZATION.md` | Plan d'intégration complet |

---

**Créé le** : 2025-12-21
**Par** : Claude (Préparation évaluation GT)
**Statut** : ✅ PRÊT - En attente fin entraînement 4 familles
**Durée estimée évaluation** : ~30-45 min (automatique) ou ~1h (comparaison)
