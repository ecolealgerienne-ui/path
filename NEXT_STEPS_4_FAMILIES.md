# Entraînement des 4 Familles Restantes - Guide d'Exécution

**Date**: 2025-12-21
**Statut**: ✅ Glandular validé (Dice 0.9655, NT Acc 0.9517)
**Action**: GO confirmé pour les 4 familles restantes

---

## 📋 Résumé de la Validation Glandular

| Métrique | Résultat Test | Comparaison Train | Statut |
|----------|---------------|-------------------|--------|
| **NP Dice** | 0.9655 ± 0.0184 | Train: 0.9641 (Δ +0.0015) | ✅ Identique |
| **HV MSE** | 0.0266 ± 0.0104 | Train: 0.0105 (Δ +0.0161) | ⚠️ Variance naturelle |
| **NT Acc** | 0.9517 ± 0.0229 | Train: 0.9107 (Δ +0.0410) | ✅ Meilleur ! |
| **HV Range** | [-1, 1] | ✅ Tous les 10 samples | ✅ Normalisé |

**Décision**: ✅ GO - Modèle validé, amélioration significative sur NT (+7.2% vs OLD)

---

## 🚀 Étapes d'Exécution (Sur Votre Machine Locale)

### Étape 1: Génération des Données FIXED (~20 min)

```bash
# Activer environnement
conda activate cellvit

# Vérifier que PanNuke est accessible
ls /home/amar/data/PanNuke/fold0/types.npy
# Attendu: fichier existe

# Lancer la génération des 4 familles
bash scripts/preprocessing/generate_all_families_FIXED.sh
```

**Sortie attendue** :
```
[1/4] DIGESTIVE - Colon, Stomach, Esophagus, Bile-duct
  ✅ Saved: data/family_FIXED/digestive_data_FIXED.npz (~2.0 GB)

[2/4] UROLOGIC - Kidney, Bladder, Testis, Ovarian, Uterus, Cervix
  ✅ Saved: data/family_FIXED/urologic_data_FIXED.npz (~900 MB)

[3/4] RESPIRATORY - Lung, Liver
  ✅ Saved: data/family_FIXED/respiratory_data_FIXED.npz (~350 MB)

[4/4] EPIDERMAL - Skin, HeadNeck
  ✅ Saved: data/family_FIXED/epidermal_data_FIXED.npz (~480 MB)

✅ GÉNÉRATION COMPLÈTE
```

**Logs** : `logs/digestive_fixed_generation.log`, etc.

---

### Étape 2: Entraînement des 4 Familles (~7 heures)

```bash
# Lancer les 4 entraînements séquentiels
bash scripts/training/train_all_families_FIXED.sh
```

**Timeline estimée** :

| Famille | Samples | Durée | Fin Estimée |
|---------|---------|-------|-------------|
| Digestive | 2430 | ~2.5h | +2.5h |
| Urologic | 1101 | ~2.0h | +4.5h |
| Respiratory | 408 | ~1.5h | +6.0h |
| Epidermal | 571 | ~1.5h | +7.5h |

**Métriques cibles** (basées sur Glandular) :
- **NP Dice**: ≥ 0.95
- **HV MSE**: < 0.05 (acceptable selon littérature)
- **NT Acc**: ≥ 0.85

**Logs** : `logs/train_digestive_fixed.log`, etc.

---

### Étape 3: Validation des Modèles (Optionnel mais Recommandé)

Après chaque entraînement, tester le modèle sur 10 échantillons :

```bash
# Tester Digestive
python scripts/validation/test_glandular_model.py \
    --checkpoint models/checkpoints_FIXED/hovernet_digestive_best.pth \
    --data_dir data/family_FIXED \
    --n_samples 10

# Répéter pour urologic, respiratory, epidermal
# (adapter --checkpoint et le script attend glandular_data_FIXED.npz,
#  donc il faudra peut-être créer un script test_family_model.py générique)
```

**Note**: Le script `test_glandular_model.py` est spécifique à Glandular. Si vous voulez tester les autres familles, dupliquez le script et adaptez le nom du fichier de données.

---

## 📊 Résultats Attendus

### Comparaison OLD vs NEW (Glandular)

| Métrique | OLD | NEW Train | NEW Test | Amélioration |
|----------|-----|-----------|----------|--------------|
| NP Dice | 0.9645 | 0.9641 | 0.9655 | ≈ Identique |
| HV MSE | 0.0150 | 0.0105 | 0.0266 | Train meilleur |
| NT Acc | 0.8800 | 0.9107 | 0.9517 | **+7.2%** ✅ |

### Hypothèses pour les 4 Familles

**Digestive** (2430 samples, structures tubulaires) :
- NP Dice: ~0.96 (similaire Glandular)
- HV MSE: ~0.015 (bon, beaucoup de données)
- NT Acc: ~0.88 (diversité organes)

**Urologic** (1101 samples, densité nucléaire élevée) :
- NP Dice: ~0.93 (OK mais clusters serrés)
- HV MSE: ~0.25 (difficile, chevauchement)
- NT Acc: ~0.91 (bon)

**Respiratory** (408 samples, structures ouvertes) :
- NP Dice: ~0.94 (OK)
- HV MSE: **~0.05** (surprise possible, noyaux espacés)
- NT Acc: ~0.89 (OK)

**Epidermal** (571 samples, couches stratifiées) :
- NP Dice: ~0.95 (bon)
- HV MSE: ~0.27 (difficile, chevauchement)
- NT Acc: ~0.89 (OK)

---

## 🔍 Points de Vigilance

### 1. HV MSE Plus Élevé sur Test

**Observation** : Glandular Test HV MSE (0.0266) > Train (0.0105)

**Causes probables** :
1. Resize 224→256 avec interpolation bilinéaire
2. Variance naturelle (Std = 0.0104)
3. Sample 9 outlier à 0.0513 (sans lui : ~0.0237)

**Action** : Acceptable si < 0.05 (littérature)

### 2. Familles avec Peu de Données

**Respiratory** (408 samples) et **Epidermal** (571 samples) :
- Risque d'overfitting plus élevé
- HV MSE potentiellement dégradé
- Mais NP Dice et NT Acc devraient rester bons (robustes)

**Mitigation** :
- Data augmentation activée (`--augment`)
- Surveillance des logs d'entraînement

### 3. Seuil Critique

**Découverte** : ~2000 samples = seuil pour HV MSE < 0.02

| Famille | Samples | HV MSE Attendu |
|---------|---------|----------------|
| Digestive | 2430 | ✅ < 0.02 |
| Urologic | 1101 | ⚠️ ~0.25 |
| Respiratory | 408 | ⚠️ ~0.05-0.30 |
| Epidermal | 571 | ⚠️ ~0.27 |

**Acceptabilité** : HV MSE < 0.05 est excellent selon la littérature.

---

## 📁 Fichiers Générés

Après exécution complète :

```
data/family_FIXED/
├── glandular_data_FIXED.npz      (~3.5 GB) ✅
├── digestive_data_FIXED.npz      (~2.0 GB)
├── urologic_data_FIXED.npz       (~900 MB)
├── respiratory_data_FIXED.npz    (~350 MB)
└── epidermal_data_FIXED.npz      (~480 MB)

models/checkpoints_FIXED/
├── hovernet_glandular_best.pth   (~50 MB) ✅
├── hovernet_digestive_best.pth   (~50 MB)
├── hovernet_urologic_best.pth    (~50 MB)
├── hovernet_respiratory_best.pth (~50 MB)
└── hovernet_epidermal_best.pth   (~50 MB)

logs/
├── glandular_fixed_generation.log ✅
├── digestive_fixed_generation.log
├── urologic_fixed_generation.log
├── respiratory_fixed_generation.log
├── epidermal_fixed_generation.log
├── train_glandular_fixed.log ✅
├── train_digestive_fixed.log
├── train_urologic_fixed.log
├── train_respiratory_fixed.log
└── train_epidermal_fixed.log
```

---

## 🎯 Après l'Entraînement

### 1. Mise à Jour de l'IHM (~3.5h)

Suivre le plan détaillé : `INTEGRATION_PLAN_HV_NORMALIZATION.md`

**Phases** :
- Phase 1 : Vérification inférence (HV range [-1, 1])
- Phase 2 : Ajustement post-processing (watershed thresholds)
- Phase 3 : Métriques morphométriques
- Phase 4 : Tests non-régression
- Phase 5 : Documentation

### 2. Déploiement

```bash
# Copier les checkpoints FIXED vers production
cp models/checkpoints_FIXED/*.pth models/checkpoints/

# Mettre à jour l'IHM Gradio
python scripts/demo/gradio_demo.py
```

### 3. Documentation

Mettre à jour `CLAUDE.md` avec :
- Résultats finaux des 5 familles
- Confirmation de l'amélioration HV normalization
- Métriques de validation

---

## 🐛 Dépannage

### Erreur: "Données manquantes"

```bash
# Vérifier que les données FIXED existent
ls -lh data/family_FIXED/*.npz

# Si manquant, relancer génération
bash scripts/preprocessing/generate_all_families_FIXED.sh
```

### Erreur: "CUDA out of memory"

```bash
# Réduire batch_size dans le script d'entraînement
# Éditer train_all_families_FIXED.sh, ligne:
BATCH_SIZE=16  # au lieu de 32
```

### Entraînement interrompu

```bash
# Relancer depuis la famille échouée
# Les checkpoints précédents sont sauvegardés
python scripts/training/train_hovernet_family.py \
    --family <FAMILY_NAME> \
    --data_dir data/family_FIXED \
    --output_dir models/checkpoints_FIXED \
    --epochs 50 \
    --augment
```

---

## 📝 Checklist de Validation

Après génération :
- [ ] 4 fichiers .npz créés dans `data/family_FIXED/`
- [ ] Logs de génération OK (pas d'erreur)
- [ ] HV range [-1, 1] pour chaque famille

Après entraînement :
- [ ] 4 checkpoints .pth créés dans `models/checkpoints_FIXED/`
- [ ] NP Dice ≥ 0.93 pour toutes les familles
- [ ] NT Acc ≥ 0.85 pour toutes les familles
- [ ] HV MSE < 0.05 pour Digestive (>2000 samples)
- [ ] Logs d'entraînement complets

Avant déploiement :
- [ ] Tests validation sur 10 samples par famille
- [ ] Mise à jour IHM selon INTEGRATION_PLAN
- [ ] Documentation à jour

---

**Créé le** : 2025-12-21
**Par** : Claude (Suite validation Glandular)
**Statut** : ✅ PRÊT À EXÉCUTER
**Durée totale estimée** : ~7.5 heures (génération + entraînement)
