# 🔍 Résumé de l'Investigation : Performances Catastrophiques

**Date** : 2025-12-22
**Statut** : 🟡 En cours - Hypothèse principale identifiée

---

## Résultats Observés

| Famille | Dice Réel | Dice Attendu | Écart | Statut |
|---------|-----------|--------------|-------|--------|
| glandular | 0.078 | 0.9648 | -92% | ❌❌❌ |
| digestive | 0.071 | 0.9634 | -93% | ❌❌❌ |
| urologic | 0.129 | 0.9318 | -86% | ❌❌ |
| respiratory | 0.052 | 0.9409 | -94% | ❌❌❌ |
| epidermal | 0.017 | 0.9542 | -98% | ❌❌❌ |

**Routage** : ✅ 100% (147/147)

---

## Hypothèses Testées

### ✅ Hypothèse #1 : Features Corrompues (INFIRMÉE)

**Test** : Vérifier CLS std dans les features d'entraînement

**Résultat** :
```
✅ fold0_features.npz  CLS std = 0.768  (OK)
✅ fold1_features.npz  CLS std = 0.768  (OK)
✅ fold2_features.npz  CLS std = 0.768  (OK)

✅ VERDICT: Features CORRECTES
   → forward_features() avec LayerNorm final OK
   → CLS std dans la plage attendue [0.70-0.90]
```

**Conclusion** : Les checkpoints ont été entraînés avec des features correctes. Le problème ne vient PAS du preprocessing H-optimus-0.

---

### ✅ Hypothèse #2 : Ground Truth Mismatch (INFIRMÉE)

**Test** : Comparer préparation GT entre entraînement et évaluation

**Résultat** :
```python
# TRAIN (prepare_family_data.py ligne 79)
np_mask = mask[:, :, 1:].sum(axis=-1) > 0

# EVAL (test_family_models_isolated.py ligne 226)
np_gt = mask[:, :, 1:].sum(axis=-1) > 0
```

**Différences identifiées** :
- **Resize direction** :
  - Train : GT 256→224 (resize GT vers taille modèle)
  - Eval : Prédictions 224→256 (resize prédictions vers taille GT)
- **Interpolation** :
  - Train NP : `nearest`, Eval NP : `linear`
  - Train NT : `nearest`, Eval NT : `linear`

**Analyse** : Les différences d'interpolation sont **mineures** et ne devraient pas causer un écart de -92%. La méthode de préparation du GT est identique (union binaire sur canaux 1-5).

**Conclusion** : Les différences de resize/interpolation ne suffisent PAS à expliquer les performances catastrophiques.

---

### 🟡 Hypothèse #3 : Prédictions Sous-Confiantes (EN COURS)

**Observation** : Dice de 0.08 suggère que **très peu de pixels sont prédits comme noyaux**.

**Analyse théorique** :
```python
# Dans compute_metrics (ligne 91)
pred_np = pred["np"] > 0.5  # Seuil de binarisation

# Si pred["np"] contient des probabilités très faibles (< 0.1)
# → Après seuil 0.5, presque tout est False
# → Très peu d'intersection avec GT
# → Dice très faible !
```

**Causes possibles** :
1. **Modèle mal calibré** : Outputs très faibles malgré entraînement correct
2. **Bias dans la dernière couche** : Initialisation incorrecte
3. **Loss non convergée** : Entraînement arrêté trop tôt
4. **Mismatch spatial** : Pixels prédits au mauvais endroit (décalage de grille)

**Script de diagnostic créé** : `scripts/evaluation/diagnose_predictions_distribution.py`

**Commande à exécuter** :
```bash
python scripts/evaluation/diagnose_predictions_distribution.py \
    results/family_validation_20251222_153551/test_samples/glandular/test_samples.npz \
    models/checkpoints/hovernet_glandular_best.pth
```

**Ce script va vérifier** :
- Distribution des probabilités NP (min/max/mean/std)
- Nombre de pixels > différents seuils (0.1, 0.2, ..., 0.9)
- Intersection avec GT après resize
- Calcul du Dice step-by-step

---

## Prochaines Étapes

### 1. Exécuter le script de diagnostic (PRIORITAIRE)

```bash
python scripts/evaluation/diagnose_predictions_distribution.py \
    results/family_validation_20251222_153551/test_samples/glandular/test_samples.npz \
    models/checkpoints/hovernet_glandular_best.pth
```

**Scénarios possibles** :

#### Scénario A : Probabilités très faibles (mean < 0.1)
```
❌ PROBLÈME CRITIQUE: Modèle sous-confiant
   → Causes: Loss non convergée, bias incorrect, mismatch subtil
   → Solution: Ré-entraîner avec monitoring renforcé
```

#### Scénario B : Seuil 0.5 trop élevé
```
⚠️  PROBLÈME: Seuil de binarisation inadapté
   → Beaucoup de pixels > 0.3 mais < 0.5
   → Solution: Ajuster seuil ou re-calibrer modèle
```

#### Scénario C : Spatial mismatch
```
⚠️  PROBLÈME: Pixels au mauvais endroit
   → Nombre de pixels correct mais faible intersection
   → Solution: Vérifier décalage de grille, resize
```

#### Scénario D : Autre problème
```
🔍 Investigation plus approfondie requise
   → Inspecter visuellement les prédictions
   → Comparer avec images d'entraînement
```

---

### 2. Si Scénario A confirmé : Ré-entraînement Contrôlé

**Durée estimée** : ~10 heures (5 familles)

**Modifications recommandées** :
1. **Monitoring renforcé** : Logger probabilités moyennes par epoch
2. **Early stopping** : Arrêter si val_loss stagne
3. **Weight initialization** : Vérifier bias initial des têtes
4. **Learning rate** : Tester avec LR plus faible (1e-4 → 5e-5)

**Commandes** :
```bash
# Tester sur une seule famille d'abord (glandular)
python scripts/training/train_hovernet_family.py \
    --family glandular \
    --epochs 50 \
    --augment \
    --lr 5e-5 \
    --monitor_probs

# Si succès, re-entraîner les 5 familles
```

---

## Chronologie des Tests

| Date | Test | Résultat | Temps |
|------|------|----------|-------|
| 2025-12-22 15:35 | Pipeline validation complet | Dice 0.08 (attendu 0.95) | 10 min |
| 2025-12-22 16:00 | Vérif dates checkpoints | Birth: 2025-12-20, Modify: 2025-12-21 | 2 min |
| 2025-12-22 16:05 | Vérif CLS std features | 0.768 (OK) ✅ | 1 min |
| 2025-12-22 16:10 | Comparaison prep GT | Identique ✅ | 5 min |
| 2025-12-22 16:20 | **EN ATTENTE** : Diagnostic prédictions | - | - |

---

## Fichiers Créés

| Fichier | Description |
|---------|-------------|
| `DIAGNOSTIC_CRITICAL_ISSUE.md` | Analyse complète (15 pages) |
| `RESUME_ACTIONS.md` | Référence rapide |
| `INVESTIGATION_SUMMARY.md` | Ce fichier |
| `scripts/evaluation/diagnose_catastrophic_results.sh` | Script de vérification automatique |
| `scripts/validation/verify_features_standalone.py` | Vérification CLS std |
| `scripts/evaluation/diagnose_predictions_distribution.py` | **Inspection prédictions (à exécuter)** |

---

## Références

- Résultats attendus : CLAUDE.md section "Résultats HoVer-Net par Famille"
- Bug #1 (ToPILImage) : CLAUDE.md section "FIX CRITIQUE: Preprocessing ToPILImage"
- Bug #2 (LayerNorm) : CLAUDE.md section "FIX CRITIQUE: LayerNorm Mismatch"
- Bug #3 (Instance Mismatch) : CLAUDE.md section "BUG #3: Training/Eval Instance Mismatch"
