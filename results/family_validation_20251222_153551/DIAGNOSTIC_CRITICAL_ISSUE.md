# 🚨 DIAGNOSTIC CRITIQUE : Performances Catastrophiques des Modèles HoVer-Net

**Date** : 2025-12-22 15:35:51
**Statut** : ❌ BLOQUANT CRITIQUE
**Gravité** : MAXIMALE

---

## Résumé Exécutif

Le pipeline de validation par famille a révélé des **performances catastrophiques** pour tous les 5 modèles HoVer-Net, avec un écart de **92% par rapport aux performances d'entraînement documentées**.

**Résultats observés** :
- Dice moyen : 0.079 (attendu : ~0.95) → **-91.7%**
- NT Acc moyen : 0.800 (attendu : ~0.90) → **-11.1%**
- HV MSE moyen : 0.225 (acceptable pour certaines familles)

**Routage OrganHead** : ✅ Parfait (147/147 = 100%)

---

## Résultats Détaillés par Famille

### Comparaison Attendu vs Réel

| Famille | Dice Attendu | Dice Réel | Écart | HV MSE Attendu | HV MSE Réel | NT Acc Attendu | NT Acc Réel | Statut |
|---------|--------------|-----------|-------|----------------|-------------|----------------|-------------|--------|
| **glandular** | 0.9648 | 0.078 | **-92%** | 0.0106 | 0.061 | 0.9111 | 0.819 | ❌❌❌ |
| **digestive** | 0.9634 | 0.071 | **-93%** | 0.0163 | 0.216 | 0.8824 | 0.784 | ❌❌❌ |
| **urologic** | 0.9318 | 0.129 | **-86%** | 0.2812 | 0.290 | 0.9139 | 0.874 | ❌❌ |
| **respiratory** | 0.9409 | 0.052 | **-94%** | 0.0500 | 0.307 | 0.9183 | 0.747 | ❌❌❌ |
| **epidermal** | 0.9542 | 0.017 | **-98%** | 0.2653 | 0.272 | 0.8857 | 0.774 | ❌❌❌ |

**Performances attendues documentées dans CLAUDE.md (section "Résultats par Famille (PanNuke)")** : Journal de Développement > 2025-12-21 — Entraînement 5 Familles COMPLET ✅

---

## Warning Features Suspectes

```
⚠️  Sample 6: ⚠️ Features SUSPECTES (CLS std=0.699, attendu 0.70-0.90)
Vérifier le preprocessing (conversion uint8, normalisation)
```

**Analyse** :
- CLS std = 0.699, juste en dessous du seuil 0.70
- Seulement 1 échantillon sur 148 (0.7%)
- **Pas suffisant pour expliquer l'échec massif**

---

## Scénario Diagnostic

D'après le guide de validation (`docs/GUIDE_VALIDATION_PAR_FAMILLE.md`), nous sommes dans le **Scénario 2** :

### ✅ Routage OK
- OrganHead accuracy : 100% (147/147)
- Mapping ORGAN_TO_FAMILY : 100%
- **Le problème NE VIENT PAS du routage**

### ❌ Tests Isolés CATASTROPHIQUES
- Tous les modèles HoVer-Net échouent
- Dice : -86% à -98% vs attendu
- NT Acc : -4% à -19% vs attendu

### 🔍 Diagnostic

**Problème d'entraînement ou de compatibilité des checkpoints avec le code d'évaluation**

---

## Hypothèses Possibles (Par Ordre de Probabilité)

### Hypothèse #1 : Checkpoints Entraînés AVANT les Fixes de Preprocessing (TRÈS PROBABLE)

**Contexte** : Deux bugs critiques ont été découverts et fixés :
- **Bug #1** (2025-12-20) : ToPILImage avec float64 → overflow couleurs → features corrompues
- **Bug #2** (2025-12-21) : LayerNorm mismatch → CLS std 0.28 vs 0.77

**Si les checkpoints datent d'avant ces fixes** :
- Checkpoints entraînés avec **features corrompues** (CLS std ~0.28)
- Évaluation utilise **features correctes** (CLS std ~0.77)
- **Mismatch total → Prédictions aléatoires**

**Vérification requise** :
```bash
# Vérifier date de création des checkpoints
stat models/checkpoints/hovernet_glandular_best.pth | grep "Birth"

# Comparer avec date des commits de fix
git log --oneline --after="2025-12-20" --before="2025-12-22" | grep -E "fix|Fix|FIX"
```

**Si confirmé** :
- ❌ Les 5 checkpoints sont **inutilisables**
- Solution : Ré-extraire features (FIXED preprocessing) + ré-entraîner 5 familles (~10 heures)

---

### Hypothèse #2 : Ground Truth Préparé Différemment (PROBABLE)

**Contexte** : Bug #3 (Instance Mismatch) - Encore non résolu :
- `connectedComponents()` fusionne les cellules qui se touchent
- Entraînement utilise GT fusionné
- Évaluation pourrait utiliser GT différent

**Vérification requise** :
```bash
# Inspecter la préparation du GT dans test_family_models_isolated.py
grep -A 10 "np_gt = mask" scripts/evaluation/test_family_models_isolated.py

# Comparer avec prepare_family_data.py
grep -A 10 "np_mask = mask" scripts/preprocessing/prepare_family_data.py
```

**Si confirmé** :
- Le GT d'évaluation diffère du GT d'entraînement
- Solution : Harmoniser la préparation du GT

---

### Hypothèse #3 : Bug dans compute_metrics (PEU PROBABLE)

**Raison de doute** : Les 5 familles échouent de façon cohérente
- Dice faible : tous ~0.05-0.13 (sauf urologic 0.13)
- NT Acc faible : tous ~0.75-0.87

**Si c'était un bug de métrique**, on s'attendrait à :
- Résultats aléatoires entre familles
- Ou bien échec total (Dice = 0)

**Vérification requise** :
```bash
# Inspecter compute_metrics
grep -A 30 "def compute_metrics" scripts/evaluation/test_family_models_isolated.py
```

---

### Hypothèse #4 : Checkpoints Corrompus ou Mal Chargés (POSSIBLE)

**Vérification requise** :
```bash
# Vérifier intégrité des checkpoints
ls -lh models/checkpoints/*.pth

# Vérifier que les clés chargées correspondent
# (le script affiche "✅ Modèle chargé" donc load_state_dict a réussi)
```

**Si confirmé** :
- Re-télécharger ou re-créer les checkpoints

---

## Actions Immédiates Requises

### 🔴 PRIORITÉ 1 : Vérifier Date des Checkpoints

```bash
# Afficher la date de création de tous les checkpoints
for ckpt in models/checkpoints/hovernet_*_best.pth; do
    echo "=== $(basename $ckpt) ==="
    stat "$ckpt" | grep -E "Birth|Modify"
    echo ""
done

# Comparer avec les commits de fix
git log --oneline --all --graph --decorate --date=short | grep -E "2025-12-20|2025-12-21" | head -20
```

**Interprétation** :
- Si checkpoints datent **d'avant 2025-12-21** → Hypothèse #1 confirmée
- Si checkpoints datent **d'après 2025-12-21** → Chercher ailleurs

---

### 🔴 PRIORITÉ 2 : Vérifier CLS std dans Features d'Entraînement

```bash
# Vérifier les features utilisées pour entraîner les checkpoints actuels
python scripts/validation/verify_features.py --features_dir data/cache/pannuke_features

# Sortie attendue si features correctes:
# ✅ Fold 0: CLS std = 0.768 (attendu: 0.70-0.90)
# ✅ Fold 1: CLS std = 0.771 (attendu: 0.70-0.90)
# ✅ Fold 2: CLS std = 0.769 (attendu: 0.70-0.90)

# Sortie si features corrompues (Bug #2):
# ❌ Fold 0: CLS std = 0.280 (attendu: 0.70-0.90)
```

---

### 🟡 PRIORITÉ 3 : Comparer Préparation GT Train vs Eval

```bash
# Extraire la logique de préparation du GT
echo "=== TRAIN (prepare_family_data.py) ==="
grep -A 20 "np_mask = mask" scripts/preprocessing/prepare_family_data.py

echo ""
echo "=== EVAL (test_family_models_isolated.py) ==="
grep -A 20 "np_gt = mask" scripts/evaluation/test_family_models_isolated.py
```

**Si différences détectées** :
- Harmoniser les deux scripts
- Re-préparer les données d'entraînement

---

### 🟢 PRIORITÉ 4 : Inspecter un Échantillon Manuellement

```bash
# Créer un script de diagnostic pour 1 échantillon
python scripts/evaluation/visualize_raw_predictions.py \
    --sample results/family_validation_20251222_153551/test_samples/glandular/test_samples.npz \
    --index 0 \
    --checkpoint models/checkpoints/hovernet_glandular_best.pth \
    --output results/diagnostic_sample_0.png
```

**Inspection visuelle** :
- Si prédictions = bruit aléatoire → Checkpoint ou features corrompus
- Si prédictions cohérentes mais décalées → Problème de GT ou métrique

---

## Plan de Résolution

### Si Hypothèse #1 Confirmée (Checkpoints entraînés avec features corrompues)

**Coût** : ~12-15 heures (extraction + entraînement 5 familles)

```bash
# Étape 1: Ré-extraire features avec preprocessing FIXED (3 folds, ~2-3h)
for fold in 0 1 2; do
    python scripts/preprocessing/extract_features.py \
        --data_dir /home/amar/data/PanNuke \
        --fold $fold \
        --batch_size 8 \
        --chunk_size 500
done

# Étape 2: Vérifier features
python scripts/validation/verify_features.py --features_dir data/cache/pannuke_features
# Attendu: CLS std ~0.77 pour tous les folds

# Étape 3: Ré-entraîner OrganHead (~30 min)
python scripts/training/train_organ_head.py --folds 0 1 2 --epochs 50

# Étape 4: Ré-entraîner 5 familles HoVer-Net (~2h par famille = 10h total)
for family in glandular digestive urologic respiratory epidermal; do
    python scripts/training/train_hovernet_family.py \
        --family $family \
        --epochs 50 \
        --augment
done

# Étape 5: Re-tester
bash scripts/evaluation/run_family_validation_pipeline.sh /home/amar/data/PanNuke models/checkpoints
```

---

### Si Hypothèse #2 Confirmée (GT différent)

**Coût** : ~2-3 heures (harmoniser GT + re-préparer données + ré-entraîner)

```bash
# Étape 1: Identifier la différence
diff <(grep -A 20 "np_mask = mask" scripts/preprocessing/prepare_family_data.py) \
     <(grep -A 20 "np_gt = mask" scripts/evaluation/test_family_models_isolated.py)

# Étape 2: Harmoniser le code

# Étape 3: Re-préparer données famille
python scripts/preprocessing/prepare_family_data.py --family glandular
# (répéter pour les 5 familles)

# Étape 4: Ré-entraîner (~10h)
```

---

## Conclusion Temporaire

**Statut** : 🚨 **BLOQUÉ** en attente de vérification des hypothèses

**Recommandation** : Exécuter les commandes de PRIORITÉ 1 et 2 pour confirmer/infirmer l'hypothèse #1 (la plus probable).

**Impact** :
- Si Hypothèse #1 : Ré-entraînement complet requis (~12-15h)
- Si Hypothèse #2 : Correction GT + ré-entraînement (~2-3h)
- Si autre hypothèse : Investigation plus approfondie requise

---

## Références

- Bug #1 (ToPILImage) : Commit 2025-12-20, CLAUDE.md section "FIX CRITIQUE: Preprocessing ToPILImage"
- Bug #2 (LayerNorm) : Commit 2025-12-21, CLAUDE.md section "FIX CRITIQUE: LayerNorm Mismatch"
- Bug #3 (Instance Mismatch) : CLAUDE.md section "BUG #3: Training/Eval Instance Mismatch"
- Résultats attendus : CLAUDE.md section "Résultats HoVer-Net par Famille (PanNuke)"
- Guide validation : `docs/GUIDE_VALIDATION_PAR_FAMILLE.md`
