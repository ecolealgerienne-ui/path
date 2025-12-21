# Diagnostic Échec Ground Truth - CRITIQUE

**Date**: 2025-12-21
**Statut**: 🚨 **ÉCHEC CRITIQUE** - Investigation en cours
**Priorité**: **BLOQUANT** pour déploiement

---

## 🚨 Résultats Test Rapide (5 échantillons)

| Métrique | Résultat | Cible | Écart | Statut |
|----------|----------|-------|-------|--------|
| **Dice** | 0.8866 | 0.95 | -6.4% | 🟠 Moyen |
| **AJI** | 0.3091 | 0.80 | **-61%** | 🔴 **CATASTROPHIQUE** |
| **PQ** | 0.1623 | 0.70 | **-77%** | 🔴 **CATASTROPHIQUE** |
| **Précision** | 14.29% | >80% | **-66%** | 🔴 **CATASTROPHIQUE** |
| **Rappel** | 35.71% | >80% | **-44%** | 🔴 **CRITIQUE** |

### Détection

```
TP:   5  (vrais positifs)
FP:  30  (faux positifs) ← 6x plus de FP que de TP !
FN:   9  (faux négatifs)

Précision: 5/(5+30) = 14.29%  ← Le modèle détecte 86% de fausses instances
Rappel:    5/(5+9)  = 35.71%  ← Le modèle manque 64% des vraies instances
```

### Type Cellulaire (Exemple le plus flagrant)

```
Epithelial:
  Expert annotées:  9 cellules
  Modèle détecte:  31 cellules  ← 3.4x sur-détection !
```

---

## 🔬 Hypothèses (Par Ordre de Probabilité)

### 1. 🎯 Watershed Post-Processing Défaillant (90% probable)

**Symptômes** :
- FP = 30 (6x plus de fausses instances que de vraies)
- Sur-détection massive Epithelial (9 → 31)

**Cause probable** : Seuils watershed trop permissifs créent trop de seeds

**Paramètres actuels** (`src/inference/hoptimus_hovernet.py:211-216`) :
```python
markers[edge > 0.3] = 0  # Supprimer bords (gradient HV)
markers = (markers > 0.7).astype(np.uint8)  # Seuil NP prob
markers = ndimage.label(markers * (dist > 2))[0]  # Distance seeds
```

**Problèmes potentiels** :
- `edge > 0.3` : **Trop permissif** ? (devrait être 0.5-0.7 pour frontières nettes)
- `dist > 2` : **Distance minimale trop faible** → Trop de seeds créés
- Résultat : Chaque petit pic de probabilité NP crée une instance

**Test à faire** :
```python
# Essayer des seuils plus stricts
markers[edge > 0.5] = 0  # Au lieu de 0.3
markers = ndimage.label(markers * (dist > 5))[0]  # Au lieu de 2
```

### 2. 🧪 Compute_HV_Maps() Incorrect (50% probable)

**Symptôme** : Les métriques d'entraînement étaient bonnes (HV MSE 0.01-0.06), mais évaluation GT catastrophique

**Cause possible** : Les targets HV pendant l'entraînement ne correspondent pas aux vraies frontières d'instances

**À vérifier** :
- Est-ce que `compute_hv_maps()` utilise les **vrais IDs d'instances** de PanNuke ?
- Ou est-ce qu'on recalcule avec `connectedComponents` qui fusionne les cellules qui se touchent ?

**Code à inspecter** (`scripts/preprocessing/prepare_family_data_FIXED.py`) :
```python
# Si on fait ça, c'est FAUX:
_, labels = cv2.connectedComponents(binary_mask)  # Fusionne cellules touchantes
hv_targets = compute_hv_maps(labels)  # HV maps avec instances FUSIONNÉES

# Il faut faire ça:
inst_map = extract_true_instance_ids_from_pannuke(mask)  # IDs réels
hv_targets = compute_hv_maps(inst_map)  # HV maps avec vraies frontières
```

### 3. 📏 Résolution Mismatch (30% probable)

**Symptôme** : Resize 224→256 pendant l'évaluation

**Cause possible** : L'interpolation bilinéaire floute les gradients HV précis

**Test** :
- Évaluer directement à 224×224 (sans resize)
- Ou utiliser INTER_NEAREST pour préserver les gradients

### 4. 🗂️ Conversion Annotations GT Incorrecte (20% probable)

**Symptôme** : Peut-être que le script `convert_annotations.py` ne produit pas le bon format

**À vérifier** :
- Les `inst_map` dans les .npz ont-ils des IDs d'instances corrects ?
- Y a-t-il des instances fusionnées par erreur ?

---

## 🛠️ Plan d'Investigation (Ordre de Priorité)

### Étape 1: Diagnostic Visuel (URGENT - 10 min)

```bash
# Lancer diagnostic sur une image
python scripts/evaluation/diagnose_gt_failure.py \
    --npz_file <PREMIER_FICHIER_NPZ_DU_TEST> \
    --checkpoint_dir models/checkpoints_FIXED \
    --output_dir results/diagnostic_gt

# Examiner visuellement:
# - Les gradients HV sont-ils suffisamment forts ?
# - Y a-t-il trop de seeds de watershed ?
# - Les instances GT sont-elles correctes ?
```

**Fichier attendu** : `results/diagnostic_gt/diagnostic_*.png`

### Étape 2: Test Watershed Thresholds (30 min)

```bash
# Créer script test_watershed_thresholds.py
# Essayer différentes combinaisons:
#   - edge_threshold: [0.3, 0.4, 0.5, 0.6, 0.7]
#   - dist_threshold: [2, 3, 5, 7, 10]

# Trouver meilleure combinaison qui:
#   - Réduit FP (actuellement 30)
#   - Améliore Rappel (actuellement 35%)
#   - Augmente AJI (actuellement 0.31)
```

### Étape 3: Vérifier Compute_HV_Maps() (1h)

```bash
# Inspecter scripts/preprocessing/prepare_family_data_FIXED.py
# Comparer avec masks PanNuke originaux
# Vérifier si on utilise les vrais IDs d'instances ou connectedComponents
```

**Si bug trouvé** : Ré-entraînement complet requis (~10h)

### Étape 4: Test Sans Resize (15 min)

```bash
# Modifier evaluate_ground_truth.py
# Évaluer à résolution native 224×224
# Comparer métriques
```

---

## 📊 Disconnect Train vs GT

**Observation clé** :

| Phase | NP Dice | HV MSE | NT Acc | Statut |
|-------|---------|--------|--------|--------|
| **Training (Glandular)** | 0.9641 | 0.0105 | 0.9107 | ✅ Excellent |
| **Validation (10 samples)** | 0.9655 | 0.0266 | 0.9517 | ✅ Excellent |
| **Ground Truth (5 samples)** | 0.8866 | ? | ? | ❌ Catastrophe |

**Pourquoi ce disconnect ?**

1. **Training/Validation** : Compare prédictions vs targets générés par `prepare_family_data_FIXED.py`
2. **Ground Truth** : Compare prédictions vs annotations expertes PanNuke originales

**Si `compute_hv_maps()` est incorrect** :
- Le modèle apprend correctement les targets incorrects → Bonnes métriques train/val
- Mais les prédictions ne matchent pas les annotations expertes → Mauvaises métriques GT

---

## 🎯 Critères de Succès (Post-Fix)

### Minimaux (GO)

| Métrique | Cible | Actuel | Requis |
|----------|-------|--------|--------|
| Dice | 0.95 | 0.8866 | > 0.93 |
| AJI | 0.80 | 0.3091 | > 0.70 |
| PQ | 0.70 | 0.1623 | > 0.60 |
| Précision | >80% | 14.29% | > 70% |
| Rappel | >80% | 35.71% | > 70% |

### Idéaux (EXCELLENT)

| Métrique | Valeur |
|----------|--------|
| Dice | > 0.95 |
| AJI | > 0.75 |
| PQ | > 0.65 |
| Précision | > 85% |
| Rappel | > 85% |

---

## 📝 Actions Immédiates

1. **[USER]** Lancer diagnostic visuel :
   ```bash
   # Trouver première image test
   find /home/amar/data -name "*.npz" | grep -i pannuke | head -1

   # Lancer diagnostic
   python scripts/evaluation/diagnose_gt_failure.py --npz_file <PATH>
   ```

2. **[CLAUDE]** Créer script test watershed thresholds

3. **[USER]** Partager image diagnostic pour analyse visuelle

4. **[CLAUDE]** Selon diagnostic, proposer fix approprié

---

## ⚠️ Impact sur le Projet

**Si Watershed Fix Suffit** : ~30 min
- Ajuster seuils
- Re-tester GT
- Déployer si OK

**Si Compute_HV_Maps Bug** : ~10h
- Corriger preprocessing
- Ré-entraîner 5 familles
- Re-tester GT

**Si Problème Fondamental Architecture** : ~1 semaine
- Revoir HoVer-Net decoder
- Ré-entraîner
- Valider

---

**Créé le** : 2025-12-21
**Par** : Claude (Investigation échec GT)
**Statut** : 🚨 INVESTIGATION EN COURS
**Priorité** : **BLOQUANT**
