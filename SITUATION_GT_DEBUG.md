# 🚨 Situation Debug Ground Truth — ÉTAT DES LIEUX

**Date**: 2025-12-21 (suite)
**Statut**: 🔧 Debugging en cours — Fix diagnostic script appliqué
**Priorité**: **BLOQUANT** pour déploiement

---

## ✅ Ce qui est FAIT

### 1. Entraînement Complet (5/5 familles)

| Famille | Samples | NP Dice | HV MSE | NT Acc | Statut |
|---------|---------|---------|--------|--------|--------|
| **Glandular** | 3391 | **0.9641** | **0.0105** | **0.9107** | ✅ |
| **Digestive** | 2430 | **0.9636** | **0.0116** | **0.8784** | ✅ |
| **Urologic** | 1101 | **0.9311** | **0.0230** | **0.9064** | ✅ 🎁 |
| **Respiratory** | 408 | **0.9339** | **0.0565** | **0.8894** | ✅ |
| **Epidermal** | 571 | **0.9533** | **0.2620** | **0.8753** | ✅ |

**Tous les critères POC atteints** :
- NP Dice ≥ 0.93 ✅
- NT Acc ≥ 0.85 ✅
- HV MSE < 0.1 pour familles >2000 samples ✅

### 2. Scripts d'Évaluation GT

| Script | Rôle | Statut |
|--------|------|--------|
| `scripts/evaluation/convert_annotations.py` | Convertir PanNuke → .npz | ✅ |
| `scripts/evaluation/evaluate_ground_truth.py` | Évaluer Dice/AJI/PQ | ✅ |
| `scripts/evaluation/quick_test_fixed.sh` | Test rapide (5 samples) | ✅ |
| `scripts/evaluation/test_fixed_models_ground_truth.sh` | Test complet (50 samples) | ✅ |
| `scripts/evaluation/diagnose_gt_failure.py` | Diagnostic visuel | ✅ **FIX APPLIQUÉ** |
| `scripts/evaluation/test_watershed_params.py` | Test seuils watershed | ✅ |

### 3. Fixes Appliqués au Diagnostic Script

**Commit**: `Fix diagnose_gt_failure.py - handle different result key names`

**Changements** :
```python
# Debug: afficher les clés disponibles
print(f"🔍 Clés dans result: {list(result.keys())}")

# Support multiple key names
pred_inst = result.get('instance_map', result.get('inst_map', np.zeros_like(gt_inst)))
pred_type = result.get('type_map', result.get('nt_map', np.zeros_like(gt_type)))
pred_np = result.get('np_prob', result.get('np_mask', np.zeros_like(pred_inst, dtype=np.float32)))
pred_hv = result.get('hv_map', result.get('hv', np.zeros((2, *gt_inst.shape), dtype=np.float32)))
```

---

## 🚨 PROBLÈME CRITIQUE Détecté

### Résultats Test Rapide (5 échantillons)

```
╔═══════════════════════════════════════════════════════════════╗
║              RÉSULTATS GROUND TRUTH - CATASTROPHE             ║
╠═══════════════════════════════════════════════════════════════╣
║ Dice Global: 0.8866  |  AJI: 0.3091  |  PQ: 0.1623            ║
╠═══════════════════════════════════════════════════════════════╣
║ DÉTECTION                                                     ║
║   TP:   5  |  FP:  30  |  FN:   9                            ║
║   Précision: 14.29%  |  Rappel: 35.71%                       ║
╠═══════════════════════════════════════════════════════════════╣
║ TYPE CELLULAIRE (Exemple le plus flagrant)                   ║
║   Epithelial: Expert=9 → Modèle=31 (3.4x sur-détection!)     ║
╚═══════════════════════════════════════════════════════════════╝
```

**Observation critique** :
- **30 faux positifs** vs **5 vrais positifs** → 6x plus de FP que de TP !
- Le modèle crée **trop d'instances** (sur-segmentation massive)
- Dice reste acceptable (0.8866) car il mesure le chevauchement binaire
- Mais **AJI 0.31** (cible 0.80) et **PQ 0.16** (cible 0.70) = CATASTROPHE

### Disconnect Train vs Ground Truth

| Phase | NP Dice | HV MSE | NT Acc | Statut |
|-------|---------|--------|--------|--------|
| **Training (Glandular)** | 0.9641 | 0.0105 | 0.9107 | ✅ Excellent |
| **Validation (10 samples)** | 0.9655 | 0.0266 | 0.9517 | ✅ Excellent |
| **Ground Truth (5 samples)** | 0.8866 | ? | ? | ❌ Catastrophe |

**Pourquoi ce disconnect ?**

1. **Training/Validation** : Compare prédictions vs targets générés par `prepare_family_data_FIXED.py`
2. **Ground Truth** : Compare prédictions vs annotations **expertes PanNuke originales**

**→ Si `compute_hv_maps()` ou post-processing watershed sont incorrects**, les métriques train/val peuvent être bonnes (le modèle apprend correctement les targets), mais les prédictions ne matchent pas les annotations expertes.

---

## 🔬 Hypothèses (Par Ordre de Probabilité)

### 1. 🎯 Watershed Post-Processing Défaillant (90% probable)

**Symptômes** :
- FP = 30 (6x plus que TP)
- Sur-détection massive Epithelial (9 → 31)

**Cause probable** : Seuils watershed **trop permissifs** créent trop de seeds.

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

**Test disponible** : `scripts/evaluation/test_watershed_params.py`

### 2. 🧪 compute_hv_maps() Incorrect (50% probable)

**Cause possible** : Les targets HV pendant l'entraînement ne correspondent pas aux vraies frontières d'instances.

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

---

## ⏭️ PROCHAINES ÉTAPES (Ordre d'Exécution)

### ÉTAPE 1 : Diagnostic Visuel (URGENT - 5 min)

**Commandes** :
```bash
# 1. Pull le fix
git pull origin claude/evaluation-ground-truth-zJB9O

# 2. Lancer diagnostic sur une image
bash scripts/evaluation/quick_test_fixed.sh

# 3. Récupérer le premier .npz créé
FIRST_NPZ=$(find data/evaluation/pannuke_fold2_converted -name "*.npz" | head -1)
echo "Premier fichier: $FIRST_NPZ"

# 4. Lancer diagnostic visuel
python scripts/evaluation/diagnose_gt_failure.py \
    --npz_file "$FIRST_NPZ" \
    --checkpoint_dir models/checkpoints_FIXED \
    --output_dir results/diagnostic_gt

# 5. Examiner l'image générée
ls -lh results/diagnostic_gt/diagnostic_*.png
```

**Sortie attendue** :
```
🔍 Clés dans result: ['instance_map', 'type_map', 'np_prob', 'hv_map', ...]

Prédictions:
  Instances: 31
  Types: [1 2 5]
  HV range: [-0.987, 0.991]

✅ Diagnostic saved: results/diagnostic_gt/diagnostic_*.png
```

**Analyser visuellement** :
- Les gradients HV sont-ils suffisamment forts ?
- Y a-t-il trop de seeds de watershed ?
- Les instances GT sont-elles correctes ?

### ÉTAPE 2 : Test Watershed Thresholds (30 min)

**SI** le diagnostic visuel montre trop de seeds/instances :

```bash
python scripts/evaluation/test_watershed_params.py \
    --npz_file "$FIRST_NPZ" \
    --checkpoint_dir models/checkpoints_FIXED \
    --output_dir results/watershed_sweep

# Examiner les heatmaps
ls -lh results/watershed_sweep/*.png
cat results/watershed_sweep/*.json
```

**Objectif** : Trouver meilleure combinaison (edge_threshold, dist_threshold) qui :
- Réduit FP (actuellement 30)
- Améliore Rappel (actuellement 35%)
- Augmente AJI (actuellement 0.31 → cible 0.70+)

### ÉTAPE 3 : Vérifier compute_hv_maps() (1h)

**SI** watershed ne suffit pas :

```bash
# Inspecter le script de préparation
cat scripts/preprocessing/prepare_family_data_FIXED.py | grep -A 20 "connectedComponents"

# Comparer avec masks PanNuke originaux
python -c "
import numpy as np
data = np.load('/home/amar/data/PanNuke/fold2/masks.npy')
print(f'Mask shape: {data.shape}')
print(f'Channels: {data.shape[-1]}')  # Devrait être 6 (BG + 5 classes)
"
```

**Vérifier** :
- Si on utilise les vrais IDs d'instances (canaux 1-5 de PanNuke)
- Ou si `connectedComponents` fusionne les cellules qui se touchent

**Si bug trouvé** : Ré-entraînement complet requis (~10h)

### ÉTAPE 4 : Test Sans Resize (15 min)

```bash
# Modifier evaluate_ground_truth.py pour évaluer à 224×224
# Comparer métriques
```

---

## 📊 Critères de Succès (Post-Fix)

| Métrique | Actuel | Cible GO | Cible EXCELLENT |
|----------|--------|----------|-----------------|
| **Dice** | 0.8866 | > 0.93 | > 0.95 |
| **AJI** | 0.3091 | > 0.70 | > 0.75 |
| **PQ** | 0.1623 | > 0.60 | > 0.65 |
| **Précision** | 14.29% | > 70% | > 85% |
| **Rappel** | 35.71% | > 70% | > 85% |

---

## ⚙️ Impact sur le Projet

### Si Watershed Fix Suffit : ~30 min
- Ajuster seuils dans `hoptimus_hovernet.py`
- Re-tester GT
- Déployer si OK

### Si compute_hv_maps() Bug : ~10h
- Corriger preprocessing
- Ré-entraîner 5 familles
- Re-tester GT

### Si Problème Fondamental Architecture : ~1 semaine
- Revoir HoVer-Net decoder
- Ré-entraîner
- Valider

---

## 📝 Fichiers Clés pour Debug

| Fichier | Rôle |
|---------|------|
| `src/inference/hoptimus_hovernet.py` | Post-processing watershed (lignes 211-216) |
| `scripts/preprocessing/prepare_family_data_FIXED.py` | Génération targets HV |
| `scripts/evaluation/diagnose_gt_failure.py` | Diagnostic visuel ✅ FIX APPLIQUÉ |
| `scripts/evaluation/test_watershed_params.py` | Test seuils watershed |
| `DIAGNOSTIC_GT_FAILURE.md` | Plan d'investigation complet |

---

**Créé le** : 2025-12-21
**Par** : Claude (Debug Ground Truth)
**Statut** : 🔧 FIX APPLIQUÉ - Attente diagnostic visuel
**Action immédiate** : Pull + Lancer `diagnose_gt_failure.py`
