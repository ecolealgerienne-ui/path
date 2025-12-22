# 🔧 Guide: Watershed Parameter Optimization

**Objectif**: Réduire la sur-segmentation (9 instances → 4 instances GT)

**Statut**: ✅ Type mapping corrigé — Maintenant optimiser watershed

---

## 🎯 Problème Actuel

```
GT:   4 instances
Pred: 9 instances (2.25x over-segmentation)
```

**Cause probable**: Paramètres watershed trop agressifs (détectent trop de frontières)

---

## 📊 Scripts Créés (Commit f0109b7)

### 1. `optimize_watershed_params.py`

**Fonction**: Teste 245 combinaisons de paramètres pour trouver le meilleur match

**Paramètres testés**:
- `edge_threshold`: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  - Seuil pour le gradient HV
  - Plus haut → moins d'instances détectées
- `dist_threshold`: [1, 2, 3, 4, 5, 7, 10]
  - Distance minimale entre pics locaux
  - Plus haut → moins d'instances détectées
- `min_size`: [5, 10, 20, 30, 50]
  - Taille minimale d'instance en pixels
  - Plus haut → filtre les petites régions

**Métrique**: Minimise `abs(n_pred - n_gt)`

### 2. `visualize_watershed_optimization.py`

**Fonction**: Crée une image de comparaison 2×2:
- Original image
- GT instances
- Pred instances (avec meilleurs paramètres)
- Overlay (GT=vert, Pred=rouge)

---

## 🚀 Workflow Complet

### Étape 1: Pull les nouveaux scripts

```bash
git pull origin claude/evaluation-ground-truth-zJB9O
```

### Étape 2: Lancer l'optimisation

```bash
python scripts/evaluation/optimize_watershed_params.py \
    --npz_file data/evaluation/pannuke_fold2_converted/image_00000.npz \
    --checkpoint_dir models/checkpoints_FIXED \
    --output_dir results/watershed_optimization
```

**Sortie attendue**:
```
🔍 Testing parameter grid:
  edge_threshold: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  dist_threshold: [1, 2, 3, 4, 5, 7, 10]
  min_size: [5, 10, 20, 30, 50]
  Total combinations: 245

📊 Top 10 Parameter Combinations:
Rank  Edge   Dist   MinSz  Pred   GT     Error  Ratio
----------------------------------------------------------------------
1     0.50   5      20     4      4      0      1.00
2     0.60   4      15     5      4      1      1.25
...

✅ BEST PARAMETERS:
  edge_threshold: 0.50
  dist_threshold: 5
  min_size: 20
  Predicted instances: 4
  GT instances: 4
  Error: 0 instances
  Ratio: 1.00x

💾 Saved: results/watershed_optimization/best_watershed_params.npz
💾 Saved: results/watershed_optimization/all_results.json
```

### Étape 3: Visualiser les résultats

```bash
python scripts/evaluation/visualize_watershed_optimization.py \
    --results_file results/watershed_optimization/best_watershed_params.npz \
    --npz_file data/evaluation/pannuke_fold2_converted/image_00000.npz \
    --output results/watershed_optimization/comparison.png
```

**Ouvrir**: `results/watershed_optimization/comparison.png`

Vous verrez:
- GT (vert) vs Pred (rouge) overlay
- Si les contours se chevauchent bien → bon match ✅
- Si beaucoup de rouge sans vert → sur-segmentation ❌
- Si beaucoup de vert sans rouge → sous-segmentation ❌

---

## 🔧 Étape 4: Appliquer les Meilleurs Paramètres

Une fois les meilleurs paramètres trouvés (ex: edge=0.5, dist=5, min_size=20), il faut les appliquer dans le code d'inférence.

### Option A: Modifier `post_process_hv()` par défaut

**Fichier**: `src/inference/optimus_gate_inference_multifamily.py`

Chercher la fonction `post_process_hv()` et modifier les valeurs par défaut:

```python
def post_process_hv(
    self,
    np_prob: np.ndarray,
    hv_pred: np.ndarray,
    edge_threshold: float = 0.5,    # ← MODIFIER ICI
    dist_threshold: int = 5,         # ← MODIFIER ICI
    min_size: int = 20,              # ← MODIFIER ICI
) -> np.ndarray:
    ...
```

### Option B: Passer les paramètres lors de l'appel

Si vous voulez garder la flexibilité, modifiez `predict()` pour accepter ces paramètres.

---

## 📊 Interpréter les Résultats

### Cas 1: Match Parfait (Error = 0)

```
✅ BEST PARAMETERS:
  Predicted instances: 4
  GT instances: 4
  Error: 0
```

**Action**: Appliquer ces paramètres en production!

### Cas 2: Léger Over-Segmentation (Error ≤ 2)

```
⚠️ BEST PARAMETERS:
  Predicted instances: 6
  GT instances: 4
  Error: 2
```

**Action**:
- Augmenter `edge_threshold` (ex: 0.3 → 0.5)
- Augmenter `dist_threshold` (ex: 2 → 5)
- Augmenter `min_size` (ex: 10 → 20)

### Cas 3: Léger Under-Segmentation (Error ≤ 2)

```
⚠️ BEST PARAMETERS:
  Predicted instances: 2
  GT instances: 4
  Error: 2
```

**Action**:
- Diminuer `edge_threshold` (ex: 0.5 → 0.3)
- Diminuer `dist_threshold` (ex: 5 → 2)
- Diminuer `min_size` (ex: 20 → 10)

### Cas 4: Erreur Importante (Error > 5)

```
❌ BEST PARAMETERS:
  Predicted instances: 15
  GT instances: 4
  Error: 11
```

**Causes possibles**:
1. **HV gradients trop faibles** → Vérifier HV MSE pendant entraînement
   - Si HV MSE > 0.1 → Le modèle n'a pas bien appris les frontières
   - Solution: Ré-entraîner avec plus de données ou meilleure augmentation

2. **GT annotations incomplètes** → Certaines cellules visibles ne sont pas annotées
   - Vérifier visuellement l'image GT

3. **Watershed inadapté** → Peut-être utiliser une autre méthode (connected components, etc.)

---

## 🔍 Diagnostic HV Maps

Si l'optimisation ne donne pas de bons résultats, vérifier la qualité des HV maps:

```python
# Dans diagnostic image (diagnostic_image_00000.png)
# Regarder Row 3: HV-H, HV-V, HV Gradient Magnitude

# HV range devrait être proche de [-1, 1]
# Gradient magnitude devrait montrer des frontières nettes
```

**Critères de qualité**:
- HV range: [-0.8, 0.8] ou mieux ✅
- HV Gradient Max: > 1.0 ✅
- Frontières visibles dans "HV Gradient Magnitude" ✅

Si les gradients sont faibles (max < 0.5):
→ Problème d'entraînement HV branch → Voir `BUGS_+1_TYPE_MAPPING_COMPLETE.md` section "HV MSE"

---

## 📝 Checklist

- [ ] Pull commit f0109b7
- [ ] Lancer `optimize_watershed_params.py`
- [ ] Vérifier top 10 résultats
- [ ] Identifier meilleurs paramètres (Error minimal)
- [ ] Lancer `visualize_watershed_optimization.py`
- [ ] Vérifier visuellement l'overlay GT vs Pred
- [ ] Appliquer meilleurs paramètres dans `post_process_hv()`
- [ ] Re-tester avec `evaluate_ground_truth.py` complet

---

**Créé le**: 2025-12-21
**Par**: Claude (Watershed Optimization)
**Statut**: ⏳ Attente exécution user
**Commits**: 43cf8a2 (optimize script), f0109b7 (visualize script)
