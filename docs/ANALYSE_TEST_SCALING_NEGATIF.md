# Analyse Test Scaling - Résultat Négatif

**Date:** 2025-12-24
**Test:** `test_hv_scaling.py` sur 10 échantillons epidermal
**Résultat:** ❌ Scaling n'améliore PAS l'AJI (0.0905 constant)

---

## Résultats Bruts

### Peaks Détectés (Échantillon 9)

| Scaling | Energy Range | Energy Mean | Peaks Found |
|---------|--------------|-------------|-------------|
| 1.0x | [0.0019, 0.0209] | 0.0095 | **137** |
| 5.0x | [0.0093, 0.1043] | 0.0476 | **137** |
| 10.0x | [0.0186, 0.2086] | 0.0953 | **137** |
| 20.0x | [0.0371, 0.4171] | 0.1905 | **137** |
| 50.0x | [0.0928, 1.0428] | 0.4763 | **137** |

### AJI Final

```
   Scale |   AJI Mean |    AJI Std |    Amélioration
────────────────────────────────────────────────────
     1.0 |     0.0905 |     0.1120 |  (baseline)
     5.0 |     0.0905 |     0.1120 |  (❌ 0.0%)
    10.0 |     0.0905 |     0.1120 |  (❌ 0.0%)
    20.0 |     0.0905 |     0.1120 |  (❌ 0.0%)
    50.0 |     0.0905 |     0.1120 |  (❌ 0.0%)
```

**Conclusion:** Aucune amélioration, même avec x50 scaling.

---

## Analyse Expert

### 1. Les Peaks Sont Détectés ✅

**Observation:** `Peaks found: 137` reste constant pour tous les scaling factors.

**Signification:**
L'algorithme `peak_local_max` identifie **déjà correctement** les centres des 137 cellules, même avec les gradients HV faibles (magnitude 0.02).

**Conclusion:**
Le problème n'est **PAS** la détection des marqueurs (seeds) pour le Watershed.

### 2. L'Energy Augmente Correctement ✅

**Observation:** Energy max passe de 0.021 (x1) à 1.043 (x50).

**Signification:**
Le scaling amplifie bien les gradients HV comme prévu.

**Mais:** L'AJI reste à 0.09 (aucun changement).

### 3. Le Watershed Ne Profite Pas du Scaling ❌

**Observation:** AJI identique (0.0905) pour tous les scaling factors.

**Signification:**
Même avec des gradients HV amplifiés (x50), le Watershed produit **exactement les mêmes instances**.

**Hypothèses:**

#### A. Giant Blob (Plus Probable)

Le Watershed fusionne toutes les cellules en **1 instance géante**, peu importe la magnitude des gradients.

**Pourquoi?**
Les gradients HV pointent peut-être tous dans la **mauvaise direction** ou sont **spatialement incohérents**.

**Test:**
Visualiser `inst_pred` - si toutes les cellules ont la même couleur → Giant Blob confirmé.

#### B. Watershed Mal Configuré

Les paramètres `watershed(-energy, markers, mask=binary_mask)` ne séparent peut-être pas correctement.

**Pourquoi?**
- Le signe négatif `-energy` pourrait être incorrect
- Le masque binaire pourrait être trop permissif
- Les markers (137 peaks) ne sont peut-être pas utilisés correctement

#### C. ID Mismatch (Moins Probable)

Les instances sont correctement séparées, mais la comparaison avec `inst_target` échoue.

**Pourquoi?**
- `inst_target` est décalé spatialement (resize mal fait)
- `inst_target` contient des IDs corrompus

**Test:**
Visualiser `inst_pred` vs `inst_target` - si les formes sont similaires mais décalées → ID Mismatch confirmé.

---

## Diagnostic Visuel Requis

### Script Créé

`scripts/evaluation/visualize_instance_maps.py`

### Exécution

```bash
python scripts/evaluation/visualize_instance_maps.py
```

### Ce que le script fait

1. Charge échantillon 9 (index 8)
2. Inférence modèle → `np_pred`, `hv_pred`
3. Resize 224→256 avec **INTER_NEAREST** (comme recommandé expert)
4. Post-processing Watershed → `inst_pred`
5. Compare avec `inst_target`

### Visualisations Générées

**Fichier:** `results/diagnostic_instance_maps_sample9.png`

**Contenu:**
```
Row 1 (Prédictions):
  [NP Pred Binary] [HV Magnitude] [Instances PRED]

Row 2 (Ground Truth):
  [NP GT Binary]   [HV GT Magnitude] [Instances GT]
```

### Questions à Répondre

**1. Instances PRED (Row 1, colonne 3):**
- ❓ Est-ce **UNE SEULE couleur** (Giant Blob)?
- ❓ Ou **PLUSIEURS couleurs** (multiples instances)?

**2. Nombre d'instances:**
- ❓ Combien d'instances dans PRED?
- ❓ Combien d'instances dans GT?

**3. Alignment:**
- ❓ Les formes sont-elles **similaires** mais décalées?
- ❓ Ou complètement **différentes**?

---

## Scénarios Possibles

### Scénario A: Giant Blob (PRED = 1 instance)

**Symptôme:**
Instances PRED montre **UNE SEULE couleur uniforme** couvrant tous les noyaux.

**Cause:**
Bug #3 (Instance Mismatch) - Le modèle a été entraîné sur des instances **fusionnées** (connectedComponents sur union binaire).

**Solution:**
Ré-entraîner avec **vraies instances** PanNuke (extraire IDs des canaux 1-4 au lieu de connectedComponents).

**Coût:** ~2 heures (re-train epidermal uniquement)

**Documentation:** CLAUDE.md lignes 745-819

### Scénario B: Sous-Segmentation (PRED = 10-50 instances)

**Symptôme:**
Instances PRED montre **plusieurs couleurs**, mais beaucoup moins que GT (ex: 30 pred vs 137 GT).

**Cause:**
Paramètres Watershed trop conservateurs (`min_distance=2`, `min_size=10`).

**Solution:**
Ajuster paramètres:
- `min_distance=1` (au lieu de 2)
- `min_size=5` (au lieu de 10)

**Coût:** 5 minutes (pas de re-train)

### Scénario C: ID Mismatch (PRED ≈ GT instances, mais AJI faible)

**Symptôme:**
Instances PRED montre **~137 couleurs** (similaire à GT), mais décalées ou mal alignées.

**Cause:**
- `inst_target` resize avec INTER_LINEAR (lisse les IDs)
- Décalage spatial d'un pixel (destroy AJI)

**Solution:**
Vérifier que `inst_target` est chargé SANS resize, ou avec INTER_NEAREST uniquement.

**Coût:** 5 minutes (fix script)

---

## Action Immédiate

### Commande

```bash
python scripts/evaluation/visualize_instance_maps.py
```

### Temps Estimé

- Chargement modèle: 10s
- Inférence: 5s
- Visualisation: 5s
- **Total: 20 secondes**

### Fichier de Sortie

```
results/diagnostic_instance_maps_sample9.png
```

### Partager Résultat

Une fois généré, partage:
1. Le nombre d'instances PRED vs GT (affiché dans le terminal)
2. Le diagnostic final (Giant Blob / ID Mismatch / OK)
3. L'image `diagnostic_instance_maps_sample9.png` (si possible)

---

## Prédiction Expert

> "Ma recommandation technique immédiate: Vérifie la ligne de ton code qui redimensionne les instances.
> Si tu utilises `cv2.resize` sur des `inst_maps` (les étiquettes d'ID), tu détruis les données."

**Script actuel utilise déjà INTER_NEAREST** ✅

Donc si le problème persiste, c'est **probablement un Giant Blob** (Bug #3).

---

## Timeline

| Étape | Temps | Action |
|-------|-------|--------|
| **Visualisation** | 20s | Exécuter `visualize_instance_maps.py` |
| **Analyse** | 2 min | Identifier scénario (A/B/C) |
| **Fix (si B ou C)** | 5 min | Ajuster paramètres ou resize |
| **Re-train (si A)** | 2h | Bug #3 - Vraies instances PanNuke |

---

## Références

- **Expert Analysis:** Messages utilisateur 2025-12-24
- **Bug #3:** CLAUDE.md lignes 745-819
- **HoVer-Net Paper:** Graham et al. 2019
- **Script créé:** `scripts/evaluation/visualize_instance_maps.py`

---

**Statut:** ⚠️ EN ATTENTE - Diagnostic visuel requis pour identifier scénario
