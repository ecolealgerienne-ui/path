# Vérification Extraction GT — État et Prochaines Étapes

**Date:** 2025-12-23
**Objectif:** Vérifier empiriquement si `connectedComponents` fusionne les cellules qui se touchent vs extraction native PanNuke

---

## Résumé de la Situation

### Problème Identifié

Le système OptimusGate montre une disparité importante dans les métriques AJI:
- **Sur données .npz (training):** AJI = 0.94 (excellent)
- **Sur images brutes PanNuke:** AJI = 0.30 (catastrophique)

### Hypothèse à Vérifier

La méthode `connectedComponents` utilisée dans `eval_aji_from_training_data.py` fusionne les cellules qui se touchent, créant une **fausse métrique** (compare "mauvaises instances vs mauvaises instances").

### Script de Vérification Créé

**Fichier:** `scripts/evaluation/verify_gt_extraction.py`

**Fonctionnement:**
1. Charge un échantillon depuis données training (.npz)
2. Extrait GT avec `connectedComponents` (méthode BUGGY)
3. Utilise `fold_id` et `image_id` pour charger l'image brute PanNuke correspondante
4. Extrait GT avec IDs natifs PanNuke (méthode CORRECTE)
5. Compare les deux méthodes:
   - Nombre d'instances détectées
   - Visualisation côte à côte
   - Pourcentage d'instances perdues

---

## Blocage Actuel ⚠️

### Problème de Mapping

Le script nécessite `fold_ids` et `image_ids` pour mapper les indices .npz → images brutes PanNuke.

**Diagnostic:**
```
Features keys: ['features']  ❌ Pas de fold_ids/image_ids
Targets keys: ['np_targets', 'hv_targets', 'nt_targets']  ❌ Pas de fold_ids/image_ids
```

### Formats de Données

Le projet utilise **deux formats** incompatibles:

| Format | Fichiers | fold_ids/image_ids | Localisation |
|--------|----------|-------------------|--------------|
| **OLD** (utilisé actuellement) | `{family}_features.npz`<br>`{family}_targets.npz` | ❌ NON | `data/cache/family_data/` |
| **FIXED** (recommandé) | `{family}_data_FIXED.npz` | ✅ OUI | `data/family_FIXED/` |

**Conclusion:** Les données actuelles (OLD format) ne permettent pas de faire la vérification.

---

## Solution: Générer Données FIXED

### Commande

```bash
python scripts/preprocessing/prepare_family_data_FIXED.py \
    --data_dir /home/amar/data/PanNuke \
    --family epidermal
```

### Ce que fait le script FIXED

1. **Extraction correcte des instances** (lignes 79-130):
   - Canaux 1-4: Utilise IDs natifs PanNuke (préserve instances séparées) ✅
   - Canal 5 (Epithelial): Binaire uniquement, utilise connectedComponents

2. **Sauvegarde fold_ids/image_ids** (lignes 245-246, 277-278):
   ```python
   chunk_fold_ids.append(fold)
   chunk_image_ids.append(idx)

   # Sauvegardé dans .npz:
   fold_ids=fold_ids_array,
   image_ids=image_ids_array,
   ```

3. **Optimisation RAM:**
   - Traitement par chunks de 500 images
   - Consommation: ~2 GB au lieu de 10+ GB

### Temps estimé

- Epidermal (571 samples): ~2-3 minutes
- Glandular (3535 samples): ~10-15 minutes
- Toutes les familles: ~30-40 minutes

---

## Étapes pour Compléter la Vérification

### 1. Générer FIXED Data (une seule famille pour test)

```bash
# Test rapide avec famille epidermal (571 samples)
python scripts/preprocessing/prepare_family_data_FIXED.py \
    --data_dir /home/amar/data/PanNuke \
    --family epidermal \
    --chunk_size 300
```

**Sortie attendue:**
```
✅ Saved: data/family_FIXED/epidermal_data_FIXED.npz
   Size: X.XX GB

📊 Statistics:
   Images: (571, 256, 256, 3)
   NP coverage: XX.XX%
   HV range: [-1.000, 1.000]
   NT classes: [0 1 2 3 4 5]
```

### 2. Vérifier Présence fold_ids

```bash
python scripts/utils/inspect_npz.py data/family_FIXED/epidermal_data_FIXED.npz
```

**Sortie attendue:**
```
Keys in epidermal_data_FIXED.npz:
  - images: shape (571, 256, 256, 3), dtype uint8
  - np_targets: shape (571, 256, 256), dtype float32
  - hv_targets: shape (571, 2, 256, 256), dtype float32
  - nt_targets: shape (571, 256, 256), dtype int64
  - fold_ids: shape (571,), dtype int32  ✅
  - image_ids: shape (571,), dtype int32  ✅
```

### 3. Lancer Vérification GT

```bash
python scripts/evaluation/verify_gt_extraction.py \
    --family epidermal \
    --sample_idx 0 \
    --data_dir /home/amar/data/PanNuke
```

**Sortie attendue:**
```
📥 Chargement données training (.npz)...
   Format: FIXED (single file with fold_ids/image_ids)
   File: data/family_FIXED/epidermal_data_FIXED.npz
   Sample: idx=0, fold=X, image_id=YYY

   Méthode connectedComponents:
      → N instances détectées

📥 Chargement PanNuke brut (fold X)...
   Image shape: (256, 256, 3), Mask shape: (256, 256, 6)

   Méthode extract_pannuke_native:
      → M instances détectées  (M > N attendu !)

══════════════════════════════════════════════════════════════════
RÉSULTATS COMPARAISON
══════════════════════════════════════════════════════════════════

connectedComponents:    N instances
PanNuke Native:         M instances
Différence:             (M - N) instances perdues
Perte:                  XX.X%

📊 Génération visualisation...
   ✅ Sauvegardé: results/verify_gt_epidermal_sample0.png
```

### 4. Analyser Visualisation

**Fichier:** `results/verify_gt_epidermal_sample0.png`

**Contenu:**
```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  Image H&E (fold X) │ connectedComponents │ PanNuke Native      │
│                     │    N instances      │    M instances      │
│                     │    (ROUGE)          │    (VERT)           │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

**Interprétation:**
- Si **M > N** (ex: 13 vs 9): Hypothèse **CONFIRMÉE** ✅
  → `connectedComponents` fusionne effectivement les cellules touchantes
- Si **M ≈ N**: Hypothèse **REJETÉE** ❌
  → Le problème vient d'ailleurs (watershed? autre?)

---

## Résultats Partiels (sample_idx=0, epidermal)

### Test avec OLD data (sans vérification complète)

```
connectedComponents: 1 instance détectée
```

**⚠️ Suspicieux:** Epidermal devrait avoir ~13 instances typiquement.

**Explications possibles:**
1. Échantillon majoritairement background (légitime)
2. Problème de resize 256→224 durant training
3. Problème de qualité des données OLD

**À vérifier avec FIXED data** pour trancher.

---

## Impact sur le Projet

### Si Hypothèse CONFIRMÉE (M > N)

**Diagnostic:**
- Les données training utilisent des instances fusionnées (connectedComponents)
- Le modèle apprend des gradients HV **faibles** aux frontières
- Le watershed ne peut pas séparer les instances car les gradients appris sont insuffisants

**Solution court terme:**
- Améliorer post-processing watershed (thresholds, markers)
- Gain attendu: AJI +40% (0.30 → 0.42)

**Solution long terme:**
- Ré-entraîner avec données FIXED (vraies instances séparées)
- Le modèle apprendra des gradients HV **forts** aux frontières
- Gain attendu: AJI +100%+ (0.30 → >0.60)
- Coût: 10h GPU (5 familles)

### Si Hypothèse REJETÉE (M ≈ N)

**Diagnostic:**
- Le problème ne vient PAS de connectedComponents
- Chercher ailleurs: watershed? HV maps corrompues? resize mismatch?

**Prochaines investigations:**
- Vérifier HV maps (dtype, range, gradients)
- Vérifier watershed (paramètres, markers)
- Comparer predictions 224×224 vs GT 256×256

---

## Résumé Exécutif

| # | Action | Temps | Statut |
|---|--------|-------|--------|
| 1 | Créer script vérification | 30 min | ✅ FAIT |
| 2 | Générer FIXED data (epidermal) | 3 min | ⏳ À FAIRE |
| 3 | Lancer vérification | 10 sec | ⏳ À FAIRE |
| 4 | Analyser résultats | 5 min | ⏳ À FAIRE |
| **TOTAL** | | **~40 min** | |

**Blocage actuel:** Besoin de générer FIXED data pour compléter la vérification.

**Commande suivante:**
```bash
python scripts/preprocessing/prepare_family_data_FIXED.py \
    --data_dir /home/amar/data/PanNuke \
    --family epidermal \
    --chunk_size 300
```
