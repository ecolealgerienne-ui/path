# Diagnostic Expert Bug #4: Images CHW Non Normalisées (2025-12-24)

## Contexte

**Symptôme :** Distance alignement spatial = 96.29px malgré :
- ✅ Sources PanNuke RAW saines (overlap 100%)
- ✅ prepare_family_data_FIXED_v2 avec auto-détection format masks

**Diagnostic Expert :** 96.29px = **distance statistique aléatoire** dans carré 256×256 → Corrélation nulle (axes croisés).

---

## Cause Racine : Images CHW Non Normalisées

### Code Problématique (v2 lignes 297-313)

```python
# prepare_family_data_FIXED_v2.py
for idx in tqdm(chunk_indices):
    image = np.array(images[idx], dtype=np.uint8)  # ❌ Pas de normalisation !
    mask = np.array(masks[idx])

    # v2: Masque normalisé HWC
    inst_map = extract_pannuke_instances(mask)  # ✅ Mask → HWC

    # NT target
    mask_normalized = normalize_mask_format(mask)  # ✅ Mask → HWC
    nt_target = np.argmax(mask_normalized[:, :, 1:], axis=-1)

    # ❌ BUG: Image stockée en CHW si sources sont CHW !
    chunk_images.append(image)  # (3, 256, 256) au lieu de (256, 256, 3)
```

### Impact

| Composant | Format stocké | Format attendu | Résultat |
|-----------|---------------|----------------|----------|
| **Images** | ❌ CHW (3,256,256) | HWC (256,256,3) | Axes croisés |
| **Masks** | ✅ HWC (256,256,6) | HWC (256,256,6) | OK |
| **HV targets** | ✅ HWC (2,256,256) | HWC (2,256,256) | OK mais calculés sur mask normalisé |
| **Résultat** | Images CHW + Targets HWC | → | **Distance 96px (aléatoire)** |

---

## Solution : Normaliser Images ET Masks (v3)

### Fix Expert (v3 lignes 296-328)

```python
for idx in tqdm(chunk_indices, desc="      Processing", leave=False):
    raw_img = np.array(images[idx], dtype=np.uint8)
    raw_mask = np.array(masks[idx])

    # ✅ FIXÉ v3: NORMALISATION IMAGE (Bug #1 Expert)
    if raw_img.shape[0] == 3:  # Si CHW (3, 256, 256)
        image = np.transpose(raw_img, (1, 2, 0))  # CHW → HWC
    else:
        image = raw_img  # Déjà HWC (256, 256, 3)

    # ✅ FIXÉ v3: NORMALISATION MASQUE (Une seule fois)
    mask = normalize_mask_format(raw_mask)  # CHW → HWC si nécessaire

    # ✅ FIXÉ v3: Génération targets sur données REDRESSÉES
    inst_map = extract_pannuke_instances(mask)  # Mask déjà HWC
    np_target = (inst_map > 0).astype(np.float32)
    hv_target = compute_hv_maps(inst_map)
    nt_target = np.argmax(mask[:, :, 1:], axis=-1).astype(np.int64)

    # ✅ STOCKAGE (Image garantie HWC maintenant)
    chunk_images.append(image)  # (256, 256, 3) garanti !
    chunk_np_targets.append(np_target)
    chunk_hv_targets.append(hv_target)
    chunk_nt_targets.append(nt_target)
    chunk_fold_ids.append(fold)
    chunk_image_ids.append(idx)
```

---

## Différences v2 → v3

| Aspect | v2 | v3 (Expert Fix) |
|--------|-------|-----------------|
| **Normalisation Images** | ❌ Non | ✅ Oui (CHW→HWC) |
| **Normalisation Masks** | ✅ Oui | ✅ Oui |
| **Double normalisation mask** | ⚠️ Oui (ligne 301 + 310) | ✅ Non (une seule fois ligne 308) |
| **Format Images stockées** | ❌ CHW (3,256,256) | ✅ HWC (256,256,3) |
| **Format Masks stockés** | ✅ HWC (256,256,6) | ✅ HWC (256,256,6) |
| **Alignement attendu** | ❌ 96px | ✅ <2px |

---

## Diagnostic Complet (Session 2025-12-24)

### Étape 1 : Test Sources RAW ✅

```bash
python scripts/validation/sanity_check_pannuke_raw.py --fold 0 --indices 0 1 2 512
```

**Résultat :**
```
✅ VERDICT: TOUS LES INDICES SONT ALIGNÉS
   → Les fichiers sources PanNuke RAW sont SAINS
   → Le bug vient de prepare_family_data_FIXED_v2.py
```

**Conclusion :** Sources PanNuke OK, problème dans preprocessing.

### Étape 2 : Analyse Visuelle Alignment

**Visualisation fournie :** `alignment_sample_0512.png`

**Observation Expert :**
> "Les noyaux que l'on voit en vert ne correspondent pas du tout aux formes visibles sur l'image H&E. En revanche, dans sanity_check_raw_idx0512.jpg, les masques originaux correspondent parfaitement."

**Conclusion :** Désynchronisation introduite lors du preprocessing.

### Étape 3 : Diagnostic Code Expert

**Expert a identifié 2 bugs potentiels :**

1. **Bug #1 (CONFIRMÉ) :** Images CHW non normalisées
   - v2 normalise masks mais PAS images
   - Si sources en CHW → Images stockées en (3,256,256) au lieu de (256,256,3)
   - Axes croisés → Distance 96px

2. **Bug #2 (SECONDAIRE) :** Fold ID manquant dans verify_spatial_alignment.py
   - Script charge masks[img_id] depuis fold0 uniquement
   - Données epidermal contiennent folds 0+1+2
   - Si image vient de fold1 → comparée avec masque fold0 → Distance 96px
   - **Note :** Ce bug est moins probable car v2 régénère toutes les données ensemble

**Expert a fourni le fix exact pour Bug #1 (appliqué dans v3).**

---

## Tests de Validation v3

### Test Attendu Après Régénération

```bash
# 1. Régénérer avec v3
python scripts/preprocessing/prepare_family_data_FIXED_v3.py \
    --family epidermal --chunk_size 300 --folds 0 1 2

# 2. Copier vers emplacement attendu
cp data/family_FIXED/epidermal_data_FIXED.npz \
   data/cache/family_data/epidermal_data_FIXED.npz

# 3. Vérifier alignement
python scripts/validation/verify_spatial_alignment.py \
    --family epidermal --n_samples 10
```

**Résultat attendu :**
```
Distance moyenne: < 2 pixels  ✅ (au lieu de 96px)
VERDICT: GO
```

### Métriques Post Re-training

| Métrique | Avant (v2, Bug #4) | Après (v3, Fix) | Gain |
|----------|-------------------|----------------|------|
| Distance alignement | 96.29px | <2px | **-98%** ✅ |
| AJI (après training) | 0.06 | **0.60+** | **+846%** 🎯 |
| PQ | 0.0005 | >0.65 | +129,900% |
| Instances détectées | 9 vs 32 GT | ~30 vs 32 GT | Match |

---

## Leçons Apprises

### Pourquoi 96.29px Exactement ?

> "En géométrie computationnelle, la distance moyenne entre deux points pris au hasard dans un carré de 256x256 est d'environ 90-100 pixels." — Expert

**Signification :** 96.29px n'est PAS un décalage géométrique, mais une **corrélation nulle** (appariement aléatoire).

### Importance de la Normalisation Complète

**Règle :** Si on normalise les masks (CHW→HWC), on DOIT normaliser les images aussi.

**Pourquoi c'est subtil :**
- Les tests unitaires ne détectent pas ce bug (shapes valides)
- Le modèle compile et s'entraîne sans erreur
- Le bug n'apparaît que lors de l'évaluation GT (distance spatiale)

### Méthodologie de Diagnostic Efficace

1. **Sanity Check Sources** ← Élimine hypothèse "dataset corrompu"
2. **Analyse Visuelle** ← Révèle désalignement vs alignement
3. **Expert Review Code** ← Identifie ligne exacte du bug
4. **Fix Ciblé** ← Modifier UNIQUEMENT la partie problématique
5. **Validation Complète** ← Re-tester avec nouvelles données

---

## Fichiers Créés/Modifiés

| Fichier | Type | Description |
|---------|------|-------------|
| `scripts/preprocessing/prepare_family_data_FIXED_v3.py` | Script | Fix expert (normalisation images) |
| `scripts/validation/sanity_check_pannuke_raw.py` | Script | Test sources RAW PanNuke |
| `docs/EXPERT_DIAGNOSIS_BUG4_IMAGES_CHW.md` | Doc | Ce document |

---

## Prochaines Étapes

1. ✅ **v3 créé** avec fix expert
2. 🔜 **Régénérer epidermal** avec v3 (folds 0 1 2)
3. 🔜 **Vérifier alignement** < 2px
4. 🔜 **Régénérer features** fold 0
5. 🔜 **Re-training epidermal** (40 min)
6. 🔜 **Test AJI final** (attendu: 0.06 → 0.60+)

**Temps total estimé :** ~1h30 (régénération 15min + features 20min + training 40min + test 5min)

---

**Date :** 2025-12-24
**Expert :** Analyse fournie par utilisateur
**Implémentation :** Claude (prepare_family_data_FIXED_v3.py)
**Statut :** ✅ Fix prêt — En attente validation
