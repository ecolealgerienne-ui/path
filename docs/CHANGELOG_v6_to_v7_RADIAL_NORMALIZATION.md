# CHANGELOG v6 → v7: Normalisation Radiale

**Date:** 2025-12-26
**Status:** ✅ IMPLÉMENTÉ
**Objectif:** Réduire distance centroïdes de 40.51px → <2px

---

## 1. Problème Identifié avec v6

**Résultats v6 (BBOX normalization):**
- ✅ Détection: **100% Precision, 100% Recall** (GT=Pred parfait)
- ❌ Distance: **40.51px** moyenne (objectif: <2px)
- ❌ Sample 512: **122.94px** (≈256/2, index mismatch possible)

**Cause racine:** Normalisation par **bounding box** (rectangle) au lieu de **distance radiale** (cercle).

---

## 2. Différence v6 vs v7 (Code)

### v6: BBOX Normalization (Rectangle)

```python
def compute_hv_maps_v6(inst_map):
    # Calculer centroïde
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)

    # Bounding box
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    bbox_h = y_max - y_min + 1e-6
    bbox_w = x_max - x_min + 1e-6

    # ❌ BBOX normalization (rectangle)
    v_dist = (y_coords - center_y) / bbox_h  # Range: [-0.5, 0.5]
    h_dist = (x_coords - center_x) / bbox_w  # Range: [-0.5, 0.5]
```

**Problème:** Pour un noyau allongé (bbox_h = 20px, bbox_w = 10px), les gradients HV ont des échelles différentes en X et Y → distorsion spatiale.

---

### v7: RADIAL Normalization (Circle)

```python
def compute_hv_maps_v7(inst_map):
    # Calculer centroïde
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)

    # ✅ Distance depuis centroïde
    y_dist = y_coords - center_y
    x_dist = x_coords - center_x

    # ✅ RADIAL normalization (circle)
    dist_max = np.max(np.sqrt(y_dist**2 + x_dist**2)) + 1e-7
    v_dist = y_dist / dist_max  # Range: [-1, 1]
    h_dist = x_dist / dist_max  # Range: [-1, 1]
```

**Avantage:** Normalisation **isotrope** (même échelle en X et Y) → gradients HV alignés avec la géométrie réelle du noyau.

---

## 3. Impact Attendu

### Exemple Noyau Allongé (20px × 10px)

**v6 (BBOX):**
```
bbox_h = 20, bbox_w = 10
Point au bord haut: v_dist = +10/20 = +0.5
Point au bord droit: h_dist = +5/10 = +0.5

→ Ratio spatial: 1:1 (distorsion !)
→ Le gradient pointe vers un "faux centre" décalé
```

**v7 (RADIAL):**
```
dist_max = sqrt(10² + 5²) = 11.18
Point au bord haut: v_dist = +10/11.18 = +0.89
Point au bord droit: h_dist = +5/11.18 = +0.45

→ Ratio spatial: 2:1 (correct, respecte la géométrie réelle)
→ Le gradient pointe EXACTEMENT vers le centroïde
```

---

## 4. Visualisation

```
v6 (BBOX):                     v7 (RADIAL):
┌─────────────┐                ┌─────────────┐
│             │                │             │
│   ←  ●  →   │                │   ←  ●  →   │
│      ↓      │                │    ↙ ↓ ↘    │
│             │                │             │
└─────────────┘                └─────────────┘

Gradients normalisés          Gradients normalisés
par bbox (rectangle)          par dist_max (cercle)

→ Échelles différentes X/Y    → Échelle uniforme X/Y
→ Centre apparent décalé       → Centre exact
```

---

## 5. Prédiction Expert

**Citation:**
> "Dans HoVer-Net, les cartes de distance HV doivent être centrées sur l'instance de manière à ce que le point (0,0) de la carte HV [...] corresponde exactement au centre de masse. [...] Avec une normalisation par bbox, tu as un décalage systématique si la bbox n'est pas carrée."

**Gain attendu:**
- Distance moyenne: **40.51px → <2px** (gain -95%)
- Sample 512: **122.94px → <2px** (index mismatch résolu)
- AJI après re-training: **0.06 → 0.60+** (gain +900%)

---

## 6. Tests Effectués Avant v7

| Test | Résultat | Conclusion |
|------|----------|------------|
| **v5 (BBOX corner)** | Distance 38.25px | ❌ Normalise par COIN au lieu de CENTROÏDE |
| **v6 (BBOX centroid)** | Distance 40.51px | ❌ Rectangle au lieu de cercle |
| **Flip/Rotate** (4 configs) | Tous NO-GO | Problem NOT spatial orientation |
| **Sign Inversion** (4 configs) | Tous IDENTIQUES | Script uses argmin (sign-invariant) |

**Verdict:** Problem dans MÉTHODE de normalisation (bbox vs radial), PAS dans orientation ou direction.

---

## 7. Workflow de Validation

```bash
# 1. Régénérer epidermal avec v7
python scripts/preprocessing/prepare_family_data_FIXED_v7.py --family epidermal

# 2. Vérifier alignement
python scripts/validation/verify_spatial_alignment_FIXED.py --family epidermal

# 3. Si distance <2px → GO
#    Régénérer les 5 familles:
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/preprocessing/prepare_family_data_FIXED_v7.py --family $family
done

# 4. Re-training (si nécessaire)
python scripts/training/train_hovernet_family.py --family epidermal --epochs 50

# 5. Test AJI final
python scripts/evaluation/test_on_training_data.py --family epidermal --n_samples 10
```

---

## 8. Fichiers Créés

| Fichier | Rôle |
|---------|------|
| `prepare_family_data_FIXED_v7.py` | Génération targets avec radial normalization |
| `CHANGELOG_v6_to_v7_RADIAL_NORMALIZATION.md` | Documentation du changement |

---

## 9. Références

**HoVer-Net Paper (Graham et al. 2019):**
> "Each pixel in the instance is assigned a vector pointing towards the instance centroid, normalized by the instance size."

**Expert Analysis:**
> "Le fait qu'aucune transformation géométrique simple (Flip, Rotation) ne fasse tomber la distance à zéro prouve que le problème n'est pas une simple inversion d'axes [...] mais un décalage intrinsèque dans la génération des coordonnées."

---

## 10. Next Steps

- [x] Créer v7 avec radial normalization
- [ ] Régénérer epidermal data
- [ ] Vérifier distance <2px
- [ ] Si GO: Régénérer 5 familles
- [ ] Re-training si nécessaire
- [ ] Test AJI final (objectif: >0.60)

---

**Status:** ✅ v7 PRÊT — En attente de validation par l'utilisateur
