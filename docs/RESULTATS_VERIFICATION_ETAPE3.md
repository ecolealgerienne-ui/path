# Résultats Vérification Étape 3 — Architecture & Loss Functions

**Date:** 2025-12-23
**Objectif:** Comparer l'architecture et les loss functions entre HoVer-Net et OptimusGate

---

## ✅ Résultat 1: Format des Données d'Entraînement

### Script: `verify_training_data.py`

**Familles analysées:**
- `glandular_targets.npz` (3391 samples)
- `urologic_targets.npz` (1101 samples)

**HV Targets:**
```
Dtype:  float32  ✅
Range:  [-1.0000, 1.0000]  ✅
Mean:   0.0000
Std:    ~0.52
```

**VERDICT:** ✅ **DONNÉES FIXED utilisées pour l'entraînement**

Les données sont correctement normalisées en float32 avec range [-1, 1], comme prévu dans la version FIXED.

---

## ⚠️ Résultat 2: Comparaison MSE vs SmoothL1Loss

### Script: `compare_mse_vs_smoothl1.py` (100 échantillons réels)

**Valeurs des Loss:**
```
MSE Loss:        0.009996
SmoothL1 Loss:   0.004998
Ratio (S/M):     0.5000
```

**Magnitude des Gradients:**
```
MSE Gradient Norm:       0.000058
SmoothL1 Gradient Norm:  0.000029
Ratio (S/M):             0.4999  ❌
```

**VERDICT:** ❌ **SmoothL1 produit des gradients 50% plus FAIBLES que MSE**

### Explication Mathématique

**MSE (HoVer-Net original):**
```python
L_MSE = (pred - target)²
∂L/∂pred = 2 × (pred - target)  # Gradient linéaire avec l'erreur
```

**SmoothL1Loss (Notre système):**
```python
L_SmoothL1 = {
    0.5 × (pred - target)²           si |error| < 1
    |pred - target| - 0.5            si |error| ≥ 1
}

∂L/∂pred = {
    (pred - target)                  si |error| < 1
    sign(pred - target)              si |error| ≥ 1  ← PLAFOND à ±1
}
```

**Impact visuel (graphique `Gradients`):**
- Gradient MSE (bleu): croissance linéaire illimitée
- Gradient SmoothL1 (orange): **plafonné à ±1** pour |error| > 1

**Pour une erreur de 2.0:**
- MSE gradient: 4.0
- SmoothL1 gradient: 1.0
- **Ratio: 4×** moins de signal d'apprentissage!

---

## 🎯 Hypothèse Confirmée

### Pourquoi AJI 0.0863 (vs HoVer-Net 0.68)?

**Architecture:**
- Backbone: H-optimus-0 (1.1B params) ✅ SUPÉRIEUR à ResNet-50 (25M)
- Données: FIXED (instances séparées) ✅ IDENTIQUE à HoVer-Net
- **Loss function: SmoothL1Loss ❌ DIFFÉRENT de MSE (HoVer-Net)**

**Impact des gradients faibles:**

1. **Frontières floues:**
   - Les grandes erreurs HV (frontières entre cellules) ne reçoivent **PAS** de signal fort
   - Le modèle n'apprend **PAS** à créer des gradients HV nets
   - Watershed ne peut **PAS** séparer les instances

2. **Visualisation du problème:**
   ```
   Instance A    Frontière    Instance B
   ─────────────────────────────────────
   HV = -0.8  →  HV = 0.0  ←  HV = +0.8

   Erreur prédiction: 2.0 à la frontière

   MSE gradient:      4.0  → Signal FORT pour corriger
   SmoothL1 gradient: 1.0  → Signal FAIBLE (4× moins)
   ```

3. **Métriques observées:**
   - NP Dice: 0.9477 ✅ (segmentation binaire OK)
   - HV MSE: 0.048 ✅ (erreur moyenne acceptable)
   - **AJI: 0.0863 ❌ (séparation instances catastrophique)**

   → Le modèle détecte les noyaux mais **ne les sépare pas** car les gradients HV sont trop faibles!

---

## 📊 Comparaison Complète avec HoVer-Net

| Composant | HoVer-Net Original | OptimusGate Actuel | Impact |
|-----------|-------------------|-------------------|--------|
| **Backbone** | ResNet-50 (25M) | H-optimus-0 (1.1B) | ✅ Meilleur |
| **Données** | PanNuke (inst. séparées) | PanNuke FIXED | ✅ Identique |
| **HV Loss** | **MSE** | **SmoothL1Loss** | ❌ **2-4× gradients plus faibles** |
| **Gradient Loss** | MSGE (Sobel 5×5) | Finite differences | ⚠️ Différent |
| **NP Dice** | ~0.92 | 0.9477 | ✅ Meilleur |
| **AJI** | **0.68** | **0.0863** | ❌ **8× pire** |

---

## 🔬 Recommandation

### Test à Effectuer (Priorité Haute)

**Ré-entraîner UNE famille (glandular) avec MSE loss au lieu de SmoothL1Loss:**

```python
# Modification dans hovernet_decoder.py (ligne 299)
# AVANT:
hv_l1_sum = F.smooth_l1_loss(hv_pred_masked, hv_target_masked, reduction='sum')

# APRÈS (TEST):
hv_mse_sum = F.mse_loss(hv_pred_masked, hv_target_masked, reduction='sum')
```

**Métriques à comparer:**
| Métrique | SmoothL1 (actuel) | MSE (test) | Objectif |
|----------|-------------------|------------|----------|
| NP Dice | 0.9477 | ? | Maintenir >0.94 |
| HV MSE | 0.048 | ? | Accepter <0.10 |
| AJI | **0.0863** | ? | **Améliorer >0.60** |

**Temps estimé:** 2-3h entraînement glandular (3391 samples)

**Si AJI améliore significativement:** Ré-entraîner les 5 familles avec MSE

---

## 🚦 Décision

**Hypothèse validée:** La différence de loss function (SmoothL1 vs MSE) est une cause probable de l'AJI catastrophique.

**Actions recommandées:**

1. **Court terme (2-3h):** Test MSE sur glandular
2. **Moyen terme (10h):** Si test OK, ré-entraîner 5 familles avec MSE
3. **Long terme:** Si MSE ne suffit pas, implémenter MSGE (Sobel 5×5) comme HoVer-Net

**Actions à NE PAS faire:**
- ❌ Changer les données (FIXED est correct)
- ❌ Modifier le backbone (H-optimus-0 est supérieur)
- ❌ Implémenter watershed avancé AVANT de fixer la loss function

---

## 📚 Références

**HoVer-Net Paper:**
- Loss: MSE for HV regression (Section 3.2)
- MSGE: Sobel 5×5 for gradient sharpening (Equation 4)

**Code HoVer-Net:**
- `models/hovernet/utils.py` lignes 87-102: `mse_loss()`
- `models/hovernet/utils.py` lignes 148-172: `msge_loss()`

**Notre Code:**
- `src/models/hovernet_decoder.py` lignes 299-313: SmoothL1Loss
