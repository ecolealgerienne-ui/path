# PLAN D'IMPLÉMENTATION: Magnitude Loss (Solution A)

**Date:** 2025-12-24
**Objectif:** Forcer le modèle à prédire des gradients HV FORTS (magnitude >0.4)
**Gain attendu:** AJI 0.09 → 0.50-0.70 (gain 5-7×)

---

## 📋 MODIFICATIONS REQUISES

### Fichier: `src/models/hovernet_decoder.py`

---

### ÉTAPE 1: Ajouter méthode magnitude_loss()

**Localisation:** Après `gradient_loss()` (ligne ~300)

**Code à ajouter:**

```python
def magnitude_loss(
    self,
    hv_pred: torch.Tensor,
    hv_target: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Force le modèle à prédire des gradients FORTS aux frontières.

    PROBLÈME RÉSOLU:
    - La loss actuelle (MSE + gradient) ne pénalise PAS magnitude faible
    - Le modèle apprend à prédire des HV maps LISSES (magnitude 0.04 vs targets 0.77)
    - Watershed ne peut pas séparer instances (AJI 0.09 catastrophique)

    SOLUTION:
    - MSE sur la MAGNITUDE (sqrt(H² + V²)) au lieu des composantes séparées
    - Force le modèle à prédire des valeurs ÉLEVÉES (proches des targets)
    - Gain attendu: magnitude 0.04 → 0.40-0.60 (10-15×)

    Args:
        hv_pred: Prédictions HV (B, 2, H, W) - float [-1, 1]
        hv_target: Targets HV (B, 2, H, W) - float [-1, 1]
        mask: Masque noyaux (B, 1, H, W) - binary [0, 1]

    Returns:
        Scalar loss (MSE sur magnitudes)

    Example:
        >>> hv_pred = torch.randn(1, 2, 224, 224)  # Magnitude faible ~0.05
        >>> hv_target = torch.randn(1, 2, 224, 224) * 0.8  # Magnitude forte ~0.8
        >>> mask = torch.ones(1, 1, 224, 224)
        >>> loss = criterion.magnitude_loss(hv_pred, hv_target, mask)
        >>> # loss élevé car écart de magnitude important
    """
    # Calculer magnitude (norme L2 des composantes H et V)
    # sqrt(H² + V²) ∈ [0, sqrt(2)] ≈ [0, 1.41]
    mag_pred = torch.sqrt((hv_pred ** 2).sum(dim=1, keepdim=True) + 1e-8)  # (B, 1, H, W)
    mag_target = torch.sqrt((hv_target ** 2).sum(dim=1, keepdim=True) + 1e-8)

    # Masquer (calcul UNIQUEMENT sur pixels de noyaux)
    if mask is not None and mask.sum() > 0:
        mag_pred_masked = mag_pred * mask
        mag_target_masked = mag_target * mask

        # MSE avec normalisation par nombre de pixels masqués
        mag_loss_sum = F.mse_loss(mag_pred_masked, mag_target_masked, reduction='sum')
        n_pixels = mask.sum()
        mag_loss = mag_loss_sum / (n_pixels + 1e-8)
    else:
        # Sans masque (fallback, ne devrait jamais arriver)
        mag_loss = F.mse_loss(mag_pred, mag_target)

    return mag_loss
```

---

### ÉTAPE 2: Modifier calcul loss totale HV

**Localisation:** Méthode `forward()`, ligne ~348

**Code AVANT:**

```python
# Gradient loss (MSGE - Graham et al.)
hv_gradient = self.gradient_loss(hv_pred, hv_target, mask=mask)
hv_loss = hv_l1 + 2.0 * hv_gradient  # Équilibré: MSE + 2× gradient
```

**Code APRÈS:**

```python
# Gradient loss (MSGE - Graham et al.)
hv_gradient = self.gradient_loss(hv_pred, hv_target, mask=mask)

# Magnitude loss (NOUVEAU - 2025-12-24)
# Force le modèle à prédire gradients FORTS (magnitude >0.4)
# Targets ont magnitude 0.77 mais prédictions seulement 0.04 → ratio 0.05 (20× trop faible!)
# Solution: MSE sur magnitude pour forcer le modèle à "muscler" ses prédictions
hv_magnitude = self.magnitude_loss(hv_pred, hv_target, mask=mask)

# Loss totale HV (3 termes)
hv_loss = hv_l1 + 2.0 * hv_gradient + 1.0 * hv_magnitude
#                                      ^^^^^^^^^^^^^^^^^^
#                                      NOUVEAU terme
```

---

### ÉTAPE 3: Ajouter monitoring magnitude loss

**Localisation:** Retour `forward()`, lignes ~369-385

**Modifications:**

#### Mode adaptive (ligne 369):

```python
return total, {
    'np': np_loss.item(),
    'hv': hv_loss.item(),
    'hv_l1': hv_l1.item(),           # AJOUTER (détail MSE)
    'hv_gradient': hv_gradient.item(),  # AJOUTER (détail gradient)
    'hv_magnitude': hv_magnitude.item(),  # AJOUTER (détail magnitude)
    'nt': nt_loss.item(),
    'w_np': w_np,
    'w_hv': w_hv,
    'w_nt': w_nt,
}
```

#### Mode poids fixes (ligne 381):

```python
return total, {
    'np': np_loss.item(),
    'hv': hv_loss.item(),
    'hv_l1': hv_l1.item(),           # AJOUTER
    'hv_gradient': hv_gradient.item(),  # AJOUTER
    'hv_magnitude': hv_magnitude.item(),  # AJOUTER
    'nt': nt_loss.item(),
}
```

---

## 📊 POIDS RECOMMANDÉS

```python
hv_loss = 1.0 * hv_l1 + 2.0 * hv_gradient + 1.0 * hv_magnitude
#         ^^^^^^^       ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
#         MSE base      Force variations    Force magnitude
#         (accuracy)    (sharpness)         (strength)
```

**Justification:**
- `1.0 × hv_l1`: Assure prédictions précises (accuracy)
- `2.0 × hv_gradient`: Force variations spatiales (sharpness) - déjà présent
- `1.0 × hv_magnitude`: Force gradients forts (strength) - NOUVEAU

**Alternative (si magnitude loss domine trop):**
```python
hv_loss = 1.0 * hv_l1 + 2.0 * hv_gradient + 0.5 * hv_magnitude
```

**Test après premier training:** Si magnitude reste <0.2, augmenter à `2.0 × hv_magnitude`

---

## 🔬 VALIDATION POST-IMPLÉMENTATION

### Test 1: Vérifier magnitude loss fonctionne

**Script de test (à créer):** `scripts/validation/test_magnitude_loss.py`

```python
import torch
from src.models.hovernet_decoder import HoVerNetLoss

# Créer loss
criterion = HoVerNetLoss(lambda_np=1.0, lambda_hv=2.0, lambda_nt=1.0)

# Cas 1: Magnitude faible (pred) vs forte (target)
hv_pred_weak = torch.randn(1, 2, 224, 224) * 0.1  # Magnitude ~0.1
hv_target_strong = torch.randn(1, 2, 224, 224) * 0.8  # Magnitude ~0.8
mask = torch.ones(1, 1, 224, 224)

mag_loss_high = criterion.magnitude_loss(hv_pred_weak, hv_target_strong, mask)
print(f"Magnitude loss (faible→forte): {mag_loss_high:.4f}")  # Attendu: >0.5

# Cas 2: Magnitude forte (pred) vs forte (target)
hv_pred_strong = torch.randn(1, 2, 224, 224) * 0.8
mag_loss_low = criterion.magnitude_loss(hv_pred_strong, hv_target_strong, mask)
print(f"Magnitude loss (forte→forte): {mag_loss_low:.4f}")  # Attendu: <0.1

assert mag_loss_high > mag_loss_low * 5, "Magnitude loss ne pénalise pas assez!"
print("✅ Magnitude loss fonctionne correctement")
```

---

### Test 2: Ré-entraîner epidermal

**Commande:**

```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_hv 2.0 \
    --batch_size 16 \
    --lr 1e-4
```

**Métriques à surveiller (logs training):**

```
Epoch 10/50
  hv_l1:        0.05   (MSE base - doit rester ~0.05)
  hv_gradient:  0.08   (Gradient loss - doit rester ~0.08)
  hv_magnitude: 0.30   (NOUVEAU - doit DIMINUER au fil des epochs)
  hv_loss:      0.51   (Somme: 0.05 + 2×0.08 + 1×0.30)

Epoch 50/50
  hv_l1:        0.04
  hv_gradient:  0.06
  hv_magnitude: 0.10   ← DIMINUTION = modèle apprend à prédire magnitude forte!
  hv_loss:      0.26
```

**Bon signe:** hv_magnitude diminue au fil des epochs (modèle prédit des magnitudes plus proches des targets)

**Mauvais signe:** hv_magnitude stagne ou augmente (poids trop faible, augmenter à 2.0)

---

### Test 3: Vérifier magnitude prédictions

**Commande:**

```bash
python scripts/evaluation/compute_hv_magnitude.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10
```

**Résultat attendu:**

```
AVANT (sans magnitude loss):
  Magnitude moyenne: 0.0423
  Status: FAIL (<0.05)

APRÈS (avec magnitude loss):
  Magnitude moyenne: 0.40-0.60   ← OBJECTIF ATTEINT
  Status: SUCCESS (>0.15)
```

---

### Test 4: Vérifier AJI Ground Truth

**Commande:**

```bash
python scripts/evaluation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10
```

**Résultat attendu:**

```
AVANT (sans magnitude loss):
  NP Dice: 0.95
  AJI:     0.09   ← CATASTROPHIQUE

APRÈS (avec magnitude loss):
  NP Dice: 0.92-0.95  (peut légèrement diminuer, acceptable)
  AJI:     0.50-0.70  ← OBJECTIF ATTEINT (gain 5-7×)
```

---

## 📈 PROGRESSION ATTENDUE

| Phase | Magnitude Pred | HV MSE | AJI | Statut |
|-------|----------------|--------|-----|--------|
| **Baseline** (λ_hv=2.0) | 0.022 | 0.16 | 0.09 | ❌ Giant Blob |
| **λ_hv=3.0** | 0.053 | 0.16 | ~0.15 | ⚠️ Partiel |
| **λ_hv=5.0** | 0.042 | 0.16 | ~0.09 | ❌ Plateau |
| **+ Magnitude Loss** | **0.40-0.60** | **0.20-0.25** | **0.50-0.70** | ✅ **SUCCÈS** |

**Notes:**
- HV MSE peut augmenter (0.16 → 0.20-0.25) — C'EST NORMAL
- Prédire des gradients forts est plus difficile → MSE plus élevé
- Mais AJI s'améliore drastiquement → C'est ce qui compte!

---

## ⚠️ POINTS D'ATTENTION

### 1. Si magnitude loss domine trop

**Symptôme:** HV MSE explose (>0.50), NP Dice chute (<0.85)

**Solution:** Réduire poids magnitude loss
```python
hv_loss = hv_l1 + 2.0 * hv_gradient + 0.3 * hv_magnitude  # Au lieu de 1.0
```

---

### 2. Si magnitude stagne malgré la loss

**Symptôme:** Après 50 epochs, magnitude reste <0.20

**Solutions possibles:**
- Augmenter poids: `2.0 * hv_magnitude` au lieu de `1.0`
- Vérifier tanh activation (ligne 118) — doit être présent
- Vérifier gradient clipping dans optimizer

---

### 3. Si AJI ne s'améliore pas malgré magnitude élevée

**Symptôme:** Magnitude pred >0.40 mais AJI toujours ~0.15

**Diagnostic:** Problème post-processing watershed
- Vérifier paramètres watershed (dist_threshold, min_size)
- Voir `scripts/evaluation/test_watershed_params.py`

---

## 🎯 CRITÈRES DE SUCCÈS

| Métrique | Avant | Cible | Seuil Succès |
|----------|-------|-------|--------------|
| **Magnitude pred** | 0.04 | 0.50 | **>0.40** ✅ |
| **AJI** | 0.09 | 0.65 | **>0.50** ✅ |
| HV MSE | 0.16 | 0.25 | <0.30 (acceptable) |
| NP Dice | 0.95 | 0.93 | >0.90 (tolérance -2%) |

**Si AJI >0.50 ET magnitude >0.40:** ✅ **SUCCÈS COMPLET**

**Si magnitude >0.40 MAIS AJI <0.40:** ⚠️ **SUCCÈS PARTIEL** (problème post-processing)

**Si magnitude <0.30:** ❌ **ÉCHEC** (augmenter poids magnitude loss)

---

## 📋 CHECKLIST D'IMPLÉMENTATION

- [ ] ÉTAPE 1: Ajouter méthode `magnitude_loss()` dans hovernet_decoder.py
- [ ] ÉTAPE 2: Modifier calcul `hv_loss` (ligne 348) pour inclure magnitude
- [ ] ÉTAPE 3: Ajouter monitoring dans retour `forward()` (lignes 369, 381)
- [ ] ÉTAPE 4: Créer `test_magnitude_loss.py` pour valider fonction
- [ ] ÉTAPE 5: Exécuter test unitaire magnitude loss
- [ ] ÉTAPE 6: Ré-entraîner epidermal (50 epochs)
- [ ] ÉTAPE 7: Vérifier magnitude prédictions (>0.40)
- [ ] ÉTAPE 8: Vérifier AJI (>0.50)
- [ ] ÉTAPE 9: Documenter résultats dans CLAUDE.md

**Temps estimé total:** 2 heures

---

**Dernière mise à jour:** 2025-12-24
**Statut:** Prêt pour implémentation
**Prochaine action:** Implémenter ÉTAPE 1 (ajouter magnitude_loss)
