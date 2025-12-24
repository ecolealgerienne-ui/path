# DIAGNOSTIC: Pourquoi HV MSE Plafonne à 0.16

**Date:** 2025-12-24
**Contexte:** Lambda_hv=3.0 et lambda_hv=5.0 donnent tous deux HV MSE ~0.16

---

## 🔍 DÉCOUVERTE: Conflit d'Objectifs dans la Loss Function

### Code Actuel (hovernet_decoder.py, lignes 333-348)

```python
# HV MSE (sur pixels de noyaux uniquement)
hv_mse_sum = F.mse_loss(hv_pred_masked, hv_target_masked, reduction='sum')
hv_l1 = hv_mse_sum / (mask.sum() * 2)  # Division par nombre de pixels

# Gradient loss (Sobel) - force variations spatiales
hv_gradient = self.gradient_loss(hv_pred, hv_target, mask=mask)

# Loss totale HV
hv_loss = hv_l1 + 2.0 * hv_gradient  # ← ÉQUILIBRÉ
```

### Problème Identifié

**Les deux termes de la loss HV sont en CONFLIT:**

| Terme | Objectif | Stratégie Optimale Modèle |
|-------|----------|---------------------------|
| `hv_l1` (MSE) | Minimiser erreur moyenne | Prédire valeurs **PROCHES DE LA MOYENNE** (lisses) |
| `hv_gradient` (Sobel) | Minimiser différence gradients | Prédire gradients **SIMILAIRES AUX TARGETS** |

**Résultat:**
- Le modèle apprend à prédire des HV maps **LISSES** (peu de variation)
- MSE faible (erreur moyenne acceptable) ✅
- Magnitude faible (pas de pics) ❌
- HV MSE plafonne à **0.16** (compromis optimal entre lissage et gradients)

---

## 📊 Comparaison Lambda_hv=3.0 vs 5.0

| Métrique | λ=3.0 | λ=5.0 | Différence |
|----------|-------|-------|------------|
| **HV MSE** | **0.1621** | **0.1617** | **-0.25%** (identique) |
| **HV Magnitude** | **0.0529** | **0.0423** | **-20%** (PIRE!) |
| NP Dice | 0.9527 | 0.9525 | -0.02% |
| NT Acc | 0.8961 | 0.9040 | +0.88% |

**Observation:** Augmenter lambda_hv ne change RIEN au HV MSE → Modèle a atteint un **plateau d'optimisation**.

---

## 🧠 Pourquoi Augmenter Lambda_hv Ne Marche Pas

### Loss Totale (mode poids fixes)

```python
total_loss = lambda_np * np_loss + lambda_hv * hv_loss + lambda_nt * nt_loss
```

**Avec lambda_hv=3.0:**
```python
hv_contribution = 3.0 × (hv_l1 + 2.0 × hv_gradient)
                = 3.0 × (0.05 + 2.0 × 0.08)  # Exemple
                = 3.0 × 0.21
                = 0.63
```

**Avec lambda_hv=5.0:**
```python
hv_contribution = 5.0 × (hv_l1 + 2.0 × hv_gradient)
                = 5.0 × (0.05 + 2.0 × 0.08)
                = 5.0 × 0.21
                = 1.05
```

**Mais si le modèle a déjà convergé vers "prédire des valeurs lisses":**
- Augmenter le poids ne change pas la **stratégie optimale**
- Le modèle reste bloqué dans le même minimum local
- HV MSE plafonne à 0.16 (limite architecturale/algorithmique)

---

## ❌ Ce Que la Loss Actuelle Ne Force PAS

**La loss actuelle pénalise:**
- ✅ Erreur moyenne (MSE)
- ✅ Différence de gradients (Sobel)

**Mais AUCUNE loss ne RÉCOMPENSE:**
- ❌ Magnitude élevée (`max(abs(HV))`)
- ❌ Pics forts aux frontières entre noyaux
- ❌ Contraste entre centre (0) et bord (±1)

**Résultat:** Le modèle peut minimiser la loss en prédisant des HV maps **PROCHES DE ZÉRO PARTOUT**, ce qui:
- Donne HV MSE acceptable (0.16)
- Mais magnitude catastrophique (0.04)
- Et AJI catastrophique (0.09)

---

## 🎯 Solutions Possibles

### Solution A: Ajouter Magnitude Loss (RECOMMANDÉ)

Forcer le modèle à prédire des valeurs ÉLEVÉES aux frontières:

```python
# Dans hovernet_decoder.py
def magnitude_loss(self, hv_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Pénalise les prédictions HV FAIBLES.

    Objectif: Forcer magnitude > 0.5 aux frontières
    """
    # Détecter frontières (pixels de noyaux adjacents au background)
    boundaries = detect_boundaries(mask)  # Morphological erosion

    # Magnitude aux frontières
    hv_magnitude = torch.abs(hv_pred).max(dim=1)[0]  # Max(|H|, |V|)
    boundary_magnitude = hv_magnitude * boundaries

    # Pénaliser si magnitude < 0.5
    target_magnitude = 0.5
    mag_loss = F.mse_loss(boundary_magnitude, torch.ones_like(boundary_magnitude) * target_magnitude)

    return mag_loss

# Loss totale
hv_loss = hv_l1 + 2.0 * hv_gradient + 0.5 * magnitude_loss(hv_pred, mask)
```

**Gain attendu:** Magnitude 0.04 → 0.20-0.50

---

### Solution B: Augmenter Poids Gradient Loss

Au lieu de `2.0 × hv_gradient`, tester `5.0 × hv_gradient`:

```python
hv_loss = hv_l1 + 5.0 * hv_gradient  # Force variations spatiales
```

**Mais:** Risque d'overfitting sur les gradients (bruit)

---

### Solution C: Power Transform sur HV Targets

Amplifier les valeurs élevées dans les targets:

```python
# Durant preprocessing (prepare_family_data.py)
hv_targets_amplified = np.sign(hv_targets) * np.abs(hv_targets) ** 0.5  # Power 0.5
# Range [-1, 1] → [-1, 1] mais les valeurs proches de ±1 sont plus fréquentes
```

**Effet:** Force le modèle à apprendre des distributions avec plus de valeurs extrêmes.

---

### Solution D: Vérifier Targets HV Magnitude

**Test critique:** Vérifier si les targets eux-mêmes ont une magnitude élevée:

```bash
python scripts/validation/verify_hv_targets_magnitude.py \
    --family epidermal \
    --n_samples 10
```

**Si targets magnitude < 0.1:** Le problème vient des données (Gaussian smoothing trop agressif)
**Si targets magnitude > 0.5:** Le problème vient du modèle (loss function inadéquate)

---

## 🔬 Comparaison avec Autres Familles

| Famille | Samples | HV MSE | HV Magnitude | Statut |
|---------|---------|--------|--------------|--------|
| Glandular | 3,391 | **0.0106** | ? | ✅ Excellent |
| Digestive | 2,430 | **0.0163** | ? | ✅ Excellent |
| Respiratory | 408 | **0.0500** | ? | ✅ Bon |
| Urologic | 1,101 | 0.2812 | ? | ⚠️ Dégradé |
| **Epidermal** | **571** | **0.1621** | **0.04** | ❌ **Plateau** |

**Observation:** Epidermal est coincé à mi-chemin entre "excellent" et "dégradé".

---

## 🎓 Leçon Apprise

**HV MSE élevé ≠ Magnitude élevée**

- **HV MSE 0.16:** Mesure ACCURACY (erreur quadratique moyenne)
- **HV Magnitude 0.04:** Mesure STRENGTH (max des valeurs absolues)

Un modèle peut avoir:
- MSE acceptable (prédictions "correctes en moyenne")
- Magnitude catastrophique (valeurs toutes proches de 0)

**Solution:** Ajouter une loss qui FORCE la magnitude (Solution A).

---

## 📋 Action Immédiate Recommandée

**Avant d'implémenter une solution, VÉRIFIER LES TARGETS:**

```bash
python scripts/validation/verify_hv_targets_magnitude.py \
    --family epidermal \
    --data_dir data/family_data \
    --n_samples 50
```

**Attendu:** Magnitude targets > 0.5 pour confirmer que le problème vient du modèle (pas des données).

**Si magnitude targets < 0.1:** Ré-générer targets avec moins de smoothing (Bug #7?).

---

**Dernière mise à jour:** 2025-12-24
**Statut:** Cause racine identifiée (conflit d'objectifs loss) — Solutions documentées
