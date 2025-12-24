# 🔧 FIX "GRADIENT KILLER" — Diagnostic Expert (2025-12-24)

**Statut:** ✅ CAUSE RACINE IDENTIFIÉE — Fix à appliquer

---

## 🎯 Résumé Exécutif

L'expert a identifié **3 bugs critiques** dans `magnitude_loss()` qui expliquent pourquoi:
- Test 3 échoue (gradients morts)
- Magnitude plafonne à 0.022 au lieu de 0.8+
- Giant Blob persiste (AJI 0.09)

**Impact attendu du fix:** Magnitude 0.02 → 0.4+ après 5 epochs, AJI 0.09 → 0.6+

---

## 🐛 Les 3 Bugs Identifiés

### Bug #1: Epsilon Mal Placé (Gradient Killer)

**Code actuel (BUGUÉ - ligne 345):**
```python
mag_pred = torch.sqrt((hv_pred ** 2).sum(dim=1, keepdim=True) + 1e-8)
#                                                               ^^^^^^
#                                                               Epsilon APRÈS la racine
```

**Problème:**
- Quand `(hv_pred ** 2).sum() ≈ 0`, on calcule `sqrt(0) = 0`
- Puis on ajoute `1e-8` → `0 + 1e-8 = 1e-8`
- Le gradient de `sqrt(0)` est **infini** ou **NaN**
- PyTorch détache le tenseur → **gradients morts** (Test 3 échoue)

**Fix:**
```python
mag_pred = torch.sqrt(torch.sum(pred**2, dim=1) + 1e-6)
#                                                ^^^^^^
#                                                Epsilon DANS la racine
```

**Explication:**
- On calcule `sqrt(0 + 1e-6) = sqrt(1e-6) ≈ 1e-3`
- Gradient de `sqrt(1e-6)` est bien défini: `0.5 / sqrt(1e-6) = 500`
- Les gradients remontent correctement → Test 3 passe ✅

---

### Bug #2: F.mse_loss Voit le Fond (Normalisation Incorrecte)

**Code actuel (BUGUÉ - lignes 350-356):**
```python
mag_pred_masked = mag_pred * mask          # Multiplie par masque
mag_target_masked = mag_target * mask      # → fond devient 0

mag_loss_sum = F.mse_loss(mag_pred_masked, mag_target_masked, reduction='sum')
#              ^^^^^^^^^^^
#              Calcule sur TOUS les pixels (y compris fond masqué à 0)

n_pixels = mask.sum()
mag_loss = mag_loss_sum / (n_pixels + 1e-8)
```

**Problème:**
- `F.mse_loss` calcule: `((mag_pred_masked - mag_target_masked) ** 2).mean()`
- Fond masqué: `(0 - 0) ** 2 = 0` → **dilue le signal**
- Cellules: `(0.02 - 0.8) ** 2 = 0.61` → signal **noyé par le fond**

**Exemple concret:**
```
Image 224×224 = 50,176 pixels
Cellules: 5,000 pixels (10%)
Fond: 45,176 pixels (90%)

MSE sur cellules: (0.8 - 0.02)^2 = 0.61
MSE sur fond: (0 - 0)^2 = 0

MSE globale = (5000×0.61 + 45176×0) / 50176 = 0.061
                ^^^^^^^^^^^  ^^^^^^^^^^^
                10% signal   90% bruit

→ Le modèle "voit" une erreur faible (0.06) alors que les cellules ont erreur 0.61!
```

**Fix (Expert):**
```python
# 1. Calculer erreur AVANT masquage
loss = (mag_true - mag_pred)**2  # (B, H, W)

# 2. Appliquer masque AVANT réduction
weighted_loss = loss * mask.squeeze(1)  # Annule le fond

# 3. Normaliser SEULEMENT par pixels de cellules
return weighted_loss.sum() / (mask.sum() + 1e-6)
```

**Explication:**
```
Erreur cellules: 0.61 (inchangée)
Erreur fond: 0 × masque = 0 (éliminée avant réduction)

Loss = 5000×0.61 / 5000 = 0.61
       ^^^^^^^^^^^  ^^^^
       Signal pur   Normalisé par cellules seulement

→ Le modèle "voit" l'erreur RÉELLE (0.61) et va corriger!
```

---

### Bug #3: Lambda Magnitude Trop Faible

**Code actuel (ligne 416):**
```python
hv_loss = hv_l1 + 2.0 * hv_gradient + 1.0 * hv_magnitude
#                                     ^^^
#                                     Seulement 1.0×
```

**Problème:**
- `hv_l1 ≈ 0.02` (MSE sur composantes H/V)
- `hv_gradient ≈ 0.01` (MSE sur gradients Sobel)
- `hv_magnitude ≈ 0.61` (MSE sur magnitude)

**Loss totale:**
```
hv_loss = 0.02 + 2.0×0.01 + 1.0×0.61 = 0.65
          ^^^^   ^^^^^^^^^   ^^^^^^^^^
          3%     3%          94%
```

Le signal magnitude domine déjà, **MAIS** il est dilué par Bug #2!

**Fix Expert:**
```python
hv_loss = hv_l1 + 3.0 * hv_gradient + 5.0 * hv_magnitude
#                 ^^^                 ^^^
#                 Augmenté            Amplifié 5×
```

**Rationale:**
- Avec Bug #2 fixé, magnitude_loss va correctement voir l'erreur 0.61
- Poids 5.0× garantit que le modèle **PRIORISE** la magnitude
- Ratio gradient/magnitude = 3/5 force variations spatiales + amplitude

---

## 🔧 LE FIX COMPLET

### 1. Modifier `src/models/hovernet_decoder.py`

**Remplacer `magnitude_loss()` (lignes 302-361):**

```python
def magnitude_loss(
    self,
    hv_pred: torch.Tensor,
    hv_target: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Version corrigée par Expert (2025-12-24):

    FIXES APPLIQUÉS:
    1. Epsilon DANS la racine (stabilise gradients)
    2. Masquage AVANT réduction (élimine dilution par fond)
    3. Erreur quadratique manuelle (pas F.mse_loss)

    BUG RÉSOLU: Magnitude plafonne à 0.02 au lieu de 0.8+
    CAUSE: Fond (90% pixels) tirait tout vers le bas
    SOLUTION: Calculer loss UNIQUEMENT sur pixels de cellules

    Args:
        hv_pred: Prédictions HV (B, 2, H, W) - float [-1, 1]
        hv_target: Targets HV (B, 2, H, W) - float [-1, 1]
        mask: Masque noyaux (B, 1, H, W) - binary [0, 1]

    Returns:
        Scalar loss (MSE sur magnitudes, masqué)
    """
    # 1. Calculer magnitude avec epsilon DANS la racine
    #    Protection contre sqrt(0) qui tue les gradients
    mag_pred = torch.sqrt(torch.sum(hv_pred**2, dim=1) + 1e-6)  # (B, H, W)
    mag_true = torch.sqrt(torch.sum(hv_target**2, dim=1) + 1e-6)

    # 2. Erreur quadratique MANUELLE (pas F.mse_loss)
    #    Pour contrôler exactement le calcul
    loss = (mag_true - mag_pred)**2  # (B, H, W)

    # 3. Application du masque AVANT la réduction
    #    Critique: élimine la dilution par le fond
    if mask is not None and mask.sum() > 0:
        # Squeeze pour matcher dimensions (B, H, W)
        weighted_loss = loss * mask.squeeze(1)

        # 4. Normaliser SEULEMENT par pixels de cellules
        #    Pas par toute l'image (50k pixels) mais par cellules (~5k)
        return weighted_loss.sum() / (mask.sum() + 1e-6)
    else:
        # Fallback sans masque (ne devrait jamais arriver en pratique)
        return loss.mean()
```

**Modifier `__init__()` (ajouter lambda_magnitude):**

```python
def __init__(
    self,
    lambda_np: float = 1.0,
    lambda_hv: float = 2.0,
    lambda_nt: float = 1.0,
    lambda_magnitude: float = 5.0,  # ← NOUVEAU paramètre
    adaptive: bool = False
):
    super().__init__()
    self.lambda_np = lambda_np
    self.lambda_hv = lambda_hv
    self.lambda_nt = lambda_nt
    self.lambda_magnitude = lambda_magnitude  # ← Stocker
    self.adaptive = adaptive
    # ... reste du code
```

**Modifier calcul HV loss (ligne 416):**

```python
# Loss totale HV (3 termes)
# EXPERT FIX (2025-12-24): lambda_magnitude=5.0 pour forcer signal fort
hv_loss = hv_l1 + 3.0 * hv_gradient + self.lambda_magnitude * hv_magnitude
#                 ^^^                 ^^^^^^^^^^^^^^^^^^^^^^^
#                 Gradient amplifié    Magnitude prioritaire (5.0×)
```

---

### 2. Modifier `scripts/training/train_hovernet_family.py`

**Ajouter argument CLI:**

```python
parser.add_argument('--lambda_magnitude', type=float, default=5.0,
                    help='Poids magnitude loss (Expert: 5.0 pour forcer signal)')
```

**Passer à HoVerNetLoss:**

```python
criterion = HoVerNetLoss(
    lambda_np=args.lambda_np,
    lambda_hv=args.lambda_hv,
    lambda_nt=args.lambda_nt,
    lambda_magnitude=args.lambda_magnitude,  # ← NOUVEAU
    adaptive=args.adaptive
).to(device)
```

---

## 🚀 Plan d'Exécution

### Étape 1: Appliquer les Fixes (5 min)

```bash
# Modifier hovernet_decoder.py
# - Fonction magnitude_loss (lignes 302-361)
# - Paramètre lambda_magnitude dans __init__
# - Ligne 416: utiliser self.lambda_magnitude

# Modifier train_hovernet_family.py
# - Ajouter --lambda_magnitude argument
# - Passer à HoVerNetLoss()
```

### Étape 2: Re-training avec Nouveaux Lambdas (40 min)

```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_hv 3.0 \
    --lambda_magnitude 5.0
```

### Étape 3: Vérification Magnitude après 5 Epochs (CRITIQUE)

```bash
# Après epoch 5, tester magnitude
python scripts/evaluation/compute_hv_magnitude.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_epoch_5.pth \
    --n_samples 10
```

**Seuil de Succès:**
- Magnitude >0.25 après 5 epochs → ✅ Fix fonctionne, continuer training
- Magnitude <0.10 → ❌ Problème persistant, investiguer

### Étape 4: Test AJI Final (après 50 epochs)

```bash
python scripts/evaluation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --data_dir data/family_data \
    --n_samples 100
```

**Attendu:**
- Magnitude: 0.02 → **0.50+** (gain 25×) 🎯
- AJI: 0.09 → **0.60+** (gain 7×) 🎯
- Instances détectées: 1 Giant Blob → 8-12 cellules séparées

---

## 📊 Explication Mathématique du Fix

### Pourquoi le modèle apprenait à prédire 0.02?

**Avec le BUG (normalisation sur toute image):**

```
Loss fonction du modèle:
L(mag) = Σ(pixels) (target - pred)² / N_total

Pour le fond (90% pixels):
  target = 0, pred = 0.02
  Erreur = (0 - 0.02)² = 0.0004

Pour les cellules (10% pixels):
  target = 0.8, pred = 0.02
  Erreur = (0.8 - 0.02)² = 0.61

Loss totale:
L = (90% × 0.0004 + 10% × 0.61) / 100%
  = 0.00036 + 0.061 = 0.061

Si le modèle monte pred à 0.5:
  Fond: (0 - 0.5)² = 0.25 (×625 pire!)
  Cellules: (0.8 - 0.5)² = 0.09 (×0.15 mieux)

  Loss = 90% × 0.25 + 10% × 0.09 = 0.234 (×3.8 PIRE!)

→ L'optimiseur REFUSE de monter la magnitude car le fond "crie" trop fort
```

**Avec le FIX (normalisation sur cellules seulement):**

```
Loss fonction du modèle:
L(mag) = Σ(cellules) (target - pred)² / N_cellules

Pour le fond: IGNORÉ (masqué à 0)

Pour les cellules (100% de la loss):
  target = 0.8, pred = 0.02
  Erreur = (0.8 - 0.02)² = 0.61

  Loss = 0.61

Si le modèle monte pred à 0.5:
  Cellules: (0.8 - 0.5)² = 0.09

  Loss = 0.09 (×0.15 MIEUX!)

→ L'optimiseur VEUT monter la magnitude car récompensé uniquement sur cellules
```

---

## 🎯 Prédictions Expert

**Quote Expert:**
> "Ton code actuel était 'sourd' au signal de magnitude à cause d'une normalisation
> sur toute l'image. En focalisant la perte sur les pixels masqués, tu libères le
> modèle de la tyrannie du fond noir."

**Prédiction après Fix:**

| Métrique | Avant Fix | Après 5 Epochs | Après 50 Epochs | Gain |
|----------|-----------|----------------|-----------------|------|
| **Magnitude** | 0.022 | **0.25+** | **0.50+** | **×23** 🎯 |
| **AJI** | 0.09 | **0.30+** | **0.60+** | **×7** 🎯 |
| **Instances** | 1 blob | 5-8 cellules | 8-12 cellules | Résolu ✅ |

**Pourquoi ça va marcher:**

1. **Gradient Killer résolu** → Gradients remontent (Test 3 passe)
2. **Dilution par fond éliminée** → Signal magnitude pur (Test 1 passe)
3. **Lambda 5.0×** → Modèle priorise amplitude sur lissage
4. **Watershed voit les creux** → Séparation instances (AJI ↑)

---

## 📝 Checklist Validation

Avant de lancer le training:

- [ ] `magnitude_loss()` modifiée (epsilon dans racine + masking avant réduction)
- [ ] `__init__` accepte `lambda_magnitude` paramètre
- [ ] Ligne 416 utilise `self.lambda_magnitude`
- [ ] `train_hovernet_family.py` accepte `--lambda_magnitude`
- [ ] Argument passé à `HoVerNetLoss()`

Après 5 epochs:

- [ ] Magnitude >0.25 (indicateur que le fix fonctionne)
- [ ] HV MSE stable ~0.20 (ne doit pas exploser)

Après 50 epochs:

- [ ] Magnitude >0.50
- [ ] AJI >0.60
- [ ] Test visuel: cellules séparées (pas de blob)

---

## 🔬 Tests Unitaires Attendus (Après Fix)

Après application du fix, les tests unitaires devraient passer:

**Test 1 (Penalization) - PASSERA:**
```python
# Avant: ratio 1.91 (fail car dilution fond)
# Après: ratio >5.0 (pass car masking correct)
```

**Test 3 (Gradients) - PASSERA:**
```python
# Avant: Has grad: False (epsilon après racine)
# Après: Has grad: True (epsilon dans racine)
```

**Test 5 (Integration) - RESTERA PASS:**
```python
# Déjà pass, continuera à passer
```

---

**STATUT:** ✅ Fix documenté et prêt à appliquer

**NEXT STEP:** Appliquer les modifications puis re-training avec `lambda_hv=3.0, lambda_magnitude=5.0`
