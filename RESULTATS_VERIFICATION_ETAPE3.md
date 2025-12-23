# Résultats Vérification - Étape 3

**Date** : 2025-12-23
**Objectif** : Comparer architecture et loss functions (HoVer-Net vs Notre système)

---

## 🏗️ PARTIE 1 : Comparaison Architecture

### HoVer-Net Original

**Fichier** : `/tmp/hover_net/models/hovernet/net_desc.py`

```
INPUT (256×256 ou 270×270 RGB)
         ↓
┌─────────────────────────────────────┐
│  ENCODER (ResNet-50 Preact)         │
│  • conv0: 7×7 conv + BN + ReLU      │
│  • d0: ResBlock 64  → 256  (×3)     │
│  • d1: ResBlock 256 → 512  (×4)     │
│  • d2: ResBlock 512 → 1024 (×6)     │
│  • d3: ResBlock 1024→ 2048 (×3)     │
│  • conv_bot: 1×1 conv 2048→1024     │
│                                     │
│  Paramètres : ~25M                  │
│  Pré-entraîné : ImageNet            │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  DECODER (3 branches parallèles)    │
│                                     │
│  Pour chaque branche (NP, HV, TP):  │
│  • u3: Conv + DenseBlock + Conv     │
│  • u2: Conv + DenseBlock + Conv     │
│  • u1: Conv (padded)                │
│  • u0: BN + ReLU + Conv 1×1         │
│                                     │
│  Skip connections: d3+u3, d2+u2, etc│
└─────────────────────────────────────┘
         ↓
OUTPUT (80×80 ou 164×164)
  • NP: 2 channels (background, nuclei)
  • HV: 2 channels (horizontal, vertical)
  • TP: n_types channels (si classification)
```

### Notre Système (OptimusGate)

**Fichier** : `/home/user/path/src/models/hovernet_decoder.py`

```
INPUT (224×224 RGB)
         ↓
┌─────────────────────────────────────┐
│  H-OPTIMUS-0 (gelé)                 │
│  • ViT-Giant/14                     │
│  • 1.1 milliard paramètres          │
│  • Pré-entraîné: 500k+ lames H&E    │
│                                     │
│  Output: (B, 261, 1536)             │
│    - 1 CLS token                    │
│    - 256 patch tokens (16×16)       │
│    - 4 register tokens              │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  HOVERNET DECODER                   │
│                                     │
│  Bottleneck (économie VRAM):        │
│  • 1×1 conv: 1536 → 256             │
│  • Reshape tokens → spatial (16×16) │
│                                     │
│  Tronc commun (upsampling):         │
│  • up1: 16→32  (256→128)            │
│  • up2: 32→64  (128→64)             │
│  • up3: 64→128 (64→64)              │
│  • up4: 128→224 (64→64)             │
│                                     │
│  Têtes spécialisées (légères):      │
│  • NP: Conv→Conv 64→2               │
│  • HV: Conv→Conv 64→2 + Tanh        │
│  • NT: Conv→Conv 64→5               │
└─────────────────────────────────────┘
         ↓
OUTPUT (224×224)
  • NP: 2 channels
  • HV: 2 channels (avec Tanh [-1, 1])
  • NT: 5 channels
```

### Différences Architecturales

| Aspect | HoVer-Net Original | Notre Système |
|--------|-------------------|---------------|
| **Backbone** | ResNet-50 (25M params) | H-optimus-0 (1.1B params) |
| **Pré-entraînement** | ImageNet (photos naturelles) | 500k+ lames H&E (domaine spécifique) |
| **Features** | 2048-dim (spatial) | 1536-dim (tokens) |
| **Décodeur** | DenseBlocks + skip connections | UpsampleBlocks simples |
| **Bottleneck** | ❌ Non (2048 direct) | ✅ Oui (1536→256, économie VRAM) |
| **Skip connections** | ✅ Oui (encoder→decoder) | ❌ Non (backbone gelé) |
| **Input size** | 256×256 ou 270×270 | 224×224 (fixe H-optimus-0) |
| **Output size** | 164×164 ou 80×80 | 224×224 |
| **Activation HV** | ❌ Non (outputs directs) | ✅ Tanh (force [-1, 1]) |

**Impact théorique** :
- ✅ **Avantage** : Backbone 44× plus gros, pré-entraîné sur domaine
- ⚠️ **Inconvénient** : Pas de skip connections (backbone gelé)
- ⚠️ **Inconvénient** : Décodeur plus simple (pas de DenseBlocks)

---

## ⚖️ PARTIE 2 : Comparaison Loss Functions

### HoVer-Net Original

**Fichier** : `/tmp/hover_net/models/hovernet/utils.py`

**Configuration** (`opt.py` lignes 47-51):
```python
"loss": {
    "np": {"bce": 1, "dice": 1},
    "hv": {"mse": 1, "msge": 1},  # ← MSE + MSGE
    "tp": {"bce": 1, "dice": 1},
},
```

**Implémentation Loss HV:**

#### 1. MSE Loss (lignes 87-102)
```python
def mse_loss(true, pred):
    """Mean squared error."""
    loss = pred - true
    loss = (loss * loss).mean()
    return loss
```

**Caractéristiques** :
- MSE simple **NON MASQUÉ**
- Calculé sur **TOUS les pixels** (background + noyaux)
- Sensible aux outliers (pénalité quadratique)

#### 2. MSGE Loss (lignes 106-172)
```python
def msge_loss(true, pred, focus):
    """Mean squared error of gradients."""
    # Sobel 5×5 kernel
    kernel_h, kernel_v = get_sobel_kernel(5)

    # Calcul gradients
    true_grad = get_gradient_hv(true)  # Sobel sur H et V
    pred_grad = get_gradient_hv(pred)

    # MSE sur gradients MASQUÉ par focus (noyaux uniquement)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    loss = loss.sum() / (focus.sum() + 1e-8)
    return loss
```

**Caractéristiques** :
- **Sobel 5×5** pour calculer gradients (smoothing + dérivée)
- MSE sur gradients **MASQUÉ** (noyaux uniquement via `focus`)
- Normalisation par nombre de pixels de noyaux

### Notre Système

**Fichier** : `/home/user/path/src/models/hovernet_decoder.py`

**Configuration** (lignes 206-208):
```python
lambda_np = 1.0
lambda_hv = 2.0  # ← Pondération 2× pour HV
lambda_nt = 1.0
```

**Implémentation Loss HV:**

#### 1. SmoothL1Loss MASQUÉ (lignes 299-313)
```python
# Créer masque des noyaux
mask = np_target.float().unsqueeze(1)  # (B, 1, H, W)

# Masquer pred et target
hv_pred_masked = hv_pred * mask
hv_target_masked = hv_target * mask

# SmoothL1 sur versions masquées
hv_l1_sum = F.smooth_l1_loss(hv_pred_masked, hv_target_masked, reduction='sum')
hv_l1 = hv_l1_sum / (mask.sum() * 2)  # Normaliser par nb pixels noyaux
```

**Caractéristiques** :
- **SmoothL1Loss** (Huber) : moins sensible aux outliers que MSE
- **MASQUÉ** (noyaux uniquement)
- Normalisation par nombre de pixels de noyaux

#### 2. Gradient Loss (lignes 244-277)
```python
def gradient_loss(self, pred, target, mask=None):
    """SmoothL1 sur gradients par différences finies."""
    # Gradient horizontal (différences finies)
    pred_h = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_h = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Gradient vertical
    pred_v = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_v = target[:, :, 1:, :] - target[:, :, :-1, :]

    if mask is not None:
        # Masquer gradients
        mask_h = mask[:, :, :, 1:]
        mask_v = mask[:, :, 1:, :]

        grad_loss_h = F.smooth_l1_loss(pred_h * mask_h, target_h * mask_h, reduction='sum')
        grad_loss_v = F.smooth_l1_loss(pred_v * mask_v, target_v * mask_v, reduction='sum')

        grad_loss = (grad_loss_h + grad_loss_v) / (mask_h.sum() + mask_v.sum() + 1e-8)
        return grad_loss
```

**Caractéristiques** :
- **Différences finies** (pas Sobel) pour gradients
- **SmoothL1Loss** au lieu de MSE
- **MASQUÉ** (noyaux uniquement)

#### 3. Loss Totale HV (ligne 319)
```python
hv_loss = hv_l1 + 0.5 * hv_gradient  # Poids 0.5× pour gradient
```

---

## ❌ DIFFÉRENCES CRITIQUES IDENTIFIÉES

### Différence #1 : MSE vs SmoothL1Loss

| Métrique | HoVer-Net (MSE) | Notre Système (SmoothL1) |
|----------|-----------------|--------------------------|
| **Formule** | `(pred - true)²` | `0.5*(pred - true)² si |diff|<1, sinon |diff|-0.5` |
| **Sensibilité outliers** | Haute (pénalité quadratique) | Basse (pénalité linéaire pour |diff|>1) |
| **Convergence** | Plus rapide sur données propres | Plus stable sur données bruitées |

**Impact théorique** :
- ⚠️ SmoothL1Loss peut produire des **gradients plus faibles** que MSE
- ⚠️ Sur données histopathologiques (parfois bruitées), SmoothL1 peut être **trop conservatif**

**Test requis** :
```python
# Comparer sur un batch
batch_mse = F.mse_loss(hv_pred, hv_target)
batch_smooth_l1 = F.smooth_l1_loss(hv_pred, hv_target)
print(f"Ratio: {batch_smooth_l1 / batch_mse:.3f}")
```

### Différence #2 : Sobel 5×5 vs Différences Finies

| Métrique | HoVer-Net (Sobel 5×5) | Notre Système (Différences Finies) |
|----------|----------------------|-----------------------------------|
| **Noyau** | 5×5 avec smoothing | 1×2 (horizontal), 2×1 (vertical) |
| **Effet** | Lisse + dérive | Dérive brute (sensible au bruit) |
| **Détection frontières** | Robuste | Précise mais bruitée |

**Impact théorique** :
- ⚠️ Différences finies sont **plus sensibles au bruit** que Sobel
- ⚠️ Sobel détecte mieux les **frontières nettes** (smoothing intégré)

### Différence #3 : Masquage

| Aspect | HoVer-Net | Notre Système |
|--------|-----------|---------------|
| **MSE/SmoothL1 masqué ?** | ❌ Non (MSE sur tous pixels) | ✅ Oui (masque noyaux) |
| **MSGE/Gradient masqué ?** | ✅ Oui (via `focus`) | ✅ Oui (via `mask`) |

**Impact** :
- ✅ **Avantage nous** : MSE masqué évite que background (70-80% pixels) domine la loss
- ❌ **Bug potentiel HoVer-Net** : MSE non masqué pourrait pousser le modèle vers HV=0 partout

**⚠️ ATTENTION** : Le code HoVer-Net montre MSE **NON masqué** (ligne 101 de utils.py). Cela semble être un **bug** ou une version différente. À vérifier dans leur README/paper.

---

## 📊 TABLEAU RÉCAPITULATIF

| Composant | HoVer-Net Original | Notre Système | Conforme ? |
|-----------|-------------------|---------------|------------|
| **Backbone** | ResNet-50 (25M) | H-optimus-0 (1.1B) | ❌ Différent (mais meilleur) |
| **Pré-entraînement** | ImageNet | 500k+ H&E | ❌ Différent (mais meilleur) |
| **Skip connections** | ✅ Oui | ❌ Non | ❌ |
| **Décodeur** | DenseBlocks | UpsampleBlocks | ❌ Plus simple |
| **NP Loss** | BCE + Dice | BCE + Dice | ✅ Identique |
| **HV Loss (base)** | MSE | SmoothL1Loss | ❌ **DIFFÉRENT** |
| **HV Loss (gradient)** | MSGE (Sobel 5×5) | Gradient Loss (Diff finies) | ❌ **DIFFÉRENT** |
| **HV Masquage** | ❌ MSE non masqué | ✅ SmoothL1 masqué | ✅ Meilleur |
| **NT Loss** | BCE + Dice | CrossEntropy | ⚠️ Similaire |
| **Activation HV** | ❌ Non | ✅ Tanh | ✅ Meilleur (force [-1, 1]) |

---

## 🎯 HYPOTHÈSES SUR AJI 0.0863

### Hypothèse #1 : Données OLD (connectedComponents) ← **PRINCIPALE**

**Preuve Étape 2** : HoVer-Net utilise instances séparées, nous utilisons OLD data fusionnées.

**Impact** : Gradients HV faibles → Watershed ne peut pas séparer.

**Statut** : ✅ **CONFIRMÉ** (Étape 2)

### Hypothèse #2 : SmoothL1Loss Trop Conservatif

**Théorie** : SmoothL1Loss pénalise moins les grandes erreurs → gradients HV plus faibles que MSE.

**Impact** : Même avec données FIXED, gradients HV pourraient être insuffisants.

**Statut** : ⚠️ **À TESTER**

**Test requis** :
```python
# Ré-entraîner UNE famille avec MSE au lieu de SmoothL1Loss
# Comparer HV MSE et AJI
```

### Hypothèse #3 : Sobel vs Différences Finies

**Théorie** : Sobel 5×5 détecte mieux les frontières que différences finies brutes.

**Impact** : Gradient loss moins efficace pour forcer variations spatiales.

**Statut** : ⚠️ **À TESTER**

**Test requis** :
```python
# Implémenter Sobel 5×5 comme HoVer-Net
# Comparer convergence HV MSE
```

### Hypothèse #4 : Skip Connections Manquantes

**Théorie** : Skip connections aident à préserver détails haute résolution.

**Impact** : Décodeur perd informations fines (frontières cellulaires).

**Statut** : ⚠️ **POSSIBLE** (mais backbone 44× plus gros devrait compenser)

---

## 🔬 TESTS RECOMMANDÉS (Par Ordre de Priorité)

### Priorité 1 : Régénérer Données FIXED (Étape 2) ← **CRITIQUE**

**Effort** : 10h calcul
**Gain estimé** : AJI 0.09 → 0.60-0.70

**Justification** : Étape 2 a prouvé que c'est la cause racine.

### Priorité 2 : Tester MSE vs SmoothL1Loss

**Effort** : 2h calcul (1 famille)
**Gain estimé** : Si confirmé, AJI +10-20%

**Méthode** :
```bash
# Entraîner Glandular avec MSE
python scripts/training/train_hovernet_family.py \
    --family glandular \
    --loss_type mse \
    --epochs 50

# Comparer HV MSE et AJI
```

### Priorité 3 : Implémenter Sobel 5×5

**Effort** : 1h dev + 2h calcul
**Gain estimé** : Si confirmé, AJI +5-10%

**Méthode** :
```python
# Modifier gradient_loss() pour utiliser Sobel 5×5
def sobel_gradient_loss(pred, target, mask):
    kernel = get_sobel_kernel_5x5()
    pred_grad = F.conv2d(pred, kernel, padding=2)
    target_grad = F.conv2d(target, kernel, padding=2)
    # ... reste identique
```

---

## ✅ DÉCISION RECOMMANDÉE

### Approche Séquentielle

**Étape A** : Régénérer données FIXED (Priorité 1)
- Utiliser `prepare_family_data_FIXED.py`
- Ré-entraîner les 5 familles
- **Vérifier AJI** → Si > 0.60, problème résolu ✅

**Étape B** : SI AJI < 0.60 après FIXED data
- Tester MSE vs SmoothL1Loss (Priorité 2)
- Implémenter Sobel 5×5 (Priorité 3)

### Justification

1. **Étape 2 a identifié la cause racine** : Données OLD fusionnées
2. **Régénérer FIXED est obligatoire** de toute façon
3. **Tester loss avant régénération = perte de temps** si les données sont le vrai problème

---

## 📝 État du Plan

- [x] **Étape 1** : Vérifier données utilisées → **COMPLÉTÉ**
- [x] **Étape 2** : Comparer preprocessing HoVer-Net → **COMPLÉTÉ**
- [x] **Étape 3** : Comparer architecture/loss → **COMPLÉTÉ** ✅
- [ ] **Étape 4** : Comparer watershed
- [ ] **Étape 5** : Tester modèle officiel

**Prochaine action** : Étape 4 (Watershed) OU décision de régénérer données FIXED
