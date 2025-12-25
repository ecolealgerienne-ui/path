# Vérification Complète du Pipeline

## Analyse Systématique Script par Script

---

## 1. prepare_family_data_FIXED_v12_COHERENT.py

### Entrées
- **Images PanNuke**: `fold{0,1,2}/images.npy` - (N, 256, 256, 3) uint8
- **Masks PanNuke**: `fold{0,1,2}/masks.npy` - (N, 256, 256, 6)
  - Channels 0-4: Instances de noyaux
  - Channel 5: Tissue (EXCLU en v12)

### Sorties
```
data/family_FIXED/{family}_data_FIXED_v12_COHERENT.npz
├── images:     (N, 256, 256, 3) uint8
├── np_targets: (N, 256, 256)    float32 [0, 1]     ← BINAIRE
├── hv_targets: (N, 2, 256, 256) float32 [-1, 1]
├── nt_targets: (N, 256, 256)    int64   [0, 1]     ← BINAIRE (2 classes!)
├── fold_ids:   (N,)             int32
└── image_ids:  (N,)             int32
```

### ⚠️ PROBLÈME CRITIQUE IDENTIFIÉ #1
**NT targets sont BINAIRES (0, 1) mais le décodeur attend 5 classes!**

```python
# v12: compute_nt_target_v12()
nt_target[nuclei_mask] = 1  # Seulement 0 ou 1

# Décodeur: HoVerNetDecoder
self.nt_head = DecoderHead(64, n_classes)  # n_classes=5 par défaut!
```

**Impact**: Le modèle a 5 sorties NT mais seulement 2 classes dans les targets.
- Classes 2, 3, 4 ne sont JAMAIS utilisées
- CrossEntropyLoss essaie de classifier parmi 5 classes avec targets binaires

---

## 2. extract_features_from_v9.py

### Entrées
- Fichier v12: `{family}_data_FIXED_v12_COHERENT.npz`

### Traitement
```python
# 1. Charge images (256, 256, 3)
images = data['images']

# 2. Transforme chaque image
transform = create_hoptimus_transform()
# → ToPILImage → Resize(224) → ToTensor → Normalize

# 3. Extrait features
features = backbone.forward_features(batch)  # (B, 261, 1536)
```

### Sorties
```
data/cache/family_data/
├── {family}_features.npz
│   └── features: (N, 261, 1536) float32
│
└── {family}_targets.npz     ← COPIE DIRECTE SANS MODIFICATION!
    ├── np_targets: (N, 256, 256) float32
    ├── hv_targets: (N, 2, 256, 256) float32
    └── nt_targets: (N, 256, 256) int64
```

### ✅ Vérification OK
- Features extraites avec transform correct
- CLS std validé (0.70-0.90)
- Targets copiés directement (seront resizés dans training)

---

## 3. train_hovernet_family.py

### Entrées
- `{family}_features.npz`: (N, 261, 1536)
- `{family}_targets.npz`: Targets à 256×256

### Traitement dans __getitem__
```python
# 1. Charge features et targets
features = self.features[idx]    # (261, 1536)
np_target = self.np_targets[idx] # (256, 256)
hv_target = self.hv_targets[idx] # (2, 256, 256)
nt_target = self.nt_targets[idx] # (256, 256)

# 2. Resize targets 256 → 224
np_target, hv_target, nt_target = resize_targets(..., target_size=224)

# 3. Convertit en tensors
features = torch.from_numpy(features)           # (261, 1536) float32
np_target = torch.from_numpy(np_target)         # (224, 224) float32
hv_target = torch.from_numpy(hv_target)         # (2, 224, 224) float32
nt_target = torch.from_numpy(nt_target).long()  # (224, 224) int64
```

### Forward Pass
```python
# Décodeur reçoit
features: (B, 261, 1536)

# Décodeur produit
np_out: (B, 2, 224, 224)  ← 2 classes: [background, foreground]
hv_out: (B, 2, 224, 224)  ← 2 canaux: [vertical, horizontal]
nt_out: (B, 5, 224, 224)  ← 5 classes: [bg, neo, inf, con, dead, epi]
```

### ⚠️ PROBLÈME CRITIQUE IDENTIFIÉ #2
**Mismatch entre nt_out (5 classes) et nt_target (2 valeurs)**

```python
# Loss NT
nt_loss = self.bce(nt_pred, nt_target.long())
# nt_pred: (B, 5, H, W) - 5 classes
# nt_target: (B, H, W) - valeurs 0 ou 1 seulement!

# CrossEntropyLoss attend:
# - nt_target contient des indices de classes [0, 4]
# - Mais v12 ne donne que [0, 1]
```

**Impact**: NT Acc à 0.15 (pire que random 0.50) car le modèle est confus.

---

## 4. Résumé des Problèmes

### Problème Principal: INCOMPATIBILITÉ v12 ↔ Décodeur

| Composant | v12 Produit | Décodeur Attend |
|-----------|-------------|-----------------|
| NP | (256, 256) float32 [0, 1] | (2, 224, 224) - OK après resize |
| HV | (2, 256, 256) float32 [-1, 1] | (2, 224, 224) - OK |
| **NT** | **(256, 256) int64 [0, 1]** | **(5, 224, 224) - MISMATCH!** |

### Pourquoi le Training Échoue

1. **NT avec 5 classes vs targets binaires**:
   - Le modèle essaie de classifier en 5 classes
   - Mais les targets ne contiennent que 0 et 1
   - Les classes 2, 3, 4 ne sont jamais activées
   - Le gradient est confus → apprentissage instable

2. **Loss totale dominée par termes HV**:
   ```python
   hv_loss = hv_l1 + 3.0 * hv_gradient + 5.0 * hv_magnitude
   total = 1.0 * np_loss + 2.0 * hv_loss + 1.0 * nt_loss
   # hv_loss peut dominer et noyer le signal NP
   ```

3. **Résultat Training**:
   ```
   NP Dice: 0.3909  ← TERRIBLE (devrait être 0.90+)
   HV MSE:  0.2399  ← Passable
   NT Acc:  0.1518  ← PIRE QUE RANDOM (0.50)
   ```

---

## 5. Solutions Recommandées

### Option A: Modifier le Décodeur pour v12 (Binaire)
```python
# Dans HoVerNetDecoder.__init__
self.nt_head = DecoderHead(64, n_classes=2)  # BINAIRE au lieu de 5
```

### Option B: Modifier v12 pour 5 Classes (Retour à PanNuke Original)
```python
# Dans compute_nt_target()
# Utiliser les vraies classes PanNuke [0-4] au lieu de binaire
```

### Option C: Solution Rapide - Ignorer NT pendant Training
```python
# Dans HoVerNetLoss
lambda_nt = 0.0  # Désactiver la perte NT temporairement
```

---

## 6. Vérification Cohérence Constants

### src/constants.py
```python
CURRENT_DATA_VERSION = "v12_COHERENT"
HOPTIMUS_INPUT_SIZE = 224
PANNUKE_IMAGE_SIZE = 256
```

### TargetFormat (src/data/preprocessing.py)
```python
@dataclass
class TargetFormat:
    nt_max: int = 4  # ⚠️ INCOHÉRENT avec v12 qui produit max=1!
```

---

## 7. Flux de Données Complet

```
┌────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1: prepare_family_data_FIXED_v12_COHERENT.py             │
├────────────────────────────────────────────────────────────────┤
│ PanNuke (256×256)                                              │
│     ↓                                                          │
│ images: (N, 256, 256, 3) uint8                                 │
│ np_targets: (N, 256, 256) float32 [0, 1]                       │
│ hv_targets: (N, 2, 256, 256) float32 [-1, 1]                   │
│ nt_targets: (N, 256, 256) int64 [0, 1] ← BINAIRE!              │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2: extract_features_from_v9.py                           │
├────────────────────────────────────────────────────────────────┤
│ Images 256×256 → Transform → 224×224 → H-optimus-0             │
│     ↓                                                          │
│ features: (N, 261, 1536) ← 1 CLS + 256 patches + 4 registers   │
│ targets: COPIE DIRECTE (toujours 256×256)                      │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ ÉTAPE 3: train_hovernet_family.py                              │
├────────────────────────────────────────────────────────────────┤
│ __getitem__:                                                   │
│   features (261, 1536) → tensor                                │
│   targets (256, 256) → resize_targets(224) → tensors           │
│                                                                │
│ HoVerNetDecoder:                                               │
│   Input: (B, 261, 1536)                                        │
│   Output: np(B,2,224,224), hv(B,2,224,224), nt(B,5,224,224)    │
│                          ↑                                     │
│                  MISMATCH: 5 classes vs 2 valeurs!             │
│                                                                │
│ Loss:                                                          │
│   np_loss = BCE + Dice                                         │
│   hv_loss = MSE + 3×gradient + 5×magnitude                     │
│   nt_loss = CrossEntropy(5 classes, targets [0,1])  ← CONFUS!  │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. Action Immédiate Recommandée

### Modifier n_classes dans le training pour matcher v12:

```python
# train_hovernet_family.py ligne ~XX
hovernet = HoVerNetDecoder(
    embed_dim=1536,
    n_classes=2,  # ← CHANGER 5 → 2 pour matcher v12 binaire
    dropout=args.dropout
).to(device)
```

OU

### Désactiver NT loss temporairement:
```python
criterion = HoVerNetLoss(
    lambda_np=1.0,
    lambda_hv=2.0,
    lambda_nt=0.0,  # ← DÉSACTIVER NT
)
```

---

## 9. Conclusion

**Le problème fondamental est un MISMATCH entre:**
- v12 qui produit des targets NT BINAIRES (0 ou 1)
- Le décodeur HoVer-Net qui attend 5 classes NT (0-4)

Ce mismatch cause:
1. NT Acc de 0.15 (pire que random)
2. Gradient confus qui perturbe aussi NP et HV
3. Training qui ne converge jamais correctement (Dice 0.39)

**La solution est d'aligner n_classes du décodeur avec les targets v12.**
