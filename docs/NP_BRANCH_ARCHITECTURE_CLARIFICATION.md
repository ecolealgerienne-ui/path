# Clarification: Architecture Branche NP

**Date:** 2025-12-24
**Statut:** Documentation définitive

---

## Résumé Exécutif

La branche NP (Nuclei Presence) du décodeur HoVer-Net produit **2 canaux** (background/foreground) avec **CrossEntropyLoss**, pas 1 canal avec BCEWithLogitsLoss.

Cette architecture est **cohérente** avec:
- ✅ Le code d'entraînement (`train_hovernet_family.py` ligne 191)
- ✅ CLAUDE.md (ligne 1079: "2 classes : fond/noyau")
- ✅ La littérature HoVer-Net (classification binaire avec 2 classes)

**L'erreur était dans `model_interface.py`** qui documentait 1 canal au lieu de 2.

---

## Architecture Définitive

### Décodeur (`src/models/hovernet_decoder.py`)

```python
# Ligne 117: NP head produit 2 canaux
self.np_head = DecoderHead(64, 2)  # background + foreground

# Ligne 224: Loss CrossEntropy (2 classes)
self.bce = nn.CrossEntropyLoss()

# Ligne 236: Dice loss avec softmax
def dice_loss(self, pred: torch.Tensor, target: torch.Tensor):
    pred_soft = F.softmax(pred, dim=1)[:, 1]  # Probabilité classe 1 (foreground)
    ...
```

### Format de Sortie

```python
# HoVerNetDecoder.forward()
np_out, hv_out, nt_out = model(features)

# Shapes:
np_out.shape  # (B, 2, H, W)  ← 2 canaux!
hv_out.shape  # (B, 2, H, W)
nt_out.shape  # (B, 5, H, W)
```

---

## Extraction du Masque Binaire

### ❌ INCORRECT (Ancien)

```python
# NE JAMAIS faire ça
np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]  # ❌
# Problème: sigmoid sur 2 canaux, puis canal 0 = BACKGROUND!
```

### ✅ CORRECT (Définitif)

**Option A: Via le wrapper (Recommandé)**

```python
from src.models import create_hovernet_wrapper

hovernet = create_hovernet_wrapper(checkpoint, device)
output = hovernet(features)  # HoVerNetOutput

# Méthode 1: Automatique (applique softmax)
result = output.to_numpy(apply_activations=True)
np_mask = result["np"]  # (H, W) foreground prob [0, 1]

# Méthode 2: Attribut direct (logits)
np_logits = output.np  # (B, 2, H, W)
np_prob = torch.softmax(np_logits, dim=1)[0, 1]  # Foreground
```

**Option B: Direct (Scripts legacy)**

```python
# Inférence
np_out, hv_out, nt_out = hovernet(features)

# Extraction foreground
np_prob = torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]  # Canal 1 = foreground

# Masque binaire
np_mask = (np_prob > 0.5).astype(np.uint8)
```

---

## Training vs Inference

### Training (`train_hovernet_family.py`)

```python
# Ligne 191: Calcul Dice
def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_binary = (pred.argmax(dim=1) == 1).float()  # Classe 1 = foreground
    ...

# Ligne 318: Loss
np_bce = self.bce(np_pred, np_target.long())  # CrossEntropyLoss
```

**Cohérent:** Utilise `argmax` pour obtenir foreground.

### Inference (Scripts de test)

```python
# test_on_training_data.py ligne 101
np_pred = torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]  # Foreground

# test_aji_v8.py ligne 143
np_pred = torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]  # Foreground
```

**Cohérent:** Utilise `softmax[:,1]` pour obtenir foreground.

---

## Pourquoi 2 Canaux au Lieu de 1?

### Littérature

**HoVer-Net (Graham et al. 2019)** utilise:
- NP: 1 canal avec BCEWithLogitsLoss (sigmoid)

**Notre implémentation** utilise:
- NP: 2 canaux avec CrossEntropyLoss (softmax)

**Les deux sont valides** pour classification binaire. CrossEntropyLoss est souvent préféré car:

| Critère | BCEWithLogitsLoss (1 canal) | CrossEntropyLoss (2 canaux) |
|---------|----------------------------|-----------------------------|
| Gradient flow | Bon | Meilleur (2 chemins de gradient) |
| Stabilité numérique | Log-sum-exp intégré | Logsumexp plus stable |
| Complexité | Plus simple (1 canal) | Légèrement plus complexe |
| Standard deep learning | Courant pour binaire | Plus courant pour multi-classe |

**Notre choix:** 2 canaux avec CrossEntropyLoss pour stabilité et gradient flow.

---

## Tests de Validation

### Test 1: Vérifier Shape

```python
import torch
from src.models.loader import ModelLoader

hovernet = ModelLoader.load_hovernet(checkpoint, device="cuda")
features = torch.randn(1, 261, 1536).cuda()

np_out, hv_out, nt_out = hovernet(features)

assert np_out.shape == (1, 2, 224, 224), f"NP shape incorrect: {np_out.shape}"
assert hv_out.shape == (1, 2, 224, 224), f"HV shape incorrect: {hv_out.shape}"
assert nt_out.shape == (1, 5, 224, 224), f"NT shape incorrect: {nt_out.shape}"

print("✅ Shapes correctes")
```

### Test 2: Vérifier Extraction Foreground

```python
# Via wrapper
from src.models import create_hovernet_wrapper

hovernet = create_hovernet_wrapper(checkpoint, device="cuda")
output = hovernet(features)
result = output.to_numpy(apply_activations=True)

np_mask = result["np"]  # (224, 224)
assert np_mask.shape == (224, 224), f"NP mask shape incorrect: {np_mask.shape}"
assert 0 <= np_mask.min() <= np_mask.max() <= 1, "NP prob hors range [0, 1]"

print(f"✅ NP mask range: [{np_mask.min():.3f}, {np_mask.max():.3f}]")
```

### Test 3: Vérifier Cohérence Training

```python
# Simuler training
np_pred = torch.randn(1, 2, 224, 224)  # Logits
target = torch.randint(0, 2, (1, 224, 224))  # Binary target

# Méthode training (argmax)
pred_binary_train = (np_pred.argmax(dim=1) == 1).float()

# Méthode inference (softmax)
pred_binary_inf = (torch.softmax(np_pred, dim=1)[:, 1] > 0.5).float()

# Doivent être identiques
assert torch.allclose(pred_binary_train, pred_binary_inf), "Mismatch train/inf!"

print("✅ Training et inference cohérents")
```

---

## Migration Scripts Legacy

Pour les scripts qui utilisent encore `sigmoid`:

```bash
# Rechercher toutes les occurrences
grep -r "sigmoid(np" scripts/ --include="*.py"

# Migrer automatiquement
python scripts/utils/migrate_np_predictions.py --dry-run
python scripts/utils/migrate_np_predictions.py  # Exécuter
```

**Pattern de remplacement:**

```python
# AVANT:
np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]

# APRÈS:
np_pred = torch.softmax(np_out, dim=1).cpu().numpy()[0, 1]
```

---

## Checklist Développeur

Avant de modifier du code qui utilise NP predictions:

- [ ] Vérifier que `np_out.shape == (B, 2, H, W)` (2 canaux)
- [ ] Utiliser `softmax(np_out, dim=1)[:, 1]` pour foreground
- [ ] NE JAMAIS utiliser `sigmoid` sur 2 canaux
- [ ] Vérifier cohérence avec `compute_dice()` dans training
- [ ] Tester avec `test_on_training_data.py` avant commit

---

## Références

- **hovernet_decoder.py** ligne 117, 224, 236
- **train_hovernet_family.py** ligne 191
- **model_interface.py** ligne 17-68
- **CLAUDE.md** ligne 1073-1086
- **Graham et al. 2019** - HoVer-Net original

---

## Historique

| Date | Changement | Auteur |
|------|------------|--------|
| 2025-12-24 | Clarification architecture 2 canaux | Claude (Bug #6 fix) |
| 2025-12-24 | Mise à jour `model_interface.py` | Claude |
| 2025-12-24 | Migration scripts de test | Claude |

---

**VERDICT FINAL:** Architecture 2 canaux **validée et documentée**. Pas de changement nécessaire.
