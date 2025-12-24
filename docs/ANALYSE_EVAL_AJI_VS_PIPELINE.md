# Analyse : eval_aji_from_images.py vs Pipeline d'Entraînement

> **Date:** 2025-12-23
>
> **Objectif:** Comparer point par point le script d'évaluation avec le pipeline documenté pour identifier TOUTES les différences.

---

## Vue d'Ensemble de la Comparaison

| Étape | Pipeline Entraînement | Script Évaluation | Conforme |
|-------|----------------------|-------------------|----------|
| 1. Chargement données | load PanNuke .npy | load_pannuke_data() | ✅ |
| 2. Filtrage famille | ORGAN_TO_FAMILY | filter_by_family() | ✅ |
| 3. GT instances | extract_pannuke_instances() | extract_gt_instances() | ✅ |
| 4. Preprocessing | preprocess_image() | preprocess_image() | ✅ |
| 5. Features H-opt | forward_features() | forward_features() | ✅ |
| 6. Validation CLS | validate_features() | validate_features() | ✅ |
| 7. HoVer-Net | hovernet(features) | hovernet(features) | ✅ |
| 8. Conversion pred | argmax NP/NT | argmax NP | ✅ |
| 9. GT resize | 256→224 nearest | 256→224 nearest | ✅ |
| 10. **Watershed** | ❌ **N'existe pas** | ✅ **watershed_from_hv()** | ⚠️ DIFFÉRENCE |
| 11. **Metrics** | Dice/MSE/Acc | **AJI** | ⚠️ DIFFÉRENCE |

---

## ✅ ÉTAPES CONFORMES (1-9)

### ÉTAPE 1 : Chargement PanNuke (Conforme)

**Pipeline (PIPELINE_COMPLET_DONNEES.md ÉTAPE 0)** :
```python
# Structure attendue
images_path = data_dir / fold_name / "images.npy"
masks_path = data_dir / fold_name / "masks.npy"
types_path = data_dir / fold_name / "types.npy"

images = np.load(images_path)  # (N, 256, 256, 3) uint8
masks = np.load(masks_path)    # (N, 256, 256, 6) int32
types = np.load(types_path)    # (N,) str
```

**Script (lignes 48-74)** :
```python
def load_pannuke_data(data_dir: Path, fold: int):
    fold_name = f"fold{fold}"

    images_path = data_dir / fold_name / "images.npy"
    masks_path = data_dir / fold_name / "masks.npy"
    types_path = data_dir / fold_name / "types.npy"

    images = np.load(images_path)
    masks = np.load(masks_path)
    types = np.load(types_path)

    return images, masks, types
```

**Verdict:** ✅ **IDENTIQUE**

---

### ÉTAPE 2 : Filtrage par Famille (Conforme)

**Pipeline (ÉTAPE 1 Traitement A2)** :
```python
family_organs = [org for org, fam in ORGAN_TO_FAMILY.items() if fam == family]
family_indices = [i for i, organ in enumerate(types) if organ in family_organs]

images_filtered = images[family_indices]
masks_filtered = masks[family_indices]
```

**Script (lignes 77-100)** :
```python
def filter_by_family(images, masks, types, family: str):
    family_organs = [organ for organ, fam in ORGAN_TO_FAMILY.items() if fam == family]
    indices = [i for i, organ in enumerate(types) if organ in family_organs]

    return images[indices], masks[indices], types[indices], indices
```

**Verdict:** ✅ **IDENTIQUE**

---

### ÉTAPE 3 : Extraction GT Instances (Conforme)

**Pipeline (ÉTAPE 1 Traitement A3)** :
```python
def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: IDs natifs
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5: connectedComponents
    epithelial_binary = (mask[:, :, 5] > 0).astype(np.uint8)
    if epithelial_binary.sum() > 0:
        _, epithelial_labels = cv2.connectedComponents(epithelial_binary)
        # ... fusion avec inst_map

    return inst_map
```

**Script (lignes 103-141)** :
```python
def extract_gt_instances(mask: np.ndarray) -> np.ndarray:
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: IDs d'instances natifs PanNuke
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = channel_mask == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 (Epithelial): binaire, utiliser connectedComponents
    epithelial_binary = (mask[:, :, 5] > 0).astype(np.uint8)
    if epithelial_binary.sum() > 0:
        _, epithelial_labels = cv2.connectedComponents(epithelial_binary)
        # ... fusion avec inst_map

    return inst_map
```

**Verdict:** ✅ **IDENTIQUE** (méthode FIXED)

---

### ÉTAPE 4 : Preprocessing Image (Conforme)

**Pipeline (ÉTAPE 2 Traitement B2)** :
```python
# B2a: Conversion uint8
if image.dtype != np.uint8:
    image = image.clip(0, 255).astype(np.uint8)

# B2b: Transform
transform = create_hoptimus_transform()
tensor = transform(image)  # (3, 224, 224)
tensor = tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)
```

**Script (ligne 251)** :
```python
tensor = preprocess_image(image, device=args.device)  # (1, 3, 224, 224)
```

**Détails `preprocess_image()`** (src/preprocessing/__init__.py) :
```python
def preprocess_image(image: np.ndarray, device: str = "cuda") -> torch.Tensor:
    # Conversion uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    # Transform
    transform = create_hoptimus_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    return tensor
```

**Verdict:** ✅ **IDENTIQUE** (utilise fonction centralisée)

---

### ÉTAPE 5 : Features H-optimus-0 (Conforme)

**Pipeline (ÉTAPE 2 Traitement B3)** :
```python
with torch.no_grad():
    features = backbone.forward_features(tensor)  # (1, 261, 1536)
```

**Script (lignes 254-255)** :
```python
with torch.no_grad():
    features = backbone.forward_features(tensor)  # (1, 261, 1536)
```

**Verdict:** ✅ **IDENTIQUE**

---

### ÉTAPE 6 : Validation CLS std (Conforme)

**Pipeline (ÉTAPE 2 Traitement B3 - Validation)** :
```python
cls_token = features[:, 0, :]
cls_std = cls_token.std().item()

assert 0.70 <= cls_std <= 0.90, f"CLS std {cls_std:.3f} hors range!"
```

**Script (lignes 258-262)** :
```python
try:
    validate_features(features)
except ValueError as e:
    print(f"\n⚠️ Warning image {i}: {e}")
    continue
```

**Détails `validate_features()`** (src/preprocessing/__init__.py) :
```python
def validate_features(features: torch.Tensor) -> dict:
    cls_token = features[:, 0, :]
    cls_std = cls_token.std().item()

    if not (0.70 <= cls_std <= 0.90):
        raise ValueError(f"CLS std {cls_std:.3f} hors range [0.70-0.90]")
```

**Verdict:** ✅ **IDENTIQUE** (utilise fonction centralisée)

---

### ÉTAPE 7 : HoVer-Net Prediction (Conforme)

**Pipeline (ÉTAPE 4 Forward Pass)** :
```python
np_out, hv_out, nt_out = hovernet(features)
```

**Script (lignes 265-266)** :
```python
with torch.no_grad():
    np_out, hv_out, nt_out = hovernet(features)
```

**Verdict:** ✅ **IDENTIQUE**

---

### ÉTAPE 8 : Conversion Prédictions (Conforme)

**Pipeline (ÉTAPE 5 Conversion Prédictions)** :
```python
# NP: argmax sur classes [Background, Nuclei]
np_pred = np_out.argmax(dim=1).float()  # (B, 224, 224) [0, 1]

# HV: pas d'activation (déjà [-1, 1])
hv_pred = hv_out  # (B, 2, 224, 224)
```

**Script (lignes 269-272)** :
```python
np_pred_logits = np_out.cpu().numpy()[0]  # (2, 224, 224)
np_pred = (np_pred_logits.argmax(axis=0)).astype(np.float32)  # (224, 224) [0, 1]

hv_pred = hv_out.cpu().numpy()[0]  # (2, 224, 224)
```

**Verdict:** ✅ **IDENTIQUE** (juste conversion torch → numpy)

---

### ÉTAPE 9 : GT Resize 256→224 (Conforme)

**Pipeline (ÉTAPE 3 Traitement C2 - Resize)** :
```python
# Resize NP: nearest (binaire)
np_target = F.interpolate(
    np_target.unsqueeze(0).unsqueeze(0),
    size=(224, 224),
    mode='nearest'
).squeeze()  # (224, 224)
```

**Script (lignes 275-282)** :
```python
# GT instances à 256×256 (méthode FIXED)
inst_gt_256 = extract_gt_instances(mask)

# Resize GT 256 → 224 (EXACTEMENT comme training!)
inst_gt = cv2.resize(inst_gt_256.astype(np.float32), (224, 224),
                   interpolation=cv2.INTER_NEAREST).astype(np.int32)

np_gt = (inst_gt > 0).astype(np.float32)  # GT binaire
```

**Verdict:** ✅ **IDENTIQUE** (resize GT vers 224, interpolation nearest)

---

## ⚠️ DIFFÉRENCES CRITIQUES (Étapes 10-11)

### ❌ ÉTAPE 10 : Watershed (N'existe PAS dans le Training)

**Pipeline (ÉTAPE 4-5)** :
- Le training ne calcule JAMAIS watershed
- Le training calcule seulement :
  - Loss NP (Cross-Entropy + Dice)
  - Loss HV (MSE masqué + gradient loss)
  - Loss NT (Cross-Entropy)
- **Il n'y a PAS de post-processing watershed dans le training**

**Script (lignes 144-191)** :
```python
def watershed_from_hv(np_pred: np.ndarray, hv_pred: np.ndarray) -> np.ndarray:
    """
    Watershed depuis HV maps.

    Args:
        np_pred: (256, 256) binary [0, 1]
        hv_pred: (2, 256, 256) float32 [-1, 1]  ← HV maps PASSÉES mais...

    Returns:
        inst_map: (256, 256) instance IDs
    """
    # Binariser NP
    np_binary = (np_pred > 0.5).astype(np.uint8)

    if np_binary.sum() == 0:
        return np.zeros_like(np_binary, dtype=np.int32)

    # Distance transform pour markers
    dist = ndimage.distance_transform_edt(np_binary)  # ← Utilise SEULEMENT NP

    # Markers = local maxima
    if dist.max() > 0:
        dist_norm = dist / dist.max()
        markers_mask = dist_norm > 0.5
        markers, _ = ndimage.label(markers_mask)

        # ... watershed avec distance transform
        energy = (255 * (1 - dist_norm) * np_binary).astype(np.uint8)
        cv2.watershed(cv2.cvtColor(energy, cv2.COLOR_GRAY2BGR), markers_ws)
```

### 🚨 PROBLÈME IDENTIFIÉ : HV Maps Ignorées !

**Les HV maps sont passées en argument (`hv_pred`) mais JAMAIS utilisées !**

Le watershed utilise SEULEMENT :
1. Distance transform sur NP binaire
2. Markers = local maxima de la distance
3. Energy = distance inversée

**Les HV maps ne sont PAS utilisées pour :**
- Calculer les gradients
- Affiner les frontières
- Guider le watershed

**Conséquence :** Le modèle apprend à prédire des HV maps pour séparer les instances, mais le watershed ne les exploite pas → Les instances ne sont PAS correctement séparées → AJI faible.

---

### ❌ ÉTAPE 11 : Metrics (AJI n'existe PAS dans le Training)

**Pipeline (ÉTAPE 5 Calcul Métriques)** :
```python
# Dice Score
intersection = (np_pred * np_target).sum()
union = np_pred.sum() + np_target.sum()
dice = (2 * intersection) / (union + 1e-8)

# HV MSE
mask = np_target.unsqueeze(1)
hv_mse = F.mse_loss(hv_pred * mask, hv_target * mask)

# NT Accuracy
mask = (np_target > 0.5)
correct = (nt_pred[mask] == nt_target[mask]).sum()
nt_acc = correct / total
```

**Métriques du training** :
- ✅ Dice (NP binaire)
- ✅ HV MSE (gradients)
- ✅ NT Accuracy (classification)
- ❌ **PAS d'AJI** (pas de séparation instances)

**Script (lignes 288-289)** :
```python
aji = compute_aji(inst_pred, inst_gt)
dice = compute_dice(np_pred, np_gt)
```

**Verdict:** ⚠️ **DIFFÉRENCE MAJEURE**

Le training ne calcule JAMAIS AJI car il ne sépare JAMAIS les instances. Il calcule seulement des métriques pixel-wise (Dice, MSE, Accuracy).

Pour calculer AJI, on DOIT faire du watershed pour obtenir des instances séparées. Mais le watershed actuel N'UTILISE PAS les HV maps prédites par le modèle !

---

## 🎯 CAUSE RACINE DU PROBLÈME AJI

### Symptômes

- **AJI sur .npz (eval_aji_from_training_data.py)** : 0.94
- **AJI sur images brutes (eval_aji_from_images.py)** : 0.30

### Cause Racine #1 : Watershed Incomplet

**Problème :** La fonction `watershed_from_hv()` reçoit les HV maps mais ne les utilise PAS.

**Comparaison avec HoVer-Net original (Graham et al. 2019)** :

| Étape | HoVer-Net Original | Script Actuel |
|-------|-------------------|---------------|
| 1. NP binary | ✅ Utilise | ✅ Utilise |
| 2. **HV gradients** | ✅ **Calcule Sobel(HV)** | ❌ **Ignoré** |
| 3. **Energy map** | ✅ **H² + V²** | ❌ **Distance transform** |
| 4. Markers | ✅ Local maxima energy | ✅ Local maxima distance |
| 5. Watershed | ✅ Sur energy map | ⚠️ Sur distance map |

**Code HoVer-Net original (simplifié)** :
```python
def watershed_hovernet(np_pred, hv_pred):
    # 1. Binariser NP
    np_binary = (np_pred > 0.5)

    # 2. Calculer gradients HV (Sobel)
    h_sobel_h = sobel(hv_pred[0, :, :], axis=1)  # Gradient H horizontal
    h_sobel_v = sobel(hv_pred[0, :, :], axis=0)  # Gradient H vertical
    v_sobel_h = sobel(hv_pred[1, :, :], axis=1)  # Gradient V horizontal
    v_sobel_v = sobel(hv_pred[1, :, :], axis=0)  # Gradient V vertical

    # 3. Energy map = magnitude des gradients
    energy = np.sqrt(h_sobel_h**2 + h_sobel_v**2 + v_sobel_h**2 + v_sobel_v**2)

    # 4. Markers = local maxima de (1 - energy) = centres des noyaux
    # Les centres ont des gradients faibles (HV ~ 0)
    # Les frontières ont des gradients forts (HV change rapidement)
    dist = 1 - energy
    markers = find_local_maxima(dist)

    # 5. Watershed avec energy comme "coût"
    inst_map = watershed(-energy, markers, mask=np_binary)

    return inst_map
```

**Code actuel (INCORRECT)** :
```python
def watershed_from_hv(np_pred, hv_pred):  # hv_pred passé mais JAMAIS utilisé!
    np_binary = (np_pred > 0.5)

    # ❌ Distance transform au lieu de HV gradients
    dist = ndimage.distance_transform_edt(np_binary)

    # ❌ Markers basés sur distance, pas sur HV
    dist_norm = dist / dist.max()
    markers_mask = dist_norm > 0.5
    markers, _ = ndimage.label(markers_mask)

    # ❌ Energy = distance inversée, pas gradients HV
    energy = (255 * (1 - dist_norm) * np_binary).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(energy, cv2.COLOR_GRAY2BGR), markers_ws)
```

### Cause Racine #2 : AJI vs .npz (Faux Négatif)

**Pourquoi AJI 0.94 sur .npz ?**

Le script `eval_aji_from_training_data.py` utilisait `connectedComponents` sur le **NP binaire** pour reconstruire les instances GT.

```python
# eval_aji_from_training_data.py (ligne 92-97)
def extract_gt_instances(np_target: np.ndarray, nt_target: np.ndarray):
    np_binary = (np_target > 0.5).astype(np.uint8)
    _, inst_map = cv2.connectedComponents(np_binary)  # ← FUSIONNE cellules qui se touchent
    return inst_map
```

**Résultat :** GT avait 0.5 instances par image au lieu de 13.3 vraies instances.

Le modèle prédisait AUSSI des instances fusionnées (car watershed basé sur distance, pas HV).

**Donc AJI 0.94 comparait "mauvaises instances" vs "mauvaises instances" → score artificiellement élevé !**

---

## 📊 TABLEAU RÉCAPITULATIF

| Étape | Entraînement | Évaluation | Conforme | Impact |
|-------|--------------|------------|----------|--------|
| Chargement PanNuke | ✅ | ✅ | ✅ | Aucun |
| Filtrage famille | ✅ | ✅ | ✅ | Aucun |
| **GT instances** | extract_pannuke_instances() | extract_gt_instances() | ✅ | Aucun |
| **Preprocessing** | preprocess_image() | preprocess_image() | ✅ | Aucun |
| **Features** | forward_features() | forward_features() | ✅ | Aucun |
| **Validation CLS** | validate_features() | validate_features() | ✅ | Aucun |
| **HoVer-Net** | hovernet(features) | hovernet(features) | ✅ | Aucun |
| **Conversion pred** | argmax NP/NT | argmax NP | ✅ | Aucun |
| **GT resize** | 256→224 nearest | 256→224 nearest | ✅ | Aucun |
| **Watershed** | ❌ N'existe pas | ✅ watershed_from_hv() | ❌ | **CRITIQUE** |
| **Metrics** | Dice/MSE/Acc | AJI | ❌ | **CRITIQUE** |

---

## 🔧 SOLUTION PROPOSÉE

### Fix #1 : Implémenter Watershed HoVer-Net Correct

**Remplacer `watershed_from_hv()` par :**

```python
from scipy.ndimage import sobel

def watershed_hovernet_correct(np_pred: np.ndarray, hv_pred: np.ndarray) -> np.ndarray:
    """
    Watershed selon méthode HoVer-Net (Graham et al. 2019).

    Args:
        np_pred: (H, W) binary [0, 1]
        hv_pred: (2, H, W) float32 [-1, 1]

    Returns:
        inst_map: (H, W) instance IDs
    """
    np_binary = (np_pred > 0.5).astype(np.uint8)

    if np_binary.sum() == 0:
        return np.zeros_like(np_binary, dtype=np.int32)

    # 1. Calculer gradients HV (Sobel)
    h_map = hv_pred[0]  # (H, W)
    v_map = hv_pred[1]  # (H, W)

    h_sobel_h = sobel(h_map, axis=1)  # Gradient H en x
    h_sobel_v = sobel(h_map, axis=0)  # Gradient H en y
    v_sobel_h = sobel(v_map, axis=1)  # Gradient V en x
    v_sobel_v = sobel(v_map, axis=0)  # Gradient V en y

    # 2. Energy map = magnitude des gradients
    energy = np.sqrt(h_sobel_h**2 + h_sobel_v**2 + v_sobel_h**2 + v_sobel_v**2)

    # 3. Normaliser
    if energy.max() > 0:
        energy_norm = energy / energy.max()
    else:
        energy_norm = energy

    # 4. Markers = local maxima de (1 - energy)
    # Les centres des noyaux ont energy faible (gradients HV ~ 0)
    dist_like = 1 - energy_norm
    dist_like = dist_like * np_binary  # Masquer background

    # Local maxima
    from scipy.ndimage import maximum_filter
    local_max = (dist_like == maximum_filter(dist_like, size=3))
    local_max = local_max & (dist_like > 0.5)  # Seuil

    markers, _ = ndimage.label(local_max)

    if markers.max() == 0:
        # Fallback: distance transform
        dist = ndimage.distance_transform_edt(np_binary)
        local_max = (dist == maximum_filter(dist, size=5)) & (dist > 2)
        markers, _ = ndimage.label(local_max)

    if markers.max() == 0:
        # Dernier recours
        _, inst_map = cv2.connectedComponents(np_binary)
        return inst_map.astype(np.int32)

    # 5. Watershed avec -energy (minima = frontières)
    markers_ws = markers.astype(np.int32)

    # OpenCV watershed attend une image 3 canaux + int32 markers
    energy_uint8 = (255 * energy_norm).astype(np.uint8)
    energy_bgr = cv2.cvtColor(energy_uint8, cv2.COLOR_GRAY2BGR)

    cv2.watershed(energy_bgr, markers_ws)

    # Clean up
    inst_map = markers_ws.copy()
    inst_map[inst_map == -1] = 0  # Frontières watershed
    inst_map[np_binary == 0] = 0  # Background

    return inst_map
```

### Fix #2 : Valider avec GT PanNuke

**Script de test :**
```bash
python scripts/evaluation/eval_aji_from_images.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --data_dir /home/amar/data/PanNuke \
    --fold 2 \
    --n_samples 20
```

**Résultat attendu :**
- AJI devrait passer de 0.30 à >0.60 (proche du 0.94 mais avec vraies instances GT)

---

## 📝 CONCLUSION

### Points Positifs ✅

**Les étapes 1-9 du pipeline sont IDENTIQUES** entre entraînement et évaluation :
1. Chargement PanNuke
2. Filtrage famille
3. GT instances (méthode FIXED)
4. Preprocessing (preprocess_image centralisé)
5. Features H-optimus-0 (forward_features)
6. Validation CLS std
7. HoVer-Net prediction
8. Conversion prédictions
9. GT resize 256→224

**Le preprocessing et l'inférence sont corrects !**

### Problème Identifié ❌

**Le watershed n'utilise PAS les HV maps prédites !**

- HoVer-Net apprend à prédire des gradients HV pour séparer les instances
- Le watershed actuel utilise SEULEMENT distance transform (ignore HV)
- Résultat : Les instances ne sont PAS correctement séparées
- AJI faible (0.30) car instances fusionnées

### Action Prioritaire 🎯

**Implémenter le watershed HoVer-Net correct qui utilise les HV gradients.**

Cela devrait résoudre le problème AJI sans avoir à ré-entraîner le modèle.

**Le modèle est BON, c'est le post-processing qui est incorrect !**
