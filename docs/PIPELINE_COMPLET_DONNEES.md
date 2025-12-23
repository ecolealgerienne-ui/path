# Pipeline Complet des Données — CellViT-Optimus

> **OBJECTIF** : Documenter CHAQUE étape de transformation des données depuis PanNuke brut jusqu'aux prédictions finales.
>
> **IMPORTANCE CRITIQUE** : Toute différence entre ce pipeline (entraînement) et le pipeline d'évaluation cause des erreurs de prédiction.

---

## Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ENTRAÎNEMENT                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DONNÉES BRUTES PANNUKE                                                 │
│         ↓                                                               │
│  ÉTAPE 1: prepare_family_data_FIXED.py                                  │
│         ↓                                                               │
│  FICHIERS .npz (images + targets)                                       │
│         ↓                                                               │
│  ÉTAPE 2: extract_features_from_fixed.py                                │
│         ↓                                                               │
│  FICHIERS .npz (features + targets)                                     │
│         ↓                                                               │
│  ÉTAPE 3: DataLoader (train_hovernet_family.py)                         │
│         ↓                                                               │
│  BATCH TENSORS (GPU)                                                    │
│         ↓                                                               │
│  ÉTAPE 4: HoVer-Net Training                                            │
│         ↓                                                               │
│  MODÈLE ENTRAÎNÉ (.pth)                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ÉTAPE 0 : Données Brutes PanNuke

### Structure sur Disque

```
/home/amar/data/PanNuke/
├── fold0/
│   ├── images.npy          # (2656, 256, 256, 3)
│   ├── masks.npy           # (2656, 256, 256, 6)
│   └── types.npy           # (2656,)
├── fold1/
│   ├── images.npy          # (2523, 256, 256, 3)
│   ├── masks.npy           # (2523, 256, 256, 6)
│   └── types.npy           # (2523,)
└── fold2/
    ├── images.npy          # (2722, 256, 256, 3)
    ├── masks.npy           # (2722, 256, 256, 6)
    └── types.npy           # (2722,)
```

### Format images.npy

| Attribut | Valeur |
|----------|--------|
| **Shape** | `(N, 256, 256, 3)` |
| **Dtype** | `uint8` |
| **Range** | `[0, 255]` |
| **Colorspace** | RGB (H&E staining) |
| **Channels** | R=Hematoxylin, G=Mixed, B=Eosin |

**Exemple valeur pixel** :
```python
image[0, 128, 128, :] = [234, 198, 217]  # Rose (tissue)
```

### Format masks.npy

| Attribut | Valeur |
|----------|--------|
| **Shape** | `(N, 256, 256, 6)` |
| **Dtype** | `int32` |
| **Range** | `[0, max_instance_id]` |
| **Channels** | 6 canaux (voir ci-dessous) |

**Structure des 6 canaux** :

| Canal | Type Cellulaire | Format | Exemple Valeurs |
|-------|----------------|--------|-----------------|
| 0 | Background | Binaire (0/1) | `[0, 0, 0, 1, 1, ...]` |
| 1 | Neoplastic | Instance IDs | `[0, 0, 88, 88, 96, 96, ...]` |
| 2 | Inflammatory | Instance IDs | `[0, 15, 15, 0, 22, ...]` |
| 3 | Connective | Instance IDs | `[0, 0, 5, 5, 0, ...]` |
| 4 | Dead | Instance IDs | `[0, 0, 0, 0, 0, ...]` |
| 5 | Epithelial | Binaire (0/1) | `[0, 0, 1, 1, 0, ...]` |

**IMPORTANT** :
- Canaux 1-4 : Chaque pixel a un ID d'instance unique (0 = background)
- Canal 5 : Binaire seulement (0/1), pas d'IDs d'instances
- Les IDs d'instances ne sont PAS consécutifs (ex: `[88, 96, 107]` pas `[1, 2, 3]`)

### Format types.npy

| Attribut | Valeur |
|----------|--------|
| **Shape** | `(N,)` |
| **Dtype** | `str` (Unicode) |
| **Values** | `['Breast', 'Colon', 'Prostate', ...]` |

**Liste complète des organes** (19) :
```python
PANNUKE_ORGANS = [
    'Adrenal_gland', 'Bile-duct', 'Bladder', 'Breast', 'Cervix',
    'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'Liver',
    'Lung', 'Ovarian', 'Pancreatic', 'Prostate', 'Skin',
    'Stomach', 'Testis', 'Thyroid', 'Uterus'
]
```

---

## ÉTAPE 1 : prepare_family_data_FIXED.py

### Script
`scripts/preprocessing/prepare_family_data_FIXED.py`

### Fonction Principale
`extract_pannuke_instances()` — Ligne 79-134

### Traitement A1 : Chargement des Données

**Code** (lignes 146-151) :
```python
images = np.load(images_path)        # (N, 256, 256, 3) uint8
masks = np.load(masks_path)          # (N, 256, 256, 6) int32
types = np.load(types_path)          # (N,) str
```

**Aucune transformation** — Lecture directe depuis disque.

### Traitement A2 : Filtrage par Famille

**Code** (lignes 153-166) :
```python
ORGAN_TO_FAMILY = {
    'Breast': 'glandular',
    'Prostate': 'glandular',
    # ... etc
}

family_organs = [org for org, fam in ORGAN_TO_FAMILY.items() if fam == family]
family_indices = [i for i, organ in enumerate(types) if organ in family_organs]

# Filtrer
images_filtered = images[family_indices]
masks_filtered = masks[family_indices]
types_filtered = types[family_indices]
```

**Transformation** :
- Input : `(N, 256, 256, 3/6)` — Tous les organes
- Output : `(M, 256, 256, 3/6)` — Seulement organes de la famille (`M < N`)

### Traitement A3 : Extraction Instances PanNuke

**Code** (lignes 79-134) :
```python
def extract_pannuke_instances(mask: np.ndarray) -> np.ndarray:
    """
    Extrait les VRAIES instances de PanNuke.

    Input:  mask (256, 256, 6) — 6 canaux avec IDs d'instances
    Output: inst_map (256, 256) — Carte d'instances unique
    """
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4 : Instance IDs natifs PanNuke
    for c in range(1, 5):  # Neoplastic, Inflammatory, Connective, Dead
        channel_mask = mask[:, :, c]
        inst_ids = np.unique(channel_mask)
        inst_ids = inst_ids[inst_ids > 0]  # Exclure background

        for inst_id in inst_ids:
            inst_mask = (channel_mask == inst_id)
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # Canal 5 : Epithelial (binaire) → connectedComponents
    epithelial_mask = mask[:, :, 5]
    if epithelial_mask.max() > 0:
        epithelial_binary = (epithelial_mask > 0).astype(np.uint8)
        num_labels, epithelial_labels = cv2.connectedComponents(epithelial_binary)

        for label_id in range(1, num_labels):
            inst_mask = (epithelial_labels == label_id)
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    return inst_map
```

**Transformation** :
- Input : `mask (256, 256, 6)` — 6 canaux avec IDs
- Output : `inst_map (256, 256)` — Carte d'instances unique
- Dtype : `int32`
- Range : `[0, num_instances]` (0 = background)

**Exemple** :
```
Input mask[:, :, 1] (Neoplastic):
[0, 0, 88, 88, 0]
[0, 96, 96, 0, 0]
[107, 107, 0, 0, 0]

Output inst_map:
[0, 0, 1, 1, 0]
[0, 2, 2, 0, 0]
[3, 3, 0, 0, 0]
```

### Traitement A4 : Calcul Target NP (Nuclear Presence)

**Code** (ligne 237) :
```python
np_target = (inst_map > 0).astype(np.float32)
```

**Transformation** :
- Input : `inst_map (256, 256)` — int32, range [0, N]
- Output : `np_target (256, 256)` — float32, range [0.0, 1.0]
- Logique : 0 si background, 1.0 si noyau

**Exemple** :
```
Input inst_map:
[0, 0, 1, 1, 0]
[0, 2, 2, 0, 0]

Output np_target:
[0.0, 0.0, 1.0, 1.0, 0.0]
[0.0, 1.0, 1.0, 0.0, 0.0]
```

### Traitement A5 : Calcul Target HV (Horizontal-Vertical)

**Code** (lignes 43-77) :
```python
def compute_hv_maps(inst_map: np.ndarray) -> np.ndarray:
    """
    Calcule les cartes de gradients HV.

    Input:  inst_map (256, 256) — Instance IDs
    Output: hv_map (2, 256, 256) — Gradients H et V
    """
    h_map = np.zeros((256, 256), dtype=np.float32)
    v_map = np.zeros((256, 256), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = (inst_map == inst_id)

        # Centroid
        y_coords, x_coords = np.where(inst_mask)
        centroid_y = y_coords.mean()
        centroid_x = x_coords.mean()

        # Taille du noyau
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        height = y_max - y_min + 1
        width = x_max - x_min + 1

        # Normalisation [-1, 1]
        for i in range(len(y_coords)):
            y, x = y_coords[i], x_coords[i]

            # Horizontal : distance normalisée au centroid en X
            if width > 1:
                h_map[y, x] = (x - centroid_x) / (width / 2.0)
            else:
                h_map[y, x] = 0.0

            # Vertical : distance normalisée au centroid en Y
            if height > 1:
                v_map[y, x] = (y - centroid_y) / (height / 2.0)
            else:
                v_map[y, x] = 0.0

    # Stack [H, V]
    hv_map = np.stack([h_map, v_map], axis=0)  # (2, 256, 256)

    # Clamp [-1, 1]
    hv_map = np.clip(hv_map, -1.0, 1.0)

    return hv_map
```

**Transformation** :
- Input : `inst_map (256, 256)` — int32, instance IDs
- Output : `hv_map (2, 256, 256)` — float32, range [-1.0, 1.0]
- Channel 0 : Horizontal gradient
- Channel 1 : Vertical gradient

**Exemple** (noyau centré en (2, 2), taille 3×3) :
```
Input inst_map:
[0, 0, 0, 0, 0]
[0, 1, 1, 1, 0]
[0, 1, 1, 1, 0]
[0, 1, 1, 1, 0]
[0, 0, 0, 0, 0]

Output hv_map[0, :, :] (H):
[0.0,  0.0,  0.0,  0.0, 0.0]
[0.0, -1.0,  0.0,  1.0, 0.0]
[0.0, -1.0,  0.0,  1.0, 0.0]
[0.0, -1.0,  0.0,  1.0, 0.0]
[0.0,  0.0,  0.0,  0.0, 0.0]

Output hv_map[1, :, :] (V):
[0.0,  0.0,  0.0,  0.0, 0.0]
[0.0, -1.0, -1.0, -1.0, 0.0]
[0.0,  0.0,  0.0,  0.0, 0.0]
[0.0,  1.0,  1.0,  1.0, 0.0]
[0.0,  0.0,  0.0,  0.0, 0.0]
```

### Traitement A6 : Calcul Target NT (Nuclear Type)

**Code** (lignes 239-253) :
```python
nt_target = np.zeros((256, 256), dtype=np.int64)

PANNUKE_TO_NT_CLASS = {
    1: 1,  # Neoplastic    → Class 1
    2: 2,  # Inflammatory  → Class 2
    3: 3,  # Connective    → Class 3
    4: 4,  # Dead          → Class 4
    5: 5,  # Epithelial    → Class 5 (mais on utilise 0-4)
}

# Note: Dans le code réel, on mappe 1-5 → 0-4
for c in range(1, 6):
    channel_mask = mask[:, :, c]
    nt_target[channel_mask > 0] = c - 1  # 1→0, 2→1, 3→2, 4→3, 5→4
```

**Transformation** :
- Input : `mask (256, 256, 6)` — 6 canaux
- Output : `nt_target (256, 256)` — int64, range [0, 4]
- Mapping : Neoplastic=0, Inflammatory=1, Connective=2, Dead=3, Epithelial=4

**ATTENTION** : Background reste à 0, donc on a collision. Vérifions le code réel...

**Code réel** (lignes 239-253) :
```python
nt_target = np.zeros((256, 256), dtype=np.int64)

# Priorité : dernier canal gagne
for c in range(1, 6):
    channel_mask = mask[:, :, c]
    class_id = c  # 1, 2, 3, 4, 5
    nt_target[channel_mask > 0] = class_id

# Background = 0, Neoplastic = 1, ..., Epithelial = 5
```

**Donc le mapping final** :
- 0 : Background
- 1 : Neoplastic
- 2 : Inflammatory
- 3 : Connective
- 4 : Dead
- 5 : Epithelial

**Mais HoVer-Net utilise 5 classes (pas 6)**, donc il faut vérifier...

En fait, regardons le code d'entraînement `train_hovernet_family.py` ligne 172 :
```python
nt_target_t = nt_target_t.long()  # Déjà [0-5]
```

Et le loss (ligne 362) :
```python
nt_loss = F.cross_entropy(nt_out, nt_target)
```

Où `nt_out` a shape `(B, n_classes, H, W)` avec `n_classes=5`.

**Code réel** (ligne 239) :
```python
nt_target = np.argmax(mask[:, :, 1:], axis=-1).astype(np.int64)
```

**Explication** :
- `mask[:, :, 1:]` → Canaux 1-5 (exclut canal 0 = background)
- Shape : `(256, 256, 5)` pour [Neoplastic, Inflammatory, Connective, Dead, Epithelial]
- `np.argmax(axis=-1)` → Retourne l'indice [0, 1, 2, 3, 4] du canal dominant

**Mapping final** :
- argmax=0 → Canal 1 dominant → **Neoplastic**
- argmax=1 → Canal 2 dominant → **Inflammatory**
- argmax=2 → Canal 3 dominant → **Connective**
- argmax=3 → Canal 4 dominant → **Dead**
- argmax=4 → Canal 5 dominant → **Epithelial**

**Transformation** :
- Input : `mask (256, 256, 6)` — 6 canaux avec IDs
- Output : `nt_target (256, 256)` — int64, range [0, 4]
- Dtype : `int64`

**Cas particulier (Background)** :
- Si tous les canaux 1-5 sont à 0 (pixel background)
- `argmax` retourne 0 (Neoplastic par défaut)
- **Ce n'est pas un problème** car NP=0 pour ces pixels
- En post-processing, on ignore NT pour les pixels où NP=0

**Exemple** :
```
Input mask[:, :, 1:] (5 canaux):
Canal 1 (Neo):  [88, 0, 0, 0, 0]
Canal 2 (Infl): [0, 15, 0, 0, 0]
Canal 3 (Conn): [0, 0, 5, 0, 0]
Canal 4 (Dead): [0, 0, 0, 0, 0]
Canal 5 (Epit): [0, 0, 0, 1, 0]

Output nt_target:
[0, 1, 2, 4, 0]
 ↑  ↑  ↑  ↑  ↑
Neo Infl Conn Epit Neo(bg)
```

### Traitement A7 : Sauvegarde .npz

**Code** (lignes 271-279) :
```python
np.savez_compressed(
    output_file,
    images=images_array,            # (M, 256, 256, 3) uint8
    np_targets=np_targets_array,    # (M, 256, 256) float32 [0.0, 1.0]
    hv_targets=hv_targets_array,    # (M, 2, 256, 256) float32 [-1.0, 1.0]
    nt_targets=nt_targets_array,    # (M, 256, 256) int64 [0, 4]
    fold_ids=fold_ids_array,        # (M,) int32
    image_ids=image_ids_array,      # (M,) int32
)
```

**Fichier de sortie** :
```
data/family_FIXED/{family}_data_FIXED.npz
```

**Structure finale** :

| Clé | Shape | Dtype | Range | Signification |
|-----|-------|-------|-------|---------------|
| `images` | (M, 256, 256, 3) | uint8 | [0, 255] | Images RGB |
| `np_targets` | (M, 256, 256) | float32 | [0.0, 1.0] | Masques binaires noyaux |
| `hv_targets` | (M, 2, 256, 256) | float32 | [-1.0, 1.0] | Gradients H/V |
| `nt_targets` | (M, 256, 256) | int64 | [0, 4] | Types cellulaires |
| `fold_ids` | (M,) | int32 | [0, 2] | Fold PanNuke d'origine |
| `image_ids` | (M,) | int32 | [0, N] | Index dans le fold |

**Exemple pour famille Epidermal** :
- Fichier : `data/family_FIXED/epidermal_data_FIXED.npz`
- Taille : ~0.10 GB
- M = 571 images (Skin + HeadNeck)

---

## ÉTAPE 2 : extract_features_from_fixed.py

### Script
`scripts/preprocessing/extract_features_from_fixed.py`

### Traitement B1 : Chargement .npz

**Code** (lignes 60-65) :
```python
data = np.load(data_file)

images = data['images']              # (M, 256, 256, 3) uint8
np_targets = data['np_targets']      # (M, 256, 256) float32
hv_targets = data['hv_targets']      # (M, 2, 256, 256) float32
nt_targets = data['nt_targets']      # (M, 256, 256) int64
fold_ids = data['fold_ids']          # (M,) int32
image_ids = data['image_ids']        # (M,) int32
```

**Aucune transformation** — Lecture directe.

### Traitement B2 : Preprocessing Image (CRITIQUE)

**Code** (lignes 85-95) :
```python
# ÉTAPE B2a: Conversion uint8 (obligatoire pour ToPILImage)
if image.dtype != np.uint8:
    if image.max() <= 1.0:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        image = image.clip(0, 255).astype(np.uint8)

# ÉTAPE B2b: Transform torchvision
transform = create_hoptimus_transform()
tensor = transform(image)  # (3, 224, 224)
tensor = tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)
```

**Détails `create_hoptimus_transform()`** (src/preprocessing/__init__.py lignes 17-29) :
```python
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

def create_hoptimus_transform():
    return transforms.Compose([
        transforms.ToPILImage(),           # numpy → PIL Image
        transforms.Resize((224, 224)),     # 256×256 → 224×224
        transforms.ToTensor(),             # PIL → tensor [0, 1]
        transforms.Normalize(              # Normalisation H-optimus-0
            mean=HOPTIMUS_MEAN,
            std=HOPTIMUS_STD
        ),
    ])
```

**Transformation détaillée** :

| Étape | Input | Output | Transformation |
|-------|-------|--------|----------------|
| 1. ToPILImage | (256, 256, 3) uint8 [0-255] | PIL Image RGB | Conversion format |
| 2. Resize | PIL 256×256 | PIL 224×224 | Bilinear interpolation |
| 3. ToTensor | PIL Image | (3, 224, 224) float32 [0.0-1.0] | Divide by 255 |
| 4. Normalize | (3, 224, 224) [0-1] | (3, 224, 224) float32 | (x - mean) / std |

**Valeurs après normalisation** :

```python
# Pixel original: [234, 198, 217] (rose tissue)
# Après ToTensor: [0.918, 0.776, 0.851]
# Après Normalize:
#   R: (0.918 - 0.707) / 0.212 = 0.996
#   G: (0.776 - 0.579) / 0.230 = 0.857
#   B: (0.851 - 0.704) / 0.178 = 0.826
# → [0.996, 0.857, 0.826]
```

**IMPORTANT — Bug Historique #1** :
- Si image est float64 [0-255], `ToPILImage()` multiplie par 255 → overflow
- **Solution** : Toujours convertir en uint8 AVANT ToPILImage
- Ce bug a causé des features corrompues (corrigé 2025-12-20)

### Traitement B3 : Extraction Features H-optimus-0

**Code** (lignes 97-101) :
```python
with torch.no_grad():
    features = backbone.forward_features(tensor)  # (1, 261, 1536)

features_np = features.cpu().numpy()[0]  # (261, 1536)
```

**Détails `backbone.forward_features()`** :
- Modèle : H-optimus-0 (ViT-Giant/14)
- Paramètres : 1.1 milliard (tous gelés)
- Architecture :
  1. Patch Embedding : 224×224 → 16×16 patches de 14×14 pixels
  2. ViT Encoder : 40 couches Transformer
  3. **LayerNorm final** (CRUCIAL)
  4. Output : CLS token (1×1536) + Patch tokens (256×1536) + Registres (4×1536)

**Transformation** :
- Input : `tensor (1, 3, 224, 224)` — Image normalisée
- Output : `features (1, 261, 1536)` — Embeddings
  - Token 0 : CLS (contexte global)
  - Tokens 1-256 : Patches (16×16 grid)
  - Tokens 257-260 : Registres (pour stabilité entraînement)

**Validation CLS std** (CRITIQUE) :
```python
cls_token = features[:, 0, :]  # (1, 1536)
cls_std = cls_token.std().item()

# DOIT être entre 0.70 et 0.90
assert 0.70 <= cls_std <= 0.90, f"CLS std {cls_std:.3f} hors range!"
```

**IMPORTANT — Bug Historique #2** :
- Si on utilise `blocks[23]` au lieu de `forward_features()`, pas de LayerNorm final
- CLS std ~0.28 au lieu de ~0.77 → prédictions fausses
- **Solution** : Toujours utiliser `forward_features()`
- Ce bug a causé des prédictions incorrectes (corrigé 2025-12-21)

### Traitement B4 : Sauvegarde Features

**Code** (lignes 120-132) :
```python
np.savez_compressed(
    features_file,
    features=features_array,    # (M, 261, 1536) float32
    fold_ids=fold_ids,          # (M,) int32
    image_ids=image_ids,        # (M,) int32
)

np.savez_compressed(
    targets_file,
    np_targets=np_targets,      # (M, 256, 256) float32
    hv_targets=hv_targets,      # (M, 2, 256, 256) float32
    nt_targets=nt_targets,      # (M, 256, 256) int64
    fold_ids=fold_ids,          # (M,) int32
    image_ids=image_ids,        # (M,) int32
)
```

**Fichiers de sortie** :
```
data/cache/family_data/{family}_features.npz   (~500 MB pour 571 images)
data/cache/family_data/{family}_targets.npz    (~100 MB pour 571 images)
```

**Séparation features/targets** :
- **Raison** : Features chargées en RAM, targets chargées au besoin
- **Bénéfice** : Économise RAM (~400 MB par famille)

---

## ÉTAPE 3 : DataLoader (train_hovernet_family.py)

### Script
`scripts/training/train_hovernet_family.py`

### Classe Dataset
`FamilyHoVerDataset` (lignes 40-150)

### Traitement C1 : Chargement des .npz

**Code** (lignes 60-75) :
```python
features_file = cache_dir / f"{family}_features.npz"
targets_file = cache_dir / f"{family}_targets.npz"

features_data = np.load(features_file)
targets_data = np.load(targets_file)

self.features = features_data['features']       # (M, 261, 1536)
self.np_targets = targets_data['np_targets']    # (M, 256, 256)
self.hv_targets = targets_data['hv_targets']    # (M, 2, 256, 256)
self.nt_targets = targets_data['nt_targets']    # (M, 256, 256)
```

**Aucune transformation** — Toutes les données chargées en RAM.

### Traitement C2 : Resize 256 → 224 (CRITIQUE)

**Code** (__getitem__ méthode, lignes 95-115) :
```python
# Features: déjà à 224×224 (implicite via H-optimus-0)
features = torch.from_numpy(self.features[idx]).float()  # (261, 1536)

# Targets: 256×256 → RESIZE vers 224×224
np_target = torch.from_numpy(self.np_targets[idx]).float()  # (256, 256)
hv_target = torch.from_numpy(self.hv_targets[idx]).float()  # (2, 256, 256)
nt_target = torch.from_numpy(self.nt_targets[idx]).long()   # (256, 256)

# Resize NP: nearest (binaire)
np_target = F.interpolate(
    np_target.unsqueeze(0).unsqueeze(0),  # (1, 1, 256, 256)
    size=(224, 224),
    mode='nearest'
).squeeze()  # (224, 224)

# Resize HV: bilinear (gradients)
hv_target = F.interpolate(
    hv_target.unsqueeze(0),  # (1, 2, 256, 256)
    size=(224, 224),
    mode='bilinear',
    align_corners=False
).squeeze(0)  # (2, 224, 224)

# Resize NT: nearest (classes discrètes)
nt_target = F.interpolate(
    nt_target.unsqueeze(0).unsqueeze(0).float(),  # (1, 1, 256, 256)
    size=(224, 224),
    mode='nearest'
).squeeze().long()  # (224, 224)
```

**Interpolation Methods** :

| Target | Mode | Raison |
|--------|------|--------|
| NP | nearest | Binaire (0/1), pas d'interpolation |
| HV | bilinear | Gradients continus [-1, 1] |
| NT | nearest | Classes discrètes [0-4] |

**IMPORTANT — Direction du Resize** :
- ✅ **CORRECT** : Targets 256 → 224 (pour matcher features à 224)
- ❌ **ERREUR** : Prédictions 224 → 256 (cause mismatch)

**align_corners=False** :
- Alignement des pixels pour interpolation
- False = centre des pixels (standard PyTorch)

### Traitement C3 : Data Augmentation (Optionnel)

**Code** (lignes 120-145) :
```python
if self.augment and random.random() < 0.5:
    # Flip horizontal
    if random.random() < 0.5:
        features_2d = features_2d.flip(dims=[1])  # Flip sur H
        np_target = np_target.flip(dims=[1])
        nt_target = nt_target.flip(dims=[1])

        hv_h = hv_target[0].flip(dims=[1]) * (-1)  # Inverser H
        hv_v = hv_target[1].flip(dims=[1])         # V inchangé
        hv_target = torch.stack([hv_h, hv_v], dim=0)

    # Rotation 90°
    k = random.randint(0, 3)  # 0=0°, 1=90°, 2=180°, 3=270°
    if k > 0:
        features_2d = torch.rot90(features_2d, k=k, dims=[0, 1])
        np_target = torch.rot90(np_target, k=k, dims=[0, 1])
        nt_target = torch.rot90(nt_target, k=k, dims=[0, 1])

        # Rotation HV: échanger H/V selon l'angle
        # k=1 (90°):  H'=-V, V'=H
        # k=2 (180°): H'=-H, V'=-V
        # k=3 (270°): H'=V, V'=-H
        hv_target = rotate_hv_maps(hv_target, k)
```

**Transformation HV lors d'augmentation** :

| Augmentation | H' | V' | Explication |
|--------------|----|----|-------------|
| Flip H | -H | V | Inverser direction horizontale |
| Flip V | H | -V | Inverser direction verticale |
| Rot 90° | -V | H | Rotation axes |
| Rot 180° | -H | -V | Inversion complète |
| Rot 270° | V | -H | Rotation inverse |

**Important** : Augmentation appliquée sur features reshape en 16×16 grid.

### Traitement C4 : Batch Preparation

**Code** (DataLoader, lignes 400-410) :
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,  # Typiquement 8-16
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # Optimisation GPU
)
```

**Structure batch** :

| Tensor | Shape | Dtype | Device |
|--------|-------|-------|--------|
| features | (B, 261, 1536) | float32 | CUDA |
| np_target | (B, 224, 224) | float32 | CUDA |
| hv_target | (B, 2, 224, 224) | float32 | CUDA |
| nt_target | (B, 224, 224) | int64 | CUDA |

**Exemple B=8** :
```python
batch = next(iter(train_loader))
features, np_target, hv_target, nt_target = batch

print(features.shape)   # torch.Size([8, 261, 1536])
print(np_target.shape)  # torch.Size([8, 224, 224])
print(hv_target.shape)  # torch.Size([8, 2, 224, 224])
print(nt_target.shape)  # torch.Size([8, 224, 224])
```

---

## ÉTAPE 4 : HoVer-Net Training

### Forward Pass

**Code** (lignes 430-435) :
```python
np_out, hv_out, nt_out = hovernet(features)
```

**Architecture HoVer-Net** (src/models/hovernet_decoder.py) :

```
Input: features (B, 261, 1536)
    ↓
Bottleneck 1×1: 1536 → 256  [Économie VRAM]
    ↓
Reshape: (B, 256, 16, 16)  [Grid 16×16]
    ↓
Upsampling Block 1: 16×16 → 32×32  [256 → 128 channels]
    ↓
Upsampling Block 2: 32×32 → 64×64  [128 → 64 channels]
    ↓
Upsampling Block 3: 64×64 → 128×128  [64 → 64 channels]
    ↓
Upsampling Block 4: 128×128 → 224×224  [64 → 64 channels]
    ↓
    ├─→ NP Head: Conv 64→2 → (B, 2, 224, 224)  [Background, Nuclei]
    ├─→ HV Head: Conv 64→2 → (B, 2, 224, 224)  [H, V gradients]
    └─→ NT Head: Conv 64→5 → (B, 5, 224, 224)  [5 classes]
```

**Shapes de sortie** :

| Output | Shape | Signification |
|--------|-------|---------------|
| np_out | (B, 2, 224, 224) | Logits [Background, Nuclei] |
| hv_out | (B, 2, 224, 224) | Gradients H/V (pas d'activation) |
| nt_out | (B, 5, 224, 224) | Logits [Neo, Infl, Conn, Dead, Epit] |

**IMPORTANT — Activations** :
- NP : Pas d'activation (Cross-Entropy attend logits)
- HV : Pas d'activation (régression directe vers [-1, 1])
- NT : Pas d'activation (Cross-Entropy attend logits)

### Loss Computation

**Code** (lignes 440-450) :
```python
loss_fn = HoVerNetLoss(
    lambda_np=args.lambda_np,   # 1.0
    lambda_hv=args.lambda_hv,   # 2.0
    lambda_nt=args.lambda_nt,   # 1.0
    adaptive=args.adaptive_loss,  # False par défaut
)

total_loss, loss_dict = loss_fn(
    np_out, hv_out, nt_out,
    np_target, hv_target, nt_target
)
```

**Détails HoVerNetLoss** (src/models/hovernet_decoder.py lignes 294-325) :

#### Loss NP (Nuclear Presence)

```python
# Cross-Entropy
np_bce = F.cross_entropy(np_out, np_target.long())

# Dice Loss
pred_soft = F.softmax(np_out, dim=1)[:, 1]  # Proba classe "Nuclei"
intersection = (pred_soft * np_target).sum()
union = pred_soft.sum() + np_target.sum()
dice = (2 * intersection + 1e-5) / (union + 1e-5)
np_dice = 1 - dice

# Total NP
np_loss = np_bce + np_dice
```

#### Loss HV (Horizontal-Vertical) — CRITIQUE

```python
# Masque: calculer loss UNIQUEMENT sur pixels de noyaux
mask = np_target.float().unsqueeze(1)  # (B, 1, 224, 224)

if mask.sum() > 0:
    # Masquer predictions et targets
    hv_pred_masked = hv_out * mask
    hv_target_masked = hv_target * mask

    # MSE masqué (changé depuis SmoothL1 - Step 3 verification)
    hv_mse_sum = F.mse_loss(hv_pred_masked, hv_target_masked, reduction='sum')
    hv_l1 = hv_mse_sum / (mask.sum() * 2)  # *2 car 2 canaux
else:
    hv_l1 = 0.0

# Gradient Loss (MSGE - Mean Squared Gradient Error)
hv_gradient = gradient_loss(hv_out, hv_target, mask)
hv_loss = hv_l1 + 0.5 * hv_gradient
```

**IMPORTANT — Masque HV** :
- Sans masque : modèle apprend à prédire HV=0 partout (background domine)
- Avec masque : modèle apprend les gradients UNIQUEMENT sur noyaux
- Amélioration : HV MSE 0.30 → 0.01 (Glandular/Digestive)

**Gradient Loss** :
```python
def gradient_loss(pred, target, mask):
    # Gradient horizontal
    pred_h = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_h = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Gradient vertical
    pred_v = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_v = target[:, :, 1:, :] - target[:, :, :-1, :]

    # MSE sur gradients
    grad_loss = F.mse_loss(pred_h, target_h) + F.mse_loss(pred_v, target_v)
    return grad_loss
```

**Raison** : Force le modèle à apprendre les variations spatiales (pas juste valeurs moyennes).

#### Loss NT (Nuclear Type)

```python
# Cross-Entropy sur TOUS les pixels
nt_loss = F.cross_entropy(nt_out, nt_target.long())
```

**Note** : Calculé sur tous les pixels (y compris background=Neoplastic).

#### Loss Totale

```python
if adaptive:
    # Uncertainty Weighting (Kendall et al. 2018)
    total = (
        torch.exp(-log_var_np) * np_loss + log_var_np +
        torch.exp(-log_var_hv) * hv_loss + log_var_hv +
        torch.exp(-log_var_nt) * nt_loss + log_var_nt
    )
else:
    # Fixed weights
    total = lambda_np * np_loss + lambda_hv * hv_loss + lambda_nt * nt_loss
```

**Poids recommandés** :
- λ_NP = 1.0 (segmentation)
- λ_HV = 2.0 (séparation instances — priorité)
- λ_NT = 1.0 (classification)

### Backward Pass

**Code** (lignes 455-460) :
```python
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

**Optimizer** : AdamW
**Learning Rate** : 1e-4
**Scheduler** : ReduceLROnPlateau (factor=0.5, patience=5)

---

## ÉTAPE 5 : Validation & Metrics

### Conversion Prédictions

**Code** (lignes 480-490) :
```python
# NP: argmax sur classes [Background, Nuclei]
np_pred = np_out.argmax(dim=1).float()  # (B, 224, 224) [0, 1]

# HV: pas d'activation (déjà [-1, 1])
hv_pred = hv_out  # (B, 2, 224, 224)

# NT: argmax sur 5 classes
nt_pred = nt_out.argmax(dim=1)  # (B, 224, 224) [0-4]
```

### Calcul Métriques

**Dice Score** :
```python
intersection = (np_pred * np_target).sum()
union = np_pred.sum() + np_target.sum()
dice = (2 * intersection) / (union + 1e-8)
```

**HV MSE** :
```python
mask = np_target.unsqueeze(1)
hv_mse = F.mse_loss(hv_pred * mask, hv_target * mask)
```

**NT Accuracy** :
```python
mask = (np_target > 0.5)  # Pixels de noyaux seulement
correct = (nt_pred[mask] == nt_target[mask]).sum()
total = mask.sum()
nt_acc = correct / total
```

**Résultats Typiques** :

| Famille | NP Dice | HV MSE | NT Acc |
|---------|---------|--------|--------|
| Glandular | 0.9648 | 0.0106 | 0.9111 |
| Digestive | 0.9634 | 0.0163 | 0.8824 |
| Urologic | 0.9318 | 0.2812 | 0.9139 |

---

## RÉSUMÉ COMPLET DU PIPELINE

### Données à Chaque Étape

| Étape | Format | Dimensions | Dtype | Range |
|-------|--------|------------|-------|-------|
| **PanNuke brut** | .npy | (N, 256, 256, 3/6) | uint8/int32 | [0-255]/[0-max_id] |
| **Images FIXED** | .npz | (M, 256, 256, 3) | uint8 | [0-255] |
| **NP targets** | .npz | (M, 256, 256) | float32 | [0.0-1.0] |
| **HV targets** | .npz | (M, 2, 256, 256) | float32 | [-1.0-1.0] |
| **NT targets** | .npz | (M, 256, 256) | int64 | [0-4] |
| **Tensor image** | torch | (1, 3, 224, 224) | float32 | Normalisé |
| **Features H-opt** | .npz | (M, 261, 1536) | float32 | Variable |
| **Batch features** | torch | (B, 261, 1536) | float32 | CUDA |
| **Batch targets** | torch | (B, [2], 224, 224) | float32/int64 | CUDA |
| **Prédictions NP** | torch | (B, 2, 224, 224) | float32 | Logits |
| **Prédictions HV** | torch | (B, 2, 224, 224) | float32 | [-inf, inf] |
| **Prédictions NT** | torch | (B, 5, 224, 224) | float32 | Logits |

### Points Critiques pour Cohérence

| # | Point Critique | Entraînement | Évaluation | Doit Être Identique |
|----|----------------|--------------|------------|---------------------|
| 1 | Conversion uint8 | ✅ Avant ToPILImage | ❓ | OUI |
| 2 | Transform | create_hoptimus_transform() | ❓ | OUI |
| 3 | Normalisation | HOPTIMUS_MEAN/STD | ❓ | OUI |
| 4 | Features | forward_features() | ❓ | OUI |
| 5 | CLS std | ~0.77 | ❓ | OUI |
| 6 | GT Instances | extract_pannuke_instances() | ❓ | OUI |
| 7 | HV targets | compute_hv_maps() float32 | ❓ | OUI |
| 8 | Resize direction | Targets 256→224 | ❓ | OUI |
| 9 | Interpolation NP | nearest | ❓ | OUI |
| 10 | Interpolation HV | bilinear | ❓ | OUI |

**TOUT doit être identique** entre entraînement et évaluation.
