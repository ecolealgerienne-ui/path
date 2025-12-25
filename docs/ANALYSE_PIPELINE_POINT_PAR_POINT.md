# ANALYSE DU PIPELINE POINT PAR POINT

> **Date:** 2025-12-25
> **Objectif:** Documentation complète du pipeline de traitement, de l'entrainement au test

---

## ORDRE D'EXÉCUTION DES SCRIPTS

```
Script 1: prepare_family_data_FIXED_v12_COHERENT.py
    ↓
Script 2: extract_features_from_v9.py
    ↓
Script 3: train_hovernet_family.py
    ↓
Script 4: test_epidermal_aji_FINAL.py
```

---

## SCRIPT 1: prepare_family_data_FIXED_v12_COHERENT.py

### Rôle
Prépare les données d'entraînement à partir des folds PanNuke bruts.

### Entrées

| Donnée | Source | Type | Structure |
|--------|--------|------|-----------|
| images.npy | PanNuke fold | np.ndarray | (N, 256, 256, 3) uint8 [0-255] |
| masks.npy | PanNuke fold | np.ndarray | (N, 256, 256, 6) int32 |
| types.npy | PanNuke fold | np.ndarray | (N,) strings (noms d'organes) |

### Structure du mask PanNuke (6 canaux)

| Canal | Contenu |
|-------|---------|
| 0 | Instances multi-types (IDs d'instances) |
| 1 | Instances Neoplastic |
| 2 | Instances Inflammatory |
| 3 | Instances Connective |
| 4 | Instances Dead |
| 5 | Tissue mask (exclu du traitement) |

### Traitements

#### 1. Filtrage par famille
```
Entrée: types (N organes)
Sortie: indices des échantillons de la famille cible
Logique: ORGAN_TO_FAMILY[organ] == family
```

Mapping organe → famille:
- **glandular**: Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland
- **digestive**: Colon, Stomach, Esophagus, Bile-duct
- **urologic**: Kidney, Bladder, Testis, Ovarian, Uterus, Cervix
- **respiratory**: Lung, Liver
- **epidermal**: Skin, HeadNeck

#### 2. Normalisation du mask
```
Entrée: mask shape (256, 256, 6) ou (6, 256, 256)
Sortie: mask shape (256, 256, 6) - format HWC
```

#### 3. Extraction instances (canaux 0-4 uniquement)
```python
def extract_pannuke_instances_NUCLEI_ONLY(mask):
    # Canal 0: instances multi-types (SOURCE PRIMAIRE)
    # Canaux 1-4: instances par classe (ajoutées si non-vides)
    # Canal 5: EXCLU (tissue mask, pas des noyaux)
```
```
Entrée: mask (256, 256, 6)
Sortie: inst_map (256, 256) int32 - IDs d'instances [0, 1, 2, ...]
        (0 = background, 1+ = noyaux individuels)
```

#### 4. Calcul NP target (Nuclear Presence)
```python
def compute_np_target_v12(mask):
    nuclei_mask = compute_nuclei_mask_v12(mask)  # mask[:,:,:5].sum() > 0
    return nuclei_mask.astype(np.float32)
```
```
Entrée: mask (256, 256, 6)
Sortie: np_target (256, 256) float32 [0.0, 1.0]
        0.0 = background, 1.0 = noyau présent
```

#### 5. Calcul HV targets (Horizontal-Vertical maps)
```python
def compute_hv_maps(inst_map):
    # Pour chaque instance:
    #   1. Trouver centroïde
    #   2. Pour chaque pixel: vecteur normalisé (pixel → centroïde)
    #   3. Valeurs dans [-1, 1]
    # Gaussian smoothing sigma=0.5
```
```
Entrée: inst_map (256, 256) int32
Sortie: hv_map (2, 256, 256) float32 [-1, 1]
        hv_map[0] = Vertical (Y)
        hv_map[1] = Horizontal (X)
```

#### 6. Calcul NT target (Nuclear Type)
```python
def compute_nt_target_v12(mask):
    nuclei_mask = compute_nuclei_mask_v12(mask)  # MÊME masque que NP
    nt_target = np.zeros((256, 256), dtype=np.int64)
    nt_target[nuclei_mask] = 1  # Classification binaire
    return nt_target
```
```
Entrée: mask (256, 256, 6)
Sortie: nt_target (256, 256) int64 [0, 1]
        0 = background, 1 = noyau
```

### Sorties

| Fichier | Clé | Type | Structure |
|---------|-----|------|-----------|
| {family}_data_FIXED_v12_COHERENT.npz | images | uint8 | (N, 256, 256, 3) |
| | np_targets | float32 | (N, 256, 256) |
| | hv_targets | float32 | (N, 2, 256, 256) |
| | nt_targets | int64 | (N, 256, 256) |
| | fold_ids | int32 | (N,) |
| | image_ids | int32 | (N,) |

**Note:** Toutes les données sont à 256×256 à cette étape.

---

## SCRIPT 2: extract_features_from_v9.py

### Rôle
Extrait les features H-optimus-0 depuis les données préparées.

### Entrées

| Donnée | Source | Type | Structure |
|--------|--------|------|-----------|
| Data file | Script 1 | npz | Contient images (N, 256, 256, 3) uint8 |

### Traitements

#### 1. Préparation image
```python
if image.dtype != np.uint8:
    image = image.clip(0, 255).astype(np.uint8)
```
```
Entrée: image (256, 256, 3) uint8 ou float
Sortie: image (256, 256, 3) uint8
```

#### 2. Transform H-optimus-0
```python
transform = create_hoptimus_transform()
# Compose([
#     ToPILImage(),
#     Resize((224, 224)),      # RESIZE 256→224
#     ToTensor(),
#     Normalize(HOPTIMUS_MEAN, HOPTIMUS_STD)
# ])
tensor = transform(image).unsqueeze(0)
```
```
Entrée: image (256, 256, 3) uint8
Sortie: tensor (1, 3, 224, 224) float32 normalisé
```

Constantes de normalisation:
- HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
- HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

#### 3. Extraction features H-optimus-0
```python
features = backbone.forward_features(tensor)
```
```
Entrée: tensor (1, 3, 224, 224)
Sortie: features (1, 261, 1536) float32
        - 1 CLS token
        - 256 patch tokens (16×16 grid)
        - 4 register tokens
```

### Sorties

| Fichier | Clé | Type | Structure |
|---------|-----|------|-----------|
| {family}_features.npz | features | float32 | (N, 261, 1536) |
| {family}_targets.npz | np_targets | float32 | (N, 256, 256) |
| | hv_targets | float32 | (N, 2, 256, 256) |
| | nt_targets | int64 | (N, 256, 256) |

**Note:** Les features sont extraites depuis images redimensionnées à 224×224, mais les targets restent à 256×256 (copie directe depuis Script 1).

---

## SCRIPT 3: train_hovernet_family.py

### Rôle
Entraîne le décodeur HoVer-Net sur les features pré-extraites.

### Entrées

| Donnée | Source | Type | Structure |
|--------|--------|------|-----------|
| Features | Script 2 | npz | (N, 261, 1536) float32 |
| Targets | Script 2 | npz | np (N, 256, 256), hv (N, 2, 256, 256), nt (N, 256, 256) |

### Architecture du Modèle

#### HoVerNetDecoder
```
Input: patch_tokens (B, 256, 1536)
    ↓
Bottleneck: Linear(1536 → 64)
    ↓
Reshape: (B, 64, 16, 16)
    ↓
Upsampling blocks (16→32→64→112→224)
    ↓
3 têtes de sortie:
    - NP head: Conv(64 → 2)     # 2 classes (background, foreground)
    - HV head: Conv(64 → 2)     # 2 canaux (V, H)
    - NT head: Conv(64 → n_classes)  # n classes
```

### Traitements

#### 1. Dataset (FamilyHoVerDataset.__getitem__)
```python
def __getitem__(self, idx):
    features = self.features[idx]     # (261, 1536)
    np_target = self.np_targets[idx]  # (256, 256)
    hv_target = self.hv_targets[idx]  # (2, 256, 256)
    nt_target = self.nt_targets[idx]  # (256, 256)

    # RESIZE TARGETS 256→224
    np_target, hv_target, nt_target = resize_targets(
        np_target, hv_target, nt_target,
        target_size=224
    )

    return features, np_target, hv_target, nt_target
```

#### 2. Fonction resize_targets (src/data/preprocessing.py)
```python
def resize_targets(np_target, hv_target, nt_target, target_size=224):
    # NP: F.interpolate mode='nearest' (valeurs binaires)
    # HV: F.interpolate mode='bilinear' (valeurs continues)
    # NT: F.interpolate mode='nearest' (labels discrets)
```
```
Entrée: np (256, 256), hv (2, 256, 256), nt (256, 256)
Sortie: np (224, 224), hv (2, 224, 224), nt (224, 224)
```

#### 3. Forward Pass
```python
patch_tokens = features[:, 1:257, :]  # Exclure CLS et registers
np_out, hv_out, nt_out = hovernet(patch_tokens)
```
```
Entrée: patch_tokens (B, 256, 1536)
Sortie:
    - np_out (B, 2, 224, 224)
    - hv_out (B, 2, 224, 224)
    - nt_out (B, n_classes, 224, 224)
```

#### 4. Fonctions de Loss

**NP Loss (CrossEntropyLoss + DiceLoss):**
```python
np_loss = CrossEntropyLoss(np_out, np_target) + DiceLoss(np_out, np_target)
```
- np_out: (B, 2, 224, 224) - logits pour 2 classes
- np_target: (B, 224, 224) - labels 0 ou 1

**HV Loss (SmoothL1Loss + GradientLoss + MagnitudeLoss):**
```python
hv_loss = SmoothL1(hv_out, hv_target) + 3.0 * gradient_loss + 5.0 * magnitude_loss
```
- hv_out: (B, 2, 224, 224) - prédictions [-1, 1]
- hv_target: (B, 2, 224, 224) - cibles [-1, 1]

**NT Loss (CrossEntropyLoss):**
```python
nt_loss = CrossEntropyLoss(nt_out, nt_target)
```
- nt_out: (B, n_classes, 224, 224) - logits
- nt_target: (B, 224, 224) - labels entiers

**Loss Totale:**
```python
total_loss = lambda_np * np_loss + lambda_hv * hv_loss + lambda_nt * nt_loss
# Valeurs par défaut: lambda_np=1.0, lambda_hv=2.0, lambda_nt=1.0
```

### Sorties

| Fichier | Contenu |
|---------|---------|
| hovernet_{family}_best.pth | Modèle entraîné (weights du décodeur) |
| hovernet_{family}_last.pth | Dernier checkpoint |

### Métriques d'Entraînement

| Métrique | Calcul | Cible |
|----------|--------|-------|
| NP Dice | 2×\|Pred∩GT\| / (\|Pred\|+\|GT\|) | > 0.90 |
| HV MSE | Mean((pred - target)²) | < 0.05 |
| NT Acc | Pixels corrects / Total pixels | > 0.85 |

---

## SCRIPT 4: test_epidermal_aji_FINAL.py

### Rôle
Évalue le modèle entraîné sur les données de test et calcule les métriques AJI, Dice, PQ.

### Entrées

| Donnée | Source | Type | Structure |
|--------|--------|------|-----------|
| Images | Script 1 | npz | (N, 256, 256, 3) uint8 |
| GT Masks | PanNuke | npy | (N, 256, 256, 6) int32 |
| Modèle | Script 3 | pth | HoVer-Net weights |

### Traitements

#### 1. Préparation image
```python
if image.dtype != np.uint8:
    image = image.clip(0, 255).astype(np.uint8)
```

#### 2. Transform (identique au training)
```python
transform = create_hoptimus_transform()  # Resize 256→224
tensor = transform(image).unsqueeze(0).to(device)
```
```
Entrée: image (256, 256, 3) uint8
Sortie: tensor (1, 3, 224, 224) float32
```

#### 3. Feature Extraction
```python
features = backbone.forward_features(tensor)
patch_tokens = features[:, 1:257, :]
```
```
Entrée: tensor (1, 3, 224, 224)
Sortie: patch_tokens (1, 256, 1536)
```

#### 4. Prédiction HoVer-Net
```python
np_out, hv_out, nt_out = hovernet(patch_tokens)
```
```
Entrée: patch_tokens (1, 256, 1536)
Sortie:
    - np_out (1, 2, 224, 224)
    - hv_out (1, 2, 224, 224)
    - nt_out (1, n_classes, 224, 224)
```

#### 5. Conversion numpy et activations
```python
np_pred = torch.softmax(np_out, dim=1)[0].cpu().numpy()  # (2, 224, 224)
np_pred = np_pred.transpose(1, 2, 0)  # (224, 224, 2)

hv_pred = hv_out[0].cpu().numpy()  # (2, 224, 224)
hv_pred = hv_pred.transpose(1, 2, 0)  # (224, 224, 2)
```

#### 6. Resize prédictions 224→256
```python
# NP: interpolation linéaire
np_pred_256 = cv2.resize(np_pred, (256, 256), interpolation=cv2.INTER_LINEAR)

# HV: interpolation linéaire par canal
hv_pred_256 = np.zeros((256, 256, 2))
hv_pred_256[:, :, 0] = cv2.resize(hv_pred[:, :, 0], (256, 256), interpolation=cv2.INTER_LINEAR)
hv_pred_256[:, :, 1] = cv2.resize(hv_pred[:, :, 1], (256, 256), interpolation=cv2.INTER_LINEAR)
```
```
Entrée: np_pred (224, 224, 2), hv_pred (224, 224, 2)
Sortie: np_pred_256 (256, 256, 2), hv_pred_256 (256, 256, 2)
```

#### 7. Post-processing: Extraction instances
```python
def get_instance_segmentation(np_pred, hv_pred):
    # 1. Binarisation NP
    prob_map = np_pred[:, :, 1]  # Probabilité foreground
    binary_mask = (prob_map > 0.5).astype(np.uint8)

    # 2. Calcul gradients HV (Sobel)
    grad_h = sobel(hv_pred[:, :, 0])
    grad_v = sobel(hv_pred[:, :, 1])

    # 3. Magnitude des gradients
    magnitude = np.sqrt(grad_h**2 + grad_v**2)

    # 4. Markers pour watershed
    markers = label(binary_mask * (magnitude < threshold))

    # 5. Watershed
    instances = watershed(-magnitude, markers, mask=binary_mask)

    return instances
```
```
Entrée: np_pred (256, 256, 2), hv_pred (256, 256, 2)
Sortie: instances (256, 256) int32 - IDs d'instances
```

#### 8. Extraction instances GT
```python
def get_correct_gt_instances(mask):
    # Utilise canal 0 comme base
    # Ajoute instances des canaux 1-4 si non-vides
    # Canal 5 exclu (tissue)
```
```
Entrée: mask (256, 256, 6)
Sortie: gt_instances (256, 256) int32
```

#### 9. Calcul métriques

**Dice Score:**
```python
dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
# Comparaison binaire: pred > 0 vs gt > 0
```

**AJI (Aggregated Jaccard Index):**
```python
def compute_aji(pred_inst, gt_inst):
    # Pour chaque instance GT:
    #   1. Trouver l'instance prédite avec meilleur IoU
    #   2. Accumuler IoU
    # AJI = sum(IoU) / (|GT| + |Pred non-matchées|)
```

**PQ (Panoptic Quality):**
```python
def compute_pq(pred_inst, gt_inst, iou_threshold=0.5):
    # PQ = DQ × SQ
    # DQ (Detection Quality) = TP / (TP + 0.5*FP + 0.5*FN)
    # SQ (Segmentation Quality) = mean(IoU) pour les matches
```

### Sorties

| Métrique | Description | Cible |
|----------|-------------|-------|
| Dice | Chevauchement binaire global | > 0.90 |
| AJI | Qualité de la segmentation d'instances | > 0.60 |
| PQ | Qualité panoptique (détection + segmentation) | > 0.65 |

---

## SYNTHÈSE DES TAILLES

| Étape | Script | Images | Targets | Features | Prédictions |
|-------|--------|--------|---------|----------|-------------|
| Préparation données | Script 1 | 256×256 | 256×256 | - | - |
| Extraction features | Script 2 | 256→224 | 256×256 (copie) | 261×1536 | - |
| Entraînement | Script 3 | - | 256→224 | 261×1536 | 224×224 |
| Test (inférence) | Script 4 | 256→224 | 256×256 | 261×1536 | 224×224 |
| Test (post-proc) | Script 4 | - | 256×256 | - | 224→256 |

---

## FLUX DE DONNÉES COMPLET

```
┌────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1: prepare_family_data_FIXED_v12_COHERENT.py             │
├────────────────────────────────────────────────────────────────┤
│ PanNuke folds (256×256)                                        │
│     ↓                                                          │
│ images: (N, 256, 256, 3) uint8                                 │
│ np_targets: (N, 256, 256) float32 [0, 1]                       │
│ hv_targets: (N, 2, 256, 256) float32 [-1, 1]                   │
│ nt_targets: (N, 256, 256) int64 [0, 1]                         │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2: extract_features_from_v9.py                           │
├────────────────────────────────────────────────────────────────┤
│ Images 256×256 → Transform (Resize 224) → H-optimus-0          │
│     ↓                                                          │
│ features: (N, 261, 1536) float32                               │
│     - 1 CLS token                                              │
│     - 256 patch tokens                                         │
│     - 4 register tokens                                        │
│                                                                │
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
│   Input: patch_tokens (B, 256, 1536)                           │
│   Output: np(B,2,224,224), hv(B,2,224,224), nt(B,n,224,224)    │
│                                                                │
│ Loss:                                                          │
│   np_loss = CrossEntropy + Dice                                │
│   hv_loss = SmoothL1 + gradient + magnitude                    │
│   nt_loss = CrossEntropy                                       │
│   total = λ_np×np + λ_hv×hv + λ_nt×nt                         │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ ÉTAPE 4: test_epidermal_aji_FINAL.py                           │
├────────────────────────────────────────────────────────────────┤
│ Inférence:                                                     │
│   image 256×256 → Transform (Resize 224) → H-optimus-0         │
│   features → HoVerNetDecoder → np/hv/nt (224×224)              │
│                                                                │
│ Post-processing:                                               │
│   np/hv (224×224) → Resize (256×256) → Watershed               │
│   → instances (256×256)                                        │
│                                                                │
│ Évaluation:                                                    │
│   instances prédites vs instances GT                           │
│   → Dice, AJI, PQ                                              │
└────────────────────────────────────────────────────────────────┘
```

---

## FICHIERS DE CONFIGURATION

### src/constants.py (constantes centralisées)

```python
# Version des données
CURRENT_DATA_VERSION = "v12_COHERENT"

# Tailles
HOPTIMUS_INPUT_SIZE = 224
PANNUKE_IMAGE_SIZE = 256

# Normalisation H-optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

# Validation features
HOPTIMUS_CLS_STD_MIN = 0.70
HOPTIMUS_CLS_STD_MAX = 0.90
```

### Chemins par défaut

| Donnée | Chemin |
|--------|--------|
| Données famille | data/family_FIXED/{family}_data_FIXED_v12_COHERENT.npz |
| Features | data/cache/family_data/{family}_features.npz |
| Targets | data/cache/family_data/{family}_targets.npz |
| Checkpoints | models/checkpoints/hovernet_{family}_best.pth |

---

## MODULES CLÉS

### src/preprocessing/__init__.py
- `create_hoptimus_transform()`: Transform canonique pour H-optimus-0
- `preprocess_image()`: Prétraitement complet image → tensor
- `validate_features()`: Validation CLS std [0.70, 0.90]

### src/data/preprocessing.py
- `resize_targets()`: Resize targets 256→224
- `validate_targets()`: Validation dtype et range des targets
- `load_targets()`: Chargement centralisé des targets

### src/models/hovernet_decoder.py
- `HoVerNetDecoder`: Architecture du décodeur
- `HoVerNetLoss`: Fonction de loss combinée

### src/models/loader.py
- `ModelLoader.load_hoptimus0()`: Chargement backbone H-optimus-0
