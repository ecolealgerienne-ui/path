# ANALYSE DU PIPELINE POINT PAR POINT

> **Date:** 2025-12-25 (mise à jour)
> **Objectif:** Documenter chaque étape de traitement, entrées/sorties, pour identifier la source de l'écart Training (Dice 0.95) vs Évaluation (Dice 0.32, AJI 0.03)

---

## 🔴 RÉSUMÉ EXÉCUTIF: BUGS IDENTIFIÉS

### Bug #1: CENTER PADDING au lieu de RESIZE (CORRIGÉ, mais pas suffisant)

| Étape | Méthode | Taille | Problème |
|-------|---------|--------|----------|
| **Training** | Image 256→224 via `Resize()` | 224×224 | L'image est **COMPRESSÉE** |
| **Training** | Targets 256→224 via `resize_targets()` | 224×224 | Targets **COMPRESSÉS** de la même façon |
| **Test** | Image 256→224 via `Resize()` | 224×224 | L'image est **COMPRESSÉE** ✅ |
| **Test** | Prédictions 224→256 via ~~CENTER PADDING~~ | 256×256 | ✅ CORRIGÉ → RESIZE |

**Statut:** ✅ Corrigé (commit fb66774) — Mais Dice toujours 0.32 après fix!

---

### 🔴🔴🔴 Bug #2: DATA MISMATCH v9 vs v12 (PROBABLE CAUSE RACINE)

**Le script `extract_features_from_v9.py` charge par défaut le fichier v9:**
```python
# Ligne 66 de extract_features_from_v9.py:
input_file = Path(f"data/family_FIXED/{args.family}_data_FIXED_v9_NUCLEI_ONLY.npz")
```

**Scénario probable:**
1. ✅ Utilisateur crée v12: `epidermal_data_FIXED_v12_COHERENT.npz`
2. ❌ Utilisateur extrait features SANS spécifier `--input_file` → utilise v9 par défaut
3. ❌ Training utilise `epidermal_features.npz` + `epidermal_targets.npz` (générés depuis v9)
4. ✅ Test utilise `epidermal_data_FIXED_v12_COHERENT.npz` (v12)

**Conséquence:** Le modèle a été entraîné sur v9 (avec bug NP/NT), mais testé contre GT compatible v12!

### Vérification nécessaire:

```bash
# Vérifier les dates des fichiers:
ls -la data/cache/family_data/epidermal*.npz
ls -la data/family_FIXED/epidermal*.npz

# Si epidermal_features.npz est PLUS ANCIEN que epidermal_data_FIXED_v12_COHERENT.npz
# → C'est le bug!
```

### Solution:

```bash
# Re-extraire features DEPUIS v12:
python scripts/preprocessing/extract_features_from_v9.py \
    --family epidermal \
    --input_file data/family_FIXED/epidermal_data_FIXED_v12_COHERENT.npz \
    --output_dir data/cache/family_data

# Puis re-entraîner:
python scripts/training/train_hovernet_family.py --family epidermal --epochs 50 --augment
```

---

### Bug #3 potentiel: GT de test vs Targets d'entraînement

**Test utilise:**
```python
gt_inst = get_correct_gt_instances(gt_mask)  # Depuis PanNuke brut (canal 0 + 1-4)
```

**Training utilise:**
```python
np_target = compute_np_target_v12(mask)  # mask[:,:,:5].sum() > 0 (union binaire)
```

Ces deux peuvent être différents si les canaux PanNuke ne correspondent pas exactement.

---

### Impact cumulé:
- Bug #1 (padding): Corrigé
- Bug #2 (v9/v12 mismatch): **PROBABLE CAUSE RACINE** - À vérifier
- Bug #3 (GT vs targets): Potentiel - À vérifier après fix de #2

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

### Entrée
| Donnée | Type | Structure | Taille |
|--------|------|-----------|--------|
| images.npy | np.ndarray | (N, 256, 256, 3) | uint8 [0-255] |
| masks.npy | np.ndarray | (N, 256, 256, 6) | int32 (instance IDs) |
| types.npy | np.ndarray | (N,) | strings (organes) |

### Traitements

#### Traitement 1: Filtrage par famille
```
Entrée: types (N organes)
Sortie: indices des échantillons de la famille cible
Logique: ORGAN_TO_FAMILY[organ] == family
```

#### Traitement 2: Normalisation du mask
```
Entrée: mask shape (256, 256, 6) ou (6, 256, 256)
Sortie: mask shape (256, 256, 6) - format HWC
Logique: normalize_mask_format() transpose si nécessaire
```

#### Traitement 3: Extraction instances (NUCLEI ONLY)
```python
def extract_pannuke_instances_NUCLEI_ONLY(mask):
    # Canal 0: instances multi-types (SOURCE PRIMAIRE)
    # Canaux 1-4: instances par classe (ajoutées si non-vides)
    # Canal 5: EXCLU (c'est du tissue, pas des noyaux)
```
```
Entrée: mask (256, 256, 6)
Sortie: inst_map (256, 256) int32 - IDs d'instances [0, 1, 2, ...]
        (0 = background, 1+ = noyaux)
```

#### Traitement 4: Calcul NP target
```python
def compute_np_target_v12(mask):
    nuclei_mask = compute_nuclei_mask_v12(mask)  # mask[:,:,:5].sum() > 0
    return nuclei_mask.astype(np.float32)
```
```
Entrée: mask (256, 256, 6)
Sortie: np_target (256, 256) float32 [0.0, 1.0]
```

#### Traitement 5: Calcul HV targets
```python
def compute_hv_maps(inst_map):
    # Pour chaque instance:
    #   1. Trouver centroïde
    #   2. Pour chaque pixel: vecteur (pixel → centroïde)
    #   3. Normalisation radiale [-1, 1]
    # Gaussian smoothing sigma=0.5
```
```
Entrée: inst_map (256, 256) int32
Sortie: hv_map (2, 256, 256) float32 [-1, 1]
        hv_map[0] = Vertical (Y)
        hv_map[1] = Horizontal (X)
```

#### Traitement 6: Calcul NT target
```python
def compute_nt_target_v12(mask):
    nuclei_mask = compute_nuclei_mask_v12(mask)  # MÊME que NP
    nt_target = np.zeros((256, 256), dtype=np.int64)
    nt_target[nuclei_mask] = 1  # Binaire: 0=bg, 1=noyau
    return nt_target
```
```
Entrée: mask (256, 256, 6)
Sortie: nt_target (256, 256) int64 [0, 1]
```

### Sortie
| Fichier | Contenu | Type | Structure |
|---------|---------|------|-----------|
| {family}_data_FIXED_v12_COHERENT.npz | images | uint8 | (N, 256, 256, 3) |
| | np_targets | float32 | (N, 256, 256) |
| | hv_targets | float32 | (N, 2, 256, 256) |
| | nt_targets | int64 | (N, 256, 256) |
| | fold_ids | int32 | (N,) |
| | image_ids | int32 | (N,) |

### ⚠️ Point Critique
> **Toutes les données sont à 256×256 à cette étape.**

---

## SCRIPT 2: extract_features_from_v9.py

### Entrée
| Donnée | Source | Type | Structure |
|--------|--------|------|-----------|
| Data file | Script 1 | npz | images (N, 256, 256, 3) uint8 |

### Traitements

#### Traitement 1: Préparation image
```python
if image.dtype != np.uint8:
    image = image.clip(0, 255).astype(np.uint8)
```
```
Entrée: image (256, 256, 3) uint8 ou float
Sortie: image (256, 256, 3) uint8
```

#### Traitement 2: Transform H-optimus-0
```python
transform = create_hoptimus_transform()
# = ToPILImage()
#   → Resize((224, 224))  # ⚠️ RESIZE 256→224
#   → ToTensor()
#   → Normalize(HOPTIMUS_MEAN, HOPTIMUS_STD)
tensor = transform(image).unsqueeze(0)
```
```
Entrée: image (256, 256, 3) uint8
Sortie: tensor (1, 3, 224, 224) float32 normalisé
```

### 🔴 POINT CRITIQUE: L'image 256×256 est COMPRESSÉE (squeezed) en 224×224

#### Traitement 3: Extraction features H-optimus-0
```python
features = backbone.forward_features(tensor)
# features shape: (1, 261, 1536)
# [CLS token (1) + 256 patch tokens] × 1536 dims
```
```
Entrée: tensor (1, 3, 224, 224)
Sortie: features (1, 261, 1536) float32
```

### Sortie
| Fichier | Contenu | Type | Structure |
|---------|---------|------|-----------|
| {family}_features.npz | features | float32 | (N, 261, 1536) |
| {family}_targets.npz | np_targets | float32 | (N, 256, 256) |
| | hv_targets | float32 | (N, 2, 256, 256) |
| | nt_targets | int64 | (N, 256, 256) |

### ⚠️ Point Critique
> **Features extraites depuis images 224×224, mais targets toujours à 256×256!**

---

## SCRIPT 3: train_hovernet_family.py

### Entrée
| Donnée | Source | Type | Structure |
|--------|--------|------|-----------|
| Features | Script 2 | npz | (N, 261, 1536) float32 |
| Targets | Script 2 | npz | np (N, 256, 256), hv (N, 2, 256, 256), nt (N, 256, 256) |

### Traitements

#### Traitement 1: Chargement Dataset (FamilyHoVerDataset)
```python
class FamilyHoVerDataset(Dataset):
    def __getitem__(self, idx):
        # Récupérer features et targets
        features = self.features[idx]
        np_target = self.np_targets[idx]
        hv_target = self.hv_targets[idx]
        nt_target = self.nt_targets[idx]

        # ⚠️ RESIZE TARGETS 256→224
        np_target, hv_target, nt_target = resize_targets(
            np_target, hv_target, nt_target,
            target_size=224  # Resize vers 224 pour matcher features
        )
```

#### Traitement 2: resize_targets (src/data/preprocessing.py)
```python
def resize_targets(np_target, hv_target, nt_target, target_size=224):
    # NP: interpolation 'linear' (probabilités)
    np_resized = cv2.resize(np_target, (target_size, target_size),
                           interpolation=cv2.INTER_LINEAR)

    # HV: interpolation 'linear' par canal
    hv_resized = np.zeros((2, target_size, target_size))
    for c in range(2):
        hv_resized[c] = cv2.resize(hv_target[c], (target_size, target_size),
                                   interpolation=cv2.INTER_LINEAR)

    # NT: interpolation 'nearest' (labels discrets)
    nt_resized = cv2.resize(nt_target, (target_size, target_size),
                           interpolation=cv2.INTER_NEAREST)

    return np_resized, hv_resized, nt_resized
```
```
Entrée: np (256, 256), hv (2, 256, 256), nt (256, 256)
Sortie: np (224, 224), hv (2, 224, 224), nt (224, 224)
```

### ✅ À cette étape: Features (224×224) et Targets (224×224) sont ALIGNÉS

#### Traitement 3: Forward Pass HoVer-Net
```python
patch_tokens = features[:, 1:257, :]  # (B, 256, 1536)
np_out, hv_out, nt_out = hovernet(patch_tokens)
# Sorties: (B, 2, 224, 224), (B, 2, 224, 224), (B, 5, 224, 224)
```
```
Entrée: patch_tokens (B, 256, 1536)
Sortie: np_out (B, 2, 224, 224), hv_out (B, 2, 224, 224), nt_out (B, 5, 224, 224)
```

#### Traitement 4: Calcul Loss
```python
# NP Loss: CrossEntropy sur (B, 2, 224, 224) vs targets (B, 224, 224)
# HV Loss: SmoothL1 sur (B, 2, 224, 224) vs targets (B, 2, 224, 224)
# NT Loss: CrossEntropy sur (B, 5, 224, 224) vs targets (B, 224, 224)
```

### Sortie
| Fichier | Contenu |
|---------|---------|
| hovernet_{family}_best.pth | Modèle entraîné à 224×224 |

### ⚠️ Point Critique
> **Le modèle apprend sur des données COMPRESSÉES 256→224 via RESIZE (cv2.INTER_LINEAR)**

---

## SCRIPT 4: test_epidermal_aji_FINAL.py

### Entrée
| Donnée | Source | Type | Structure |
|--------|--------|------|-----------|
| Images | Script 1 (v12) | npz | (N, 256, 256, 3) uint8 |
| GT Masks | PanNuke brut | npy | (N, 256, 256, 6) int32 |
| Modèle | Script 3 | pth | HoVer-Net |

### Traitements

#### Traitement 1: Préparation image
```python
if image.dtype != np.uint8:
    image = image.clip(0, 255).astype(np.uint8)
```

#### Traitement 2: Transform (IDENTIQUE au training)
```python
transform = create_hoptimus_transform()  # Resize 256→224
tensor = transform(image).unsqueeze(0).to(device)
```
```
Entrée: image (256, 256, 3) uint8
Sortie: tensor (1, 3, 224, 224) float32
```

### ✅ Jusqu'ici cohérent avec training

#### Traitement 3: Feature Extraction
```python
features = backbone.forward_features(tensor)
patch_tokens = features[:, 1:257, :]  # (1, 256, 1536)
```

#### Traitement 4: Prédiction
```python
np_out, hv_out, nt_out = hovernet(patch_tokens)
# Sorties à 224×224
```

#### Traitement 5: Conversion numpy + axes
```python
np_pred = torch.softmax(np_out, dim=1)[0].cpu().numpy().transpose(1, 2, 0)  # (224, 224, 2)
hv_pred = hv_out[0].cpu().numpy().transpose(1, 2, 0)  # (224, 224, 2)
```

#### ✅ Traitement 6: RESIZE 224→256 (CORRIGÉ - commit fb66774)
```python
# APRÈS FIX (lignes 321-329):
# Resize NP (interpolation linéaire pour probabilités)
np_pred_256 = cv2.resize(np_pred, (256, 256), interpolation=cv2.INTER_LINEAR)

# Resize HV (interpolation linéaire par canal)
hv_pred_256 = np.zeros((256, 256, 2), dtype=hv_pred.dtype)
hv_pred_256[:, :, 0] = cv2.resize(hv_pred[:, :, 0], (256, 256), interpolation=cv2.INTER_LINEAR)
hv_pred_256[:, :, 1] = cv2.resize(hv_pred[:, :, 1], (256, 256), interpolation=cv2.INTER_LINEAR)
```

### ✅ Bug #1 corrigé — MAIS Dice toujours 0.32!

```
PROBLÈME:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  TRAINING:                                                             │
│  Image 256×256 → Resize() → Image 224×224                             │
│  [                ]      [            ]                                │
│  Target 256×256 → Resize() → Target 224×224                           │
│  [                ]      [            ]                                │
│  → L'image est COMPRESSÉE, le target aussi                            │
│  → ALIGNEMENT PARFAIT                                                  │
│                                                                        │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                        │
│  TEST:                                                                 │
│  Image 256×256 → Resize() → Image 224×224                             │
│  [                ]      [            ]                                │
│                                                                        │
│  Prédiction 224×224 → CENTER PADDING → Prédiction 256×256             │
│        [            ]           → [   [            ]   ]              │
│                                    ↑16px        ↑16px                  │
│                                    border       border                 │
│                                                                        │
│  GT reste à 256×256 (original)                                         │
│  [                ]                                                    │
│                                                                        │
│  → La prédiction est DÉCALÉE de 16px par rapport au GT!               │
│  → Le contenu prédit correspond à l'image COMPRESSÉE                   │
│  → Mais il est PADÉ au lieu d'être RE-ÉTIRÉ                           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

#### Traitement 7: Extraction instances GT
```python
gt_inst = get_correct_gt_instances(gt_mask)
# Utilise canal 0 + canaux 1-4 si non-vides
# GT à 256×256 original
```

#### Traitement 8: Calcul métriques
```python
aji = compute_aji(pred_inst, gt_inst)  # Comparaison de deux maps 256×256
dice = compute_dice(prob_map > 0.5, gt_inst > 0)
pq = compute_panoptic_quality(pred_inst, gt_inst)
```

---

## 📊 SYNTHÈSE DES TAILLES

| Étape | Script | Image | Targets/GT | Prédiction |
|-------|--------|-------|------------|------------|
| Préparation | Script 1 | 256×256 | 256×256 | - |
| Extraction | Script 2 | 256→224 (resize) | 256×256 | - |
| Training | Script 3 | 224×224 | 256→224 (resize) | 224×224 |
| Test (input) | Script 4 | 256→224 (resize) | 256×256 | 224×224 |
| Test (output) | Script 4 | - | 256×256 | 224→256 (**PADDING**) |

---

## 🔴 LE BUG EN VISUEL

```
TRAINING (CORRECT):
┌─────────────────┐     ┌─────────────┐
│                 │     │             │
│  Image 256×256  │ ──▶ │ Image       │
│                 │     │ 224×224     │
└─────────────────┘     └─────────────┘
        ↓ resize              ↓ correspond exactement
┌─────────────────┐     ┌─────────────┐
│                 │     │             │
│  Target 256×256 │ ──▶ │ Target      │
│                 │     │ 224×224     │
└─────────────────┘     └─────────────┘

TEST (BUG):
┌─────────────────┐     ┌─────────────┐     ┌───────────────────┐
│                 │     │             │     │    padding 16px   │
│  Image 256×256  │ ──▶ │ Image       │ ──▶ │  ┌─────────────┐  │
│                 │     │ 224×224     │     │  │ Pred 224    │  │
└─────────────────┘     └─────────────┘     │  └─────────────┘  │
                                            └───────────────────┘
                                                    VS
                              ┌─────────────────┐
                              │                 │
                              │  GT 256×256     │  ← Non modifié
                              │  (original)     │
                              └─────────────────┘

RÉSULTAT: La prédiction (contenu compressé dans zone centrale)
          ne correspond PAS au GT (contenu à l'échelle originale)
```

---

## ✅ SOLUTION PROPOSÉE

### Option A: Modifier le test pour utiliser RESIZE au lieu de CENTER PADDING

```python
# AVANT (BUG):
diff = (256 - 224) // 2
np_pred_256 = np.zeros((256, 256, 2))
np_pred_256[diff:diff+h, diff:diff+w, :] = np_pred  # CENTER PADDING

# APRÈS (FIX):
np_pred_256 = cv2.resize(np_pred, (256, 256), interpolation=cv2.INTER_LINEAR)
hv_pred_256 = np.zeros((256, 256, 2))
for c in range(2):
    hv_pred_256[:, :, c] = cv2.resize(hv_pred[:, :, c], (256, 256),
                                       interpolation=cv2.INTER_LINEAR)
```

### Pourquoi ça marchera:
1. Training: Image 256→224 (resize), Target 256→224 (resize)
2. Test: Image 256→224 (resize), Pred 224→256 (resize INVERSE)
3. Le resize inverse restaure la correspondance spatiale avec le GT

---

## 📋 VÉRIFICATION

Après correction, les métriques attendues:
- Dice: 0.35 → **~0.95** (comme training)
- AJI: 0.04 → **>0.60** (objectif)
- PQ: 0.00 → **>0.65** (objectif)

---

## ANNEXE: Commentaires trompeurs dans le code

Le script `test_epidermal_aji_FINAL.py` contient ces commentaires (lignes 309-315):

```python
# 2. CENTER PADDING 224→256 (au lieu de resize qui déforme)
#    ===================================================================
#    FIX EXPERT #2 (2025-12-24): PADDING au lieu de RESIZE
#    ===================================================================
#    CAUSE: H-optimus extrait crops centraux 224×224 d'images 256×256
#    AVANT: cv2.resize() étirait → décalage spatial → PQ=0.00
#    APRÈS: Center padding préserve positions exactes
```

**Ces commentaires sont ERRONÉS:**
- H-optimus-0 NE fait PAS de "crop central"
- `create_hoptimus_transform()` fait un `Resize((224, 224))` qui COMPRESSE l'image entière
- Le center padding introduit en fait le décalage spatial qu'il prétend corriger

---

## RÉSUMÉ FINAL

| Aspect | Status |
|--------|--------|
| Bug identifié | ✅ CENTER PADDING au lieu de RESIZE dans test |
| Cause racine | Incompréhension du fonctionnement de H-optimus-0 transform |
| Impact | Décalage spatial systématique → métriques catastrophiques |
| Solution | Remplacer center padding par cv2.resize() |
| Temps de fix | ~5 minutes |
