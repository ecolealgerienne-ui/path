# CellViT-Optimus — Contexte Projet

> **Version:** V13 Smart Crops + FPN Chimique + Macenko
> **Date:** 2025-12-30
> **Objectif:** AJI ≥ 0.68

---

## Historique Complet

Pour l'historique complet du développement (bugs résolus, décisions techniques, journal de développement), voir: **[claude_history.md](./claude_history.md)**

---

## ⚠️ CONSIGNES CRITIQUES POUR CLAUDE

> **🚫 INTERDICTION ABSOLUE DE TESTER LOCALEMENT**
>
> Claude NE DOIT JAMAIS essayer d'exécuter des commandes de test, d'entraînement, ou d'évaluation dans son environnement.
>
> **Actions AUTORISÉES :**
> - ✅ Lire des fichiers (code, configs, documentation)
> - ✅ Créer/modifier du code Python
> - ✅ Créer des scripts que L'UTILISATEUR lancera
> - ✅ Faire de la review de code
> - ✅ Créer de la documentation
>
> **Actions INTERDITES :**
> - ❌ `python scripts/training/...` (pas d'env)
> - ❌ `python scripts/evaluation/...` (pas de données)
> - ❌ Toute commande nécessitant GPU/données

---

## Vue d'ensemble

**CellViT-Optimus** est un système de segmentation et classification de noyaux cellulaires pour l'histopathologie.

**Architecture actuelle:** V13 Smart Crops + FPN Chimique + Macenko Normalization

**Résultat Respiratory:** AJI 0.6872 = **101% de l'objectif 0.68** ✅

---

## Pipeline Complet (Data Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE CELLVIT-OPTIMUS                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  PanNuke Dataset    │
│  (7,904 images)     │
│  256×256 RGB        │
│  fold0/, fold1/,    │
│  fold2/             │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1: NORMALISATION MACENKO                                             │
│  Script: normalize_staining_source.py                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Charge images 256×256 depuis PanNuke                                     │
│  • Applique Macenko stain normalization (réduction variance ~30%)           │
│  • Sauvegarde: data/family_FIXED/{family}_data_FIXED.npz                   │
│  • Contenu: images, fold_ids, image_ids, organ_names                        │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2: GÉNÉRATION SMART CROPS                                            │
│  Script: prepare_v13_smart_crops.py                                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Source images: family_FIXED/ (avec --use_normalized)                     │
│  • Source masks: PanNuke raw (fold{N}/masks.npy)                           │
│  • 5 crops 224×224 par image + rotations déterministes                      │
│  • Split CTO: train/val par source_image_ids (ZERO leakage)                │
│  • Sauvegarde: data/family_data_v13_smart_crops/{family}_{split}.npz       │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 3: EXTRACTION FEATURES H-OPTIMUS-0                                   │
│  Script: extract_features_v13_smart_crops.py                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Backbone: H-optimus-0 (ViT-Giant/14, 1.1B params, GELÉ)                  │
│  • Entrée: 224×224 RGB normalisé                                            │
│  • Sortie: (B, 261, 1536) = CLS + 4 registers + 256 patches                 │
│  • Cache: data/cache/family_data/{family}_{split}_features.pt              │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 4: ENTRAÎNEMENT HOVERNET DECODER                                     │
│  Script: train_hovernet_family_v13_smart_crops.py                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Architecture: FPN Chimique + h_alpha learnable                           │
│  • Injection H-channel: 5 niveaux (16→32→64→112→224)                        │
│  • Losses: NP (BCE) + HV (MSE) + NT (CE)                                    │
│  • Checkpoint: models/checkpoints_v13_smart_crops/                          │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 5: ÉVALUATION AJI                                                    │
│  Script: test_v13_smart_crops_aji.py                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Post-processing: HV-guided Watershed                                     │
│  • Métriques: AJI, Dice, mPQ                                                │
│  • Paramètres optimisés par famille                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Scripts de Validation

| Script | Usage | Vérifications |
|--------|-------|---------------|
| `verify_v13_smart_crops_data.py` | Après étape 2 | HV targets, inst_maps, normalisation Macenko |
| `verify_pipeline_integrity.py` | Après étape 4 | H-channel, h_alpha, dimensions, gradients |

---

## Architecture V13 Smart Crops + FPN Chimique

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IMAGE H&E SOURCE (256×256)                           │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                    5 Crops Stratégiques (224×224)
                    + Rotations Déterministes
                               │
┌─────────────────────────────────────────────────────────────────────────┐
│              H-OPTIMUS-0 (ViT-Giant/14, 1.1B params, gelé)              │
│  • Entrée: 224×224 @ 0.5 MPP                                            │
│  • Sortie: CLS token (1536) + 256 Patches (1536)                       │
└─────────────────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
┌──────────────────────┐            ┌──────────────────────────────────────┐
│  CLS Token (1536)    │            │  Patch Tokens (256, 1536)            │
│         │            │            │         │                            │
│    OrganHead         │            │    FPN Chimique                      │
│   (99.94% acc)       │            │   + H-Channel Injection              │
│         │            │            │         │                            │
│   19 Organes         │            │  ┌──────┴─────┬───────┐              │
│   + OOD              │            │  NP       HV       NT               │
└──────────────────────┘            └──────────────────────────────────────┘
                                                    │
                               ┌────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    WATERSHED POST-PROCESSING                            │
│  • beta=0.50, min_size=30, np_threshold=0.40, min_distance=5           │
│  • Formule: marker_energy = dist × (1 - hv_magnitude^beta)             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architecture FPN Chimique

Injection multi-échelle du canal Hématoxyline (H-channel) à 5 niveaux:

```
Niveau 0: Bottleneck 256 + H@16×16   (sémantique)
Niveau 1: Features 128 + H@32×32
Niveau 2: Features 64 + H@64×64
Niveau 3: Features 32 + H@112×112
Niveau 4: Features 16 + H@224×224    (détails)

Paramètres FPN: 2,696,017
```

### Stratégie 5 Crops (Split-First-Then-Rotate)

Chaque image source 256×256 génère 5 crops 224×224 avec rotations:

| Position | Coordonnées | Rotation |
|----------|-------------|----------|
| Centre | (16, 16) | 0° |
| Haut-Gauche | (0, 0) | 90° CW |
| Haut-Droit | (32, 0) | 180° |
| Bas-Gauche | (0, 32) | 270° CW |
| Bas-Droit | (32, 32) | Flip H |

**Principe CTO:** Split train/val par source_image_ids AVANT rotation → ZERO data leakage

---

## Résultats Actuels (SANS Normalisation Macenko)

> **⚠️ IMPORTANT:** Tous les résultats ci-dessous ont été obtenus **SANS normalisation Macenko**.
> Un test comparatif AVEC normalisation est en cours sur Respiratory.

### Récapitulatif 5/5 Familles

| Famille | Samples | AJI | Progress | Paramètres Watershed |
|---------|---------|-----|----------|----------------------|
| **Respiratory** | 408 | **0.6872** | **101.1%** ✅ | beta=0.50, min_size=30, np_thr=0.40, min_dist=5 |
| **Urologic** | 1101 | **0.6743** | **99.2%** | beta=0.50, min_size=30, np_thr=0.45, min_dist=2 |
| **Glandular** | 3391 | **0.6566** | **96.6%** | beta=0.50, min_size=50, np_thr=0.40, min_dist=3 |
| Epidermal | 574 | 0.6203 | 91.2% | beta=1.00, min_size=20, np_thr=0.45, min_dist=3 |
| Digestive | 2430 | 0.6160 | 90.6% | beta=2.00, min_size=60, np_thr=0.45, min_dist=5 |

**Objectif atteint:** 1/5 (Respiratory) | **Proche (>96%):** 3/5

---

## Pipeline Complet (Commandes)

**Exemple pour famille `respiratory`** — Remplacer par la famille souhaitée.

> **Important:** Adapter `--pannuke_dir` à votre installation locale.

### 1. Normalisation Macenko (Recommandé)

```bash
# Charge depuis PanNuke 256×256, sauve dans family_FIXED/
python scripts/preprocessing/normalize_staining_source.py \
    --family respiratory \
    --pannuke_dir /chemin/vers/PanNuke

# Résultat attendu: variance ~21 → ~14 (réduction ~30%)
```

**Contenu généré:** `data/family_FIXED/respiratory_data_FIXED.npz`
- `images`: (N, 256, 256, 3) normalisées
- `fold_ids`: (N,) origine fold
- `image_ids`: (N,) index dans fold
- `organ_names`: (N,) nom organe

### 2. Générer Smart Crops

```bash
# AVEC normalisation Macenko (RECOMMANDÉ)
python scripts/preprocessing/prepare_v13_smart_crops.py \
    --family respiratory \
    --use_normalized \
    --pannuke_dir /chemin/vers/PanNuke \
    --max_samples 5000

# SANS normalisation (legacy)
python scripts/preprocessing/prepare_v13_smart_crops.py \
    --family respiratory \
    --pannuke_dir /chemin/vers/PanNuke \
    --max_samples 5000

# Pour un organe spécifique
python scripts/preprocessing/prepare_v13_smart_crops.py \
    --family glandular \
    --organ Breast \
    --pannuke_dir /chemin/vers/PanNuke \
    --max_samples 5000
```

### 3. Vérifier Données Générées

```bash
# Vérifier split train (inclut détection normalisation)
python scripts/validation/verify_v13_smart_crops_data.py --family respiratory --split train

# Vérifier split val
python scripts/validation/verify_v13_smart_crops_data.py --family respiratory --split val

# Résultats attendus:
#   ✅ Normalisation Macenko DÉTECTÉE (variance < 18)
#   ✅ HV targets: float32 [-1, 1]
#   ✅ inst_maps: LOCAL relabeling OK
```

### 4. Extraire Features H-optimus-0

```bash
python scripts/preprocessing/extract_features_v13_smart_crops.py --family epidermal --split train
python scripts/preprocessing/extract_features_v13_smart_crops.py --family epidermal --split val

# Vérifier les features générées
ls -la data/cache/family_data/
```

### 5. Entraînement FPN Chimique

```bash
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal \
    --epochs 60 \
    --use_hybrid \
    --use_fpn_chimique \
    --use_h_alpha
```

**⚠️ IMPORTANT:** `--use_fpn_chimique` nécessite TOUJOURS `--use_hybrid`

### 6. Évaluation AJI

```bash
# Respiratory (AJI 0.6872 ✅)
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth \
    --family respiratory \
    --n_samples 50 \
    --use_hybrid \
    --use_fpn_chimique \
    --np_threshold 0.40 \
    --min_size 30 \
    --min_distance 5

# Urologic (AJI 0.6743)
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_urologic_v13_smart_crops_hybrid_fpn_best.pth \
    --family urologic \
    --n_samples 50 \
    --use_hybrid \
    --use_fpn_chimique \
    --np_threshold 0.45 \
    --min_size 30 \
    --min_distance 2

# Epidermal (AJI 0.6203)
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth \
    --family epidermal \
    --n_samples 50 \
    --use_hybrid \
    --use_fpn_chimique \
    --np_threshold 0.45 \
    --min_size 20 \
    --beta 1.0 \
    --min_distance 3

# Glandular (AJI 0.6566)
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_glandular_v13_smart_crops_hybrid_fpn_best.pth \
    --family glandular \
    --n_samples 50 \
    --use_hybrid \
    --use_fpn_chimique \
    --np_threshold 0.40 \
    --min_size 50 \
    --beta 0.5 \
    --min_distance 3

# Digestive (AJI 0.6160)
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_digestive_v13_smart_crops_hybrid_fpn_best.pth \
    --family digestive \
    --n_samples 50 \
    --use_hybrid \
    --use_fpn_chimique \
    --np_threshold 0.45 \
    --min_size 60 \
    --beta 2.0 \
    --min_distance 5
```

**Paramètres Watershed optimisés par famille (SANS normalisation):**

| Famille | np_threshold | min_size | beta | min_distance | AJI | Status |
|---------|--------------|----------|------|--------------|-----|--------|
| Respiratory | 0.40 | 30 | 0.50 | 5 | **0.6872** | ✅ Objectif |
| Urologic | 0.45 | 30 | 0.50 | 2 | **0.6743** | 99.2% |
| Glandular | 0.40 | 50 | 0.50 | 3 | **0.6566** | 96.6% |
| Epidermal | 0.45 | 20 | 1.00 | 3 | 0.6203 | 91.2% |
| Digestive | 0.45 | 60 | 2.00 | 5 | 0.6160 | 90.6% |

### 7. Optimisation Watershed (optionnel)

```bash
python scripts/evaluation/optimize_watershed_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth \
    --family epidermal \
    --n_samples 50
```

---

## 5 Familles HoVer-Net

| Famille | Organes | Samples |
|---------|---------|---------|
| **Glandular** | Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland | 3391 |
| **Digestive** | Colon, Stomach, Esophagus, Bile-duct | 2430 |
| **Urologic** | Kidney, Bladder, Testis, Ovarian, Uterus, Cervix | 1101 |
| **Respiratory** | Lung, Liver | 408 |
| **Epidermal** | Skin, HeadNeck | 574 |

---

## Constantes Importantes

### Normalisation H-optimus-0

```python
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224
```

### Structure Features

```
features (B, 261, 1536):
├── features[:, 0, :]       # CLS token → OrganHead
├── features[:, 1:5, :]     # 4 Register tokens (IGNORER)
└── features[:, 5:261, :]   # 256 Patch tokens → HoVer-Net
```

---

## Règles Critiques

### 1. Ne Pas Modifier l'Existant

> **"On touche pas l'existant"** — Les scripts existants fonctionnent. Toute modification requiert validation explicite.

### 2. Modules Partagés OBLIGATOIRES

> **🚫 JAMAIS de duplication de code critique**
>
> Les algorithmes critiques DOIVENT être dans `src/` et importés par tous les scripts.
> **NE JAMAIS copier-coller** une fonction entre scripts — créer un module partagé.

**Modules partagés existants:**

| Module | Fonction | Usage |
|--------|----------|-------|
| `src/postprocessing/watershed.py` | `hv_guided_watershed()` | Segmentation instances |
| `src/metrics/ground_truth_metrics.py` | `compute_aji()` | Calcul AJI+ |
| `src/evaluation/instance_evaluation.py` | `run_inference()`, `evaluate_sample()`, `evaluate_batch_with_params()` | Évaluation complète |

**Import obligatoire:**

```python
# ✅ CORRECT - Single source of truth
from src.postprocessing import hv_guided_watershed
from src.metrics.ground_truth_metrics import compute_aji
from src.evaluation import run_inference, evaluate_batch_with_params

# ❌ INTERDIT - Duplication de code
def hv_guided_watershed(...):  # Copie locale
def run_inference(...):        # Copie locale
```

**Pourquoi:** Évite les divergences d'algorithme entre scripts (bug découvert 2025-12-29: scipy.ndimage.label vs skimage.measure.label causait -2.8% AJI).

### 3. FPN Chimique = use_hybrid + use_fpn_chimique

```bash
# ✅ CORRECT (Training ET Évaluation)
--use_hybrid --use_fpn_chimique

# ❌ INCORRECT
--use_fpn_chimique  # Sans --use_hybrid → Erreur
```

### 4. Nommage des Checkpoints

```bash
# FPN Chimique checkpoint:
hovernet_{family}_v13_smart_crops_hybrid_fpn_best.pth

# Exemple:
hovernet_epidermal_v13_smart_crops_hybrid_fpn_best.pth
```

### 5. Validation CLS std

Le CLS token std doit être entre **0.70 et 0.90**.

### 6. Transfer Learning Inter-Famille

Pour transférer un modèle entraîné sur une famille vers une autre (ex: Respiratory → Epidermal):

```bash
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal \
    --pretrained_checkpoint models/checkpoints_v13_smart_crops/hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth \
    --finetune_lr 1e-5 \
    --epochs 30 \
    --use_hybrid \
    --use_fpn_chimique
```

**Différences avec `--resume`:**

| Aspect | `--resume` | `--pretrained_checkpoint` |
|--------|-----------|---------------------------|
| Usage | Même famille | Famille différente |
| Epoch | Continue depuis sauvegardé | Reset à 0 |
| Optimizer | Reprend état sauvegardé | Nouveau avec LR ultra-bas |
| LR par défaut | `args.lr` (1e-4) | `args.finetune_lr` (1e-5) |

**Paramètres recommandés:**
- LR: 1e-5 ou 5e-6 (évite catastrophic forgetting)
- λ_hv: 10.0 (maintient skills séparation instances)
- Epochs: 20-30 (adaptation, pas réapprentissage)

---

## Environnement

| Composant | Version |
|-----------|---------|
| OS | WSL2 Ubuntu 24.04.2 LTS |
| GPU | RTX 4070 SUPER (12.9 GB VRAM) |
| Python | 3.10 (Miniconda) |
| PyTorch | 2.6.0+cu124 |
| Conda env | `cellvit` |

---

## Documentation Clé

| Document | Description |
|----------|-------------|
| [claude_history.md](./claude_history.md) | Historique complet du développement |
| [docs/V13_SMART_CROPS_STRATEGY.md](./docs/V13_SMART_CROPS_STRATEGY.md) | Stratégie V13 (CTO validée) |
| [docs/sessions/2025-12-29_respiratory_v13_smart_crops_results.md](./docs/sessions/2025-12-29_respiratory_v13_smart_crops_results.md) | Résultats Respiratory |

---

## Prochaines Étapes

1. **Glandular** (3391 samples) — Plus grand dataset, attendu >0.68 AJI
2. **Digestive** (2430 samples) — Deuxième plus grand
3. **Epidermal** (574 samples) — Challenge tissus stratifiés
4. **Urologic** (1101 samples) — Tissus denses

---

## Références

- H-optimus-0: https://huggingface.co/bioptimus/H-optimus-0
- HoVer-Net: Graham et al., Medical Image Analysis 2019
- PanNuke: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
