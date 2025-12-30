# CellViT-Optimus — Contexte Projet

> **Version:** V13 Smart Crops + FPN Chimique (Raw Images)
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

**Architecture actuelle:** V13 Smart Crops + FPN Chimique (Raw Images — sans normalisation Macenko)

**Résultat Respiratory:** AJI 0.6872 = **101% de l'objectif 0.68** ✅

---

## 🔬 Découverte Stratégique: Ruifrok vs Macenko (2025-12-30)

> **VERDICT: Macenko DÉSACTIVÉ pour la production V13**

### Résultat Expérimental

| Configuration | AJI Respiratory | Δ |
|---------------|-----------------|---|
| **SANS Macenko (Raw)** | **0.6872** ✅ | Baseline |
| AVEC Macenko | 0.6576 | **-4.3%** ❌ |

### Analyse Technique: Le "Shift de Projection"

Le FPN Chimique utilise la **déconvolution Ruifrok** pour extraire le canal Hématoxyline (H-channel):

```python
# Vecteur Ruifrok FIXE (constantes physiques Beer-Lambert)
stain_matrix = [0.650, 0.704, 0.286]  # Direction pure Hématoxyline
```

**Le Conflit:**
1. **Ruifrok** = Projection sur vecteur physique FIXE (absorption optique H&E)
2. **Macenko** = Rotation ADAPTATIVE dans l'espace OD pour aligner vers une référence
3. **Résultat:** Macenko déplace la composante Éosine vers le vecteur Hématoxyline
4. **Conséquence:** Le canal H extrait contient des "fantômes" de cytoplasme → bruit dans HV-MSE

### Pourquoi Raw Images > Macenko pour V13

| Aspect | Ruifrok (FPN Chimique) | Macenko |
|--------|------------------------|---------|
| **Philosophie** | Bio-Physique (Loi de Beer-Lambert) | Statistique (SVD/variance) |
| **Vecteurs** | Fixes (universels) | Adaptatifs (par image) |
| **Impact ADN** | Préserve contrastes fins (texture) | Lisse intensités (uniformité) |
| **Score AJI** | **Optimisé (0.6872)** | Dégradé (0.6576) |

### Implication Production

> *"The system leverages physical absorption constants (Ruifrok) which are intrinsically superior to adaptive statistical normalization (Macenko) for preserving nuclear chromatin texture."*

**Recommandations:**
1. ✅ **Verrouillage:** Macenko désactivé pour V13 production
2. ✅ **Data Augmentation:** Légère augmentation luminosité/contraste aléatoire (si nécessaire)
3. ❌ **Éviter:** Normalisation stain lourde qui détruit la texture chromatinienne

---

## Pipeline Complet (Data Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PIPELINE CELLVIT-OPTIMUS (Raw Images)                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  PanNuke Dataset    │
│  (7,904 images)     │
│  256×256 RGB RAW    │  ← Images brutes (PAS de normalisation Macenko)
│  fold0/, fold1/,    │
│  fold2/             │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1: GÉNÉRATION SMART CROPS                                            │
│  Script: prepare_v13_smart_crops.py                                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Source images: PanNuke RAW (fold{N}/images.npy) ← SANS --use_normalized  │
│  • Source masks: PanNuke raw (fold{N}/masks.npy)                           │
│  • 5 crops 224×224 par image + rotations déterministes                      │
│  • Split CTO: train/val par source_image_ids (ZERO leakage)                │
│  • Sauvegarde: data/family_data_v13_smart_crops/{family}_{split}.npz       │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2: EXTRACTION FEATURES H-OPTIMUS-0                                   │
│  Script: extract_features_v13_smart_crops.py                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Backbone: H-optimus-0 (ViT-Giant/14, 1.1B params, GELÉ)                  │
│  • Entrée: 224×224 RGB                                                      │
│  • Sortie: (B, 261, 1536) = CLS + 4 registers + 256 patches                 │
│  • Cache: data/cache/family_data/{family}_{split}_features.pt              │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 3: ENTRAÎNEMENT HOVERNET DECODER                                     │
│  Script: train_hovernet_family_v13_smart_crops.py                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Architecture: FPN Chimique + h_alpha learnable                           │
│  • Injection H-channel via Ruifrok: 5 niveaux (16→32→64→112→224)           │
│  • Losses: NP (BCE) + HV (MSE) + NT (CE)                                    │
│  • Checkpoint: models/checkpoints_v13_smart_crops/                          │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 4: ÉVALUATION AJI                                                    │
│  Script: test_v13_smart_crops_aji.py                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Post-processing: HV-guided Watershed                                     │
│  • Métriques: AJI, Dice, mPQ                                                │
│  • Paramètres optimisés par famille                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Note:** Macenko normalization est disponible via `--use_normalized` mais **déconseillée**
> pour V13 (régression -4.3% AJI due au conflit Ruifrok/Macenko)

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

## Résultats Actuels (Raw Images — Production)

> **✅ VALIDÉ (2025-12-30):** Images brutes (sans Macenko) = configuration optimale pour V13.
> Test comparatif: Macenko cause -4.3% AJI (voir section "Découverte Stratégique").

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

### 1. Générer Smart Crops (Raw Images)

```bash
# ✅ PRODUCTION: Images brutes depuis PanNuke (RECOMMANDÉ)
python scripts/preprocessing/prepare_v13_smart_crops.py \
    --family respiratory \
    --pannuke_dir /chemin/vers/PanNuke \
    --max_samples 5000

# ⚠️ DÉCONSEILLÉ: Avec normalisation Macenko (cause -4.3% AJI)
# python scripts/preprocessing/prepare_v13_smart_crops.py \
#     --family respiratory --use_normalized --pannuke_dir /chemin/vers/PanNuke
```

### 2. Vérifier Données Générées

```bash
# Vérifier split train
python scripts/validation/verify_v13_smart_crops_data.py --family respiratory --split train

# Vérifier split val
python scripts/validation/verify_v13_smart_crops_data.py --family respiratory --split val

# Résultats attendus (Raw Images):
#   ⚠️ Normalisation Macenko NON détectée (variance > 18) ← CORRECT pour V13
#   ✅ HV targets: float32 [-1, 1]
#   ✅ inst_maps: LOCAL relabeling OK
```

### 3. Extraire Features H-optimus-0

```bash
python scripts/preprocessing/extract_features_v13_smart_crops.py --family epidermal --split train
python scripts/preprocessing/extract_features_v13_smart_crops.py --family epidermal --split val

# Vérifier les features générées
ls -la data/cache/family_data/
```

### 4. Entraînement FPN Chimique

```bash
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal \
    --epochs 60 \
    --use_hybrid \
    --use_fpn_chimique \
    --use_h_alpha
```

**⚠️ IMPORTANT:** `--use_fpn_chimique` nécessite TOUJOURS `--use_hybrid`

### 5. Évaluation AJI

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

### 6. Optimisation Watershed (optionnel)

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
| [docs/UI_COCKPIT.md](./docs/UI_COCKPIT.md) | **R&D Cockpit (IHM Gradio)** — Architecture, API, Phases |

---

## Prochaines Étapes

> **Stratégie:** Toujours utiliser les modèles par **famille** (pas de modèles organ-specific).

### Priorités d'Amélioration

| Famille | AJI Actuel | Gap vs 0.68 | Priorité |
|---------|------------|-------------|----------|
| **Epidermal** | 0.6203 | -8.8% | Haute |
| **Digestive** | 0.6160 | -9.4% | Haute |
| **Glandular** | 0.6566 | -3.4% | Moyenne |
| Urologic | 0.6743 | -0.8% | Basse |
| Respiratory | 0.6872 | ✅ | Done |

### Pistes d'Optimisation

1. **Watershed tuning** — Continuer optimisation des paramètres par famille
2. **Data augmentation** — Augmentations légères (luminosité, contraste)
3. **Transfer learning** — Utiliser Respiratory comme pretrained pour les autres familles

---

## Références

- H-optimus-0: https://huggingface.co/bioptimus/H-optimus-0
- HoVer-Net: Graham et al., Medical Image Analysis 2019
- PanNuke: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
