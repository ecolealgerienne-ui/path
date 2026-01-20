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
│   (99.75% acc)       │            │   + H-Channel Injection              │
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

### 🔬 Optimisation Organ-Level (2025-12-31)

> **Découverte:** L'optimisation par organe révèle des paramètres watershed très différents
> masqués par l'approche famille. Gain potentiel significatif.

#### Respiratory: Lung vs Liver

| Organe | AJI | Beta | Min Size | NP Thr | Min Dist | Status |
|--------|-----|------|----------|--------|----------|--------|
| **Liver** | **0.7207** | 2.0 | 40 | 0.45 | 2 | ✅ **+6% vs objectif** |
| Lung | 0.6498 | 0.5 | 40 | 0.50 | 2 | 95.6% |
| *Famille Respiratory* | *0.6872* | *0.50* | *30* | *0.40* | *5* | *moyenne pondérée* |

**Insight clé:** Beta optimal varie de **0.5 (Lung)** à **2.0 (Liver)** — les noyaux hépatiques
nécessitent plus de pondération HV pour la séparation des instances.

#### Epidermal: Skin vs HeadNeck

| Organe | AJI | Beta | Min Size | NP Thr | Min Dist | Status |
|--------|-----|------|----------|--------|----------|--------|
| Skin | 0.6359 | 1.5 | 30 | 0.50 | 2 | 93.5% |
| HeadNeck | 0.6289 | 2.0 | 30 | 0.50 | 4 | 92.5% |
| *Famille Epidermal* | *0.6203* | *1.0* | *20* | *0.45* | *3* | *91.2%* |

**Insight:** Paramètres similaires entre Skin et HeadNeck (contrairement à Respiratory).
Amélioration organ-level: +1.4% à +2.5% vs famille. Gap restant ~6-7% vs objectif.

#### Digestive: Colon, Stomach, Esophagus, Bile-duct

| Organe | AJI | Beta | Min Size | NP Thr | Min Dist | Status |
|--------|-----|------|----------|--------|----------|--------|
| **Bile-duct** | **0.6980** | 1.0 | 30 | 0.50 | 3 | ✅ **102.6%** |
| **Stomach** | **0.6869** | 1.0 | 70 | 0.50 | 3 | ✅ **101%** |
| Esophagus | 0.6583 | 0.5 | 30 | 0.45 | 2 | 96.8% |
| Colon | 0.5730 | 0.5 | 50 | 0.45 | 2 | ❌ 84.3% |
| *Famille Digestive* | *0.6160* | *2.0* | *60* | *0.45* | *5* | *90.6%* |

**Insights:**
- **Bile-duct & Stomach** atteignent l'objectif avec params identiques (beta=1.0, np_thr=0.50, min_dist=3)
- **Stomach min_size=70** — noyaux glandulaires larges, filtre les lymphocytes
- **Colon = problème majeur** (84.3%) — mucine + inflammation. Écart-type 0.179 (le plus élevé)
- Le Colon tire la moyenne famille vers le bas; les 3 autres organes sont tous > 0.65

#### Urologic: Kidney, Bladder, Testis, Ovarian, Uterus, Cervix

| Organe | AJI | Beta | Min Size | NP Thr | Min Dist | Status |
|--------|-----|------|----------|--------|----------|--------|
| **Bladder** | **0.6997** | 2.0 | 20 | 0.50 | 4 | ✅ **102.9%** |
| **Kidney** | **0.6944** | 1.0 | 20 | 0.50 | 1 | ✅ **102.1%** |
| **Cervix** | **0.6872** | 0.5 | 20 | 0.50 | 2 | ✅ **101.1%** |
| Testis | 0.6650 | 2.0 | 50 | 0.50 | 2 | 97.8% |
| Ovarian | 0.6306 | 0.5 | 40 | 0.50 | 3 | 92.7% |
| Uterus | 0.6173 | 1.0 | 10 | 0.50 | 1 | 90.8% |
| *Famille Urologic* | *0.6743* | *0.50* | *30* | *0.45* | *2* | *99.2%* |

**Insights:**
- **3 organes Grade Clinique:** Bladder, Kidney, Cervix
- **Kidney min_distance=1** — le plus agressif, possible grâce à l'injection H-channel
- **np_threshold=0.50** optimal pour toute la famille (haute confiance)
- **Uterus min_size=10** — noyaux très petits, filtrage minimal nécessaire

#### Glandular: Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland

| Organe | AJI | Beta | Min Size | NP Thr | Min Dist | Status |
|--------|-----|------|----------|--------|----------|--------|
| **Adrenal_gland** | **0.7236** | 1.0 | 50 | 0.45 | 4 | ✅ **106.4%** 🏆 |
| Pancreatic | 0.6763 | 1.0 | 20 | 0.50 | 2 | 99.5% |
| Thyroid | 0.6722 | 1.5 | 30 | 0.50 | 2 | 98.9% |
| Breast | 0.6566 | 1.5 | 40 | 0.45 | 2 | 96.6% |
| Prostate | 0.6164 | 1.0 | 10 | 0.50 | 2 | 90.6% |
| *Famille Glandular* | *0.6566* | *0.50* | *50* | *0.40* | *3* | *96.6%* |

**Insights:**
- **Adrenal_gland = RECORD ABSOLU** (0.7236) — dépasse même Liver (0.7207)
- **Pancreatic & Thyroid** très proches de l'objectif (99.5% et 98.9%)
- **min_distance=4** pour Adrenal_gland — tissus bien séparés, noyaux réguliers
- **Prostate min_size=10** — noyaux très petits (similaire à Uterus)

#### Commande Optimisation Organ-Level

```bash
# Phase 1: Exploration rapide (20 samples, 400 configs)
python scripts/evaluation/optimize_watershed_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_{family}_v13_smart_crops_hybrid_fpn_best.pth \
    --family {family} \
    --organ {Organ} \
    --n_samples 20

# Phase 2: Copier-coller la commande générée automatiquement (100 samples, ~81 configs)
```

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

### 2. Modules Partagés OBLIGATOIRES — SINGLE SOURCE OF TRUTH

> **🚫 JAMAIS de duplication de code critique**
>
> Les algorithmes critiques DOIVENT être dans `src/` et importés par tous les scripts.
> **NE JAMAIS copier-coller** une fonction, constante ou liste entre scripts.

**Modules partagés existants:**

| Module | Fonction/Constante | Usage |
|--------|-------------------|-------|
| `src/postprocessing/watershed.py` | `hv_guided_watershed()` | Segmentation instances |
| `src/metrics/ground_truth_metrics.py` | `compute_aji()` | Calcul AJI+ |
| `src/evaluation/instance_evaluation.py` | `run_inference()`, `evaluate_sample()`, `evaluate_batch_with_params()` | Évaluation complète |
| `src/models/organ_head.py` | `PANNUKE_ORGANS`, `OrganPrediction`, `predict_with_ood()` | Prédiction organe |
| `src/preprocessing/__init__.py` | `preprocess_image()`, `HOPTIMUS_MEAN`, `HOPTIMUS_STD` | Normalisation images |
| `src/constants.py` | Toutes les constantes globales | Configuration |

**Import obligatoire:**

```python
# ✅ CORRECT - Single source of truth
from src.postprocessing import hv_guided_watershed
from src.metrics.ground_truth_metrics import compute_aji
from src.evaluation import run_inference, evaluate_batch_with_params
from src.models.organ_head import PANNUKE_ORGANS, OrganPrediction
from src.preprocessing import preprocess_image, HOPTIMUS_MEAN, HOPTIMUS_STD

# ❌ INTERDIT - Duplication de code
def hv_guided_watershed(...):  # Copie locale
def run_inference(...):        # Copie locale
ORGAN_NAMES = ["Adrenal_gland", ...]  # Liste dupliquée
organ_probs = torch.softmax(logits, dim=1)  # Réimplémentation au lieu de predict_with_ood()
MEAN = (0.485, 0.456, 0.406)  # Constante dupliquée
```

**🔍 Comment vérifier AVANT de coder:**

1. **Avant d'écrire une fonction** → `grep -r "def ma_fonction" src/` — existe-t-elle déjà ?
2. **Avant de définir une constante** → `grep -r "MA_CONSTANTE" src/` — est-elle déjà définie ?
3. **Avant de définir une liste** → `grep -r "ORGAN\|FAMILY\|PANNUKE" src/` — existe-t-elle ?
4. **Avant d'appeler un modèle** → Vérifier si une méthode officielle existe (ex: `predict_with_ood()`)

**🔍 Audit périodique (à faire lors des reviews):**

```bash
# Chercher des duplications de listes d'organes
grep -rn "Adrenal_gland.*Bile-duct" --include="*.py" | grep -v "organ_head.py"

# Chercher des réimplémentations de softmax pour OrganHead
grep -rn "softmax.*organ\|organ.*softmax" --include="*.py" | grep -v "organ_head.py"

# Chercher des constantes de normalisation dupliquées
grep -rn "0.707223\|0.485.*0.456" --include="*.py" | grep -v "constants.py\|preprocessing"
```

**Pourquoi:**
- Bug 2025-12-29: scipy.ndimage.label vs skimage.measure.label → -2.8% AJI
- Bug 2025-12-31: softmax brut vs Temperature Scaling → confiance OrganHead 0.66 au lieu de 0.90+

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
| [docs/specs/V14_WSI_TRIAGE_SPEC.md](./docs/specs/V14_WSI_TRIAGE_SPEC.md) | **Spec v14.0** — Triage WSI pyramidal (< 2 min/lame) |
| **[docs/cytology/V14_CYTOLOGY_BRANCH.md](./docs/cytology/V14_CYTOLOGY_BRANCH.md)** | **Spec v14.0 Cytologie** — Architecture en Y, Router, Métriques |
| **[docs/cytology/V14_MASTER_SLAVE_ARCHITECTURE.md](./docs/cytology/V14_MASTER_SLAVE_ARCHITECTURE.md)** | **Architecture Maître/Esclave** — CellPose dual-model, KPIs, Business model |
| **[docs/cytology/V14_PIPELINE_EXECUTION_ORDER.md](./docs/cytology/V14_PIPELINE_EXECUTION_ORDER.md)** | **🔥 Pipeline V14 Ordre d'Exécution** — Clarification CRITIQUE: Séquentiel PUIS Parallèle |
| **[docs/cytology/V14_MACENKO_STRATEGY.md](./docs/cytology/V14_MACENKO_STRATEGY.md)** | **Normalisation Macenko** — Router-Dependent (Cyto ON / Histo OFF) |

---

## Prochaines Étapes

> **Stratégie:** Toujours utiliser les modèles par **famille** (pas de modèles organ-specific).

### 🚧 V14 Cytologie Branch — Architecture Maître/Esclave

**Objectif:** Fusionner pipeline Histologie V13 avec nouveau pipeline Cytologie (Dubai Edition)

**Statut:** 🎯 Architecture validée (2026-01-18) | **Phase 1 en cours** (2026-01-20)

**Changement Architectural Majeur:**
> Suite à analyse approfondie, remplacement de l'approche CellPose `cyto2` unique par **orchestration intelligente de 2 modèles spécialisés** (Maître/Esclave).

#### 🎯 Décision Stratégique Critique (2026-01-20)

> **CellPose inadapté pour SIPaKMeD** — Stratégie duale Phase 1/Phase 2

**Validation Expérimentale:**
```
CellPose sur SIPaKMeD (cellule isolée 168×156):
  Diameter=50: 4 objets détectés au lieu de 1 → Sur-segmentation
  Cause: CellPose optimisé pour tissus (groupes cellulaires), pas cellules isolées
```

**Stratégie Adoptée:**

| Phase | Dataset | Segmentation | But |
|-------|---------|--------------|-----|
| **Phase 1 (Dev)** | SIPaKMeD (cellules isolées) | **Masques GT** | Valider architecture |
| **Phase 2 (Prod)** | Lames réelles (groupes) | **CellPose** | Déploiement clinique |

**Justification:**
- SIPaKMeD ≠ domaine cible (1 cellule isolée vs 100+ cellules/patch)
- H-Optimus = extracteur de features → fonctionne sur tout patch propre
- CellPose sera utilisé en production sur lames réelles (urinaires, thyroïdiennes)

#### Architecture en "Y"

```
INPUT IMAGE → ROUTER → Histo (V13) OU Cyto (Maître/Esclave)
                │
                ├─ Histologie: FPN Chimique (V13 existant)
                │
                └─ Cytologie: NOUVEAU Pipeline Séquentiel
                   ├─ MAÎTRE (nuclei): 100% activation → Features nucléaires
                   └─ ESCLAVE (cyto3): 30% activation → N/C ratio (si requis)
```

#### Gains Mesurés

| Aspect | Cyto2 Seul (Initial) | Maître/Esclave | Gain |
|--------|---------------------|----------------|------|
| **Performance** | 2s/image | 0.5-1.8s (adaptatif) | **2× plus rapide** |
| **GPU Load** | 100% constant | 30-100% adaptatif | **46% économie** |
| **Business** | Forfait unique | 4 packages (€5k-€12k) | **Monetization** |

#### Métriques Critiques Cytologie

**⚠️ Changement Paradigme:** AJI (V13 Histologie) est INADAPTÉ pour cytologie.

**Principe:** **Sensibilité > Précision** (Safety First — Ne jamais rater un cancer)

**KPIs Validation:**

| Métrique | Seuil Cible | Priorité |
|----------|-------------|----------|
| **Sensibilité Malin** | **> 98%** | 🔴 CRITIQUE |
| **FROC (FP/WSI @ 98% sens)** | **< 2.0** | 🔴 CRITIQUE |
| **Cohen's Kappa** | **> 0.80** | 🔴 CRITIQUE |
| IoU Noyau | > 0.85 | 🟡 Important |
| AP50 (COCO) | > 0.90 | 🟡 Important |

**Argument Commercial Dubai:**
> *"Notre système ne rate JAMAIS une cellule anormale (Sensibilité 99%), là où un humain fatigué en rate 5-10%."*

#### Documentation Complète

| Document | Description |
|----------|-------------|
| [V14_CYTOLOGY_BRANCH.md](./docs/V14_CYTOLOGY_BRANCH.md) | Spécifications complètes V14 |
| **[V14_MASTER_SLAVE_ARCHITECTURE.md](./docs/V14_MASTER_SLAVE_ARCHITECTURE.md)** | **Architecture Maître/Esclave (détails techniques)** |

#### Alertes Critiques

- ⚠️ **Macenko:** Régression -4.3% AJI sur V13 → Approche router-dependent (Macenko uniquement Cyto)
- ✅ **Non-régression V13:** AJI Respiratory ≥ 0.6872 OBLIGATOIRE
- 🎯 **Métriques:** Sensibilité > Accuracy (paradigme différent de V13)

#### Questions Validées

✅ **Segmentation Cytoplasme:** CellPose `cyto3` (esclave) activé conditionnellement selon organe
✅ **Décision Trigger:** Config JSON par organe (Urine/Thyroïde → cyto3 auto, Gynéco → skip)
✅ **Gestion Erreurs:** Matching géométrique nuclei→cytoplasme avec cas orphelins (N/C = None, pas de blocage rapport)

#### Questions En Attente

1. Datasets cytologie: Sources pour 70k images (TB-PANDA ✅, Herlev ✅, Urine ❓)
2. Priorité organes: Thyroïde > Col > Urine?
3. Tests validation: Benchmarks CellPose nuclei vs cyto3 consistency

---

### Priorités d'Amélioration V13 (Histologie)

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

## 🔬 Insights Biologiques & R&D Future (2025-12-31)

> **Contexte:** L'optimisation organ-level a révélé des signatures biologiques encodées
> dans les paramètres watershed optimaux. Ces découvertes ouvrent des pistes R&D avancées.

### Découvertes Clés

#### 1. Le Paradoxe du Beta (Liver β=2.0 vs Lung β=0.5)

| Organe | Beta | Morphologie Nucléaire | Explication |
|--------|------|----------------------|-------------|
| **Liver** | 2.0 | Noyaux vésiculeux (clairs) + nucléole central proéminent | Beta élevé → ignore micro-variations NP, se focalise sur gradient HV |
| **Lung** | 0.5 | Noyaux denses, ratio N/C élevé, débris inflammatoires | Beta bas → pondère plus la probabilité NP |

**Conclusion:** Plus un noyau est "vésiculeux" (clair avec point sombre), plus β doit être élevé.
Le foie est le "Gold Standard" de cette morphologie.

#### 2. Signal/Bruit par Tissu

| Tissu | Caractéristique | Impact sur AJI |
|-------|-----------------|----------------|
| **Liver** | Déterministe (organisé, hépatocytes réguliers) | AJI élevé (0.72) |
| **Lung** | Stochastique (inflammatoire, débris, N/C variable) | AJI plus bas (0.65) |

Le gap de 10% AJI reflète la complexité tissulaire intrinsèque, pas uniquement la qualité du modèle.

#### 3. Efficacité de l'Injection H-Channel (Ruifrok)

L'injection du canal Hématoxyline via déconvolution Ruifrok permet:
- `min_distance=2` sans sur-fusion (impossible sans H-channel)
- Séparation précise des noyaux adjacents
- "Lubrifiant géométrique" pour le Watershed

> *"Sans l'injection Hybrid V2, descendre à min_distance=2 causerait une explosion de fusions."*

### Pistes R&D Future

#### Piste 1: Régression Dynamique des Paramètres (Meta-Segmentation)

**Concept:** Utiliser les probabilités OrganHead pour interpoler les paramètres watershed.

```
β_final = P_lung × β_lung + P_liver × β_liver
```

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | Moyenne |
| Impact | Moyen |
| Limitation | OrganHead opère au niveau IMAGE, pas noyau. Interpolation uniforme sur tout le patch. |

#### Piste 2: Watershed Adaptatif par Incertitude ⭐ PRIORITAIRE

**Concept:** Moduler β et min_distance localement selon la carte d'incertitude.

```python
# Pseudo-code
if uncertainty[region] > 0.7:
    beta_local = beta_base * 1.5      # Plus conservateur
    min_dist_local = min_dist_base - 1  # Plus prudent
```

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | **Haute** |
| Impact | **Haut** |
| Avantage | L'incertitude est déjà calculée. Adaptation locale zone par zone. |

#### Piste 3: Test-Time Adaptation (TTA)

**Concept:** Exécuter le Watershed avec N configurations, sélectionner selon métrique de compacité.

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | Basse |
| Impact | Moyen |
| Limitation | Latence × N configs. Critère "compacité" pas toujours corrélé à la justesse. |

#### Piste 4: Watershed "Z-Aware" Multi-Échelle

**Concept:** Deux passes Watershed en parallèle pour gérer la stratification tissulaire (couche basale vs superficielle).

```python
# Passe "Basale" (noyaux petits, denses)
params_basal = {"min_distance": 2, "min_size": 20, "beta": 1.0}

# Passe "Superficielle" (noyaux grands, espacés)
params_superficial = {"min_distance": 5, "min_size": 40, "beta": 2.0}

# Sélection locale basée sur magnitude gradient HV
if hv_gradient_magnitude[region] > threshold:
    use_basal_params()
else:
    use_superficial_params()
```

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | Moyenne |
| Impact | Moyen-Haut |
| Limitation | Risque d'artefacts aux frontières entre zones. Critère de sélection à valider empiriquement. |
| Cas d'usage | **Epidermal** (Skin/HeadNeck) où l'écart-type AJI est élevé (0.12-0.14). |

#### Piste 5: Attention Spatiale via Patch Tokens H-Optimus-0 ⭐

**Concept:** Utiliser les 256 patch tokens (features[:, 5:261, :]) pour pondérer les paramètres Watershed localement.

```python
# Les patch tokens encodent la texture locale (kératine, mélanine, etc.)
patch_features = features[:, 5:261, :]  # (B, 256, 1536)

# MLP léger pour prédire les paramètres locaux
local_params = param_predictor(patch_features)  # → beta, min_size par patch
```

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | Moyenne-Haute |
| Impact | **Haut** |
| Avantage | Les patch tokens encodent DÉJÀ la texture locale. Pas besoin de feature supplémentaire. |
| Cas d'usage | Détection automatique zones kératine → augmente min_size. Zones mélanine → ajuste beta. |

### Investigations Prioritaires

> **⚠️ AVANT d'implémenter les pistes avancées:**
>
> L'écart-type élevé (0.12-0.14) sur Epidermal nécessite une investigation des outliers.
> Certains samples avec AJI < 0.50 pourraient avoir un staining H&E défaillant qui
> "trompe" l'extracteur Ruifrok. Vérifier avant d'investir en R&D avancée.

### Pistes Exploratoires (Risque Variable)

#### Piste 6: Extraction H-Channel Adaptative (Macenko Dynamique)

> **⚠️ ATTENTION: CONTRADICTION AVEC RÉSULTATS V13**
>
> Cette piste **contredit** les résultats documentés: Macenko cause **-4.3% AJI** vs Raw.
> Le conflit Ruifrok/Macenko est établi. Explorer avec précaution.

**Concept:** Estimer les vecteurs de densité optique (OD) par patch au lieu de vecteurs Ruifrok fixes.

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | Moyenne |
| Impact | Incertain |
| **Risque** | **ÉLEVÉ** — Macenko déplace Éosine vers vecteur H → "fantômes" cytoplasme |
| Statut | ❌ Non recommandé sans investigation approfondie |

#### Piste 7: Exploitation des Register Tokens (H-Optimus-0)

**Concept:** Utiliser les 4 register tokens (features[:, 1:5, :]) actuellement ignorés pour pondérer β dynamiquement.

```python
# Register tokens capturent structure globale / type de stroma
register_tokens = features[:, 1:5, :]  # (B, 4, 1536)

# Si stroma fibreux dense détecté → augmente β
beta_modifier = stroma_classifier(register_tokens)
beta_final = beta_base * beta_modifier
```

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | Moyenne |
| Impact | Moyen |
| Avantage | Tokens déjà disponibles, pas de coût d'extraction supplémentaire |
| Limitation | Nécessite recherche sur ce que H-Optimus-0 encode dans ces tokens |

#### Piste 8: FPN Chimique Multispectrale (CLAHE/LBP)

**Concept:** Injecter des canaux de texture (CLAHE, LBP) en plus du canal H dans les couches hautes de la FPN.

```python
# Injection multi-canal dans FPN
h_channel = ruifrok_extract(image)      # Canal Hématoxyline
clahe_channel = apply_clahe(image)       # Contraste local adaptatif
lbp_channel = compute_lbp(image)         # Texture Local Binary Pattern

fpn_input = concat([h_channel, clahe_channel, lbp_channel])
```

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | Basse |
| Impact | Moyen-Haut |
| Limitation | Requiert modification architecture + réentraînement complet |
| Cas d'usage | Tissus haute hétérogénéité (Epidermal, Grade III) |

#### Piste 9: Watershed Itératif par Densité Nucléaire ⭐

**Concept:** Deux passes — estimer densité locale, puis ajuster min_distance.

```python
# Passe 1: Segmentation rapide → estimation densité
quick_seg = watershed(np_pred, hv_pred, min_distance=3)
density = count_nuclei(quick_seg) / area_mm2

# Passe 2: Ajustement local
if density > 2500:  # Amas dense (noyaux/mm²)
    min_distance = 2
elif density < 1000:  # Zone éparse
    min_distance = 5
else:
    min_distance = 3

final_seg = watershed(np_pred, hv_pred, min_distance=min_distance)
```

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | **Haute** |
| Impact | **Haut** |
| Avantage | Implémentable sans réentraînement. Critère densité = métrique pathologique standard. |
| Complémentaire | Combine bien avec Piste 4 (Z-Aware) |

#### Piste 10: NC-based Beta-Switch (Auto-Tuner) ⭐⭐ PRIORITAIRE

**Concept:** Utiliser la branche NC (Nuclear Classification) pour switcher dynamiquement les paramètres watershed selon le contexte tissulaire.

```python
# Extraction du ratio de pixels classés "Connective" (ID 2)
prob_map_nc = outputs['nc']
connective_ratio = (torch.argmax(prob_map_nc, dim=1) == 2).float().mean()

# Switch dynamique des hyper-paramètres
if connective_ratio > 0.40:
    beta = 0.5        # Tissus fibreux → priorité forme
    min_distance = 4  # Évite sur-segmentation noyaux fusiformes
else:
    beta = 2.0        # Tissus épithéliaux → priorité séparation HV
    min_distance = 2  # Agressif sur amas denses
```

| Aspect | Évaluation |
|--------|------------|
| Faisabilité | **Haute** |
| Impact | **Haut** |
| Avantage | NC déjà calculé (gratuit). Adaptation contextuelle automatique. |
| Cas d'usage | **Uterus/Ovarian** (tissus mésenchymateux, AJI < 0.65) |

### Architecture V13 Production Finale (2025-12-31)

```
┌─────────────────────────────────────────────────────────────────┐
│              V13 PRODUCTION ARCHITECTURE FINALE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. MOTEUR PHYSIQUE (Verrouillé ✅)                             │
│     └── Extraction Ruifrok FIXE (Beer-Lambert)                  │
│         • Vecteurs constants: [0.650, 0.704, 0.286]             │
│         • Macenko INTERDIT (cause -4.3% AJI)                    │
│                                                                  │
│  2. OPTIMISATION VISUELLE (Optionnel 🔬)                        │
│     └── CLAHE Post-Ruifrok sur canal H uniquement               │
│         • Préserve intégrité Beer-Lambert                       │
│         • Améliore contraste noyaux vésiculeux                  │
│                                                                  │
│  3. INTELLIGENCE CONTEXTUELLE (Nouveau 🎯)                      │
│     └── NC-based Beta-Switch (Auto-Tuner)                       │
│         • connective_ratio > 0.40 → β=0.5, min_dist=4           │
│         • connective_ratio ≤ 0.40 → β=2.0, min_dist=2           │
│         • Cible: Uterus/Ovarian (tissus mésenchymateux)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Bilan Organ-Level (2025-12-31)

**7 organes "Grade Clinique" (AJI ≥ 0.68):**

| Rang | Organe | Famille | AJI |
|------|--------|---------|-----|
| 🏆 1 | **Adrenal_gland** | Glandular | **0.7236** |
| 2 | Liver | Respiratory | 0.7207 |
| 3 | Bladder | Urologic | 0.6997 |
| 4 | Bile-duct | Digestive | 0.6980 |
| 5 | Kidney | Urologic | 0.6944 |
| 6 | Cervix | Urologic | 0.6872 |
| 7 | Stomach | Digestive | 0.6869 |

**Prochains objectifs:**
- Pancreatic (99.5%), Thyroid (98.9%), Testis (97.8%) — quick wins potentiels
- Colon (84.3%) — nécessite investigation outliers (mucine)
- Uterus (90.8%), Ovarian (92.7%), Prostate (90.6%) — cibles NC Beta-Switch

### Production: Avantage Compétitif

> **⚠️ RAPPEL CRITIQUE (2025-12-25):**
>
> La configuration **Marquage Virtuel Hybride** (Fusion H-Channel via Ruifrok au décodeur)
> est le cœur de l'avantage compétitif V13. Chaque nouveau modèle d'organe DOIT conserver
> cette injection à 100% pour maintenir les scores AJI au-dessus de 0.68.

---

## Références

- H-optimus-0: https://huggingface.co/bioptimus/H-optimus-0
- HoVer-Net: Graham et al., Medical Image Analysis 2019
- PanNuke: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
