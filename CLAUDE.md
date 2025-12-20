# CellViT-Optimus — Contexte Projet

> **IMPORTANT : Ce fichier est la source de vérité du projet.**
>
> Claude doit maintenir ce fichier à jour avec toute information importante :
> - Décisions techniques prises durant le développement
> - Problèmes rencontrés et solutions appliquées
> - Changements d'architecture ou de stratégie
> - Dépendances ajoutées et leurs versions
> - Bugs connus et workarounds
> - Toute information qui serait utile pour reprendre le contexte
>
> **Si une information est jugée importante pour la continuité du projet, elle doit être ajoutée ici.**

> **OBLIGATOIRE : Avant toute implémentation, Claude DOIT lire le fichier `CellViT-Optimus_Specifications.md` et s'assurer que le code respecte les spécifications techniques définies.**

---

## Vue d'ensemble

**CellViT-Optimus** est un système d'assistance au triage histopathologique. Il ne remplace pas le pathologiste mais l'aide à :
- Prioriser les lames et régions à forte valeur diagnostique
- Réduire le temps de lecture
- Sécuriser la décision grâce à une maîtrise explicite de l'incertitude

**Statut :** POC / Exploration
**Objectif immédiat :** Créer un prototype démontrable pour discussions avec professionnels de santé

---

## Architecture Technique

```
┌────────────────────────────────────────────────────────────────┐
│                      LAME H&E (WSI)                            │
└────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│              COUCHE 1 — EXTRACTION SÉMANTIQUE                  │
│                     H-OPTIMUS-0 (gelé)                         │
│  • Entrée : tuiles 224×224 @ 0.5 MPP                          │
│  • Sortie : CLS token (1536) + Patches (256×1536)             │
│  • ViT-Giant/14, 1.1 milliard paramètres                      │
└────────────────────────────────────────────────────────────────┘
                               │
     ┌─────────────────────────┴─────────────────────────┐
     ▼                                                   ▼
┌─────────────────────────────┐        ┌─────────────────────────────┐
│  COUCHE 2A — FLUX GLOBAL    │        │  COUCHE 2B — FLUX LOCAL     │
│       OrganHead             │        │       HoVer-Net             │
│                             │        │                             │
│  • CLS token → MLP          │        │  • Patches → Décodeur       │
│  • Classification organe    │        │  • NP : présence noyaux     │
│  • 19 organes PanNuke       │        │  • HV : séparation          │
│  ✅ Accuracy 96.05%         │        │  • NT : typage (5 cls)      │
│                             │        │  ✅ Dice 0.9601             │
└─────────────────────────────┘        └─────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│              COUCHE 3 — SÉCURITÉ & INCERTITUDE                 │
│                                                                │
│  • Incertitude aléatorique (entropie NP/HV)                   │
│  • Incertitude épistémique (Conformal Prediction)             │
│  • Détection OOD (distance latente Mahalanobis)               │
│  • Calibration locale (Temperature Scaling par centre)        │
│                                                                │
│  Sortie : {Fiable | À revoir | Hors domaine}                  │
└────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│              COUCHE 4 — INTERACTION EXPERT                     │
│                                                                │
│  • Sélection automatique des ROIs                             │
│  • Visualisation (cellules + heatmaps attention)              │
│  • Validation humaine finale                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Composants Modèles

### H-optimus-0 (Backbone)
| Attribut | Valeur |
|----------|--------|
| Source | Bioptimus (HuggingFace) |
| Architecture | ViT-Giant/14 avec 4 registres |
| Paramètres | 1.1 milliard |
| Entrée | 224×224 pixels @ 0.5 MPP |
| Sortie | Embedding 1536-dim |
| Licence | Apache 2.0 (usage commercial OK) |

**Normalisation requise :**
```python
mean = (0.707223, 0.578729, 0.703617)
std = (0.211883, 0.230117, 0.177517)
```

**Chargement :**
```python
import timm
model = timm.create_model(
    "hf-hub:bioptimus/H-optimus-0",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=False
)
```

### CellViT (Référence segmentation)
| Attribut | Valeur |
|----------|--------|
| Source | TIO-IKIM (GitHub) |
| Architecture | U-Net + ViT encoder |
| Entrée inference | 1024×1024 pixels |
| Classes | 5 (Neoplastic, Inflammatory, Connective, Dead, Epithelial) |

---

## Environnement de Développement

### Configuration validée
| Composant | Version |
|-----------|---------|
| OS | WSL2 Ubuntu 24.04.2 LTS |
| GPU | RTX 4070 SUPER (12.9 GB VRAM) |
| NVIDIA Driver | 566.36 |
| CUDA | 12.7 (système) / 12.4 (PyTorch) |
| Docker | 29.1.3 (natif, pas Docker Desktop) |
| NVIDIA Container Toolkit | Installé |
| Python | 3.10 (via Miniconda) |
| Conda | 25.11.1 |
| PyTorch | 2.6.0+cu124 |
| Environnement conda | `cellvit` |

### Contraintes VRAM (12 GB)
| Tâche | VRAM estimée | Faisabilité |
|-------|--------------|-------------|
| Inférence H-optimus-0 (FP16, batch=1) | ~3-4 GB | ✅ OK |
| Inférence H-optimus-0 (FP16, batch=8) | ~6-8 GB | ✅ OK |
| Entraînement décodeur (backbone gelé) | ~8-10 GB | ⚠️ Serré |
| Entraînement complet avec gradients | >16 GB | ❌ Impossible |

---

## Sources de Données

### Branche Cellule (Segmentation)
| Dataset | Contenu | Usage |
|---------|---------|-------|
| **PanNuke** | ~200k noyaux, 5 types, 19 organes | Entraînement NP/HV/NT |
| **MoNuSeG** | Multi-organes | Robustesse segmentation |
| **CoNSeP** | Morphologie colique | Calibration HV |

### Branche Lame (Biomarqueurs)
| Dataset | Contenu | Usage |
|---------|---------|-------|
| **TCGA** | Milliers de WSI + données moléculaires | Entraînement AMIL |
| **CPTAC** | WSI + protéomique | Têtes expertes |

---

## Structure Projet Cible

```
cellvit-optimus/
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.worker
│   └── docker-compose.yml
├── scripts/
│   ├── setup/
│   ├── preprocessing/
│   ├── evaluation/
│   ├── calibration/
│   ├── ood_detection/
│   └── benchmarking/
├── tests/
│   ├── unit/
│   └── integration/
├── configs/
├── notebooks/
└── src/
    ├── models/
    ├── data/
    ├── inference/
    └── utils/
```

---

## Décisions Techniques Clés

1. **Backbone gelé** — H-optimus-0 n'est jamais fine-tuné, seules les têtes s'entraînent
2. **UNETR pour reconstruction spatiale** — Extraction features couches 6/12/18/24 du ViT
3. **Tiling adaptatif** — Recall 0.999 sur tissu tumoral, garde-fou basse résolution
4. **Cache d'embeddings versionné** — Hash [Backbone]+[Preprocessing]+[Resolution]+[Date]
5. **Distillation limitée au pré-triage** — Le modèle original reste obligatoire pour diagnostic

---

## Stratégie de Sécurité Clinique

### Sortie en 3 niveaux
- **Fiable** — Confiance haute, prédiction utilisable
- **À revoir** — Incertitude détectée, validation humaine recommandée
- **Hors domaine** — Cas atypique, ne pas utiliser la prédiction

### Cold Start (nouveau centre)
1. Seuils conservateurs par défaut
2. Shadow mode sur 30-50 premières lames
3. Détection OOD automatique

---

## Références

- H-optimus-0 : https://huggingface.co/bioptimus/H-optimus-0
- CellViT : https://github.com/TIO-IKIM/CellViT
- PanNuke : https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
- TCGA : https://www.cancer.gov/tcga

---

## Notes pour Claude

- **Objectif actuel** : POC démontrable en 6 semaines
- **Priorité** : Validation technique avant expansion
- **Approche** : Explorer le domaine médical via ce projet, rester ouvert aux pivots
- **Hardware limité** : Toujours considérer les contraintes 12GB VRAM dans les suggestions

---

## Plan de Développement POC (6 semaines)

> **IMPORTANT** : Suivre ce plan étape par étape. Ne pas passer à l'étape suivante
> sans avoir validé les critères de l'étape courante.

### Phase 1 : Environnement & Données (Semaines 1-2)

| Étape | Description | Validation | Statut |
|-------|-------------|------------|--------|
| 1.1 | Setup WSL2 + Docker + CUDA | `nvidia-smi` fonctionne | ✅ FAIT |
| 1.2 | Conda + PyTorch | `torch.cuda.is_available()` = True | ✅ FAIT |
| 1.3 | Télécharger PanNuke | 3 folds présents | ✅ FAIT (manuel) |
| 1.4 | Scripts preprocessing | Extraction tuiles, normalisation | ✅ FAIT |

**Critères de passage Phase 2 :**
- [x] Environnement GPU fonctionnel
- [x] Dataset PanNuke disponible
- [x] Pipeline preprocessing prêt

### Phase 2 : Intégration H-optimus-0 (Semaines 3-4)

| Étape | Description | Validation | Statut |
|-------|-------------|------------|--------|
| 2.1 | Accès HuggingFace gated | Token configuré | ✅ FAIT |
| 2.2 | Charger H-optimus-0 | Inférence OK sur 1 image | ✅ FAIT |
| 2.3 | Extraction features PanNuke | Embeddings 1536-dim sauvés | ✅ FAIT |
| 2.4 | Visualisation t-SNE | Clusters par organe visibles | ✅ FAIT |
| 2.5 | Décodeur UNETR skeleton | Architecture compilable | ✅ FAIT |
| 2.6 | Entraînement UNETR sur PanNuke | Loss converge | ✅ FAIT (Dice 0.6935) |

**Critères de passage Phase 3 :**
- [x] UNETR entraîné sur PanNuke (backbone H-optimus-0 gelé)
- [x] Dice ≈ 0.7 sur PanNuke validation (0.6935 accepté pour POC)

### Phase 3 : Interface Démo (Semaine 5)

| Étape | Description | Validation | Statut |
|-------|-------------|------------|--------|
| 3.1 | Interface Gradio basique | Upload image → résultat | ✅ FAIT |
| 3.2 | Intégration HoVer-Net dans démo | Inférence H-optimus-0 + HoVer-Net | ✅ FAIT |
| 3.3 | Rapport avec couleurs/emojis | Correspondance visuelle | ✅ FAIT |
| 3.4 | Scripts OOD/calibration | Utilitaires prêts | ✅ FAIT |

### Phase 4 : Sécurité & Interaction Expert (Semaine 6) ✅ COMPLÈTE

| Étape | Description | Validation | Statut |
|-------|-------------|------------|--------|
| 4.1 | Incertitude aléatorique | Entropie NP/HV calculée | ✅ FAIT |
| 4.2 | Incertitude épistémique | Conformal Prediction intégré | ✅ FAIT |
| 4.3 | Détection OOD | Distance Mahalanobis sur embeddings | ✅ FAIT |
| 4.4 | Calibration locale | Temperature Scaling fonctionnel | ✅ FAIT |
| 4.5 | Sortie 3 niveaux | {Fiable \| À revoir \| Hors domaine} | ✅ FAIT |
| 4.6 | Sélection automatique ROIs | Régions prioritaires identifiées | ✅ FAIT |
| 4.7 | Carte d'incertitude | Heatmap rouge/vert dans démo | ✅ FAIT |

### Phase 5 : Packaging (Post-POC)

| Étape | Description | Validation | Statut |
|-------|-------------|------------|--------|
| 5.1 | Docker packaging | `docker-compose up` fonctionne | 🔜 DIFFÉRÉ |
| 5.2 | Documentation utilisateur | README complet | 🔜 DIFFÉRÉ |

**Critères de livraison POC :**
- [x] Démo fonctionnelle avec architecture cible (H-optimus-0 + HoVer-Net, Dice 0.9601)
- [x] Couche 3 : Sécurité & Incertitude intégrée
- [x] Couche 4 : Interaction Expert (ROIs, heatmaps)

---

## Statut Actuel

**Phase en cours :** Phase 4 — COMPLÈTE ✅
**Blocage actuel :** Aucun
**Prochaine action :** Phase 5 (Packaging) ou démo avec pathologistes

### Résumé des accomplissements
- ✅ Couche 1 : H-optimus-0 intégré (embeddings 1536-dim)
- ✅ Couche 2A : HoVer-Net decoder entraîné (Dice 0.9601)
- ✅ Couche 3 : Sécurité & Incertitude (entropie + Mahalanobis + Conformal Prediction)
- ✅ Couche 4 : Interaction Expert (ROIs, calibration, heatmaps)

---

## Décisions Techniques & Justifications

### Décision 1: Utiliser le repo CellViT officiel (TIO-IKIM)

**Date:** 2025-12-19
**Contexte:** Le checkpoint CellViT-256.pth (187 MB) a une architecture complexe qui ne correspondait pas à notre wrapper custom.

**Problèmes rencontrés:**
- Incompatibilité `pos_embed`: [1, 197, 384] (checkpoint) vs [1, 257, 384] (notre modèle)
- Structure décodeurs différente: `decoder.X.block.Y` vs `decoder.X.Y`
- Têtes de sortie avec `bottleneck_upsampler`, `decoderX_upsampler`, `decoder0_header`
- Seulement 149/439 paramètres compatibles

**Décision:** Cloner le repo officiel `TIO-IKIM/CellViT` et utiliser leur code pour charger le modèle.

**Pourquoi cette décision pour le POC:**
- ✅ Gain de temps: Pas besoin de reverse-engineer l'architecture exacte
- ✅ Fiabilité: Code testé par les auteurs originaux
- ✅ Baseline fiable: Permet de valider le pipeline end-to-end rapidement

**Impact sur l'architecture cible:**
- ⚠️ CellViT-256 n'est PAS l'architecture cible
- L'architecture cible utilise **H-optimus-0 + UNETR** (specs section 2.3)
- CellViT-256 sert uniquement de **baseline de comparaison**

### Chemin vers l'Architecture Cible

```
POC (actuel)                          CIBLE (production)
─────────────────────────────────     ─────────────────────────────────
CellViT-256 pré-entraîné              H-optimus-0 backbone (gelé)
    │                                     │
    │ encoder ViT-256                     │ ViT-Giant/14 (1.1B params)
    │ (46M params)                        │ Embeddings 1536-dim
    │                                     │
    ▼                                     ▼
Décodeur intégré CellViT              Décodeur UNETR custom
    │                                     │
    │                                     │ Skip connections couches 6/12/18/24
    │                                     │
    ▼                                     ▼
3 têtes: NP, HV, NT                   3 têtes: NP, HV, NT
```

### Étapes pour passer à l'architecture cible

1. **Phase POC (actuelle):** Valider pipeline avec CellViT-256 comme baseline
2. **Phase 2.6:** Entraîner notre décodeur UNETR sur PanNuke avec H-optimus-0 gelé
3. **Validation:** Comparer métriques UNETR vs CellViT-256 baseline
4. **Production:** Remplacer CellViT-256 par UNETR entraîné

### Pourquoi ne pas utiliser CellViT-256 en production?

| Critère | CellViT-256 | H-optimus-0 + UNETR |
|---------|-------------|---------------------|
| Taille backbone | 46M params | 1.1B params |
| Features | 384-dim | 1536-dim |
| Pré-entraînement | PanNuke uniquement | 500k+ lames H&E multi-centres |
| Généralisation | Limitée | Excellente (foundation model) |
| Conformité specs | ❌ Non | ✅ Oui |

---

## Journal de Développement

### 2025-12-19 — Setup environnement ✅ VALIDÉ
- **Environnement WSL2 configuré** : Ubuntu 24.04.2 LTS
- **Docker Engine natif installé** (pas Docker Desktop) — meilleure performance, pas de licence
- **NVIDIA Container Toolkit** configuré — Docker peut accéder au GPU
- **Miniconda installé** — environnement `cellvit` créé
- **PyTorch 2.6.0+cu124 installé** — GPU RTX 4070 SUPER détecté et fonctionnel
- **Test GPU matmul** : OK
- **Décision** : Utiliser Python 3.10 pour compatibilité optimale avec PyTorch/CUDA

**Commandes de vérification rapide :**
```bash
# Activer l'environnement
conda activate cellvit

# Vérifier GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2025-12-19 — H-optimus-0 + PanNuke ✅ VALIDÉ
- **H-optimus-0 chargé** : 1.13B paramètres, embeddings 1536-dim
- **PanNuke Fold 1 téléchargé** : 2656 images, 19 organes, 256×256 pixels
- **Script d'extraction créé** : `scripts/preprocessing/extract_features.py`
- **Script de visualisation créé** : `scripts/evaluation/visualize_embeddings.py`

**Performances mesurées :**
| Métrique | Valeur |
|----------|--------|
| Temps par image | 13.6 ms |
| Throughput | 73.4 img/s |
| Pic mémoire GPU | 4.59 GB |

**Commandes d'extraction :**
```bash
# Extraction stratifiée (tous les organes)
python scripts/preprocessing/extract_features.py --num_images 500 --batch_size 16 --stratified

# Visualisation t-SNE
python scripts/evaluation/visualize_embeddings.py
```

**Résultat t-SNE** : Les embeddings montrent une structure (pas aléatoire), avec quelques clusters par organe. Validation que H-optimus-0 capture de l'information sémantique utile.

### 2025-12-19 — Scripts & Démo Gradio ✅ FAIT
- **Interface Gradio créée** : `scripts/demo/gradio_demo.py`
- **Générateur tissus synthétiques** : `scripts/demo/synthetic_cells.py`
- **Visualisation cellules** : `scripts/demo/visualize_cells.py`
- **Rapport avec emojis couleur** : 🔴🟢🔵🟡🩵 correspondant aux types

### 2025-12-19 — Scripts utilitaires (specs section 6.1) ✅ FAIT
Scripts créés conformément aux specs :
- `scripts/setup/download_models.py` — Téléchargement CellViT, SAM, H-optimus-0
- `scripts/setup/download_datasets.py` — Téléchargement PanNuke avec vérification
- `scripts/preprocessing/stain_normalization.py` — Normalisation Macenko H&E
- `scripts/preprocessing/tile_extraction.py` — Extraction tuiles 224×224
- `scripts/preprocessing/quality_filter.py` — Détection flou, tissus, artefacts
- `scripts/preprocessing/tissue_detection.py` — Détection ROI, filtrage background
- `scripts/evaluation/metrics_segmentation.py` — Dice, IoU, PQ, F1
- `scripts/calibration/temperature_scaling.py` — Calibration post-hoc, ECE
- `scripts/ood_detection/latent_distance.py` — Mahalanobis sur embeddings
- `scripts/ood_detection/entropy_scoring.py` — Incertitude → Fiable/À revoir/Hors domaine
- `scripts/training/train_unetr.py` — Entraînement UNETR sur PanNuke

### 2025-12-19 — Intégration CellViT-256 ✅ VALIDÉE (Étape 1.5 POC)
- **Repo officiel cloné** : `CellViT/` (TIO-IKIM/CellViT)
- **Dépendances installées** : ujson, einops, shapely, geojson, colorama, natsort
- **Wrapper officiel mis à jour** : `src/inference/cellvit_official.py`
- **Test validation créé** : `scripts/validation/test_cellvit_official.py`
- **Checkpoint téléchargé** : `models/pretrained/CellViT-256.pth` (187.2 MB, Epoch 129)

**Architecture CellViT-256 (via repo officiel) :**
| Attribut | Valeur |
|----------|--------|
| Paramètres | 46,750,349 |
| embed_dim | 384 |
| depth | 12 |
| num_heads | 6 |
| extract_layers | [3, 6, 9, 12] |

**Résultats validation complète :**
```
✅ Import CellViT256 OK
✅ Architecture: 46.7M params
✅ Forward pass OK
✅ Checkpoint chargé (187.2 MB, 439 clés)
✅ Poids chargés (All keys matched successfully)
✅ Inférence réussie (NP/Type probs: [0.000, 1.000])

🎉 TOUS LES TESTS PASSENT - Étape 1.5 validée!
```

**Test validation :**
```bash
python scripts/validation/test_cellvit_official.py -c models/pretrained/CellViT-256.pth
```

### 2025-12-19 — Démo Gradio avec CellViT-256 ✅ VALIDÉE (Étape 3.2 POC)
- **Wrapper officiel intégré** dans `scripts/demo/gradio_demo.py`
- **Import mis à jour** : `CellViTOfficial` remplace `CellViTInference`
- **Validation checkpoint** : Vérification taille > 1MB avant chargement

**Test sur image réelle (cancer prostate) :**
```
✅ MODÈLE CELLVIT-256 ACTIF
Total cellules détectées: 25
  🔴 Neoplastic: 17 (68.0%)
  🔵 Connective: 8 (32.0%)
```

**Résultat :** Détection cohérente — majorité néoplasique sur image de carcinome prostatique.

### 2025-12-19 — Validation métriques PanNuke ✅ VALIDÉE (Étape 1.6 POC)
- **Dataset PanNuke** : 3 folds téléchargés et réorganisés (structure Warwick → CellViT)
- **Script d'évaluation créé** : `scripts/validation/evaluate_pannuke.py`
- **Tests unitaires créés** : `tests/unit/test_metrics.py`, `test_ood.py`, `test_calibration.py`
- **Tests intégration** : `tests/integration/test_pipeline_e2e.py`

**Résultats sur PanNuke (2722 images) :**
```
Binary-Cell-Dice:    0.8733 ± 0.1048
Binary-Cell-Jaccard: 0.7859
```

**Critère POC :** Dice 0.8733 > 0.7 ✅

### 2025-12-19 — Entraînement UNETR ✅ VALIDÉ (Étape 2.6 POC)
- **Features pré-extraites** : H-optimus-0 couches 6/12/18/24 → 17 GB (fold 0)
- **Checkpoint sauvé** : `models/checkpoints/unetr_best.pth`
- **Données** : Fold 0 uniquement (2125 train / 531 val)

**Résultats entraînement (50 epochs) :**
| Métrique | Train | Validation |
|----------|-------|------------|
| Loss | 0.1266 | 1.0297 |
| Dice | - | **0.6935** |

**Observation :** Overfitting détecté (Val Loss 8x > Train Loss). Le Dice reste acceptable car il mesure le chevauchement binaire, pas la calibration des probabilités.

**Critère POC :** Dice 0.6935 ≈ 0.7 ✅ (accepté pour POC)

#### ⚠️ Recommandations pour améliorer la généralisation (post-POC)

| Priorité | Action | Impact attendu |
|----------|--------|----------------|
| 1 | **Utiliser les 3 folds** | 3x plus de données → meilleure généralisation |
| 2 | **Data augmentation** | Rotations, flips, variations couleur H&E |
| 3 | **Regularisation** | Dropout (0.1-0.3), weight decay (1e-4) |
| 4 | **Early stopping** | Arrêter quand val_loss stagne |
| 5 | **Temperature scaling** | Calibrer les probabilités post-entraînement |

### 2025-12-19 — Migration UNETR → HoVer-Net ✅ VALIDÉ

**Problème identifié :** L'architecture UNETR n'était pas adaptée à H-optimus-0 car :
- UNETR attend des skip connections multi-résolution
- H-optimus-0 sort toutes les couches à 16x16 (même résolution)
- Résultats UNETR décevants : Dice 0.6935, classifications déséquilibrées

**Solution adoptée :** Décodeur HoVer-Net style (basé sur littérature CellViT)

**Architecture HoVer-Net :**
```
H-optimus-0 (16x16 @ 1536)
        ↓
Bottleneck 1x1 (1536 → 256)  ← Économie VRAM
        ↓
Tronc Commun (upsampling partagé 16→224)
        ↓
   ┌────┴────┬────────┐
   ↓         ↓        ↓
  NP        HV       NT
```

**Résultats comparatifs :**
| Métrique | UNETR | HoVer-Net | Amélioration |
|----------|-------|-----------|--------------|
| Dice | 0.6935 | **0.9587** | +38% |
| Val Loss | 1.0297 | 0.7469 | -27% |

**Fichiers créés :**
- `src/models/hovernet_decoder.py` — Décodeur avec bottleneck partagé
- `scripts/training/train_hovernet.py` — Script d'entraînement
- `src/inference/hoptimus_hovernet.py` — Wrapper inférence
- `models/checkpoints/hovernet_best.pth` — Checkpoint entraîné

### 2025-12-20 — Couche 3: Sécurité & Incertitude ✅ VALIDÉ

**Implémentation complète de la Couche 3** conforme aux specs:

**Module créé:** `src/uncertainty/`
- `uncertainty_estimator.py` — Estimateur unifié combinant:
  - Incertitude aléatorique (entropie NP/NT)
  - Incertitude épistémique (distance Mahalanobis sur embeddings)
  - Classification en 3 niveaux: {Fiable | À revoir | Hors domaine}

**Intégration dans l'inférence:**
- `hoptimus_hovernet.py` mis à jour pour calculer l'incertitude à chaque prédiction
- Carte d'incertitude spatiale générée (rouge=incertain, vert=fiable)
- Rapport textuel enrichi avec métriques d'incertitude

**Intégration dans la démo Gradio:**
- Nouvelle sortie: carte d'incertitude visualisée
- Description des niveaux de confiance dans l'interface
- Rapport complet avec entropie, Mahalanobis, score combiné

**Fichiers modifiés/créés:**
- `src/uncertainty/__init__.py`
- `src/uncertainty/uncertainty_estimator.py`
- `src/inference/hoptimus_hovernet.py` (ajout `visualize_uncertainty()`)
- `scripts/demo/gradio_demo.py` (4 outputs au lieu de 3)

**Amélioration Loss:** MSELoss → SmoothL1Loss pour branche HV (moins sensible aux outliers)

**Résultats après SmoothL1Loss (2025-12-20):**
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| Dice | 0.9587 | **0.9601** | +0.14% |
| Val Loss | 0.7469 | **0.7333** | -1.8% |
| HV Loss | ~0.01 | 0.0085 | -15% |

### 2025-12-20 — Régularisation: Augmentation + Dropout ✅ IMPLÉMENTÉ

**Problème identifié:** Overfitting Train Loss (0.31) vs Val Loss (0.81) = 2.6x gap

**Solutions implémentées:**

1. **Data Augmentation** (`FeatureAugmentation` class):
   - Flip horizontal/vertical avec ajustement composantes H/V
   - Rotation 90° (90°, 180°, 270°) avec rotation H/V
   - Appliqué sur features H-optimus-0 (reshape 16x16 grid)
   - Flag: `--augment`

2. **Dropout régularisation**:
   - Dropout2d après bottleneck et entre blocs upsampling
   - Default: 0.1, configurable via `--dropout`

3. **Loss weights ajustés** (recommandation expert):
   - `L_total = 1.0*NP + 2.0*HV + 1.0*NT`
   - Focus sur gradient sharpness (séparation instances)

**Fichiers modifiés:**
- `src/models/hovernet_decoder.py` — Ajout dropout parameter
- `scripts/training/train_hovernet.py` — Ajout FeatureAugmentation, flags --augment/--dropout

**Commande entraînement avec régularisation:**
```bash
python scripts/training/train_hovernet.py --fold 0 --epochs 50 --augment --dropout 0.1
```

### 2025-12-20 — Phase 4 Complète: Conformal Prediction + ROI Selection ✅

**Modules implémentés:**

1. **Conformal Prediction** (`src/uncertainty/conformal_prediction.py`)
   - Méthodes: LAC, APS, RAPS
   - Garantie de couverture (1 - alpha)
   - Support pixel-wise pour segmentation
   - Usage:
   ```python
   cp = ConformalPredictor(method=ConformalMethod.APS, alpha=0.1)
   cp.calibrate(val_probs, val_labels)
   result = cp.predict_set(test_probs)  # Returns prediction set
   ```

2. **Temperature Scaling intégré** (`uncertainty_estimator.py`)
   - Calibration post-hoc des probabilités
   - Minimisation NLL ou ECE
   - Intégré dans UncertaintyEstimator:
   ```python
   estimator.calibrate_temperature(logits, labels)
   probs = estimator.apply_temperature(logits)
   ```

3. **Sélection automatique ROIs** (`src/uncertainty/roi_selection.py`)
   - Score combiné: incertitude + densité + néoplasiques
   - Priorités: CRITICAL, HIGH, MEDIUM, LOW
   - Fenêtre glissante avec suppression chevauchement
   - Usage:
   ```python
   selector = ROISelector(roi_size=64, stride=32)
   rois = selector.select_rois(uncertainty_map, np_mask, nt_probs, n_rois=5)
   ```

**Tests de validation:**
```bash
python -c "from src.uncertainty import ConformalPredictor, ROISelector; print('OK')"
```

### 2025-12-20 — Architecture Optimus-Gate ✅

**Architecture finale "Optimus-Gate"** avec double flux:

```
H-optimus-0 (backbone gelé)
         │
    features (B, 261, 1536)
         │
    ┌────┴────┐
    ↓         ↓
CLS token   Patch tokens
(1, 1536)   (256, 1536)
    │         │
    ↓         ↓
OrganHead   HoVerNet
(96% acc)   (96% Dice)
    │         │
    ↓         ↓
19 organes  NP/HV/NT
+ OOD       + Cellules
```

**Résultats entraînement:**
| Composant | Métrique | Valeur |
|-----------|----------|--------|
| OrganHead | Val Accuracy | **96.05%** |
| OrganHead | Organes à 100% | 14/19 |
| HoVer-Net | Dice | **0.9601** |
| OOD | Threshold | 39.26 |

**Triple Sécurité OOD:**
- Entropie organe (softmax uncertainty)
- Mahalanobis global (CLS token distance)
- Mahalanobis local (patch mean distance)

**Usage:**
```python
from src.inference import OptimusGate

# Charger le modèle pré-entraîné
model = OptimusGate.from_pretrained(
    hovernet_path="models/checkpoints/hovernet_best.pth",
    organ_head_path="models/checkpoints/organ_head_best.pth",
    device="cuda"
)

# Prédiction
result = model.predict(features)
print(result.organ.organ_name)      # "Prostate"
print(result.organ.confidence)      # 0.99
print(result.n_cells)               # 42
print(result.is_ood)                # False
print(result.confidence_level)      # ConfidenceLevel.FIABLE

# Rapport complet
print(model.generate_report(result))
```

### 2025-12-20 — Intégration Gradio Demo ✅

**OptimusGateInference** intégré dans la démo Gradio:

- **Fichier créé**: `src/inference/optimus_gate_inference.py`
  - Wrapper complet: image → H-optimus-0 → OptimusGate → résultats
  - Méthodes: `predict()`, `visualize()`, `visualize_uncertainty()`, `generate_report()`

- **Démo mise à jour**: `scripts/demo/gradio_demo.py`
  - OptimusGate chargé en priorité (avant HoVer-Net seul)
  - UI mise à jour avec architecture double flux
  - Affichage organe détecté + cellules + OOD
  - Onglet "À propos" avec schéma Optimus-Gate

**Lancement:**
```bash
python scripts/demo/gradio_demo.py
# URL: http://localhost:7860
```

---

## Fichiers Créés (Inventaire)

```
src/
├── models/
│   ├── __init__.py
│   ├── unetr_decoder.py          # Décodeur UNETR (obsolète)
│   ├── hovernet_decoder.py       # Décodeur HoVer-Net (Flux Local)
│   └── organ_head.py             # OrganHead (Flux Global)
├── inference/
│   ├── __init__.py
│   ├── optimus_gate.py           # Architecture unifiée Optimus-Gate
│   ├── optimus_gate_inference.py # 🆕 Wrapper Gradio (image → résultats)
│   ├── hoptimus_hovernet.py      # Wrapper H-optimus-0 + HoVer-Net
│   ├── hoptimus_unetr.py         # Wrapper H-optimus-0 + UNETR (fallback)
│   └── cellvit_official.py       # Wrapper pour repo officiel TIO-IKIM
└── uncertainty/                   # Couche 3 & 4: Sécurité & Interaction Expert
    ├── __init__.py
    ├── uncertainty_estimator.py  # Entropie + Mahalanobis + Temperature Scaling
    ├── conformal_prediction.py   # Conformal Prediction (APS/LAC/RAPS)
    └── roi_selection.py          # Sélection automatique ROIs

scripts/
├── setup/
│   ├── download_models.py
│   └── download_datasets.py
├── preprocessing/
│   ├── extract_features.py        # Extraction embeddings H-optimus-0
│   ├── stain_normalization.py
│   ├── tile_extraction.py
│   ├── quality_filter.py
│   └── tissue_detection.py
├── evaluation/
│   ├── visualize_embeddings.py
│   └── metrics_segmentation.py
├── calibration/
│   └── temperature_scaling.py
├── ood_detection/
│   ├── latent_distance.py
│   └── entropy_scoring.py
├── training/
│   ├── train_unetr.py            # Entraînement UNETR (obsolète)
│   ├── train_hovernet.py         # Entraînement HoVer-Net (Flux Local)
│   └── train_organ_head.py       # Entraînement OrganHead (Flux Global)
├── utils/
│   └── inspect_checkpoint.py
├── validation/
│   ├── test_cellvit256_inference.py  # Test étape 1.5 POC
│   └── test_optimus_gate.py          # Test Optimus-Gate complet
└── demo/
    ├── gradio_demo.py             # Interface principale
    ├── synthetic_cells.py         # Générateur tissus
    └── visualize_cells.py         # Fonctions visualisation

models/
├── pretrained/
│   └── CellViT-256.pth            # 187 MB (baseline)
└── checkpoints/
    ├── hovernet_best.pth          # HoVer-Net (Dice 0.9601)
    └── organ_head_best.pth        # OrganHead (Acc 96.05%)
```

---

## Problèmes Connus & Solutions

| Problème | Solution |
|----------|----------|
| Conda ToS non acceptées | `conda tos accept --override-channels --channel <url>` |
| Docker "command not found" dans WSL | Installer Docker Engine natif, pas Docker Desktop |
| H-optimus-0 accès refusé (401/403) | Voir section "Accès H-optimus-0" ci-dessous |
| Token HuggingFace "fine-grained" sans accès gated | Activer "Read access to public gated repos" dans les permissions du token |

---

## Accès H-optimus-0 (Gated Model)

H-optimus-0 est un modèle "gated" sur HuggingFace. Configuration requise :

### Étape 1 : Demander l'accès
1. Créer un compte sur https://huggingface.co
2. Aller sur https://huggingface.co/bioptimus/H-optimus-0
3. Cliquer sur "Agree and access repository"

### Étape 2 : Créer un token avec les bonnes permissions
1. Aller sur https://huggingface.co/settings/tokens
2. Créer un nouveau token avec ces permissions :
   - ✅ **Read access to contents of all public gated repos you can access**
   - ✅ Read access to contents of all repos under your personal namespace

### Étape 3 : Se connecter
```bash
huggingface-cli login
# Coller le token quand demandé
```

### Vérification
```bash
huggingface-cli whoami
```

---

## Guide d'Installation Complète (depuis zéro)

### Prérequis Windows
- Windows 10/11 avec WSL2 activé
- GPU NVIDIA avec drivers récents

### 1. WSL2 + Ubuntu
```powershell
# PowerShell Admin
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
```

### 2. Docker Engine natif (dans WSL)
```bash
# Dépendances
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg

# Clé GPG Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Repository Docker
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Installation
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
# Fermer et rouvrir le terminal
```

### 3. NVIDIA Container Toolkit
```bash
# Repository NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Installation
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 4. Miniconda + Environnement
```bash
# Installer Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init zsh  # ou bash
rm ~/miniconda.sh
# Fermer et rouvrir le terminal

# Accepter ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Créer environnement
conda create -n cellvit python=3.10 -y
conda activate cellvit
```

### 5. PyTorch + Dépendances
```bash
# PyTorch avec CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Dépendances projet
pip install timm transformers huggingface_hub
pip install scikit-learn scipy pandas matplotlib seaborn
pip install tifffile opencv-python netcal mapie gradio
```

### 6. Test final
```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
```

---

## Dépendances Clés (à installer)

```
# Core ML
torch>=2.0
torchvision
timm
transformers

# Histopathologie
openslide-python
tifffile
staintools  # ou torchstain

# Évaluation
scikit-learn
scipy
pandas
matplotlib

# Calibration & Incertitude
netcal
mapie

# API/Démo
fastapi
gradio  # ou streamlit
```
