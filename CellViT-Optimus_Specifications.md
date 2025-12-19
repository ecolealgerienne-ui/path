# CellViT-Optimus — Spécifications Techniques

**Version:** 1.0  
**Date:** Décembre 2024  
**Statut:** POC / Exploration

---

## Table des matières

1. [Vision & Objectifs](#1-vision--objectifs)
2. [Architecture Technique](#2-architecture-technique)
3. [Composants Modèles](#3-composants-modèles)
4. [Sources de Données](#4-sources-de-données)
5. [Infrastructure & Environnement](#5-infrastructure--environnement)
6. [Scripts & Outils](#6-scripts--outils)
7. [Stratégie de Sécurité Clinique](#7-stratégie-de-sécurité-clinique)
8. [Optimisation Inférence](#8-optimisation-inférence)
9. [Plan de Développement](#9-plan-de-développement)
10. [Contraintes & Risques](#10-contraintes--risques)

---

## 1. Vision & Objectifs

### 1.1 Objectif du système

CellViT-Optimus est un **système d'assistance au triage histopathologique**. Il ne vise pas à remplacer le pathologiste, mais à :

- Prioriser les lames et régions à forte valeur diagnostique
- Réduire le temps de lecture
- Sécuriser la décision grâce à une maîtrise explicite de l'incertitude

### 1.2 Paradigme

**Foundation Model + Heads expertes** : combinaison d'un backbone robuste pré-entraîné (H-optimus-0) avec des têtes spécialisées légères.

### 1.3 Ce que le système fait

- Analyse des lames H&E entières (WSI)
- Segmentation et caractérisation cellulaire (5 types)
- Prédiction de biomarqueurs moléculaires à des fins de triage
- Identification explicite des cas incertains ou hors domaine

### 1.4 Ce que le système ne fait pas

- Diagnostic autonome
- Remplacement direct de l'IHC ou des tests moléculaires
- Décision clinique sans validation humaine

---

## 2. Architecture Technique

### 2.1 Vue d'ensemble

```
┌────────────────────────────────────────────────────────────────┐
│                      LAME H&E (WSI)                            │
│                  Entrée : image multi-gigapixels               │
└────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│              COUCHE 1 — EXTRACTION SÉMANTIQUE                  │
│                     H-OPTIMUS-0 (gelé)                         │
│                                                                │
│  • Entrée : tuiles 224×224 @ 0.5 MPP                          │
│  • Sortie : embeddings 1536-dim                                │
│  • Features multi-couches pour UNETR (couches 6, 12, 18, 24)  │
└────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│  COUCHE 2A — CELLULAIRE  │          │  COUCHE 2B — LAME        │
│    Décodeur UNETR        │          │    Attention-MIL         │
│                          │          │                          │
│  • NP : présence noyaux  │          │  • Agrégation régions    │
│  • HV : séparation       │          │  • Pondération attention │
│  • NT : typage (5 cls)   │          │  • Score biomarqueur     │
│                          │          │                          │
│  Sortie : objets cellules│          │  Sortie : prédiction WSI │
└──────────────────────────┘          └──────────────────────────┘
          │                                         │
          │         ┌───────────────────────────────┘
          │         │ Feedback cellulaire (Phase 4)
          ▼         ▼
┌────────────────────────────────────────────────────────────────┐
│              COUCHE 3 — SÉCURITÉ & INCERTITUDE                 │
│                                                                │
│  • Incertitude aléatorique (entropie NP/HV)                   │
│  • Incertitude épistémique (Conformal Prediction)             │
│  • Détection OOD (distance latente Mahalanobis)               │
│  • Calibration locale (Temperature Scaling par centre)         │
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
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Principes directeurs

1. Le backbone foundation n'est jamais décisionnel seul
2. Les décisions cliniques sont toujours filtrées par une couche de sécurité
3. Le pathologiste reste l'autorité finale

### 2.3 Décodeur UNETR — Reconstruction spatiale

**Problème :** H-optimus-0 (ViT) produit des embeddings à résolution fixe (patches 14×14). La segmentation cellulaire requiert une précision pixel-level.

**Solution :** Architecture UNETR
- Extraction de features à différentes profondeurs du Transformer (couches 6, 12, 18, 24)
- Blocs de convolution transposée pour reconstruire la pyramide de résolution
- Conservation de la richesse sémantique + récupération de la granularité spatiale

### 2.4 Articulation des branches

**Phase initiale :** Branches parallèles (modularité, validation indépendante)

**Phase 4 :** Fusion hybride
- Branche Lame utilise embeddings Optimus bruts
- Injection des métriques cellulaires (% tumorales, densité lymphocytes) comme features supplémentaires dans le classifieur AMIL

---

## 3. Composants Modèles

### 3.1 H-optimus-0 (Backbone)

| Attribut | Valeur |
|----------|--------|
| Source | Bioptimus (HuggingFace) |
| Architecture | ViT-Giant/14 avec 4 registres |
| Paramètres | 1.1 milliard |
| Entrée | 224×224 pixels @ 0.5 MPP |
| Sortie | Embedding 1536-dim |
| Licence | Apache 2.0 (usage commercial OK) |
| Entraînement | 500k+ lames H&E multi-centres |

**Normalisation requise :**
```python
mean = (0.707223, 0.578729, 0.703617)
std = (0.211883, 0.230117, 0.177517)
```

**Utilisation :**
```python
import timm
model = timm.create_model(
    "hf-hub:bioptimus/H-optimus-0",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=False
)
```

### 3.2 CellViT (Référence segmentation)

| Attribut | Valeur |
|----------|--------|
| Source | TIO-IKIM (GitHub) |
| Architecture | U-Net + ViT encoder |
| Checkpoints | CellViT-SAM-H, CellViT-256 |
| Entrée inference | 1024×1024 pixels |
| Classes | 5 (Neoplastic, Inflammatory, Connective, Dead, Epithelial) |
| Dataset | PanNuke |

### 3.3 Alternative légère : H0-mini

| Attribut | Valeur |
|----------|--------|
| Architecture | ViT-Base/14 (distillé de H-optimus-0) |
| Paramètres | ~86M (vs 1.1B) |
| Performance | Comparable à H-optimus-0 |
| Licence | CC-BY-NC-ND (**non-commercial uniquement**) |

---

## 4. Sources de Données

### 4.1 Branche Cellule (Segmentation & Typage)

| Dataset | Contenu | Usage |
|---------|---------|-------|
| **PanNuke** | ~200k noyaux, 5 types, 19 organes | Entraînement principal NP/HV/NT |
| **MoNuSeG** | Multi-organes, segmentation | Robustesse segmentation |
| **CoNSeP** | Morphologie colique | Calibration branche HV |

### 4.2 Branche Lame (Biomarqueurs)

| Dataset | Contenu | Usage |
|---------|---------|-------|
| **TCGA** | Milliers de WSI + données moléculaires | Entraînement AMIL (MSI, mutations, survie) |
| **CPTAC** | WSI + protéomique | Têtes expertes biomarqueurs |

### 4.3 Stratégie de préparation

1. **Normalisation couleur :** Macenko sur tous les datasets
2. **Filtrage qualité :** Détection artefacts (bulles, flou) → données d'entraînement OOD

---

## 5. Infrastructure & Environnement

### 5.1 Configuration de développement

| Composant | Spécification |
|-----------|---------------|
| OS | WSL2 Ubuntu 22.04 LTS |
| GPU | RTX 4070 (12 GB VRAM) |
| CUDA | 12.x |
| Python | 3.10+ |
| Framework | PyTorch 2.x |

### 5.2 Architecture Docker

```
┌─────────────────────────────────────────────────────────────────┐
│                      DOCKER COMPOSE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  cellvit-base    │  │  cellvit-worker  │  │  cellvit-api  │  │
│  │                  │  │                  │  │               │  │
│  │  - CUDA 12.x     │  │  - H-optimus-0   │  │  - FastAPI    │  │
│  │  - PyTorch 2.x   │  │  - Décodeur      │  │  - Endpoints  │  │
│  │  - timm          │  │  - Inférence     │  │  - Health     │  │
│  │  - Base commune  │  │  - GPU required  │  │  - CPU only   │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                  │
│  Volumes partagés :                                              │
│  - /data/models     (checkpoints)                               │
│  - /data/cache      (embeddings versionnés)                     │
│  - /data/outputs    (résultats)                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Contraintes VRAM (RTX 4070 - 12 GB)

| Tâche | VRAM estimée | Faisabilité |
|-------|--------------|-------------|
| Inférence H-optimus-0 (FP16, batch=1) | ~3-4 GB | ✅ OK |
| Inférence H-optimus-0 (FP16, batch=8) | ~6-8 GB | ✅ OK |
| Entraînement décodeur (backbone gelé) | ~8-10 GB | ⚠️ Serré |
| Entraînement complet avec gradients | >16 GB | ❌ Impossible |

---

## 6. Scripts & Outils

### 6.1 Arborescence projet

```
cellvit-optimus/
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.worker
│   ├── Dockerfile.api
│   └── docker-compose.yml
│
├── scripts/
│   ├── setup/
│   │   ├── check_environment.py
│   │   ├── download_models.py
│   │   └── download_datasets.py
│   │
│   ├── preprocessing/
│   │   ├── tile_extraction.py
│   │   ├── stain_normalization.py
│   │   ├── quality_filter.py
│   │   └── tissue_detection.py
│   │
│   ├── evaluation/
│   │   ├── metrics_segmentation.py
│   │   ├── metrics_detection.py
│   │   ├── metrics_classification.py
│   │   └── aggregate_results.py
│   │
│   ├── calibration/
│   │   ├── temperature_scaling.py
│   │   ├── conformal_prediction.py
│   │   └── calibration_metrics.py
│   │
│   ├── ood_detection/
│   │   ├── latent_distance.py
│   │   ├── entropy_scoring.py
│   │   ├── ood_evaluation.py
│   │   └── ood_visualization.py
│   │
│   ├── benchmarking/
│   │   ├── latency_test.py
│   │   ├── throughput_test.py
│   │   ├── memory_profiling.py
│   │   └── stress_test.py
│   │
│   └── validation/
│       ├── sanity_checks.py
│       ├── regression_tests.py
│       └── cross_center_eval.py
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_metrics.py
│   │   ├── test_ood.py
│   │   └── test_calibration.py
│   │
│   └── integration/
│       ├── test_pipeline_e2e.py
│       └── test_docker_deployment.py
│
├── configs/
│   ├── eval_pannuke.yaml
│   ├── eval_tcga.yaml
│   └── thresholds_clinical.yaml
│
├── notebooks/
│   ├── 01_exploration_hoptimus.ipynb
│   ├── 02_cellvit_inference.ipynb
│   └── 03_integration_poc.ipynb
│
└── src/
    ├── models/
    ├── data/
    ├── inference/
    └── utils/
```

### 6.2 Librairies réutilisables

| Besoin | Librairie |
|--------|-----------|
| Métriques segmentation | CellViT repo (Panoptic Quality, F1) |
| Stain normalization | `torchstain` ou `staintools` |
| Temperature scaling | `netcal` |
| Conformal prediction | `mapie` |
| Profiling mémoire | `pytorch_memlab` |

---

## 7. Stratégie de Sécurité Clinique

### 7.1 Types d'incertitude

| Type | Source | Mécanisme de détection |
|------|--------|------------------------|
| Aléatorique | Qualité coupe/signal | Entropie prédictions NP/HV |
| Épistémique | Limites du modèle | Conformal Prediction |
| Hors distribution | Cas jamais vus | Distance Mahalanobis latente |

### 7.2 Stratégie Cold Start (nouveau centre)

```
1. Seuils Conservateurs (Défaut)
   └── Haute confiance requise, priorité précision > rappel

2. Shadow Mode (30-50 premières lames)
   └── Système prédit, expert valide
   └── Erreurs → ajustement Temperature Scaling local

3. Détection OOD automatique
   └── Si distribution trop éloignée → revue humaine systématique
```

### 7.3 Sortie clinique

Signal simple en 3 niveaux :
- **Fiable** — Confiance haute, prédiction utilisable
- **À revoir** — Incertitude détectée, validation humaine recommandée
- **Hors domaine** — Cas atypique, ne pas utiliser la prédiction

---

## 8. Optimisation Inférence

### 8.1 Tiling Adaptatif

**Objectif :** Réduire les tuiles à analyser (~70% de gain)

**Méthode :** Détection de contenu à basse résolution (5x ou 1.25x)

**Seuil critique :** Recall = 0.999 sur tissu tumoral
- Accepter ~10x faux positifs pour ne jamais rater un vrai positif
- Garde-fou : analyse systématique basse résolution de toute la lame

### 8.2 Cache d'Embeddings

**Versioning strict (hash-based) :**
```
ID = [Version_Backbone] + [Version_Preprocessing] + [Resolution] + [Date_Extraction]
```

**Règle :** Si version différente détectée → invalidation automatique du cache

### 8.3 Distillation (Phase 7)

**Objectif :** Modèle léger (MobileViT/ViT-Small) pour pré-triage

**Contraintes :**
- Ne remplace PAS H-optimus-0 pour diagnostic final en V1
- Test de Stress Histologique obligatoire :
  - Mesure performance sur sous-types rares (bague à chaton, sarcomatoïdes)
  - Si perte > 2% sur un sous-type → Teacher obligatoire pour ces cas

---

## 9. Plan de Développement

### 9.1 Phases

| Phase | Objectif | Durée estimée |
|-------|----------|---------------|
| 1 | Données : sélection organe/biomarqueur, dataset multi-centres | 2 sem |
| 2 | Intégration backbone : extraction features, stabilité inter-centres | 2 sem |
| 3 | Branche Cellule : décodeur UNETR, validation PanNuke | 4 sem |
| 4 | Branche Lame : AMIL + feedback cellulaire | 3 sem |
| 5 | Sécurité : OOD, calibration, seuils cliniques | 3 sem |
| 6 | Validation interne : tests rétrospectifs | 2 sem |
| 7 | Optimisation : inférence, distillation éventuelle | 2 sem |
| 8 | Préparation produit : gel pipeline, documentation | 2 sem |

**Total estimé : 20 semaines (~5 mois)**

### 9.2 POC Minimal (objectif démo)

| Semaine | Livrables |
|---------|-----------|
| 1-2 | Environnement + inférence CellViT pré-entraîné |
| 3-4 | Intégration H-optimus-0, extraction features |
| 5-6 | Interface démo (Gradio/Streamlit), packaging |

**Livrable POC : 6 semaines**

---

## 10. Contraintes & Risques

### 10.1 Contraintes techniques

| Contrainte | Impact | Mitigation |
|------------|--------|------------|
| VRAM limitée (12 GB) | Batch size réduit, pas de fine-tuning complet | Mixed precision, gradient accumulation |
| Poids modèles (~4 GB) | Temps de chargement | Cache en mémoire |
| Images Docker lourdes (~15-20 GB) | Stockage, temps de build | Multi-stage builds, layer caching |

### 10.2 Risques projet

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Features UNETR insuffisantes | Moyenne | Élevé | Valider empiriquement sur PanNuke en phase 3 |
| Performance sous benchmarks publiés | Moyenne | Moyen | Itérer sur architecture décodeur |
| Drift silencieux (cache embeddings) | Faible | Élevé | Versioning strict implémenté |

### 10.3 Périmètre Certifié vs R&D

**Inclus dans le périmètre certifiable (V1) :**
- Backbone H-optimus-0 gelé
- Une tête experte active (organe + usage définis)
- Pipeline de sécurité (OOD, conformal, calibration)
- Seuils cliniques validés
- Workflow human-in-the-loop

**Hors périmètre certifié :**
- Typage cellulaire détaillé à 5 classes
- Analyses exploratoires avancées
- Visualisations non décisionnelles
- Fonctions désactivées en production

---

## Annexes

### A. Commandes de vérification environnement

```bash
# WSL2
wsl --version
wsl -l -v

# CUDA dans WSL
nvidia-smi

# Python/PyTorch
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### B. Références

- H-optimus-0 : https://huggingface.co/bioptimus/H-optimus-0
- CellViT : https://github.com/TIO-IKIM/CellViT
- PanNuke : https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
- TCGA : https://www.cancer.gov/tcga

---

*Document généré le 18 décembre 2024*
