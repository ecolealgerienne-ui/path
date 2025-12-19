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
│  • Sortie : embeddings 1536-dim                                │
│  • ViT-Giant/14, 1.1 milliard paramètres                      │
└────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│  COUCHE 2A — CELLULAIRE  │          │  COUCHE 2B — LAME        │
│    Décodeur UNETR        │          │    Attention-MIL         │
│                          │          │                          │
│  • NP : présence noyaux  │          │  • Agrégation régions    │
│  • HV : séparation       │          │  • Score biomarqueur     │
│  • NT : typage (5 cls)   │          │                          │
└──────────────────────────┘          └──────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│              COUCHE 3 — SÉCURITÉ & INCERTITUDE                 │
│  • Incertitude aléatorique (entropie NP/HV)                   │
│  • Incertitude épistémique (Conformal Prediction)             │
│  • Détection OOD (distance Mahalanobis)                       │
│  Sortie : {Fiable | À revoir | Hors domaine}                  │
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

### Phase 1 : Environnement & Inférence CellViT (Semaines 1-2)

| Étape | Description | Validation | Statut |
|-------|-------------|------------|--------|
| 1.1 | Setup WSL2 + Docker + CUDA | `nvidia-smi` fonctionne | ✅ FAIT |
| 1.2 | Conda + PyTorch | `torch.cuda.is_available()` = True | ✅ FAIT |
| 1.3 | Télécharger CellViT-256 | Fichier 187 MB présent | ✅ FAIT (manuel) |
| 1.4 | Télécharger PanNuke | 3 folds présents | ✅ FAIT (manuel) |
| 1.5 | Inférence CellViT-256 | Détection cellules sur image test | ✅ FAIT |
| 1.6 | Valider métriques | Dice > 0.7 sur PanNuke fold3 | ✅ FAIT (Dice=0.8733) |

**Critères de passage Phase 2 :**
- [x] CellViT-256 fonctionne sur GPU
- [x] Détection visible sur image réelle
- [x] Métriques de base calculées

### Phase 2 : Intégration H-optimus-0 (Semaines 3-4)

| Étape | Description | Validation | Statut |
|-------|-------------|------------|--------|
| 2.1 | Accès HuggingFace gated | Token configuré | ✅ FAIT |
| 2.2 | Charger H-optimus-0 | Inférence OK sur 1 image | ✅ FAIT |
| 2.3 | Extraction features PanNuke | Embeddings 1536-dim sauvés | ✅ FAIT |
| 2.4 | Visualisation t-SNE | Clusters par organe visibles | ✅ FAIT |
| 2.5 | Décodeur UNETR skeleton | Architecture compilable | ✅ FAIT |
| 2.6 | Entraînement UNETR sur PanNuke | Loss converge | ⏳ À FAIRE |

**Critères de passage Phase 3 :**
- [ ] UNETR entraîné sur PanNuke (backbone gelé)
- [ ] Performance proche de CellViT-256 baseline

### Phase 3 : Interface Démo & Packaging (Semaines 5-6)

| Étape | Description | Validation | Statut |
|-------|-------------|------------|--------|
| 3.1 | Interface Gradio basique | Upload image → résultat | ✅ FAIT |
| 3.2 | Intégration CellViT-256 dans démo | Inférence réelle | ✅ FAIT |
| 3.3 | Rapport avec couleurs/emojis | Correspondance visuelle | ✅ FAIT |
| 3.4 | Scripts OOD/calibration | Utilitaires prêts | ✅ FAIT |
| 3.5 | Docker packaging | `docker-compose up` fonctionne | ⏳ À FAIRE |
| 3.6 | Documentation utilisateur | README complet | ⏳ À FAIRE |

**Critères de livraison POC :**
- [ ] Démo fonctionnelle end-to-end
- [ ] Docker déployable
- [ ] Documentation claire

---

## Statut Actuel

**Phase en cours :** Phase 3 (étapes 1.6 et 3.2 validées)
**Blocage actuel :** Aucun
**Prochaine action :** Étape 2.6 (entraînement UNETR) ou 3.5 (Docker packaging)

---

## Architecture POC vs Cible

> **ATTENTION** : L'implémentation actuelle est un POC de validation.
> Certains choix ne correspondent pas à l'architecture cible.

| Composant | POC (actuel) | Cible (production) |
|-----------|--------------|-------------------|
| Segmentation | CellViT-256 pré-entraîné | UNETR sur H-optimus-0 |
| Backbone | CellViT encoder (ViT-256, 46M params) | H-optimus-0 (1.1B params) |
| Données démo | Synthétiques | PanNuke + images réelles |
| Détection cellules | Seuillage simple (fallback) | Modèle entraîné |
| Incertitude | Non implémenté | Conformal Prediction |
| OOD | Scripts prêts (non intégrés) | Pipeline complet |

**Objectif POC :** Valider la faisabilité technique, pas l'architecture finale.

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

---

## Fichiers Créés (Inventaire)

```
src/
├── models/
│   ├── __init__.py
│   └── unetr_decoder.py          # Décodeur UNETR pour H-optimus-0
└── inference/
    ├── __init__.py
    ├── cellvit_inference.py       # Wrapper CellViT-256 simplifié
    ├── cellvit256_model.py        # Architecture CellViT-256 locale
    └── cellvit_official.py        # Wrapper pour repo officiel TIO-IKIM

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
│   └── train_unetr.py
├── utils/
│   └── inspect_checkpoint.py
├── validation/
│   └── test_cellvit256_inference.py  # Test étape 1.5 POC
└── demo/
    ├── gradio_demo.py             # Interface principale
    ├── synthetic_cells.py         # Générateur tissus
    └── visualize_cells.py         # Fonctions visualisation

models/
└── pretrained/
    └── CellViT-256.pth            # 187 MB (téléchargé manuellement)
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
