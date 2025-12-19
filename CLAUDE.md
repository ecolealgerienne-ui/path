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
| GPU | RTX 4070 Super (12 GB VRAM) |
| NVIDIA Driver | 566.36 |
| CUDA | 12.7 |
| Docker | 29.1.3 (natif, pas Docker Desktop) |
| NVIDIA Container Toolkit | Installé |
| Python | 3.10 (via Miniconda) |
| Conda | 25.11.1 |

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

## Journal de Développement

### 2024-12-19 — Setup environnement
- **Environnement WSL2 configuré** : Ubuntu 24.04.2 LTS
- **Docker Engine natif installé** (pas Docker Desktop) — meilleure performance, pas de licence
- **NVIDIA Container Toolkit** configuré — Docker peut accéder au GPU
- **Miniconda installé** — prêt pour environnement Python isolé
- **Décision** : Utiliser Python 3.10 pour compatibilité optimale avec PyTorch/CUDA

---

## Problèmes Connus & Solutions

| Problème | Solution |
|----------|----------|
| Conda ToS non acceptées | `conda tos accept --override-channels --channel <url>` |
| Docker "command not found" dans WSL | Installer Docker Engine natif, pas Docker Desktop |

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
