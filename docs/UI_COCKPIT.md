# CellViT-Optimus R&D Cockpit

> **Version:** POC v4.3 (Auto Params + Phase 3 Sync)
> **Date:** 2025-12-30
> **Status:** Fonctionnel — Architecture partagée R&D/Pathologiste

---

## Vue d'ensemble

Le **R&D Cockpit** est une interface Gradio pour l'exploration et la validation du moteur IA CellViT-Optimus. Ce n'est **pas** une IHM clinique — c'est un instrument de développement.

### Positionnement

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ⚠️ OUTIL D'AIDE — NE REMPLACE PAS LE DIAGNOSTIC MÉDICAL               │
├─────────────────────────────────────────────────────────────────────────┤
│  • Document d'aide à la décision (réglementaire)                        │
│  • Validation par pathologiste OBLIGATOIRE                              │
│  • Jamais de verdict binaire (malin/bénin)                              │
│  • Jamais de recommandation thérapeutique                               │
│  • Toujours afficher l'incertitude                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Objectifs

1. **Moment WOW en 30 secondes** — Upload → Segmentation visible → Métriques
2. **Exploration des prédictions** — Overlays activables, debug pipeline
3. **Validation scientifique** — Métriques morphométriques, alertes cliniques
4. **Debug IA** — Visualisation NP/HV/Instances, détection fusions/sur-segmentations

---

## Lancement

### Méthode 1: Script (recommandé)

```bash
./scripts/run_cockpit.sh
```

Options:
- `--preload` : Précharge le moteur au démarrage
- `--share` : Crée un lien public Gradio
- `--port 8080` : Port personnalisé

### Méthode 2: Python direct

```bash
conda activate cellvit
python -m src.ui.app
```

### Méthode 3: Avec préchargement

```bash
python -m src.ui.app --preload --organ Lung
# Ou avec modèle dédié:
python -m src.ui.app --preload --organ Breast  # ★ modèle dédié
```

---

## Interface

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CellViT-Optimus — R&D Cockpit                                          │
│  ⚠️ Document d'aide à la décision — Validation médicale requise         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │                          │  │ MÉTRIQUES                           │ │
│  │      IMAGE + OVERLAY     │  │ • Organe: Lung (98.2%)              │ │
│  │                          │  │ • Noyaux: 127                       │ │
│  │    [Clic = sélection]    │  │ • Densité: 2340/mm²                 │ │
│  │                          │  │ • Index mitotique: 3/10 HPF         │ │
│  └──────────────────────────┘  │                                      │ │
│                                │ DISTRIBUTION                         │ │
│  ☑ Segmentation  ☑ Contours   │ ████████░░ Néoplasique 42%           │ │
│  ☐ Incertitude  ☐ Densité     │ ███░░░░░░░ Inflammatoire 15%         │ │
│                                │                                      │ │
│  [Analyser]                    │ ALERTES                              │ │
│                                │ 🔍 Suspicion d'anisocaryose          │ │
├────────────────────────────────┴──────────────────────────────────────┤
│  ▶ Debug IA (fermé par défaut)                                        │
│    NP Probability | HV Horizontal | HV Vertical | Instances           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Contraintes d'Entrée

### Images acceptées

| Critère | Valeur | Raison |
|---------|--------|--------|
| **Taille** | 224×224 pixels **exactement** | Entrée native H-optimus-0 |
| **Format** | PNG, JPG, TIFF | RGB 3 canaux |
| **Résolution** | 0.5 MPP | Calibration PanNuke |

### Validation en amont

```python
# Dans app.py - Rejet automatique si ≠ 224×224
if h != 224 or w != 224:
    return error_message("Image {w}×{h} non acceptée. Requis: 224×224")
```

**Note:** Les images PanNuke sources sont 256×256. Les Smart Crops 224×224 sont extraits lors du preprocessing (voir `prepare_v13_smart_crops.py`).

---

## Architecture Technique

### Pipeline d'Inférence (Détail)

> **Alignement garanti avec `test_v13_smart_crops_aji.py`** — Même softmax, mêmes paramètres watershed.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1: PRÉTRAITEMENT                                                 │
│  inference_engine.py:456-480                                            │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Input: Image RGB 224×224 (uint8)                                     │
│  • Normalisation H-optimus-0:                                           │
│    - mean = (0.707223, 0.578729, 0.703617)                              │
│    - std  = (0.211883, 0.230117, 0.177517)                              │
│  • Output: tensor (1, 3, 224, 224) float32                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2: EXTRACTION FEATURES (H-OPTIMUS-0)                             │
│  inference_engine.py:514-522                                            │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Backbone: ViT-Giant/14 (1.1B params, GELÉ)                           │
│  • Output: (1, 261, 1536)                                               │
│    ├── [:, 0, :]      → CLS token (1536)     → OrganHead                │
│    ├── [:, 1:5, :]    → 4 Register tokens    → IGNORÉS                  │
│    └── [:, 5:261, :]  → 256 Patch tokens     → HoVer-Net                │
│  • Validation: CLS std ∈ [0.70, 0.90]                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────────────┐
│  ÉTAPE 3a: ORGANHEAD             │    │  ÉTAPE 3b: HOVERNET DECODER              │
│  inference_engine.py:529-554     │    │  inference_engine.py:556-567             │
│  ────────────────────────        │    │  ──────────────────────────────────────  │
│  • Input: CLS token (1536)       │    │  • Input: 256 Patch tokens + RGB image   │
│  • MLP: 1536→512→19              │    │  • FPN Chimique + H-channel injection    │
│  • Output:                       │    │  • run_inference() [SSOT]                │
│    - organ_name                  │    │    → softmax(dim=1)[1] ✅                │
│    - organ_confidence            │    │  • Output:                               │
│    - predicted_family            │    │    ├── NP: Probabilité [0, 1]            │
│    - watershed_params (override) │    │    └── HV: Gradients [-1, 1]             │
└──────────────────────────────────┘    └──────────────────────────────────────────┘
                    │                               │
                    │                               ▼
                    │               ┌──────────────────────────────────────────────┐
                    │               │  ÉTAPE 4: WATERSHED                          │
                    │               │  inference_engine.py:566-571                 │
                    └──────────────►│  ──────────────────────────────────────────  │
                                    │  • hv_guided_watershed() [SSOT]              │
                                    │  • Params = get_model_for_organ(PREDICTED)   │
                                    │    (Override organ-specific si disponible)   │
                                    │  • Output: instance_map (224×224)            │
                                    │    → Chaque noyau = ID unique (FINAL)        │
                                    └──────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 5: MORPHOMÉTRIE (PARTIELLE)                                      │
│  inference_engine.py:603-616                                            │
│  ─────────────────────────────────────────────────────────────────────  │
│  • MorphometryAnalyzer.analyze()                                        │
│  • Métriques par noyau: area, circularity, perimeter                    │
│  • Détection anomalies: fusions, sur-segmentations                      │
│  • ⚠️ mitotic_index_per_10hpf = None si surface < 0.1 mm²               │
│    (Patch 224×224 = 0.0125 mm² = seulement 6.4% d'un HPF)               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 6: ANALYSE SPATIALE (Phase 3)                                    │
│  inference_engine.py:619-649                                            │
│  ─────────────────────────────────────────────────────────────────────  │
│  • run_spatial_analysis() — READ-ONLY (ne modifie pas instance_map)     │
│  • Pléomorphisme: score 1-3 (CV aire + ratio taille)                    │
│  • Hotspots: zones haute densité (>1.5× moyenne)                        │
│  • Mitoses candidates: seuils absolus 25-180 µm²                        │
│  • Chromatine: entropie Shannon, hétérogénéité                          │
│  • Topologie Voronoï: graphe adjacence                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 5b: FINALISATION MORPHOMETRY + SYNC (v4.3)                       │
│  inference_engine.py:651-680                                            │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Phase 3 = source autoritative pour mitoses                           │
│  • morphometry.refresh_mitosis_alerts():                                │
│    → Supprime ancienne alerte mitose                                    │
│    → Ajoute nouvelle alerte avec count Phase 3                          │
│  • morphometry.refresh_confidence_after_phase3():                       │
│    → Dégrade confiance si complexité élevée:                            │
│      - Pléomorphisme 3 (sévère) → -1 niveau                             │
│      - > 10 mitoses → -1 niveau                                         │
│      - > 20% chromatine hétérogène → -1 niveau                          │
│  • mitotic_index_per_10hpf reste None (sanity check)                    │
│  • Affichage: "X candidat(s) (patch unique)"                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SORTIE: AnalysisResult                                                 │
│  inference_engine.py:675-706                                            │
│  ─────────────────────────────────────────────────────────────────────  │
│  • image_rgb, instance_map, type_map, n_nuclei                          │
│  • organ_name, organ_confidence, family (PRÉDIT)                        │
│  • morphometry (finalisée avec Phase 3)                                 │
│  • spatial_analysis, pleomorphism_score, hotspots, mitosis_candidates   │
│  • watershed_params (organ-specific)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Points Critiques Pipeline

| Point | Implémentation | Fichier:ligne |
|-------|----------------|---------------|
| **Activation NP** | `softmax(dim=1)[1]` (pas sigmoid!) | `instance_evaluation.py:61` |
| **Watershed Params** | `get_model_for_organ(predicted_organ)` | `inference_engine.py:550-551` |
| **Phase 3** | READ-ONLY, enrichit métadonnées sans modifier segmentation | `spatial_analysis.py:585` |
| **HPF Sanity Check** | Si surface < 0.1 mm², index = None | `morphometry.py:220-229` |
| **Mitoses Absolues** | Seuils 25-180 µm² (pas relatifs à mean_area) | `spatial_analysis.py:264-273` |
| **Alertes Mitoses (v4.3)** | 3 niveaux: >10 TRÈS élevée, >3 élevée, >0 présentes | `format_clinical.py:188` |
| **Sync Confiance (v4.3)** | Dégrade après Phase 3 si complexité élevée | `morphometry.py:146` |

### Modules Partagés (Single Source of Truth)

| Module | Import | Usage |
|--------|--------|-------|
| `src.preprocessing` | `preprocess_image`, `validate_features` | Normalisation H-optimus-0 |
| `src.postprocessing.watershed` | `hv_guided_watershed` | Segmentation instances |
| `src.evaluation.instance_evaluation` | `run_inference` | Inférence NP/HV (softmax!) |
| `src.metrics.morphometry` | `MorphometryAnalyzer` | Métriques morphologiques |

### Architecture UI Partagée (v4.3)

> **Principe:** Les deux UIs (R&D et Pathologiste) partagent la même logique d'analyse.

```
src/ui/core/engine_ops.py
│
├── run_analysis_core()          ← PARTAGÉ (Single Source of Truth)
│   ├── Validation image 224×224
│   ├── Auto/Manual params
│   ├── state.engine.analyze()
│   └── state.current_result = result
│
├── analyze_image_core()         ← R&D UI wrapper
│   ├── Appelle run_analysis_core()
│   └── Ajoute: debug panels, anomaly overlays
│
└── (autres fonctions partagées...)

src/ui/app.py (R&D)              src/ui/app_pathologist.py (Clinique)
│                                 │
└── analyze_image()               └── analyze_image()
    └── analyze_image_core()          └── run_analysis_core()
        └── run_analysis_core()       └── overlays cliniques
```

**Garanties:**

| Fonctionnalité | Partagée? |
|----------------|-----------|
| `engine.analyze()` | ✅ Identique |
| Auto params (organ_config) | ✅ Identique |
| `refresh_mitosis_alerts()` | ✅ Identique |
| `refresh_confidence_after_phase3()` | ✅ Identique |
| Visualisations | ❌ Différentes (R&D vs Clinique) |

### Structure Fichiers

```
src/ui/
├── __init__.py           # Exports: CellVitEngine, AnalysisResult, visualizations, export
├── organ_config.py       # Configuration organes (SOURCE UNIQUE DE VÉRITÉ)
│   ├── ORGANS                # 19 organes PanNuke
│   ├── ORGAN_TO_FAMILY       # Mapping organe → famille
│   ├── ORGANS_WITH_DEDICATED_MODEL  # {Breast, Colon}
│   └── get_model_for_organ() # Retourne checkpoint + params
├── inference_engine.py   # CellVitEngine (wrapper unifié)
│   ├── _set_organ()          # Configure moteur pour un organe
│   ├── _load_hovernet()      # Charge modèle + détecte flags checkpoint
│   ├── _preprocess_image()   # Preprocessing centralisé
│   ├── change_organ()        # Change d'organe à chaud
│   └── analyze()             # Pipeline complet
├── visualizations.py     # Overlays et rendus
│   ├── create_segmentation_overlay()
│   ├── create_contour_overlay()
│   ├── create_uncertainty_overlay()
│   └── create_debug_panel()
├── spatial_analysis.py   # Analyse spatiale Phase 3
│   ├── compute_pleomorphism_score()
│   ├── compute_chromatin_features()
│   └── run_spatial_analysis()
├── export.py             # Export Phase 4
│   ├── create_report_pdf()
│   ├── export_nuclei_csv()
│   ├── export_summary_csv()
│   └── process_batch()
├── app.py               # R&D Cockpit (développeurs)
│   ├── Sélection par organe (19)
│   ├── Tous les overlays (9)
│   ├── Sliders Watershed
│   ├── Debug IA complet
│   └── Export JSON/CSV/PDF
└── app_pathologist.py   # Interface Pathologiste (cliniciens)
    ├── Sélection par organe (19)
    ├── Overlays simplifiés (4)
    ├── Métriques interprétées
    ├── Badge Confiance IA
    └── Export PDF uniquement
```

### Configuration Organes (organ_config.py)

Fichier source unique de vérité pour le mapping organe → modèle:

```python
# Les 19 organes PanNuke groupés par famille
ORGAN_TO_FAMILY = {
    # Glandular (5 organes)
    "Breast": "glandular",
    "Prostate": "glandular",
    "Thyroid": "glandular",
    "Pancreatic": "glandular",
    "Adrenal_gland": "glandular",
    # Digestive (4 organes)
    "Colon": "digestive",
    "Stomach": "digestive",
    "Esophagus": "digestive",
    "Bile-duct": "digestive",
    # Urologic (6 organes)
    "Kidney": "urologic",
    # ... etc
}

# STRATÉGIE V13: Tous les organes utilisent les modèles par FAMILLE
# Les modèles organ-specific ont été abandonnés
ORGANS_WITH_DEDICATED_MODEL = set()  # Vide

# Usage
from src.ui.organ_config import get_model_for_organ

info = get_model_for_organ("Breast")
# {
#   'checkpoint_path': 'models/.../hovernet_glandular_...best.pth',
#   'family': 'glandular',
#   'is_dedicated': False,
#   'watershed_params': {'np_threshold': 0.40, 'min_size': 50, 'beta': 0.50, 'min_distance': 3},
#   'display_name': 'Breast (glandular)'
# }

# Override organ-specific pour watershed (optionnel)
ORGAN_WATERSHED_PARAMS = {
    "Breast": {"np_threshold": 0.50, "min_size": 30, "beta": 0.50, "min_distance": 2},
}
```

**Note:** Les paramètres watershed sont récupérés pour l'organe **prédit** par OrganHead, pas l'organe sélectionné manuellement. Cela garantit que les overrides organ-specific sont appliqués correctement.

---

## Fonctionnalités Phase 1

### Segmentation

- **Upload image** : Glisser-déposer une image H&E (224×224 **obligatoire**)
- **Analyse automatique** : Segmentation + Morphométrie
- **Overlays** :
  - Segmentation colorée (par type cellulaire)
  - Contours des noyaux
  - Carte d'incertitude (ambre)
  - Heatmap densité

### Métriques

- **Organe détecté** : Prédiction OrganHead + confiance
- **Comptage** : Nombre de noyaux détectés
- **Morphométrie** :
  - Aire moyenne ± std
  - Circularité
  - Densité (noyaux/mm²)
  - Index mitotique
  - Ratios (néoplasique, I/E)
- **TILs status** : chaud/froid/exclu

### Interaction

- **Clic sur noyau** : Affiche métriques individuelles
  - ID, Type, Aire, Périmètre, Circularité
  - Confiance, Status (incertain/mitose)

### Debug IA

- **Pipeline visuel** :
  - NP Probability (heatmap rouge)
  - HV Horizontal (bleu-rouge)
  - HV Vertical (bleu-rouge)
  - Instances finales (couleurs)

---

## Paramètres Watershed

### Mode Auto vs Manuel (v4.3)

> **Nouveau:** Checkbox "Params Auto (organ_config.py)" dans l'UI R&D.

| Mode | Comportement | Usage |
|------|--------------|-------|
| **Auto** (défaut) | Utilise `organ_config.py` pour l'organe **prédit** | Production |
| **Manuel** | Utilise les valeurs des sliders | Debug/Exploration |

**Exemple:** Si OrganHead prédit "Breast", le mode Auto applique automatiquement `min_distance=2` (optimisé pour Breast) même si le slider affiche 5.

**Note:** L'interface Pathologiste utilise **toujours** le mode Auto (pas de sliders exposés).

### Sliders (Mode Manuel)

Les paramètres sont ajustables en temps réel :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| Seuil NP | 0.40 | Binarisation de la probabilité nucléaire |
| Taille min | 30 | Pixels minimum par instance |
| Beta | 0.50 | Poids HV magnitude |
| Distance min | 5 | Distance entre peaks |

### Valeurs optimales par famille (Source: CLAUDE.md)

| Famille | NP Thr | Min Size | Beta | Min Dist | AJI | Status |
|---------|--------|----------|------|----------|-----|--------|
| **Respiratory** | 0.40 | 30 | 0.50 | 5 | **0.6872** | ✅ Objectif |
| **Urologic** | 0.45 | 30 | 0.50 | 2 | **0.6743** | 99.2% |
| **Glandular** | 0.40 | 50 | 0.50 | 3 | **0.6566** | 96.6% |
| Epidermal | 0.45 | 20 | 1.00 | 3 | 0.6203 | 91.2% |
| Digestive | 0.45 | 60 | 2.00 | 5 | 0.6160 | 90.6% |

**Override Organ-Specific:**

| Organe | NP Thr | Min Size | Beta | Min Dist | Raison |
|--------|--------|----------|------|----------|--------|
| Breast | 0.50 | 30 | 0.50 | 2 | Noyaux pléomorphes denses |

---

## API CellVitEngine

### Initialisation

```python
from src.ui import CellVitEngine

# Charger moteur avec organe spécifique
engine = CellVitEngine(
    device="cuda",           # ou "cpu"
    organ="Lung",            # Nom de l'organe (ex: Lung, Breast, Colon)
    load_backbone=True,      # H-optimus-0 (~5s)
    load_organ_head=True     # OrganHead
)

# Vérifier status
print(engine.get_status())
# {
#   'models_loaded': True,
#   'organ': 'Lung',
#   'family': 'respiratory',
#   'is_dedicated_model': False,   # True si modèle dédié (Breast, Colon)
#   'is_hybrid': True,
#   'use_fpn_chimique': True,
#   ...
# }
```

### Analyse

```python
import numpy as np
from PIL import Image

# Charger image 224×224
image = np.array(Image.open("sample.png"))
assert image.shape == (224, 224, 3), "Image must be 224×224"

# Analyser
result = engine.analyze(
    image,
    watershed_params={"np_threshold": 0.40},  # Override optionnel
    compute_morphometry=True,
    compute_uncertainty=True
)

# Résultats
print(f"Noyaux: {result.n_nuclei}")
print(f"Organe: {result.organ_name} ({result.organ_confidence:.1%})")
print(f"Temps: {result.inference_time_ms:.0f}ms")
```

### Résultats disponibles

```python
result.image_rgb         # (224, 224, 3) Image analysée
result.instance_map      # (224, 224) IDs instances [0=background]
result.np_pred           # (224, 224) Probabilité nucléaire [0,1]
result.hv_pred           # (2, 224, 224) Gradients HV [-1,1]
result.n_nuclei          # int Nombre de noyaux
result.nucleus_info      # List[NucleusInfo] Détails par noyau
result.morphometry       # MorphometryReport Métriques globales
result.uncertainty_map   # (224, 224) Incertitude [0,1]
result.organ_name        # str Organe prédit
result.organ_confidence  # float Confiance [0,1]
result.watershed_params  # dict Paramètres utilisés
result.inference_time_ms # float Temps total
```

### Changement d'organe

```python
# Recharge HoVer-Net pour autre organe
engine.change_organ("Breast")  # Utilise modèle dédié ★

# Vérifier quel modèle est utilisé
print(engine.is_dedicated_model)  # True pour Breast, Colon
print(engine.family)              # "glandular" (famille parent)

# Nouveaux paramètres watershed appliqués automatiquement
print(engine.watershed_params)
# {'np_threshold': 0.40, 'min_size': 30, 'beta': 0.50, 'min_distance': 5}

# Pour un organe sans modèle dédié → utilise modèle famille
engine.change_organ("Skin")  # Utilise modèle epidermal
print(engine.is_dedicated_model)  # False
print(engine.family)              # "epidermal"
```

---

## Détection Automatique du Modèle

Le moteur lit les flags directement du checkpoint (alignement avec training):

```python
# Dans _load_hovernet()
checkpoint = torch.load(path, weights_only=False)

use_hybrid = checkpoint.get("use_hybrid", False)
use_fpn_chimique = checkpoint.get("use_fpn_chimique", False)
use_h_alpha = checkpoint.get("use_h_alpha", False)

# Fallback pour anciens checkpoints
if not use_hybrid:
    use_hybrid = any("fpn" in k for k in state_dict.keys())
```

---

## Prérequis

### Dépendances Python

```bash
pip install gradio>=4.0.0
```

### Modèles requis

1. **H-optimus-0** — Téléchargé automatiquement depuis HuggingFace
2. **OrganHead** — `models/checkpoints/organ_head_best.pth`
3. **HoVer-Net** — `models/checkpoints_v13_smart_crops/hovernet_{family}_v13_smart_crops_hybrid_fpn_best.pth`

---

## Limitations (POC v1)

- Image unique 224×224 (pas WSI, pas crops multiples)
- Pas de sauvegarde/export des résultats
- Pas de mode batch
- Pas de comparaison avec Ground Truth
- Pas de détection mitose avancée

---

## Phases de Développement

### Phase 1 — Fondation ✅ (Actuelle)

**Objectif:** "Moment WOW" en <30 secondes

| Composant | Status | Description |
|-----------|--------|-------------|
| `CellVitEngine` | ✅ | Wrapper unifié moteur IA |
| Interface Gradio | ✅ | Upload + Visualisation |
| Overlay segmentation | ✅ | Couleurs par type cellulaire |
| Overlay contours | ✅ | Bordures des noyaux |
| Overlay incertitude | ✅ | Heatmap ambre |
| Métriques globales | ✅ | Comptage, morphométrie |
| Validation 224×224 | ✅ | Rejet images non conformes |
| Alignement pipeline | ✅ | preprocess_image(), validate_features() |

**Livrables:**
- `src/ui/inference_engine.py` — Moteur d'inférence
- `src/ui/visualizations.py` — Overlays
- `src/ui/app.py` — Interface Gradio
- `scripts/run_cockpit.sh` — Script lancement

---

### Phase 2 — Couches IA ✅ (Complétée)

**Objectif:** Debug pipeline et interaction

| Composant | Status | Description |
|-----------|--------|-------------|
| Mode Debug amélioré | ✅ | Panneau NP/HV/Instances + alertes anomalies |
| Détection fusions | ✅ | Noyaux avec aire > 2× moyenne (magenta) |
| Détection sur-segmentation | ✅ | Noyaux avec aire < 0.5× moyenne (cyan) |
| Clic sur noyau | ✅ | Métriques individuelles + status anomalie |
| Export JSON | ✅ | `result.to_json()` avec métadonnées complètes |
| Vue anomalies | ✅ | Overlay avec indicateurs F/S |

**Livrables Phase 2:**
- `NucleusInfo.is_potential_fusion/is_potential_over_seg` — Flags anomalies
- `AnalysisResult.fusion_ids/over_seg_ids` — Listes IDs anomalies
- `AnalysisResult.to_json()` — Export JSON complet
- `create_anomaly_overlay()` — Visualisation anomalies
- `create_debug_panel_enhanced()` — Panneau debug avec alertes

**Fonctionnalités Debug:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  ▶ Debug IA                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │ NP Prob  │ │ HV Horiz │ │ HV Vert  │ │ Instances│                   │
│  │          │ │          │ │          │ │          │                   │
│  │ [0,1]    │ │ [-1,1]   │ │ [-1,1]   │ │ Colors   │                   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                   │
│                                                                         │
│  ⚠️ Alertes:                                                            │
│  • 3 fusions potentielles (aire > 2× moyenne)                          │
│  • 5 sur-segmentations (aire < 0.5× moyenne)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 3 — Intelligence Spatiale ✅ (Complétée)

**Objectif:** Biomarqueurs avancés

| Composant | Status | Description |
|-----------|--------|-------------|
| Pléomorphisme | ✅ | Score anisocaryose [1-3] basé sur CV aire/circularité |
| Chromatine | ✅ | Texture LBP, entropie Shannon, détection hétérogénéité |
| Topologie Voronoï | ✅ | Graphe adjacence cellules, moyenne voisins |
| Clustering spatial | ✅ | Hotspots = zones haute densité (>1.5× moyenne) |
| Mitoses améliorées | ✅ | Détection par forme + chromatine + intensité |
| Overlays Phase 3 | ✅ | Hotspots 🟠, Mitoses 🔴, Chromatine 🟣, Voronoï |

**Livrables Phase 3:**
- `src/ui/spatial_analysis.py` — Module d'analyse spatiale complet
  - `compute_pleomorphism_score()` — Score anisocaryose 1-3
  - `compute_chromatin_features()` — LBP + entropie
  - `build_voronoi_topology()` — Tessellation + graphe adjacence
  - `find_spatial_clusters()` — Détection hotspots
  - `detect_mitosis_advanced()` — Forme + chromatine + intensité
  - `run_spatial_analysis()` — Pipeline complet Phase 3
- `NucleusInfo` enrichi avec champs Phase 3
- `AnalysisResult` avec `spatial_analysis`, `pleomorphism_score`, etc.
- Visualisations: `create_hotspot_overlay()`, `create_mitosis_overlay()`, etc.
- Panneau debug Phase 3 avec score pléomorphisme

**Score Pléomorphisme:**
```python
# Score basé sur variance des caractéristiques morphologiques
pleomorphism = compute_pleomorphism_score(areas, circularities)

# Critères:
# - CV aire < 0.25: faible, 0.25-0.50: modéré, > 0.50: sévère
# - Ratio taille max/min < 3: faible, 3-6: modéré, > 6: sévère
# - Score final = max des composantes (approche conservative)

# Résultat: PleomorphismScore
#   score: 1 (faible), 2 (modéré), 3 (sévère)
#   description: "Pléomorphisme sévère — forte anisocaryose"
```

**Détection Mitoses Avancée:**
```python
# Critères multi-facteurs (score cumulatif):
# - Forme irrégulière (circularité < 0.5): +0.4
# - Taille moyenne-grande (0.7-2.0× moyenne): +0.2
# - Intensité foncée (< 100): +0.2
# - Entropie chromatine élevée (> 3.5): +0.2
# - Contraste élevé (> 40): +0.1
# Seuil candidat mitose: score ≥ 0.5
```

**Clustering Hotspots:**
```python
# Grille de densité 16×16 pixels
# Seuil hotspot = 1.5× densité moyenne
# Connected components pour clusters
# Minimum 5 noyaux par cluster
```

---

### Phase 4 — Polish & Export ✅ (Complétée)

**Objectif:** Prêt pour validation clinique

| Composant | Status | Description |
|-----------|--------|-------------|
| Export PDF | ✅ | Rapport clinique formaté 2 pages |
| Export CSV Noyaux | ✅ | Données détaillées par noyau |
| Export CSV Résumé | ✅ | Métriques globales et paramètres |
| Export JSON | ✅ | Données complètes structurées |
| Traçabilité | ✅ | Audit trail (analysis_id, timestamp, image_hash) |
| Batch processing | ✅ | Traitement multi-images (API) |
| Support WSI | ⏳ | Tiles OpenSeadragon (future) |
| Comparaison GT | ⏳ | Overlay ground truth (future) |

**Livrables Phase 4:**
- `src/ui/export.py` — Module d'export complet
  - `AuditMetadata` — Dataclass traçabilité (analysis_id, timestamp, hash, etc.)
  - `create_audit_metadata()` — Génère métadonnées pour chaque analyse
  - `export_nuclei_csv()` — CSV avec 22 colonnes par noyau
  - `export_summary_csv()` — Résumé métriques globales
  - `create_report_pdf()` — Rapport PDF 2 pages avec visualisations
  - `process_batch()` — Traitement batch d'images
  - `BatchResult` — Résultats agrégés batch
- Interface UI avec boutons export (PDF, CSV, JSON)
- Téléchargement direct des fichiers générés

**Format Export CSV Noyaux:**
```csv
id,centroid_y,centroid_x,area_um2,perimeter_um,circularity,cell_type,type_idx,
confidence,is_uncertain,is_mitotic,is_potential_fusion,is_potential_over_seg,
anomaly_reason,chromatin_entropy,chromatin_heterogeneous,is_mitosis_candidate,
mitosis_score,n_neighbors,is_in_hotspot
1,45,67,52.30,28.40,0.812,Neoplastic,1,0.945,False,False,...
```

**Format Rapport PDF:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  RAPPORT D'ANALYSE — CellViT-Optimus                                    │
│  DOCUMENT D'AIDE À LA DÉCISION — VALIDATION MÉDICALE REQUISE            │
├─────────────────────────────────────────────────────────────────────────┤
│  Organe détecté: Lung (98.2%)                                           │
│  Famille: respiratory      ID Analyse: A1B2C3D4                         │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐    MÉTRIQUES GLOBALES                           │
│  │                    │    • Noyaux détectés: 127                       │
│  │  Segmentation      │    • Densité: 2340 noyaux/mm²                   │
│  │  Overlay           │    • Aire moyenne: 45.2 ± 12.3 µm²              │
│  │                    │    • Index mitotique: 3/10 HPF                  │
│  └────────────────────┘                                                 │
│                            INTELLIGENCE SPATIALE                        │
│                            • Pléomorphisme: 2/3 (Modéré)                │
│                            • Hotspots: 3 zones                          │
│                            • Mitoses candidates: 5                      │
├─────────────────────────────────────────────────────────────────────────┤
│  ALERTES                                                                │
│  • Pléomorphisme modéré                                                 │
│  • 5 mitoses suspectes — activité proliférative                         │
├─────────────────────────────────────────────────────────────────────────┤
│  PARAMÈTRES WATERSHED                                                   │
│  np_threshold: 0.40, min_size: 30, beta: 0.50, min_distance: 5          │
│                                                                         │
│  CellViT-Optimus v3.0 — Généré le 2025-12-30 15:30:00                   │
│  Ce document est un outil d'aide à la décision et ne remplace pas       │
│  le diagnostic médical.                                                 │
└─────────────────────────────────────────────────────────────────────────┘

PAGE 2: Distribution des types cellulaires (pie chart + table)
```

**Traçabilité (AuditMetadata):**
```python
@dataclass
class AuditMetadata:
    analysis_id: str       # UUID unique (ex: "A1B2C3D4")
    timestamp: str         # ISO 8601
    user_id: str           # Identifiant utilisateur
    session_id: str        # Session Gradio
    model_family: str      # Famille HoVer-Net
    model_checkpoint: str  # Nom du checkpoint
    model_version: str     # "v3.0"
    watershed_params: dict # Paramètres utilisés
    image_hash: str        # SHA256[:16] de l'image
    image_size: tuple      # (224, 224)
    inference_time_ms: float
```

---

## Roadmap Résumé

```
Phase 1 ████████████████████ 100% ✅ Fondation
Phase 2 ████████████████████ 100% ✅ Couches IA
Phase 3 ████████████████████ 100% ✅ Intelligence Spatiale
Phase 4 ████████████████████ 100% ✅ Polish & Export
```

**Toutes les phases complètes!** Le R&D Cockpit est maintenant prêt pour la validation clinique avec:
- Export PDF rapport clinique
- Export CSV données tabulaires
- Traçabilité complète
- API batch processing

---

## Positionnement: POC Technique R&D

> **Ce cockpit est un instrument technique pour l'équipe de développement, PAS une interface utilisateur finale.**

### Objectif actuel

Le R&D Cockpit sert à:

| Usage | Description |
|-------|-------------|
| **Debug IA** | Visualiser le pipeline NP/HV/Instances, détecter les anomalies |
| **Validation scientifique** | Vérifier les métriques morphométriques, les biomarqueurs |
| **Exploration** | Tester différents paramètres watershed, comparer les familles |
| **Export données** | Générer des rapports pour analyse externe |

### Ce que ce cockpit n'est PAS

- Une interface pour pathologistes
- Une IHM clinique validée
- Un outil de diagnostic
- Une interface ergonomique pour non-techniciens

### Indicateurs techniques affichés

Les indicateurs bruts (HV gradients, NP probability, entropie chromatine, etc.) sont **intentionnellement** visibles car:
- Ils servent au debug et à la compréhension du modèle
- Ils permettent de détecter des problèmes de prédiction
- Ils sont essentiels pour l'amélioration continue de l'IA

---

## Évolution: Écran Pathologiste ✅ (Implémenté)

> **Interface dédiée aux pathologistes — `app_pathologist.py`**

### Lancement

```bash
# Interface Pathologiste (port 7861)
./scripts/run_pathologist.sh --preload

# ou directement
python -m src.ui.app_pathologist --preload --organ Lung

# Avec modèle dédié:
python -m src.ui.app_pathologist --preload --organ Breast  # ★
```

### Design implémenté

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CellViT-Optimus — Analyse Histopathologique                            │
│  Document d'aide à la décision — Validation médicale requise            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────────┐   ┌────────────────────────────────────┐   │
│  │                        │   │  ┌──────────────────────────────┐  │   │
│  │     IMAGE + OVERLAY    │   │  │   Confiance IA : Élevée      │  │   │
│  │                        │   │  └──────────────────────────────┘  │   │
│  │     [Clic = détails]   │   │                                    │   │
│  │                        │   │  ### Poumon                        │   │
│  └────────────────────────┘   │  Confiance: 98%                    │   │
│                               │  Famille: Respiratory              │   │
│  ☑ Types cellulaires         │                                    │   │
│  ☑ Contours                  │  Noyaux détectés: 127              │   │
│  ☑ Zones denses              │  Densité: Élevée (2340/mm²)        │   │
│  ☑ Mitoses                   │  Index mitotique: 3/10 HPF (Faible)│   │
│                               │  Pléomorphisme: Modéré (grade II)  │   │
│  [Analyser]                   │                                    │   │
│                               │  DISTRIBUTION [chart]              │   │
├───────────────────────────────┴────────────────────────────────────────┤
│  POINTS D'ATTENTION                                                    │
│  🟡 Anisocaryose modérée — variation notable                           │
│  🟠 Zones hypercellulaires — 2 cluster(s) identifié(s)                 │
├─────────────────────────────────────────────────────────────────────────┤
│  [Télécharger le rapport PDF]                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Différences clés vs. R&D Cockpit

| Aspect | R&D Cockpit | Écran Pathologiste |
|--------|-------------|-------------------|
| **Public** | Développeurs IA | Pathologistes |
| **Port** | 7860 | 7861 |
| **Métriques** | Brutes (debug) | Interprétées (clinique) |
| **Overlays** | 9 (HV, NP, Voronoï, etc.) | 4 (types, contours, hotspots, mitoses) |
| **Paramètres** | Sliders Watershed | Automatiques (masqués) |
| **Export** | JSON/CSV/PDF technique | PDF clinique uniquement |
| **Confiance** | Valeurs brutes | Badge visuel (Élevée/Modérée/Faible) |

### Fonctionnalités implémentées

| Fonction | Status | Description |
|----------|--------|-------------|
| Badge Confiance IA | ✅ | Indicateur visuel global (vert/orange/rouge) |
| Métriques interprétées | ✅ | "Densité: Élevée" au lieu de "2340/mm²" |
| Grades cliniques | ✅ | "Modéré (compatible grade II)" |
| Overlays simplifiés | ✅ | 4 checkboxes au lieu de 9 |
| Alertes priorisées | ✅ | Emojis 🔴🟡🟠 + langage clinique |
| Détails avancés | ✅ | Accordéon optionnel pour experts |
| Export PDF | ✅ | Rapport formaté pour dossier patient |

### Ce qui est masqué pour le pathologiste

```python
HIDDEN_FOR_PATHOLOGIST = [
    # Debug IA
    "np_pred",              # Probabilité nucléaire brute
    "hv_pred",              # Gradients HV
    "debug_panel",          # Panneau debug NP/HV/Instances

    # Paramètres techniques
    "watershed_sliders",    # np_threshold, beta, min_size, min_distance

    # Métriques brutes
    "chromatin_entropy",    # Valeur entropie → "hétérogène" oui/non
    "mitosis_score",        # Score 0-1 → "candidat" oui/non
    "n_neighbors",          # Nombre voisins Voronoï
    "area_cv",              # CV aire → score pléomorphisme

    # Overlays debug
    "voronoi_overlay",      # Tessellation technique
    "uncertainty_overlay",  # Incertitude modèle
    "anomaly_overlay",      # Fusions/sur-seg (R&D)
]
```

### Langage clinique

| Métrique brute | Interprétation clinique |
|----------------|------------------------|
| `density = 2340` | "Densité: Élevée (2340/mm²)" |
| `pleomorphism_score = 2` | "Modéré (compatible grade II)" |
| `mitotic_index = 3.0` | "3/10 HPF (Faible)" |
| `uncertainty_mean < 0.3` | Badge "Confiance IA: Élevée" |

### Évolutions futures (WSI)

1. **Viewer WSI zoomable** — OpenSeadragon (à intégrer)
2. **Annotations** — Marquer des régions d'intérêt
3. **Workflow séquentiel** — Valider et passer au suivant
4. **Historique** — Traçabilité des validations

---

## Configuration Overlays

### Palette de couleurs standardisée

Définie dans `src/ui/visualizations.py`:

```python
OVERLAY_CONFIG = {
    # Transparence
    "segmentation_alpha": 0.4,
    "contour_thickness": 1,
    "anomaly_alpha": 0.5,

    # Couleurs Phase 1 (RGB)
    "uncertainty_color": (255, 191, 0),     # Ambre
    "density_cmap": "YlOrRd",               # Jaune-Orange-Rouge

    # Couleurs Phase 2 (RGB)
    "fusion_color": (255, 0, 255),          # Magenta
    "over_seg_color": (0, 255, 255),        # Cyan

    # Couleurs Phase 3 (RGB)
    "hotspot_color": (255, 165, 0),         # Orange
    "mitosis_high_color": (255, 0, 0),      # Rouge
    "mitosis_low_color": (255, 255, 0),     # Jaune
    "chromatin_color": (148, 0, 211),       # Violet
    "voronoi_color": (100, 100, 100),       # Gris
}
```

### Ordre de superposition (z-index)

```python
OVERLAY_ORDER = [
    "density",          # Fond
    "segmentation",     # Couleurs par type
    "contours",         # Bordures
    "voronoi",          # Tessellation
    "uncertainty",      # Zones incertaines
    "hotspots",         # Clusters
    "chromatin",        # Hétérogénéité
    "mitoses",          # Candidats
    "anomalies",        # Dernier = plus visible
]
```

---

## Troubleshooting

### "Erreur : Image {w}×{h} pixels"

L'image doit être exactement 224×224. Utilisez les Smart Crops générés par `prepare_v13_smart_crops.py`.

### "Moteur non chargé"

Cliquer sur "Charger le moteur" après avoir sélectionné la famille.

### Erreur CUDA out of memory

```bash
python -m src.ui.app --device cpu
```

### Features validation warning

Si CLS std ∉ [0.70, 0.90], vérifier:
1. Image bien en uint8 [0-255]
2. Pas de pré-normalisation externe
3. Format RGB (pas BGR)

### Checkpoint non trouvé

Vérifier que les fichiers existent dans `models/checkpoints_v13_smart_crops/`.

### Gradio non trouvé

```bash
pip install gradio>=4.0.0
```
