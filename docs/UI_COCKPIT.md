# CellViT-Optimus R&D Cockpit

> **Version:** POC v1.1 (Phase 1)
> **Date:** 2025-12-30
> **Status:** Fonctionnel — Aligné avec pipeline training/evaluation

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
python -m src.ui.app --preload --family respiratory
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

### Pipeline d'Inférence

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IMAGE RGB (224×224, uint8)                           │
└─────────────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
┌──────────────────────────┐         ┌──────────────────────────────────┐
│ preprocess_image()       │         │ ToTensor() → [0,1]               │
│ src.preprocessing        │         │ images_rgb pour FPN Chimique     │
│ (ToPILImage+Normalize)   │         │                                  │
└──────────────────────────┘         └──────────────────────────────────┘
          │                                         │
          ▼                                         │
┌──────────────────────────┐                       │
│ H-optimus-0              │                       │
│ forward_features()       │                       │
│ → (1, 261, 1536)         │                       │
└──────────────────────────┘                       │
          │                                         │
          ├──► validate_features()                  │
          │    CLS std ∈ [0.70, 0.90]              │
          │                                         │
          ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    HoVerNetDecoderHybrid                                │
│  • use_hybrid=True (FPN multi-échelle)                                  │
│  • use_fpn_chimique=True (H-channel injection)                          │
│  • use_h_alpha=False (optionnel)                                        │
│                                                                         │
│  Forward: model(features, images_rgb=images_rgb)                        │
└─────────────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
┌──────────────────────────┐         ┌──────────────────────────────────┐
│ NP Output (2, H, W)      │         │ HV Output (2, H, W)              │
│ softmax(dim=1)[1]        │         │ Gradients [-1, 1]                │
│ → np_pred [0, 1]         │         │                                  │
└──────────────────────────┘         └──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    hv_guided_watershed()                                │
│  src.postprocessing.watershed (SINGLE SOURCE OF TRUTH)                  │
│                                                                         │
│  Paramètres: np_threshold, beta, min_size, min_distance                 │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Instance Map (H, W)                                  │
│  + Morphométrie via MorphometryAnalyzer                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Modules Partagés (Single Source of Truth)

| Module | Import | Usage |
|--------|--------|-------|
| `src.preprocessing` | `preprocess_image`, `validate_features` | Normalisation H-optimus-0 |
| `src.postprocessing.watershed` | `hv_guided_watershed` | Segmentation instances |
| `src.evaluation.instance_evaluation` | `run_inference` | Inférence NP/HV (softmax!) |
| `src.metrics.morphometry` | `MorphometryAnalyzer` | Métriques morphologiques |

### Structure Fichiers

```
src/ui/
├── __init__.py           # Exports: CellVitEngine, AnalysisResult, visualizations
├── inference_engine.py   # CellVitEngine (wrapper unifié)
│   ├── _load_hovernet()      # Charge modèle + détecte flags checkpoint
│   ├── _preprocess_image()   # Preprocessing centralisé
│   └── analyze()             # Pipeline complet
├── visualizations.py     # Overlays et rendus
│   ├── create_segmentation_overlay()
│   ├── create_contour_overlay()
│   ├── create_uncertainty_overlay()
│   └── create_debug_panel()
└── app.py               # Interface Gradio
    ├── Validation 224×224
    ├── Chargement moteur
    └── Callbacks analyse
```

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

Les paramètres sont ajustables en temps réel :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| Seuil NP | 0.40 | Binarisation de la probabilité nucléaire |
| Taille min | 30 | Pixels minimum par instance |
| Beta | 0.50 | Poids HV magnitude |
| Distance min | 5 | Distance entre peaks |

### Valeurs optimales par famille

| Famille | NP Thr | Min Size | Beta | Min Dist | AJI |
|---------|--------|----------|------|----------|-----|
| Respiratory | 0.40 | 30 | 0.50 | 5 | **0.6872** ✅ |
| Urologic | 0.45 | 30 | 0.50 | 2 | 0.6743 |
| Epidermal | 0.45 | 20 | 1.00 | 3 | 0.6203 |
| Digestive | 0.45 | 60 | 2.00 | 5 | 0.6160 |

---

## API CellVitEngine

### Initialisation

```python
from src.ui import CellVitEngine

# Charger moteur avec famille spécifique
engine = CellVitEngine(
    device="cuda",           # ou "cpu"
    family="respiratory",    # famille HoVer-Net
    load_backbone=True,      # H-optimus-0 (~5s)
    load_organ_head=True     # OrganHead
)

# Vérifier status
print(engine.get_status())
# {'models_loaded': True, 'is_hybrid': True, 'use_fpn_chimique': True, ...}
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

### Changement de famille

```python
# Recharge HoVer-Net pour autre famille
engine.change_family("epidermal")

# Nouveaux paramètres watershed appliqués automatiquement
print(engine.watershed_params)
# {'np_threshold': 0.45, 'min_size': 20, 'beta': 1.0, 'min_distance': 3}
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

### Phase 2 — Couches IA (À venir)

**Objectif:** Debug pipeline et interaction

| Composant | Status | Description |
|-----------|--------|-------------|
| Mode Debug | ⏳ | Panneau dépliable NP/HV/Instances |
| Détection fusions | ⏳ | Identifier noyaux fusionnés |
| Détection sur-segmentation | ⏳ | Identifier faux positifs |
| Comparaison watershed | ⏳ | Avant/après paramètres |
| Clic sur noyau | ⏳ | Métriques individuelles |
| Export JSON | ⏳ | Résultats analyse |

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

### Phase 3 — Intelligence Spatiale (À venir)

**Objectif:** Biomarqueurs avancés

| Composant | Status | Description |
|-----------|--------|-------------|
| Pléomorphisme | ⏳ | Score anisocaryose [1-3] |
| Chromatine | ⏳ | Texture LBP, entropie |
| Topologie Voronoï | ⏳ | Graphe adjacence cellules |
| Clustering spatial | ⏳ | Hotspots, patterns |
| Mitoses améliorées | ⏳ | Détection par forme + chromatine |
| Ratio I/E spatial | ⏳ | Distribution TILs |

**Métriques Pléomorphisme:**
```python
# Score basé sur variance des caractéristiques morphologiques
pleomorphism_score = compute_pleomorphism(
    area_cv,           # Coefficient de variation aire
    circularity_cv,    # CV circularité
    chromatin_entropy, # Entropie texture
    nuclear_ratio_var  # Variance N/C ratio
)
# Résultat: 1 (faible), 2 (modéré), 3 (sévère)
```

---

### Phase 4 — Polish & Export (À venir)

**Objectif:** Prêt pour validation clinique

| Composant | Status | Description |
|-----------|--------|-------------|
| Support WSI | ⏳ | Tiles OpenSeadragon |
| Navigation panoramique | ⏳ | Zoom/Pan sur WSI |
| Export PDF | ⏳ | Rapport clinique formaté |
| Export CSV | ⏳ | Métriques tabulaires |
| Traçabilité | ⏳ | Audit trail (qui, quand, quoi) |
| Multi-images | ⏳ | Batch processing |
| Comparaison GT | ⏳ | Overlay ground truth |

**Format Rapport PDF:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│  RAPPORT D'ANALYSE — CellViT-Optimus                                    │
│  ⚠️ DOCUMENT D'AIDE À LA DÉCISION — VALIDATION MÉDICALE REQUISE        │
├─────────────────────────────────────────────────────────────────────────┤
│  Patient: [Anonymisé]     Date: 2025-12-30     Organe: Lung            │
│  Famille: Respiratory     Confiance: 98.2%                              │
├─────────────────────────────────────────────────────────────────────────┤
│  MÉTRIQUES GLOBALES                                                     │
│  • Noyaux détectés: 127                                                 │
│  • Densité: 2340 noyaux/mm²                                             │
│  • Aire moyenne: 45.2 ± 12.3 µm²                                        │
│  • Index mitotique: 3/10 HPF                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ALERTES                                                                │
│  🔍 Anisocaryose modérée (score 2/3)                                    │
│  🔍 TILs status: Chaud (>50% stroma)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  [Image + Overlay]                [Distribution types]                  │
│                                                                         │
│  Paramètres: np_thr=0.40, beta=0.50, min_size=30                       │
│  Modèle: hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth      │
│  Version: CellViT-Optimus v1.1                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Roadmap Résumé

```
Phase 1 ████████████████████ 100% ✅ Fondation
Phase 2 ░░░░░░░░░░░░░░░░░░░░   0%    Couches IA
Phase 3 ░░░░░░░░░░░░░░░░░░░░   0%    Intelligence Spatiale
Phase 4 ░░░░░░░░░░░░░░░░░░░░   0%    Polish & Export
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
