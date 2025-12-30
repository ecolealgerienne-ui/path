# CellViT-Optimus R&D Cockpit

> **Version:** POC v1 (Phase 1)
> **Date:** 2025-12-30
> **Status:** Fonctionnel

---

## Vue d'ensemble

Le **R&D Cockpit** est une interface Gradio pour l'exploration et la validation du moteur IA CellViT-Optimus. Ce n'est **pas** une IHM clinique — c'est un instrument de développement.

### Objectifs

1. **Moment WOW en 30 secondes** — Upload → Segmentation visible → Métriques
2. **Exploration des prédictions** — Overlays activables, debug pipeline
3. **Validation scientifique** — Métriques morphométriques, alertes cliniques
4. **Debug IA** — Visualisation NP/HV/Instances

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

## Fonctionnalités Phase 1

### Segmentation

- **Upload image** : Glisser-déposer une image H&E (224×224)
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
| Respiratory | 0.40 | 30 | 0.50 | 5 | 0.6872 |
| Urologic | 0.45 | 30 | 0.50 | 2 | 0.6743 |
| Epidermal | 0.45 | 20 | 1.00 | 3 | 0.6203 |
| Digestive | 0.45 | 60 | 2.00 | 5 | 0.6160 |

---

## Architecture Code

```
src/ui/
├── __init__.py           # Exports publics
├── inference_engine.py   # CellVitEngine (wrapper unifié)
├── visualizations.py     # Overlays et rendus
└── app.py               # Interface Gradio
```

### CellVitEngine

```python
from src.ui import CellVitEngine

engine = CellVitEngine(device="cuda", family="respiratory")
result = engine.analyze(image_rgb)

# Résultats
result.instance_map      # (H, W) IDs instances
result.n_nuclei          # Nombre de noyaux
result.morphometry       # MorphometryReport
result.organ_name        # Organe prédit
result.uncertainty_map   # (H, W) incertitude
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
3. **HoVer-Net** — `models/checkpoints_v13_smart_crops/hovernet_{family}_*.pth`

---

## Limitations (POC v1)

- Image unique (pas WSI)
- Pas de sauvegarde/export
- Pas de mode batch
- Pas de comparaison GT

---

## Roadmap

### Phase 2 (À venir)
- Mode Debug avancé (fusions, sur-segmentations)
- Comparaison avant/après watershed
- Export métriques JSON

### Phase 3 (À venir)
- Pléomorphisme (chromatine)
- Topologie spatiale (Voronoï)
- Détection mitoses améliorée

### Phase 4 (À venir)
- Support WSI (via OpenSeadragon)
- Export rapport clinique
- Traçabilité audit

---

## Troubleshooting

### "Moteur non chargé"

Cliquer sur "Charger le moteur" après avoir sélectionné la famille.

### Erreur CUDA

```bash
python -m src.ui.app --device cpu
```

### Gradio non trouvé

```bash
pip install gradio>=4.0.0
```

### Checkpoint non trouvé

Vérifier que les fichiers existent dans `models/checkpoints_v13_smart_crops/`.
