# Prompt Nouvelle Session — CellViT-Optimus V15.3 Cytologie

> **Dernière mise à jour:** 2026-01-25
> **Session précédente:** `docs/sessions/2026-01-24_v15_3_h_channel_session.md`

---

## Contexte Projet

Tu travailles sur **CellViT-Optimus**, un système de segmentation et classification de cellules pour la pathologie numérique.

### Deux Branches Principales

| Branche | Version | Objectif | Status |
|---------|---------|----------|--------|
| **Histologie** | V13 | Segmentation noyaux (AJI ≥ 0.68) | 1/5 familles validées |
| **Cytologie** | V15.3 | Classification Bethesda cervicale (>95% recall) | **En cours** |

### Performance Actuelle Pipeline Cytologie V15.2/V15.3

| Composant | Métrique | Valeur | Status |
|-----------|----------|--------|--------|
| Cell Triage V1 | Recall | **96.28%** | Production |
| Binary Head | Recall (Abnormal) | **97.12%** | Production |
| Severity Head | Recall (High-grade) | **81.53%** | Production |
| Fine-grained Head | Balanced Accuracy | **60.34%** | Production |

### Architecture V15.3 (Validée)

```
┌─────────────────────────────────────────────────────────────────────┐
│              V15.3 ARCHITECTURE (FINALE)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TRAINING (Deep Learning Only)                                       │
│  └── H-Optimus-0 CLS Token (1536D) → Cell Triage V1 → MultiHead     │
│                                                                      │
│  POST-PROCESSING (H-Channel)                                         │
│  ├── Confidence Boosting (validation prédictions)                   │
│  ├── Nuclei Detection (visualisation cell-level)                    │
│  └── H-Stats (métriques, comptage)                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Principe fondamental validé expérimentalement:**
> **Training:** Deep Learning (H-Optimus) uniquement
> **Post-Processing:** H-Channel (Ruifrok) pour validation et visualisation
> **Ne JAMAIS mélanger** les deux (cause -5.8% recall — "Noise Feature Poisoning")

---

## Ce Qui a Été Fait (Session 2026-01-24)

### Phases Complétées

| Phase | Fichiers | Status |
|-------|----------|--------|
| **Phase 1** - Module H-Channel | `src/preprocessing/h_channel.py` | 41/41 tests |
| **Phase 2** - Cell Triage v2 | `07b_train_cell_triage_v2.py` | **ABANDONNÉ** (-5.8%) |
| **Phase 3** - Confidence Boosting | `apply_confidence_boosting()` | Implémenté |
| **Phase 4** - Visualisation | `12_visualize_predictions.py` | 5 modes |

### Modes de Visualisation Disponibles

```bash
# Patch-level (défaut V15.2)
python scripts/cytology/12_visualize_predictions.py --image img.jpg

# Cell-level (V15.3 - contours noyaux)
python scripts/cytology/12_visualize_predictions.py --image img.jpg --cell_level

# Heatmap (zones suspectes uniquement)
python scripts/cytology/12_visualize_predictions.py --image img.jpg --heatmap

# Comparaison GT vs Prédictions
python scripts/cytology/12_visualize_predictions.py --input_dir val/images --compare_gt

# Métriques TP/FP/FN
python scripts/cytology/12_visualize_predictions.py --input_dir val/images --compare_gt --metrics
```

---

## Prochaines Étapes À Faire

### Phase 5: Validation (Priorité Haute)

- [ ] **Benchmark Confidence Boosting vs baseline**
  - Mesurer la réduction de faux positifs avec `apply_confidence_boosting()`
  - Script à créer: comparer résultats avec/sans boosting sur dataset val

- [ ] **Validation qualitative visualisation cell-level**
  - Générer visualisations sur ~20 images représentatives
  - Vérifier cohérence des contours de noyaux détectés

- [ ] **Mettre à jour CLAUDE.md** avec les résultats finaux V15.3

### Court Terme — Production

- [ ] **API REST pour intégration clinique** (FastAPI)
  - Endpoint `POST /diagnose` — Upload image, retourne diagnostic
  - Endpoint `GET /health` — Status de l'API
  - S'inspirer de `11_unified_inference.py`

### Moyen Terme — Amélioration

- [ ] **Optimiser threshold Severity** pour meilleur recall high-grade
  - Actuel: 81.53% recall → Objectif: >85%

- [ ] **Data augmentation** pour ASCH et SCC (classes sous-représentées)
  - ASCH: 33% recall seulement
  - SCC: 23 exemples dans val

### Long Terme — R&D

- [ ] Fine-tuning H-Optimus sur données cytologiques
- [ ] Multi-instance learning pour classification WSI complète
- [ ] Attention mechanisms pour interprétabilité

---

## Fichiers Clés

### Scripts Cytologie

| Script | Description |
|--------|-------------|
| `07_train_cell_triage.py` | Entraînement Cell Triage V1 (à utiliser) |
| `08_train_multihead_bethesda.py` | Entraînement MultiHead Bethesda |
| `10_train_multihead_combined.py` | Training avec APCData + SIPaKMeD |
| `11_unified_inference.py` | Pipeline d'inférence unifié |
| `12_visualize_predictions.py` | Visualisation (5 modes) |

### Modules Partagés

| Module | Fonctions |
|--------|-----------|
| `src/preprocessing/h_channel.py` | `extract_h_channel_ruifrok()`, `compute_h_stats()`, `detect_nuclei_for_visualization()`, `apply_confidence_boosting()` |
| `src/preprocessing/stain_separation.py` | `ruifrok_extract_h_channel()` |

### Modèles (à entraîner par l'utilisateur)

```
models/cytology/
├── cell_triage.pt              # Cell Triage V1 (96.28% recall)
├── multihead_bethesda.pt       # MultiHead Bethesda (APCData seul)
└── multihead_bethesda_combined.pt  # MultiHead (APCData + SIPaKMeD)
```

### Documentation

| Document | Description |
|----------|-------------|
| `CLAUDE.md` | Instructions projet (À METTRE À JOUR) |
| `docs/cytology/V15_3_H_CHANNEL_AUGMENTED_SPEC.md` | Spec V15.3 complète |
| `docs/cytology/V15_2_PIPELINE_PROGRESS.md` | Résultats V15.2 |
| `docs/sessions/2026-01-24_v15_3_h_channel_session.md` | Session précédente |

---

## Règles CRITIQUES

### 1. Utilise TOUJOURS l'Existant

```python
# ✅ CORRECT - Importer depuis src/
from src.preprocessing.h_channel import compute_h_stats, detect_nuclei_for_visualization
from src.postprocessing import hv_guided_watershed

# ❌ INTERDIT - Copier-coller du code
def compute_h_stats(...):  # Réimplémentation locale
```

**Avant de coder, vérifie:**
```bash
grep -r "def ma_fonction" src/
grep -r "MA_CONSTANTE" src/
```

### 2. Pas d'Initiatives Sans Raison

- Ne modifie **PAS** les scripts existants qui fonctionnent
- Ne crée **PAS** de nouveaux fichiers sans nécessité
- Demande **AVANT** de refactorer ou réorganiser

### 3. Inspire-toi des Scripts Existants

Pour créer un nouveau script, regarde d'abord:
- `07_train_cell_triage.py` — Pattern d'entraînement avec cache features
- `11_unified_inference.py` — Pattern de pipeline d'inférence
- `12_visualize_predictions.py` — Pattern de visualisation

### 4. Mettre à Jour CLAUDE.md

Après chaque changement significatif:
- Ajouter les nouveaux résultats
- Mettre à jour les prochaines étapes
- Documenter les décisions architecturales

### 5. Interdictions Absolues

- ❌ `python scripts/...` — Claude ne peut PAS exécuter de code
- ❌ Modifier l'architecture V15.3 validée
- ❌ Réintroduire H-Stats dans l'entraînement (régression prouvée -5.8%)
- ❌ Créer des fichiers sans nécessité absolue

---

## Leçons de la Session Précédente

### Ce qui Fonctionne

| Approche | Pourquoi |
|----------|----------|
| H-Optimus CLS seul pour training | Signal propre, 96%+ recall |
| H-Channel pour post-processing | Validation, visualisation |
| Architecture séparée DL/Heuristique | Pas de contamination |

### Ce qui NE Fonctionne PAS

| Approche | Pourquoi |
|----------|----------|
| Mélanger DL + H-Stats | "Noise Feature Poisoning" -5.8% |
| Otsu sur "Empty" bruité | Patches vides contiennent du bruit |
| Cell Triage v2 | ABANDONNÉ — garder V1 |

---

## Constantes Importantes

```python
# H-Optimus-0
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224

# Bethesda classes
BETHESDA_CLASSES = {
    0: "NILM", 1: "ASCUS", 2: "ASCH",
    3: "LSIL", 4: "HSIL", 5: "SCC"
}

# Severity mapping
SEVERITY_MAPPING = {
    0: "Normal",     # NILM
    1: "Low-grade",  # ASCUS
    2: "High-grade", # ASCH
    3: "Low-grade",  # LSIL
    4: "High-grade", # HSIL
    5: "High-grade"  # SCC
}

# Thresholds optimaux
CELL_TRIAGE_THRESHOLD = 0.01  # Très bas pour maximiser recall
BINARY_THRESHOLD = 0.3        # Haute sensibilité
SEVERITY_THRESHOLD = 0.4      # Équilibré
```

---

## Commandes Utiles

```bash
# Voir structure du projet
ls -la scripts/cytology/
ls -la src/preprocessing/

# Chercher une fonction existante
grep -r "def fonction_name" src/

# Voir les tests
python -m pytest tests/unit/test_h_channel.py -v

# Visualiser des prédictions (5 modes)
python scripts/cytology/12_visualize_predictions.py \
    --input_dir data/raw/apcdata/APCData_YOLO/val/images \
    --output results/visualizations/ \
    --max_images 5

# Mode cell-level (contours noyaux)
python scripts/cytology/12_visualize_predictions.py \
    --image img.jpg --cell_level

# Mode heatmap (zones suspectes)
python scripts/cytology/12_visualize_predictions.py \
    --image img.jpg --heatmap

# Mode métriques TP/FP/FN
python scripts/cytology/12_visualize_predictions.py \
    --input_dir val/images --compare_gt --metrics
```

---

## Checklist Début de Session

1. [ ] **Lire** ce prompt en entier
2. [ ] **Consulter** `CLAUDE.md` pour le contexte global
3. [ ] **Vérifier** `docs/cytology/V15_3_H_CHANNEL_AUGMENTED_SPEC.md` pour l'état actuel
4. [ ] **Demander** à l'utilisateur quelle tâche prioriser parmi les "Prochaines Étapes"
5. [ ] **Utiliser** les scripts et modules existants
6. [ ] **Documenter** les changements dans `docs/sessions/` à la fin

---

## Résumé des Commits Récents

```
d8be129 feat(cytology): Add script to extract patches for histology module testing
3a38191 feat(v15.3): Add TP/FP/FN metrics visualization mode
8719f63 feat(v15.3): Add GT vs Heatmap comparison mode
2adaea8 feat(v15.3): Add side-by-side original/heatmap comparison view
e5f97dd feat(v15.3): Add heatmap visualization mode for suspicious areas
d94d3a4 fix(v15.3): Rewrite nuclei detection for Pap-stained cytology
8178694 feat(v15.3): Add cell-level visualization mode (Phase 4)
bf4a881 docs(v15.3): Document Cell Triage V2 experiment and finalize architecture
d77d1c0 feat(v15.3): Add Cell Triage v2 with H-Channel augmentation (Phase 2)
37ea97a feat(v15.3): Implement H-Channel extraction module (Phase 1)
```

---

**Bonne session !**
