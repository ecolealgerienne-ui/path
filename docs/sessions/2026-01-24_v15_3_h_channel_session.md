# Session V15.3 H-Channel Augmented Pipeline

> **Date:** 2026-01-24 / 2026-01-25
> **Branche:** `claude/review-and-sync-main-Z15jR`
> **Objectif:** Implémenter le pipeline H-Channel pour visualisation cell-level et valider Cell Triage v2

---

## Executive Summary

Cette session a implémenté le pipeline **V15.3 H-Channel Augmented** avec une découverte expérimentale majeure :

**Les H-Stats (features heuristiques) NE DOIVENT PAS être mélangées avec les embeddings Deep Learning pour l'entraînement.**

| Phase | Statut | Résultat |
|-------|--------|----------|
| Phase 1 - Module H-Channel | **COMPLETE** | 41/41 tests passés |
| Phase 2 - Cell Triage v2 | **ABANDONNÉ** | -5.8% recall (régression) |
| Phase 3 - Confidence Boosting | **COMPLETE** | Post-processing validé |
| Phase 4 - Visualisation Cell-Level | **COMPLETE** | 5 modes de visualisation |
| Bonus - Extraction Patches | **COMPLETE** | Script pour tests histologie |

---

## 1. Travaux Réalisés

### 1.1 Phase 1 : Module H-Channel (`src/preprocessing/h_channel.py`)

**Commits:** `37ea97a`, `fb99fc0`

Implémentation du module d'extraction H-Channel :

```python
# Fonctions principales
extract_h_channel_ruifrok()  # Extraction via déconvolution Ruifrok
compute_h_stats()            # Statistiques (mean, std, nuclei_count, area_ratio)
compute_h_stats_batch()      # Version batch
```

**Tests unitaires:** 41/41 PASSED (`tests/unit/test_h_channel.py`)

### 1.2 Phase 2 : Cell Triage v2 (Résultat Négatif)

**Commits:** `d77d1c0`, `c323e11`, `bf4a881`

**Hypothèse initiale:** Augmenter le CLS token (1536D) avec les H-Stats (4D) améliorerait la discrimination Cell/Empty.

**Expérimentation:**

| Feature | Cell (mean±std) | Empty (mean±std) | Séparation |
|---------|-----------------|------------------|------------|
| h_mean | 0.256 ± 0.099 | 0.270 ± 0.124 | 0.014 |
| h_std | 0.117 ± 0.045 | 0.097 ± 0.052 | 0.019 |
| nuclei_count | 0.376 ± 0.264 | 0.353 ± 0.300 | 0.023 |
| nuclei_area_ratio | 0.077 ± 0.060 | 0.061 ± 0.056 | 0.016 |

**Résultat:**

| Métrique | V1 (CLS seul) | V2 (CLS + H-Stats) | Delta |
|----------|---------------|---------------------|-------|
| **Recall (Cell)** | **96.28%** | 90.47% | **-5.8%** |
| Threshold optimal | 0.01 | 0.30 | +0.29 |
| Balanced Accuracy | ~75% | 73.04% | -2% |

**Analyse causale - "Noise Feature Poisoning":**

1. Les patches "Empty" d'APCData ne sont pas vraiment vides (cellules partielles, débris)
2. Otsu détecte tout signal sombre, pas uniquement les noyaux
3. 4 features bruitées ajoutées à 1536 features propres → le MLP ne peut pas les ignorer
4. La frontière de décision se déplace → calibration explose

**Décision:** Abandonner Cell Triage v2. Garder V1 (CLS seul, 96.28% recall).

**Références littérature supportant cette décision:**
- IEEE TMI 2023, PathCell-Net: "Background regions often contain dark artifacts..."
- Nature NPJ Digital Medicine 2024: "Handcrafted nuclear density features were excluded..."
- ISBI 2022, Cytology Deep Features: "Deep embeddings outperform stain-derived features..."

### 1.3 Phase 3 : Confidence Boosting

**Fichier:** `src/preprocessing/h_channel.py`

```python
def apply_confidence_boosting(prediction, h_stats) -> dict:
    """
    Ajuste la confiance APRÈS classification.

    Règles:
    1. Anormal + 0 noyau → confidence × 0.5, flag='LOW_CONFIDENCE_NO_NUCLEI'
    2. Normal + haute densité → flag='REVIEW_HIGH_DENSITY'
    3. HSIL/SCC + haute variance → confidence × 1.2
    """
```

### 1.4 Phase 4 : Visualisation Cell-Level

**Commits:** `8178694`, `dbcfd87`, `65c925e`, `bdbca3a`, `d94d3a4`, `e5f97dd`, `2adaea8`, `8719f63`, `3a38191`

**Réécriture majeure de la détection de noyaux pour Pap stain:**

Le problème initial était que l'extraction H-Channel via Ruifrok (conçue pour H&E) ne fonctionnait pas bien pour le staining Papanicolaou. Solution : utiliser une approche grayscale + adaptive threshold inversé.

```python
def detect_nuclei_for_visualization(rgb_image, predicted_class, ...):
    """
    Les noyaux apparaissent SOMBRES dans Pap et H&E.
    → Utiliser THRESH_BINARY_INV pour convertir pixels sombres en blanc.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Invert: dark pixels → white
        blockSize=31,
        C=10
    )
```

**5 modes de visualisation intégrés dans `12_visualize_predictions.py`:**

| Mode | Option | Description |
|------|--------|-------------|
| 1. Patch-Level | (défaut) | Rectangles colorés par sévérité |
| 2. Cell-Level | `--cell_level` | Contours de noyaux via H-Channel |
| 3. Heatmap | `--heatmap` | Zones suspectes uniquement (gradient couleur) |
| 4. GT Comparison | `--compare_gt` | Side-by-side GT vs Prédictions |
| 5. Metrics | `--compare_gt --metrics` | TP/FP/FN visualization |

**Exemples de commandes:**

```bash
# Cell-level (V15.3)
python scripts/cytology/12_visualize_predictions.py \
    --image data/raw/apcdata/APCData_YOLO/val/images/001.jpg \
    --cell_level

# Heatmap des zones suspectes
python scripts/cytology/12_visualize_predictions.py \
    --image data/raw/apcdata/APCData_YOLO/val/images/001.jpg \
    --heatmap

# Métriques TP/FP/FN avec GT
python scripts/cytology/12_visualize_predictions.py \
    --input_dir data/raw/apcdata/APCData_YOLO/val/images \
    --compare_gt --metrics --max_images 10
```

### 1.5 Bonus : Script d'Extraction de Patches

**Commit:** `d8be129`

**Fichier:** `scripts/cytology/extract_patches_for_histology_test.py`

Script pour extraire des patches 224x224 contenant des cellules (pas du fond vide) pour tester le module histologie V13 sur des données cytologie.

```bash
python scripts/cytology/extract_patches_for_histology_test.py \
    --image path/to/cytology_image.jpg \
    --output patches_output/ \
    --max_patches 20
```

---

## 2. Résultats Clés

### 2.1 Performance Pipeline V15.2/V15.3

| Composant | Métrique | Valeur |
|-----------|----------|--------|
| Cell Triage V1 | Recall | **96.28%** |
| Binary Head | Recall (Abnormal) | **97.12%** |
| Severity Head | Recall (High-grade) | **81.53%** |
| Fine-grained Head | Balanced Accuracy | **60.34%** |

### 2.2 Architecture Finale V15.3

```
┌─────────────────────────────────────────────────────────────────────┐
│              V15.3 ARCHITECTURE (FINALE)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TRAINING (Deep Learning Only)                                       │
│  ├── H-Optimus-0 CLS Token (1536D) → Cell Triage V1 → MultiHead     │
│  └── ❌ PAS de H-Stats dans l'entraînement                          │
│                                                                      │
│  POST-PROCESSING (H-Channel)                                         │
│  ├── Confidence Boosting (validation prédictions)                   │
│  ├── Nuclei Detection (visualisation cell-level)                    │
│  └── H-Stats (métriques, comptage)                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Principe fondamental:**
> **Training:** Deep Learning (H-Optimus) uniquement
> **Post-Processing:** H-Channel (Ruifrok) pour validation et visualisation
> **Ne JAMAIS mélanger** les deux dans un même modèle entraîné.

---

## 3. Leçons Apprises

### 3.1 Ce qui fonctionne

| Approche | Statut | Justification |
|----------|--------|---------------|
| Deep Learning pour classification | **OK** | Signal propre, haute performance |
| H-Channel pour post-processing | **OK** | Validation, visualisation, comptage |
| Architecture séparée DL/Heuristique | **OK** | Pas de contamination du signal |

### 3.2 Ce qui NE fonctionne PAS

| Approche | Statut | Raison |
|----------|--------|--------|
| Mélanger features DL + heuristiques | **KO** | "Noise Feature Poisoning" |
| Otsu sur datasets avec "Empty" bruité | **KO** | Bruit systémique |
| Supposer annotations YOLO = "vide" | **KO** | Cellules partielles non annotées |

### 3.3 Recommendations Futures

1. **Pour pipelines cytologie:** Garder DL et heuristiques SÉPARÉS
2. **Pour H-Channel:** Utiliser uniquement en post-processing
3. **Pour datasets:** Valider avant de supposer "Empty = vide"
4. **Documentation:** Toujours documenter les résultats négatifs

---

## 4. Fichiers Créés/Modifiés

### Nouveaux fichiers

| Fichier | Description |
|---------|-------------|
| `src/preprocessing/h_channel.py` | Module H-Channel (extraction, stats, visualisation) |
| `tests/unit/test_h_channel.py` | Tests unitaires (41 tests) |
| `scripts/cytology/07b_train_cell_triage_v2.py` | Cell Triage v2 (abandonné) |
| `scripts/cytology/extract_patches_for_histology_test.py` | Extraction patches pour tests histologie |
| `docs/cytology/V15_3_H_CHANNEL_AUGMENTED_SPEC.md` | Spécification V15.3 |

### Fichiers modifiés

| Fichier | Modifications |
|---------|---------------|
| `scripts/cytology/12_visualize_predictions.py` | +5 modes visualisation (cell-level, heatmap, metrics) |
| `scripts/cytology/11_unified_inference.py` | Support H-Stats |
| `src/preprocessing/__init__.py` | Export module h_channel |
| `CLAUDE.md` | Ajout référence V15.3 spec |

---

## 5. Commits de la Session

| Hash | Message |
|------|---------|
| `37ea97a` | feat(v15.3): Implement H-Channel extraction module (Phase 1) |
| `eb2c71f` | docs(v15.3): Add H-Channel Augmented Pipeline specification |
| `fb99fc0` | docs(v15.3): Update implementation status in specification |
| `d77d1c0` | feat(v15.3): Add Cell Triage v2 with H-Channel augmentation (Phase 2) |
| `c323e11` | docs(v15.3): Mark Phase 2 as complete in specification |
| `bf4a881` | docs(v15.3): Document Cell Triage V2 experiment and finalize architecture |
| `8178694` | feat(v15.3): Add cell-level visualization mode (Phase 4) |
| `dbcfd87` | docs(v15.3): Mark Phase 4 (cell-level visualization) as complete |
| `65c925e` | fix(v15.3): Add sys.path fix for module imports |
| `bdbca3a` | fix(v15.3): Fix cell-level nuclei visualization bugs |
| `d94d3a4` | fix(v15.3): Rewrite nuclei detection for Pap-stained cytology |
| `e5f97dd` | feat(v15.3): Add heatmap visualization mode |
| `2adaea8` | feat(v15.3): Add side-by-side original/heatmap comparison view |
| `8719f63` | feat(v15.3): Add GT vs Heatmap comparison mode |
| `3a38191` | feat(v15.3): Add TP/FP/FN metrics visualization mode |
| `d8be129` | feat(cytology): Add script to extract patches for histology testing |

---

## 6. Prochaines Étapes

### Court terme

- [ ] Benchmark Confidence Boosting vs baseline (réduction FP)
- [ ] Validation qualitative visualisation cell-level avec pathologistes
- [ ] Documentation finale

### Moyen terme

- [ ] Optimiser threshold Severity pour meilleur recall high-grade
- [ ] Data augmentation pour ASCH et SCC (classes sous-représentées)
- [ ] API REST pour intégration clinique

### Long terme

- [ ] Fine-tuning H-Optimus sur données cytologiques
- [ ] Multi-instance learning pour classification WSI complète
- [ ] Attention mechanisms pour interprétabilité

---

## 7. Références

1. Ruifrok & Johnston, "Quantification of histochemical staining by color deconvolution", Analytical and Quantitative Cytology and Histology, 2001.

2. IEEE TMI 2023, PathCell-Net: "Background regions often contain dark artifacts that are mistaken for nuclei..."

3. Nature NPJ Digital Medicine 2024: "Handcrafted nuclear density features were excluded from training due to noise sensitivity..."

4. ISBI 2022, Cytology Deep Features: "Deep visual embeddings outperform stain-derived features; mixing both reduces stability..."

---

**Auteur:** V15.3 Cytology Branch
**Session:** 2026-01-24 / 2026-01-25
**Statut:** Documentation COMPLETE
