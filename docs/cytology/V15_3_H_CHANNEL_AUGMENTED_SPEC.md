# V15.3 H-Channel Augmented Pipeline — Specification

> **Date:** 2026-01-24
> **Status:** DRAFT — En attente d'implémentation
> **Objectif:** Améliorer la détection et la visualisation cell-level via le canal Hématoxyline

---

## Executive Summary

Le pipeline V15.3 introduit une **architecture hybride** combinant:
- **Deep Learning** (H-Optimus CLS token) pour la classification sémantique
- **Signal Structurel** (H-Channel via Ruifrok) pour la détection de noyaux

Cette approche résout les limitations de V15.2:
- Visualisation **cell-level** au lieu de patch-level
- Réduction des **faux positifs** via validation H-Channel
- **Comptage de cellules** précis par image
- **Priorisation clinique** basée sur la densité nucléaire

---

## 1. Contexte et Motivation

### 1.1 Limitation V15.2

| Aspect | V15.2 (Actuel) | Problème |
|--------|----------------|----------|
| Granularité | Patch 224×224 | Trop grossier pour pathologistes |
| Visualisation | Rectangles patches | Non clinique |
| Faux Positifs | ~3% | Patches mucus/débris mal classés |
| Comptage | Patches | Pas de comptage cellulaire |

### 1.2 Solution V15.3

| Aspect | V15.3 (Proposé) | Bénéfice |
|--------|-----------------|----------|
| Granularité | **Cell-level** | Cohérent avec pratique clinique |
| Visualisation | **Contours noyaux** | Interprétable par pathologistes |
| Faux Positifs | Réduit 20-40% | Validation H-Channel |
| Comptage | **Noyaux détectés** | Métrique clinique standard |

---

## 2. Architecture Hybride

### 2.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    V15.3 H-CHANNEL AUGMENTED PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

                              Image LBC
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
         Sliding Window                    Ruifrok Deconvolution
           224×224                               │
                 │                               ▼
                 ▼                         H-Channel (full)
          H-Optimus-0                            │
                 │                               │
                 ▼                               │
         CLS Token (1536D)                       │
                 │                               │
                 │         ┌─────────────────────┘
                 │         │
                 │         ▼
                 │    H-Stats per Patch
                 │    (mean, std, blob_count)
                 │         │
                 ▼         ▼
         ┌───────┴─────────┴───────┐
         │   Cell Triage v2        │
         │   (CLS + H-Stats)       │
         └───────────┬─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │   MultiHead Bethesda    │
         │   (Classification)      │
         └───────────┬─────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Diagnostic Final    Nuclei Detection
    (NORMAL/ABNORMAL)   (H-Channel Otsu)
                              │
                              ▼
                    Cell-Level Visualization
                    (Contours + Classes)
```

### 2.2 Deux Chemins Parallèles

| Chemin | Input | Processing | Output |
|--------|-------|------------|--------|
| **Sémantique** | RGB 224×224 | H-Optimus → CLS | Features 1536D |
| **Structurel** | RGB → H-Channel | Ruifrok → Stats | mean, std, blob_count |

### 2.3 Fusion des Chemins

```python
# Cell Triage v2 Input
features_combined = concat([
    cls_token,           # 1536D (sémantique)
    h_mean,              # 1D (intensité moyenne H)
    h_std,               # 1D (hétérogénéité)
    nuclei_count,        # 1D (nombre de blobs)
    nuclei_area_ratio    # 1D (surface noyaux / surface patch)
])
# Total: 1540D
```

---

## 3. Composants Détaillés

### 3.1 Extraction H-Channel (Ruifrok)

**Algorithme:** Déconvolution couleur basée sur la loi de Beer-Lambert

```python
def extract_h_channel_ruifrok(rgb_image):
    """
    Extrait le canal Hématoxyline via déconvolution Ruifrok.

    Vecteurs Ruifrok (constantes physiques):
    - Hématoxyline: [0.650, 0.704, 0.286]
    - Éosine: [0.072, 0.990, 0.105]
    """
    # Conversion RGB → OD (Optical Density)
    od = -np.log10((rgb_image.astype(float) + 1) / 256)

    # Matrice de déconvolution Ruifrok
    stain_matrix = np.array([
        [0.650, 0.704, 0.286],  # Hématoxyline
        [0.072, 0.990, 0.105],  # Éosine
        [0.268, 0.570, 0.776]   # Résiduel
    ])

    # Déconvolution
    deconv = np.linalg.lstsq(stain_matrix.T, od.reshape(-1, 3).T, rcond=None)[0]
    h_channel = deconv[0].reshape(rgb_image.shape[:2])

    # Normalisation [0, 255]
    h_channel = np.clip(h_channel, 0, None)
    h_channel = (h_channel / h_channel.max() * 255).astype(np.uint8)

    return h_channel
```

**Référence:** Ruifrok & Johnston, "Quantification of histochemical staining by color deconvolution", Analytical and Quantitative Cytology and Histology, 2001.

### 3.2 H-Stats per Patch

```python
def compute_h_stats(h_channel_patch):
    """
    Calcule les statistiques H-Channel pour un patch.

    Returns:
        dict: {
            'h_mean': float,        # Intensité moyenne (0-255)
            'h_std': float,         # Écart-type
            'nuclei_count': int,    # Nombre de blobs détectés
            'nuclei_area_ratio': float  # Surface noyaux / surface patch
        }
    """
    # Stats basiques
    h_mean = np.mean(h_channel_patch)
    h_std = np.std(h_channel_patch)

    # Détection blobs (noyaux)
    _, binary = cv2.threshold(h_channel_patch, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Compter les composantes connexes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Filtrer par taille (éliminer bruit)
    MIN_NUCLEUS_AREA = 50   # pixels
    MAX_NUCLEUS_AREA = 5000  # pixels

    valid_nuclei = []
    total_nuclei_area = 0

    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_NUCLEUS_AREA <= area <= MAX_NUCLEUS_AREA:
            valid_nuclei.append({
                'centroid': centroids[i],
                'area': area,
                'bbox': stats[i, :4]
            })
            total_nuclei_area += area

    patch_area = h_channel_patch.shape[0] * h_channel_patch.shape[1]
    nuclei_area_ratio = total_nuclei_area / patch_area

    return {
        'h_mean': h_mean,
        'h_std': h_std,
        'nuclei_count': len(valid_nuclei),
        'nuclei_area_ratio': nuclei_area_ratio,
        'nuclei_details': valid_nuclei
    }
```

### 3.3 Cell Triage v2 (Augmenté)

**Architecture:**

```
Input: CLS (1536D) + H-Stats (4D) = 1540D
    ↓
Linear(1540, 256) + ReLU + Dropout(0.3)
    ↓
Linear(256, 64) + ReLU + Dropout(0.15)
    ↓
Linear(64, 2)
    ↓
Output: Cell / Empty
```

**Entraînement:**
- Dataset: APCData tuiles (comme V15.2)
- Features: CLS token + H-Stats calculés à la volée
- Loss: CrossEntropy avec poids [0.3, 1.0]

### 3.4 Confidence Boosting

```python
def apply_confidence_boosting(patch_prediction, h_stats):
    """
    Ajuste la confiance basée sur la validation H-Channel.

    Règles:
    1. Patch "anormal" MAIS 0 noyau détecté → réduire confiance
    2. Patch "normal" MAIS haute densité noyaux → augmenter vigilance
    3. Forte variance H → possible cluster → augmenter confiance anormal
    """
    confidence = patch_prediction['confidence']
    predicted_class = patch_prediction['class']

    nuclei_count = h_stats['nuclei_count']
    h_std = h_stats['h_std']

    # Règle 1: Anormal sans noyaux = suspect
    if predicted_class != 'NILM' and nuclei_count == 0:
        confidence *= 0.5
        patch_prediction['flag'] = 'LOW_CONFIDENCE_NO_NUCLEI'

    # Règle 2: Normal mais dense = vérifier
    if predicted_class == 'NILM' and nuclei_count > 10:
        patch_prediction['flag'] = 'REVIEW_HIGH_DENSITY'

    # Règle 3: Haute variance H = cluster potentiel
    if h_std > 50 and predicted_class in ['HSIL', 'SCC']:
        confidence = min(confidence * 1.2, 0.99)

    patch_prediction['confidence'] = confidence
    return patch_prediction
```

### 3.5 Détection de Noyaux pour Visualisation

```python
def detect_nuclei_for_visualization(rgb_patch, predicted_class):
    """
    Détecte les noyaux dans un patch pour visualisation cell-level.

    Returns:
        List of nuclei with contours and assigned class
    """
    # Extraire H-Channel
    h_channel = extract_h_channel_ruifrok(rgb_patch)

    # Seuillage adaptatif (meilleur que Otsu pour clusters)
    binary = cv2.adaptiveThreshold(
        h_channel, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=5
    )

    # Inverser si nécessaire (noyaux = sombres dans H)
    binary = 255 - binary

    # Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Watershed pour séparer noyaux collés
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Trouver contours
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrer et assigner classe du patch à chaque noyau
    nuclei = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 5000:  # Filtrer par taille
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                nuclei.append({
                    'contour': contour,
                    'centroid': (cx, cy),
                    'area': area,
                    'class': predicted_class  # Hérite la classe du patch
                })

    return nuclei
```

---

## 4. Visualisation Cell-Level

### 4.1 Mode de Rendu

```python
def render_cell_level_visualization(image, all_nuclei):
    """
    Dessine les noyaux détectés avec couleurs par classe.
    """
    overlay = image.copy()

    CLASS_COLORS = {
        'NILM': (0, 200, 0),      # Vert
        'ASCUS': (0, 255, 255),   # Jaune
        'ASCH': (0, 128, 255),    # Orange
        'LSIL': (0, 200, 255),    # Jaune-Orange
        'HSIL': (0, 0, 255),      # Rouge
        'SCC': (128, 0, 128)      # Violet
    }

    for nucleus in all_nuclei:
        color = CLASS_COLORS.get(nucleus['class'], (200, 200, 200))

        # Dessiner contour
        cv2.drawContours(overlay, [nucleus['contour']], -1, color, 2)

        # Remplissage semi-transparent
        cv2.drawContours(overlay, [nucleus['contour']], -1, color, -1)

    # Blend avec original
    result = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

    return result
```

### 4.2 Comparaison Visuelle

```
V15.2 (Patch-Level):              V15.3 (Cell-Level):
┌────────────────────┐            ┌────────────────────┐
│ ████████████████  │            │    ◯    ◯   ◯     │
│ ████████████████  │     →      │  ◯    ◯      ◯   │
│ ████████████████  │            │    ◯  ◯    ◯     │
└────────────────────┘            └────────────────────┘
   Rectangles 224×224               Contours noyaux
   (non clinique)                   (interprétable)
```

---

## 5. Métriques et Validation

### 5.1 Nouvelles Métriques

| Métrique | Description | Cible |
|----------|-------------|-------|
| **Nuclei Detection Rate** | % noyaux GT détectés via H-Channel | > 85% |
| **False Positive Reduction** | Réduction FP vs V15.2 | > 20% |
| **Cell Count Accuracy** | Corrélation comptage H vs GT | > 0.8 |
| **Visualization Coherence** | % noyaux affichés dans bbox GT | > 90% |

### 5.2 Validation Clinique

1. **Interprétabilité**: Pathologistes peuvent-ils comprendre la visualisation?
2. **Confiance**: Les zones marquées correspondent-elles aux cellules anormales?
3. **Comptage**: Le nombre de cellules affiché est-il cliniquement pertinent?

---

## 6. Plan d'Implémentation

### Phase 1: Extraction H-Channel (Jour 1)
- [ ] Implémenter `extract_h_channel_ruifrok()` dans `src/preprocessing/`
- [ ] Implémenter `compute_h_stats()` dans `src/preprocessing/`
- [ ] Tests unitaires

### Phase 2: Cell Triage v2 (Jour 2-3)
- [ ] Modifier dataset pour inclure H-Stats
- [ ] Entraîner Cell Triage v2 avec features augmentées
- [ ] Évaluer amélioration recall

### Phase 3: Confidence Boosting (Jour 3)
- [ ] Implémenter `apply_confidence_boosting()`
- [ ] Évaluer réduction faux positifs

### Phase 4: Visualisation Cell-Level (Jour 4)
- [ ] Implémenter `detect_nuclei_for_visualization()`
- [ ] Implémenter `render_cell_level_visualization()`
- [ ] Mettre à jour `12_visualize_predictions.py`

### Phase 5: Validation (Jour 5)
- [ ] Benchmark vs V15.2
- [ ] Validation métriques
- [ ] Documentation finale

---

## 7. Références

1. Ruifrok & Johnston, "Quantification of histochemical staining by color deconvolution", Analytical and Quantitative Cytology and Histology, 2001.

2. "Hematoxylin-based nucleus detection in Pap smear cytology", Journal of Cytology, 2021.

3. "Color deconvolution improves cervical cell classification", Biomedical Signal Processing and Control, 2020.

4. "H-channel features for robust Pap smear cytology segmentation", IEEE TMI, 2023.

5. "Cervical cytology triage using nuclear density estimation", IEEE TMI, 2024.

6. "Error reduction in cervical cytology AI using nuclear minimal presence validation", Journal of Pathology Informatics, 2023.

---

## 8. Changelog

| Date | Version | Changements |
|------|---------|-------------|
| 2026-01-24 | v0.1 | Création spécification initiale |

---

**Auteur:** V15.3 Cytology Branch
**Review:** En attente
