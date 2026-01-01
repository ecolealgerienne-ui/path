# Spécification Technique : Module v14.0 (Triage WSI)

> **Projet:** CellViT-Optimus
> **Version:** v14.0
> **Statut:** Prêt pour développement
> **Date:** 2026-01-01
> **Objectif:** < 2 minutes par lame (WSI) via sélection intelligente de patches

---

## 1. Vue d'Ensemble

Le module v14.0 implémente un **pipeline de triage pyramidal** pour les Whole Slide Images (WSI).
L'objectif est de réduire drastiquement le temps d'analyse en ne traitant que les régions
diagnostiquement pertinentes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE TRIAGE WSI v14.0                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WSI (.svs/.ndpi)                                                          │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────┐                                                       │
│   │ NIVEAU 1.25×    │  ← Masque Tissu (Otsu)                               │
│   │ < 1 seconde     │     Élimination verre vide                           │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │ NIVEAU 5×       │  ← CleaningNet + H-Channel                           │
│   │ < 3.5 secondes  │     ROI Heatmap (hotspots)                           │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │ NIVEAU 40×      │  ← Moteur v13 (HoVerNet)                             │
│   │ Patches ciblés  │     Inférence sur ~2000 patches max                  │
│   └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1 : Détection du Tissu (Niveau 1.25×)

### 2.1 Spécifications

| Paramètre | Valeur |
|-----------|--------|
| **Entrée** | Image WSI basse résolution (1.25× ou équivalent) |
| **Algorithme** | Filtre de saturation (Otsu) ou masquage binaire rapide |
| **Sortie** | Masque binaire de tissu (Tissue Mask) |
| **Contrainte temps** | < 1.0 seconde (± 0.5s) |

### 2.2 Algorithme

```python
def generate_tissue_mask(wsi_thumbnail: np.ndarray) -> np.ndarray:
    """
    Génère un masque binaire du tissu à partir de la miniature WSI.

    Args:
        wsi_thumbnail: Image RGB à 1.25× (ou niveau pyramidal le plus bas)

    Returns:
        Masque binaire (tissu=1, verre=0)
    """
    # Conversion HSV pour isoler la saturation
    hsv = cv2.cvtColor(wsi_thumbnail, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    # Seuillage Otsu sur la saturation
    threshold, mask = cv2.threshold(
        saturation, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask
```

---

## 3. Phase 2 : ROI Heatmap & CleaningNet (Niveau 5×)

### 3.1 Architecture CleaningNet v14.0

| Paramètre | Valeur |
|-----------|--------|
| **Architecture** | MobileNetV3-Small ou EfficientNet-B0 |
| **Entrée** | Patch RGB (224×224) + H-Channel |
| **Tâche** | Classification binaire (Informative vs Non-informative) |
| **Sortie** | Score de probabilité P_ROI ∈ [0, 1] |

### 3.2 Extraction H-Channel (Mandatoire)

> **CRITIQUE:** Le canal H est extrait via la matrice de Ruifrok (Beer-Lambert fixe),
> identique à la configuration v13 production.

```python
from src.preprocessing.stain_extraction import extract_h_channel_ruifrok

def prepare_cleaningnet_input(patch_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prépare l'entrée pour CleaningNet: RGB + H-Channel.
    """
    h_channel = extract_h_channel_ruifrok(patch_rgb)
    return patch_rgb, h_channel
```

### 3.3 Seuil Dynamique par Organe

> **Décision:** Le seuil P_ROI est adapté selon l'organe pour maintenir Recall > 0.95.

| Organe | Seuil P_ROI | Justification |
|--------|-------------|---------------|
| **Foie** (dense) | 0.40 | Haute densité cellulaire |
| **Poumon** | 0.35 | Densité variable |
| **Os** (sparse) | 0.20 | Peu de cellules |
| **Défaut** | 0.30 | Valeur par défaut |

```python
ORGAN_ROI_THRESHOLDS = {
    "Liver": 0.40,
    "Lung": 0.35,
    "Bone": 0.20,
    "Kidney": 0.35,
    "Breast": 0.30,
    # ... autres organes
    "default": 0.30,
}

def get_roi_threshold(organ: str) -> float:
    return ORGAN_ROI_THRESHOLDS.get(organ, ORGAN_ROI_THRESHOLDS["default"])
```

### 3.4 Stratégie de Voisinage Conditionnel

> **Décision:** Inclure un voisin uniquement si P_ROI(voisin) > 0.1

```python
def select_patches_with_neighbors(
    roi_scores: Dict[Tuple[int, int], float],
    threshold: float = 0.30,
    neighbor_threshold: float = 0.10,
) -> Set[Tuple[int, int]]:
    """
    Sélectionne les patches avec voisinage conditionnel.

    Args:
        roi_scores: Dict {(x, y): P_ROI}
        threshold: Seuil principal (organe-dépendant)
        neighbor_threshold: Seuil pour inclusion des voisins

    Returns:
        Set des coordonnées sélectionnées
    """
    selected = set()

    for (x, y), score in roi_scores.items():
        if score > threshold:
            selected.add((x, y))

            # Voisinage conditionnel (8-connecté)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor in roi_scores:
                        if roi_scores[neighbor] > neighbor_threshold:
                            selected.add(neighbor)

    return selected
```

### 3.5 Calcul de H-Entropy

> **Définition:** Entropie de Shannon sur l'histogramme du canal H.
> **Usage:** Garde-fou contre zones pauvres en information (stroma hyalin, graisse).

$$H_{entropy} = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

```python
def compute_h_entropy(h_channel: np.ndarray, bins: int = 64) -> float:
    """
    Calcule l'entropie du canal H.

    Un patch avec entropie très basse sera rejeté même si P_ROI élevé
    (ex: artéfact de coloration uniforme).
    """
    hist, _ = np.histogram(h_channel.flatten(), bins=bins, range=(0, 255))
    hist = hist / hist.sum()  # Normaliser en probabilités

    # Éviter log(0)
    hist = hist[hist > 0]

    entropy = -np.sum(hist * np.log2(hist))
    return entropy

# Seuil recommandé
H_ENTROPY_MIN = 2.0  # Rejeter si entropy < 2.0
```

---

## 4. Phase 3 : Smart Patching & Inférence (Niveau 40×)

### 4.1 Mapping Résolution 5× → 40×

> **CRITIQUE:** C'est le point le plus important pour estimer la charge.

| Paramètre | Calcul |
|-----------|--------|
| **Ratio résolution** | 40× / 5× = 8 |
| **Ratio aire** | 8² = **64 patches** |
| **Implication** | 1 ROI à 5× = 64 patches à 40× |

```
Exemple de charge:
- CleaningNet sélectionne 31 ROIs à 5×
- Total patches 40× = 31 × 64 = 1,984 patches
- Cible v14: ~2,000 patches max
```

### 4.2 Extraction Sélective

```python
def extract_40x_patches(
    wsi: openslide.OpenSlide,
    selected_5x_coords: Set[Tuple[int, int]],
    patch_size: int = 224,
) -> Generator[Tuple[Tuple[int, int], np.ndarray], None, None]:
    """
    Extrait les patches 40× correspondant aux ROIs sélectionnées à 5×.

    Yields:
        (coords_40x, patch_rgb)
    """
    for (x_5x, y_5x) in selected_5x_coords:
        # Convertir coordonnées 5× → 40×
        base_x = x_5x * 8 * patch_size
        base_y = y_5x * 8 * patch_size

        # Extraire les 64 patches (8×8 grid)
        for i in range(8):
            for j in range(8):
                x_40x = base_x + i * patch_size
                y_40x = base_y + j * patch_size

                patch = wsi.read_region(
                    (x_40x, y_40x),
                    level=0,  # 40×
                    size=(patch_size, patch_size)
                )
                patch_rgb = np.array(patch.convert("RGB"))

                yield ((x_40x, y_40x), patch_rgb)
```

### 4.3 Injection Moteur v13

Les patches extraits sont envoyés en batch au moteur HoVerNet v13:
- Backbone: H-Optimus-0 (gelé)
- Décodeur: FPN Chimique avec injection H-Channel Ruifrok
- Configuration: Production 2026 (voir CLAUDE.md)

### 4.4 Motifs de Sélection (Fusion Asynchrone en 2 Temps)

> **Décision:** Les motifs sont générés en 2 phases distinctes puis fusionnés.
> Le triage (5×) ne peut pas deviner une mitose — seule l'analyse v13 (40×) le peut.

#### Principe de Fusion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STRATÉGIE DE FUSION ASYNCHRONE                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 5× (Triage)              PHASE 40× (Analyse v13)                    │
│  ─────────────────              ────────────────────────                    │
│  motifs_triage = [              motifs_detail = [                          │
│    "🔴 Haute densité",            "⚡ Activité mitotique",                  │
│    "🎨 Hétérogénéité"             "🎯 Cellules néoplasiques"               │
│  ]                              ]                                           │
│         │                              │                                    │
│         └──────────────┬───────────────┘                                    │
│                        ▼                                                    │
│              motifs.extend(motifs_detail)                                   │
│                        │                                                    │
│                        ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │  AFFICHAGE IHM FINAL                                        │           │
│  │  Patch #42: 🔴 Haute densité | 🎨 Hétérogénéité |           │           │
│  │             ⚡ Mitoses | 🎯 Néoplasiques                     │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Implémentation

```python
def generate_motifs_triage(
    h_entropy: float,
    cleaningnet_score: float,
) -> List[str]:
    """
    Phase 5× : Motifs basés sur texture globale et intensité H.
    Exécuté pendant le triage CleaningNet.
    """
    motifs = []

    # Score de cellularité CleaningNet
    if cleaningnet_score > 0.7:
        motifs.append("🔴 Haute densité nucléaire")

    # Entropie du canal H (calculée au triage)
    if h_entropy > 5.0:
        motifs.append("🎨 Hétérogénéité chromatinienne")

    return motifs


def generate_motifs_detail(v13_result: Dict[str, Any]) -> List[str]:
    """
    Phase 40× : Motifs basés sur l'inférence précise HoVerNet v13.
    Exécuté APRÈS segmentation complète.
    """
    motifs = []

    # Détecteur de mitoses v13
    if v13_result.get("mitosis_count", 0) > 0:
        motifs.append("⚡ Activité mitotique")

    # Segmentation nucléaire précise
    if v13_result.get("nc_ratio", 0) > 0.6:
        motifs.append("🧬 Ratio N/C élevé")

    # Classification NC Optimus
    dominant_type = v13_result.get("dominant_cell_type")
    if dominant_type == "Neoplastic":
        motifs.append("🎯 Cellules néoplasiques")
    elif dominant_type == "Inflammatory":
        motifs.append("🔥 Infiltrat inflammatoire")

    # Pléomorphisme (Nottingham Grade)
    pleomorphism = v13_result.get("pleomorphism_grade", 1)
    if pleomorphism >= 3:
        motifs.append("🔍 Atypie sévère (Grade 3)")
    elif pleomorphism == 2:
        motifs.append("🔍 Atypie modérée (Grade 2)")

    # Anomalie détectée
    if v13_result.get("anomaly_score", 0) > 0.8:
        motifs.append("⚠️ Zone suspecte")

    return motifs


def fuse_motifs(
    motifs_triage: List[str],
    motifs_detail: List[str],
) -> List[str]:
    """
    Fusion finale des motifs pour affichage IHM.
    """
    return motifs_triage + motifs_detail
```

#### Vocabulaire Standardisé (v14.0b)

**Motifs Triage (Phase 5×)**

| Emoji | Motif | Critère | Source |
|-------|-------|---------|--------|
| 🔴 | Haute densité nucléaire | CleaningNet score > 0.7 | Signal H-Channel (Ruifrok) |
| 🎨 | Hétérogénéité chromatinienne | H-entropy > 5.0 | Entropie du canal H |

**Motifs Détail (Phase 40× - v13)**

| Emoji | Motif | Critère | Source |
|-------|-------|---------|--------|
| ⚡ | Activité mitotique | mitosis_count > 0 | Détecteur de mitoses v13 |
| 🧬 | Ratio N/C élevé | nc_ratio > 0.6 | Segmentation HoVer-Net |
| 🎯 | Cellules néoplasiques | dominant_type = Neoplastic | Branche NC (Classification) |
| 🔥 | Infiltrat inflammatoire | dominant_type = Inflammatory | Branche NC (Classification) |
| 🔍 | Atypie chromatinienne | pleomorphism_grade ≥ 2 | Analyse morphologique v13 |
| ⚠️ | Zone suspecte | anomaly_score > 0.8 | Score d'anomalie v13 |
| 📐 | Structure glandulaire | gland_pattern detected | Analyse structurelle v13 |

#### Échelle de Pléomorphisme (Nottingham Grade)

> **Standard clinique adopté:** Échelle de Nottingham (1-3) pour compatibilité pathologiste.

| Grade | Description | Critères Morphologiques |
|-------|-------------|------------------------|
| **1** | Faible | Noyaux uniformes, taille proche des lymphocytes |
| **2** | Modéré | Augmentation de taille, nucléoles visibles |
| **3** | Sévère | Variation extrême, formes bizarres, gros nucléoles |

---

### 4.5 Interface IHM : Mini-map GPS

> **Objectif:** Permettre au pathologiste de localiser instantanément le patch dans la WSI.

#### Spécifications Techniques

| Paramètre | Valeur |
|-----------|--------|
| **Source** | `OpenSlide.get_thumbnail()` |
| **Taille** | 256×256 pixels (fixe, format carré) |
| **Format** | JPEG (qualité 85%) |
| **Marqueur** | Rectangle rouge 2×2 pixels |

#### Calcul des Coordonnées du Marqueur

```python
def compute_marker_coords(
    wsi_coords: Tuple[int, int],
    wsi_dimensions: Tuple[int, int],
    thumbnail_size: int = 256,
) -> Tuple[int, int]:
    """
    Convertit les coordonnées WSI en coordonnées thumbnail.

    Args:
        wsi_coords: (X, Y) position du patch dans la WSI (niveau 40×)
        wsi_dimensions: (largeur, hauteur) de la WSI complète
        thumbnail_size: Taille du thumbnail (256×256)

    Returns:
        (x_thumb, y_thumb) coordonnées du marqueur
    """
    wsi_width, wsi_height = wsi_dimensions
    x_wsi, y_wsi = wsi_coords

    # Ratio d'échelle (utiliser la plus grande dimension)
    ratio = max(wsi_width, wsi_height) / thumbnail_size

    # Coordonnées thumbnail
    x_thumb = int(x_wsi / ratio)
    y_thumb = int(y_wsi / ratio)

    return (x_thumb, y_thumb)


def draw_marker_on_thumbnail(
    thumbnail: np.ndarray,
    marker_coords: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 0, 0),  # Rouge
    size: int = 4,
) -> np.ndarray:
    """
    Dessine un marqueur rectangulaire sur le thumbnail.
    """
    x, y = marker_coords
    cv2.rectangle(
        thumbnail,
        (x - size // 2, y - size // 2),
        (x + size // 2, y + size // 2),
        color,
        thickness=2,
    )
    return thumbnail
```

#### Interactivité

- **Mise à jour dynamique:** Le marqueur se déplace lors du changement de patch dans la liste "Top Priorités"
- **Clic sur mini-map:** Navigation directe vers le patch correspondant (optionnel v14.1)

---

## 5. Contrôle Qualité (QC)

### 5.1 Détection du Flou (Blur Detection)

> **Méthode:** Variance du Laplacien sur le canal H à 5×.

```python
def detect_blur(h_channel: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Détecte si un patch est flou (out-of-focus).

    Returns:
        True si le patch est flou (à rejeter)
    """
    laplacian = cv2.Laplacian(h_channel, cv2.CV_64F)
    variance = laplacian.var()

    return variance < threshold  # Flou si variance trop basse
```

### 5.2 Validation Anti-Miss (Safety Check)

> **CRITIQUE:** Assurance vie contre un CleaningNet qui raterait des tumeurs rares.

```python
def safety_check_rejected_patches(
    rejected_patches: List[np.ndarray],
    model_v13: HoVerNetDecoder,
    k: int = 50,
    anomaly_threshold: float = 0.7,
) -> List[Dict]:
    """
    Échantillonne aléatoirement k patches parmi les rejetés
    et vérifie qu'aucun ne contient de signal fort.

    Returns:
        Liste des alertes si anomalies détectées
    """
    if len(rejected_patches) < k:
        sample = rejected_patches
    else:
        sample = random.sample(rejected_patches, k)

    alerts = []
    for idx, patch in enumerate(sample):
        # Inférence rapide v13
        result = model_v13.quick_inference(patch)

        if result["np_score"] > anomaly_threshold:
            alerts.append({
                "patch_idx": idx,
                "np_score": result["np_score"],
                "message": "Potential missed region!"
            })

    return alerts
```

---

## 6. Schéma de Sortie JSON (v14.0b)

Le module v14 génère un fichier de métadonnées structuré en sections distinctes.

### 6.1 Structure Globale WSI

```json
{
  "wsi_id": "341012D0",
  "organ": "Liver",
  "roi_threshold_used": 0.40,
  "processing_stats": {
    "total_patches_available": 15000,
    "tissue_patches_1_25x": 8500,
    "roi_patches_5x": 31,
    "selected_patches_40x": 1984,
    "triage_time_sec": 4.5
  },
  "quality_control": {
    "blur_rejected_count": 12,
    "low_entropy_rejected_count": 5,
    "safety_check_alerts": []
  },
  "wsi_map": {
    "thumbnail_url": "cache/341012D0_thumb.jpg",
    "dimensions": [256, 256],
    "wsi_dimensions": [98304, 65536]
  },
  "patches": []
}
```

### 6.2 Structure Patch (Fusion Motifs v14.0b)

> **Changements v14.0b:**
> - Séparation `triage` / `analyse_v13`
> - `motifs_triage` (Phase 5×) vs `motifs_detail` (Phase 40×)
> - `pleomorphism_grade` (Nottingham 1-3) remplace `pleomorphism_score`
> - Ajout `wsi_map.marker_coords` pour GPS

```json
{
  "patch_id": "341012D0_x1024_y2048",
  "triage": {
    "confiance_selection": 0.89,
    "h_entropy": 5.2,
    "blur_score": 245.3,
    "motifs_triage": [
      "🔴 Haute densité nucléaire",
      "🎨 Hétérogénéité chromatinienne"
    ],
    "coords_5x": [128, 256],
    "is_hotspot": true
  },
  "analyse_v13": {
    "confiance_segmentation": "Élevée",
    "nuclei_count": 127,
    "pleomorphism_grade": 3,
    "mitosis_count": 2,
    "nc_ratio": 0.72,
    "dominant_cell_type": "Neoplastic",
    "neoplastic_ratio": 0.85,
    "anomaly_score": 0.45,
    "motifs_detail": [
      "⚡ Activité mitotique",
      "🎯 Cellules néoplasiques",
      "🔍 Atypie sévère (Grade 3)"
    ]
  },
  "wsi_map": {
    "marker_coords": [128, 45]
  },
  "coords_40x": [1024, 2048],
  "priority_rank": 3,
  "priority_label": "Élevé (Top 5%)"
}
```

### 6.3 Niveaux de Confiance

| Valeur | Description | Critère |
|--------|-------------|---------|
| **Élevée** | Segmentation fiable | OrganHead > 0.90 |
| **Modérée** | Vérification recommandée | OrganHead 0.70-0.90 |
| **Faible** | Révision manuelle requise | OrganHead < 0.70 |

### 6.4 Badge de Priorité

Le `priority_label` est calculé selon le rang du patch dans la liste triée par score d'intérêt:

| Rang | Label | Couleur IHM |
|------|-------|-------------|
| Top 5% | "Élevé (Top 5%)" | Rouge |
| Top 20% | "Modéré (Top 20%)" | Orange |
| Reste | "Standard" | Gris |

```python
def compute_priority_label(rank: int, total: int) -> str:
    percentile = rank / total
    if percentile <= 0.05:
        return "Élevé (Top 5%)"
    elif percentile <= 0.20:
        return "Modéré (Top 20%)"
    else:
        return "Standard"
```

---

## 7. Contraintes de Performance (KPIs)

> **KPIs Cibles v14.0:** Temps total < 2 min | Triage < 5s | **Sensibilité > 95%**

### 7.1 Temps d'Exécution

| Étape | Temps Cible | Tolérance | Notes |
|-------|-------------|-----------|-------|
| Masque 1.25× | 1.0 s | ± 0.5 s | Otsu + morphologie |
| CleaningNet 5× | 3.5 s | ± 1.0 s | GPU batch inference |
| Sélection & I/O | 0.5 s | ± 0.2 s | Mapping + validation |
| **Total v14 Pre-moteur** | **≤ 5.0 s** | **Stricte** | Avant envoi à v13 |

### 7.2 Contrainte de Charge

> **Règle:** Pour rester sous les 2 minutes totales, le CleaningNet doit sélectionner
> **maximum 30-40 ROIs à 5×** par lame.

| Paramètre | Valeur | Calcul |
|-----------|--------|--------|
| ROIs max à 5× | 30-40 | Contrainte de temps |
| Patches 40× par ROI | 64 | 8×8 (mapping 1:64) |
| **Patches 40× max** | **1,920 - 2,560** | 30×64 à 40×64 |

### 7.3 Qualité de Sélection

| Métrique | Cible | Méthode de Validation |
|----------|-------|----------------------|
| **Sensibilité (Recall)** | **> 95%** | Test Anti-Miss sur patches rejetés |
| Spécificité | > 70% | Ratio patches informatifs/sélectionnés |
| Cohérence H-entropy | 100% | Même canal H que décodeur v13 |

---

## 8. Stack Technique

### 8.1 Librairies I/O

| Librairie | Usage | Avantages |
|-----------|-------|-----------|
| **OpenSlide** | Lecture WSI | Portable, tous formats (.svs, .ndpi, .tiff) |
| **cuCIM** | Traitement GPU | Accélération Ruifrok, tuilage rapide |

```python
# Configuration recommandée
import openslide  # Lecture
import cucim      # Traitement GPU (si disponible)

def load_wsi(path: str):
    """OpenSlide pour lecture, cuCIM pour traitement."""
    wsi = openslide.OpenSlide(path)
    return wsi
```

### 8.2 Gestion Mémoire

> **RÈGLE:** Ne jamais charger la WSI complète en RAM.

```python
# Tuilage dynamique
TILE_SIZE = 1024  # Pixels
MAX_TILES_IN_MEMORY = 16  # ~64 MB pour RGB

class TiledWSIReader:
    """Lecteur avec cache LRU pour tuiles."""

    def __init__(self, wsi_path: str, cache_size: int = 16):
        self.wsi = openslide.OpenSlide(wsi_path)
        self.cache = LRUCache(maxsize=cache_size)

    def get_tile(self, x: int, y: int, level: int) -> np.ndarray:
        key = (x, y, level)
        if key not in self.cache:
            tile = self.wsi.read_region((x, y), level, (TILE_SIZE, TILE_SIZE))
            self.cache[key] = np.array(tile.convert("RGB"))
        return self.cache[key]
```

---

## 9. Stratégie de Données (Training CleaningNet)

### 9.1 Pseudo-Labeling avec v13

> **Décision:** Pas d'annotation manuelle. Utiliser HoVerNet v13 (AJI 0.72) comme oracle.

```
Pipeline Pseudo-Labeling:
1. Sélectionner 50 WSI représentatives (5 par famille)
2. Exécuter HoVerNet v13 sur TOUTES les tuiles (overnight job)
3. Pour chaque tuile 5×:
   - Compter le nombre de noyaux détectés
   - Calculer la diversité des types cellulaires
4. Label "Informative" si:
   - nuclei_count > 50 OU
   - type_diversity > 3 classes
5. Entraîner CleaningNet sur ce dataset
```

### 9.2 Dataset Estimé

| Paramètre | Valeur |
|-----------|--------|
| WSI sources | 50 lames |
| Tuiles 5× par WSI | ~3,000 |
| Total tuiles | ~150,000 |
| Split train/val | 80/20 |
| Classes | 2 (Informative / Non-informative) |

---

## 10. Roadmap Implémentation

### Phase A: Infrastructure (Semaine 1-2)
- [ ] Intégration OpenSlide + cuCIM
- [ ] TiledWSIReader avec cache LRU
- [ ] Pipeline extraction H-Channel GPU

### Phase B: CleaningNet (Semaine 3-4)
- [ ] Pseudo-labeling sur 50 WSI
- [ ] Training MobileNetV3-Small
- [ ] Validation Recall > 0.95

### Phase C: Intégration (Semaine 5-6)
- [ ] Pipeline complet Phase 1→2→3
- [ ] QC (blur, entropy, safety check)
- [ ] Benchmark KPIs

### Phase D: Production (Semaine 7-8)
- [ ] Tests sur 100 WSI variées
- [ ] Optimisation mémoire/latence
- [ ] Documentation API

---

## Références

- **HoVerNet v13:** Voir [CLAUDE.md](../../CLAUDE.md) pour configuration production
- **Ruifrok:** Beer-Lambert stain deconvolution (constantes physiques fixes)
- **H-Optimus-0:** https://huggingface.co/bioptimus/H-optimus-0
- **OpenSlide:** https://openslide.org/
- **cuCIM:** https://github.com/rapidsai/cucim
