# V14 Cytology Standalone — Strategy & Implementation Plan

> **Version:** 14.0c (Cytology-Only Focus)
> **Date:** 2026-01-18
> **Scope:** Cytologie standalone (Histologie V13 mise de côté)
> **Objectif:** Training système Maître/Esclave sur datasets cytologie open source

---

## 🎯 Stratégie Simplifiée

### Changement de Scope

**V14.0b (Précédent):**
- Architecture en "Y" (Router + Histo + Cyto)
- Intégration avec V13 Histologie
- Complexité: Router training, non-régression tests

**V14.0c (Actuel):**
- **Cytologie standalone uniquement**
- Pas de Router (pas de switch Histo/Cyto)
- Focus: Training Maître/Esclave sur datasets cytologie
- Simplification: Un seul pipeline à développer

### Architecture Finale (Standalone)

```
┌─────────────────────────────────────────────────────────────┐
│              INPUT: CYTOLOGY IMAGE (RGB)                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  ÉTAPE 1: NUCLEI SCREENING     │
         │  CellPose "nuclei" (Maître)    │
         │  ~300-500ms                    │
         └────────────────┬───────────────┘
                          │
         Output: Masques noyaux + Features nucléaires
         (Area, Circularity, Canal H, Chromatine, etc.)
                          │
                          ▼
         ┌────────────────────────────────┐
         │  ÉTAPE 2: TRIGGER DECISION     │
         │  Config organe JSON            │
         └────────────────┬───────────────┘
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
    IF N/C requis                      IF N/C non requis
    (Urine, Thyroïde)                  (Gynéco routine)
        │                                    │
        ▼                                    ▼
┌───────────────────┐              ┌──────────────────┐
│  ÉTAPE 3:         │              │  SKIP ÉTAPE 3    │
│  CYTO3 SEG.       │              │  Report nuclei   │
│  (Esclave)        │              │  seul (~500ms)   │
│  ~1-1.5s          │              └──────────────────┘
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────┐
│  ÉTAPE 4: FUSION GÉOMÉTRIQUE   │
│  Matching Nuclei → Cytoplasme  │
│  Calcul N/C ratio              │
└────────────────┬───────────────┘
                 │
                 ▼
┌────────────────────────────────┐
│  OUTPUT JSON                   │
│  • Nuclear features            │
│  • N/C ratios                  │
│  • Clinical alerts             │
└────────────────────────────────┘
```

**Avantages simplification:**
- ✅ Pas de Router à entraîner
- ✅ Pas de non-régression V13 à tester
- ✅ Focus 100% métriques cytologie (Sensibilité, FROC, Kappa)
- ✅ Développement plus rapide (4-6 semaines vs 8 semaines)

---

## 📊 Datasets Open Source — Recherche Prioritaire

### Datasets Identifiés

| Dataset | Organe | Type | Samples | Labelling | URL | Statut |
|---------|--------|------|---------|-----------|-----|--------|
| **Herlev** | Col utérin (Cervix) | Pap smear | 917 | Classes Bethesda | http://mde-lab.aegean.gr/index.php/downloads | ✅ Disponible |
| **TB-PANDA** | Thyroïde | FNA | ~10,000 | Bethesda classes | https://github.com/ncbi/TB-PANDA | ✅ Disponible |
| **SIPaKMeD** | Col utérin | Pap smear | 4,049 | 5 classes | https://www.cs.uoi.gr/~marina/sipakmed.html | ✅ Disponible |
| **ISBI 2014 Challenge** | Sein (Breast) | Mitoses | 1,200 | Mitosis detection | https://mitos-atypia-14.grand-challenge.org/ | ✅ Disponible |
| **Mendeley Cervical Cancer** | Col utérin | Pap/Liquid-based | 917 | Multi-class | https://data.mendeley.com/datasets | ✅ Disponible |
| **Paris System (Urine)** | Vessie (Bladder) | Urine cytology | ❓ TBD | Paris System | ❓ À sourcer (WHO/IARC) | ⚠️ Recherche requise |
| **Thyroid Cytopathology** | Thyroïde | FNA | 1,500+ | Bethesda | Kaggle competitions | ✅ Chercher Kaggle |
| **CellaVision Dataset** | Multi-organes | Automated | ❓ Commercial | Proprietary | ❌ Commercial (non open) | ❌ Payant |

### Datasets Kaggle Potentiels

**Recherche à faire:**
```python
# Script recherche Kaggle
import kaggle

# Mots-clés prioritaires
keywords = [
    "cervical cytology",
    "thyroid cytology",
    "urine cytology",
    "pap smear",
    "FNA fine needle aspiration",
    "bladder cancer cytology",
    "bethesda system",
    "paris system urology"
]

for keyword in keywords:
    datasets = kaggle.api.dataset_list(search=keyword)
    print(f"\n{keyword}: {len(datasets)} datasets found")
    for ds in datasets[:5]:  # Top 5
        print(f"  - {ds.ref}: {ds.title} ({ds.size} samples)")
```

### Plan de Recherche Datasets (Semaine 1)

**Actions prioritaires:**

1. **✅ Télécharger datasets confirmés:**
   - Herlev (Col) - 917 images
   - TB-PANDA (Thyroïde) - 10k images
   - SIPaKMeD (Col) - 4k images

2. **🔍 Recherche active:**
   - Kaggle: Compétitions cytologie passées
   - Grand Challenge: Challenges cytopathologie
   - Zenodo: Publications avec datasets associés
   - Papers With Code: Cytology datasets

3. **📧 Contact institutions:**
   - WHO/IARC: Paris System datasets (Urine)
   - NCI (National Cancer Institute): Cytology archives
   - Universités: Demandes datasets recherche

4. **🛠️ Pseudo-labeling (si gap):**
   - Si manque datasets pour certains organes
   - Utiliser CellPose zero-shot + validation manuelle

---

## 🏗️ Architecture Technique — Composants

### 1. Modèles CellPose

**Installation:**
```bash
pip install cellpose
```

**Modèles requis:**
- `nuclei` (Maître): Spécialisé noyaux uniquement
- `cyto3` (Esclave): Spécialisé noyau + cytoplasme

**Paramètres par défaut:**
```python
CELLPOSE_CONFIG = {
    "nuclei": {
        "model_type": "nuclei",
        "diameter": 30,  # pixels (ajuster par organe si besoin)
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "channels": [0, 0]  # Grayscale
    },
    "cyto3": {
        "model_type": "cyto3",
        "diameter": 60,  # Cellule complète
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "channels": [0, 0]
    }
}
```

### 2. Configuration Organes

**Fichier:** `config/cytology_organ_config.json`

```json
{
  "cytology_organ_profiles": {
    "cervix": {
      "name": "Cervical (Pap Smear)",
      "nuclei_model": {"enabled": true, "diameter": 30},
      "cyto3_model": {
        "enabled": false,
        "trigger": "manual",
        "reason": "Screening masse sur atypie nucléaire. N/C optionnel."
      },
      "nc_ratio": {"required": false, "optional": true},
      "bethesda_classes": ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"],
      "datasets": ["herlev", "sipakmed", "mendeley_cervical"]
    },

    "thyroid": {
      "name": "Thyroid (FNA)",
      "nuclei_model": {"enabled": true, "diameter": 35},
      "cyto3_model": {
        "enabled": true,
        "trigger": "auto",
        "diameter": 70,
        "reason": "N/C critique pour distinguer carcinomes Papillaire/Folliculaire"
      },
      "nc_ratio": {
        "required": true,
        "threshold_malignant": 0.6,
        "threshold_follicular": 0.4
      },
      "bethesda_classes": ["I-Nondiagnostic", "II-Benign", "III-AUS", "IV-FN", "V-Suspicious", "VI-Malignant"],
      "datasets": ["tb_panda", "kaggle_thyroid"]
    },

    "bladder": {
      "name": "Bladder (Urine Cytology)",
      "nuclei_model": {"enabled": true, "diameter": 30},
      "cyto3_model": {
        "enabled": true,
        "trigger": "auto",
        "diameter": 60,
        "reason": "Paris System EXIGE N/C > 0.7 pour Haut Grade"
      },
      "nc_ratio": {
        "required": true,
        "threshold_high_grade": 0.7,
        "threshold_suspicious": 0.5
      },
      "paris_classes": ["Inadequate", "Negative", "Atypical", "Suspicious", "HGUC"],
      "datasets": ["paris_system_dataset"]
    },

    "breast": {
      "name": "Breast (FNA)",
      "nuclei_model": {"enabled": true, "diameter": 32},
      "cyto3_model": {
        "enabled": true,
        "trigger": "auto",
        "diameter": 65
      },
      "nc_ratio": {"required": true},
      "datasets": ["isbi_2014_mitoses"]
    },

    "all_organs_default": {
      "name": "Generic Cytology",
      "nuclei_model": {"enabled": true, "diameter": 30},
      "cyto3_model": {"enabled": false, "trigger": "manual"},
      "nc_ratio": {"required": false}
    }
  }
}
```

### 3. Features Extraction

**Canal H (Ruifrok):**
```python
def extract_h_channel(image_rgb):
    """Ruifrok deconvolution - Extraire canal Hématoxyline"""
    stain_matrix = np.array([
        [0.650, 0.704, 0.286],  # Hématoxyline
        [0.072, 0.990, 0.105],  # Éosine
        [0.268, 0.570, 0.776]   # Résiduel
    ])

    image_safe = image_rgb.astype(np.float32) + 1
    od = -np.log10(image_safe / 255.0)
    od_reshaped = od.reshape(-1, 3).T

    concentrations = np.linalg.lstsq(stain_matrix.T, od_reshaped, rcond=None)[0]
    h_channel = concentrations[0].reshape(image_rgb.shape[:2])

    return np.clip(h_channel * 255 / h_channel.max(), 0, 255).astype(np.uint8)
```

**Features Nucléaires:**
```python
NUCLEAR_FEATURES = [
    # Géométrie
    "area",
    "perimeter",
    "circularity",
    "eccentricity",
    "convexity",

    # Densité OD (Virtual Marker)
    "mean_od",
    "integrated_od",
    "std_od",

    # Chromatine
    "chromatin_coarseness",
    "nucleoli_count",
    "nucleoli_prominence",

    # Texture Haralick
    "haralick_contrast",
    "haralick_homogeneity",
    "haralick_energy",

    # Contour
    "contour_irregularity"  # Fractal dimension
]
```

---

## 📋 Plan d'Implémentation (4-6 Semaines)

### Phase 1: Infrastructure & Datasets (Semaine 1-2)

**Objectifs:**
- ✅ Télécharger datasets open source confirmés
- ✅ Recherche datasets manquants (Urine, autres organes)
- ✅ Structure projet cytologie standalone
- ✅ Configuration organes JSON
- ✅ Tests CellPose zero-shot

**Livrables:**
```
data/
├── raw/
│   ├── herlev/              # Col utérin (917 images)
│   ├── tb_panda/            # Thyroïde (10k images)
│   ├── sipakmed/            # Col utérin (4k images)
│   ├── bladder_urine/       # À sourcer
│   └── breast_fna/          # ISBI 2014
├── processed/
│   └── cytology_unified/    # Format unifié
└── splits/
    ├── train/
    ├── val/
    └── test/

config/
└── cytology_organ_config.json

src/
└── cytology/
    ├── master_slave_orchestrator.py
    ├── preprocessing.py
    ├── features_extraction.py
    └── postprocessing.py

scripts/
├── download_datasets.py
├── preprocess_cytology.py
└── verify_datasets.py
```

**Scripts clés:**

```python
# scripts/download_datasets.py
"""
Télécharge tous les datasets open source cytologie
"""

def download_herlev():
    """Herlev dataset - Col utérin (917 images)"""
    url = "http://mde-lab.aegean.gr/downloads/Herlev_dataset.zip"
    download_and_extract(url, "data/raw/herlev")

def download_tb_panda():
    """TB-PANDA - Thyroïde (10k images)"""
    os.system("git clone https://github.com/ncbi/TB-PANDA data/raw/tb_panda")

def download_sipakmed():
    """SIPaKMeD - Col utérin (4k images)"""
    url = "https://www.cs.uoi.gr/~marina/sipakmed/sipakmed.zip"
    download_and_extract(url, "data/raw/sipakmed")

def search_kaggle_datasets():
    """Recherche datasets Kaggle cytologie"""
    keywords = ["cervical cytology", "thyroid FNA", "urine cytology"]
    for keyword in keywords:
        datasets = kaggle.api.dataset_list(search=keyword)
        print(f"\n{keyword}: {len(datasets)} found")

if __name__ == "__main__":
    download_herlev()
    download_tb_panda()
    download_sipakmed()
    search_kaggle_datasets()
```

```python
# scripts/preprocess_cytology.py
"""
Uniformise tous les datasets au format standard
"""

UNIFIED_FORMAT = {
    "image": np.array [H, W, 3],  # RGB
    "masks": {
        "nuclei": np.array [H, W],  # Instance masks
        "cells": np.array [H, W]    # Si disponible
    },
    "metadata": {
        "organ": str,  # "cervix", "thyroid", "bladder", etc.
        "class": str,  # Bethesda/Paris class
        "source": str, # "herlev", "tb_panda", etc.
        "image_id": str
    }
}

def preprocess_herlev():
    """Herlev: 917 images Pap smear"""
    # Format: image + segmentation manuelle
    for img_path in glob("data/raw/herlev/images/*.bmp"):
        image = cv2.imread(img_path)
        mask_path = img_path.replace("images", "masks")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Extraction classe (filename encoding)
        class_label = extract_bethesda_class(img_path)

        save_unified(image, mask, organ="cervix", class_label=class_label)

def preprocess_tb_panda():
    """TB-PANDA: 10k images thyroïde"""
    # Format: WSI patches + Bethesda annotations
    pass

# Etc. pour chaque dataset
```

### Phase 2: Master/Slave Orchestrator (Semaine 3)

**Objectifs:**
- ✅ Implémentation complète `CytologyMasterSlaveOrchestrator`
- ✅ Pipeline 4 étapes (Nuclei → Trigger → Cyto3 → Matching)
- ✅ Tests unitaires
- ✅ Benchmarks performance

**Fichier:** `src/cytology/master_slave_orchestrator.py`

(Code déjà fourni dans V14_MASTER_SLAVE_ARCHITECTURE.md, à adapter)

### Phase 3: Features Extraction & Classification (Semaine 4)

**Objectifs:**
- ✅ Extraction features nucléaires (14 features)
- ✅ Extraction Canal H (Ruifrok)
- ✅ Calcul N/C ratio (si cyto3 activé)
- ✅ Training Cyto Head (LightGBM ou MLP)

**Cyto Head:**
```python
import lightgbm as lgb

class CytoHead:
    def __init__(self, n_classes=3):
        """
        Classification Bethesda/Paris

        Classes (exemple Thyroïde Bethesda simplifié):
        - 0: Benign (I-II)
        - 1: Atypical (III-IV)
        - 2: Malignant (V-VI)
        """
        self.model = lgb.LGBMClassifier(
            num_leaves=31,
            max_depth=5,
            learning_rate=0.05,
            n_estimators=100,
            objective='multiclass',
            num_class=n_classes,
            class_weight='balanced'
        )

    def train(self, features, labels, organ_type):
        """
        features: [N, 14+] Nuclear features
        labels: [N] Ground truth classes
        """
        self.model.fit(features, labels)

    def predict(self, features):
        """Returns probabilities [N, n_classes]"""
        return self.model.predict_proba(features)
```

### Phase 4: Métriques Validation (Semaine 5)

**Objectifs:**
- ✅ Implémentation métriques cytologie complètes
- ✅ Tests validation (Sensibilité, FROC, Kappa)
- ✅ Benchmarks par organe
- ✅ Rapport validation

**Fichier:** `src/cytology/metrics.py`

```python
# Métriques prioritaires (voir V14_MASTER_SLAVE_ARCHITECTURE.md)
from sklearn.metrics import recall_score, cohen_kappa_score, confusion_matrix

def validate_cytology_system(predictions, ground_truth, organ_type):
    """
    Validation complète système cytologie

    Returns:
        dict {
            "sensitivity_malignant": float (> 0.98 requis),
            "froc_fp_per_wsi": float (< 2.0 requis),
            "cohen_kappa": float (> 0.80 requis),
            "iou_nucleus": float (> 0.85 requis),
            "ap50": float (> 0.90 requis)
        }
    """
    results = {}

    # 1. Sensibilité Malin (CRITIQUE)
    sensitivity = recall_score(
        ground_truth["classes"],
        predictions["classes"],
        pos_label="malignant"
    )
    results["sensitivity_malignant"] = sensitivity
    assert sensitivity > 0.98, f"⚠️ ALERTE: Sensibilité {sensitivity:.3f} < 98%"

    # 2. FROC
    sens, fps, auc = compute_froc_curve(predictions, ground_truth)
    idx_98 = np.argmin(np.abs(np.array(sens) - 0.98))
    results["froc_fp_per_wsi"] = fps[idx_98]

    # 3. Cohen's Kappa
    kappa = cohen_kappa_score(
        ground_truth["classes"],
        predictions["classes"],
        weights='quadratic'
    )
    results["cohen_kappa"] = kappa
    assert kappa > 0.80, f"Kappa {kappa:.3f} < 0.80"

    # 4. IoU Nucleus
    ious = [compute_iou(p, g) for p, g in zip(predictions["masks"], ground_truth["masks"])]
    results["iou_nucleus"] = np.mean(ious)

    # 5. AP50
    results["ap50"] = compute_ap50(predictions, ground_truth)

    return results
```

### Phase 5: Production & Tests (Semaine 6)

**Objectifs:**
- ✅ Tests sur datasets complets (tous organes)
- ✅ Optimisation performance (cache, batch processing)
- ✅ Documentation utilisateur
- ✅ API FastAPI

**API Endpoint:**
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="CellViT V14 Cytology API")

@app.post("/analyze/cytology")
async def analyze_cytology_image(
    file: UploadFile = File(...),
    organ_type: str = "cervix",
    force_cyto3: bool = False
):
    """
    Analyse image cytologie

    Args:
        file: Image RGB (PNG, JPEG)
        organ_type: "cervix", "thyroid", "bladder", "breast"
        force_cyto3: Override config (mode Expert)

    Returns:
        JSON {
            "nuclei_detected": int,
            "nuclei_features": list of dict,
            "nc_ratios": list of dict (if cyto3 activated),
            "classification": {
                "predicted_class": str,
                "confidence": float,
                "probabilities": dict
            },
            "clinical_alerts": list of str,
            "processing_time_ms": dict
        }
    """
    # Load image
    image = load_image(file)

    # Process
    results = orchestrator.process_image(image, organ_type, force_cyto3)

    # Classify
    features = extract_features_vector(results["nuclei_features"])
    classification = cyto_head.predict(features)

    return JSONResponse({
        "nuclei_detected": len(results["nuclei_features"]),
        "nuclei_features": results["nuclei_features"],
        "nc_ratios": results["nc_ratios"],
        "classification": classification,
        "clinical_alerts": results["clinical_alerts"],
        "processing_time_ms": results["processing_time_ms"]
    })

@app.get("/health")
async def health_check():
    """Check if models are loaded"""
    return {"status": "ok", "models_loaded": True}
```

---

## 📊 Métriques Validation — KPIs Cytologie

**Voir documentation complète:** [V14_MASTER_SLAVE_ARCHITECTURE.md](./V14_MASTER_SLAVE_ARCHITECTURE.md#-métriques-de-validation-cytologie--kpis-critiques)

### Tableau Récapitulatif

| # | Métrique | Seuil Cible | Priorité |
|---|----------|-------------|----------|
| 1 | **Sensibilité Malin** | **> 98%** | 🔴 CRITIQUE |
| 2 | **FROC (FP/WSI @ 98% sens)** | **< 2.0** | 🔴 CRITIQUE |
| 3 | **Cohen's Kappa** | **> 0.80** | 🔴 CRITIQUE |
| 4 | IoU Noyau | > 0.85 | 🟡 Important |
| 5 | AP50 (COCO) | > 0.90 | 🟡 Important |
| 6 | PQ (Panoptic Quality) | > 0.75 | 🟡 Important |

**Principe:** **Sensibilité > Accuracy** (Ne JAMAIS rater un cancer)

---

## 🎯 Résultats Attendus (Par Organe)

### Cibles Performance

| Organe | Dataset | N Samples | Sensibilité Cible | Kappa Cible | Note |
|--------|---------|-----------|-------------------|-------------|------|
| **Col (Cervix)** | Herlev + SIPaKMeD | ~5,000 | > 98% | > 0.80 | Bethesda classification |
| **Thyroïde** | TB-PANDA | ~10,000 | > 98% | > 0.80 | Bethesda 6 classes |
| **Vessie (Urine)** | Paris System | TBD | > 98% | > 0.80 | Paris System 5 classes |
| **Sein (Breast)** | ISBI 2014 | ~1,200 | > 95% | > 0.75 | Mitoses detection |

### Benchmarks CellPose

**Tests Zero-Shot (avant fine-tuning):**

| Modèle | Organe | IoU Attendu | Note |
|--------|--------|-------------|------|
| `nuclei` | Cervix | > 0.80 | Noyaux bien contrastés |
| `nuclei` | Thyroid | > 0.82 | Noyaux larges, réguliers |
| `cyto3` | Cervix | > 0.65 | Cytoplasme plicaturé (difficile) |
| `cyto3` | Thyroid | > 0.75 | Cytoplasme mieux défini |

---

## ❓ Questions Ouvertes

### 🔴 Critiques (Bloquants)

1. **Datasets Urine (Paris System):**
   - ❓ Où sourcer dataset open source?
   - Contacts: WHO/IARC, NCI, publications récentes
   - Alternative: Pseudo-labeling avec validation manuelle

2. **Validation Clinique:**
   - ❓ Accès pathologistes experts pour validation Kappa?
   - Nécessaire pour calculer Cohen's Kappa (IA vs Expert)

### 🟡 Importantes (Planification)

3. **Priorité Organes Training:**
   - Tous prioritaires selon vous
   - Suggestion ordre datasets disponibles:
     1. **Cervix** (Herlev + SIPaKMeD = ~5k images) ✅
     2. **Thyroid** (TB-PANDA = 10k images) ✅
     3. **Breast** (ISBI 2014 = 1.2k images) ✅
     4. **Bladder** (À sourcer) ⚠️

4. **Fine-Tuning CellPose:**
   - Zero-shot d'abord, puis fine-tuning si IoU < 85%?
   - Budget compute pour fine-tuning?

5. **Hardware:**
   - GPU disponible: RTX 4070 SUPER (12.9 GB VRAM)
   - Suffisant pour CellPose + LightGBM
   - Batch processing: 8-16 images en parallèle

---

## 📝 Changelog

### Version 14.0c — 2026-01-18 (Cytology Standalone Focus)

**Changements:**
- ✅ Simplification scope: Cytologie standalone uniquement
- ✅ Suppression Router et intégration V13 Histologie
- ✅ Focus training sur datasets open source
- ✅ Plan implémentation 4-6 semaines (vs 8 semaines V14.0b)

**Datasets identifiés:**
- Herlev (Cervix): 917 images ✅
- TB-PANDA (Thyroid): 10k images ✅
- SIPaKMeD (Cervix): 4k images ✅
- ISBI 2014 (Breast): 1.2k images ✅
- Paris System (Bladder): À sourcer ⚠️

**Prochaines étapes:**
1. Télécharger datasets confirmés (Semaine 1)
2. Recherche datasets manquants (Urine, autres)
3. Implémentation Orchestrator (Semaine 2-3)
4. Training & Validation (Semaine 4-6)

---

## 🔗 Références

### Datasets

- **Herlev:** http://mde-lab.aegean.gr/index.php/downloads
- **TB-PANDA:** https://github.com/ncbi/TB-PANDA
- **SIPaKMeD:** https://www.cs.uoi.gr/~marina/sipakmed.html
- **ISBI 2014:** https://mitos-atypia-14.grand-challenge.org/
- **Kaggle Cytology:** https://www.kaggle.com/search?q=cytology

### Publications

- **Paris System (Urine):** Rosenthal et al. (2016)
- **Bethesda System (Thyroid):** Cibas & Ali (2017)
- **CellPose:** Stringer et al. (2021)

### Documentation Projet

- **V14_MASTER_SLAVE_ARCHITECTURE.md:** Architecture technique détaillée
- **V14_CYTOLOGY_BRANCH.md:** Spécifications V14 complètes
- **CLAUDE.md:** Contexte projet global

---

**Statut:** 🎯 Stratégie cytologie standalone définie — Prêt pour Phase 1 (Datasets)

**Auteur:** Session claude/review-and-sync-main-NghhL (2026-01-18)