# Spécification Pipeline WSI Industriel — CellViT-Optimus

> **Version:** 1.0
> **Date:** 2026-01-26
> **Statut:** DRAFT — En attente validation utilisateur
> **Branche Histologie:** V13 Smart Crops + FPN Chimique

---

## Executive Summary

Ce document spécifie l'architecture d'un pipeline industriel pour le traitement de lames entières (WSI - Whole Slide Images) intégré au système CellViT-Optimus V13 existant.

**Objectifs :**
- Traiter des lames H&E complètes (multi-GB) de manière automatisée
- Exploiter le système V13 existant sans modification
- Respecter les standards industriels (DICOM, IHE, CAP)
- Permettre une évolution vers une certification FDA/CE à terme

**Principe fondamental :**
> Le système V13 (HoVerNet + FPN Chimique) reste le moteur de segmentation.
> Le pipeline WSI est une **couche d'orchestration** autour de l'existant.

---

## Table des Matières

1. [Standards Industriels](#1-standards-industriels)
2. [Architecture Globale](#2-architecture-globale)
3. [Workflow Utilisateur](#3-workflow-utilisateur)
4. [Spécifications Techniques](#4-spécifications-techniques)
5. [Formats Supportés](#5-formats-supportés)
6. [Qualité et Validation](#6-qualité-et-validation)
7. [Intégrations Futures](#7-intégrations-futures)
8. [Roadmap](#8-roadmap)
9. [Références](#9-références)

---

## 1. Standards Industriels

### 1.1 DICOM WSI (Digital Imaging and Communications in Medicine)

**Source:** [NEMA DICOM WSI](https://dicom.nema.org/dicom/dicomwsi/)

Le standard DICOM pour les WSI existe depuis 2010 (Supplement 145) et devient le standard d'interopérabilité de facto.

**Avantages DICOM :**
- Interopérabilité multi-vendeurs (scanners, viewers, PACS)
- Métadonnées patient/spécimen standardisées
- Profils colorimétriques ICC intégrés
- Compatibilité avec l'écosystème hospitalier existant

**Connectathon 2025 :** [Proscia DICOM WSI Connectathon](https://tissuepathology.com/2025/06/26/proscia-demonstrates-seamless-interoperability-at-the-2025-dicom-wsi-connectathon/)
- 8 fabricants de scanners connectés (3DHISTECH, Hamamatsu, Leica, etc.)
- Validation des protocoles C-STORE et STOW
- Confirmation que DICOM WSI est un standard mature

**Recommandation :**
> Support DICOM en lecture (Phase 2) pour compatibilité hospitalière.
> Priorité Phase 1 : formats natifs (.svs, .ndpi, .mrxs) via OpenSlide.

### 1.2 IHE PaLM (Integrating the Healthcare Enterprise - Pathology and Laboratory Medicine)

**Profil DPIA (Digital Pathology Image Acquisition) :**
- Communication des métadonnées patient/spécimen via HL7 V2
- Récupération des identifiants depuis le code-barres de la lame
- Standardisation de l'interface LIS ↔ Scanner

**Workflow IHE :**
```
LIS → Worklist → Scanner → WSI → Image Management System → Viewer
         ↓           ↓              ↓                        ↓
      HL7 V2     Barcode        DICOM/Native           HL7/FHIR
```

**Recommandation :**
> Non prioritaire pour Phase 1 (prototypage).
> À considérer pour intégration LIS en Phase 3.

### 1.3 CAP Guidelines (College of American Pathologists)

**Source:** [CAP WSI Validation Guidelines](https://www.cap.org/protocols-and-guidelines/cap-guidelines/current-cap-guidelines/validating-whole-slide-imaging-for-diagnostic-purposes-in-pathology)

**Exigences clés pour validation diagnostique :**

| Exigence | Description |
|----------|-------------|
| **60 cas minimum** | Par application (diagnostic primaire, frozen section, etc.) |
| **Concordance intra-observateur** | Comparaison verre vs digital, ≥2 semaines d'écart |
| **Documentation QA** | Traçabilité complète du processus |
| **Contrôle des artefacts** | Identification des risques technologiques |

**Document 2025 :** [CAP Practical Tips](https://documents.cap.org/documents/Practical-Tips-to-Assist-Implementation-of-Whole-Slide-Imaging-2025_10_01.pdf)
- Importance de la qualité pré-analytique (grossing, processing)
- Consistency du protocole H&E pour qualité de numérisation optimale

**Recommandation :**
> Prévoir un module de QC (Quality Control) automatisé.
> Logger tous les traitements pour audit trail.

### 1.4 FDA 510(k) — Parcours Réglementaire US

**Clearances récentes (2025) :**

| Produit | Fabricant | Date | Type |
|---------|-----------|------|------|
| [AISight Dx](https://www.pathai.com/news/pathai-receives-fda-clearance-for-aisight-dx-platform-for-primary-diagnosis) | PathAI | Juin 2025 | Image Management |
| [HALO AP Dx](https://indicalab.com/news/press-release/fda-cleared-digital-pathology/) | Indica Labs | Déc 2025 | Enterprise Platform |
| [Prostate Detect](https://www.targetedonc.com/view/fda-grants-510-k-clearance-to-ibex-prostate-detect-ai-for-prostate-cancer) | Ibex | 2025 | AI Diagnostic |

**Tendance : PCCP (Predetermined Change Control Plan)**
- Permet des mises à jour logicielles sans nouvelle soumission 510(k)
- Exige un plan de contrôle des modifications pré-approuvé
- PathAI a obtenu un PCCP pour AISight Dx

**Recommandation :**
> Architecture modulaire facilitant la traçabilité des versions.
> Documentation des datasets d'entraînement et de validation.

### 1.5 Vendors Leaders — Architectures de Référence

#### Sectra Digital Pathology

**Source:** [Sectra Digital Pathology Solution](https://medical.sectra.com/product/sectra-digital-pathology-solution/)

**Points clés :**
- Enterprise Imaging (EI) — plateforme unifiée radio + pathologie
- Workflow orchestration engine avec règles configurables
- Intégration native LIS (Epic Beaker) et PACS existant
- [Pas de silo séparé](https://medical.sectra.com/resources/digitizing-pathology-dont-create-another-silo/) — réutilisation infrastructure existante

**Architecture Sectra :**
```
Scanner (Leica/Aperio) → Sectra EI → Epic Beaker LIS
                              ↓
                         PACS/VNA existant
                              ↓
                         EMR (accès cliniciens)
```

#### Philips IntelliSite

**Source:** [Philips DICOM in Digital Pathology](https://www.usa.philips.com/healthcare/article/dicom-in-digital-pathology)

- Premier système WSI approuvé FDA (2017)
- Format natif .tiff (iSyntax)
- Focus sur le DICOM pour interopérabilité

#### Leica Biosystems / Aperio

**Source:** [Leica DICOM White Paper](https://www.leicabiosystems.com/sites/default/files/media_product-download/2024-12/White_Paper_-_DICOM_3_DEC_2024_240796_Rev_B.pdf)

- Format natif .svs (le plus répandu)
- Premier à offrir DICOM dans un système FDA-approved
- Collaboration avec Sectra pour intégration

---

## 2. Architecture Globale

### 2.1 Principe : Couche d'Orchestration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE WSI INDUSTRIEL                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     COUCHE ORCHESTRATION (NOUVEAU)                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│  │  │ WSI     │  │ Tissue  │  │ Tile    │  │ Feature │  │ Agreg.  │   │   │
│  │  │ Loader  │→ │ Mask    │→ │ Extract │→ │ Cache   │→ │ Report  │   │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     MOTEUR V13 (EXISTANT — INTOUCHÉ)                 │   │
│  │                                                                      │   │
│  │   H-Optimus-0 → FPN Chimique → HoVerNet Decoder → HV-Watershed      │   │
│  │   (1.1B params)   (H-Channel)   (NP + HV + NT)    (Instances)       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     COUCHE STOCKAGE                                  │   │
│  │                                                                      │   │
│  │   Phase 1: Disque Local  →  Phase 2: NAS/S3  →  Phase 3: PACS/VNA  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     COUCHE PRÉSENTATION (IHM GRADIO)                │   │
│  │                                                                      │   │
│  │   Liste Lames → Preview → Lancement → Progress → Résultats/Heatmap │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Composants Détaillés

| Composant | Responsabilité | Technologie | Standard |
|-----------|----------------|-------------|----------|
| **WSI Loader** | Lecture multi-format, métadonnées | OpenSlide + tiffslide | - |
| **Tissue Segmentation** | Segmentation tissu HSV | CLAM pipeline | **CLAM** |
| **QC Artefacts** | Détection pen/blur/folds/bubbles | HistoQC | **HistoQC** |
| **Content Filter** | Exclusion adipose/stroma/low-entropy | H-Channel + Entropie | **HistoROI** |
| **Tile Extract** | Découpage 224×224 sur ROIs filtrés | tile_extraction.py | **CLAM** |
| **Feature Cache** | Cache features H-Optimus-0 | .pt / .npz | - |
| **Inference V13** | Segmentation nucléaire | HoVerNet + FPN Chimique | - |
| **Aggregation** | Stats slide-level, heatmap | ABMIL style | **CLAM** |
| **Report Generator** | JSON structuré, export | Custom | - |

### 2.3 Flux de Données (avec Filtrage CLAM/HistoQC)

```
WSI File (.svs/.ndpi/.mrxs)
    │
    ▼
┌───────────────────────────────────────┐
│ WSI LOADER                            │
│ • Lecture métadonnées (MPP, dims)     │
│ • Génération thumbnail (1024px)       │
│ • Détection format automatique        │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ NIVEAU 1: TISSUE SEGMENTATION (CLAM)  │
│ • Downscale niveau 5× ou 10×          │
│ • Conversion HSV → canal Saturation   │
│ • Otsu thresholding (sthresh=8)       │
│ • Median filter (mthresh=7)           │
│ • Morphological closing (close=4)     │
│ • Extraction contours (four_pt)       │
│ • Élimine ~50-60% (fond blanc)        │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ NIVEAU 2: QC ARTEFACTS (HistoQC)      │
│ • Détection pen markers (HSV color)   │
│ • Détection tissue folds (gradient)   │
│ • Détection air bubbles (circular)    │
│ • Détection blur (Laplacian var)      │
│ • Élimine ~10-20% (artefacts)         │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ NIVEAU 3: CONTENT FILTER (HistoROI)   │
│ • Exclusion adipose tissue            │
│ • Exclusion low entropy (<4.0)        │
│ • Exclusion no nuclei (H-channel<5%)  │
│ • Élimine ~10-15% (non informatif)    │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ TILE EXTRACTION (niveau 40× / 0.5MPP) │
│ • Grille 224×224 sur ROIs filtrés     │
│ • ~20-30% des tiles initiaux gardés   │
│ • Sauvegarde tiles + coordonnées      │
│ • Metadata filtrage par tile          │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ H-OPTIMUS-0 FEATURE EXTRACTION        │
│ • Batch inference (GPU)               │
│ • 261 tokens × 1536D par tile         │
│ • Cache .pt pour réutilisation        │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ HOVERNET V13 INFERENCE                │
│ • FPN Chimique + H-Channel Ruifrok    │
│ • Branches NP + HV + NT               │
│ • Watershed HV-guided                 │
│ • Instance masks + centroids          │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│ AGGREGATION & REPORT                  │
│ • Comptage nucléaire par type         │
│ • Densité par mm²                     │
│ • Heatmap overlay sur thumbnail       │
│ • JSON structuré + audit trail        │
└───────────────────────────────────────┘
```

### 2.4 Preprocessing & Filtrage Intelligent (Standards CLAM/HistoQC)

> **Référence industrielle :** [CLAM - Mahmood Lab (Harvard)](https://github.com/mahmoodlab/CLAM) + [HistoQC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6552675/)
>
> Ces outils sont le standard de facto pour le preprocessing WSI en pathologie computationnelle.

#### Pipeline de Filtrage Multi-Niveau

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PIPELINE FILTRAGE INTELLIGENT (STANDARDS INDUSTRIELS)          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EXTRACTION BRUTE                                                           │
│  └── ~10,000 tiles potentiels (lame 2GB @ 40×)                              │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  NIVEAU 1 : TISSUE SEGMENTATION (CLAM Standard)                     │   │
│  │  ───────────────────────────────────────────────                    │   │
│  │  Méthode : Binary thresholding canal Saturation HSV @ basse résol.  │   │
│  │                                                                      │   │
│  │  Paramètres CLAM :                                                   │   │
│  │  • seg_level: -1 (auto, typiquement niveau 5× ou 10×)               │   │
│  │  • sthresh: 8 (seuil saturation, plus haut = moins de foreground)   │   │
│  │  • mthresh: 7 (median filter pour lisser)                           │   │
│  │  • close: 4 (morphological closing)                                  │   │
│  │  • contour_fn: 'four_pt' (4 points autour du centre dans contour)   │   │
│  │                                                                      │   │
│  │  Élimine : ~50-60% (fond blanc, verre, zones hors tissu)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  NIVEAU 2 : QUALITY CONTROL (HistoQC Standard)                      │   │
│  │  ───────────────────────────────────────────────                    │   │
│  │  Détection et exclusion des artefacts :                              │   │
│  │                                                                      │   │
│  │  • Pen markers     : Détection couleur (bleu, vert, rouge, noir)    │   │
│  │  • Tissue folds    : Détection gradient + texture anormale          │   │
│  │  • Air bubbles     : Détection zones circulaires claires            │   │
│  │  • Blur/Focus      : Variance Laplacien < seuil                     │   │
│  │  • Coverslip edge  : Détection bords artefactuels                   │   │
│  │                                                                      │   │
│  │  Métriques calculées :                                               │   │
│  │  • Color histograms (détection batch effects)                       │   │
│  │  • Brightness/Contrast                                               │   │
│  │  • Edge density                                                      │   │
│  │                                                                      │   │
│  │  Élimine : ~10-20% supplémentaires (artefacts, zones défocalisées)  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  NIVEAU 3 : CONTENT FILTERING (HistoROI / Domain-specific)          │   │
│  │  ───────────────────────────────────────────────────────            │   │
│  │  Filtrage basé sur le contenu tissulaire :                          │   │
│  │                                                                      │   │
│  │  • Adipose tissue  : Exclusion zones graisseuses (blanc + texture)  │   │
│  │  • Necrosis        : Détection zones nécrotiques (si non pertinent) │   │
│  │  • Mucin           : Détection mucine (optionnel selon application) │   │
│  │  • Stroma only     : Exclusion stroma pur sans cellules épithéliales│   │
│  │                                                                      │   │
│  │  Méthodes :                                                          │   │
│  │  • Entropie Shannon < 4.0 → zone homogène, exclure                  │   │
│  │  • H-Channel density < 5% → pas de noyaux visibles, exclure         │   │
│  │                                                                      │   │
│  │  Élimine : ~10-15% supplémentaires (tissus non informatifs)         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TILES FILTRÉS → INFERENCE V13                                      │   │
│  │  ─────────────────────────────────                                  │   │
│  │  ~20-30% des tiles initiaux (2,000-3,000 sur 10,000)                │   │
│  │                                                                      │   │
│  │  Gain performance : 3-5× plus rapide                                 │   │
│  │  Gain qualité : Moins de bruit, meilleure agrégation                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Paramètres CLAM Recommandés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `seg_level` | -1 (auto) | Niveau de downsampling pour segmentation |
| `sthresh` | 8 | Seuil saturation HSV (8 = standard) |
| `mthresh` | 7 | Taille filtre médian |
| `close` | 4 | Kernel morphological closing |
| `use_otsu` | True | Utiliser Otsu au lieu de seuil fixe |
| `contour_fn` | 'four_pt' | Vérifier 4 points autour du centre |
| `area_thresh` | 16 | Aire minimale contour (pixels²) |

#### Artefacts HistoQC Détectés

| Artefact | Méthode de Détection | Action |
|----------|---------------------|--------|
| **Pen markers** | Couleur HSV (bleu H:100-130, vert H:35-85) | Exclure région |
| **Tissue folds** | Gradient magnitude élevé + texture anormale | Exclure région |
| **Air bubbles** | Contours circulaires + haute luminosité | Exclure région |
| **Blur** | Variance Laplacien < 100 | Exclure tile |
| **Coverslip crack** | Lignes droites + faible saturation | Exclure région |
| **Thick section** | Saturation très élevée uniformément | Warning QC |

#### Métriques de Filtrage (Output)

```json
{
  "filtering_stats": {
    "level_1_tissue_segmentation": {
      "tiles_input": 10234,
      "tiles_output": 4521,
      "filtered_ratio": 0.558,
      "method": "CLAM_HSV_saturation"
    },
    "level_2_quality_control": {
      "tiles_input": 4521,
      "tiles_output": 3890,
      "filtered_ratio": 0.140,
      "artifacts_detected": {
        "pen_marker": 23,
        "blur": 456,
        "fold": 89,
        "bubble": 63
      },
      "method": "HistoQC"
    },
    "level_3_content_filtering": {
      "tiles_input": 3890,
      "tiles_output": 2845,
      "filtered_ratio": 0.269,
      "content_excluded": {
        "adipose": 567,
        "low_entropy": 234,
        "no_nuclei": 244
      },
      "method": "H-Channel_entropy"
    },
    "total": {
      "tiles_initial": 10234,
      "tiles_final": 2845,
      "overall_filtered_ratio": 0.722,
      "speedup_factor": 3.6
    }
  }
}
```

#### Intégration avec Outils Existants

| Outil | Intégration | Usage |
|-------|-------------|-------|
| **CLAM** | Natif Python | Tissue segmentation + tiling |
| **HistoQC** | Via PySlyde ou direct | Artifact detection |
| **PySlyde** | Package Python 2025 | Wrapper unifié (supporte H-Optimus) |
| **TRIDENT** | Mahmood Lab 2025 | Feature extraction + MIL |

**Note :** PySlyde (Nov 2025) intègre nativement la détection de tissu compatible CLAM et le support des artefacts via HistoQC, tout en supportant H-Optimus-0 pour l'extraction de features.

---

## 3. Workflow Utilisateur

### 3.1 Vue IHM Principale

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CELLVIT-OPTIMUS — WSI PROCESSING                            [User: Admin] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────┐    ┌────────────────────────────────────────┐  │
│  │  📋 LAMES EN ATTENTE   │    │  🔬 DÉTAIL LAME                        │  │
│  │  ────────────────────  │    │  ──────────────                        │  │
│  │                        │    │                                        │  │
│  │  ○ slide_001.svs      │    │  ┌──────────────────────────────────┐  │  │
│  │    📁 pending          │    │  │                                  │  │  │
│  │    2.3 GB | Sein       │    │  │         THUMBNAIL                │  │  │
│  │                        │    │  │         (tissue mask overlay)    │  │  │
│  │  ● slide_002.ndpi  ←──┼────│  │                                  │  │  │
│  │    📁 pending          │    │  └──────────────────────────────────┘  │  │
│  │    1.8 GB | Colon      │    │                                        │  │
│  │                        │    │  📊 MÉTADONNÉES                        │  │
│  │  ⏳ slide_003.svs      │    │  ├─ Dimensions: 98,304 × 65,536       │  │
│  │    🔄 processing (34%) │    │  ├─ MPP: 0.25 (40×)                   │  │
│  │    ETA: 8 min          │    │  ├─ Scanner: Hamamatsu NDP            │  │
│  │                        │    │  ├─ Tiles estimés: ~12,400            │  │
│  │  ✓ slide_004.svs      │    │  └─ Temps estimé: ~15 min             │  │
│  │    ✅ completed        │    │                                        │  │
│  │    AJI: 0.71           │    │  ┌──────────────────────────────────┐  │  │
│  │                        │    │  │  🚀 LANCER LE TRAITEMENT         │  │  │
│  │  ✗ slide_005.svs      │    │  └──────────────────────────────────┘  │  │
│  │    ❌ failed (QC)      │    │                                        │  │
│  │                        │    │  ┌──────────────────────────────────┐  │  │
│  └────────────────────────┘    │  │  ⚙️ OPTIONS AVANCÉES              │  │  │
│                                │  │  □ Force reprocess                │  │  │
│  ┌────────────────────────┐    │  │  □ Export DICOM                   │  │  │
│  │  📤 IMPORTER LAMES     │    │  │  Famille: [Auto-detect ▼]        │  │  │
│  │  Drag & drop ou Browse │    │  └──────────────────────────────────┘  │  │
│  └────────────────────────┘    └────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Vue Résultats

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RÉSULTATS — slide_004.svs                                    [← Retour]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │                      HEATMAP OVERLAY                                 │   │
│  │                      (navigable, zoomable)                           │   │
│  │                                                                      │   │
│  │   ┌────────────────────────────────────────────────────────────┐    │   │
│  │   │                                                            │    │   │
│  │   │                    [WSI + Heatmap]                         │    │   │
│  │   │                                                            │    │   │
│  │   │         🔴 High density (>5000/mm²)                        │    │   │
│  │   │         🟡 Medium density (2000-5000/mm²)                  │    │   │
│  │   │         🟢 Low density (<2000/mm²)                         │    │   │
│  │   │                                                            │    │   │
│  │   └────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │   [Zoom: 5×] [Pan] [Reset] [Toggle Heatmap] [Download PNG]          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐   │
│  │  📊 STATISTIQUES     │  │  🔬 NOYAUX PAR TYPE  │  │  📋 ACTIONS     │   │
│  │  ─────────────────   │  │  ─────────────────   │  │  ───────────    │   │
│  │                      │  │                      │  │                 │   │
│  │  Total: 1,847,293    │  │  Neoplastic: 12.7%   │  │  [📥 JSON]      │   │
│  │  Tiles: 8,234        │  │  Inflammatory: 4.8%  │  │  [📥 CSV]       │   │
│  │  Durée: 12m 34s      │  │  Connective: 24.7%   │  │  [📥 Heatmap]   │   │
│  │  Densité: 4,523/mm²  │  │  Dead: 0.7%          │  │  [📥 DICOM]     │   │
│  │                      │  │  Epithelial: 57.1%   │  │                 │   │
│  └──────────────────────┘  └──────────────────────┘  └─────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 États d'une Lame

```
                    ┌─────────┐
                    │ PENDING │ (fichier détecté, non traité)
                    └────┬────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ QC_CHECK │   │ QUEUED   │   │ SKIPPED  │
    │ (analyse │   │ (en file │   │ (user    │
    │  qualité)│   │ d'attente│   │  ignore) │
    └────┬─────┘   └────┬─────┘   └──────────┘
         │              │
         │    ┌─────────┘
         ▼    ▼
    ┌────────────┐
    │ PROCESSING │ (traitement en cours)
    │  [0-100%]  │
    └─────┬──────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌───────┐   ┌────────┐
│ DONE  │   │ FAILED │
│  ✅   │   │   ❌   │
└───────┘   └────────┘
```

---

## 4. Spécifications Techniques

### 4.1 Performance Cibles

| Métrique | Cible | Justification |
|----------|-------|---------------|
| **Temps/lame 2GB** | < 15 min | Comparable aux solutions commerciales |
| **Temps/lame 500MB** | < 5 min | Cas biopsie standard |
| **Throughput** | 50 lames/jour | 1 GPU RTX 4070 SUPER |
| **Mémoire GPU** | < 10 GB | Marge pour batching |
| **Mémoire RAM** | < 32 GB | Streaming tiles, pas tout en mémoire |

### 4.2 Librairies WSI — Benchmark et Recommandation

**Sources:**
- [OpenSlide Python](https://openslide.org/api/python/)
- [tiffslide GitHub](https://github.com/Bayer-Group/tiffslide)
- [PyVips Performance](https://github.com/libvips/pyvips/issues/100)

| Librairie | Avantages | Inconvénients | Recommandation |
|-----------|-----------|---------------|----------------|
| **OpenSlide** | Standard, tous formats, DICOM 4.0 | Plus lent que tiffslide sur TIFF | ✅ **Principal** |
| **tiffslide** | Plus rapide sur TIFF standard | 10× plus lent sur JPEG2000 (.svs TCGA) | ⚠️ Fallback TIFF |
| **PyVips** | Très rapide, faible mémoire | API moins intuitive | ✅ **Pour thumbnails** |

**Stratégie recommandée :**
```python
# Pseudo-code
def load_wsi(path):
    if path.suffix in ['.svs', '.ndpi', '.mrxs']:
        return OpenSlide(path)  # Meilleur support formats propriétaires
    elif path.suffix in ['.tiff', '.tif']:
        return tiffslide.open(path)  # Plus rapide sur TIFF standard
    elif path.suffix == '.dcm':
        return OpenSlide(path)  # Support DICOM depuis v4.0
```

### 4.3 Paramètres Extraction Tiles

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **tile_size** | 224 × 224 | Standard H-Optimus-0 |
| **target_mpp** | 0.5 | Résolution optimale V13 |
| **overlap** | 0 | Pas de chevauchement (efficacité) |
| **tissue_threshold** | 0.5 | 50% minimum de tissu |
| **background_threshold** | 220 | Pixel gris > 220 = fond |
| **blur_threshold** | 100 | Variance Laplacien < 100 = flou |

### 4.4 Structure Stockage

```
data/
└── wsi/
    ├── inbox/                      # Upload utilisateur
    │   └── *.svs, *.ndpi, *.mrxs
    │
    ├── pending/                    # En attente de traitement
    │   ├── slide_001.svs
    │   └── slide_001.json          # Métadonnées extraites
    │
    ├── processing/                 # En cours (1 seule à la fois)
    │   └── slide_002/
    │       ├── metadata.json
    │       ├── thumbnail.png
    │       ├── tissue_mask.png
    │       ├── tiles/              # Tiles 224×224
    │       ├── features/           # Cache H-Optimus
    │       └── progress.json       # État du traitement
    │
    ├── completed/                  # Terminé avec succès
    │   └── slide_003/
    │       ├── metadata.json
    │       ├── thumbnail.png
    │       ├── tissue_mask.png
    │       ├── tiles/
    │       ├── features/
    │       ├── predictions/        # NP, HV, NT par tile
    │       ├── instances/          # Masks instances
    │       ├── heatmap.png         # Overlay densité
    │       ├── report.json         # Rapport final
    │       └── audit.log           # Traçabilité
    │
    └── failed/                     # Échec (QC, erreur)
        └── slide_004/
            ├── metadata.json
            ├── error.log
            └── thumbnail.png       # Pour debug visuel
```

### 4.5 Format Rapport JSON

```json
{
  "version": "1.0",
  "slide_id": "slide_003",
  "timestamp": "2026-01-26T14:30:00Z",
  "status": "completed",

  "source": {
    "filename": "slide_003.svs",
    "path": "completed/slide_003/",
    "format": "Aperio SVS",
    "checksum_sha256": "a1b2c3d4..."
  },

  "metadata": {
    "dimensions_px": [98304, 65536],
    "dimensions_mm": [24.576, 16.384],
    "mpp": 0.25,
    "magnification": "40x",
    "scanner": {
      "vendor": "Leica",
      "model": "Aperio AT2",
      "serial": "AT2-12345"
    },
    "staining": "H&E",
    "organ_detected": "Breast",
    "organ_confidence": 0.94
  },

  "processing": {
    "pipeline_version": "V13.1",
    "started_at": "2026-01-26T14:15:00Z",
    "completed_at": "2026-01-26T14:30:00Z",
    "duration_seconds": 900,
    "tiles": {
      "total_possible": 15234,
      "extracted": 8234,
      "filtered_background": 5890,
      "filtered_blur": 1110
    },
    "gpu": {
      "device": "NVIDIA RTX 4070 SUPER",
      "memory_peak_mb": 9234
    }
  },

  "quality_control": {
    "tissue_ratio": 0.54,
    "blur_ratio": 0.07,
    "staining_uniformity": 0.89,
    "focus_score": 0.92,
    "passed": true
  },

  "results": {
    "nuclei": {
      "total_count": 1847293,
      "by_type": {
        "Neoplastic": {"count": 234567, "ratio": 0.127},
        "Inflammatory": {"count": 89012, "ratio": 0.048},
        "Connective": {"count": 456789, "ratio": 0.247},
        "Dead": {"count": 12345, "ratio": 0.007},
        "Epithelial": {"count": 1054580, "ratio": 0.571}
      },
      "density_per_mm2": {
        "mean": 4523.7,
        "std": 1234.5,
        "min": 120.3,
        "max": 12456.8
      }
    },
    "regions_of_interest": [
      {
        "id": "roi_001",
        "center_px": [45000, 23000],
        "center_mm": [11.25, 5.75],
        "size_px": [2240, 2240],
        "density_per_mm2": 8923.4,
        "neoplastic_ratio": 0.45,
        "confidence": 0.92
      }
    ],
    "heatmap_path": "completed/slide_003/heatmap.png"
  },

  "audit": {
    "operator": "system",
    "model_checkpoints": {
      "h_optimus": "bioptimus/H-optimus-0",
      "hovernet": "hovernet_breast_v13_smart_crops_hybrid_fpn_best.pth"
    },
    "parameters": {
      "watershed": {
        "np_threshold": 0.40,
        "min_size": 40,
        "beta": 1.5,
        "min_distance": 2
      }
    }
  }
}
```

---

## 5. Formats Supportés

### 5.1 Formats Prioritaires (Phase 1)

| Format | Extension | Vendeur | Support |
|--------|-----------|---------|---------|
| **Aperio SVS** | .svs | Leica | ✅ OpenSlide natif |
| **Hamamatsu NDPI** | .ndpi | Hamamatsu | ✅ OpenSlide natif |
| **MIRAX** | .mrxs | 3DHISTECH | ✅ OpenSlide natif |
| **Generic TIFF** | .tif/.tiff | Multiple | ✅ tiffslide |

### 5.2 Formats Futurs (Phase 2+)

| Format | Extension | Vendeur | Support |
|--------|-----------|---------|---------|
| **DICOM WSI** | .dcm | Standard | ⏳ OpenSlide 4.0 |
| **Philips iSyntax** | .isyntax | Philips | ⏳ SDK propriétaire |
| **Ventana BIF** | .bif | Roche | ⏳ SDK propriétaire |
| **Zeiss CZI** | .czi | Zeiss | ⏳ python-bioformats |

### 5.3 Détection Automatique

```python
FORMAT_SIGNATURES = {
    b'APER': 'aperio_svs',
    b'NDPI': 'hamamatsu_ndpi',
    b'MRXS': 'mirax',
    b'II*\x00': 'generic_tiff',
    b'MM\x00*': 'generic_tiff_be',
    b'DICM': 'dicom',
}

def detect_format(path):
    with open(path, 'rb') as f:
        header = f.read(4)
    return FORMAT_SIGNATURES.get(header, 'unknown')
```

---

## 6. Qualité et Validation

### 6.1 Quality Control Automatisé

Chaque lame passe par un QC avant traitement :

| Check | Critère | Action si échec |
|-------|---------|-----------------|
| **Format valide** | Header reconnu | → failed/ |
| **Lisibilité** | OpenSlide.read_region OK | → failed/ |
| **Tissue ratio** | > 10% de la lame | → failed/ (lame vide) |
| **Focus score** | Variance Laplacien > seuil | → warning (revue manuelle) |
| **Staining** | Détection H&E valide | → warning |

### 6.2 Métriques de Validation (CAP-aligned)

Pour validation diagnostique future :

| Métrique | Cible | Méthode |
|----------|-------|---------|
| **Concordance intra-observateur** | > 95% | Comparaison V13 vs pathologiste |
| **Reproductibilité** | CV < 5% | Même lame, 3 runs |
| **Sensibilité détection** | > 90% | Noyaux annotés GT |
| **Spécificité** | > 85% | Faux positifs / total |

### 6.3 Audit Trail

Chaque traitement génère un log complet :

```
[2026-01-26 14:15:00] INFO  | slide_003 | Processing started
[2026-01-26 14:15:01] INFO  | slide_003 | Format detected: aperio_svs
[2026-01-26 14:15:02] INFO  | slide_003 | Dimensions: 98304x65536, MPP: 0.25
[2026-01-26 14:15:05] INFO  | slide_003 | Tissue mask generated, ratio: 0.54
[2026-01-26 14:15:10] INFO  | slide_003 | QC passed: focus=0.92, staining=0.89
[2026-01-26 14:16:00] INFO  | slide_003 | Tiles extracted: 8234/15234 (54%)
[2026-01-26 14:20:00] INFO  | slide_003 | Features extracted: batch 100/103
[2026-01-26 14:28:00] INFO  | slide_003 | Inference complete: 1,847,293 nuclei
[2026-01-26 14:30:00] INFO  | slide_003 | Report generated, status: completed
```

---

## 7. Intégrations Futures

### 7.1 Phase 2 : Stockage Distribué

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ARCHITECTURE STOCKAGE PHASE 2                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LOCAL                      NAS/S3                        CDN              │
│   ─────                      ─────                         ───              │
│                                                                             │
│   inbox/  ───sync───→  s3://bucket/inbox/                                  │
│   pending/ ───sync───→  s3://bucket/pending/                               │
│   completed/ ←──lazy──  s3://bucket/completed/  ───cache───→  CloudFront   │
│                                                                             │
│   Tiles et features restent sur stockage rapide (SSD local ou NVMe NAS)    │
│   Reports et heatmaps peuvent être sur S3 (accès moins fréquent)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Phase 3 : Intégration LIS/PACS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INTÉGRATION HOSPITALIÈRE PHASE 3                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐     HL7 V2      ┌─────────────┐     DICOM      ┌─────────┐  │
│   │   LIS   │ ───────────────→│  CellViT    │ ──────────────→│  PACS   │  │
│   │ (Epic,  │ ←───────────────│  Optimus    │ ←──────────────│  /VNA   │  │
│   │ Cerner) │   HL7 Results   │             │   Query/Retrieve          │  │
│   └─────────┘                 └─────────────┘                └─────────┘  │
│        │                            │                             │        │
│        │                            │                             │        │
│        └────────────────────────────┼─────────────────────────────┘        │
│                                     │                                       │
│                                     ▼                                       │
│                              ┌─────────────┐                               │
│                              │     EMR     │                               │
│                              │  (Accès     │                               │
│                              │  cliniciens)│                               │
│                              └─────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 API REST (Production)

```
POST   /api/v1/slides              # Upload nouvelle lame
GET    /api/v1/slides              # Liste toutes les lames
GET    /api/v1/slides/{id}         # Détail d'une lame
POST   /api/v1/slides/{id}/process # Lancer traitement
GET    /api/v1/slides/{id}/status  # Statut traitement
GET    /api/v1/slides/{id}/report  # Télécharger rapport
GET    /api/v1/slides/{id}/heatmap # Télécharger heatmap
DELETE /api/v1/slides/{id}         # Supprimer lame

GET    /api/v1/health              # Healthcheck
GET    /api/v1/metrics             # Métriques Prometheus
```

---

## 8. Roadmap

### Phase 1 : Prototype (Actuel)

**Objectif :** Pipeline fonctionnel sur données open-source

| Tâche | Priorité | Effort | Dépendances |
|-------|----------|--------|-------------|
| WSI Loader (OpenSlide) | P1 | 2j | - |
| Intégration IHM Gradio | P1 | 3j | WSI Loader |
| Pipeline orchestrateur | P1 | 4j | WSI Loader |
| Tests CAMELYON16/TCGA | P1 | 2j | Pipeline |
| Documentation utilisateur | P2 | 1j | Pipeline |

**Datasets cibles :**
- CAMELYON16 (400 WSI, ganglions sein)
- TCGA (subset 50 WSI, multi-organes)

### Phase 2 : Consolidation

**Objectif :** Robustesse et performance

| Tâche | Priorité | Effort | Dépendances |
|-------|----------|--------|-------------|
| Support multi-format (.ndpi, .mrxs) | P1 | 2j | Phase 1 |
| QC automatisé complet | P1 | 3j | Phase 1 |
| Stockage S3/NAS | P2 | 3j | Phase 1 |
| Optimisation performance | P2 | 4j | Phase 1 |
| API REST FastAPI | P2 | 3j | Phase 1 |

### Phase 3 : Production

**Objectif :** Déploiement client

| Tâche | Priorité | Effort | Dépendances |
|-------|----------|--------|-------------|
| Support DICOM WSI | P2 | 3j | Phase 2 |
| Intégration LIS (HL7) | P3 | 2 sem | Phase 2, Client |
| Export QuPath/ASAP | P3 | 2j | Phase 2 |
| Validation CAP (60 cas) | P3 | 4 sem | Client, Pathologiste |
| Documentation FDA-ready | P3 | 2 sem | Validation |

---

## 9. Références

### Standards et Guidelines

- [DICOM WSI Standard (NEMA)](https://dicom.nema.org/dicom/dicomwsi/)
- [CAP WSI Validation Guidelines](https://www.cap.org/protocols-and-guidelines/cap-guidelines/current-cap-guidelines/validating-whole-slide-imaging-for-diagnostic-purposes-in-pathology)
- [IHE PaLM Technical Framework](https://www.ihe.net/Technical_Framework/PaLM/)
- [FDA Digital Pathology Guidance](https://www.fda.gov/medical-devices/digital-health-center-excellence)

### Vendors et Solutions

- [Sectra Digital Pathology](https://medical.sectra.com/product/sectra-digital-pathology-solution/)
- [Philips IntelliSite](https://www.usa.philips.com/healthcare/solutions/pathology)
- [Leica Biosystems Aperio](https://www.leicabiosystems.com/digital-pathology/manage/aperio-ehealth-solutions/)
- [PathAI AISight](https://www.pathai.com/)
- [Indica Labs HALO](https://indicalab.com/)

### Librairies Techniques

- [OpenSlide](https://openslide.org/)
- [tiffslide (Bayer)](https://github.com/Bayer-Group/tiffslide)
- [PyVips](https://github.com/libvips/pyvips)
- [CLAM (Mahmood Lab)](https://github.com/mahmoodlab/CLAM)

### Publications FDA 2025

- [PathAI AISight Dx FDA Clearance](https://www.pathai.com/news/pathai-receives-fda-clearance-for-aisight-dx-platform-for-primary-diagnosis)
- [Indica Labs HALO AP Dx FDA Clearance](https://indicalab.com/news/press-release/fda-cleared-digital-pathology/)
- [Ibex Prostate Detect FDA Clearance](https://www.targetedonc.com/view/fda-grants-510-k-clearance-to-ibex-prostate-detect-ai-for-prostate-cancer)

---

## Changelog

| Version | Date | Auteur | Modifications |
|---------|------|--------|---------------|
| 1.0 | 2026-01-26 | Claude | Création initiale |

---

**Document maintenu par:** Équipe CellViT-Optimus
**Dernière revue:** 2026-01-26
