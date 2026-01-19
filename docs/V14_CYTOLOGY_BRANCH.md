# V14 Cytology Branch — Dubai Edition

> **Version:** 14.0 (Spécifications initiales)
> **Date:** 2026-01-18
> **Statut:** 🚧 En spécification
> **Objectif:** Fusionner pipeline Histologie V13 avec nouveau pipeline Cytologie

---

## 📋 Vue d'ensemble

Le système V14 introduit une **architecture en "Y"** permettant de traiter automatiquement:
- **Histologie:** Coupes tissulaires H&E (pipeline V13 existant)
- **Cytologie:** Cellules isolées (frottis Pap, ponctions) — **NOUVEAU**

**Cas d'usage Dubai:** Déploiement multi-scanners nécessitant normalisation et calibration robustes.

---

## ✅ Specs Techniques Validées (Expert — 2026-01-19)

> **Source:** Validation Expert Architecture V14
> **Statut:** ✅ Conforme et Prêt pour Implémentation

### Pipeline de Traitement (Data Flow)

```
┌─────────────────────────────────────────────────────────────────────┐
│  PIPELINE V14 CYTOLOGIE (Séquentiel → Parallèle → Fusionné)         │
└─────────────────────────────────────────────────────────────────────┘

ÉTAPE 1 — Détection & Segmentation des Noyaux (CellPose)
    • Type: Séquentiel
    • Modèle: CellPose v2 pré-entraîné (nuclei)
    • Output: N bounding boxes + N masques
    • ⚠️ Aucun entraînement de CellPose dans V14

ÉTAPE 2 — Génération des Patchs 224×224 (Preprocessing)
    • Type: Séquentiel
    • Actions: Crop + Padding blanc + Normalisation Macenko ✅
    • Output: N patches 224×224 + masques alignés

ÉTAPE 3 — Analyse en deux branches (PARALLÈLE)
    ┌─────────────────────────────────┬──────────────────────────────┐
    │ Branche A: H-Optimus            │ Branche B: Morphométrie      │
    │ • Statut: Figé (non entraîné)   │ • Source: Masques CellPose   │
    │ • Input: Patch 224×224          │ • Calcul: 20 features        │
    │ • Output: Embedding 1536D       │ • Output: Vecteur 20D        │
    └─────────────────────────────────┴──────────────────────────────┘

ÉTAPE 4 — Fusion Multimodale (Concatenation)
    • combined[i] = concat(embedding[i], morpho_features[i])
    • Vecteur final: 1556D (1536 + 20)

ÉTAPE 5 — Classification Finale (MLP)
    • Architecture: 1556 → 512 → 256 → 7 classes
    • BatchNorm sur morpho features (CRITIQUE)
    • Focal Loss (déséquilibre classes)
    • Output: Probabilités par classe + Confiance
```

### Modules Entraînés vs Non-Entraînés

| Module | Statut | Justification |
|--------|--------|---------------|
| **CellPose** | ✅ Pré-entraîné, figé | Détection généraliste robuste |
| **H-Optimus-0** | ✅ Pré-entraîné, figé | Feature extractor 1.1B params |
| **MLP Morphométrique** | ⚠️ Facultatif | Normalisation features (si requis) |
| **MLP Classification** | 🔵 **ENTRAÎNÉ** | Fusion multimodale → Diagnostic |

### 20 Features Morphométriques (SINGLE SOURCE OF TRUTH)

**⚠️ CRITIQUE:** Features calculées sur masques CellPose, JAMAIS lues depuis CSV/Excel externe.

| # | Feature | Source | Importance Clinique |
|---|---------|--------|---------------------|
| 1 | area_nucleus | regionprops | Criterion 1 (Size of Nuclei) |
| 2-10 | Géométrie | regionprops | Forme, circularité, solidité |
| 11-13 | Intensité + H-channel | Ruifrok | Criterion 3 (Chromatin Density) |
| 14-16 | Haralick texture | GLCM | Granularité chromatine |
| 17-18 | **N/C ratio** | regionprops | **Paris System (> 0.7 = High Grade)** |
| 19-20 | Feret, roundness | regionprops | Dimensions max |

**Implémentation:** `src/cytology/morphometry.py` (complète avec 20 features)

### Architecture MLP avec BatchNorm (Fusion Multimodale)

```python
# ═════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE VALIDÉE EXPERT
# ═════════════════════════════════════════════════════════════════════════════

input_embedding = Input(shape=(1536,))    # H-Optimus
input_morpho = Input(shape=(20,))         # Morphométrie

# 1. NORMALISATION MORPHO (VITAL pour équilibrage gradients)
norm_morpho = BatchNormalization()(input_morpho)

# 2. FUSION
merged = Concatenate()([input_embedding, norm_morpho])

# 3. CLASSIFICATION HEAD
x = Dense(512, activation='relu')(merged)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(num_classes, activation='softmax')(x)
```

**Pourquoi BatchNorm est CRITIQUE:**
- Embedding: 1536 dims, valeurs normalisées ~[-1, +1]
- Morpho: 20 dims, valeurs brutes (area=500, nc_ratio=0.7)
- Sans BatchNorm → Gradient écrase features morpho (1536 >> 20)
- Avec BatchNorm → Fusion réellement multimodale

**Implémentation:** `src/models/cytology_classifier.py` (complète avec Focal Loss)

### Macenko Normalization: Router-Dependent ✅

| Branche | Macenko | Justification |
|---------|---------|---------------|
| **Cytologie** | ✅ **ON** | Scanners multiples Dubai + Pas de FPN Chimique |
| **Histologie** | ❌ **OFF** | Régression -4.3% AJI (conflit Ruifrok/FPN) |

**Documentation complète:** [V14_MACENKO_STRATEGY.md](./V14_MACENKO_STRATEGY.md)

### Avantages Architecture V14 (Validation Expert)

1. ✅ **Pas d'annotation manuelle** — CellPose détection automatique
2. ✅ **Intelligence visuelle** — H-Optimus 1.1B params
3. ✅ **Explicabilité** — 20 features quantifiables (médecins comprennent)
4. ✅ **Localisation robuste** — CellPose élimine erreurs
5. ✅ **Performance avec peu de données** — Transfer learning (SIPaKMeD 4,049 + augmentation)
6. ✅ **Maintenabilité** — Modules découplés (CellPose v3 → swap sans toucher reste)
7. ✅ **Production** — Pipeline rapide (~0.5s/cellule)

### Résumé One-Liner

> **V14 = CellPose localise + Optimus comprend + Morphométrie quantifie + MLP décide**

---

## 🚨 ALERTES CRITIQUES — Conflits avec V13 Production

### ⚠️ Alerte 1: Macenko Normalization = Régression -4.3% AJI

**Découverte V13 (2025-12-30):**

| Configuration | AJI Respiratory | Δ |
|---------------|-----------------|---|
| **SANS Macenko (Raw)** | **0.6872** ✅ | Baseline |
| AVEC Macenko | 0.6576 | **-4.3%** ❌ |

**Cause:** Le **"Shift de Projection"**
- Ruifrok = Vecteurs physiques FIXES (Beer-Lambert)
- Macenko = Rotation ADAPTATIVE dans l'espace optique
- **Conflit:** Macenko déplace Éosine vers vecteur Hématoxyline → "fantômes" cytoplasme dans canal H → bruit dans HV-MSE

**Référence:** `CLAUDE.md` section "Découverte Stratégique: Ruifrok vs Macenko"

**Impact sur V14:**
- ❌ **Spec initiale:** "Preprocessing : Normalisation Macenko (Standardisation couleur)" pour TOUTES les images
- ✅ **Recommandation:** Normalisation **Router-Dependent** (voir Architecture Proposée)

### ⚠️ Alerte 2: Non-Régression V13 Obligatoire

**Requirement critique:**
Le pipeline Histologie V14 DOIT maintenir les performances V13:
- Respiratory: AJI ≥ 0.6872
- Urologic: AJI ≥ 0.6743
- Glandular: AJI ≥ 0.6566

**Tests obligatoires:**
```python
# tests/test_v14_non_regression.py
def test_v13_histo_unchanged():
    """V13 AJI doit rester inchangé après intégration Router"""
    model_v14 = V14HybridSystem(histo_branch=load_v13_checkpoint())

    aji_v14_respiratory = evaluate_aji(model_v14, respiratory_val, force_branch="histo")
    assert aji_v14_respiratory >= 0.6872, "Régression V13 détectée!"
```

---

## 🏗️ Architecture Globale

> **⚠️ CLARIFICATION CRITIQUE (2026-01-19):**
> Le pipeline Cytologie est **Séquentiel PUIS Parallèle**, pas "parallèle pur".
> Voir [V14_PIPELINE_EXECUTION_ORDER.md](./V14_PIPELINE_EXECUTION_ORDER.md) pour détails complets.

### Architecture en "Y" (High-Level)

**Note:** Ce diagramme montre l'architecture globale (Router Histo/Cyto).
Pour l'ordre d'exécution AU SEIN de la branche Cytologie, voir section "Pipeline Cytologie Détaillé" ci-dessous.

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (RGB)                             │
│              WSI Tile / Cytology Smear                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │  PREPROCESSING (Router-Based) │
         │  • Cyto → Macenko ON          │
         │  • Histo → RAW (V13)          │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │   H-OPTIMUS-0 BACKBONE        │
         │   (1.1B params, FROZEN)       │
         │   Output: CLS + 256 Patches   │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │      ROUTER HEAD (MLP)        │
         │   Input: CLS Token            │
         │   Output: P(Cytology)         │
         └───────────────┬───────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
      P > 0.85                   P < 0.15
            │                         │
            ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│  CYTOLOGY BRANCH    │   │  HISTOLOGY BRANCH   │
│  (NOUVEAU)          │   │  (V13 EXISTANT)     │
├─────────────────────┤   ├─────────────────────┤
│ • CellPose Segm.    │   │ • FPN Chimique      │
│ • Morphométrie      │   │ • HV-Guided         │
│ • Virtual Marker    │   │   Watershed         │
│ • N/C Ratio         │   │ • AJI ≥ 0.68        │
│ • Cyto Head         │   │                     │
└─────────────────────┘   └─────────────────────┘
            │                         │
            └────────────┬────────────┘
                         ▼
         ┌───────────────────────────────┐
         │   CALIBRATION CLINIQUE        │
         │   • h_channel_gain            │
         │   • pixel_size_microns        │
         │   • scanner_profile           │
         └───────────────┬───────────────┘
                         ▼
         ┌───────────────────────────────┐
         │      JSON OUTPUT              │
         │   • pipeline_branch           │
         │   • predictions               │
         │   • clinical_metrics          │
         └───────────────────────────────┘
```

### Zone d'Incertitude Router

**Problème:** Spec initiale = Décision binaire (P > 0.5)

**Recommandation:** Ajouter zone grise pour images ambiguës

```python
ROUTER_THRESHOLDS = {
    "cyto_confident": 0.85,    # P > 0.85 → Cytologie
    "histo_confident": 0.15,   # P < 0.15 → Histologie
    # 0.15 ≤ P ≤ 0.85 → UNCERTAIN (flag review)
}
```

**Cas d'usage "Uncertain":**
- Images mal préparées (artéfacts)
- Biopsies liquides (mixte tissu + cellules)
- Coupes fines ressemblant à frottis

---

## 🔄 Pipeline Cytologie Détaillé (Ordre d'Exécution)

> **CLARIFICATION CRITIQUE (2026-01-19):**
> Le pipeline Cytologie n'est PAS "parallèle pur". C'est **Séquentiel PUIS Parallèle**.

### Pourquoi Séquentiel d'Abord?

**Problème:** H-Optimus ne peut PAS analyser directement une image 1024×1024 avec N cellules.
- H-Optimus attend **224×224 centrées sur UNE cellule**
- Il ne fait pas de détection d'objets (pas de bounding boxes)
- Il est un **encodeur** (image → vecteur), pas un **détecteur**

**Solution:** CellPose DOIT venir en premier pour localiser les cellules.

### Les 5 Étapes (Ordre Strict)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE SÉQUENTIELLE (Obligatoire)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT: Image WSI 1024×1024 (après Router → branch="cytology")

ÉTAPE 1: CellPose Master (nuclei)
    ↓
    Détecte N cellules → N bounding boxes + N masques noyaux
    Ex: 50 cellules détectées

ÉTAPE 1.5: CellPose Slave (cyto3) — CONDITIONNEL
    ↓
    Si organe requiert N/C ratio (Thyroid, Bladder) → Segmente cytoplasme
    Sinon (Cervix) → Skip

ÉTAPE 2: Crop + Padding
    ↓
    Pour chaque bbox → Crop 224×224 + Padding blanc (PadIfNeeded)
    Résultat: 50 patches 224×224 prêts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE PARALLÈLE (Optimisée — Batch Processing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pour CHAQUE patch (50 itérations parallélisées):

ÉTAPE 3A: H-Optimus               ÉTAPE 3B: Morphométrie
(GPU Batch)                        (CPU Multi-thread)
    ↓                                  ↓
Embedding 1536D                    14 Features + Canal H
(Texture, couleur)                 (Forme, taille, N/C)
    │                                  │
    └──────────────┬───────────────────┘
                   ↓

ÉTAPE 4: Fusion (Concatenation)
    ↓
    Vecteur 1550D = [1536 + 14]

ÉTAPE 5: MLP Classification Head
    ↓
    Softmax → Classe (ex: "Carcinoma in situ") + Confiance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT: Rapport pour les 50 cellules
        "5 cellules suspectes sur 50 analysées"
```

### Rôles des Composants

| Composant | Rôle | Input | Output | Timing |
|-----------|------|-------|--------|--------|
| **CellPose Master** | Localisation | Image 1024×1024 | N bounding boxes | Séquentiel |
| **CellPose Slave** | Contexte (N/C) | Image 1024×1024 | N masques cyto | Séquentiel |
| **Crop + Pad** | Préparation | Bboxes | N patches 224×224 | Séquentiel |
| **H-Optimus** | Encodage | Patch 224×224 | Embedding 1536D | **Parallèle** |
| **Morphométrie** | Mesure | Masques | 14 features | **Parallèle** |
| **MLP Head** | Décision | Vecteur 1550D | Classe + Confiance | Séquentiel |

**Principe Clé:** CellPose et H-Optimus ne sont PAS parallèles. H-Optimus et Morphométrie le sont.

### Optimisation Batch (Implémentation)

```python
# Une fois les patches générés (après CellPose):
patches_tensor = torch.stack(patches)  # (50, 3, 224, 224)

# ✅ PARALLÈLE: Batch inference GPU
with torch.no_grad():
    embeddings = h_optimus(patches_tensor)  # (50, 1536) — ~0.1s

# ✅ PARALLÈLE: Multi-thread CPU
with concurrent.futures.ThreadPoolExecutor() as executor:
    morpho_features = list(executor.map(compute_morphometry, masks))  # ~0.05s

# Total: ~0.15s pour 50 cellules (vs 50 × 0.02s = 1s si séquentiel)
```

**Documentation complète:** [V14_PIPELINE_EXECUTION_ORDER.md](./V14_PIPELINE_EXECUTION_ORDER.md)

---

## 📐 Module A: Pre-Processing & Normalisation

### Approche Router-Dependent (Recommandée)

**Principe:** Normalisation conditionnelle pour isoler le risque Macenko

```python
def preprocess_v14(image, pipeline_branch):
    """
    Preprocessing adaptatif selon branche détectée par Router

    Args:
        image: RGB Tensor [3, H, W]
        pipeline_branch: "cytology" | "histology" | "uncertain"

    Returns:
        image_processed: Tensor normalisé
        h_channel: Canal Hématoxyline (Ruifrok)
    """
    if pipeline_branch == "cytology":
        # Cytologie: Macenko OK (pas de FPN Chimique downstream)
        image_normalized = macenko_normalize(
            image,
            target_template=load_reference_template("pap_smear_ref.png")
        )
        h_channel = ruifrok_deconvolution(image_normalized)
        return image_normalized, h_channel

    else:  # histology ou uncertain
        # Histologie: RAW images (V13 prouvé)
        # Extraction H-channel sur RAW (préserve physique Beer-Lambert)
        h_channel = ruifrok_deconvolution(image)
        # Pas de normalisation Macenko
        return image, h_channel
```

### Implémentation Macenko (Cytologie uniquement)

**Librairie:** `torch-stain` ou custom OpenCV

```python
from torchstain import MacenkoNormalizer

class CytologyPreprocessor:
    def __init__(self, target_template_path):
        self.normalizer = MacenkoNormalizer()

        # Charger image de référence (Pap smear parfait)
        target = cv2.imread(target_template_path)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        self.normalizer.fit(target)

    def normalize(self, image):
        """
        Transforme image entrante pour matcher template

        Args:
            image: np.array [H, W, 3] RGB
        Returns:
            normalized: np.array [H, W, 3] RGB
        """
        normalized, _, _ = self.normalizer.normalize(image)
        return normalized
```

**Image de Référence (Target Template):**
- Format: PNG RGB 224×224
- Contenu: Frottis Pap bien coloré (noyaux bleus nets, cytoplasme rose)
- Stockage: `data/references/pap_smear_ref.png`
- QC: Vérifier histogrammes RGB centrés

---

## 📡 Module B: Router (The Switch)

### Architecture Actuelle (Spec)

```python
class RouterHead(nn.Module):
    def __init__(self, input_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, cls_token):
        """
        Args:
            cls_token: [B, 1536] CLS token de H-Optimus-0
        Returns:
            prob_cyto: [B, 1] Probabilité Cytologie
        """
        return self.mlp(cls_token)
```

### Architecture Enrichie (Recommandée)

**Ajout:** Variance patches pour capturer différence texture Histo/Cyto

```python
class RouterHeadEnhanced(nn.Module):
    def __init__(self, input_dim=1536):
        super().__init__()
        # CLS: 1536, Patch variance: 1536 → Total: 3072
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features: [B, 261, 1536] (CLS + 4 registers + 256 patches)
        Returns:
            prob_cyto: [B, 1]
        """
        cls_token = features[:, 0, :]  # [B, 1536]
        patch_tokens = features[:, 5:261, :]  # [B, 256, 1536]

        # Variance patches = proxy de texture
        # Cyto: fond blanc + cellules isolées → haute variance
        # Histo: tissu dense → variance modérée
        patch_variance = patch_tokens.var(dim=1)  # [B, 1536]

        router_input = torch.cat([cls_token, patch_variance], dim=1)  # [B, 3072]
        return self.mlp(router_input)
```

### Logique de Routing avec Incertitude

```python
def route_image(features, thresholds=None):
    """
    Route image vers pipeline approprié avec zone d'incertitude

    Args:
        features: [B, 261, 1536] Features H-Optimus-0
        thresholds: dict {"cyto_confident": 0.85, "histo_confident": 0.15}

    Returns:
        branch: "cytology" | "histology" | "uncertain"
        confidence: float (0-1)
    """
    if thresholds is None:
        thresholds = {"cyto_confident": 0.85, "histo_confident": 0.15}

    prob_cyto = router_head(features)

    if prob_cyto > thresholds["cyto_confident"]:
        return "cytology", prob_cyto.item()
    elif prob_cyto < thresholds["histo_confident"]:
        return "histology", 1 - prob_cyto.item()
    else:
        # Zone grise: Nécessite review ou exécution double pipeline
        return "uncertain", max(prob_cyto.item(), 1 - prob_cyto.item())
```

### Dataset pour Training Router

**Sources:**

| Type | Dataset | Samples | Usage |
|------|---------|---------|-------|
| **Histologie** | PanNuke (toutes familles) | ~7,904 | Training Router (label=0) |
| **Cytologie** | Herlev (Col utérin) | 917 | Training Router (label=1) |
| **Cytologie** | TB-PANDA (Thyroïde) | ~10,000 | Training Router (label=1) |
| **Cytologie** | Urine (à sourcer) | TBD | Training Router (label=1) |

**Target:** ≥ 5,000 images par classe (balanced)

**Training:**
```python
# Pseudo-code
router_dataset = {
    "train": 4000 Histo + 4000 Cyto,
    "val": 1000 Histo + 1000 Cyto
}

# Binary Cross-Entropy Loss
criterion = nn.BCELoss()

# Validation: Accuracy > 98% requis
```

---

## 🧬 Module C: Branche Cytologie (NOUVEAU)

### C.1. Segmentation — Architecture Maître/Esclave ⭐

> **🎯 CHANGEMENT ARCHITECTURAL MAJEUR (2026-01-18)**
>
> Suite à analyse approfondie, l'approche CellPose `cyto2` unique est remplacée par une **orchestration intelligente de 2 modèles spécialisés**.
>
> **Gains:** 2× plus rapide, 46% économie GPU, modularité commerciale
>
> **Documentation détaillée:** [V14_MASTER_SLAVE_ARCHITECTURE.md](./V14_MASTER_SLAVE_ARCHITECTURE.md)

#### Philosophie: "Nuclei First" (Maître/Esclave)

**Principe:** Orchestration séquentielle de 2 modèles CellPose spécialisés

```
┌─────────────────────────────────────────────────────────────┐
│  MAÎTRE: CellPose "nuclei"                                  │
│  • Spécialisation: Noyaux UNIQUEMENT                        │
│  • Activation: 100% des images (screening universel)        │
│  • Temps: ~300-500ms                                        │
│  • Output: Masques noyaux + Features nucléaires             │
└─────────────────────────────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │  TRIGGER (Intelligent)│
              │  Config organe        │
              └───────────┬───────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
    IF Urine/Thyroïde                IF Gynéco/Autre
    (N/C requis)                     (N/C non requis)
        │                                   │
        ▼                                   ▼
┌────────────────────┐            ┌──────────────────┐
│  ESCLAVE ACTIVÉ    │            │  ESCLAVE SKIP    │
│  CellPose "cyto3"  │            │  (70% économie)  │
│  • Noyau + Cyto    │            │  • Rapport       │
│  • Lourd (~1.5s)   │            │    nuclei seul   │
│  • 30% images      │            │  • ~500ms        │
└────────────────────┘            └──────────────────┘
```

**Avantages clés:**

| Aspect | Cyto2 Seul (Initial) | Maître/Esclave | Gain |
|--------|---------------------|----------------|------|
| **Performance** | 2s/image | 0.5-1.8s (adaptatif) | **2× plus rapide** |
| **GPU Load** | 100% constant | 30-100% adaptatif | **46% économie** |
| **Spécialisation** | Générique | nuclei (ADN) + cyto3 (N/C) | **Précision** |
| **Business** | Forfait unique | 4 packages (€5k-€12k) | **Monetization** |

#### Matrice de Décision par Organe

| Organe | Nuclei (Maître) | Cyto3 (Esclave) | N/C Ratio | Justification |
|--------|-----------------|-----------------|-----------|---------------|
| **Gynéco (Col)** | ✅ Actif | ❌ Inactif (Option) | Optionnel | Dépistage masse sur atypie nucléaire. Cytoplasme plicaturé. |
| **Urine (Vessie)** | ✅ Actif | ✅ **Auto-Actif** | **Requis** | Paris System EXIGE N/C > 0.7 pour Haut Grade. INDISPENSABLE. |
| **Thyroïde (FNA)** | ✅ Actif | ✅ **Auto-Actif** | **Requis** | Bethesda: N/C critique pour carcinomes Papillaire/Folliculaire. |
| **Liquides (Plèvre)** | ✅ Actif | ❌ Inactif | N/A | Amas 3D/cellules géantes. Segmentation cyto difficile, peu utile. |
| **Ganglion (Lymphome)** | ✅ Actif | ❌ Inactif | N/A | Lymphocytes quasi que noyau (N/C ~ 0.9). Cyto3 échouerait. |

#### Installation & Usage

```bash
pip install cellpose
```

**Code Orchestrateur:**
```python
from cellpose import models

class CytologyMasterSlaveOrchestrator:
    """
    Orchestrateur intelligent V14

    Architecture:
    - Master (nuclei): 100% activation
    - Slave (cyto3): Activation conditionnelle (organe-dependent)
    """

    def __init__(self, organ_config_path="config/cytology_organ_config.json"):
        # Charger config organes
        with open(organ_config_path, 'r') as f:
            self.organ_config = json.load(f)['cytology_organ_profiles']

        # Modèle MAÎTRE (léger, toujours actif)
        self.nuclei_model = models.Cellpose(
            gpu=True,
            model_type='nuclei'  # Spécialisé noyaux
        )

        # Modèle ESCLAVE (lourd, conditionnel)
        self.cyto3_model = models.Cellpose(
            gpu=True,
            model_type='cyto3'  # Spécialisé noyau + cytoplasme
        )

    def process_image(self, image_rgb, organ_type, force_cyto3=False):
        """
        Pipeline séquentiel intelligent (4 étapes)

        Args:
            image_rgb: np.array [H, W, 3]
            organ_type: str ("urology_bladder", "gynecology_cervix", etc.)
            force_cyto3: bool Override config (mode Expert)

        Returns:
            dict {
                "nuclei_masks": np.array,
                "nuclei_features": list of dict,
                "cyto3_masks": np.array or None,
                "nc_ratios": list of dict or None,
                "clinical_alerts": list,
                "processing_time_ms": dict,
                "pipeline_branch": str
            }
        """
        results = {}

        # STEP 1: NUCLEI SCREENING (MAÎTRE - 100% images)
        t_start = time.time()
        nuclei_masks, _, _ = self.nuclei_model.eval(
            image_rgb,
            diameter=30,
            channels=[0, 0],
            flow_threshold=0.4
        )
        results["processing_time_ms"]["nuclei"] = (time.time() - t_start) * 1000
        results["nuclei_masks"] = nuclei_masks

        # Extraire features nucléaires
        h_channel = self._extract_h_channel(image_rgb)
        nuclei_features = self._extract_nuclear_features(nuclei_masks, h_channel)
        results["nuclei_features"] = nuclei_features

        # STEP 2: TRIGGER DECISION
        organ_cfg = self.organ_config.get(organ_type, {})
        cyto3_cfg = organ_cfg.get("cyto3_model", {})

        should_run_cyto3 = (
            force_cyto3  # Override utilisateur
            or cyto3_cfg.get("enabled", False)
            or cyto3_cfg.get("trigger") == "auto"
        )

        if not should_run_cyto3:
            # SKIP STEP 3: Rapport nuclei seul
            results["cyto3_masks"] = None
            results["nc_ratios"] = None
            results["pipeline_branch"] = "master_only"
            return results

        # STEP 3: CYTO3 SEGMENTATION (ESCLAVE - Conditionnel)
        t_start = time.time()
        cyto3_masks, _, _ = self.cyto3_model.eval(
            image_rgb,
            diameter=60,  # Cellule complète
            channels=[0, 0],
            flow_threshold=0.4
        )
        results["processing_time_ms"]["cyto3"] = (time.time() - t_start) * 1000
        results["cyto3_masks"] = cyto3_masks
        results["pipeline_branch"] = "master_slave_full"

        # STEP 4: FUSION GÉOMÉTRIQUE (Matching Nuclei → Cytoplasme)
        nc_ratios = self._match_nuclei_to_cytoplasm(nuclei_masks, cyto3_masks)
        results["nc_ratios"] = nc_ratios

        # Clinical alerts
        results["clinical_alerts"] = self._check_clinical_alerts(nc_ratios, organ_cfg)

        return results

    def _match_nuclei_to_cytoplasm(self, nuclei_masks, cyto3_masks):
        """
        Matching géométrique: Pour chaque noyau, trouver cytoplasme contenant

        Gestion erreurs:
        - Cas A: Match parfait (1N→1C) → N/C calculé
        - Cas B: Noyau orphelin (pas de C) → N/C = None
        - Cas C: Cytoplasme vide (pas de N) → Ignoré
        """
        nuclei_props = regionprops(nuclei_masks)
        cyto3_props = regionprops(cyto3_masks)

        nc_ratios = []

        for nucleus_prop in nuclei_props:
            nucleus_centroid = nucleus_prop.centroid
            nucleus_area = nucleus_prop.area

            # Chercher cytoplasme contenant ce noyau
            matched_cyto = None
            for cyto_prop in cyto3_props:
                if self._point_in_mask(nucleus_centroid, cyto3_masks, cyto_prop.label):
                    matched_cyto = cyto_prop
                    break

            if matched_cyto is not None:
                # Cas A: Match parfait
                cytoplasm_area = matched_cyto.area
                nc_ratio = nucleus_area / cytoplasm_area

                nc_ratios.append({
                    "nucleus_id": nucleus_prop.label,
                    "cytoplasm_id": matched_cyto.label,
                    "nc_ratio": nc_ratio,
                    "status": "matched"
                })
            else:
                # Cas B: Noyau orphelin → Ne PAS bloquer rapport!
                nc_ratios.append({
                    "nucleus_id": nucleus_prop.label,
                    "nc_ratio": None,
                    "status": "orphan",
                    "warning": "Cytoplasm not detected - Use nuclear metrics only"
                })

        return nc_ratios
```

**Voir documentation complète:** [V14_MASTER_SLAVE_ARCHITECTURE.md](./V14_MASTER_SLAVE_ARCHITECTURE.md)

- Pipeline séquentiel (4 étapes détaillées)
- Implémentation complète `CytologyMasterSlaveOrchestrator`
- Benchmarks performance (2× gain vs cyto2 seul)
- Business model (4 packages €5k-€12k)

### C.2. Virtual Marker — Canal H (Ruifrok)

**Algorithme:** Ruifrok & Johnston deconvolution

**Implémentation:**
```python
import numpy as np

def ruifrok_deconvolution(image_rgb):
    """
    Sépare image H&E ou Pap en composantes optiques

    Args:
        image_rgb: np.array [H, W, 3] RGB (0-255)

    Returns:
        h_channel: np.array [H, W] Canal Hématoxyline (densité optique)
        e_channel: np.array [H, W] Canal Éosine
    """
    # Vecteurs Ruifrok (constantes physiques Beer-Lambert)
    # H&E staining
    stain_matrix = np.array([
        [0.650, 0.704, 0.286],  # Hématoxyline (bleu)
        [0.072, 0.990, 0.105],  # Éosine (rose)
        [0.268, 0.570, 0.776]   # Résiduel
    ])

    # Conversion RGB → Optical Density
    image_rgb = image_rgb.astype(np.float32) + 1  # Éviter log(0)
    od = -np.log10(image_rgb / 255.0)

    # Résolution système linéaire
    od_reshaped = od.reshape(-1, 3).T  # [3, H*W]
    concentrations = np.linalg.lstsq(stain_matrix.T, od_reshaped, rcond=None)[0]

    h_channel = concentrations[0].reshape(image_rgb.shape[:2])
    e_channel = concentrations[1].reshape(image_rgb.shape[:2])

    # Normalisation 0-255
    h_channel = np.clip(h_channel * 255 / h_channel.max(), 0, 255).astype(np.uint8)

    return h_channel, e_channel
```

**Note Pap Staining:**
Pour frottis Papanicolaou (non H&E), adapter vecteurs:
```python
# Pap staining (OG-6, EA-50, Hematoxylin)
stain_matrix_pap = np.array([
    [0.610, 0.740, 0.280],  # Hématoxyline (noyaux bleus)
    [0.450, 0.820, 0.350],  # OG-6 (cytoplasme kératinisé orange)
    [0.670, 0.600, 0.440]   # EA-50 (cytoplasme vert/rose)
])
```

### C.3. Morphométrie Avancée

**Features de Base (Spec):**
```python
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops

def extract_basic_features(mask, h_channel):
    """
    Features géométriques + densité optique

    Args:
        mask: Binary mask [H, W]
        h_channel: Canal H [H, W]

    Returns:
        dict de features
    """
    props = regionprops(mask.astype(int), intensity_image=h_channel)[0]

    features = {
        # Géométrie
        "area": props.area,  # pixels²
        "perimeter": props.perimeter,
        "circularity": 4 * np.pi * props.area / (props.perimeter ** 2),
        "eccentricity": props.eccentricity,  # 0=rond, 1=ligne
        "convexity": props.area / props.convex_area,

        # Densité (Virtual Marker)
        "mean_od": props.mean_intensity,  # Mean Optical Density
        "integrated_od": props.mean_intensity * props.area,  # IOD (proxy ploïdie)
        "std_od": np.std(h_channel[mask > 0]),
    }

    return features
```

**Features Avancées (Recommandées):**

```python
def extract_advanced_cytology_features(nucleus_mask, cytoplasm_mask, h_channel, rgb_image):
    """
    Features spécifiques cytopathologie

    Args:
        nucleus_mask: Masque noyau [H, W]
        cytoplasm_mask: Masque cytoplasme [H, W]
        h_channel: Canal H [H, W]
        rgb_image: Image RGB [H, W, 3]

    Returns:
        dict de features cytopathologiques
    """
    nucleus_props = regionprops(nucleus_mask.astype(int), intensity_image=h_channel)[0]

    # --- CRITIQUE EN CYTOLOGIE ---
    # 1. Nuclear-to-Cytoplasmic Ratio (N/C)
    nucleus_area = nucleus_props.area
    cytoplasm_area = np.sum(cytoplasm_mask) - nucleus_area
    nc_ratio = nucleus_area / cytoplasm_area if cytoplasm_area > 0 else np.nan

    # 2. Chromatin Pattern (Coarseness)
    h_nucleus = h_channel[nucleus_mask > 0]
    chromatin_coarseness = np.std(h_nucleus) / np.mean(h_nucleus) if len(h_nucleus) > 0 else 0

    # 3. Nucleoli Detection
    # Nucleoli = zones TRÈS denses dans H-channel (seuil > mean + 2*std)
    nucleoli_threshold = np.mean(h_nucleus) + 2 * np.std(h_nucleus)
    nucleoli_pixels = h_nucleus > nucleoli_threshold
    nucleoli_count = measure.label(nucleoli_pixels.reshape(nucleus_mask.shape)).max()

    # 4. Nuclear Contour Irregularity (Fractal Dimension)
    contour = find_contours(nucleus_mask, 0.5)[0]
    fractal_dim = compute_fractal_dimension(contour)  # Voir implémentation ci-dessous

    # 5. Texture Haralick (sur H-channel)
    h_nucleus_2d = h_channel.copy()
    h_nucleus_2d[nucleus_mask == 0] = 0
    h_quantized = (h_nucleus_2d / 16).astype(np.uint8)  # 16 niveaux de gris
    glcm = graycomatrix(
        h_quantized,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=16,
        symmetric=True,
        normed=True
    )

    features_advanced = {
        # Cytologie clinique
        "nc_ratio": nc_ratio,
        "chromatin_coarseness": chromatin_coarseness,
        "nucleoli_count": nucleoli_count,
        "nucleoli_prominence": np.max(h_nucleus) / np.mean(h_nucleus) if len(h_nucleus) > 0 else 0,
        "contour_irregularity": fractal_dim,

        # Texture Haralick
        "haralick_contrast": graycoprops(glcm, 'contrast')[0, 0],
        "haralick_homogeneity": graycoprops(glcm, 'homogeneity')[0, 0],
        "haralick_energy": graycoprops(glcm, 'energy')[0, 0],
        "haralick_correlation": graycoprops(glcm, 'correlation')[0, 0],
    }

    return features_advanced

def compute_fractal_dimension(contour, max_box_size=None):
    """
    Box-counting fractal dimension (irrégularité contour)

    Interprétation:
    - FD ~ 1.0 = Contour lisse (cercle parfait)
    - FD ~ 1.3-1.5 = Contour irrégulier (cellules normales)
    - FD > 1.5 = Contour très irrégulier (malignité)
    """
    # Normaliser coordonnées
    contour = contour - contour.min(axis=0)

    # Grille de tailles de boîtes (puissances de 2)
    if max_box_size is None:
        max_box_size = int(np.max(contour))

    sizes = 2 ** np.arange(1, int(np.log2(max_box_size)) + 1)
    counts = []

    for size in sizes:
        grid = contour // size
        counts.append(len(np.unique(grid, axis=0)))

    # Régression log-log
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]  # Dimension fractale
```

**Référence Clinique (Bethesda System — Thyroïde):**

| Feature | Normal | Atypique | Malin |
|---------|--------|----------|-------|
| **N/C Ratio** | < 0.3 | 0.3 - 0.5 | > 0.5 |
| **Chromatin Coarseness** | < 0.3 | 0.3 - 0.5 | > 0.5 |
| **Nucleoli Count** | 0-1 | 1-2 | ≥ 2 |
| **Contour Irregularity** | < 1.3 | 1.3 - 1.5 | > 1.5 |

### C.4. Cyto Head (Classification)

**Architecture:** LightGBM ou MLP

**Input Features:**
```python
# Concaténation H-Optimus embeddings + Morphométrie
input_vector = concat([
    cls_token,                    # 1536 dims (H-Optimus CLS)
    geometric_features,            # 5 dims (area, circularity, etc.)
    optical_density_features,      # 3 dims (mean_od, integrated_od, std_od)
    advanced_cytology_features,    # 9 dims (nc_ratio, chromatin, nucleoli, etc.)
    haralick_features             # 4 dims (texture)
])
# Total: 1536 + 5 + 3 + 9 + 4 = 1557 dims
```

**Classes de Sortie:**

```python
CYTOLOGY_CLASSES = {
    0: "Bénin / Normal",           # NILM (Negative for Intraepithelial Lesion)
    1: "Atypique / Incertain",     # ASC-US, FLUS, etc.
    2: "Malin / Haut Grade"        # HSIL, Carcinome
}
```

**Implémentation LightGBM (Recommandée):**

```python
import lightgbm as lgb

class CytoHead:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            num_leaves=31,
            max_depth=5,
            learning_rate=0.05,
            n_estimators=100,
            objective='multiclass',
            num_class=3,
            class_weight='balanced'  # Gère déséquilibre classes
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        X_train: [N, 1557] Features (embeddings + morpho)
        y_train: [N] Labels (0, 1, 2)
        """
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=10
        )

    def predict(self, features):
        """
        Returns:
            probs: [N, 3] Probabilités par classe
            preds: [N] Classe prédite
        """
        probs = self.model.predict_proba(features)
        preds = np.argmax(probs, axis=1)
        return probs, preds
```

---

## 🔧 Module D: Calibration Clinique

### Fichier de Configuration

**Structure:** `config/calibration_config.json`

```json
{
  "site_id": "dubai_hospital_01",
  "installation_date": "2026-02-01",

  "scanner": {
    "manufacturer": "Aperio",
    "model": "AT2",
    "serial_number": "SN123456789",
    "pixel_size_microns": 0.25,
    "last_maintenance": "2026-01-15",
    "notes": "Scanner principal, histologie + cytologie"
  },

  "stain_normalization": {
    "histology": {
      "enabled": false,
      "reason": "V13 production - Raw images prouvé optimal (AJI -4.3% si Macenko)"
    },
    "cytology": {
      "enabled": true,
      "method": "macenko",
      "target_template_path": "/data/references/pap_smear_ref_scanner_at2.png",
      "lambda_regularization": 0.1,
      "notes": "Template calibré sur scanner AT2 le 2026-01-20"
    }
  },

  "h_channel_calibration": {
    "gain": 1.1,
    "offset": 0.02,
    "expected_background_od": 0.15,
    "alert_threshold_percent": 30,
    "validation_roi": {
      "description": "Zone vide pour contrôle qualité (fond lame)",
      "x": 100,
      "y": 100,
      "width": 50,
      "height": 50
    },
    "notes": "Gain augmenté +10% car scanner AT2 produit images légèrement pâles"
  },

  "morphometry_thresholds": {
    "min_nucleus_area_um2": 15,
    "max_nucleus_area_um2": 300,
    "min_nucleus_circularity": 0.3,
    "nc_ratio_alert": 0.7,
    "notes": "Seuils basés sur Bethesda System (thyroïde)"
  },

  "clinical_alerts": {
    "enabled": true,
    "nc_ratio_high_grade": 0.6,
    "nucleoli_count_malignant": 2,
    "chromatin_coarse_threshold": 0.5
  },

  "performance_targets": {
    "tile_512x512_max_seconds": 2.0,
    "wsi_average_max_minutes": 5.0,
    "gpu_model": "Tesla T4"
  }
}
```

### Logique d'Application Calibration

```python
import json

class ClinicalCalibrator:
    def __init__(self, config_path="config/calibration_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def apply_h_channel_calibration(self, h_channel_raw):
        """
        Applique gain et offset au canal H

        Args:
            h_channel_raw: np.array [H, W] Densité optique brute

        Returns:
            h_calibrated: np.array [H, W] Calibré pour scanner
        """
        gain = self.config["h_channel_calibration"]["gain"]
        offset = self.config["h_channel_calibration"]["offset"]

        h_calibrated = h_channel_raw * gain + offset
        return h_calibrated

    def validate_calibration(self, h_channel, image_rgb):
        """
        Vérifie calibration via ROI de contrôle (fond vide)

        Returns:
            is_valid: bool
            drift_percent: float (déviation vs expected)
        """
        roi_cfg = self.config["h_channel_calibration"]["validation_roi"]
        x, y, w, h = roi_cfg["x"], roi_cfg["y"], roi_cfg["width"], roi_cfg["height"]

        # Extraire ROI fond
        roi_h = h_channel[y:y+h, x:x+w]
        measured_bg = np.mean(roi_h)

        expected_bg = self.config["h_channel_calibration"]["expected_background_od"]
        alert_threshold = self.config["h_channel_calibration"]["alert_threshold_percent"]

        drift_percent = abs(measured_bg - expected_bg) / expected_bg * 100
        is_valid = drift_percent < alert_threshold

        if not is_valid:
            print(f"⚠️ ALERTE CALIBRATION: Drift {drift_percent:.1f}% détecté")
            print(f"   Mesuré: {measured_bg:.3f}, Attendu: {expected_bg:.3f}")

        return is_valid, drift_percent

    def convert_pixels_to_microns(self, area_pixels):
        """
        Convertit aire pixels² → µm²

        Args:
            area_pixels: float (nombre de pixels)

        Returns:
            area_um2: float (aire en µm²)
        """
        pixel_size = self.config["scanner"]["pixel_size_microns"]
        area_um2 = area_pixels * (pixel_size ** 2)
        return area_um2

    def check_clinical_alerts(self, features):
        """
        Vérifie seuils cliniques critiques

        Args:
            features: dict de features morphométriques

        Returns:
            alerts: list de strings (alertes déclenchées)
        """
        alerts = []
        cfg = self.config["clinical_alerts"]

        if not cfg["enabled"]:
            return alerts

        # N/C Ratio élevé (malignité)
        if features.get("nc_ratio", 0) > cfg["nc_ratio_high_grade"]:
            alerts.append(f"N/C Ratio élevé: {features['nc_ratio']:.2f} (seuil {cfg['nc_ratio_high_grade']})")

        # Nucléoles multiples
        if features.get("nucleoli_count", 0) >= cfg["nucleoli_count_malignant"]:
            alerts.append(f"Nucléoles multiples détectés: {features['nucleoli_count']}")

        # Chromatine grossière
        if features.get("chromatin_coarseness", 0) > cfg["chromatin_coarse_threshold"]:
            alerts.append(f"Chromatine grossière: {features['chromatin_coarseness']:.2f}")

        return alerts
```

### Workflow Calibration sur Site

**Étape 1: Installation initiale**
```bash
# 1. Générer image de référence (scan lame contrôle)
python scripts/calibration/generate_reference_template.py \
    --input /path/to/control_slide.svs \
    --output data/references/pap_smear_ref_scanner_at2.png

# 2. Mesurer background OD
python scripts/calibration/measure_background_od.py \
    --reference data/references/pap_smear_ref_scanner_at2.png

# 3. Générer config initial
python scripts/calibration/init_config.py \
    --site dubai_hospital_01 \
    --scanner aperio_at2 \
    --pixel_size 0.25
```

**Étape 2: Validation périodique (mensuelle)**
```bash
# Test lame contrôle
python scripts/calibration/validate_calibration.py \
    --control_slide /path/to/monthly_control.svs \
    --config config/calibration_config.json

# Output:
# ✅ Calibration OK - Drift: 2.3% (< 30%)
# OU
# ⚠️ ALERTE - Drift: 35% → Maintenance scanner requise
```

---

## 📊 Stack Technique

### Frameworks & Librairies

| Composant | Version | Usage |
|-----------|---------|-------|
| **Python** | 3.10+ | Langage principal |
| **PyTorch** | 2.6.0+ | Backbone H-Optimus, Router |
| **CellPose** | 2.0+ | Segmentation cytologie |
| **LightGBM** | 3.3+ | Cyto Head classification |
| **OpenCV** | 4.8+ | Traitement image |
| **Scikit-Image** | 0.21+ | Morphométrie, Haralick |
| **Torch-Stain** | 1.2+ | Normalisation Macenko |
| **FastAPI** | 0.104+ | API REST |
| **Redis** | 7.0+ | Cache features |
| **Celery** | 5.3+ | Queue jobs asynchrones |
| **ONNX Runtime** | 1.16+ | Optimisation inférence |

### Optimisations Performance

**1. H-Optimus-0 → ONNX**

```bash
# Export PyTorch → ONNX
python scripts/optimization/export_hoptimus_onnx.py \
    --checkpoint models/h_optimus_0.pth \
    --output models/h_optimus_0.onnx \
    --opset_version 17

# Quantization INT8 (optionnel, gain 2-3× vitesse)
python -m onnxruntime.quantization.quantize_dynamic \
    --model_input models/h_optimus_0.onnx \
    --model_output models/h_optimus_0_int8.onnx \
    --per_channel
```

**Attention TensorRT:** ViT-Giant (1.1B params) peut échouer avec TensorRT custom layers. ONNX Runtime plus stable.

**2. Cache Redis Features**

```python
import redis
import pickle

class FeatureCache:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.ttl = 3600  # 1 heure

    def get_features(self, tile_hash):
        """Récupère features depuis cache"""
        cached = self.redis.get(f"features:{tile_hash}")
        if cached:
            return pickle.loads(cached)
        return None

    def set_features(self, tile_hash, features):
        """Sauvegarde features dans cache"""
        self.redis.setex(
            f"features:{tile_hash}",
            self.ttl,
            pickle.dumps(features)
        )
```

**3. API Endpoints**

```python
from fastapi import FastAPI, UploadFile
from celery import Celery

app = FastAPI()
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@app.post("/analyze/tile")
async def analyze_tile_sync(file: UploadFile):
    """
    Analyse synchrone (temps réel)
    Target: < 2s sur GPU T4
    """
    image = load_image(file)
    result = process_tile(image)  # Pipeline complet
    return result

@app.post("/analyze/wsi")
async def analyze_wsi_async(file: UploadFile):
    """
    Analyse asynchrone (queue)
    Target: Complétion < 5 min
    """
    task = celery_app.send_task('process_wsi', args=[file.filename])
    return {"task_id": task.id, "status": "queued"}

@celery_app.task
def process_wsi(wsi_path):
    """Task Celery pour WSI complète"""
    # Tiling → Process tiles → Aggregate
    pass
```

---

## 📋 Plan d'Implémentation

### Phase 1: Infrastructure (Semaines 1-2)

**Objectif:** Architecture en Y fonctionnelle

**Livrables:**
```
[ ] Architecture V14HybridSystem (router + 2 branches)
[ ] Preprocessing router-dependent (Macenko conditionnel)
[ ] CellPose intégration (tests zero-shot sur Herlev)
[ ] Calibration config JSON + ClinicalCalibrator class
[ ] Tests non-régression V13 (AJI Respiratory ≥ 0.6872)
```

**Scripts à créer:**
```
src/models/v14_hybrid_system.py
src/preprocessing/router_dependent_preprocessing.py
src/calibration/clinical_calibrator.py
tests/test_v14_non_regression.py
```

### Phase 2: Router Training (Semaine 3)

**Objectif:** Router accuracy > 98%

**Dataset:**
- Histologie: 5,000 images PanNuke (label=0)
- Cytologie: 5,000 images Herlev + TB-PANDA (label=1)

**Livrables:**
```
[ ] Dataset router préparé (train/val split)
[ ] RouterHead ou RouterHeadEnhanced training
[ ] Validation accuracy > 98%
[ ] Implémentation zone "uncertain" (0.15 < P < 0.85)
[ ] Export router checkpoint: models/router_v14.pth
```

**Métriques:**
- Accuracy: > 98%
- Recall Cyto: > 97% (critique: ne pas manquer cytologie)
- Recall Histo: > 99% (V13 ne doit pas être perturbé)

### Phase 3: Cytology Pipeline (Semaines 4-6)

**Objectif:** Pipeline Cyto complet fonctionnel

**Livrables:**
```
[ ] CellPose fine-tuning (si zero-shot < 90%)
[ ] Extraction features morphométriques complètes
    [ ] Géométrie (area, circularity, eccentricity)
    [ ] Canal H (mean_od, integrated_od, std_od)
    [ ] Features avancées (nc_ratio, chromatin, nucleoli, fractal_dim)
    [ ] Texture Haralick
[ ] Pseudo-labeling sur dataset 70k images
[ ] Training Cyto Head (LightGBM)
[ ] Calibration h_channel_gain validation
```

**Datasets:**
- TB-PANDA (Thyroïde): 10,000 images
- Herlev (Col): 917 images
- Urine: TBD (à sourcer)

**Target Performance:**
- Sensibilité "Malin": > 95%
- Spécificité "Bénin": > 90%
- Classe "Atypique": Recall > 80%

### Phase 4: Optimisation & Validation (Semaine 7)

**Objectif:** Prêt pour production

**Livrables:**
```
[ ] Export H-Optimus → ONNX (gain vitesse 2-3×)
[ ] Cache Redis features
[ ] API FastAPI endpoints (/analyze/tile, /analyze/wsi)
[ ] Tests multi-scanners (3 scanners différents)
[ ] Validation variance IOD < 10% après calibration
[ ] Documentation calibration sur site
```

**Performance Targets:**
- Tile 512×512: < 2s sur GPU T4
- WSI complète: < 5 min (moyenne)

### Phase 5: Production (Semaine 8)

**Objectif:** Déploiement Dubai

**Livrables:**
```
[ ] Installation sur site (Dubai Hospital)
[ ] Calibration scanner initial (template + config)
[ ] Formation pathologistes
[ ] Tests 100 lames réelles (50 Histo + 50 Cyto)
[ ] Monitoring performance
[ ] Rapport validation clinique
```

---

## 📊 Métriques de Validation Cytologie — KPIs Critiques

> **⚠️ CHANGEMENT PARADIGME:** L'AJI (métrique V13 Histologie) est INADAPTÉ pour la cytologie.
>
> En cytologie, le problème n'est pas de séparer noyaux collés, mais de **trouver l'aiguille dans la botte de foin** (cellule rare anormale).
>
> **Focus:** **Sensibilité > Précision** (Safety First — Ne jamais rater un cancer)
>
> **Documentation complète:** [V14_MASTER_SLAVE_ARCHITECTURE.md#métriques-de-validation-cytologie](./V14_MASTER_SLAVE_ARCHITECTURE.md#-métriques-de-validation-cytologie--kpis-critiques)

### Tableau Récapitulatif KPIs

| # | Catégorie | Métrique | Seuil Cible | Justification |
|---|-----------|----------|-------------|---------------|
| 1 | **Segmentation** | IoU Noyau | **> 0.85** | Précision géométrique pour Canal H et N/C ratio |
| 2 | Segmentation | IoU Cytoplasme | > 0.70 | Bords flous (tolérance large) |
| 3 | Segmentation | AP50 (COCO) | > 0.90 | Standard Kaggle, valide détection + segmentation |
| 4 | Segmentation | PQ (Panoptic Quality) | > 0.75 | Métrique moderne (remplace AJI) |
| 5 | **Dépistage (CRITIQUE)** | **Sensibilité Malin** | **> 98%** | **Safety First — Ne jamais rater un cancer** |
| 6 | Dépistage | Sensibilité Atypique | > 95% | Surveillance rapprochée requise |
| 7 | Dépistage | **FROC (FP/WSI @ 98% sens)** | **< 2.0** | **Productivité pathologiste** |
| 8 | Dépistage | Spécificité | > 60-70% | Éviter surcharge fausses alertes |
| 9 | **Diagnostic** | **Cohen's Kappa** | **> 0.80** | **Accord Expert Level avec pathologiste** |
| 10 | Diagnostic | Matrice Confusion | 0 cancer raté | Vérifier erreurs critiques |

### Pourquoi Sensibilité > Accuracy?

**Exemple trompeur:**
- Dataset Cytologie typique: 95% Normal, 5% Anormal
- Modèle naïf prédisant "TOUT Normal" → **Accuracy = 95%** ✅
- Mais **Sensibilité = 0%** (rate 100% des cancers!) ❌

**Métriques prioritaires V14 Cytologie (ordre):**
1. **Sensibilité Malin** (> 98%) — Ne JAMAIS rater un cancer
2. **FROC** (< 2 FP/WSI @ 98% sens) — Productivité pathologiste
3. **Cohen's Kappa** (> 0.80) — Accord expert
4. IoU/AP50 (> 0.85/0.90) — Précision segmentation
5. Spécificité (> 60%) — Éviter surcharge

### Argument Commercial Dubai

> *"Notre système V14 Cytologie ne rate JAMAIS une cellule anormale (Sensibilité 99%), là où un humain fatigué en rate 5 à 10% (études montrent Sensibilité humaine ~90-95% en routine)."*

**Différenciateur vs Genius (Roche):**

| Aspect | Genius (Roche) | CellViT V14 Cytologie | Avantage |
|--------|----------------|----------------------|----------|
| **Sensibilité Malin** | ~95% (estimé) | **> 98%** ✅ | Safety First |
| **FROC (FP/WSI)** | ~3-4 FP/WSI | **< 2 FP/WSI** ✅ | Productivité |
| **Cohen's Kappa** | ~0.75 | **> 0.80 (Expert Level)** ✅ | Confiance clinique |
| **Focus** | Accuracy globale | **Sensibilité (ne jamais rater cancer)** | Priorité sécurité |

### Implémentation Tests Validation

**Tests critiques requis:**

```python
# Test 1: Sensibilité Malin (CRITICAL)
from sklearn.metrics import recall_score

sensitivity_malin = recall_score(
    y_true_binary,
    y_pred_binary,
    pos_label="malignant"
)

assert sensitivity_malin > 0.98, \
    f"⚠️ SAFETY CRITICAL: Sensibilité {sensitivity_malin:.3f} < 98%"

# Test 2: FROC (Productivité)
sens, fps_per_wsi, auc_froc = compute_froc_curve(predictions, ground_truth, n_wsi)

idx_98_sens = np.argmin(np.abs(np.array(sens) - 0.98))
fp_at_98_sens = fps_per_wsi[idx_98_sens]

assert fp_at_98_sens < 2.0, \
    f"⚠️ FROC KPI NON ATTEINT: {fp_at_98_sens:.1f} FP/WSI à 98% sensibilité"

# Test 3: Cohen's Kappa (Expert Level)
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(expert_labels, ai_predictions, weights='quadratic')

assert kappa > 0.80, f"Kappa trop bas: {kappa:.3f} (vs 0.80 requis)"

# Test 4: Matrice Confusion (0 cancer raté)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred, labels=classes)

malin_missed = cm[classes.index("malignant"), classes.index("normal")]
assert malin_missed == 0, \
    f"⚠️ ERREUR CRITIQUE: {malin_missed} cancers classés Normal"
```

**Documentation détaillée:**
- Formules mathématiques
- Implémentation complète
- Benchmarks comparatifs
- Voir: [V14_MASTER_SLAVE_ARCHITECTURE.md#métriques](./V14_MASTER_SLAVE_ARCHITECTURE.md#-métriques-de-validation-cytologie--kpis-critiques)

---

## ❓ Questions Ouvertes

### 🔴 Critiques (Bloquants)

1. **Macenko Strategy:**
   - ✅ **DÉCISION REQUISE:** Router-dependent (Macenko uniquement Cyto) ou autre approche?
   - Impact: Architecture preprocessing

2. **Router Dataset:**
   - ❓ Avez-vous déjà images Cyto labellisées pour router training?
   - Si non: Plan pseudo-labeling initial?

3. **Cytoplasm Segmentation:**
   - ❓ CellPose peut segmenter cytoplasme OU seulement noyau?
   - Impact: Calcul N/C ratio (critique en cytologie)
   - Alternative: Watershed expansion depuis noyau?

4. **V13 Non-Régression:**
   - ✅ **CONFIRMATION:** Respiratory AJI ≥ 0.6872 est hard requirement?
   - Test automatisé dans CI/CD?

### 🟡 Importantes (Planification)

5. **Priorité Organes Cyto:**
   - ❓ Ordre: Thyroïde, Col utérin, Urine?
   - Impact: Datasets à sourcer en priorité

6. **Datasets Cytologie:**
   - TB-PANDA (Thyroïde): ✅ Identifié
   - Herlev (Col): ✅ Identifié
   - Urine: ❓ Source TBD
   - Total 70k images: ❓ Répartition?

7. **Target Template Macenko:**
   - ❓ Une seule référence globale OU par organe (thyroïde/col/urine)?
   - Impact: Nombre de configs calibration

### 🟢 Techniques (Optimisations)

8. **Router Architecture:**
   - RouterHead simple (CLS only) OU RouterHeadEnhanced (CLS + patch variance)?
   - A/B test recommandé

9. **Cyto Head:**
   - LightGBM (recommandé) OU MLP PyTorch?
   - LightGBM = Plus rapide, interprétable
   - MLP = Plus flexible, end-to-end training

10. **ONNX vs TensorRT:**
    - ONNX Runtime: Stable, supporté ViT
    - TensorRT: Plus rapide mais risque échec custom layers
    - Test requis sur H-Optimus-0 (1.1B params)

---

## 📝 Changelog

### Version 14.0 — 2026-01-18 (Spécifications Initiales)

**Ajouté:**
- Architecture en Y (Router + Histo + Cyto)
- Spécifications techniques complètes Module A-D
- Alertes critiques Macenko vs V13
- Features morphométriques avancées (N/C, chromatin, nucleoli, fractal)
- Calibration clinique multi-scanners
- Plan d'implémentation 8 semaines
- Questions ouvertes (10 items)

**Recommandations Clés:**
1. Preprocessing router-dependent (Macenko uniquement Cyto)
2. Zone d'incertitude Router (0.15-0.85)
3. RouterHeadEnhanced (CLS + patch variance)
4. LightGBM pour Cyto Head
5. Tests non-régression V13 obligatoires

**Décisions en Attente:**
- Validation approche Macenko router-dependent
- Confirmation hard requirement AJI V13
- Stratégie segmentation cytoplasme (N/C ratio)
- Priorité organes cytologie

---

## 🔗 Références

### Documentation Projet

- **CLAUDE.md:** Source de vérité (règles, résultats V13, découvertes)
- **claude_history.md:** Historique complet développement
- **V13_SMART_CROPS_STRATEGY.md:** Stratégie architecture V13
- **UI_COCKPIT.md:** IHM Gradio R&D

### Datasets

- **PanNuke:** https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
- **Herlev (Col utérin):** http://mde-lab.aegean.gr/index.php/downloads
- **TB-PANDA (Thyroïde):** https://github.com/ncbi/TB-PANDA

### Publications Scientifiques

- **Ruifrok & Johnston (2001):** "Quantification of histochemical staining by color deconvolution"
- **Bethesda System:** Thyroid cytopathology classification
- **Papanicolaou System:** Cervical cytology classification
- **CellPose (Stringer et al. 2021):** "Cellpose: a generalist algorithm for cellular segmentation"

### Librairies Techniques

- **H-optimus-0:** https://huggingface.co/bioptimus/H-optimus-0
- **CellPose:** https://github.com/MouseLand/cellpose
- **Torch-Stain:** https://github.com/EIDOSLAB/torchstain
- **LightGBM:** https://lightgbm.readthedocs.io/

---

**Statut:** 🚧 En spécification — Attend validation approche Macenko et décisions techniques critiques

**Prochaine Étape:** Réponses questions ouvertes → Création scripts Phase 1 (Infrastructure)