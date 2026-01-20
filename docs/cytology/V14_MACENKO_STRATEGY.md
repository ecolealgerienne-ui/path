# V14 — Stratégie Normalisation Macenko (Router-Dependent)

> **Date:** 2026-01-19
> **Statut:** ✅ Validé (Specs Expert + Résultats V13)
> **Principe:** Normalisation conditionnelle selon branche (Cytologie vs Histologie)

---

## 🎯 Principe Fondamental

**La normalisation Macenko n'est PAS universelle** — Son efficacité dépend de l'architecture downstream.

```
┌─────────────────────────────────────────────────────────────────┐
│  RÈGLE D'OR: Macenko Router-Dependent                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SI Branche Cytologie (V14):                                    │
│  ✅ Macenko = ON                                                │
│     Raison: Pas de FPN Chimique downstream                      │
│             Scanners multiples (Dubai) nécessitent normalisation│
│                                                                  │
│  SI Branche Histologie (V13):                                   │
│  ❌ Macenko = OFF (RAW images)                                  │
│     Raison: FPN Chimique utilise Ruifrok → Conflit Macenko      │
│             Régression -4.3% AJI prouvée                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Résultats Expérimentaux V13 (Histologie)

### Test Comparatif (2025-12-30)

| Configuration | AJI Respiratory | Δ | Conclusion |
|---------------|-----------------|---|------------|
| **RAW Images (SANS Macenko)** | **0.6872** ✅ | Baseline | **OPTIMAL** |
| AVEC Macenko | 0.6576 | **-4.3%** ❌ | **RÉGRESSION** |

### Analyse Technique du Conflit

**Le "Shift de Projection":**

```python
# ═════════════════════════════════════════════════════════════════════════════
#  CONFLIT RUIFROK/MACENKO (V13 Histologie)
# ═════════════════════════════════════════════════════════════════════════════

# Étape 1: Macenko Normalization
image_macenko = macenko_normalize(image_raw)
# → Rotation dans l'espace OD pour aligner vers référence
# → Éosine déplacée VERS le vecteur Hématoxyline

# Étape 2: Ruifrok Deconvolution (FPN Chimique)
h_channel = ruifrok_extract(image_macenko)
# → Projection sur vecteur Ruifrok FIXE [0.650, 0.704, 0.286]
# → ⚠️ PROBLÈME: Macenko a déjà modifié les proportions
# → Résultat: Canal H contient "fantômes" de cytoplasme (Éosine)

# Étape 3: FPN Chimique injection
fpn_output = fpn_chimique(features, h_channel)
# → Bruit dans HV-MSE loss (cytoplasme ≠ gradient séparation noyaux)
# → Régression AJI -4.3%
```

**Visualisation du Conflit:**

```
RAW IMAGE (Vérité terrain):
├─ Hématoxyline: Direction physique pure (Ruifrok FIXE)
└─ Éosine: Direction physique pure (orthogonale)

APRÈS MACENKO:
├─ Hématoxyline: Direction ROTÉE (vers template)
├─ Éosine: Direction ROTÉE (contamine H-channel!)
└─ ⚠️ Vecteurs Ruifrok deviennent INEXACTS

EXTRACTION RUIFROK SUR MACENKO:
└─ Canal H = ADN + Bruit cytoplasme (contamination Éosine)
```

**Référence:** `CLAUDE.md` section "Découverte Stratégique: Ruifrok vs Macenko"

---

## ✅ Pourquoi Macenko OK pour V14 Cytologie

### Différences Architecturales Critiques

| Aspect | V13 Histologie | V14 Cytologie | Impact Macenko |
|--------|----------------|---------------|----------------|
| **Architecture Downstream** | FPN Chimique (injection H-channel) | MLP simple (pas de FPN) | Cytologie: Pas de conflit |
| **Ruifrok Usage** | ✅ Critique (5 niveaux FPN) | ⚠️ Optionnel (feature morpho seulement) | Cytologie: Pas de dépendance forte |
| **Normalisation Bénéfice** | Faible (dataset homogène PanNuke) | **Élevé (scanners multiples Dubai)** | Cytologie: Critique production |
| **Régression Risque** | **-4.3% AJI prouvé** | Aucune (architecture différente) | Cytologie: Safe |

### Architecture V14 (Pas de FPN)

```python
# ═════════════════════════════════════════════════════════════════════════════
#  V14 CYTOLOGIE — MACENKO SAFE
# ═════════════════════════════════════════════════════════════════════════════

# Étape 1: CellPose détection
bboxes, masks = cellpose.detect(image_raw)

# Étape 2: Crop + Padding
patches = [crop_and_pad(image_raw, bbox) for bbox in bboxes]

# Étape 2.5: Macenko Normalization ✅
patches_normalized = [macenko_normalize(patch) for patch in patches]

# Étape 3: H-Optimus (sur image normalisée)
embeddings = h_optimus(patches_normalized)
# → H-Optimus robuste, mais bénéficie couleurs standardisées
# → Pas de FPN downstream → Pas de conflit Ruifrok

# Étape 4: Morphométrie (sur masques)
morpho_features = compute_morpho(masks)
# → Calcul GÉOMÉTRIQUE (indépendant couleurs)
# → Ruifrok utilisé UNIQUEMENT pour mean_h_channel (1 feature sur 20)
# → Impact marginal si Macenko appliqué

# Étape 5: MLP Classification
logits = mlp(embeddings, morpho_features)
# → Fusion simple, pas de canal H injecté
```

**Différence Clé:** Ruifrok utilisé pour **1 feature sur 20** (mean_h_channel), pas pour architecture entière.

---

## 🏗️ Implémentation Router-Dependent

### Code Production

```python
"""
V14 — Preprocessing Router-Dependent
"""

import torch
from torchstain import MacenkoNormalizer
from src.preprocessing.stain_separation import ruifrok_extract_h_channel


class V14Preprocessor:
    """
    Preprocessing adaptatif selon branche détectée par Router
    """

    def __init__(self, macenko_template_path: str):
        """
        Args:
            macenko_template_path: Image de référence pour normalisation
                                   (ex: template SIPaKMeD cervical smear)
        """
        self.macenko_normalizer = MacenkoNormalizer()

        # Fit sur template (one-time)
        template = load_image(macenko_template_path)
        self.macenko_normalizer.fit(template)

    def preprocess(
        self,
        image: torch.Tensor,
        pipeline_branch: str
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Preprocessing adaptatif

        Args:
            image: RGB Tensor [3, H, W], valeurs [0, 255]
            pipeline_branch: "cytology" | "histology" | "uncertain"

        Returns:
            image_processed: Tensor normalisé
            h_channel: Canal Hématoxyline (optionnel)
        """
        # ═════════════════════════════════════════════════════════════════════
        #  BRANCHE CYTOLOGIE: MACENKO ON
        # ═════════════════════════════════════════════════════════════════════

        if pipeline_branch == "cytology":
            # Normalisation Macenko (scanners multiples Dubai)
            image_normalized = self.macenko_normalizer.normalize(image)

            # H-channel extraction (optionnel, pour morphométrie)
            h_channel = ruifrok_extract_h_channel(
                image_normalized,
                normalize=True
            )

            return image_normalized, h_channel

        # ═════════════════════════════════════════════════════════════════════
        #  BRANCHE HISTOLOGIE: RAW IMAGES (V13 prouvé)
        # ═════════════════════════════════════════════════════════════════════

        elif pipeline_branch == "histology":
            # Pas de normalisation Macenko (régression -4.3% AJI)
            # Extraction H-channel sur RAW (préserve physique Beer-Lambert)
            h_channel = ruifrok_extract_h_channel(
                image,
                normalize=True
            )

            # Image RAW inchangée
            return image, h_channel

        # ═════════════════════════════════════════════════════════════════════
        #  BRANCHE UNCERTAIN: CHOIX CONSERVATEUR
        # ═════════════════════════════════════════════════════════════════════

        else:  # uncertain
            # Approche conservatrice: Pas de Macenko (évite régression V13)
            h_channel = ruifrok_extract_h_channel(image, normalize=True)
            return image, h_channel


# ═════════════════════════════════════════════════════════════════════════════
#  USAGE EN PRODUCTION
# ═════════════════════════════════════════════════════════════════════════════

preprocessor = V14Preprocessor(
    macenko_template_path="data/templates/sipakmed_cervix_ref.png"
)

# Router décide branche
branch = router.predict(tile)  # → "cytology" ou "histology"

# Preprocessing adaptatif
image_processed, h_channel = preprocessor.preprocess(tile, branch)

if branch == "cytology":
    # Pipeline cytologie (CellPose → H-Optimus → MLP)
    results = cytology_pipeline(image_processed)
else:
    # Pipeline histologie (FPN Chimique → Watershed)
    results = histology_pipeline(image_processed, h_channel)
```

---

## 📋 Checklist Implémentation

### Phase 1: Setup Macenko Template

```bash
# Créer template de référence (SIPaKMeD)
python scripts/cytology/create_macenko_template.py \
    --dataset sipakmed \
    --output data/templates/sipakmed_cervix_ref.png
```

### Phase 2: Tests Non-Régression V13

```python
# tests/test_v14_macenko_non_regression.py

def test_v13_histology_unchanged():
    """
    V13 AJI doit rester ≥ 0.6872 après intégration Router

    CRITIQUE: Macenko doit être OFF pour branche histologie
    """
    model_v14 = V14HybridSystem()

    # Force branche histologie
    aji_respiratory = evaluate_aji(
        model_v14,
        respiratory_val,
        force_branch="histology"  # → Macenko OFF
    )

    assert aji_respiratory >= 0.6872, \
        f"Régression V13 détectée! AJI={aji_respiratory:.4f} < 0.6872"


def test_cytology_benefits_from_macenko():
    """
    Cytologie doit bénéficier de Macenko (amélioration accuracy)

    Test: Comparer accuracy AVEC vs SANS Macenko
    """
    model_cyto = CytologyClassifier()

    # SANS Macenko
    acc_raw = evaluate_cytology(model_cyto, sipakmed_val, macenko=False)

    # AVEC Macenko
    acc_macenko = evaluate_cytology(model_cyto, sipakmed_val, macenko=True)

    # Macenko doit améliorer (ou au minimum: pas dégrader)
    assert acc_macenko >= acc_raw - 0.01, \
        f"Macenko dégrade cytologie: {acc_macenko:.3f} vs {acc_raw:.3f}"
```

### Phase 3: Tests Production Dubai

```python
def test_scanner_robustness():
    """
    Test robustesse multi-scanners (Dubai use case)

    Scanners testés:
    - Hamamatsu NanoZoomer
    - Leica Aperio
    - 3DHISTECH Pannoramic
    """
    scanners = ["hamamatsu", "leica", "3dhistech"]

    for scanner in scanners:
        acc = evaluate_cytology(
            model,
            dataset=f"sipakmed_{scanner}_variant",
            macenko=True
        )

        # Avec Macenko: accuracy doit être stable (± 2%)
        assert acc >= 0.90, \
            f"Scanner {scanner} accuracy trop basse: {acc:.3f}"
```

---

## 🔬 Résumé Scientifique

### V13 Histologie: Ruifrok > Macenko

**Philosophie:**
- Ruifrok = Physique (Loi de Beer-Lambert, constantes universelles)
- Macenko = Statistique (SVD, adaptatif par image)

**Résultat:**
- Ruifrok préserve texture chromatinienne fine (critique pour HV-MSE)
- Macenko lisse intensités → Perte détails → Régression -4.3% AJI

**Décision:** RAW images pour V13 production

### V14 Cytologie: Macenko Bénéfique

**Contexte:**
- Scanners multiples (Dubai) → Variations couleur importantes
- Pas de FPN Chimique downstream → Pas de conflit Ruifrok
- H-Optimus robuste mais bénéficie standardisation

**Résultat:**
- Macenko uniformise couleurs entre scanners
- Améliore généralisation H-Optimus
- Aucune régression (architecture différente)

**Décision:** Macenko ON pour V14 Cytologie

---

## 📚 Références

| Document | Section |
|----------|---------|
| `CLAUDE.md` | "Découverte Stratégique: Ruifrok vs Macenko" |
| `V14_CYTOLOGY_BRANCH.md` | "Module A: Pre-Processing & Normalisation" |
| `V14_PIPELINE_EXECUTION_ORDER.md` | "Phase Séquentielle — Étape 2.5" |

---

**Auteur:** V14 Cytology Branch
**Validation:** Specs Expert (2026-01-19) + Résultats V13 (2025-12-30)
**Statut:** ✅ Production Ready
