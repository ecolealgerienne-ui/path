# Analyse du Code - Besoins de Refactoring

**Date:** 2025-12-22
**Analyse par:** Claude Code
**Contexte:** Session de review après bugs critiques de normalisation

---

## 🔴 Problèmes Critiques Identifiés

### 1. Duplication Massive de Code (102 fichiers Python)

#### A. Fonctions de Preprocessing Dupliquées

**Locations identifiées:**
- `src/inference/optimus_gate_inference.py` → `create_hoptimus_transform()`, `preprocess()`
- `src/inference/hoptimus_hovernet.py` → `create_hoptimus_transform()`, `preprocess()`
- `src/inference/optimus_gate_inference_multifamily.py` → `create_hoptimus_transform()`, `preprocess()`
- `src/inference/hoptimus_unetr.py` → `create_hoptimus_transform()`, `preprocess()`
- `src/inference/cellvit_inference.py` → `preprocess()`
- `src/inference/cellvit_official.py` → `preprocess()`
- `scripts/preprocessing/extract_features.py` → logique de preprocessing inline
- `scripts/preprocessing/extract_fold_features.py` → logique de preprocessing inline
- `scripts/demo/gradio_demo.py` → logique de preprocessing inline
- Et 15+ autres fichiers...

**Impact:** Chaque modification doit être répliquée manuellement dans ~15 fichiers → risque d'oubli → bugs !

#### B. Constantes de Normalisation Dupliquées

**Trouvées dans 15+ fichiers:**
```python
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
```

**Risque:** Si on veut changer ces valeurs (ex: nouvelle version du modèle), il faut les modifier dans 15+ endroits.

#### C. Logique de Chargement de Modèle Dupliquée

Chaque script charge H-optimus-0 différemment:
- Certains avec `timm.create_model()`
- Certains avec `torch.hub.load()`
- Certains avec gestion du cache HuggingFace
- Certains sans gestion d'erreur

### 2. Incohérences de Normalisation (Bugs récents)

**Historique des bugs découverts:**

| Bug | Description | Impact | Statut |
|-----|-------------|--------|--------|
| **ToPILImage float64** | ToPILImage multiplie les floats par 255 | Features corrompues, ré-entraînement complet | ✅ Corrigé |
| **LayerNorm mismatch** | `blocks[23]` vs `forward_features()` | CLS std 0.28 vs 0.77, prédictions fausses | ✅ Corrigé |
| **Instance mismatch** | Training avec connectedComponents, GT avec vraies instances | Watershed ne sépare pas les cellules | ⚠️ En cours |

**Cause racine commune:** Pas de module centralisé pour normalisation → chacun fait sa version → incohérences.

### 3. Scripts de Validation/Test Multiples

**Trouvés:**
- `scripts/validation/verify_features.py`
- `scripts/validation/verify_pipeline.py`
- `scripts/validation/validate_preprocessing_pipeline.py`
- `scripts/validation/diagnose_organ_prediction.py`
- `scripts/validation/diagnose_ood_issue.py`
- `scripts/evaluation/verify_training_features.py`
- `scripts/evaluation/compare_train_vs_inference.py`
- `scripts/evaluation/verify_dice_bug.py`

**Problème:** Beaucoup de chevauchement de fonctionnalités, pas de tests unitaires structurés.

---

## 📊 Statistiques du Code

```
Total fichiers Python: 102
Duplications identifiées:
  - create_hoptimus_transform(): ~8 copies
  - preprocess(): ~10 copies
  - HOPTIMUS_MEAN/STD: ~15 copies
  - Logique de chargement modèle: ~12 copies
```

---

## ✅ Plan de Refactoring Proposé

### Phase 1: Modules Centralisés (Priorité CRITIQUE)

#### 1.1 Créer `src/preprocessing/` Module

**Fichier:** `src/preprocessing/__init__.py`

```python
"""
Module centralisé pour preprocessing des images H&E.

TOUTES les opérations de normalisation DOIVENT passer par ce module.
"""

from torchvision import transforms
import torch
import numpy as np

# ============================================================================
# CONSTANTES GLOBALES (source unique de vérité)
# ============================================================================

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_IMAGE_SIZE = 224

# ============================================================================
# TRANSFORM CANONIQUE
# ============================================================================

def create_hoptimus_transform() -> transforms.Compose:
    """
    Crée la transformation CANONIQUE pour H-optimus-0.

    RÈGLES:
    1. L'image d'entrée DOIT être uint8 [0-255] avant ToPILImage
    2. Cette fonction DOIT être utilisée PARTOUT (train + inference)
    3. Ne JAMAIS modifier sans mettre à jour TOUS les modèles

    Returns:
        Transform torchvision
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((HOPTIMUS_IMAGE_SIZE, HOPTIMUS_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPTIMUS_MEAN, std=HOPTIMUS_STD),
    ])

# ============================================================================
# PREPROCESSING UNIFIÉ
# ============================================================================

def preprocess_image(
    image: np.ndarray,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Prétraite une image H&E pour inférence.

    ÉTAPES CRITIQUES:
    1. Conversion uint8 (évite bug ToPILImage)
    2. Transform canonique
    3. Batch dimension
    4. Device placement

    Args:
        image: Image RGB (H, W, 3) - uint8 ou float
        device: Device PyTorch

    Returns:
        Tensor (1, 3, 224, 224) normalisé, prêt pour H-optimus-0

    Raises:
        ValueError: Si l'image n'est pas RGB ou dimensions invalides
    """
    # Validation
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got {image.shape}")

    # CRITIQUE: Conversion uint8 AVANT ToPILImage
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            # Image normalisée [0, 1]
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            # Image déjà [0, 255]
            image = image.clip(0, 255).astype(np.uint8)

    # Transform canonique
    transform = create_hoptimus_transform()
    tensor = transform(image)

    # Batch dimension + device
    tensor = tensor.unsqueeze(0).to(device)

    return tensor

# ============================================================================
# VALIDATION
# ============================================================================

def validate_features(features: torch.Tensor) -> dict:
    """
    Valide que les features extraites sont correctes.

    CRITÈRES:
    - CLS token std doit être entre 0.70 et 0.90
    - Shape doit être (B, 261, 1536) pour H-optimus-0

    Args:
        features: Features de H-optimus-0

    Returns:
        dict avec {valid: bool, cls_std: float, message: str}
    """
    cls_token = features[:, 0, :]
    cls_std = cls_token.std().item()

    valid = 0.70 <= cls_std <= 0.90

    return {
        "valid": valid,
        "cls_std": cls_std,
        "shape": tuple(features.shape),
        "message": (
            f"✅ Features valides (CLS std={cls_std:.3f})"
            if valid else
            f"❌ Features CORROMPUES (CLS std={cls_std:.3f}, attendu 0.70-0.90)"
        )
    }
```

#### 1.2 Créer `src/models/loader.py` Module

**Fichier:** `src/models/loader.py`

```python
"""
Module centralisé pour chargement des modèles.
"""

import timm
import torch
from pathlib import Path
from typing import Optional

class ModelLoader:
    """Chargeur unifié pour tous les modèles du projet."""

    @staticmethod
    def load_hoptimus0(
        device: str = "cuda",
        cache_dir: Optional[Path] = None
    ) -> torch.nn.Module:
        """
        Charge H-optimus-0 depuis HuggingFace.

        Args:
            device: Device PyTorch
            cache_dir: Répertoire de cache HuggingFace

        Returns:
            Modèle H-optimus-0 gelé en mode eval

        Raises:
            RuntimeError: Si accès refusé (token HF invalide)
        """
        try:
            model = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False
            )
            model = model.to(device)
            model.eval()

            # Geler tous les paramètres
            for param in model.parameters():
                param.requires_grad = False

            return model

        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                raise RuntimeError(
                    "Accès H-optimus-0 refusé. Vérifiez votre token HuggingFace:\n"
                    "1. huggingface-cli login\n"
                    "2. Token doit avoir 'Read access to public gated repos'\n"
                    f"Erreur: {e}"
                )
            raise

    @staticmethod
    def load_organ_head(
        checkpoint_path: Path,
        device: str = "cuda"
    ) -> torch.nn.Module:
        """Charge OrganHead depuis checkpoint."""
        from src.models.organ_head import OrganHead

        model = OrganHead(embed_dim=1536, num_organs=19)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        return model

    @staticmethod
    def load_hovernet(
        checkpoint_path: Path,
        device: str = "cuda"
    ) -> torch.nn.Module:
        """Charge HoVer-Net depuis checkpoint."""
        from src.models.hovernet_decoder import HoVerNetDecoder

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = HoVerNetDecoder(
            embed_dim=1536,
            num_classes=6,
            dropout=0.1
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        return model
```

#### 1.3 Mise à Jour de Tous les Fichiers d'Inférence

**Avant:**
```python
# Dans chaque fichier
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

def create_hoptimus_transform():
    # 10 lignes de code dupliqué
    ...

def preprocess(self, image):
    # 20 lignes de code dupliqué
    ...
```

**Après:**
```python
from src.preprocessing import preprocess_image, validate_features
from src.models.loader import ModelLoader

class OptimusGateInference:
    def __init__(self, ...):
        self.backbone = ModelLoader.load_hoptimus0(device)
        self.organ_head = ModelLoader.load_organ_head(organ_path, device)
        self.hovernet = ModelLoader.load_hovernet(hovernet_path, device)

    def predict(self, image: np.ndarray):
        # Preprocessing unifié
        tensor = preprocess_image(image, self.device)

        # Extraction features
        features = self.backbone.forward_features(tensor)

        # Validation
        validation = validate_features(features)
        if not validation["valid"]:
            raise RuntimeError(validation["message"])

        # Reste de la logique...
```

---

### Phase 2: Tests Structurés

#### 2.1 Créer `tests/` Hiérarchie Propre

```
tests/
├── unit/
│   ├── test_preprocessing.py       # Tests unitaires preprocessing
│   ├── test_model_loading.py       # Tests unitaires loader
│   ├── test_organ_head.py          # Tests unitaires OrganHead
│   └── test_hovernet.py            # Tests unitaires HoVer-Net
├── integration/
│   ├── test_pipeline_e2e.py        # Test complet image→résultat
│   └── test_train_inference_consistency.py  # Cohérence train/inference
└── fixtures/
    ├── sample_images/               # Images de test
    └── expected_outputs/            # Résultats attendus
```

#### 2.2 Tests de Non-Régression

```python
# tests/integration/test_train_inference_consistency.py

def test_preprocessing_consistency():
    """Vérifie que le preprocessing est identique partout."""
    from src.preprocessing import preprocess_image

    image = load_test_image()

    # Le preprocessing doit donner le même résultat
    tensor1 = preprocess_image(image, "cuda")
    tensor2 = preprocess_image(image, "cuda")

    assert torch.allclose(tensor1, tensor2)

def test_cls_token_std():
    """Vérifie que CLS std est dans la plage attendue."""
    from src.preprocessing import preprocess_image, validate_features
    from src.models.loader import ModelLoader

    backbone = ModelLoader.load_hoptimus0()
    image = load_test_image()

    tensor = preprocess_image(image)
    features = backbone.forward_features(tensor)

    validation = validate_features(features)
    assert validation["valid"], validation["message"]
```

---

### Phase 3: Documentation du Code

#### 3.1 Docstrings Complètes

**Standard à adopter:**
```python
def preprocess_image(
    image: np.ndarray,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Prétraite une image H&E pour inférence H-optimus-0.

    Cette fonction est la SOURCE UNIQUE DE VÉRITÉ pour le preprocessing.
    TOUTES les opérations d'inférence et d'entraînement DOIVENT l'utiliser.

    Étapes:
        1. Validation de l'image (RGB, shape correcte)
        2. Conversion uint8 (évite bug ToPILImage sur float64)
        3. Transform torchvision canonique:
           - ToPILImage() [après conversion uint8!]
           - Resize(224, 224)
           - ToTensor()
           - Normalize(HOPTIMUS_MEAN, HOPTIMUS_STD)
        4. Ajout batch dimension
        5. Transfert vers device

    Args:
        image: Image RGB (H, W, 3), uint8 [0-255] ou float [0-1] ou [0-255]
        device: Device PyTorch ("cuda", "cpu")

    Returns:
        Tensor (1, 3, 224, 224) normalisé, prêt pour H-optimus-0

    Raises:
        ValueError: Si image n'est pas RGB ou dimensions invalides

    Example:
        >>> image = cv2.imread("breast.png")
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> tensor = preprocess_image(image)
        >>> features = backbone.forward_features(tensor)

    Notes:
        - La conversion uint8 est CRITIQUE pour éviter le bug ToPILImage
        - CLS token std doit être entre 0.70-0.90 après forward_features()
        - Cette fonction est testée dans tests/unit/test_preprocessing.py

    See Also:
        - validate_features(): Valide les features extraites
        - create_hoptimus_transform(): Transform sous-jacent
    """
    ...
```

#### 3.2 Type Hints Partout

```python
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

def predict(
    self,
    image: np.ndarray
) -> Dict[str, Union[np.ndarray, List[Dict], float]]:
    """Type hints clairs pour toutes les fonctions."""
    ...
```

---

## 📋 Plan d'Exécution (Ordre Recommandé)

### Semaine 1: Modules Centralisés (CRITIQUE)

| Jour | Tâche | Effort | Priorité |
|------|-------|--------|----------|
| 1 | Créer `src/preprocessing/__init__.py` | 2h | 🔴 CRITIQUE |
| 1 | Créer `src/models/loader.py` | 2h | 🔴 CRITIQUE |
| 2 | Mettre à jour `src/inference/*.py` (6 fichiers) | 4h | 🔴 CRITIQUE |
| 3 | Mettre à jour `scripts/preprocessing/*.py` (3 fichiers) | 3h | 🔴 CRITIQUE |
| 4 | Mettre à jour `scripts/demo/gradio_demo.py` | 2h | 🔴 CRITIQUE |
| 5 | Tests de non-régression | 3h | 🔴 CRITIQUE |

### Semaine 2: Tests & Validation

| Jour | Tâche | Effort | Priorité |
|------|-------|--------|----------|
| 1 | Créer `tests/unit/test_preprocessing.py` | 2h | 🟠 HAUTE |
| 2 | Créer `tests/unit/test_model_loading.py` | 2h | 🟠 HAUTE |
| 3 | Créer `tests/integration/test_pipeline_e2e.py` | 3h | 🟠 HAUTE |
| 4 | Consolider scripts de validation | 3h | 🟡 MOYENNE |
| 5 | Documentation + Review | 2h | 🟡 MOYENNE |

---

## ✅ Critères de Succès

### Objectifs Mesurables

1. **Duplication Code**
   - Avant: ~15 copies de `create_hoptimus_transform()`
   - Après: 1 seule source de vérité dans `src/preprocessing/`

2. **Cohérence Preprocessing**
   - Avant: Bugs LayerNorm, ToPILImage, etc.
   - Après: 100% des fichiers utilisent le même module

3. **Couverture Tests**
   - Avant: Scripts de validation ad-hoc
   - Après: Suite de tests pytest structurée

4. **Maintenabilité**
   - Avant: Modification = 15+ fichiers à toucher
   - Après: Modification = 1 module central

---

## 🚨 Risques & Mitigations

### Risque 1: Casser le Code Existant
**Probabilité:** HAUTE
**Impact:** HAUTE
**Mitigation:**
- Tests de non-régression AVANT refactoring
- Refactoring incrémental (un module à la fois)
- Valider sur images de test après chaque changement

### Risque 2: Oublier des Fichiers
**Probabilité:** MOYENNE
**Impact:** HAUTE
**Mitigation:**
- Grep systématique de tous les patterns
- Checklist de validation
- Review par une autre personne

### Risque 3: Temps Sous-Estimé
**Probabilité:** MOYENNE
**Impact:** MOYENNE
**Mitigation:**
- Buffer de 20% sur estimations
- Priorisation stricte (modules centraux d'abord)

---

## 📚 Références

- CLAUDE.md Section "⚠️ GUIDE CRITIQUE: Préparation des Données"
- Bugs historiques: ToPILImage (2025-12-20), LayerNorm (2025-12-21), Instance mismatch (2025-12-21)
- Best practices Python: PEP 8, Type hints (PEP 484), Docstrings (PEP 257)

---

**Prochaine Action Recommandée:**
Créer les modules `src/preprocessing/` et `src/models/loader.py` pour établir une fondation solide.
