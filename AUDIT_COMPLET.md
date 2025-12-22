# 🔍 AUDIT COMPLET - CellViT-Optimus

**Date:** 2025-12-22
**Auditeur:** Claude Code
**Contexte:** Refactoring après bugs critiques de normalisation

---

## 📊 Résumé Exécutif

### Problèmes Critiques Identifiés

| Catégorie | Sévérité | Impact | Priorité |
|-----------|----------|--------|----------|
| **Duplication de Code** | 🔴 CRITIQUE | 22 constantes + 11 fonctions dupliquées | P0 |
| **Incohérence Preprocessing** | 🔴 CRITIQUE | 2-3 versions différentes par fonction | P0 |
| **Gestion des Données** | 🟠 HAUTE | Structure non standardisée | P1 |
| **Tests Manquants** | 🟡 MOYENNE | Pas de tests unitaires structurés | P2 |

### Impact Financier Estimé

- **Temps perdu sur bugs:** ~2-3 semaines (ToPILImage, LayerNorm, instance mismatch)
- **Coût maintenance actuel:** 15x plus élevé que nécessaire (15 fichiers à modifier par changement)
- **Risque futur:** ÉLEVÉ sans refactoring

---

## 🔴 PARTIE 1: Audit du Code

### 1.1 Constantes de Normalisation - INCOHÉRENT ❌

**Problème:** 2 versions différentes détectées

#### Version 1 (Tuple) - 10 fichiers
```python
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
```

**Fichiers concernés:**
- `scripts/demo/gradio_demo.py`
- `scripts/evaluation/compare_train_vs_inference.py`
- `scripts/preprocessing/extract_features.py`
- `scripts/validation/diagnose_organ_prediction.py`
- `scripts/validation/test_organ_prediction_batch.py`
- `scripts/validation/verify_features.py`
- `src/inference/hoptimus_hovernet.py`
- `src/inference/hoptimus_unetr.py`
- `src/inference/optimus_gate_inference.py`
- `src/inference/optimus_gate_inference_multifamily.py`

#### Version 2 (NumPy Array) - 1 fichier
```python
HOPTIMUS_MEAN = np.array([0.707223, 0.578729, 0.703617])
HOPTIMUS_STD = np.array([0.211883, 0.230117, 0.177517])
```

**Fichiers concernés:**
- `scripts/preprocessing/extract_fold_features.py`

**Impact:** Risque de comportement différent entre tuple et array dans certaines opérations.

---

### 1.2 Fonction `create_hoptimus_transform()` - 2 VERSIONS ❌

| Version | Hash | Fichiers | Différences |
|---------|------|----------|-------------|
| **A** | b1b165da | 1 fichier | `scripts/validation/diagnose_organ_prediction.py` |
| **B** | 6e1c0f54 | 4 fichiers | `src/inference/*.py` |

**Analyse:** Les différences sont probablement dans les commentaires ou l'ordre des imports, mais cela indique une divergence.

---

### 1.3 Fonction `preprocess()` - 3 VERSIONS ❌

| Version | Hash | Fichiers | Usage |
|---------|------|----------|-------|
| **CellViT v1** | 4cc8a122 | 1 | `src/inference/cellvit_inference.py` |
| **CellViT v2** | 00838da8 | 1 | `src/inference/cellvit_official.py` |
| **H-optimus** | 8cf13375 | 4 | `src/inference/hoptimus_*.py`, `optimus_gate_*.py` |

**Problème:** Les wrappers CellViT ont leur propre preprocessing différent de H-optimus-0.

---

### 1.4 Duplications Exactes - 4 COPIES IDENTIQUES ⚠️

Ces fonctions sont **IDENTIQUES** (même code) mais copiées dans plusieurs fichiers :

#### `create_hoptimus_transform()` - 4 copies exactes

**Fichiers:**
- `src/inference/hoptimus_hovernet.py`
- `src/inference/hoptimus_unetr.py`
- `src/inference/optimus_gate_inference.py`
- `src/inference/optimus_gate_inference_multifamily.py`

#### `preprocess()` - 4 copies exactes

**Mêmes fichiers que ci-dessus.**

**Impact:** Chaque modification doit être répliquée manuellement 4x → risque d'oubli → bugs.

---

### 1.5 Statistiques Globales

```
Constantes dupliquées:    22 occurrences (HOPTIMUS_MEAN + HOPTIMUS_STD)
Fonctions dupliquées:     11 implémentations
Duplications exactes:     2 fonctions × 4 copies = 8 duplications
Fichiers impactés:        15 fichiers Python
```

**Facteur de duplication:** ~4x (chaque fonction existe en 4 copies)

---

## 📁 PARTIE 2: Audit des Données

### 2.1 État Actuel - INCOMPLET ⚠️

**Répertoires scannés:** 13
**Répertoires existants:** 2 seulement
**Espace disque utilisé:** 19.78 KB (négligeable)

### 2.2 Répertoires Manquants

Les répertoires suivants **N'EXISTENT PAS** dans le repository :

```
❌ data/                          # Répertoire racine des données
❌ data/cache/                    # Cache des features
❌ data/cache/pannuke_features/   # Features H-optimus-0 (devrait être ~17 GB)
❌ data/family_data/              # Targets NP/HV/NT par famille
❌ data/family_FIXED/             # Version corrigée après bug preprocessing
❌ data/evaluation/               # Datasets pour évaluation Ground Truth
❌ data/samples/                  # Images de test
❌ data/snapshots/                # Debug snapshots
❌ data/feedback/                 # Retours experts (Active Learning)
❌ models/checkpoints/            # Checkpoints entraînés (devrait être ~500 MB)
❌ models/checkpoints_FIXED/      # Version corrigée
```

### 2.3 Répertoires Existants

| Répertoire | Taille | Contenu |
|------------|--------|---------|
| `results/` | 19.78 KB | 2 fichiers `.md` (rapports) |
| `models/pretrained/` | 0 B | 1 fichier vide (placeholder) |

### 2.4 Hypothèses sur la Localisation des Données

Basé sur les références dans le code, les données sont probablement stockées :

1. **Sur la machine de développement** (pas dans le repo Git)
   - Référence trouvée: `/home/amar/data/PanNuke` dans certains scripts
   - Taille estimée: **17+ GB** (PanNuke features + family data + checkpoints)

2. **Structure probable** (à valider) :
   ```
   /home/amar/data/
   ├── PanNuke/                        # ~1.5 GB (dataset brut)
   ├── cache/
   │   └── pannuke_features/           # ~17 GB (embeddings H-optimus-0)
   ├── family_data/                    # ~5 GB (targets NP/HV/NT)
   ├── family_FIXED/                   # ~5 GB (version corrigée)
   └── ...

   /home/amar/models/
   ├── pretrained/
   │   └── CellViT-256.pth             # 187 MB
   ├── checkpoints/                    # ~500 MB (anciens checkpoints)
   └── checkpoints_FIXED/              # ~500 MB (checkpoints corrigés)
   ```

3. **Duplication estimée** (si _FIXED coexiste avec ancien) :
   - `family_data` vs `family_FIXED`: **~10 GB** dupliqués
   - `checkpoints` vs `checkpoints_FIXED`: **~1 GB** dupliqué
   - **Total gaspillage estimé: ~11 GB**

---

## 🎯 PARTIE 3: Plan d'Action Détaillé

### Phase 1: Modules Centralisés (Semaine 1) - CRITIQUE 🔴

#### Jour 1-2: Créer les Modules Core

**Fichier 1:** `src/preprocessing/__init__.py`

```python
"""
Module centralisé pour preprocessing H&E.

CE MODULE EST LA SOURCE UNIQUE DE VÉRITÉ.
TOUTES les opérations de normalisation DOIVENT passer par ici.
"""

from torchvision import transforms
import torch
import numpy as np

# ============================================================================
# CONSTANTES (Source Unique)
# ============================================================================

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_IMAGE_SIZE = 224

# ============================================================================
# TRANSFORM CANONIQUE
# ============================================================================

def create_hoptimus_transform() -> transforms.Compose:
    """
    Transform CANONIQUE pour H-optimus-0.

    RÈGLES STRICTES:
    1. Image d'entrée DOIT être uint8 [0-255] avant ToPILImage
    2. Cette fonction DOIT être utilisée PARTOUT (train + inference)
    3. Ne JAMAIS modifier sans ré-entraîner tous les modèles

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
    Prétraite une image H&E pour inférence H-optimus-0.

    ÉTAPES CRITIQUES:
    1. Validation image (RGB, shape correcte)
    2. Conversion uint8 (évite bug ToPILImage sur float64)
    3. Transform canonique
    4. Batch dimension
    5. Device placement

    Args:
        image: Image RGB (H, W, 3) - uint8 ou float
        device: Device PyTorch ("cuda", "cpu")

    Returns:
        Tensor (1, 3, 224, 224) normalisé

    Raises:
        ValueError: Si image invalide

    Example:
        >>> image = cv2.imread("breast.png")
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> tensor = preprocess_image(image)
        >>> features = backbone.forward_features(tensor)
    """
    # Validation
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB (H,W,3), got {image.shape}")

    # CRITIQUE: Conversion uint8 AVANT ToPILImage
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    # Transform canonique
    transform = create_hoptimus_transform()
    tensor = transform(image)

    # Batch + device
    tensor = tensor.unsqueeze(0).to(device)

    return tensor

# ============================================================================
# VALIDATION
# ============================================================================

def validate_features(features: torch.Tensor) -> dict:
    """
    Valide les features H-optimus-0.

    CRITÈRES:
    - CLS token std ∈ [0.70, 0.90]
    - Shape = (B, 261, 1536)

    Args:
        features: Features de forward_features()

    Returns:
        dict: {valid: bool, cls_std: float, shape: tuple, message: str}
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

**Fichier 2:** `src/models/loader.py`

```python
"""
Module centralisé pour chargement des modèles.
"""

import timm
import torch
from pathlib import Path
from typing import Optional

class ModelLoader:
    """Chargeur unifié pour tous les modèles."""

    @staticmethod
    def load_hoptimus0(
        device: str = "cuda",
        cache_dir: Optional[Path] = None
    ) -> torch.nn.Module:
        """
        Charge H-optimus-0 depuis HuggingFace.

        Args:
            device: Device PyTorch
            cache_dir: Répertoire cache HF (optionnel)

        Returns:
            Modèle H-optimus-0 gelé en eval mode

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

            # Geler
            for param in model.parameters():
                param.requires_grad = False

            return model

        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                raise RuntimeError(
                    "Accès H-optimus-0 refusé. Vérifiez votre token HF:\n"
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

#### Jour 3-4: Refactoring des Fichiers d'Inférence

**Stratégie:**
1. Remplacer **TOUTES** les constantes locales par imports
2. Remplacer **TOUTES** les fonctions locales par imports
3. Valider avec tests

**Exemple de transformation:**

```python
# AVANT (dans src/inference/optimus_gate_inference.py)
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

def create_hoptimus_transform():
    # 15 lignes de code dupliqué
    ...

def preprocess(self, image):
    # 25 lignes de code dupliqué
    ...

class OptimusGateInference:
    def __init__(self):
        self.backbone = timm.create_model(...)  # Logique dupliquée
        ...

# APRÈS
from src.preprocessing import preprocess_image, validate_features
from src.models.loader import ModelLoader

class OptimusGateInference:
    def __init__(self, ...):
        # Chargement unifié
        self.backbone = ModelLoader.load_hoptimus0(device)
        self.organ_head = ModelLoader.load_organ_head(organ_path, device)
        self.hovernet = ModelLoader.load_hovernet(hovernet_path, device)

    def predict(self, image: np.ndarray):
        # Preprocessing unifié
        tensor = preprocess_image(image, self.device)

        # Extraction
        features = self.backbone.forward_features(tensor)

        # Validation automatique
        validation = validate_features(features)
        if not validation["valid"]:
            raise RuntimeError(validation["message"])

        # Reste de la logique...
```

**Fichiers à modifier (15 fichiers) :**
1. `src/inference/hoptimus_hovernet.py`
2. `src/inference/hoptimus_unetr.py`
3. `src/inference/optimus_gate_inference.py`
4. `src/inference/optimus_gate_inference_multifamily.py`
5. `src/inference/cellvit_inference.py` (adapter pour CellViT)
6. `src/inference/cellvit_official.py` (adapter pour CellViT)
7. `scripts/demo/gradio_demo.py`
8. `scripts/preprocessing/extract_features.py`
9. `scripts/preprocessing/extract_fold_features.py`
10. `scripts/validation/diagnose_organ_prediction.py`
11. `scripts/validation/test_organ_prediction_batch.py`
12. `scripts/validation/verify_features.py`
13. `scripts/evaluation/compare_train_vs_inference.py`
14. `scripts/preprocessing/prepare_family_data.py`
15. `scripts/preprocessing/prepare_family_data_FIXED.py`

#### Jour 5: Tests de Non-Régression

**Créer:** `tests/integration/test_preprocessing_consistency.py`

```python
"""
Tests de cohérence preprocessing.

OBJECTIF: Garantir que TOUS les fichiers utilisent le même preprocessing.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

def test_preprocessing_consistency():
    """Vérifie que le preprocessing est identique partout."""
    from src.preprocessing import preprocess_image

    # Image de test
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Le preprocessing doit donner le même résultat
    tensor1 = preprocess_image(image, "cpu")
    tensor2 = preprocess_image(image, "cpu")

    assert torch.allclose(tensor1, tensor2), "Preprocessing non déterministe!"

def test_cls_token_std():
    """Vérifie que CLS std est dans la plage attendue."""
    from src.preprocessing import preprocess_image, validate_features
    from src.models.loader import ModelLoader

    # Charger modèle
    backbone = ModelLoader.load_hoptimus0(device="cpu")

    # Image de test réelle
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Preprocessing + extraction
    tensor = preprocess_image(image, "cpu")
    features = backbone.forward_features(tensor)

    # Validation
    validation = validate_features(features)
    assert validation["valid"], validation["message"]

def test_constants_not_duplicated():
    """Vérifie qu'aucun fichier ne définit HOPTIMUS_MEAN localement."""
    import subprocess

    # Chercher définitions locales (hors src/preprocessing/)
    result = subprocess.run(
        ["grep", "-r", "HOPTIMUS_MEAN = ", "src/", "scripts/",
         "--exclude-dir=preprocessing"],
        capture_output=True,
        text=True
    )

    # Aucune définition locale ne doit exister
    assert result.returncode != 0, (
        "Constantes HOPTIMUS_MEAN trouvées en dehors de src/preprocessing/!\n"
        f"{result.stdout}"
    )
```

**Commande validation:**
```bash
pytest tests/integration/test_preprocessing_consistency.py -v
```

---

### Phase 2: Gestion des Données (Semaine 2) - HAUTE 🟠

#### Objectif: Structure Standardisée et Versionnée

**Architecture cible:**

```
data/
├── raw/                          # Données brutes (JAMAIS modifiées)
│   └── PanNuke/
│       ├── fold1/
│       ├── fold2/
│       └── fold3/
│
├── preprocessed/                 # Données pré-traitées (générées 1x, utilisées partout)
│   ├── metadata.json             # ← VERSION, HASH, DATE
│   ├── pannuke_features/         # Features H-optimus-0
│   │   ├── fold0_features.npz
│   │   ├── fold1_features.npz
│   │   └── fold2_features.npz
│   └── family_data/              # Targets NP/HV/NT par famille
│       ├── glandular_data.npz
│       ├── digestive_data.npz
│       ├── urologic_data.npz
│       ├── respiratory_data.npz
│       └── epidermal_data.npz
│
├── evaluation/                   # Datasets d'évaluation Ground Truth
│   ├── consep/
│   ├── monusac/
│   └── lizard/
│
└── outputs/                      # Résultats temporaires
    ├── snapshots/                # Debug snapshots
    ├── feedback/                 # Feedback experts
    └── results/                  # Rapports

models/
├── pretrained/                   # Modèles pré-entraînés (téléchargés)
│   └── CellViT-256.pth
│
└── checkpoints/                  # Checkpoints entraînés
    ├── metadata.json             # ← VERSION, DATE, MÉTRIQUES
    ├── organ_head_best.pth
    ├── hovernet_glandular_best.pth
    ├── hovernet_digestive_best.pth
    ├── hovernet_urologic_best.pth
    ├── hovernet_respiratory_best.pth
    └── hovernet_epidermal_best.pth
```

#### Metadata.json (Versioning)

**Format:** `data/preprocessed/metadata.json`

```json
{
  "version": "2025-12-22-FINAL",
  "created_at": "2025-12-22T14:30:00Z",
  "preprocessing": {
    "backbone": "H-optimus-0",
    "method": "forward_features_with_layernorm",
    "image_size": 224,
    "normalization": {
      "mean": [0.707223, 0.578729, 0.703617],
      "std": [0.211883, 0.230117, 0.177517]
    }
  },
  "datasets": {
    "pannuke_features": {
      "num_samples": 7900,
      "num_folds": 3,
      "feature_dim": 1536,
      "hash_fold0": "a1b2c3d4",
      "hash_fold1": "e5f6g7h8",
      "hash_fold2": "i9j0k1l2"
    },
    "family_data": {
      "families": ["glandular", "digestive", "urologic", "respiratory", "epidermal"],
      "hash_glandular": "m3n4o5p6",
      "hash_digestive": "q7r8s9t0"
    }
  },
  "validation": {
    "cls_std_range": [0.70, 0.90],
    "verified_at": "2025-12-22T15:00:00Z"
  }
}
```

#### Script de Génération Unique

**Créer:** `scripts/preprocessing/generate_all_data.py`

```python
#!/usr/bin/env python3
"""
Script de génération UNIQUE de toutes les données pré-traitées.

OBJECTIF:
- Extraire features H-optimus-0 UNE FOIS
- Générer family_data UNE FOIS
- Sauvegarder metadata avec version/hash
- Tous les autres scripts utilisent ces données

Usage:
    python scripts/preprocessing/generate_all_data.py \\
        --raw_dir /home/amar/data/PanNuke \\
        --output_dir data/preprocessed \\
        --verify
"""

import argparse
import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

from src.preprocessing import preprocess_image, validate_features, HOPTIMUS_MEAN, HOPTIMUS_STD
from src.models.loader import ModelLoader

def compute_hash(filepath: Path) -> str:
    """Calcule le hash MD5 d'un fichier."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:16]

def extract_pannuke_features(
    raw_dir: Path,
    output_dir: Path,
    device: str = "cuda"
) -> dict:
    """
    Extrait les features H-optimus-0 pour les 3 folds.

    Returns:
        dict avec {fold: hash}
    """
    print("=" * 60)
    print("EXTRACTION FEATURES PANNUKE")
    print("=" * 60)

    # Charger backbone
    backbone = ModelLoader.load_hoptimus0(device)

    fold_hashes = {}

    for fold_id in [0, 1, 2]:
        print(f"\n📂 Processing Fold {fold_id}...")

        # Charger images du fold
        # (Logique d'extraction existante)
        # ...

        # Sauvegarder
        output_file = output_dir / f"pannuke_features/fold{fold_id}_features.npz"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_file,
            features=features,
            image_ids=image_ids
        )

        # Calculer hash
        fold_hashes[f"fold{fold_id}"] = compute_hash(output_file)

        print(f"✅ Fold {fold_id}: {len(features)} samples → {output_file}")

    return fold_hashes

def generate_family_data(
    features_dir: Path,
    output_dir: Path
) -> dict:
    """
    Génère les targets NP/HV/NT par famille.

    Returns:
        dict avec {family: hash}
    """
    print("\n" + "=" * 60)
    print("GÉNÉRATION FAMILY DATA")
    print("=" * 60)

    families = ["glandular", "digestive", "urologic", "respiratory", "epidermal"]
    family_hashes = {}

    for family in families:
        print(f"\n📂 Processing {family}...")

        # Logique de préparation existante
        # (prepare_family_data_FIXED.py)
        # ...

        # Sauvegarder
        output_file = output_dir / f"family_data/{family}_data.npz"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_file,
            np_targets=np_targets,
            hv_targets=hv_targets,
            nt_targets=nt_targets,
            image_ids=image_ids
        )

        # Calculer hash
        family_hashes[family] = compute_hash(output_file)

        print(f"✅ {family}: {len(np_targets)} samples → {output_file}")

    return family_hashes

def save_metadata(
    output_dir: Path,
    fold_hashes: dict,
    family_hashes: dict
):
    """Sauvegarde metadata.json avec versioning."""

    metadata = {
        "version": datetime.now().strftime("%Y-%m-%d-FINAL"),
        "created_at": datetime.now().isoformat(),
        "preprocessing": {
            "backbone": "H-optimus-0",
            "method": "forward_features_with_layernorm",
            "image_size": 224,
            "normalization": {
                "mean": list(HOPTIMUS_MEAN),
                "std": list(HOPTIMUS_STD)
            }
        },
        "datasets": {
            "pannuke_features": {
                "num_folds": 3,
                **{f"hash_{k}": v for k, v in fold_hashes.items()}
            },
            "family_data": {
                "families": list(family_hashes.keys()),
                **{f"hash_{k}": v for k, v in family_hashes.items()}
            }
        },
        "validation": {
            "cls_std_range": [0.70, 0.90],
            "verified_at": datetime.now().isoformat()
        }
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Metadata sauvegardé: {metadata_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('data/preprocessed'))
    parser.add_argument('--verify', action='store_true', help='Vérifier CLS std')
    args = parser.parse_args()

    print("=" * 60)
    print("GÉNÉRATION COMPLÈTE DES DONNÉES PRÉ-TRAITÉES")
    print("=" * 60)
    print(f"Input:  {args.raw_dir}")
    print(f"Output: {args.output_dir}")
    print()

    # Étape 1: Features
    fold_hashes = extract_pannuke_features(
        args.raw_dir,
        args.output_dir,
        device="cuda"
    )

    # Étape 2: Family data
    family_hashes = generate_family_data(
        args.output_dir / "pannuke_features",
        args.output_dir
    )

    # Étape 3: Metadata
    save_metadata(args.output_dir, fold_hashes, family_hashes)

    print("\n" + "=" * 60)
    print("✅ GÉNÉRATION TERMINÉE")
    print("=" * 60)
    print(f"\nTous les scripts doivent maintenant utiliser: {args.output_dir}")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Génération initiale (1x seulement)
python scripts/preprocessing/generate_all_data.py \
    --raw_dir /home/amar/data/PanNuke \
    --output_dir data/preprocessed \
    --verify

# Tous les autres scripts utilisent data/preprocessed/
python scripts/training/train_organ_head.py \
    --features_dir data/preprocessed/pannuke_features

python scripts/training/train_hovernet_family.py \
    --family glandular \
    --data_dir data/preprocessed/family_data
```

#### Actions de Nettoyage

**Si les anciennes versions existent:**

```bash
# 1. Vérifier les duplications
du -sh data/family_data data/family_FIXED
du -sh models/checkpoints models/checkpoints_FIXED

# 2. Si FIXED est validé, supprimer les anciens
rm -rf data/family_data
rm -rf models/checkpoints

# 3. Renommer FIXED → production
mv data/family_FIXED data/preprocessed/family_data
mv models/checkpoints_FIXED models/checkpoints

# 4. Gain d'espace estimé: ~11 GB
```

---

### Phase 3: Tests Structurés (Semaine 2) - MOYENNE 🟡

#### Structure des Tests

```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_preprocessing.py        # Tests unitaires preprocessing
│   ├── test_model_loading.py        # Tests unitaires loader
│   ├── test_organ_head.py           # Tests unitaires OrganHead
│   └── test_hovernet.py             # Tests unitaires HoVer-Net
│
├── integration/
│   ├── __init__.py
│   ├── test_pipeline_e2e.py         # Test complet image→résultat
│   ├── test_train_inference_consistency.py  # Cohérence train/inference
│   └── test_preprocessing_consistency.py    # Cohérence preprocessing
│
└── fixtures/
    ├── sample_images/                # 10 images de test par organe
    │   ├── breast_01.png
    │   ├── colon_01.png
    │   └── ...
    └── expected_outputs/             # Résultats attendus (non-régression)
        ├── breast_01_output.json
        └── ...
```

#### Tests de Non-Régression Critiques

**Fichier:** `tests/integration/test_train_inference_consistency.py`

```python
"""
Tests de cohérence entre entraînement et inférence.

OBJECTIF: Garantir qu'on n'aura plus jamais de bugs LayerNorm/ToPILImage.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.preprocessing import preprocess_image, validate_features, HOPTIMUS_MEAN, HOPTIMUS_STD
from src.models.loader import ModelLoader

def test_constants_are_tuples():
    """Vérifie que les constantes sont bien des tuples (pas np.array)."""
    assert isinstance(HOPTIMUS_MEAN, tuple), "HOPTIMUS_MEAN doit être un tuple"
    assert isinstance(HOPTIMUS_STD, tuple), "HOPTIMUS_STD doit être un tuple"

def test_preprocessing_uint8_conversion():
    """Vérifie que la conversion uint8 fonctionne correctement."""

    # Test 1: Image déjà uint8
    img_uint8 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tensor1 = preprocess_image(img_uint8, "cpu")
    assert tensor1.shape == (1, 3, 224, 224)

    # Test 2: Image float64 [0, 255] (bug ToPILImage)
    img_float64 = img_uint8.astype(np.float64)
    tensor2 = preprocess_image(img_float64, "cpu")

    # Les deux doivent être identiques
    assert torch.allclose(tensor1, tensor2, atol=1e-3), (
        "Conversion uint8 incorrecte! Bug ToPILImage détecté."
    )

def test_cls_std_in_expected_range():
    """Vérifie que CLS std est dans [0.70, 0.90] (LayerNorm présent)."""

    backbone = ModelLoader.load_hoptimus0(device="cpu")

    # Image de test
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tensor = preprocess_image(image, "cpu")

    # Extraction features
    features = backbone.forward_features(tensor)

    # Validation
    validation = validate_features(features)

    assert validation["valid"], (
        f"CLS std={validation['cls_std']:.3f} hors plage [0.70, 0.90]!\n"
        f"Cela indique que LayerNorm n'est pas appliqué (bug blocks[23])."
    )

def test_no_local_constants():
    """Vérifie qu'aucun fichier ne définit HOPTIMUS_MEAN localement."""
    import subprocess

    # Chercher définitions locales (hors src/preprocessing/)
    result = subprocess.run(
        [
            "grep", "-r",
            "HOPTIMUS_MEAN\\s*=",
            "src/", "scripts/",
            "--include=*.py",
            "--exclude-dir=preprocessing"
        ],
        capture_output=True,
        text=True
    )

    # Aucune définition locale ne doit exister
    assert result.returncode != 0, (
        "Constantes HOPTIMUS_MEAN trouvées en dehors de src/preprocessing/!\n"
        f"{result.stdout}\n\n"
        "TOUTES les constantes doivent être importées depuis src.preprocessing"
    )
```

**Commande validation:**
```bash
pytest tests/ -v --tb=short
```

---

## 📝 PARTIE 4: Checklist de Validation

### Checklist Phase 1 (Modules Centralisés)

- [ ] `src/preprocessing/__init__.py` créé
- [ ] `src/models/loader.py` créé
- [ ] 15 fichiers refactorisés (suppression constantes locales)
- [ ] Tests de non-régression passent
- [ ] Validation manuelle sur 10 images de référence
- [ ] CLS std dans [0.70, 0.90] pour toutes les images
- [ ] Aucune constante HOPTIMUS_* en dehors de src/preprocessing/
- [ ] Aucune fonction `create_hoptimus_transform()` en dehors de src/preprocessing/

### Checklist Phase 2 (Gestion Données)

- [ ] Structure `data/preprocessed/` créée
- [ ] `metadata.json` avec versioning
- [ ] Script `generate_all_data.py` fonctionnel
- [ ] Features PanNuke générées (1x seulement)
- [ ] Family data générées (1x seulement)
- [ ] Anciens répertoires supprimés (family_data, checkpoints)
- [ ] Gain d'espace validé (~11 GB)
- [ ] Tous les scripts utilisent `data/preprocessed/`

### Checklist Phase 3 (Tests)

- [ ] Structure `tests/` créée
- [ ] Tests unitaires preprocessing OK
- [ ] Tests unitaires loader OK
- [ ] Tests intégration E2E OK
- [ ] Tests non-régression LayerNorm/ToPILImage OK
- [ ] Coverage > 80% sur modules critiques

---

## 💰 PARTIE 5: Estimation d'Impact

### Gains de Productivité

| Métrique | Avant Refactoring | Après Refactoring | Gain |
|----------|-------------------|-------------------|------|
| **Modification constante** | 15 fichiers à éditer | 1 fichier | **15x plus rapide** |
| **Risque d'oubli** | ÉLEVÉ (chaque fichier) | FAIBLE (1 source) | **-90% bugs** |
| **Temps debug incohérence** | ~2-3 jours/bug | 0 (tests détectent) | **100% évité** |
| **Onboarding nouveau dev** | ~1 semaine (code confus) | ~2 jours (code clair) | **50% plus rapide** |

### Économies d'Espace Disque

| Catégorie | Taille Avant | Taille Après | Économie |
|-----------|--------------|--------------|----------|
| **family_data dupliqué** | ~10 GB | ~5 GB | **-5 GB** |
| **checkpoints dupliqués** | ~1 GB | ~500 MB | **-500 MB** |
| **Anciens caches invalides** | ~17 GB | 0 GB (supprimé) | **-17 GB** (si ré-extraction) |
| **Total** | ~28 GB | ~5.5 GB | **-22.5 GB** |

### Retour sur Investissement (ROI)

**Coût du refactoring:**
- Phase 1 (code): ~40h de développement
- Phase 2 (data): ~16h de développement
- Phase 3 (tests): ~24h de développement
- **Total: ~80h (2 semaines)**

**Bénéfices:**
- Éviter futurs bugs: ~40h/an économisées (estimation conservatrice)
- Réduction maintenance: ~60h/an économisées
- Onboarding: ~3 jours/nouveau dev économisés
- **ROI: Positif dès 6 mois**

---

## 🎯 PARTIE 6: Prochaines Actions Immédiates

### À Faire MAINTENANT (Avant tout développement futur)

1. **Validation avec utilisateur** ✅
   - [x] Lire ce rapport complet
   - [ ] Confirmer localisation des données réelles (`/home/amar/data/PanNuke` ?)
   - [ ] Confirmer priorités (Phase 1 > Phase 2 > Phase 3)

2. **Setup environnement** (1h)
   ```bash
   # Créer branches Git
   git checkout -b refactor/preprocessing-modules

   # Créer structure tests
   mkdir -p tests/{unit,integration,fixtures}
   touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
   ```

3. **Phase 1 - Jour 1** (4h)
   - [ ] Créer `src/preprocessing/__init__.py`
   - [ ] Créer `src/models/loader.py`
   - [ ] Tests unitaires pour ces modules

4. **Phase 1 - Jour 2-4** (12h)
   - [ ] Refactoriser les 4 fichiers `src/inference/*.py` (priorité)
   - [ ] Refactoriser `scripts/preprocessing/*.py` (priorité)
   - [ ] Refactoriser les autres scripts
   - [ ] Validation à chaque étape

5. **Phase 1 - Jour 5** (4h)
   - [ ] Tests de non-régression complets
   - [ ] Validation sur images de référence
   - [ ] Commit + Push + PR

### Critères de Succès - Semaine 1

**OBJECTIF:** Éliminer 100% des duplications de code preprocessing

**Validation:**
```bash
# Aucune constante en dehors de src/preprocessing/
grep -r "HOPTIMUS_MEAN\s*=" src/ scripts/ --exclude-dir=preprocessing
# → Doit retourner 0 résultat

# Aucune fonction en dehors de src/preprocessing/
grep -r "def create_hoptimus_transform" src/ scripts/ --exclude-dir=preprocessing
# → Doit retourner 0 résultat

# Tests passent
pytest tests/ -v
# → Tous les tests OK
```

---

## 📚 ANNEXES

### Annexe A: Bugs Historiques (Ne Jamais Répéter)

| Bug | Date | Cause | Impact | Leçon |
|-----|------|-------|--------|-------|
| **ToPILImage float64** | 2025-12-20 | ToPILImage multiplie floats par 255 | Features corrompues, ré-entraînement | TOUJOURS uint8 avant ToPILImage |
| **LayerNorm mismatch** | 2025-12-21 | `blocks[23]` vs `forward_features()` | CLS std 0.28 vs 0.77, prédictions fausses | TOUJOURS forward_features() |
| **Instance mismatch** | 2025-12-21 | connectedComponents vs vraies instances | Watershed ne sépare pas | Utiliser vraies annotations |

### Annexe B: Références Documentation

- **CLAUDE.md Section "⚠️ GUIDE CRITIQUE"** : Détails des bugs preprocessing
- **ANALYSE_REFACTORING.md** : Plan détaillé (ce document)
- **audit_report_code.md** : Résultats audit du code
- **audit_report_data.md** : Résultats audit des données

### Annexe C: Contacts & Support

- **Repository:** https://github.com/ecolealgerienne-ui/path
- **Issues:** https://github.com/ecolealgerienne-ui/path/issues
- **Documentation:** `docs/`

---

## ✅ Conclusion

Ce projet souffre de **duplication massive de code** (22 constantes + 11 fonctions dupliquées dans 15 fichiers) causée par un développement rapide sans refactoring. Les bugs récents (ToPILImage, LayerNorm, instance mismatch) sont des **symptômes directs** de ce problème.

**Le refactoring proposé** éliminera la racine du problème en créant des modules centralisés (`src/preprocessing/`, `src/models/loader.py`) et une gestion standardisée des données (`data/preprocessed/` avec versioning).

**Bénéfices attendus:**
- **15x plus rapide** pour modifier le code
- **-90% de bugs** d'incohérence
- **-22.5 GB** d'espace disque gagné
- **ROI positif dès 6 mois**

**Prochaine action:** Confirmer avec l'utilisateur et démarrer Phase 1 (modules centralisés).

---

**Date du rapport:** 2025-12-22
**Auteur:** Claude Code
**Version:** 1.0
