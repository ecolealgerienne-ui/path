# Guide de Migration vers les Wrappers Standardisés

**Date:** 2025-12-22
**Objectif:** Migrer progressivement les scripts existants vers l'interface standardisée

---

## Pourquoi Migrer ?

Les wrappers standardisés offrent:
- ✅ **Robustesse** : Code qui ne casse pas lors de changements d'implémentation
- ✅ **Simplicité** : Interface unifiée pour tous les modèles
- ✅ **Sécurité** : Validation automatique des features

---

## Scripts Prioritaires (Non Migrés)

| Script | Ligne | Fragile | Priorité |
|--------|-------|---------|----------|
| `test_family_models_isolated.py` | 210-216 | ❌ Dict access | Haute |
| `test_organ_routing.py` | - | ✅ (OrganHead seulement) | Basse |

---

## Exemple de Migration: test_family_models_isolated.py

### Avant (Fragile)

```python
# Ligne 210-216 (VERSION ACTUELLE)
from src.models.loader import ModelLoader

hovernet = ModelLoader.load_hovernet(checkpoint_path, device)
backbone = ModelLoader.load_hoptimus0(device)

# Inférence
features = backbone.forward_features(tensor)
patch_tokens = features[:, 1:257, :]
np_out, hv_out, nt_out = hovernet(patch_tokens)  # ❌ Dépend de l'implémentation (tuple)

# Conversion manuelle
np_pred = torch.sigmoid(np_out).cpu().numpy()[0, 0]
hv_pred = hv_out.cpu().numpy()[0]
nt_pred = torch.softmax(nt_out, dim=1).cpu().numpy()[0]
```

**Problèmes:**
- Si HoVer-Net change de tuple → dict, le script casse
- Conversion manuelle répétée dans chaque script
- Pas de validation automatique des features

### Après (Robuste)

```python
# VERSION RECOMMANDÉE
from src.models import (
    create_hovernet_wrapper,
    create_backbone_wrapper,
)

hovernet = create_hovernet_wrapper(checkpoint_path, device)
backbone = create_backbone_wrapper(device)

# Inférence avec validation auto
features = backbone(tensor, validate=True)  # ✅ Valide CLS std automatiquement
patch_tokens = backbone.extract_patch_tokens(features)
output = hovernet(patch_tokens)  # ✅ Toujours HoVerNetOutput

# Conversion automatique avec activations
result = output.to_numpy(apply_activations=True)
np_pred = result["np"]  # (H, W) dans [0, 1]
hv_pred = result["hv"]  # (2, H, W) dans [-1, 1]
nt_pred = result["nt"]  # (n_classes, H, W) dans [0, 1]
```

**Avantages:**
- ✅ Fonctionne même si HoVer-Net change d'implémentation
- ✅ Validation automatique (détecte Bug #1 et Bug #2)
- ✅ Code plus lisible et maintenable

---

## Checklist de Migration

Pour chaque script à migrer:

- [ ] Remplacer `ModelLoader.load_hovernet()` par `create_hovernet_wrapper()`
- [ ] Remplacer `ModelLoader.load_organ_head()` par `create_organ_head_wrapper()`
- [ ] Remplacer `ModelLoader.load_hoptimus0()` par `create_backbone_wrapper()`
- [ ] Remplacer unpacking manuel (`np_out, hv_out, nt_out`) par `output.to_numpy()`
- [ ] Utiliser `backbone.extract_cls_token()` et `backbone.extract_patch_tokens()` au lieu d'indexation manuelle
- [ ] Activer validation automatique: `backbone(tensor, validate=True)`
- [ ] Tester que le script fonctionne toujours

---

## Exemple Complet: Script d'Évaluation

### Structure Complète avec Wrappers

```python
#!/usr/bin/env python3
"""
Exemple d'évaluation utilisant l'interface standardisée.
"""

import torch
from pathlib import Path
from src.models import (
    create_hovernet_wrapper,
    create_organ_head_wrapper,
    create_backbone_wrapper,
)
from src.preprocessing import preprocess_image

# Configuration
device = "cuda"
checkpoint_dir = Path("models/checkpoints")

# Charger tous les modèles (wrappers)
backbone = create_backbone_wrapper(device=device)
organ_head = create_organ_head_wrapper(
    checkpoint_path=checkpoint_dir / "organ_head_best.pth",
    temperature=0.5  # Temperature scaling pour calibration
)
hovernet_glandular = create_hovernet_wrapper(
    checkpoint_path=checkpoint_dir / "hovernet_glandular_best.pth",
    device=device
)

# Pipeline complet
def evaluate_image(image):
    """Évalue une image avec le pipeline complet."""

    # 1. Preprocessing (centralisé)
    tensor = preprocess_image(image, device=device)

    # 2. Extraction features avec validation auto
    features = backbone(tensor, validate=True)  # ✅ Valide CLS std

    # 3. Prédiction organe
    cls_token = backbone.extract_cls_token(features)
    organ_output = organ_head(cls_token)

    print(f"Organe: {organ_output.organ_name} ({organ_output.confidence:.1%})")

    # 4. Sélection du bon HoVer-Net (selon la famille)
    # (Simplifié ici, voir OptimusGate pour version complète)
    patch_tokens = backbone.extract_patch_tokens(features)
    hovernet_output = hovernet_glandular(patch_tokens)

    # 5. Conversion avec activations
    result = hovernet_output.to_numpy(apply_activations=True)

    return {
        "organ": organ_output.organ_name,
        "confidence": organ_output.confidence,
        "np_mask": result["np"],  # (H, W) dans [0, 1]
        "hv_maps": result["hv"],  # (2, H, W) dans [-1, 1]
        "nt_probs": result["nt"],  # (n_classes, H, W) dans [0, 1]
    }

# Test
if __name__ == "__main__":
    import numpy as np

    # Image de test
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Évaluation
    result = evaluate_image(image)

    print(f"Segmentation: {result['np_mask'].shape}")
    print(f"Gradients HV: {result['hv_maps'].shape}")
    print(f"Types cellules: {result['nt_probs'].shape}")
```

---

## Migration par Étapes

### Étape 1: Scripts d'Évaluation (Priorité Haute)

- `scripts/evaluation/test_family_models_isolated.py`
- `scripts/evaluation/test_organ_routing.py` (déjà partiellement conforme)

### Étape 2: Scripts d'Inférence (Priorité Moyenne)

- `src/inference/optimus_gate_inference.py`
- `src/inference/optimus_gate_inference_multifamily.py`

### Étape 3: Scripts de Validation (Priorité Basse)

- `scripts/validation/diagnose_organ_prediction.py`
- `scripts/validation/test_organ_prediction_batch.py`

---

## FAQ Migration

**Q: Dois-je tout migrer d'un coup ?**
R: Non, migration progressive. Commencer par les nouveaux scripts, puis migrer les anciens au besoin.

**Q: Les wrappers ont un coût en performance ?**
R: Négligeable (~0.1ms overhead). La validation peut être désactivée si critique: `backbone(tensor, validate=False)`.

**Q: Que faire si un modèle custom n'a pas de wrapper ?**
R: Créer un wrapper dans `src/models/model_interface.py` en suivant le pattern existant.

**Q: Les anciens scripts vont-ils casser ?**
R: Non, les wrappers sont additifs. Les anciens imports (`from src.models.loader`) fonctionnent toujours.

---

## Ressources

- **Code source:** `src/models/model_interface.py`
- **Documentation:** `CLAUDE.md` section "Interface Standardisée des Modèles"
- **Exports:** `src/models/__init__.py`

---

**Dernière mise à jour:** 2025-12-22
**Auteur:** Phase 1 Refactoring + Interface Standardisée
