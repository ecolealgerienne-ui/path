# ✅ VALIDATION PHASE 1.2 — H-Channel Features Extraction

## Script Créé

**Fichier**: `scripts/preprocessing/extract_h_features_v13.py`

**Architecture CNN implémentée**:
- ✅ 3 Conv layers (1→32→64→128)
- ✅ AdaptiveAvgPool2d (global pooling)
- ✅ FC layer (128→256)
- ✅ Total: ~148k params (négligeable vs 1.1B H-optimus-0)

---

## 🔧 Commande d'Exécution

```bash
# Activer environnement cellvit
conda activate cellvit

# Lancer extraction features H (Epidermal)
python scripts/preprocessing/extract_h_features_v13.py \
    --data_file data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz \
    --family epidermal \
    --batch_size 32 \
    --device cuda

# OU version courte (détecte automatiquement le fichier):
python scripts/preprocessing/extract_h_features_v13.py --family epidermal
```

---

## ✅ Critères de Validation

### 1. Exécution Sans Erreur
```
Attendu:
================================================================================
EXTRACTING H-CHANNEL FEATURES: EPIDERMAL
================================================================================

📂 Loading hybrid dataset: ...
  ✅ Loaded 2514 H-channels
  Shape: (2514, 224, 224), dtype: uint8

🔧 Initializing H-Channel CNN...
  ✅ CNN initialized: 148,256 parameters
  Architecture: 3 Conv + Pool + FC → 256-dim

🔬 Extracting features...
  Device: cuda
  Batch size: 32
Processing batches: 100%|████████████████| 79/79 [XX:XX<00:00, XXit/s]
  ✅ H-features extracted: (2514, 256), float32

📊 H-features statistics:
  Mean: X.XXXX
  Std: X.XXXX
  Range: [X.XXXX, X.XXXX]
  ✅ H-features std looks reasonable

💾 Saving to: data/cache/family_data/epidermal_h_features_v13.npz
  ✅ Saved: X.XX MB

================================================================================
✅ H-CHANNEL FEATURES EXTRACTION COMPLETE: EPIDERMAL
================================================================================
```

### 2. Validation Features Statistics

| Critère | Valeur Attendue | Statut |
|---------|-----------------|--------|
| **H-features shape** | `(2514, 256)` | ⏳ À vérifier |
| **H-features dtype** | `float32` | ⏳ À vérifier |
| **H-features std** | **[0.1, 2.0]** | ⏳ À vérifier |
| **CNN params** | ~148k | ⏳ À vérifier |

**⚠️ ALERTE** si std < 0.01: CNN output near-zero → Problème initialization
**⚠️ ALERTE** si std > 10.0: CNN output unstable → Check weights

### 3. Fichier de Sortie

**Vérification manuelle**:
```bash
# Vérifier fichier créé
ls -lh data/cache/family_data/epidermal_h_features_v13.npz

# Taille attendue: ~2-3 MB (beaucoup plus petit que RGB features)
# Calcul: 2514 crops × 256 features × 4 bytes (float32) ≈ 2.6 MB

# Inspecter contenu
python -c "
import numpy as np
data = np.load('data/cache/family_data/epidermal_h_features_v13.npz')
print('Keys:', list(data.keys()))
print()
for key in data.keys():
    if hasattr(data[key], 'shape'):
        print(f'{key}: {data[key].shape}, {data[key].dtype}')
    else:
        print(f'{key}: {data[key]}')
"

# Output attendu:
# Keys: ['h_features', 'cnn_params', 'feature_mean', 'feature_std']
#
# h_features: (2514, 256), float32
# cnn_params: 148256
# feature_mean: X.XXXX
# feature_std: X.XXXX
```

### 4. Test Rapide Gradient Flow (Optionnel)

```bash
# Créer test unitaire
python -c "
import torch
import torch.nn as nn
from scripts.preprocessing.extract_h_features_v13 import HChannelCNN

# Initialize CNN
model = HChannelCNN(output_dim=256)

# Dummy input with gradient
h_input = torch.randn(2, 1, 224, 224, requires_grad=True)

# Forward
features = model(h_input)  # (2, 256)

# Backward
loss = features.sum()
loss.backward()

# Check gradients
assert h_input.grad is not None, '❌ No gradient flow'
print(f'✅ Gradient flow OK: norm={h_input.grad.norm():.4f}')
print(f'✅ Features shape: {features.shape}')
print(f'✅ CNN params: {model.get_num_params():,}')
"
```

---

## 🔍 Diagnostic en Cas d'Échec

### Problème 1: "H-features std < 0.01 (near-zero output)"

**Cause**: CNN initialization ou BatchNorm en train mode au lieu de eval.

**Solution**:
```python
# Vérifier ligne 118 dans extract_h_features_v13.py:
model.eval()  # DOIT être en eval mode

# Vérifier initialization (lignes 52-62):
# Kaiming initialization pour Conv2d
# Constant initialization pour BatchNorm
```

### Problème 2: "CUDA out of memory"

**Cause**: Batch size trop élevé pour GPU.

**Solution**:
```bash
# Réduire batch size
python scripts/preprocessing/extract_h_features_v13.py \
    --family epidermal \
    --batch_size 16  # Au lieu de 32

# OU utiliser CPU (plus lent mais aucun risque OOM)
python scripts/preprocessing/extract_h_features_v13.py \
    --family epidermal \
    --device cpu
```

### Problème 3: "FileNotFoundError: hybrid dataset not found"

**Cause**: Phase 1.1 pas complétée.

**Solution**:
```bash
# Vérifier que Phase 1.1 a réussi
ls -lh data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz

# Si absent, relancer Phase 1.1:
python scripts/preprocessing/prepare_v13_hybrid_dataset.py --family epidermal
```

---

## ✅ Checklist de Validation

- [ ] Script s'exécute sans erreur
- [ ] H-features shape = (2514, 256) ✅
- [ ] H-features dtype = float32 ✅
- [ ] H-features std ∈ [0.1, 2.0]
- [ ] CNN params = ~148k
- [ ] Fichier output existe et taille ~2-3 MB
- [ ] Gradient flow test OK (optionnel)

---

## 🎯 Prochaine Étape si Validation OK

**Phase 2**: Créer `src/models/hovernet_decoder_hybrid.py` avec fusion additive RGB + H.

**Composants**:
1. `HoVerNetDecoderHybrid` class
2. Bottleneck RGB (1536 → 256)
3. Bottleneck H (256 → 256)
4. Fusion additive
5. Branches NP/HV/NT (identiques V13)

**Temps estimé**: 3-4h (dev + tests unitaires)

---

**Date**: 2025-12-26
**Phase**: 1.2 - H-Channel Features Extraction
**Statut**: ⏳ En attente validation utilisateur
