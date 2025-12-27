# ✅ VALIDATION PHASE 1.1 — Hybrid Dataset Preparation

## Script Créé

**Fichier**: `scripts/preprocessing/prepare_v13_hybrid_dataset.py`

**Fonctionnalités implémentées**:
- ✅ Macenko normalization (implémentation intégrée, pas de dépendance externe)
- ✅ H-channel extraction via rgb2hed (skimage)
- ✅ Validation H-channel quality (std range)
- ✅ Validation HV targets dtype/range (prévention Bug #3)
- ✅ Validation finale avant sauvegarde

---

## 🔧 Commande d'Exécution

```bash
# Activer environnement cellvit
conda activate cellvit

# Lancer préparation Epidermal
python scripts/preprocessing/prepare_v13_hybrid_dataset.py --family epidermal

# Options disponibles:
#   --family: glandular|digestive|urologic|epidermal|respiratory
#   --v13_data_dir: Répertoire source V13 (défaut: data/family_data_v13_multi_crop)
#   --output_dir: Répertoire sortie (défaut: data/family_data_v13_hybrid)
#   --no_macenko: Désactiver Macenko normalization
```

---

## ✅ Critères de Validation

### 1. Exécution Sans Erreur
```
Attendu:
================================================================================
PREPARING V13-HYBRID DATASET: EPIDERMAL
================================================================================

📂 Loading V13 data: ...
  ✅ Loaded 2514 crops
  Images: (2514, 224, 224, 3), uint8
  NP targets: (2514, 224, 224), float32
  HV targets: (2514, 2, 224, 224), float32
  NT targets: (2514, 224, 224), int64

🔍 Validating HV targets...
  ✅ HV dtype: float32
  ✅ HV range: [-1.0000, 1.0000]

🎨 Initializing Macenko normalizer...
  ✅ Macenko normalizer fitted

🔬 Extracting H-channels...
Processing crops: 100%|████████████████| 2514/2514 [XX:XX<00:00, XXit/s]
  ✅ H-channels extracted: (2514, 224, 224), uint8

📊 Validating H-channel quality...
  H-channel std (normalized [0, 1]):
    Mean: 0.XXX
    Range: [0.XXX, 0.XXX]
    Valid samples (std ∈ [0.15, 0.35]): XXXX/2514 (XX.X%)
  ✅ H-channel quality OK (XX.X% valid)

💾 Saving to: data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz
  ✅ Saved: XX.XX MB

================================================================================
✅ V13-HYBRID DATASET PREPARATION COMPLETE: EPIDERMAL
================================================================================
```

### 2. Validation H-Channel Quality

| Critère | Valeur Attendue | Statut |
|---------|-----------------|--------|
| **H-channel std mean** | [0.15, 0.35] | ⏳ À vérifier |
| **H-channel std range** | Min > 0.10, Max < 0.50 | ⏳ À vérifier |
| **Valid samples %** | **> 80%** | ⏳ À vérifier |

**⚠️ ALERTE** si % valid < 80%: Vérifier que Macenko normalization fonctionne correctement.

### 3. Validation HV Targets (Prévention Bug #3)

| Critère | Valeur Attendue | Statut |
|---------|-----------------|--------|
| **HV dtype** | `float32` | ⏳ À vérifier |
| **HV range** | `[-1.0, 1.0]` | ⏳ À vérifier |

**❌ STOP** si dtype ≠ float32 ou range incorrect: Régénérer données V13 source.

### 4. Fichier de Sortie

**Vérification manuelle**:
```bash
# Vérifier fichier créé
ls -lh data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz

# Taille attendue: ~1.0-1.5 GB
# Si < 500 MB: Problème de sauvegarde

# Inspecter contenu
python -c "
import numpy as np
data = np.load('data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz')
print('Keys:', list(data.keys()))
print('Shapes:')
for key in data.keys():
    if hasattr(data[key], 'shape'):
        print(f'  {key}: {data[key].shape}, {data[key].dtype}')
"

# Output attendu:
# Keys: ['images_224', 'h_channels_224', 'np_targets', 'hv_targets', 'nt_targets',
#        'source_image_ids', 'crop_positions', 'macenko_applied',
#        'h_channel_std_mean', 'h_channel_std_range']
# Shapes:
#   images_224: (2514, 224, 224, 3), uint8
#   h_channels_224: (2514, 224, 224), uint8
#   np_targets: (2514, 224, 224), float32
#   hv_targets: (2514, 2, 224, 224), float32
#   nt_targets: (2514, 224, 224), int64
#   source_image_ids: (2514,), int32
#   crop_positions: (2514,), int32
```

---

## 🔍 Diagnostic en Cas d'Échec

### Problème 1: "HV dtype is not float32"

**Cause**: Données V13 source ont le Bug #3 (HV int8).

**Solution**:
```bash
# Vérifier source V13
python -c "
import numpy as np
data = np.load('data/family_data_v13_multi_crop/epidermal_data_v13_multi_crop.npz')
print(f'HV dtype: {data[\"hv_targets\"].dtype}')
print(f'HV range: [{data[\"hv_targets\"].min()}, {data[\"hv_targets\"].max()}]')
"

# Si dtype=int8 ou range=[-127, 127]:
# Régénérer données V13 avec script FIXED
python scripts/preprocessing/prepare_family_data_v13_multi_crop.py --family epidermal
```

### Problème 2: "H-channel std invalid (<80% valid)"

**Causes possibles**:
1. Images V13 source corrompues (Bug #1 ToPILImage)
2. Macenko normalization échoue (images trop sombres/claires)

**Diagnostic**:
```bash
# Visualiser quelques H-channels
python -c "
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz')
h_channels = data['h_channels_224']

# Afficher 9 échantillons
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(h_channels[i], cmap='gray')
    std = h_channels[i].std() / 255.0
    ax.set_title(f'Std: {std:.3f}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('results/h_channels_diagnostic.png')
print('Saved: results/h_channels_diagnostic.png')
"

# Si H-channels sont uniformes (gris plat): Problème Macenko
# Si H-channels ont du contraste: OK, juste ajuster seuils [0.15, 0.35]
```

### Problème 3: "Macenko fitting failed"

**Cause**: Image de référence (première crop) atypique.

**Solution**:
```python
# Modifier prepare_v13_hybrid_dataset.py ligne 268:
# Au lieu de ref_image = images_224[0]
# Utiliser une image médiane en termes de luminosité

ref_idx = np.argmin(np.abs(images_224.mean(axis=(1,2,3)) - images_224.mean()))
ref_image = images_224[ref_idx]
normalizer.fit(ref_image)
```

---

## ✅ Checklist de Validation

- [ ] Script s'exécute sans erreur
- [ ] HV dtype = float32 ✅
- [ ] HV range = [-1.0, 1.0] ✅
- [ ] H-channel std mean ∈ [0.15, 0.35]
- [ ] H-channel valid samples > 80%
- [ ] Fichier output existe et taille > 500 MB
- [ ] Toutes les clés présentes dans .npz
- [ ] Shapes correctes (vérification manuelle)

---

## 🎯 Prochaine Étape si Validation OK

**Phase 1.2**: Créer `extract_h_features_v13.py` pour extraire features CNN du canal H.

**Commande**:
```bash
python scripts/preprocessing/extract_h_features_v13.py \
    --data_file data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz
```

**Output attendu**: `data/cache/family_data/epidermal_h_features_v13.npz` (~2-3 MB)

---

**Date**: 2025-12-26
**Phase**: 1.1 - Hybrid Dataset Preparation
**Statut**: ⏳ En attente validation utilisateur
