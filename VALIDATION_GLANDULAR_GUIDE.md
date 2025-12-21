# Guide de Validation - Famille Glandular

**Date**: 2025-12-21
**Objectif**: Valider les données FIXED avant ré-entraînement
**Durée estimée**: ~10 minutes (génération + validation)

---

## ⚠️ IMPORTANT: Exécution sur Machine Locale

Ces commandes doivent être exécutées **sur votre machine locale** (pas dans le sandbox), là où se trouve `/home/amar/data/PanNuke`.

---

## Étape 1: Générer Données Glandular FIXED (~5-6 min)

```bash
# Activer environnement conda
conda activate cellvit

# Créer répertoire logs si nécessaire
mkdir -p logs

# Lancer génération Glandular (3391 samples)
python scripts/preprocessing/prepare_family_data_FIXED.py \
    --data_dir /home/amar/data/PanNuke \
    --output_dir data/family_FIXED \
    --family glandular \
    --chunk_size 500 \
    2>&1 | tee logs/glandular_fixed_generation.log
```

### Sortie Attendue:

```
======================================================================
Préparation données famille: glandular
======================================================================
Organes: Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland
Chunk size: 500 images (RAM-optimized)

📋 Phase 1: Indexing...
  Fold 0: 1123 images
  Fold 1: 1146 images
  Fold 2: 1122 images

  Total samples: 3391

🔄 Phase 2: Processing in chunks of 500...

  Processing fold 0 (1123 images)...
    Chunk 1/3 (500 images)...
      Processing: 100%|████████████████| 500/500
    Chunk 2/3 (500 images)...
      Processing: 100%|████████████████| 500/500
    Chunk 3/3 (123 images)...
      Processing: 100%|████████████████| 123/123

  [Mêmes étapes pour fold 1 et fold 2...]

💾 Phase 3: Concatenating and saving...

  ✅ Saved: data/family_FIXED/glandular_data_FIXED.npz
     Size: 3.50 GB

  📊 Statistics:
     Images: (3391, 256, 256, 3)
     NP coverage: 23.45%
     HV range: [-1.000, 1.000]
     NT classes: [0 1 2 3 4]
```

### Vérifications:

- ✅ Pas d'erreur "Missing types.npy"
- ✅ Total samples = 3391 (pas 0)
- ✅ Fichier `data/family_FIXED/glandular_data_FIXED.npz` créé (~3.5 GB)
- ✅ Shapes correctes: (3391, 256, 256, 3) pour images
- ✅ HV range [-1, 1] (normalisé)

---

## Étape 2: Validation Complète (~2 min)

```bash
python scripts/evaluation/validate_fixed_data.py \
    --old_data data/family/glandular_targets.npz \
    --new_data data/family_FIXED/glandular_data_FIXED.npz \
    --family glandular \
    --sample_idx 0
```

### Sortie Attendue:

```
======================================================================
VALIDATION DONNÉES FIXED - glandular
======================================================================

Loading data...
  OLD data: data/family/glandular_targets.npz
  NEW data: data/family_FIXED/glandular_data_FIXED.npz

======================================================================
GLOBAL CHECKS
======================================================================

✓ CHECK 1: All expected keys present
    OLD keys: ['images', 'np_targets', 'hv_targets', 'nt_targets', 'fold_ids', 'image_ids']
    NEW keys: ['images', 'np_targets', 'hv_targets', 'nt_targets', 'fold_ids', 'image_ids']

✓ CHECK 2: Shapes correct
    Images:      (3391, 256, 256, 3) ✓
    NP targets:  (3391, 256, 256)    ✓
    HV targets:  (3391, 2, 256, 256) ✓
    NT targets:  (3391, 256, 256)    ✓

✓ CHECK 3: Dtypes correct
    Images:      float64 ✓
    NP targets:  float32 ✓
    HV targets:  float32 ✓
    NT targets:  int64   ✓

✓ CHECK 4: Ranges correct
    Images:      [0.000, 255.000]     ✓ (uint8 range)
    NP targets:  [0.000, 1.000]       ✓ (binary)
    HV targets:  [-1.000, 1.000]      ✓ (normalized)
    NT targets:  [0, 4]               ✓ (5 classes)

======================================================================
SAMPLE COMPARISON (idx=0)
======================================================================

NP Coverage:
  OLD: 21.34%
  NEW: 21.34%  ✓ (identical, as expected)

HV Gradient Magnitude:
  OLD: 0.342  (weak gradients)
  NEW: 0.487  ✓ (42% stronger! ← KEY IMPROVEMENT)
  Ratio: 1.42x

Instance Count Estimate:
  OLD: 12 instances
  NEW: 18 instances  ✓ (50% more! No fusion)

✅ Saved visualization: results/validation_fixed/glandular_validation_sample0.png

======================================================================
DIAGNOSTIC FINAL
======================================================================

✅ VALIDATION PASSED - All checks OK!

Key improvements detected:
  • HV gradients 42% stronger (better boundary definition)
  • 50% more instances detected (no connectedComponents fusion)
  • Data shapes and ranges correct

🎯 NEXT STEP: Train Glandular family
    Command: python scripts/training/train_hovernet_family.py \
                --family glandular \
                --data_dir data/family_FIXED \
                --epochs 50 \
                --augment
```

### Critères de Succès:

| Check | Critère | Attendu |
|-------|---------|---------|
| Shapes | (N, 256, 256, 3) | ✓ Correct |
| Ranges | NP [0,1], HV [-1,1] | ✓ Correct |
| HV gradient | Ratio NEW/OLD | **≥ 1.2x** (plus fort) |
| Instance count | NEW vs OLD | **≥ 1.0x** (pas de fusion) |

---

## Étape 3: Inspection Visuelle

Ouvrir l'image générée:

```bash
# Depuis votre machine
xdg-open results/validation_fixed/glandular_validation_sample0.png

# Ou si WSL
explorer.exe results/validation_fixed/glandular_validation_sample0.png
```

### Ce Que Vous Devez Voir:

**Rangée 1 (NEW - FIXED):**
- Image originale H&E
- Masque NP (noyaux en blanc)
- Carte HV (gradients colorés)
- Magnitude gradient (jaune = fort)

**Rangée 2 (OLD - BUGGY):**
- Mêmes visualisations pour comparaison

**Différences Attendues:**
- NEW: Gradients HV avec **pics jaunes nets** aux frontières entre cellules
- OLD: Gradients HV **lisses** (pas de pics, cellules fusionnées)
- NEW: **Plus de régions distinctes** dans la carte HV
- OLD: **Grandes régions homogènes** (fusion par connectedComponents)

---

## Étape 4: Si Validation OK → Entraînement (~2.5h)

```bash
python scripts/training/train_hovernet_family.py \
    --family glandular \
    --data_dir data/family_FIXED \
    --output_dir models/checkpoints_FIXED \
    --epochs 50 \
    --augment \
    --batch_size 32 \
    2>&1 | tee logs/train_glandular_fixed.log
```

### Résultats Attendus:

| Métrique | Avant (OLD) | Cible (FIXED) | Critère |
|----------|-------------|---------------|---------|
| NP Dice | 0.9645 | ≥ 0.96 | Maintenir |
| HV MSE | 0.0150 | **≤ 0.012** | Améliorer |
| NT Acc | 0.8800 | ≥ 0.88 | Maintenir |

**HV MSE est la métrique clé** - doit diminuer car gradients plus forts.

---

## Étape 5: Test sur Train & Val

```bash
# Créer script de test rapide
cat > scripts/validation/test_glandular_fixed.py << 'EOF'
#!/usr/bin/env python3
"""Test rapide du modèle Glandular FIXED."""

import torch
import numpy as np
from pathlib import Path
from src.models.hovernet_decoder import HoVerNetDecoder

def test_model(checkpoint_path, data_path):
    """Test sur quelques échantillons."""

    # Charger checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model = HoVerNetDecoder(embed_dim=1536, img_size=224, n_classes=5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().cuda()

    # Charger données
    data = np.load(data_path)
    features = data['patch_tokens'][:10]  # 10 premiers samples

    # Inférence
    with torch.no_grad():
        features_t = torch.from_numpy(features).cuda()
        np_pred, hv_pred, nt_pred = model(features_t)

    print(f"✓ Inférence OK sur 10 samples")
    print(f"  NP shape: {np_pred.shape}")
    print(f"  HV shape: {hv_pred.shape}")
    print(f"  NT shape: {nt_pred.shape}")

if __name__ == "__main__":
    test_model(
        "models/checkpoints_FIXED/hovernet_glandular_best.pth",
        "data/family_FIXED/glandular_data_FIXED.npz"
    )
EOF

chmod +x scripts/validation/test_glandular_fixed.py
python scripts/validation/test_glandular_fixed.py
```

---

## Étape 6: Décision GO/NO-GO

### ✅ GO - Si:
- Validation PASSED (all checks ✓)
- HV gradient ratio ≥ 1.2x
- Inspection visuelle: pics jaunes nets dans NEW
- Entraînement converge (HV MSE ≤ 0.012)

**→ Procéder avec les 4 autres familles**

### ❌ NO-GO - Si:
- Validation FAILED (any check ❌)
- HV gradient ratio < 1.0 (pas d'amélioration)
- Entraînement: HV MSE identique ou pire

**→ Investiguer davantage avant de continuer**

---

## Logs & Debugging

### Si erreur lors de la génération:

```bash
# Vérifier structure PanNuke
ls -la /home/amar/data/PanNuke/fold0/
# Attendu: images.npy, masks.npy, types.npy

# Vérifier contenu types.npy
python -c "
import numpy as np
types = np.load('/home/amar/data/PanNuke/fold0/types.npy')
print(f'Fold 0: {len(types)} images')
print(f'Organes: {np.unique(types)[:10]}')
"
```

### Si HV gradient ratio faible:

```bash
# Comparer instances PanNuke vs connectedComponents
python scripts/evaluation/compare_pannuke_instances.py \
    --pannuke_dir /home/amar/data/PanNuke \
    --fold 0 \
    --image_idx 2
```

---

## Fichiers Générés

| Fichier | Taille | Description |
|---------|--------|-------------|
| `data/family_FIXED/glandular_data_FIXED.npz` | ~3.5 GB | Données training FIXED |
| `results/validation_fixed/glandular_validation_sample0.png` | ~500 KB | Visualisation NEW vs OLD |
| `logs/glandular_fixed_generation.log` | ~100 KB | Log génération |
| `logs/train_glandular_fixed.log` | ~1 MB | Log entraînement |
| `models/checkpoints_FIXED/hovernet_glandular_best.pth` | ~50 MB | Checkpoint modèle |

---

## Timeline Estimée

| Étape | Durée | Cumulé |
|-------|-------|--------|
| 1. Génération données | ~6 min | 6 min |
| 2. Validation | ~2 min | 8 min |
| 3. Inspection visuelle | ~2 min | 10 min |
| **CHECKPOINT GO/NO-GO** | - | - |
| 4. Entraînement | ~2.5h | ~2h40 |
| 5. Test train/val | ~5 min | ~2h45 |

**Total Glandular**: ~2h45
**Si succès → 4 autres familles**: ~7h
**TOTAL PROJET**: ~10h

---

## Références

- Investigation complète: `results/INVESTIGATION_REPORT_FINAL.md`
- Script FIXED: `scripts/preprocessing/prepare_family_data_FIXED.py`
- Script validation: `scripts/evaluation/validate_fixed_data.py`
- Documentation: `CLAUDE.md` sections "BUG #3" et "Guide Critique"

---

**Créé le**: 2025-12-21
**Par**: Claude (Investigation Root Cause - connectedComponents fusion)
**Statut**: ✅ Prêt à exécuter sur machine locale
