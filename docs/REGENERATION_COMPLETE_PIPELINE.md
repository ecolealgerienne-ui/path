# Pipeline Complet de Régénération des Données

**Date:** 2025-12-26
**Situation:** Toutes les données ont été nettoyées, régénération complète nécessaire

---

## 📋 Vue d'ensemble du Pipeline

```
1. PanNuke Raw Data (Téléchargement)
         ↓
2. Family Data FIXED (prepare_family_data_FIXED_v12_COHERENT.py)
         ↓
3. V13-Hybrid Dataset (prepare_v13_hybrid_dataset.py)
         ↓
4. H-Features Extraction (extract_h_features_v13.py)
         ↓
5. Training (train_hovernet_family_v13_hybrid.py)
```

---

## Phase 1: Téléchargement PanNuke (~10 minutes)

### Option A: Téléchargement Manuel

**Emplacement officiel:** https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke

**Structure attendue:**
```bash
/home/amar/data/PanNuke/
├── fold0/
│   ├── images.npy       # (2656, 256, 256, 3) uint8
│   ├── masks.npy        # (2656, 256, 256, 6) int32
│   └── types.npy        # (2656,) string
├── fold1/
│   ├── images.npy
│   ├── masks.npy
│   └── types.npy
└── fold2/
    ├── images.npy
    ├── masks.npy
    └── types.npy
```

**Commandes:**
```bash
# Créer répertoire
mkdir -p /home/amar/data/PanNuke

# Télécharger depuis Warwick (lien à obtenir sur le site)
# Exemple:
cd /home/amar/data/PanNuke
wget https://warwick.ac.uk/.../pannuke_fold0.zip
wget https://warwick.ac.uk/.../pannuke_fold1.zip
wget https://warwick.ac.uk/.../pannuke_fold2.zip

# Décompresser
unzip pannuke_fold0.zip
unzip pannuke_fold1.zip
unzip pannuke_fold2.zip
```

### Option B: Script de Téléchargement (si disponible)

```bash
python scripts/setup/download_datasets.py --dataset pannuke --output_dir /home/amar/data
```

**⚠️ Note:** Vérifier si ce script existe et fonctionne.

### Validation Phase 1

```bash
# Vérifier structure
ls -lh /home/amar/data/PanNuke/fold0/

# Vérifier taille des fichiers
# images.npy: ~515 MB
# masks.npy: ~393 MB
# types.npy: ~21 KB
```

---

## Phase 2: Génération Family Data FIXED (~30 minutes, 5 familles)

**Script:** `prepare_family_data_FIXED_v12_COHERENT.py`

**Pourquoi v12 ?**
- ✅ Fix cohérence NP/NT (0% conflit garanti)
- ✅ HV targets float32 [-1, 1] (Bug #3 résolu)
- ✅ Vraies instances PanNuke (pas connectedComponents)

**Commandes:**
```bash
# Activer environnement
conda activate cellvit

# Générer pour chaque famille
for family in glandular digestive urologic epidermal respiratory; do
    echo "========================================="
    echo "Generating FIXED data for: $family"
    echo "========================================="

    python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py \
        --family $family \
        --data_dir /home/amar/data/PanNuke
done
```

**Sortie attendue par famille:**
```
data/family_FIXED/
├── glandular_data_FIXED.npz      # ~1.5 GB (3535 samples)
├── digestive_data_FIXED.npz      # ~1.0 GB (2274 samples)
├── urologic_data_FIXED.npz       # ~500 MB (1153 samples)
├── epidermal_data_FIXED.npz      # ~250 MB (571 samples)
└── respiratory_data_FIXED.npz    # ~180 MB (408 samples)
```

**Temps estimé:**
- Glandular: ~8 min
- Digestive: ~5 min
- Urologic: ~3 min
- Epidermal: ~2 min
- Respiratory: ~1 min
**Total:** ~20-30 minutes

### Validation Phase 2

```bash
# Vérifier fichiers générés
ls -lh data/family_FIXED/

# Vérifier cohérence NP/NT (doit être 0%)
python scripts/validation/check_np_nt_conflict.py \
    --data_file data/family_FIXED/epidermal_data_FIXED.npz

# Attendu: "Conflit NP/NT: 0.00%"
```

---

## Phase 3: Génération V13-Hybrid Dataset (~10 minutes, 5 familles)

**Script:** `prepare_v13_hybrid_dataset.py` (avec Clean Split)

**Nouveauté:** 🔒 Clean Split (Grouped Split) pour prévenir data leakage

**Commandes:**
```bash
for family in glandular digestive urologic epidermal respiratory; do
    echo "========================================="
    echo "Preparing V13-Hybrid dataset for: $family"
    echo "========================================="

    python scripts/preprocessing/prepare_v13_hybrid_dataset.py \
        --family $family \
        --source_data_dir data/family_FIXED
done
```

**Sortie attendue:**
```
data/family_data_v13_hybrid/
├── glandular_data_v13_hybrid.npz      # ~1.5 GB
├── digestive_data_v13_hybrid.npz      # ~1.0 GB
├── urologic_data_v13_hybrid.npz       # ~500 MB
├── epidermal_data_v13_hybrid.npz      # ~250 MB
└── respiratory_data_v13_hybrid.npz    # ~180 MB
```

**Chaque .npz contient:**
- `images_224`: Images RGB 224×224
- `h_channels_224`: H-channels (Macenko + HED)
- `np_targets`, `hv_targets`, `nt_targets`: Targets
- `source_image_ids`, `fold_ids`, `crop_position_ids`: Métadonnées
- 🔒 **`split_types`**: 0=Train, 1=Val (Clean Split)

**Temps estimé:** ~2 min par famille = ~10 minutes total

### Validation Phase 3

```bash
# Vérifier Clean Split (CRITIQUE)
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/validation/verify_clean_split.py \
        --data_file data/family_data_v13_hybrid/${family}_data_v13_hybrid.npz
done

# Attendu pour chaque:
# ✅ ALL CHECKS PASSED - Clean Split is VALID!
# ✅ No source ID overlap
# ✅ All source IDs assigned
```

---

## Phase 4: Extraction H-Features (~5 minutes, 5 familles)

**Script:** `extract_h_features_v13.py`

**Commandes:**
```bash
for family in glandular digestive urologic epidermal respiratory; do
    echo "========================================="
    echo "Extracting H-features for: $family"
    echo "========================================="

    python scripts/preprocessing/extract_h_features_v13.py \
        --family $family \
        --hybrid_data_dir data/family_data_v13_hybrid
done
```

**Sortie attendue:**
```
data/cache/family_data/
├── glandular_h_features_v13.npz      # ~15 MB
├── digestive_h_features_v13.npz      # ~10 MB
├── urologic_h_features_v13.npz       # ~5 MB
├── epidermal_h_features_v13.npz      # ~2.5 MB
└── respiratory_h_features_v13.npz    # ~1.8 MB
```

**Temps estimé:** ~1 min par famille = ~5 minutes total

### Validation Phase 4

```bash
# Vérifier H-features shape
python -c "
import numpy as np
data = np.load('data/cache/family_data/epidermal_h_features_v13.npz')
print('H-features shape:', data['h_features'].shape)  # Attendu: (N, 256)
print('H-features dtype:', data['h_features'].dtype)  # Attendu: float32
"
```

---

## Phase 5: Extraction RGB Features (OPTIONNEL, si OrganHead nécessaire)

**Script:** `extract_features.py`

**Note:** Nécessaire seulement si vous voulez ré-entraîner OrganHead.

**Commandes:**
```bash
# Extraire features H-optimus-0 pour fold 0
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 0 \
    --batch_size 8 \
    --chunk_size 300

# Répéter pour fold 1 et 2 si nécessaire
```

**Sortie:**
```
data/cache/pannuke_features/
├── fold0_features.npz      # ~4.3 GB
├── fold1_features.npz      # ~4.0 GB
└── fold2_features.npz      # ~4.4 GB
```

**Temps estimé:** ~10 min par fold = ~30 minutes (3 folds)

---

## Phase 6: Entraînement V13-Hybrid (~40 minutes par famille)

**Script:** `train_hovernet_family_v13_hybrid.py`

**Commandes:**
```bash
# Entraîner une famille (exemple: epidermal)
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --epochs 30 \
    --batch_size 16 \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0 \
    --lambda_h_recon 0.1

# Entraîner toutes les familles (parallèle possible si multi-GPU)
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/training/train_hovernet_family_v13_hybrid.py \
        --family $family --epochs 30 --batch_size 16
done
```

**Sortie:**
```
models/checkpoints_v13_hybrid/
├── hovernet_epidermal_v13_hybrid_best.pth      # Best checkpoint
└── hovernet_epidermal_v13_hybrid_history.json  # Training history
```

**Temps estimé:** ~40 min par famille = ~3-4 heures (5 familles)

---

## Phase 7: Évaluation avec Clean Split (~5 minutes par famille)

**Script:** `test_v13_hybrid_aji.py`

**Commandes:**
```bash
# Évaluer une famille
python scripts/evaluation/test_v13_hybrid_aji.py \
    --checkpoint models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_best.pth \
    --family epidermal \
    --n_samples 50

# Évaluer toutes les familles
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/evaluation/test_v13_hybrid_aji.py \
        --checkpoint models/checkpoints_v13_hybrid/hovernet_${family}_v13_hybrid_best.pth \
        --family $family \
        --n_samples 100
done
```

**Métriques attendues (avec Clean Split):**

| Famille | Dice | AJI (cible) | Status |
|---------|------|-------------|--------|
| Glandular | ~0.93 | **≥0.68** | 🎯 Objectif |
| Digestive | ~0.93 | ≥0.60 | ✅ Bon |
| Urologic | ~0.90 | ≥0.55 | ⚠️ Acceptable |
| Epidermal | ~0.93 | ≥0.60 | ✅ Bon |
| Respiratory | ~0.91 | ≥0.60 | ✅ Bon |

**Note:** AJI peut baisser légèrement (2-5%) avec Clean Split vs ancien split (normal, élimine le leakage).

---

## 📊 Récapitulatif Temps Estimé

| Phase | Description | Temps |
|-------|-------------|-------|
| 1 | Téléchargement PanNuke | ~10 min |
| 2 | Family Data FIXED (5 familles) | ~30 min |
| 3 | V13-Hybrid Dataset (5 familles) | ~10 min |
| 4 | H-Features Extraction (5 familles) | ~5 min |
| 5 | RGB Features (optionnel) | ~30 min |
| 6 | Training V13-Hybrid (5 familles) | ~3-4h |
| 7 | Évaluation (5 familles) | ~25 min |
| **TOTAL** | **Sans training** | **~1h** |
| **TOTAL** | **Avec training** | **~4-5h** |

---

## ⚠️ Points de Vigilance

### 1. Espace Disque Nécessaire

| Données | Taille Estimée |
|---------|---------------|
| PanNuke Raw (3 folds) | ~3.5 GB |
| Family FIXED (5 familles) | ~3.5 GB |
| V13-Hybrid (5 familles) | ~3.5 GB |
| H-Features (5 familles) | ~35 MB |
| RGB Features (optionnel) | ~12.7 GB |
| Checkpoints (5 familles) | ~500 MB |
| **TOTAL (sans RGB)** | **~11 GB** |
| **TOTAL (avec RGB)** | **~24 GB** |

### 2. RAM Requise

- Génération Family FIXED: ~8-10 GB par famille
- Génération V13-Hybrid: ~5-6 GB par famille
- Extraction H-Features: ~2-3 GB
- Training: ~10-12 GB (GPU + CPU)

**Recommandation:** Au moins 16 GB RAM système

### 3. GPU VRAM

- Extraction RGB features: ~4-5 GB (RTX 4070 SUPER OK)
- Training V13-Hybrid: ~8-10 GB (batch_size=16)

**Recommandation:** Réduire batch_size à 8 si VRAM < 10 GB

---

## 🚀 Script de Régénération Automatique

**Créer un script bash complet:**

```bash
#!/bin/bash
# regenerate_full_pipeline.sh

set -e  # Exit on error

echo "========================================"
echo "CellViT-Optimus - Full Pipeline Regeneration"
echo "========================================"

# Activate environment
conda activate cellvit

# Phase 2: Family FIXED
echo "Phase 2: Generating Family FIXED data..."
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py \
        --family $family --data_dir /home/amar/data/PanNuke
done

# Phase 3: V13-Hybrid
echo "Phase 3: Preparing V13-Hybrid datasets..."
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/preprocessing/prepare_v13_hybrid_dataset.py \
        --family $family --source_data_dir data/family_FIXED
done

# Phase 3b: Verify Clean Split
echo "Phase 3b: Verifying Clean Split..."
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/validation/verify_clean_split.py \
        --data_file data/family_data_v13_hybrid/${family}_data_v13_hybrid.npz
done

# Phase 4: H-Features
echo "Phase 4: Extracting H-features..."
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/preprocessing/extract_h_features_v13.py \
        --family $family --hybrid_data_dir data/family_data_v13_hybrid
done

echo "========================================"
echo "✅ Pipeline regeneration complete!"
echo "========================================"
echo "Next: Run training with:"
echo "  python scripts/training/train_hovernet_family_v13_hybrid.py --family epidermal"
```

**Rendre exécutable:**
```bash
chmod +x regenerate_full_pipeline.sh
./regenerate_full_pipeline.sh
```

---

## ✅ Checklist de Validation

**Avant de commencer:**
- [ ] PanNuke raw data téléchargé et décompressé
- [ ] Environnement conda activé (`cellvit`)
- [ ] Au moins 11 GB d'espace disque disponible
- [ ] Au moins 16 GB RAM système

**Après chaque phase:**
- [ ] Phase 2: Tous les fichiers `*_data_FIXED.npz` créés
- [ ] Phase 2: Conflit NP/NT = 0% pour toutes les familles
- [ ] Phase 3: Tous les fichiers `*_data_v13_hybrid.npz` créés
- [ ] Phase 3: Clean Split validation passed (ALL CHECKS PASSED)
- [ ] Phase 4: Tous les fichiers `*_h_features_v13.npz` créés
- [ ] Phase 6: Training converge (Dice >0.90, HV MSE <0.05)
- [ ] Phase 7: AJI ≥0.60 avec Clean Split

---

**Document Version:** 1.0
**Date:** 2025-12-26
**Auteur:** Claude AI (CellViT-Optimus Development)
