# V13 Smart Crops Pipeline - Prochaines Étapes

## Situation Actuelle

✅ **Complété hier (2025-12-26):**
- Validation HV rotation (100% negative divergence)
- Scripts créés: `prepare_v13_smart_crops.py`, `extract_features_v13_smart_crops.py`, `train_hovernet_family_v13_smart_crops.py`, `test_v13_smart_crops_aji.py`
- Script d'orchestration: `run_v13_smart_crops_pipeline.sh`
- Script de validation: `validate_v13_smart_crops_data.py`

⚠️ **Problème identifié aujourd'hui:**
- Répertoire `data/` n'existe pas → Aucune donnée V13 générée
- L'extraction features lancée hier a réussi MAIS le fichier n'est pas dans le projet actuel
- Cause: Pipeline jamais exécuté depuis le début (étape 1 manquante)

## Commandes à Exécuter (Terminal avec conda activé)

### Option A: Pipeline Complet Automatisé (RECOMMANDÉ)

```bash
# 1. Activer environment conda
conda activate cellvit

# 2. Vérifier que vous êtes dans le bon répertoire
cd ~/cellvit-optimus  # Ou le chemin de votre projet

# 3. Lancer le pipeline complet (~1h)
bash scripts/run_v13_smart_crops_pipeline.sh epidermal

# Le script va:
# - Étape 1: Générer données (5 crops + rotations) (~5 min)
# - Étape 2: Valider HV rotation (~2 min)
# - Étape 3: Extraire features train (~1 min)
# - Étape 4: Extraire features val (~1 min)
# - Étape 5: Entraîner HoVer-Net (~40 min)
# - Étape 6: Évaluer AJI (~5 min)
```

### Option B: Pipeline Étape par Étape (DEBUG)

Si le pipeline automatisé échoue, exécuter manuellement:

```bash
# 1. Activer environment
conda activate cellvit
cd ~/cellvit-optimus

# 2. Générer données V13 Smart Crops
python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal

# 3. Valider HV rotation (optionnel, déjà validé hier)
python scripts/validation/validate_hv_rotation.py \
    --data_file data/family_data_v13_smart_crops/epidermal_train_v13_smart_crops.npz \
    --n_samples 5

# 4. Extraire features train
python scripts/preprocessing/extract_features_v13_smart_crops.py \
    --family epidermal --split train --batch_size 8

# 5. Extraire features val
python scripts/preprocessing/extract_features_v13_smart_crops.py \
    --family epidermal --split val --batch_size 8

# 6. Valider données avant training
python scripts/validation/validate_v13_smart_crops_data.py --family epidermal

# 7. Entraîner HoVer-Net
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal --epochs 30 --batch_size 16

# 8. Évaluer AJI
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_epidermal_best.pth \
    --family epidermal --n_samples 50
```

## Prérequis (À Vérifier)

```bash
# 1. Données source epidermal FIXED existent
ls -lh data/family_FIXED/epidermal_data_FIXED.npz

# Si manquant, générer d'abord:
python scripts/preprocessing/prepare_family_data_FIXED.py --family epidermal

# 2. HuggingFace authentifié (pour H-optimus-0)
huggingface-cli whoami

# Si non authentifié:
huggingface-cli login
# Coller token avec "Read access to public gated repos"

# 3. GPU disponible
nvidia-smi
```

## Résultats Attendus

| Métrique | V13 POC (baseline) | V13 Smart Crops (cible) | Amélioration |
|----------|-------------------|------------------------|--------------|
| Dice | 0.76 ± 0.14 | ≥ 0.78 | +3% |
| **AJI** | **0.57 ± 0.14** | **≥ 0.68** | **+18%** 🎯 |
| PQ | ~0.51 | ≥ 0.62 | +20% |
| Over-seg Ratio | 1.30× | ~0.95× | Optimal |

## Temps Estimé

- **Option A (automatisé):** ~55 minutes (GPU RTX 4070 SUPER)
  - Preparation: 5 min
  - Validation: 2 min
  - Features extraction: 2 min
  - Training: 40 min
  - Evaluation: 5 min

- **Option B (manuel):** ~60 minutes + temps debugging si problèmes

## Fichiers Créés par le Pipeline

```
data/
├── family_data_v13_smart_crops/
│   ├── epidermal_train_v13_smart_crops.npz  (~500 MB)
│   └── epidermal_val_v13_smart_crops.npz    (~125 MB)
├── cache/
│   └── family_data/
│       ├── epidermal_rgb_features_v13_smart_crops_train.npz  (~3 GB)
│       └── epidermal_rgb_features_v13_smart_crops_val.npz    (~750 MB)

models/
└── checkpoints_v13_smart_crops/
    ├── hovernet_epidermal_best.pth  (~100 MB)
    └── hovernet_epidermal_history.json

results/
└── v13_smart_crops/
    └── epidermal_aji_evaluation_*.json
```

## En Cas d'Erreur

### Erreur 1: "data/family_FIXED/epidermal_data_FIXED.npz not found"
```bash
python scripts/preprocessing/prepare_family_data_FIXED.py --family epidermal
```

### Erreur 2: "401 Unauthorized - HuggingFace"
```bash
huggingface-cli login
# Entrer token avec "Read access to public gated repos"
```

### Erreur 3: "CUDA out of memory"
```bash
# Réduire batch size
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal --epochs 30 --batch_size 8  # Au lieu de 16
```

### Erreur 4: "CLS std hors range"
```bash
# Vérifier preprocessing est correct
python scripts/validation/verify_features.py \
    --features_dir data/cache/family_data
```

## Résumé

🎯 **Objectif:** AJI ≥ 0.68 (+18% vs V13 POC baseline 0.57)

📋 **Prochaine Action Immédiate:**
```bash
conda activate cellvit
cd ~/cellvit-optimus
bash scripts/run_v13_smart_crops_pipeline.sh epidermal
```

⏱️ **Temps estimé:** ~1h

✅ **Critère de succès:** AJI ≥ 0.68 sur 50 échantillons de validation
