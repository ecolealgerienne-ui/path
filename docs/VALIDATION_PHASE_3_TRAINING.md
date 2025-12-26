# ✅ VALIDATION PHASE 3 — Training V13-Hybrid

## Fichier Créé

**Script**: `scripts/training/train_hovernet_family_v13_hybrid.py` (~550 lignes)

**Composants implémentés**:
- ✅ HybridDataset class (charge RGB + H features + targets)
- ✅ HybridLoss avec Focal Loss (NP), SmoothL1 (HV masqué), CrossEntropy (NT)
- ✅ Optimizer AdamW avec LR séparés (RGB: 1e-4, H: 5e-5)
- ✅ CosineAnnealingLR scheduler
- ✅ Training & Validation loops
- ✅ Checkpoint saving (best Dice)
- ✅ History logging (JSON)

---

## 🔧 Commande d'Entraînement

```bash
# Activer environnement cellvit
conda activate cellvit

# Lancer training (Epidermal - 30 epochs)
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --epochs 30 \
    --batch_size 16 \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0 \
    --lambda_h_recon 0.1

# Si succès, sortie attendue:
# TRAINING COMPLETE
# Best Dice: 0.XXX (Epoch XX)
```

---

## ✅ Critères de Validation

### Test 1: Dataset Loading
```
ATTENDU:
- Train samples: ~2011 (80% de 2514)
- Val samples: ~503 (20% de 2514)
- RGB features shape: (256, 1536)
- H features shape: (256,)
- Targets validés (HV float32 [-1, 1])
```

### Test 2: Training Convergence
```
ATTENDU après 30 epochs:
- Train Loss: diminue régulièrement
- Val Loss: diminue sans overfitting (gap <50%)
- Val Dice: > 0.90 (cible V13-Hybrid)
- Val HV MSE: < 0.05 (meilleur que V13 POC)
- Val NT Acc: > 0.85
```

**⚠️ ALERTE** si:
- Val Loss > 2× Train Loss → Overfitting (réduire epochs ou augmenter dropout)
- Val Dice stagne < 0.80 → Problème architecture ou données
- HV MSE > 0.10 → Gradients HV faibles (augmenter lambda_hv)

### Test 3: Gradient Flow Équilibré
```
ATTENDU:
- LR RGB (1e-4) : bottleneck_rgb, decoder, heads
- LR H (5e-5) : bottleneck_h (plus faible car moins de données)
- Ratio gradients RGB/H : entre 1.5 et 3.0

Vérification:
- Observer convergence HV MSE
- Si HV MSE reste élevé → augmenter lr_h à 1e-4
```

### Test 4: Checkpoint Saving
```
ATTENDU:
✅ Fichier créé: models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_best.pth
✅ Taille: ~20-30 MB
✅ Contenu: model_state_dict, optimizer, epoch, best_dice, val_metrics

Vérification:
import torch
ckpt = torch.load("models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_best.pth")
print(f"Best Dice: {ckpt['best_dice']:.4f}")
print(f"Best Epoch: {ckpt['epoch']}")
print(f"Val Metrics: {ckpt['val_metrics']}")
```

### Test 5: History Logging
```
ATTENDU:
✅ Fichier créé: models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_history.json
✅ Contenu: train_loss, val_loss, val_dice, val_hv_mse, val_nt_acc (30 valeurs chacun)

Vérification:
import json
with open("models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_history.json") as f:
    history = json.load(f)

print(f"Final Dice: {history['val_dice'][-1]:.4f}")
print(f"Final HV MSE: {history['val_hv_mse'][-1]:.4f}")
```

---

## 📊 Sortie Attendue (Training Complet)

```
================================================================================
TRAINING V13-HYBRID: EPIDERMAL
================================================================================

Creating datasets...
Loading hybrid data from data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz...
Validating targets...
  ✅ HV dtype: float32 (valid)
  ✅ HV range: [-1.000, 1.000] (valid)
Loading H-channel features from data/cache/family_data/epidermal_h_features_v13.npz...
Loading RGB features from data/cache/pannuke_features/fold0_features.npz...
Extracting RGB features for train split (2011 samples)...
Dataset initialized: 2011 samples (train)
[...]
Dataset initialized: 503 samples (val)
Train: 2011 samples, Val: 503 samples

Initializing model...
Model parameters: XX,XXX,XXX

Starting training for 30 epochs...

Epoch 1 [Train]: 100%|████████| 126/126 [01:23<00:00]
  loss: 0.8234, np: 0.4532, hv: 0.3012, nt: 0.0690

Epoch 1 [Val]: 100%|████████| 32/32 [00:15<00:00]
  dice: 0.7234, hv_mse: 0.0823, nt_acc: 0.7456

Epoch 1/30
  Train Loss: 0.8234
  Val Loss:   0.7892
  Val Dice:   0.7234
  Val HV MSE: 0.0823
  Val NT Acc: 0.7456
  ✅ Saved best checkpoint: models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_best.pth

[... 29 epochs plus tard ...]

Epoch 30/30
  Train Loss: 0.3456
  Val Loss:   0.4123
  Val Dice:   0.9234
  Val HV MSE: 0.0234
  Val NT Acc: 0.8876
  ✅ Saved best checkpoint: models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_best.pth

================================================================================
TRAINING COMPLETE
================================================================================
Best Dice: 0.9234 (Epoch 28)
History saved: models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_history.json
```

---

## 🔍 Diagnostic en Cas d'Échec

### Problème 1: "FileNotFoundError: Hybrid data not found"

**Cause**: Phase 1.1 ou 1.2 pas complétée.

**Solution**:
```bash
# Vérifier présence fichiers
ls -lh data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz
ls -lh data/cache/family_data/epidermal_h_features_v13.npz
ls -lh data/cache/pannuke_features/fold0_features.npz

# Si absent, relancer Phases 1.1 et 1.2
python scripts/preprocessing/prepare_v13_hybrid_dataset.py --family epidermal
python scripts/preprocessing/extract_h_features_v13.py --family epidermal
```

### Problème 2: "CUDA out of memory"

**Cause**: Batch size trop élevé pour GPU.

**Solution**:
```bash
# Réduire batch size
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --batch_size 8  # Au lieu de 16

# Ou utiliser CPU (beaucoup plus lent)
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --device cpu
```

### Problème 3: "Val Dice stagne < 0.80"

**Cause**: Problème architecture ou données corrompues.

**Diagnostic**:
```bash
# Vérifier targets
python -c "
import numpy as np
data = np.load('data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz')
hv = data['hv_targets']
print(f'HV dtype: {hv.dtype}')  # Doit être float32
print(f'HV range: [{hv.min():.3f}, {hv.max():.3f}]')  # Doit être [-1, 1]
"

# Si HV dtype = int8 → BUG #3 détecté !
# Régénérer données Phase 1.1
```

**Solution si données OK**:
- Augmenter epochs (30 → 50)
- Augmenter lambda_hv (2.0 → 5.0)
- Réduire dropout (0.1 → 0.05)

### Problème 4: "HV MSE reste élevé (>0.10)"

**Cause**: Branche HV reçoit gradients trop faibles.

**Solution**:
```bash
# Augmenter lambda_hv
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --lambda_hv 5.0  # Au lieu de 2.0

# Ou égaliser LR (H = RGB)
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --lr_h 1e-4  # Au lieu de 5e-5
```

### Problème 5: "Overfitting (Val Loss >> Train Loss)"

**Cause**: Modèle mémorise les données.

**Solution**:
```bash
# Augmenter dropout
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --dropout 0.3  # Au lieu de 0.1

# Réduire epochs
python scripts/training/train_hovernet_family_v13_hybrid.py \
    --family epidermal \
    --epochs 20  # Au lieu de 30
```

---

## ✅ Checklist de Validation

- [ ] Script s'exécute sans erreur
- [ ] Train/Val datasets chargés (2011/503 samples)
- [ ] Training converge (Val Dice > 0.90)
- [ ] HV MSE < 0.05 (meilleur que V13 POC)
- [ ] NT Acc > 0.85
- [ ] Checkpoint sauvegardé (~20-30 MB)
- [ ] History sauvegardée (JSON avec 30 valeurs)
- [ ] Pas d'overfitting (Val Loss / Train Loss < 1.5)

---

## 🎯 Prochaine Étape si Validation OK

**Phase 4**: Créer `scripts/evaluation/test_v13_hybrid_aji.py` avec HV-guided watershed.

**Composants**:
1. Chargement checkpoint best
2. Inférence sur test samples
3. Post-processing HV-guided watershed
4. Calcul métriques AJI, PQ, Dice
5. Comparaison V13 POC vs V13-Hybrid

**Temps estimé**: 1-2h (dev + test)

**Objectif AJI**: ≥0.68 (+18% vs V13 POC baseline 0.57)

---

**Date**: 2025-12-26
**Phase**: 3 - Training V13-Hybrid
**Statut**: ✅ Script prêt — ⏳ En attente exécution

