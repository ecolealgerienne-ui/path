# Refactorisation Scripts - Module Centralisé Preprocessing

**Date:** 2025-12-23
**Objectif:** Éliminer les duplications de code et utiliser le module centralisé `src/data/preprocessing.py`

---

## Problème Identifié

Suite à la remarque de l'utilisateur : *"tu n'as pas modifier les scripts pour prendre en charge tes modifs"*

Les scripts faisaient des conversions manuelles HV int8 → float32 au lieu d'utiliser le module centralisé créé lors de la Phase 1 de factorisation.

---

## Scripts Modifiés ✅

### 1. `scripts/training/train_hovernet_family.py` (CRITIQUE - Entraînement)

**Avant :**
```python
# Chargement manuel
targets_data = np.load(targets_path)
self.np_targets = targets_data['np_targets']

hv_raw = targets_data['hv_targets']
if hv_raw.dtype == np.int8:
    self.hv_targets = hv_raw.astype(np.float32) / 127.0
    print(f"  ⚠️  HV format OLD détecté (int8) - conversion en float32")
else:
    self.hv_targets = hv_raw

self.nt_targets = targets_data['nt_targets']

# Resize manuel
np_target_t = torch.from_numpy(np_target)
hv_target_t = torch.from_numpy(hv_target)
nt_target_t = torch.from_numpy(nt_target)

np_target_t = F.interpolate(np_target_t.unsqueeze(0).unsqueeze(0),
                            size=(224, 224), mode='nearest').squeeze()
hv_target_t = F.interpolate(hv_target_t.unsqueeze(0),
                            size=(224, 224), mode='bilinear',
                            align_corners=False).squeeze(0)
nt_target_t = F.interpolate(nt_target_t.float().unsqueeze(0).unsqueeze(0),
                            size=(224, 224), mode='nearest').squeeze().long()
```

**Après :**
```python
# Import centralisé
from src.data.preprocessing import load_targets, resize_targets

# Chargement avec module centralisé
self.np_targets, self.hv_targets, self.nt_targets = load_targets(
    targets_path,
    validate=True,          # Valide automatiquement les targets
    auto_convert_hv=True    # Convertit int8 → float32 si nécessaire
)

# Resize avec module centralisé
np_target, hv_target, nt_target = resize_targets(
    np_target, hv_target, nt_target,
    target_size=224,
    mode="training"
)
```

**Lignes éliminées:** ~25 lignes de code dupliqué

---

### 2. `scripts/validation/compare_mse_vs_smoothl1.py`

**Avant :**
```python
data = np.load(data_file)
hv_targets = data['hv_targets']
np_targets = data['np_targets']

if hv_targets.dtype == np.int8:
    print(f"   ⚠️  Conversion int8 → float32")
    hv_targets = hv_targets.astype(np.float32) / 127.0
```

**Après :**
```python
from src.data.preprocessing import load_targets

np_targets, hv_targets, _ = load_targets(
    data_file,
    validate=True,
    auto_convert_hv=True
)
```

**Lignes éliminées:** ~8 lignes

---

### 3. Scripts NON modifiés (intentionnel)

| Script | Raison |
|--------|--------|
| `validate_fixed_data.py` | Compare OLD vs FIXED → conversion manuelle intentionnelle pour comparaison |
| `trace_pipeline.py` | Trace étape par étape → conversion explicite pour diagnostic |

---

## Avantages de la Refactorisation

### ✅ Single Source of Truth
- Chargement targets : 1 fonction au lieu de patterns éparpillés
- Resize : 1 implémentation au lieu de duplications
- Validation : automatique à chaque chargement

### ✅ Détection Automatique Bug #3
```python
def validate_targets(...):
    if hv_target.dtype == np.int8:
        raise ValueError(
            "HV dtype est int8 [-127, 127] au lieu de float32 [-1, 1] ! "
            "Cela cause MSE ~4681 au lieu de ~0.01. "
            "Re-generer targets avec prepare_family_data_FIXED.py"
        )
```

### ✅ Cohérence Garantie
- Entraînement et validation utilisent le MÊME preprocessing
- Impossible d'avoir des divergences train/eval
- Changements futurs propagés automatiquement

### ✅ Maintenabilité
- Modification de logique en 1 seul endroit
- Code plus lisible (imports au lieu de duplications)
- Moins de risques de bugs

---

## Prochaines Étapes (Commandes à Exécuter)

### Étape 1 : Générer Données FIXED pour Epidermal

**Commande :**
```bash
# Activer environnement conda
conda activate cellvit

# Générer données epidermal (571 samples, ~2 min)
python scripts/preprocessing/prepare_family_data_FIXED.py \
    --data_dir /home/amar/data/PanNuke \
    --family epidermal \
    --chunk_size 300 \
    --output_dir data/family_FIXED
```

**Sortie attendue :**
```
✅ Saved: data/family_FIXED/epidermal_data_FIXED.npz
   Size: X.XX GB

📊 Statistics:
   Images: (571, 256, 256, 3)
   NP coverage: XX.XX%
   HV range: [-1.000, 1.000]  ← FLOAT32 !
   NT classes: [0 1 2 3 4 5]
```

---

### Étape 2 : Vérifier Format FIXED

**Commande :**
```bash
python scripts/utils/inspect_npz.py data/family_FIXED/epidermal_data_FIXED.npz
```

**Sortie attendue :**
```
Keys in epidermal_data_FIXED.npz:
  - images: shape (571, 256, 256, 3), dtype uint8
  - np_targets: shape (571, 256, 256), dtype float32
  - hv_targets: shape (571, 2, 256, 256), dtype float32 ✅
  - nt_targets: shape (571, 256, 256), dtype int64
  - fold_ids: shape (571,), dtype int32
  - image_ids: shape (571,), dtype int32
```

---

### Étape 3 : Extraire Features H-optimus-0 pour Epidermal

**Commande :**
```bash
python scripts/preprocessing/extract_features_from_fixed.py \
    --family epidermal \
    --batch_size 8 \
    --output_dir data/cache/family_data_FIXED
```

**Sortie attendue :**
```
✅ Saved: data/cache/family_data_FIXED/epidermal_features.npz
   Shape: (571, 261, 1536)
   Size: ~1.2 GB
```

---

### Étape 4 : Ré-entraîner HoVer-Net sur Epidermal FIXED

**Commande :**
```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0
```

**Temps estimé :** ~30-45 minutes (571 samples)

**Métriques cibles :**
| Métrique | OLD (int8) | FIXED (float32) | Objectif |
|----------|------------|-----------------|----------|
| NP Dice | 0.9542 | ? | > 0.93 (stable) |
| HV MSE | 0.2733 | ? | < 0.30 (amélioration ou stable) |
| NT Acc | 0.8871 | ? | > 0.85 (stable) |

---

### Étape 5 : Valider Performances FIXED vs OLD

**Commande :**
```bash
python scripts/evaluation/validate_fixed_data.py \
    --family epidermal \
    --n_samples 10
```

**Comparaison attendue :**
```
OLD (int8 → float32 conversion):
  NP Dice:  X.XXXX
  HV MSE:   X.XXXX
  NT Acc:   X.XXXX

FIXED (native float32):
  NP Dice:  X.XXXX
  HV MSE:   X.XXXX  ← Devrait être similaire ou meilleur
  NT Acc:   X.XXXX
```

---

## Pourquoi Epidermal en Premier ?

| Raison | Détail |
|--------|--------|
| **Plus petite famille** | 571 samples (vs 3391 glandular) |
| **Temps minimal** | ~30 min entraînement (vs 2h glandular) |
| **Validation rapide** | Confirme pipeline avant re-training complet |
| **Risque faible** | Si échec, peu de temps perdu |

---

## Si Epidermal Réussit → Expansion

```bash
# Glandular (3391 samples, ~2h)
python scripts/training/train_hovernet_family.py --family glandular --epochs 50 --augment

# Digestive (2430 samples, ~1.5h)
python scripts/training/train_hovernet_family.py --family digestive --epochs 50 --augment

# Urologic (1101 samples, ~45 min)
python scripts/training/train_hovernet_family.py --family urologic --epochs 50 --augment

# Respiratory (408 samples, ~20 min)
python scripts/training/train_hovernet_family.py --family respiratory --epochs 50 --augment
```

**Temps total 5 familles :** ~5-6 heures

---

## Modifications de Code Commitées

**Fichiers modifiés :**
1. `scripts/training/train_hovernet_family.py` (+2 imports, -25 lignes duplication)
2. `scripts/validation/compare_mse_vs_smoothl1.py` (+1 import, -8 lignes duplication)

**Commit message suggéré :**
```
refactor: Use centralized preprocessing module in training/validation scripts

- train_hovernet_family.py: Replace manual HV conversion with load_targets()
- train_hovernet_family.py: Replace manual resize with resize_targets()
- compare_mse_vs_smoothl1.py: Use load_targets() for consistency
- Eliminates ~33 lines of duplicated code
- Enables automatic Bug #3 detection (int8 vs float32)
- Guarantees train/eval consistency

Refs: PLAN_DECISION_DONNEES.md, PROOF_HV_NORMALIZATION_BUG.md
```

---

## Tests de Régression Recommandés

Après modifications, vérifier que les scripts fonctionnent toujours:

```bash
# Test 1: Vérifier module centralisé
python -c "from src.data.preprocessing import load_targets, resize_targets; print('✅ Imports OK')"

# Test 2: Vérifier train script (dry-run)
python scripts/training/train_hovernet_family.py --help

# Test 3: Vérifier validation script
python scripts/validation/compare_mse_vs_smoothl1.py --help
```

---

## Résumé

| # | Action | Statut |
|---|--------|--------|
| 1 | Modifier `train_hovernet_family.py` | ✅ FAIT |
| 2 | Modifier `compare_mse_vs_smoothl1.py` | ✅ FAIT |
| 3 | Documenter refactorisation | ✅ FAIT (ce document) |
| 4 | Générer données FIXED epidermal | ⏳ À FAIRE (commande fournie) |
| 5 | Ré-entraîner epidermal | ⏳ À FAIRE (après étape 4) |
| 6 | Valider FIXED vs OLD | ⏳ À FAIRE (après étape 5) |

---

**Prochaine action immédiate :** Exécuter l'étape 1 dans l'environnement conda `cellvit`.
