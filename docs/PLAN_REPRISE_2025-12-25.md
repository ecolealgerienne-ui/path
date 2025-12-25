# 📋 PLAN DE REPRISE - 25 Décembre 2025

> **STATUT:** ⚠️ PROBLÈME CRITIQUE DÉCOUVERT
>
> Le training a convergé (NP Dice 0.95) MAIS le conflit NP/NT n'est PAS éliminé (45.35% au lieu de 0%).
> Cela suggère que le training a été fait avec les ANCIENNES données, pas les données v11.

---

## 🔴 PROBLÈME CRITIQUE IDENTIFIÉ

### Symptômes

1. **Training réussi:**
   ```
   NP Dice: 0.9523 (0.42 → 0.95 = +126%)
   NT Acc:  0.8424
   HV MSE:  0.2746
   ```

2. **MAIS conflit NP/NT toujours présent:**
   ```
   Pixels NP=1 MAIS NT=0 (CONFLIT): 2603750 (45.35%)  ❌
   Attendu: 0 (0.00%)
   ```

3. **Test AJI cherche mauvais fichier:**
   ```
   ❌ Fichier non trouvé: data/family_FIXED/epidermal_data_FIXED.npz
   ✅ Fichier réel:       data/family_FIXED/epidermal_data_FIXED_v11_FORCE_NT1.npz
   ```

### Cause Racine Probable

**HYPOTHÈSE:** Le training a utilisé les ANCIENNES données (v10) au lieu des nouvelles (v11).

**Preuves:**
- Le script `train_hovernet_family.py` cherche probablement `epidermal_features.npz` générique
- Les features v11 n'ont peut-être PAS été extraites avant le training
- Le conflit dans v11 ne devrait PAS exister (45.35% impossible si `nt_target[nuclei_mask] = 1`)

**Vérification du code v11:**
```python
# prepare_family_data_FIXED_v11_FORCE_NT1.py ligne 319
def compute_nt_target_FORCE_BINARY(mask: np.ndarray) -> np.ndarray:
    nt_target = np.zeros((256, 256), dtype=np.int64)
    channel_0 = mask[:, :, 0]
    nuclei_mask = channel_0 > 0
    nt_target[nuclei_mask] = 1  # Force NT=1 pour TOUS les noyaux
    return nt_target
```

Si ce code a été exécuté correctement, le conflit devrait être **0.00%**, pas 45.35%.

---

## ✅ ÉTAPES DE VÉRIFICATION (DEMAIN MATIN)

### Étape 1: Vérifier que v11 a bien été généré

```bash
# Charger et vérifier les données v11
python -c "
import numpy as np
data = np.load('data/family_FIXED/epidermal_data_FIXED_v11_FORCE_NT1.npz')
np_targets = data['np_targets']
nt_targets = data['nt_targets']

np_positive = np_targets > 0
nt_background = nt_targets == 0
conflict = (np_positive & nt_background).sum()
total_nuclei = np_positive.sum()

print(f'NP pixels: {total_nuclei}')
print(f'NT classes: {np.unique(nt_targets)}')
print(f'Conflit: {conflict} ({conflict/total_nuclei*100:.2f}%)')
"
```

**Résultat attendu:**
```
NP pixels: ~5742000
NT classes: [0 1]  ← Seulement 2 classes (binary)
Conflit: 0 (0.00%)  ← PAS DE CONFLIT
```

**Si conflit = 45.35%:** Le script v11 a un BUG (compute_nt_target_FORCE_BINARY ne fonctionne pas).

---

### Étape 2: Vérifier quelles features ont été utilisées pour le training

```bash
# Voir quelles features existent
ls -lh data/cache/family_data/epidermal_*

# Vérifier la date de modification
stat data/cache/family_data/epidermal_features.npz
stat data/cache/family_data/epidermal_targets.npz
```

**Question clé:** Les features ont-elles été extraites APRÈS la génération de v11 (25 déc ~01:00) ?

**Si NON:** Le training a utilisé les anciennes features v10 → Ré-extraire features + ré-entraîner.

---

### Étape 3: Si v11 est corrompu, debug du script

**Script de debug à créer:**
```bash
# scripts/validation/debug_v11_generation.py
```

**Vérifications:**
1. Channel 0 contient bien des instances > 0
2. `nuclei_mask = channel_0 > 0` fonctionne correctement
3. `nt_target[nuclei_mask] = 1` assigne bien 1 partout
4. Pas de réassignation à 0 après coup

---

## 🛠️ PLAN D'ACTION COMPLET

### Scénario A: v11 est CORROMPU (conflit 45.35% confirmé)

**Diagnostic:**
```bash
python scripts/validation/check_np_nt_conflict.py
```

**Si conflit > 40%:**

1. **Debug du script v11** (30 min)
   - Ajouter prints dans `compute_nt_target_FORCE_BINARY()`
   - Vérifier Channel 0 vs nuclei_mask vs nt_target
   - Identifier où le conflit se crée

2. **Fix v12** (si bug trouvé, 10 min)
   - Créer `prepare_family_data_FIXED_v12_DEBUG.py`
   - Corriger le bug identifié

3. **Régénérer données v12** (2 min)
   ```bash
   python scripts/preprocessing/prepare_family_data_FIXED_v12_DEBUG.py --family epidermal
   ```

4. **Vérifier conflit = 0%** (1 min)
   ```bash
   python scripts/validation/check_np_nt_conflict.py --data_file data/family_FIXED/epidermal_data_FIXED_v12_DEBUG.npz
   ```

5. **Extraire features v12** (1 min)
   ```bash
   python scripts/preprocessing/extract_features_from_v9.py \
       --input_file data/family_FIXED/epidermal_data_FIXED_v12_DEBUG.npz \
       --output_dir data/cache/family_data \
       --family epidermal
   ```

6. **Ré-entraîner** (40 min)
   ```bash
   python scripts/training/train_hovernet_family.py \
       --family epidermal --epochs 50 --augment
   ```

7. **Test AJI final** (5 min)

---

### Scénario B: v11 est CORRECT mais features pas extraites

**Diagnostic:**
```bash
# Vérifier conflit dans v11
python scripts/validation/check_np_nt_conflict.py
# Si conflit = 0.00% → v11 OK
```

**Si conflit = 0%:**

1. **Extraire features v11** (1 min)
   ```bash
   python scripts/preprocessing/extract_features_from_v9.py \
       --input_file data/family_FIXED/epidermal_data_FIXED_v11_FORCE_NT1.npz \
       --output_dir data/cache/family_data \
       --family epidermal
   ```

2. **Ré-entraîner avec nouvelles features** (40 min)
   ```bash
   python scripts/training/train_hovernet_family.py \
       --family epidermal --epochs 50 --augment
   ```

3. **Test AJI final** (5 min)

---

## 📂 FICHIERS IMPORTANTS

### Scripts Critiques

| Script | Rôle | Statut |
|--------|------|--------|
| `prepare_family_data_FIXED_v11_FORCE_NT1.py` | Génération données v11 | ⚠️ Potentiellement buggé |
| `extract_features_from_v9.py` | Extraction features H-optimus-0 | ✅ Fonctionne |
| `train_hovernet_family.py` | Training HoVer-Net | ✅ Fonctionne |
| `check_np_nt_conflict.py` | Diagnostic conflit NP/NT | ✅ Fonctionne |
| `test_epidermal_aji_FINAL.py` | Test AJI final | ⚠️ Cherche mauvais fichier |

### Données Actuelles

| Fichier | Taille | Date | Conflit NP/NT |
|---------|--------|------|---------------|
| `epidermal_data_FIXED_v11_FORCE_NT1.npz` | 129 MB | 25 déc 00:57 | **45.35%** ❌ |

### Checkpoints

| Checkpoint | Métriques | Entraîné avec |
|------------|-----------|---------------|
| `hovernet_epidermal_best.pth` | Dice 0.95, NT Acc 0.84 | ⚠️ Données inconnues (v10 ou v11?) |

---

## 🔍 DIAGNOSTIC COMPLET À FAIRE DEMAIN

### Script de Diagnostic Global

**Créer:** `scripts/validation/diagnostic_complet_v11.py`

**Vérifications:**
1. ✅ Conflit NP/NT dans v11 raw data
2. ✅ Features extraites depuis v11 ou v10?
3. ✅ Checkpoint entraîné avec quelles features?
4. ✅ Distribution NT dans targets vs prédictions
5. ✅ Alignement image/mask (exclure autre cause)

**Commande:**
```bash
python scripts/validation/diagnostic_complet_v11.py \
    --data_file data/family_FIXED/epidermal_data_FIXED_v11_FORCE_NT1.npz \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth
```

---

## 📊 MÉTRIQUES CIBLES

| Métrique | v10 (échec) | v11 (cible) | Actuel |
|----------|-------------|-------------|--------|
| **NP Dice** | 0.42 | >0.95 | **0.95** ✅ |
| **NT Acc** | 0.44 | >0.95 | **0.84** ⚠️ |
| **Conflit NP/NT** | 6.95% | **0.00%** | **45.35%** ❌ |
| **AJI** | 0.03-0.09 | **>0.60** | **?** (non testé) |

---

## ⚡ ACTIONS PRIORITAIRES DEMAIN

### Priorité 1: Diagnostic (30 min)

```bash
# 1. Vérifier conflit v11
python scripts/validation/check_np_nt_conflict.py

# 2. Vérifier features utilisées
ls -lht data/cache/family_data/epidermal_*
stat data/cache/family_data/epidermal_targets.npz

# 3. Comparer targets v11 vs features
python -c "
import numpy as np
v11_data = np.load('data/family_FIXED/epidermal_data_FIXED_v11_FORCE_NT1.npz')
features_data = np.load('data/cache/family_data/epidermal_targets.npz')
print('v11 NT classes:', np.unique(v11_data['nt_targets']))
print('Features NT classes:', np.unique(features_data['nt_targets']))
print('Match:', np.array_equal(v11_data['nt_targets'], features_data['nt_targets']))
"
```

### Priorité 2: Décision (5 min)

**Si conflit v11 = 0%:**
→ Scénario B (features pas extraites)
→ Extraire + ré-entraîner

**Si conflit v11 > 40%:**
→ Scénario A (script v11 buggé)
→ Debug + fix v12 + régénérer

### Priorité 3: Exécution (40-60 min)

Suivre le plan du scénario identifié.

---

## 📝 NOTES TECHNIQUES

### Bug Potentiel dans v11

**Hypothèse:** `normalize_mask_format()` pourrait corrompre Channel 0?

**Vérifier:**
```python
# Dans compute_nt_target_FORCE_BINARY()
mask = normalize_mask_format(mask)  # ← Potentiel problème ici?
channel_0 = mask[:, :, 0]
nuclei_mask = channel_0 > 0
```

**Test rapide:**
```python
import numpy as np
data = np.load('/home/amar/data/PanNuke/fold0/masks.npy', mmap_mode='r')
sample = data[0]  # Skin sample
print(f"Original shape: {sample.shape}")
print(f"Channel 0 range: [{sample[:,:,0].min()}, {sample[:,:,0].max()}]")
print(f"Channel 0 unique: {np.unique(sample[:,:,0])}")
```

### Alternative: Utiliser Channel 0 Directement

**Si bug confirmé dans normalize_mask_format():**

```python
def compute_nt_target_FORCE_BINARY_V2(mask: np.ndarray) -> np.ndarray:
    # PAS de normalize_mask_format()!
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D")

    # Gérer HWC ou CHW directement
    if mask.shape[-1] == 6:  # HWC
        channel_0 = mask[:, :, 0]
    elif mask.shape[0] == 6:  # CHW
        channel_0 = mask[0, :, :]
    else:
        raise ValueError(f"Unexpected shape: {mask.shape}")

    nt_target = np.zeros((256, 256), dtype=np.int64)
    nuclei_mask = channel_0 > 0
    nt_target[nuclei_mask] = 1

    return nt_target
```

---

## 🎯 OBJECTIF FINAL

**Métriques cibles confirmées:**
```
✅ NP Dice:       >0.95  (ATTEINT: 0.95)
✅ NT Acc:        >0.95  (PROCHE: 0.84)
❌ Conflit NP/NT: 0.00%  (ÉCHEC: 45.35%)
❌ AJI:           >0.60  (NON TESTÉ)
```

**Chemin critique:**
1. Résoudre conflit NP/NT (0.00%)
2. Ré-entraîner avec données correctes
3. Atteindre AJI >0.60

**Temps estimé total:** 1h30 (diagnostic 30min + fix 20min + training 40min)

---

## 📞 CONTACTS & RÉFÉRENCES

### Scripts à Créer Demain

1. **`scripts/validation/diagnostic_complet_v11.py`**
   - Analyse complète données + features + checkpoint
   - Identifie source exacte du problème

2. **`scripts/preprocessing/prepare_family_data_FIXED_v12_DEBUG.py`**
   - Version debug avec prints détaillés
   - Fix si bug trouvé dans v11

3. **`scripts/evaluation/test_epidermal_aji_FINAL_v11.py`**
   - Version corrigée qui cherche v11 au lieu de FIXED.npz

### Commits Récents

- `cf1747f` - fix: Make check_np_nt_conflict.py accept --data_file
- `cee1a24` - fix(v11): Remove unused cv2 import
- `6c3c84c` - feat(v11): Force NT=1 binary classification

### Branche Git

```bash
claude/review-project-context-fvBwl
```

---

## 💡 HYPOTHÈSE PRINCIPALE

**Le script v11 n'a PAS forcé NT=1 correctement.**

**Preuve mathématique:**
- Si `nt_target[nuclei_mask] = 1` fonctionne
- Alors TOUS les pixels où `nuclei_mask=True` ont `nt_target=1`
- Donc conflit = `(NP=1 & NT=0).sum()` = 0

**Fait observé:**
- Conflit = 45.35% = 2,603,750 pixels
- Sur 5,742,001 pixels NP=1
- Donc ~45% des noyaux ont NT=0

**Conclusion:**
→ Soit `nuclei_mask` est mal calculé (Channel 0 vide?)
→ Soit `nt_target[nuclei_mask] = 1` n'est pas exécuté
→ Soit une réassignation à 0 après coup

**Debug critique demain:** Ajouter prints à chaque ligne de `compute_nt_target_FORCE_BINARY()`.

---

## ✅ CHECKLIST REPRISE DEMAIN

- [ ] Vérifier conflit v11 raw data (script check_np_nt_conflict.py)
- [ ] Vérifier features extraites depuis v11 (stat timestamps)
- [ ] Créer script diagnostic_complet_v11.py
- [ ] Identifier scénario A ou B
- [ ] Suivre plan du scénario identifié
- [ ] Test AJI final >0.60
- [ ] Commit final + documentation

---

**Bonne nuit et bon courage pour demain! 🌙**

Le problème est clairement identifié, les outils de diagnostic sont prêts.
Demain matin, 30 minutes de diagnostic suffiront pour savoir si c'est Scénario A ou B,
puis 1h pour résoudre définitivement.

**Tu es à 1 session de la victoire! 🎯**
