# Bug #7: Training Contamination - Modèle Entraîné sur TISSU au lieu de NOYAUX

## 🔴 DIAGNOSTIC FINAL (Expert Pathologiste)

**Problème:** AJI catastrophique (0.03-0.08 au lieu de >0.60) malgré Dice correct (0.82)

**Symptôme:** Le modèle prédit un "Giant Blob" (1 instance massive) au lieu de noyaux séparés

**Cause racine:** Le modèle a été entraîné à segmenter le TISSU au lieu des NOYAUX

---

## 📊 Preuve Empirique: Analyse Channel 5

Résultats de `inspect_gt_instances.py` sur échantillon test:

```
Channel 0 (Type 0 - NOYAUX):
  Unique values: 15 ([0.0, 3.0, 4.0, 12.0, 16.0, 26.0...68.0])
  Nonzero pixels: 7,411 (~11% de l'image) ✅
  Max value: 68.0
  → INSTANCES DÉTECTÉES (IDs séparés)

Channel 5 (Epithelial - TISSU):
  Unique values: [0.0, 1.0]
  Nonzero pixels: 56,475 (~86% de l'image) ❌
  Max value: 1.0
  → MASQUE BINAIRE (pas d'instances séparées)
```

**Observation critique:**
- Noyaux (Channels 0-4): **7,411 pixels (11%)**
- Tissu (Channel 5): **56,475 pixels (86%)**
- Ratio: **86% / 11% = 7.8× plus de tissu que de noyaux!**

---

## 🐛 Bug dans prepare_family_data_FIXED_v8.py

### Bug #1: compute_np_target() (Ligne 233)

```python
# ❌ BUG v8
def compute_np_target(mask: np.ndarray) -> np.ndarray:
    # Union binaire des canaux 1-5 (excluant canal 0 = background)
    np_target = mask[:, :, 1:].sum(axis=-1) > 0  # Inclut Channel 5 ❌
    return np_target.astype(np.float32)
```

**Problème:** `mask[:, :, 1:]` en Python signifie "channels 1, 2, 3, 4, **ET 5**"

**Impact:**
- NP target inclut Channel 5 (tissu, 86% pixels)
- Le modèle apprend à prédire: "où est le tissu?" au lieu de "où sont les noyaux?"
- Training Dice = 0.95 parce que le modèle segmente parfaitement le tissu!
- Inference AJI = 0.08 parce que le modèle compare tissu (86%) vs noyaux (11%)

### Bug #2: extract_pannuke_instances() (Lignes 201-211)

```python
# ❌ BUG v8
# Canal 5 (Epithelial): binaire, utiliser connectedComponents
epithelial_mask = mask[:, :, 5]
if epithelial_mask.max() > 0:
    _, epithelial_labels = cv2.connectedComponents(epithelial_mask.astype(np.uint8))
    # ... ajoute à inst_map
```

**Problème:** Ajoute les instances de Channel 5 (tissu) à l'instance map

**Impact:**
- HV maps calculés incluent le tissu comme "instances"
- Les gradients HV pointent vers le centre du tissu, pas vers les centres des noyaux
- Watershed crée 1 énorme instance de tissu au lieu de 10-15 noyaux séparés

### Bug #3: Ignore Channel 0

```python
# ❌ BUG v8
# Canaux 1-4: IDs d'instances natifs PanNuke (déjà séparés)
for c in range(1, 5):  # Commence à 1, ignore Channel 0!
    # ...
```

**Problème:** Ignore Channel 0 qui contient les vraies instances multi-types

**Impact pour epidermal:**
- Channels 1-4 sont souvent VIDES pour epidermal (pas de Neoplastic/Inflammatory/etc)
- Channel 0 contient 15 instances avec IDs [3, 4, 12...68]
- En ignorant Channel 0, on perd 15 noyaux et on garde seulement le tissu (Channel 5)

---

## ✅ FIX v9: NUCLEI ONLY (EXCLUT TISSU)

Script créé: `prepare_family_data_FIXED_v9_NUCLEI_ONLY.py`

### Fix #1: compute_np_target_NUCLEI_ONLY()

```python
# ✅ FIX v9
def compute_np_target_NUCLEI_ONLY(mask: np.ndarray) -> np.ndarray:
    """
    Génère le target NP UNIQUEMENT pour les NOYAUX (Channels 0-4).
    EXCLUT le channel 5 (Epithelial/Tissue).
    """
    mask = normalize_mask_format(mask)

    # ✅ Union binaire des canaux 0-4 (NOYAUX SEULEMENT)
    # [:5] signifie channels 0, 1, 2, 3, 4 (exclut 5)
    np_target = mask[:, :, :5].sum(axis=-1) > 0

    return np_target.astype(np.float32)
```

**Changement:** `mask[:, :, 1:]` → `mask[:, :, :5]`
- **Avant:** Channels 1, 2, 3, 4, 5 (inclut tissu)
- **Après:** Channels 0, 1, 2, 3, 4 (noyaux uniquement)

### Fix #2: extract_pannuke_instances_NUCLEI_ONLY()

```python
# ✅ FIX v9
def extract_pannuke_instances_NUCLEI_ONLY(mask: np.ndarray) -> np.ndarray:
    """
    Extrait UNIQUEMENT les instances de NOYAUX (Channels 0-4).
    EXCLUT le channel 5 (Epithelial/Tissue).
    """
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # ✅ PRIORITÉ 1: Channel 0 (multi-type instances) - SOURCE PRIMAIRE
    channel_0 = mask[:, :, 0]
    if channel_0.max() > 0:
        inst_ids_0 = np.unique(channel_0)
        inst_ids_0 = inst_ids_0[inst_ids_0 > 0]

        for inst_id in inst_ids_0:
            inst_mask = channel_0 == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1

    # ✅ PRIORITÉ 2: Canaux 1-4 (supplémentaires si non-vide)
    for c in range(1, 5):
        channel_mask = mask[:, :, c]
        if channel_mask.max() > 0:
            inst_ids = np.unique(channel_mask)
            inst_ids = inst_ids[inst_ids > 0]

            for inst_id in inst_ids:
                inst_mask = channel_mask == inst_id
                inst_mask_new = inst_mask & (inst_map == 0)  # Évite duplication

                if inst_mask_new.sum() > 0:
                    inst_map[inst_mask_new] = instance_counter
                    instance_counter += 1

    # ❌ Channel 5 (Epithelial/Tissue): EXCLU COMPLÈTEMENT
    # (commenté, pas de code pour Channel 5)

    return inst_map
```

**Changements:**
1. Utilise Channel 0 comme SOURCE PRIMAIRE (ignoré dans v8)
2. Ajoute Channels 1-4 comme supplémentaires (sans duplication)
3. EXCLUT Channel 5 (tissu) complètement

---

## 📈 Impact Attendu

### Métriques Avant (v8 - Bug):
```
Training:
- NP Dice: 0.9507 (excellent, mais sur TISSU!)
- HV MSE: 0.2749 (correct)
- NT Acc: 0.8800 (correct)

Evaluation:
- Dice: 0.3487 (catastrophique)
- AJI: 0.0311 (catastrophique)
- PQ: 0.0000 (catastrophique)
- Instances: 1 Giant Blob au lieu de 10-15 noyaux
```

### Métriques Attendues (v9 - Fix):
```
Training:
- NP Dice: ~0.95 (sur NOYAUX cette fois!)
- HV MSE: <0.05 (gradients vers centres de noyaux)
- NT Acc: ~0.88 (inchangé)

Evaluation:
- Dice: >0.85 (gain +144%)
- AJI: >0.60 (gain +1830% - de 0.03 à 0.60!)
- PQ: >0.50 (gain infini - de 0.00 à 0.50)
- Instances: 10-15 noyaux séparés correctement
```

**Gain AJI attendu: 0.0311 → 0.60 = +1830%**

---

## 🎯 Explication Biologique (Expert Pathologiste)

Citation de l'expert:

> "C'est impossible d'avoir 86% de noyaux dans une image. Ces sont des noyaux **DANS** du tissu. Vous avez entraîné un segmenteur de tissu, pas HoVer-Net."

**Règle biologique:**
- Noyaux: ~10-15% de la surface tissulaire (taille typique)
- Tissu: ~80-90% de la surface (cytoplasme + matrice extracellulaire)

**Conséquence de l'erreur:**
- Modèle apprend: "Prédire où est le tissu épithélial" (tâche facile, Dice 0.95)
- On veut: "Séparer les noyaux individuels dans le tissu" (tâche difficile, AJI 0.60+)

C'est comme entraîner un détecteur de visages avec:
- **Voulue:** Photos de visages individuels (10% de l'image)
- **Réelle:** Photos de foules entières (90% de l'image)

Le modèle apprend à détecter la foule, pas les visages!

---

## 📝 Plan de Récupération

### Étape 1: Régénérer Données (15-20 min)

```bash
python scripts/preprocessing/prepare_family_data_FIXED_v9_NUCLEI_ONLY.py \
    --family epidermal \
    --pannuke_dir /home/amar/data/PanNuke \
    --output_dir data/family_FIXED \
    --folds 0 1 2
```

**Vérifications attendues:**
- NP coverage: ~10-15% (pas 86%!)
- Fichier: `epidermal_data_FIXED_v9_NUCLEI_ONLY.npz`
- Taille: ~50-100 MB (pas 200+ MB avec tissu)

### Étape 2: Re-training (40-50 min)

```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --data_file data/family_FIXED/epidermal_data_FIXED_v9_NUCLEI_ONLY.npz \
    --epochs 50 \
    --augment \
    --lambda_hv 2.0
```

**Métriques training attendues:**
- NP Dice: ~0.95 (inchangé)
- HV MSE: <0.05 (amélioration vs 0.27)
- NT Acc: ~0.88 (inchangé)

### Étape 3: Évaluation Finale (5 min)

```bash
python scripts/evaluation/test_epidermal_aji_FINAL.py \
    --checkpoint models/checkpoints/hovernet_epidermal_best_v9.pth \
    --n_samples 50
```

**Résultats attendus:**
```
✅ Dice: 0.85+ (vs 0.35 avant)
✅ AJI: 0.60+ (vs 0.03 avant, gain +1830%)
✅ PQ: 0.50+ (vs 0.00 avant)
✅ Instances: 10-15 séparés (vs 1 Giant Blob avant)
```

---

## 🔬 Validation Technique

### Test de Vérité: Vérifier NP Coverage

Après régénération, vérifier dans le fichier v9:

```python
import numpy as np

data = np.load("data/family_FIXED/epidermal_data_FIXED_v9_NUCLEI_ONLY.npz")
np_targets = data['np_targets']

coverage = np_targets.mean() * 100
print(f"NP coverage: {coverage:.2f}%")

# Attendu: ~10-15% (noyaux)
# v8 bugué: ~86% (tissu)
```

### Test de Sanité: Compter Instances

```python
from scipy.ndimage import label

# Charger masque GT
pannuke_masks = np.load("/home/amar/data/PanNuke/fold2/masks.npy")
mask = pannuke_masks[0]  # Premier échantillon

# Méthode v8 (bugué)
np_v8 = mask[:, :, 1:].sum(axis=-1) > 0
inst_v8, n_v8 = label(np_v8)
print(f"v8 (bug): {n_v8} instances")  # Attendu: 1-3 (tissu fusionné)

# Méthode v9 (fixé)
np_v9 = mask[:, :, :5].sum(axis=-1) > 0
inst_v9, n_v9 = label(np_v9)
print(f"v9 (fix): {n_v9} instances")  # Attendu: 10-15 (noyaux séparés)
```

---

## 📚 Leçons Apprises

### 1. Training Dice ≠ Modèle Correct

**Problème:** Dice 0.95 en training semblait excellent, mais le modèle apprenait la mauvaise tâche

**Leçon:** Toujours vérifier:
- Quelle est la **définition biologique** de la tâche?
- Les targets d'entraînement correspondent-ils à cette définition?
- Dice élevé peut cacher un problème de définition

### 2. Channel 5 de PanNuke n'est PAS des Noyaux

**Documentation PanNuke:**
- Channels 0-4: **Instances de noyaux** (séparées avec IDs)
- Channel 5: **Masque de tissu épithélial** (binaire, pas d'instances)

**Erreur:** Inclure Channel 5 dans les targets de noyaux

**Conséquence:** Modèle segmente tissu au lieu de noyaux

### 3. Array Slicing Python: Attention aux Bornes

```python
mask[:, :, 1:]   # Channels 1, 2, 3, 4, ET 5 (borne supérieure exclue)
mask[:, :, :5]   # Channels 0, 1, 2, 3, 4 (exclut 5)
```

**Erreur subtile:** `1:` signifie "de 1 jusqu'à la FIN" (inclut 5!)

### 4. Diagnostic: Paradoxe Dice-AJI

**Observation:** Dice 0.82 avec AJI 0.03

**Signification:**
- Dice mesure le **chevauchement global** (masse)
- AJI mesure la **séparation des instances** (précision géométrique)
- Dice élevé + AJI faible = "Segmentation fantôme" (bonne masse, mauvaise position)

**Ici:** Dice 0.82 parce que le modèle prédit ~70% de pixels (tissu ≈ 86%)
AJI 0.03 parce qu'il compare 1 blob de tissu vs 15 noyaux séparés

---

## ✅ Checklist Avant Re-training

- [ ] Vérifier v9 utilise `mask[:, :, :5]` (pas `1:`)
- [ ] Vérifier v9 exclut complètement Channel 5
- [ ] Vérifier v9 utilise Channel 0 comme priorité 1
- [ ] Régénérer données epidermal avec v9
- [ ] Vérifier NP coverage ~10-15% (pas 86%)
- [ ] Re-entraîner modèle avec nouvelles données
- [ ] Évaluer AJI final (objectif >0.60)

---

## 🎯 Prédiction Expert

Citation:

> "Ton Dice à 0.97 sur le crop 224 montre que ton décodeur est hyper-puissant. Il a juste besoin d'apprendre sur un terrain où les cibles ne bougent pas. Une fois le re-training terminé avec des données synchronisées (v9), ton AJI va passer de 0.06 à 0.65 en une seule session."

**Attendu:** AJI 0.08 → 0.65 (+712%) après re-training v9
