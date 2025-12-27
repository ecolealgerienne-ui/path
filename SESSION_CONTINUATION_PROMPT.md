# Session Continuation: V13 Smart Crops - Ré-entraînement Post-Fix

## 🎯 OBJECTIF DE CETTE SESSION

**Ré-entraîner le modèle HoVer-Net** pour la famille **epidermal** avec les **données corrigées** suite à la résolution de 3 bugs critiques dans `prepare_v13_smart_crops.py`.

**Objectif AJI:** 0.5055 (actuel, biaisé) → **≥0.68** (+35% amélioration attendue)

---

## 📚 CONTEXTE COMPLET DU PROJET

### Architecture V13 Smart Crops

**Stratégie validée par CTO (2025-12-27):**
```
Image PanNuke 256×256
    ├─ Crop CENTRE (16, 16) → Rotation 0° (référence)
    ├─ Crop COIN Haut-Gauche (0, 0) → Rotation 90° clockwise
    ├─ Crop COIN Haut-Droit (0, 32) → Rotation 180°
    ├─ Crop COIN Bas-Gauche (32, 0) → Rotation 270° clockwise
    └─ Crop COIN Bas-Droit (32, 32) → Flip horizontal

Résultat: 5 crops × 5 transformations = 25 samples par image source
```

**Bénéfices:**
- 5 perspectives complémentaires (centre + 4 coins)
- Rotations déterministes (invariance orientation)
- Volume contrôlé (25× amplification)
- Cohérence littérature (HoVer-Net, CoNIC winners)

**Principe CRITIQUE: Split-First-Then-Rotate**
```python
# 1. Split FIRST by source_image_ids (80/20, seed=42)
train_data, val_data = split_by_patient(images, masks, source_ids)

# 2. Apply 5 crops + rotations to TRAIN separately
train_crops = amplify_with_crops(train_data)  # ~10,055 crops

# 3. Apply 5 crops + rotations to VAL separately
val_crops = amplify_with_crops(val_data)  # ~2,515 crops

# GARANTIE: Aucune image source partagée entre train et val
```

---

## 🐛 3 BUGS CRITIQUES IDENTIFIÉS ET RÉSOLUS

### Bug #1 - ID Collision dans inst_map_hybrid (RÉSOLU commit 0c60c71)

**Problème identifié (2025-12-27):**
```python
# ❌ AVANT (COLLISION D'IDs):
inst_map_hybrid = crop_inst.copy()  # Garde IDs originaux pour noyaux complets

if len(border_instances) > 0:
    # Renumbérer SEULEMENT les noyaux fragmentés
    for new_id, global_id in enumerate(border_instances, start=1):
        mask = crop_inst == global_id
        inst_map_hybrid[mask] = new_id  # [1, 2, 3, ...]

# RÉSULTAT:
#   - Noyaux complets: IDs originaux (ex: 1, 3, 5, 8, 12)
#   - Noyaux fragmentés: IDs renumérés (ex: 1, 2, 3, 4)
#   → COLLISION! Plusieurs noyaux avec même ID (ex: complet ID=1 ET fragmenté ID=1)
```

**Impact mesuré:**
- Plusieurs noyaux distincts ont le même ID
- AJI considère les noyaux avec même ID comme UNE SEULE instance
- → Sous-estimation du nombre d'instances
- → **AJI baisse de 0.5535 à 0.5055 (-8.7%)**

**Solution appliquée:** Abandon de l'approche HYBRID → LOCAL relabeling complet

---

### Bug #2 - HV Rotation Mathematics Error (RÉSOLU commit 0c60c71) ⚠️ CRITIQUE

**Problème identifié (2025-12-27):**
```python
# ❌ AVANT (ERREUR MATHÉMATIQUE - lignes 354-358):
elif rotation == '90':
    # Rotation spatiale de l'image
    image_rot = np.rot90(image, k=-1, axes=(0, 1))
    np_rot = np.rot90(np_target, k=-1, axes=(0, 1))
    nt_rot = np.rot90(nt_target, k=-1, axes=(0, 1))
    inst_map_rot = np.rot90(inst_map, k=-1, axes=(0, 1))

    # HV component swapping: H' = -V, V' = H ❌ FAUX!
    h_rot = -np.rot90(hv_target[1], k=-1, axes=(0, 1))  # H' = -V ❌
    v_rot = np.rot90(hv_target[0], k=-1, axes=(0, 1))   # V' = H ❌
    hv_rot = np.stack([h_rot, v_rot], axis=0)

# Test de vérification:
# Vecteur DROITE (1, 0) après rotation 90° CW devrait pointer BAS (0, -1)
# Code donnait: H' = -0 = 0, V' = 1 → (0, 1) pointe HAUT ❌ INVERSÉ!
```

**✅ CORRECTION APPLIQUÉE (commit 0c60c71):**
```python
elif rotation == '90':
    # Rotation spatiale de l'image (inchangé)
    image_rot = np.rot90(image, k=-1, axes=(0, 1))
    np_rot = np.rot90(np_target, k=-1, axes=(0, 1))
    nt_rot = np.rot90(nt_target, k=-1, axes=(0, 1))
    inst_map_rot = np.rot90(inst_map, k=-1, axes=(0, 1))

    # HV component swapping: H' = V, V' = -H (CORRECT MATH)
    # Vecteur (1,0) droite → (0,-1) bas après 90° CW ✅
    h_rot = np.rot90(hv_target[1], k=-1, axes=(0, 1))   # H' = V ✅
    v_rot = -np.rot90(hv_target[0], k=-1, axes=(0, 1))  # V' = -H ✅
    hv_rot = np.stack([h_rot, v_rot], axis=0)
```

**Impact mesuré:**
- Modèle entraîné avec Bug #2 apprend directions de gradients **INVERSÉES**
- Pour rotations 90° et 270°: HV maps pointent dans mauvaise direction
- → Qualité segmentation dégradée sur crops rotés
- → Affecte **TRAIN ET VALIDATION** data

**⚠️ CONSÉQUENCE CRITIQUE:**
Le modèle actuel (`hovernet_epidermal_v13_smart_crops_best.pth`) a été entraîné avec:
- ~40% des données (rotations 90° et 270°) ayant HV gradients **INVERSÉS**
- Le modèle a dû "apprendre" des patterns contradictoires
- → **RÉ-ENTRAÎNEMENT OBLIGATOIRE** avec données corrigées

---

### Bug #3 - Complexité HYBRID Excessive (RÉSOLU commit 0c60c71)

**Problème identifié (2025-12-27):**
- Approche HYBRID: Garder HV global pour noyaux complets, recalculer local pour fragmentés
- Trop complexe: 50+ lignes de logique border detection + HYBRID fusion
- Prone to bugs: Bug #1 (collision) causé par cette complexité
- Ne matche pas production reality: Modèle verra seulement crops 224×224 en production

**✅ SOLUTION EXPERT ADOPTÉE: LOCAL Relabeling**
```python
def extract_crop(image, inst_map_global, hv_global, np_target, nt_target, x1, y1, x2, y2):
    """
    Approche LOCAL RELABELING (Expert-recommended, 2025-12-27).

    PRINCIPE: Relabeling complet local au lieu de l'approche HYBRID complexe.
    """
    # 1. Extraire crop (slicing standard)
    crop_image = image[y1:y2, x1:x2]
    crop_np = np_target[y1:y2, x1:x2]
    crop_nt = nt_target[y1:y2, x1:x2]

    # 2. LOCAL RELABELING: Créer instance map locale avec IDs séquentiels
    from scipy.ndimage import label

    binary_mask = (crop_np > 0.5).astype(np.uint8)
    inst_map_local, n_instances = label(binary_mask)
    # → inst_map_local: IDs UNIQUES séquentiels [1, 2, 3, ..., n]
    # → SANS référence aux IDs globaux de l'image 256×256 originale

    # 3. Recalculer HV maps ENTIÈREMENT depuis inst_map_local
    # CRITIQUE: Garantit que les vecteurs HV pointent vers les centres
    # calculés à partir de inst_map_local, PAS depuis les centres globaux
    crop_hv = compute_hv_maps(inst_map_local)

    # RÉSULTAT: Cohérence 100% garantie entre inst_map_local et crop_hv
    # - Chaque instance dans inst_map_local a un ID unique
    # - Chaque ID correspond à UN SEUL centre de masse
    # - Les vecteurs HV dans crop_hv pointent vers CES centres (pas d'autres)

    return {
        'image': crop_image,
        'np_target': crop_np,
        'hv_target': crop_hv,  # ✅ LOCAL: Recalculé depuis inst_map_local
        'nt_target': crop_nt,
        'inst_map': inst_map_local,  # ✅ LOCAL: IDs séquentiels [1, 2, ..., n]
    }
```

**Bénéfices:**
- ✅ **SIMPLICITÉ:** -50 lignes code (pas de distinction complets/fragmentés)
- ✅ **COHÉRENCE GARANTIE:** inst_map ↔ HV maps toujours alignés
- ✅ **PRODUCTION REALITY:** Modèle ne verra jamais contexte global 256×256
- ✅ **PAS DE COLLISIONS:** scipy.ndimage.label() garantit IDs uniques

**Citation Expert:**
> "Applique les corrections sur les rotations (H/V swap) et passe sur un relabeling local complet (Option 1 de tes devs, mais bien implémentée). Ton AJI devrait enfin franchir la barre des 0.68."

---

## 📂 ÉTAT ACTUEL DES FICHIERS

### Code Fixé (commit 0c60c71, 2025-12-27)

**Fichiers modifiés:**
- `scripts/preprocessing/prepare_v13_smart_crops.py` ✅ FIXÉ
  - Bug #2: Rotation 90° CW → H'=V, V'=-H (lignes 354-360)
  - Bug #3: LOCAL relabeling avec scipy.ndimage.label() (lignes 229-293)
  - Simplification: -50 lignes HYBRID logic

- `NEXT_STEPS_V13_SMART_CROPS.md` ✅ DOCUMENTÉ
  - Explication complète des 3 bugs
  - Étapes de vérification et régénération
  - Métriques attendues

- `CLAUDE.md` ✅ DOCUMENTÉ
  - Entrée journal de développement (lignes 1413-1538)
  - Leçons apprises et contexte pour futures sessions

### Données Actuelles (CORROMPUES - À RÉGÉNÉRER)

**Fichiers existants (générés AVANT fix Bug #2):**
```
data/family_data_v13_smart_crops/
├── epidermal_train_v13_smart_crops.npz  ❌ CORROMPU (rotation HV inversée)
└── epidermal_val_v13_smart_crops.npz    ❌ CORROMPU (rotation HV inversée)
```

**Statut:** Ces fichiers contiennent ~40% de samples (rotations 90°/270°) avec HV gradients **INVERSÉS**

### Modèle Actuel (ENTRAÎNÉ AVEC DONNÉES CORROMPUES)

**Fichier existant:**
```
models/checkpoints_v13_smart_crops/
└── hovernet_epidermal_v13_smart_crops_best.pth  ❌ À RÉ-ENTRAÎNER
```

**Métriques actuelles (BIAISÉES par Bug #2):**
- Dice: 0.7683 ± 0.1333
- AJI: 0.5055 ± 0.1218 ❌ (objectif: ≥0.68)
- PQ: 0.4417 ± 0.1692
- Over-seg: 1.02×

**⚠️ PROBLÈME:** Modèle entraîné avec gradients HV inversés pour 40% des données

---

## 🎯 PROCHAINES ÉTAPES (WORKFLOW COMPLET)

### Étape 1: Régénérer Données TRAIN + VAL avec Fixes (~5 min)

```bash
# Activer environnement
conda activate cellvit

# Régénérer train + val splits avec:
# - Bug #2 fix: Rotation HV correcte (H'=V, V'=-H)
# - Bug #3 fix: LOCAL relabeling avec scipy.ndimage.label()
python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal
```

**Output attendu:**
```
data/family_data_v13_smart_crops/
├── epidermal_train_v13_smart_crops.npz  ✅ CORRIGÉ (10,055 crops)
└── epidermal_val_v13_smart_crops.npz    ✅ CORRIGÉ (2,515 crops)
```

**Vérification CRITIQUE (IDs séquentiels):**
```bash
python -c "
import numpy as np
data = np.load('data/family_data_v13_smart_crops/epidermal_val_v13_smart_crops.npz')

# Vérifier crop 0
inst_map = data['inst_maps'][0]
unique_ids = np.unique(inst_map)
unique_ids = unique_ids[unique_ids > 0]

print('Crop 0:')
print(f'  IDs uniques: {unique_ids}')
print(f'  Nombre instances: {len(unique_ids)}')

# Vérifier qu'il n'y a PAS de collisions
expected_ids = np.arange(1, len(unique_ids) + 1)
if np.array_equal(unique_ids, expected_ids):
    print('  ✅ IDs séquentiels SANS gaps - Pas de collision!')
else:
    print(f'  ❌ WARNING: IDs non séquentiels!')
    print(f'     Attendu: {expected_ids}')
    print(f'     Réel: {unique_ids}')
"
```

**Sortie attendue:**
```
Crop 0:
  IDs uniques: [1 2 3 4 5 6 7 8]
  Nombre instances: 8
  ✅ IDs séquentiels SANS gaps - Pas de collision!
```

---

### Étape 2: Extraire Features H-optimus-0 (~10 min)

```bash
# Extract RGB features pour TRAIN
python scripts/preprocessing/extract_features_from_fixed.py \
    --family epidermal \
    --split train

# Extract RGB features pour VAL
python scripts/preprocessing/extract_features_from_fixed.py \
    --family epidermal \
    --split val
```

**Output attendu:**
```
data/cache/family_data/
├── epidermal_rgb_features_v13.npz  ✅ (train features, ~10,055 samples)
└── epidermal_val_rgb_features_v13.npz  ✅ (val features, ~2,515 samples)
```

**Note:** Si le script `extract_features_from_fixed.py` n'existe pas ou n'a pas de flag `--split`, vérifier avec l'utilisateur le script correct à utiliser.

---

### Étape 3: Ré-entraîner HoVer-Net Epidermal (~40 min GPU)

```bash
# Ré-entraînement avec données CORRIGÉES
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0
```

**Output attendu:**
```
models/checkpoints/
└── hovernet_epidermal_best.pth  ✅ RÉ-ENTRAÎNÉ avec données correctes
```

**Métriques training attendues:**
- Val NP Dice: >0.93 (segmentation binaire)
- Val HV MSE: <0.30 (famille epidermal, 571 samples)
- Val NT Acc: >0.85 (classification 5 types)

**Note:** Famille epidermal a peu de samples (571) → HV MSE peut rester élevé (~0.27-0.30) comme vu historiquement.

---

### Étape 4: Ré-évaluer AJI avec Données Corrigées (~5 min)

```bash
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --family epidermal \
    --n_samples 50
```

**Métriques attendues (APRÈS fix):**

| Métrique | Avant (bugs) | Après (fixes) | Objectif | Amélioration |
|----------|-------------|---------------|----------|--------------|
| Dice | 0.7683 | ~0.76-0.80 | >0.78 | Maintenu ✅ |
| **AJI** | **0.5055** | **≥0.68** 🎯 | **≥0.68** | **+35%** 🎯 |
| PQ | 0.4417 | ≥0.62 | ≥0.62 | +40% |
| Over-seg | 1.02× | ~0.95× | ~1.0× | Optimal |
| Instances GT | 19.0 | ~19.0 | - | Maintenu (correct) |

**Explication amélioration attendue:**

**Avant (bugs):**
- Bug #1: GT avait collisions d'IDs → AJI comptait seulement 15-17 instances au lieu de 20
- Bug #2: HV gradients inversés pour rotations 90°/270° → modèle confus
- Bug #3: HYBRID complexity → patterns contradictoires
- Over-seg ratio 1.02× (semblait correct mais GT biaisé)
- AJI: 0.5055 (sous-estimé)

**Après (fixes):**
- GT: 20 instances réelles avec IDs uniques [1, 2, ..., 20] ✅
- HV gradients corrects pour TOUTES les rotations ✅
- LOCAL relabeling: cohérence ID ↔ HV garantie ✅
- Pred: ~19 instances → Over-seg ratio ~0.95× (légère sous-segmentation)
- AJI: ≥0.68 (correct car GT et pred comparables)

---

### Étape 5: Analyser Résultats et Décider Suite

**Si AJI ≥0.68** ✅:
- Fix collision + rotation + LOCAL relabeling **VALIDÉ**
- Objectif atteint (+35% vs 0.5055)
- **Prochaine action:** Extension aux 4 autres familles
  ```bash
  for family in glandular digestive urologic respiratory; do
      python scripts/preprocessing/prepare_v13_smart_crops.py --family $family
      # ... extract features ...
      # ... train ...
      # ... evaluate ...
  done
  ```

**Si 0.60 ≤ AJI < 0.68** ⚠️:
- Proche objectif (progrès significatif vs 0.5055)
- Possibilités:
  1. Tuning watershed parameters (beta, min_size)
  2. Vérifier HV magnitude et gradients visuellement
  3. Augmenter epochs training (50 → 60-70)

**Si AJI encore < 0.60** ❌:
- Problème plus profond à investiguer
- Actions de diagnostic:
  1. Vérifier visuellement HV maps générés (sont-ils cohérents?)
  2. Tester sur quelques samples si watershed fonctionne
  3. Comparer HV gradients AVANT/APRÈS fix rotation
  4. Vérifier que model predictions sont correctes

---

## 📊 MÉTRIQUES HISTORIQUES (RÉFÉRENCE)

### Famille Epidermal - Historique Performance

| Version | Samples | NP Dice | HV MSE | NT Acc | AJI | Statut |
|---------|---------|---------|--------|--------|-----|--------|
| V12 (resize 256→224) | 571 | 0.9542 | 0.2653 | 0.8857 | 0.57* | Baseline |
| V13 POC Multi-Crop | 2855 (5×) | 0.95 | 0.03 | 0.88 | 0.57* | Data leakage |
| V13-Hybrid | N/A | 0.7066 | N/A | N/A | N/A | Échec (-26%) |
| **V13 Smart Crops (bugs)** | **12,775 (25×)** | **0.7683** | **N/A** | **N/A** | **0.5055** | **Bugs #1/#2/#3** |
| **V13 Smart Crops (CIBLE)** | **12,775 (25×)** | **≥0.78** | **<0.30** | **>0.85** | **≥0.68** 🎯 | **Après fix** |

*Note: AJI 0.57 pour V12/V13 POC mesuré sur données d'entraînement (invalidé par data leakage)

### Corrélation Samples vs Performance (Autres Familles)

| Famille | Samples | HV MSE Historique | Note |
|---------|---------|-------------------|------|
| Glandular | 3,535 | 0.0106 | Excellent (>2000 samples) |
| Digestive | 2,274 | 0.0163 | Excellent (>2000 samples) |
| Urologic | 1,153 | 0.2812 | Dégradé (<2000 samples) |
| **Epidermal** | **571** | **0.2653** | **Dégradé (<2000 samples)** |
| Respiratory | 408 | 0.0500 | Surprise! (architecture ouverte) |

**Observation:** Seuil critique ~2000 samples pour HV MSE < 0.05

**Implication pour Epidermal:**
- Peu de samples (571) → HV MSE peut rester élevé (~0.27-0.30)
- **Mais:** LOCAL relabeling + rotation correcte devraient améliorer AJI significativement
- Objectif AJI ≥0.68 reste réaliste malgré HV MSE élevé

---

## 🔧 SCRIPTS PERTINENTS

### Scripts de Preprocessing
- `scripts/preprocessing/prepare_v13_smart_crops.py` — Génération crops + rotations ✅ FIXÉ
- `scripts/preprocessing/extract_features_from_fixed.py` — Features H-optimus-0 (à vérifier si existe)

### Scripts de Training
- `scripts/training/train_hovernet_family.py` — Entraînement HoVer-Net par famille

### Scripts d'Évaluation
- `scripts/evaluation/test_v13_smart_crops_aji.py` — Évaluation AJI/PQ/Dice

### Scripts de Validation
- `scripts/validation/validate_hv_rotation.py` — Validation rotation HV (si existe)

---

## 📝 LEÇONS APPRISES (DOCUMENTATION)

### 1. Renumbering partiel = Collision garantie
- Si renumbering SEULEMENT une partie → collision avec l'autre partie
- **Solution:** LOCAL relabeling complet avec scipy.ndimage.label()

### 2. HV rotation = Transformation vectorielle, PAS scalaire
- Rotation spatiale ≠ Rotation vectorielle
- **Formule correcte 90° CW:** (H, V) → (V, -H), **PAS** (-V, H)
- **Test unitaire:** Vecteur (1,0) droite → (0,-1) bas après 90° CW

### 3. LOCAL relabeling > HYBRID complexity
- Approche HYBRID: Complexe, prone to bugs, ne matche pas production
- Approche LOCAL: Simple, cohérence garantie, production-ready
- **Expert validation:** "Passe sur un relabeling local complet"

### 4. Production reality matche training
- Modèle en production verra seulement crops 224×224
- Entraîner avec contexte LOCAL = meilleure préparation
- Approche HYBRID créait gap entre training et production

### 5. Always verify rotation mathematics
- Tester transformations avec vecteurs unitaires
- Vérifier que directions finales sont correctes
- Bug #2 aurait pu être détecté plus tôt avec tests unitaires

---

## 📚 RÉFÉRENCES DOCUMENTATION

### Fichiers de Documentation
- `NEXT_STEPS_V13_SMART_CROPS.md` — Guide complet étapes de vérification
- `CLAUDE.md` (lignes 1413-1538) — Journal de développement entrée 2025-12-27
- `docs/V13_SMART_CROPS_STRATEGY.md` — Stratégie complète (si existe)

### Commits Pertinents
- `0c60c71` — "feat(v13-smart-crops): Implement LOCAL relabeling + Fix HV rotation mathematics (CRITICAL)"
- `15fb4e5` — "docs: Add journal entry for LOCAL relabeling + rotation fix (2025-12-27)"
- `b0e54b0` — Tentative fix partielle Bug #1 (avant adoption LOCAL relabeling)

### Expert Feedback (Citation)
> "Applique les corrections sur les rotations (H/V swap) et passe sur un relabeling local complet (Option 1 de tes devs, mais bien implémentée). Ton AJI devrait enfin franchir la barre des 0.68."

---

## ⚙️ CONFIGURATION TECHNIQUE

### Environnement
- **Conda env:** `cellvit`
- **Python:** 3.10
- **GPU:** RTX 4070 SUPER (12.9 GB VRAM)
- **PyTorch:** 2.6.0+cu124

### Dépendances Critiques
- `scipy` — Pour `scipy.ndimage.label()` (LOCAL relabeling)
- `numpy` — Opérations arrays
- `torch` — Training HoVer-Net
- `cv2` — Watershed post-processing

### Hardware Constraints
- **VRAM disponible:** 12 GB
- **Batch size max:** 16 (training) pour eviter OOM
- **Temps estimé training:** ~40 min (50 epochs, famille epidermal)

---

## ✅ CHECKLIST AVANT ENTRAÎNEMENT

Avant de lancer l'entraînement, **VÉRIFIER:**

- [ ] **Données régénérées** avec fix Bug #2 (rotation HV correcte)
  ```bash
  ls -lh data/family_data_v13_smart_crops/epidermal_*_v13_smart_crops.npz
  ```

- [ ] **IDs séquentiels vérifiés** (pas de collisions Bug #1)
  ```bash
  python -c "import numpy as np; data = np.load('data/family_data_v13_smart_crops/epidermal_val_v13_smart_crops.npz'); inst_map = data['inst_maps'][0]; unique_ids = np.unique(inst_map); unique_ids = unique_ids[unique_ids > 0]; expected_ids = np.arange(1, len(unique_ids) + 1); print('✅ OK' if np.array_equal(unique_ids, expected_ids) else '❌ ERREUR')"
  ```

- [ ] **Features RGB extraites** pour train + val
  ```bash
  ls -lh data/cache/family_data/epidermal_*_rgb_features_v13.npz
  ```

- [ ] **Environnement activé**
  ```bash
  conda activate cellvit
  ```

- [ ] **GPU disponible**
  ```bash
  nvidia-smi
  ```

- [ ] **Anciens checkpoints sauvegardés** (backup avant remplacement)
  ```bash
  mkdir -p models/checkpoints_backup
  cp models/checkpoints/hovernet_epidermal_best.pth models/checkpoints_backup/hovernet_epidermal_best_BEFORE_FIX.pth
  ```

---

## 🎯 OBJECTIFS SESSION

### Objectif Principal
**Ré-entraîner modèle HoVer-Net epidermal** avec données corrigées → **AJI ≥0.68**

### Objectifs Secondaires
1. Valider que fix Bug #2 (rotation HV) améliore effectivement performance
2. Valider que LOCAL relabeling (Bug #3) garantit cohérence ID ↔ HV
3. Documenter résultats dans CLAUDE.md
4. Décider si extension aux 4 autres familles est justifiée

### Métriques de Succès
- ✅ **AJI ≥0.68** (+35% vs 0.5055)
- ✅ **Over-seg ratio ~0.95-1.05×** (optimal)
- ✅ **Dice ≥0.76** (maintenu vs baseline)
- ✅ **Pas de collisions ID** dans inst_maps

---

## 📞 CONTACT EN CAS DE PROBLÈME

### Scripts Manquants
Si `extract_features_from_fixed.py` n'existe pas:
- Chercher script alternatif: `extract_features.py`, `extract_features_v13.py`
- Demander à l'utilisateur quel script utiliser

### Erreurs Runtime
Si erreurs durant training/évaluation:
- Vérifier stack trace complet
- Vérifier shapes des tensors (inst_maps, features, targets)
- Vérifier compatibilité versions (scipy, numpy, torch)

### Métriques Inattendues
Si AJI < 0.60 après ré-entraînement:
- Diagnostic HV maps visuellement
- Tester watershed post-processing sur quelques samples
- Comparer HV gradients AVANT/APRÈS fix

---

## 🚀 COMMANDE RAPIDE (TOUT-EN-UN)

```bash
# Activer environnement
conda activate cellvit

# Étape 1: Régénérer données (~5 min)
python scripts/preprocessing/prepare_v13_smart_crops.py --family epidermal

# Étape 2: Vérifier IDs séquentiels (~1 min)
python -c "import numpy as np; data = np.load('data/family_data_v13_smart_crops/epidermal_val_v13_smart_crops.npz'); inst_map = data['inst_maps'][0]; unique_ids = np.unique(inst_map); unique_ids = unique_ids[unique_ids > 0]; expected_ids = np.arange(1, len(unique_ids) + 1); print('Crop 0:'); print(f'  IDs: {unique_ids}'); print(f'  ✅ OK' if np.array_equal(unique_ids, expected_ids) else f'  ❌ ERREUR')"

# Étape 3: Extraire features (~10 min)
# À ADAPTER selon script disponible
python scripts/preprocessing/extract_features_from_fixed.py --family epidermal --split train
python scripts/preprocessing/extract_features_from_fixed.py --family epidermal --split val

# Étape 4: Ré-entraîner (~40 min)
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0

# Étape 5: Ré-évaluer (~5 min)
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --family epidermal \
    --n_samples 50
```

**Temps total estimé:** ~61 minutes

---

## 📌 RAPPELS IMPORTANTS

### ⚠️ CRITIQUE
- Le modèle actuel a été entraîné avec **rotation HV inversée** (Bug #2)
- **40% des données** (rotations 90°/270°) avaient gradients inversés
- → **RÉ-ENTRAÎNEMENT OBLIGATOIRE** pour exploiter fix Bug #2

### ✅ FIX APPLIQUÉS (commit 0c60c71)
1. **Bug #1:** ID collision → LOCAL relabeling avec scipy.ndimage.label()
2. **Bug #2:** Rotation HV → H'=V, V'=-H (correct pour 90° CW)
3. **Bug #3:** HYBRID complexity → LOCAL relabeling simplifié

### 🎯 OBJECTIF
**AJI 0.5055 → ≥0.68** (+35% amélioration attendue)

---

**Dernière mise à jour:** 2025-12-27
**Commits de référence:** 0c60c71, 15fb4e5
**Statut:** ✅ Code fixé — ⏳ Modèle à ré-entraîner
