# Guide d'Utilisation des Scripts (Session 2025-12-24)

> **Contexte:** Suite au diagnostic Bug #4 (Data Mismatch Temporel), plusieurs scripts ont été créés pour faciliter la récupération et éviter les tests inutiles dans l'environnement Claude.

---

## 📋 Scripts Créés

### 1. `scripts/utils/inspect_environment.py` 🆕

**Objectif:** Collecter TOUTES les informations d'environnement pour que Claude puisse analyser sans jamais tester lui-même.

**Usage:**
```bash
python scripts/utils/inspect_environment.py > environment_report.txt
```

**Ce qu'il fait:**
- ✅ Informations système (OS, Python, GPU)
- ✅ État CUDA/nvidia-smi
- ✅ Packages Python critiques (torch, timm, etc.)
- ✅ Localisation données PanNuke
- ✅ État des caches (features, family_data)
- ✅ Checkpoints modèles disponibles
- ✅ Structure du projet
- ✅ État Git
- ✅ Tests imports modules custom

**Sortie:** Rapport texte complet à copier/coller pour Claude

**Quand l'utiliser:**
- Au début d'une nouvelle session
- Après avoir déplacé/modifié des données
- Pour diagnostiquer un problème d'environnement

---

### 2. `scripts/validation/verify_spatial_alignment.py` 🆕

**Objectif:** Vérification CRITIQUE de l'alignement pixel-perfect entre HV targets et images (GO/NO-GO avant re-training).

**Usage:**
```bash
python scripts/validation/verify_spatial_alignment.py \
    --family glandular \
    --n_samples 5 \
    --output_dir results/spatial_alignment
```

**Ce qu'il fait:**
1. Charge images + HV targets + NP targets
2. Calcule gradients HV (magnitude + direction)
3. Superpose sur l'image avec visualisations:
   - Image + contours NP
   - Image + vecteurs HV (quiver plot)
   - HV magnitude + contours
4. Calcule distance moyenne entre pics HV et contours réels
5. **Verdict GO/NO-GO:**
   - ✅ **GO**: distance ≤ 2 px → Training peut être lancé
   - ⚠️ **CAUTION**: distance ≤ 5 px → Vérifier visuellement
   - ❌ **NO-GO**: distance > 5 px → **NE PAS LANCER LE TRAINING**

**Arguments:**
- `--family`: Famille à vérifier (glandular, digestive, urologic, epidermal, respiratory)
- `--n_samples`: Nombre d'échantillons à vérifier (défaut: 5)
- `--output_dir`: Répertoire pour les visualisations (défaut: results/spatial_alignment)

**Exit codes:**
- `0`: GO (parfait)
- `1`: CAUTION (acceptable)
- `2`: NO-GO (problématique)
- `3`: Erreur d'exécution

**Quand l'utiliser:**
- ✅ **OBLIGATOIRE** après régénération des features (Étape 3 du plan de sauvetage)
- ✅ Avant tout re-training
- ✅ Si suspicion de Bug #4 (Data Mismatch Temporel)

**Exemple d'output:**
```
==================================================
VÉRIFICATION PIXEL-PERFECT DE L'ALIGNEMENT SPATIAL
==================================================
Famille: glandular
Échantillons: 5

✅ Données chargées depuis: data/cache/family_data/glandular_data_FIXED.npz
   Images: (3391, 256, 256, 3)
   HV targets: (3391, 2, 256, 256), dtype=float32, range=[-1.000, 1.000]
   NP targets: (3391, 256, 256), dtype=int64

Vérification échantillons:
----------------------------------------
  [1/5] Sample 42: distance=1.23 px ✅
  [2/5] Sample 158: distance=1.87 px ✅
  [3/5] Sample 891: distance=2.45 px ⚠️
  [4/5] Sample 1542: distance=1.05 px ✅
  [5/5] Sample 2983: distance=1.68 px ✅

==================================================
RÉSULTATS
==================================================
Distance moyenne: 1.66 pixels
Distance min:     1.05 pixels
Distance max:     2.45 pixels

✅ VERDICT: GO
   Alignement EXCELLENT - Training peut être lancé

==================================================
Visualisations sauvées dans: results/spatial_alignment
==================================================
```

**Visualisations générées:**

Chaque échantillon produit une image avec 6 subplots:

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Image Originale │  NP Target      │ HV Magnitude    │
│                 │  (Noyaux)       │ (Frontières)    │
├─────────────────┼─────────────────┼─────────────────┤
│ Image +         │ Image +         │ HV Magnitude +  │
│ Contours NP     │ Vecteurs HV     │ Contours        │
│ (vert)          │ (flèches jaunes)│ (cyan)          │
└─────────────────┴─────────────────┴─────────────────┘
```

**Critères de validation:**
- ✅ Les flèches HV doivent pointer VERS les centres des noyaux
- ✅ Les pics de magnitude HV doivent coïncider avec les contours verts
- ❌ Si décalage visible > 2-3 pixels → **NO-GO**

---

## 🚀 Workflow Complet de Récupération (Bug #4)

Basé sur `docs/ETAT_DES_LIEUX_2025-12-23.md`:

### Étape 1: Inspection Environnement (5 min)

```bash
# Générer rapport complet
python scripts/utils/inspect_environment.py > environment_report.txt

# Envoyer à Claude
cat environment_report.txt
# (Copier/coller dans la conversation)
```

### Étape 2: Purge Cache Features (5 min)

```bash
# Sauvegarder anciennes features
mv data/cache/pannuke_features data/cache/pannuke_features_OLD_CORRUPTED_20251224

# Créer nouveau répertoire
mkdir -p data/cache/pannuke_features
```

### Étape 3: Régénération Features Fold 0 (15-20 min)

```bash
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 0 \
    --batch_size 8 \
    --chunk_size 300
```

**Vérification:**
```bash
python scripts/validation/verify_features.py \
    --features_dir data/cache/pannuke_features

# Attendu:
# ✅ CLS std: 0.7680 ± 0.005 (dans [0.70, 0.90])
```

### Étape 4: Vérification Pixel-Perfect ⚠️ CRITIQUE (5 min)

```bash
python scripts/validation/verify_spatial_alignment.py \
    --family glandular \
    --n_samples 10 \
    --output_dir results/spatial_alignment_check

# Vérifier les visualisations dans results/spatial_alignment_check/
```

**⚠️ POINT DE DÉCISION GO/NO-GO:**

- **Si GO (distance ≤ 2 px):**
  ```bash
  echo "✅ Alignement OK - Procéder au re-training"
  ```

- **Si NO-GO (distance > 5 px):**
  ```bash
  echo "❌ Alignement KO - NE PAS RE-ENTRAÎNER"
  echo "   → Vérifier preprocessing"
  echo "   → Consulter visualisations"
  echo "   → Demander aide à Claude"
  ```

### Étape 5: Re-training (SI GO) (30-40 min)

```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_np 1.0 \
    --lambda_hv 2.0 \
    --lambda_nt 1.0
```

### Étape 6: Test de Vérité Final (5 min)

```bash
python scripts/evaluation/test_crop_truth.py \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 50
```

**Résultat attendu (Expert):**
```
✅ Dice:  > 0.95  (stable)
✅ AJI:   > 0.60  (BOND de 0.06 → 0.60+, gain +900%)
✅ PQ:    > 0.65  (restauré)

Instances: ~30 pred vs ~32 GT (match)
```

---

## 📊 Checklist de Validation

Avant de commencer:
- [ ] Lancer `inspect_environment.py` et vérifier que:
  - [ ] GPU disponible
  - [ ] PanNuke trouvé
  - [ ] PyTorch + CUDA OK
  - [ ] Modules custom importent sans erreur

Après régénération features:
- [ ] `verify_features.py` → CLS std dans [0.70, 0.90]
- [ ] `verify_spatial_alignment.py` → Verdict GO

Après re-training:
- [ ] Dice > 0.95
- [ ] HV MSE < 0.05
- [ ] NT Acc > 0.85
- [ ] **AJI > 0.60** ← Objectif principal

---

## 🚫 Rappel Consignes Claude

Claude NE DOIT JAMAIS:
- ❌ Exécuter `python scripts/training/...`
- ❌ Exécuter `python scripts/evaluation/...`
- ❌ Essayer de tester quoi que ce soit localement

Claude DOIT:
- ✅ Créer des scripts que VOUS lancez
- ✅ Analyser les résultats que VOUS lui fournissez
- ✅ Proposer des corrections basées sur les outputs

---

## 📞 Contact Claude

Pour toute question ou problème:

1. **Avant d'exécuter:** Demandez à Claude de vérifier le script
2. **Après exécution:** Copiez/collez l'output complet à Claude
3. **En cas d'erreur:** Copiez le traceback complet

Claude peut analyser n'importe quel output, mais ne peut pas tester lui-même.

---

**Dernière mise à jour:** 2025-12-24
**Auteur:** Claude (session de préparation Option B)
**Statut:** Scripts créés et documentés - Prêts pour exécution utilisateur
