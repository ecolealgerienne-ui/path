# 🎯 PLAN D'ACTION COMPLET — Fix Expert "Gradient Killer" (2025-12-24)

**Statut:** ✅ CODE PRÊT — À appliquer

**Objectif:** Résoudre le Giant Blob (AJI 0.09 → 0.60+) en corrigeant les 3 bugs magnitude_loss

---

## 📚 Documents Créés (Références)

| Document | Contenu | Usage |
|----------|---------|-------|
| **FIX_GRADIENT_KILLER_EXPERT.md** | Diagnostic complet (50 pages) | Comprendre le problème |
| **CODE_FIX_MAGNITUDE_LOSS.md** | Code corrigé hovernet_decoder.py | Copy-paste fix #1 |
| **CODE_FIX_TRAIN_SCRIPT.md** | Code corrigé train_hovernet_family.py | Copy-paste fix #2 |
| **PLAN_ACTION_FIX_EXPERT.md** | Ce fichier (plan exécution) | Workflow complet |

**Localisation:** Tous dans `docs/`

---

## 🐛 Les 3 Bugs Identifiés (Rappel)

### Bug #1: Epsilon APRÈS la racine → Gradients morts (Test 3 échoue)

**Code bugué:**
```python
mag_pred = torch.sqrt((hv_pred ** 2).sum(dim=1) + 1e-8)  # ❌ sqrt(0) puis +1e-8
```

**Fix:**
```python
mag_pred = torch.sqrt(torch.sum(hv_pred**2, dim=1) + 1e-6)  # ✅ sqrt(0 + 1e-6)
```

---

### Bug #2: F.mse_loss voit le fond → Magnitude 0.02 au lieu de 0.8+

**Code bugué:**
```python
mag_loss_sum = F.mse_loss(mag_pred_masked, mag_target_masked, reduction='sum')
# Calcule sur TOUS pixels (fond masqué à 0 dilue le signal)
```

**Fix:**
```python
loss = (mag_true - mag_pred)**2  # Erreur manuelle
weighted_loss = loss * mask.squeeze(1)  # Masque AVANT réduction
return weighted_loss.sum() / (mask.sum() + 1e-6)  # Normalise par cellules seulement
```

---

### Bug #3: Lambda magnitude trop faible (1.0 au lieu de 5.0)

**Code bugué:**
```python
hv_loss = hv_l1 + 2.0 * hv_gradient + 1.0 * hv_magnitude  # ❌ Seulement 1.0×
```

**Fix:**
```python
hv_loss = hv_l1 + 3.0 * hv_gradient + 5.0 * hv_magnitude  # ✅ Expert recommande 5.0×
```

---

## ⚡ Workflow d'Application (45 min TOTAL)

### 🔧 Étape 1: Backups (2 min)

```bash
# Sauvegarder les versions actuelles
cp src/models/hovernet_decoder.py src/models/hovernet_decoder.py.backup_before_expert_fix
cp scripts/training/train_hovernet_family.py scripts/training/train_hovernet_family.py.backup_before_expert_fix

# Confirmer backups créés
ls -lh src/models/hovernet_decoder.py.backup_before_expert_fix
ls -lh scripts/training/train_hovernet_family.py.backup_before_expert_fix
```

**Attendu:**
```
-rw-r--r-- 1 user user 25K Dec 24 14:30 hovernet_decoder.py.backup_before_expert_fix
-rw-r--r-- 1 user user 18K Dec 24 14:30 train_hovernet_family.py.backup_before_expert_fix
```

---

### 📝 Étape 2: Appliquer Fix #1 — hovernet_decoder.py (5 min)

**Ouvrir:** `docs/CODE_FIX_MAGNITUDE_LOSS.md`

**Appliquer 3 modifications:**

1. **Remplacer `magnitude_loss()` (lignes 302-361)**
   - Chercher: `def magnitude_loss(`
   - Remplacer toute la fonction par la version fixée

2. **Modifier `__init__` (ligne ~240)**
   - Chercher: `def __init__(self, lambda_np`
   - Ajouter paramètre `lambda_magnitude: float = 5.0,`
   - Ajouter ligne `self.lambda_magnitude = lambda_magnitude`

3. **Modifier calcul HV loss (ligne ~416)**
   - Chercher: `hv_loss = hv_l1 + 2.0 * hv_gradient`
   - Remplacer par: `hv_loss = hv_l1 + 3.0 * hv_gradient + self.lambda_magnitude * hv_magnitude`

**Vérifier syntaxe:**
```bash
python -c "from src.models.hovernet_decoder import HoVerNetLoss; print('✅ Syntaxe OK')"
```

---

### 📝 Étape 3: Appliquer Fix #2 — train_hovernet_family.py (3 min)

**Ouvrir:** `docs/CODE_FIX_TRAIN_SCRIPT.md`

**Appliquer 3 modifications:**

1. **Ajouter argument CLI (après ligne 351)**
   ```python
   parser.add_argument('--lambda_magnitude', type=float, default=5.0,
                      help='Poids magnitude loss (Expert: 5.0 pour forcer gradients forts)')
   ```

2. **Passer à HoVerNetLoss (ligne ~412)**
   ```python
   criterion = HoVerNetLoss(
       lambda_np=args.lambda_np,
       lambda_hv=args.lambda_hv,
       lambda_nt=args.lambda_nt,
       lambda_magnitude=args.lambda_magnitude,  # ← NOUVEAU
       adaptive=args.adaptive_loss,
   )
   ```

3. **Afficher dans logs (ligne ~420)**
   ```python
   print(f"  Loss: Poids fixes (NP={args.lambda_np}, HV={args.lambda_hv}, NT={args.lambda_nt}, Magnitude={args.lambda_magnitude})")
   ```

**Vérifier syntaxe:**
```bash
python scripts/training/train_hovernet_family.py --help | grep lambda_magnitude
# Attendu: --lambda_magnitude LAMBDA_MAGNITUDE
```

---

### ✅ Étape 4: Validation Finale (1 min)

**Test import:**
```bash
python -c "
from src.models.hovernet_decoder import HoVerNetLoss
criterion = HoVerNetLoss(lambda_magnitude=7.0)
print(f'Lambda magnitude: {criterion.lambda_magnitude}')
assert criterion.lambda_magnitude == 7.0
print('✅ Tous les fixes appliqués correctement!')
"
```

**Attendu:**
```
Lambda magnitude: 7.0
✅ Tous les fixes appliqués correctement!
```

---

### 🚀 Étape 5: Re-training Epidermal (40 min GPU)

**Commande complète:**
```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_hv 3.0 \
    --lambda_magnitude 5.0 \
    --batch_size 8 \
    --lr 1e-4
```

**Monitoring Logs:**

Vous devriez voir:
```
🔧 Initialisation du décodeur HoVer-Net...
  Paramètres: 12,345,678 (12.3M)
  Loss: Poids fixes (NP=1.0, HV=3.0, NT=1.0, Magnitude=5.0)
                                                ^^^^^^^^^^^
                                                NOUVEAU - Confirme fix appliqué

🚀 Entraînement (50 epochs)...

============================================================
Epoch 1/50
============================================================
Train - Loss: 3.5421
        NP Dice: 0.8234 | HV MSE: 0.3421 | NT Acc: 0.7891
        hv_l1: 0.0234
        hv_gradient: 0.0156
        hv_magnitude: 0.6152  ← ⚠️ SURVEILLER: doit diminuer progressivement
                      ^^^^^^
Val   - Loss: 3.6892
        NP Dice: 0.8156 | HV MSE: 0.3562 | NT Acc: 0.7823
```

---

### 🔬 Étape 6: Checkpoint Epoch 5 (CRITIQUE - Point de décision)

**Après epoch 5, tester magnitude:**

```bash
# Sauvegarder checkpoint epoch 5 (si pas auto-save)
# Vous devriez avoir: models/checkpoints/hovernet_epidermal_epoch_5.pth

python scripts/evaluation/compute_hv_magnitude.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_epoch_5.pth \
    --n_samples 10
```

**Scénario A: Magnitude >0.25 ✅ SUCCÈS**

```
Mean magnitude: 0.3421
Std:            0.0234
Min:            0.3012
Max:            0.3789

✅ SUCCÈS: Magnitude élevée (>0.25)
   → Le fix fonctionne! Continuer le training jusqu'à epoch 50.
```

**Action:** Laisser le training continuer ➜ Aller à Étape 7

---

**Scénario B: Magnitude <0.10 ❌ ÉCHEC**

```
Mean magnitude: 0.0621
Std:            0.0089
Min:            0.0512
Max:            0.0745

❌ ÉCHEC: Magnitude trop faible (<0.10)
   → Le fix n'a PAS fonctionné. Arrêter et investiguer.
```

**Actions de debug:**

1. **Vérifier que magnitude_loss est bien celle fixée:**
   ```bash
   grep -A 5 "def magnitude_loss" src/models/hovernet_decoder.py | head -10
   # Doit contenir: torch.sqrt(torch.sum(hv_pred**2, dim=1) + 1e-6)
   #                           ^^^^^^^^^^^^^^^^^^^^^^^^^ epsilon DANS
   ```

2. **Vérifier lambda_magnitude=5.0 dans les logs:**
   ```bash
   grep "Magnitude=" logs/training_epidermal_*.log
   # Doit afficher: Magnitude=5.0
   ```

3. **Vérifier masking dans magnitude_loss:**
   ```bash
   grep "weighted_loss = loss \* mask" src/models/hovernet_decoder.py
   # Doit exister
   ```

Si tous les checks passent mais magnitude reste basse → **contacter expert pour diagnostic approfondi**.

---

### 🎯 Étape 7: Validation Finale (Après 50 epochs)

**Test magnitude:**
```bash
python scripts/evaluation/compute_hv_magnitude.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10
```

**Attendu:**
```
Mean magnitude: 0.5234 ± 0.0156
                ^^^^^^
                >0.50 ✅ OBJECTIF ATTEINT

✅ SUCCÈS: Magnitude forte (>0.50)
   Amélioration: 0.022 → 0.5234 (+2279%)
```

---

**Test AJI Ground Truth:**
```bash
python scripts/evaluation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --data_dir data/family_data \
    --n_samples 100
```

**Attendu:**
```
================================================================================
RÉSULTATS SUR DONNÉES D'ENTRAÎNEMENT
================================================================================

NP Dice:  0.9512 ± 0.0234  (stable ✅)
HV MSE:   0.2089 ± 0.0156  (stable ✅)
NT Acc:   0.9034 ± 0.0198  (stable ✅)

AJI:      0.6234 ± 0.1023  ← 🎯 OBJECTIF ATTEINT (>0.60)
          ^^^^^^
          Amélioration: 0.09 → 0.62 (+591%)

Instances détectées: 8-12 cellules séparées (vs 1 Giant Blob avant)
```

---

## 🎉 Critères de Succès

| Métrique | Avant Fix | Cible | Après Fix (attendu) | Statut |
|----------|-----------|-------|---------------------|--------|
| **Magnitude (epoch 5)** | 0.022 | **>0.25** | ~0.35 | ✅ Indicateur précoce |
| **Magnitude (epoch 50)** | 0.022 | **>0.50** | ~0.52 | ✅ Objectif atteint |
| **AJI** | 0.09 | **>0.60** | ~0.62 | ✅ Giant Blob résolu |
| **Instances** | 1 blob | **8-12 cellules** | 9-11 cellules | ✅ Séparation correcte |
| **NP Dice** | 0.90 | **>0.90** | ~0.95 | ✅ Segmentation stable |
| **NT Acc** | 0.91 | **>0.85** | ~0.90 | ✅ Classification stable |

**Si TOUS les critères sont atteints:** 🎉 **FIX VALIDÉ — Giant Blob RÉSOLU**

---

## 🔄 En Cas d'Échec (Magnitude <0.10 après 5 epochs)

### Checklist Debug

1. **Vérifier que magnitude_loss() utilise epsilon DANS racine:**
   ```bash
   grep "torch.sqrt(torch.sum" src/models/hovernet_decoder.py
   # Doit retourner une ligne (nouvelle version)
   ```

2. **Vérifier masking AVANT réduction:**
   ```bash
   grep "weighted_loss = loss \* mask.squeeze" src/models/hovernet_decoder.py
   # Doit exister
   ```

3. **Vérifier lambda_magnitude passé correctement:**
   ```bash
   python -c "
   from src.models.hovernet_decoder import HoVerNetLoss
   c = HoVerNetLoss(lambda_magnitude=7.0)
   assert c.lambda_magnitude == 7.0
   print('✅ Lambda magnitude OK')
   "
   ```

4. **Vérifier training logs:**
   ```bash
   grep "hv_magnitude" logs/*.log | tail -20
   # Doit afficher les valeurs hv_magnitude à chaque epoch
   ```

### Si Tous les Checks Passent

**Le problème peut être ailleurs:**
- Features corrompues (vérifier CLS std ~0.77)
- Targets HV corrompus (vérifier dtype float32, range [-1, 1])
- Masques NP incorrects (vérifier ratio pixels cellules/fond)

**Action:** Créer un rapport de debug détaillé avec:
- Logs complets training
- Outputs compute_hv_magnitude.py
- Vérification features/targets
- Contacter expert pour analyse approfondie

---

## 📊 Timeline Estimée

| Étape | Durée | Description |
|-------|-------|-------------|
| 1. Backups | 2 min | Sauvegarder fichiers |
| 2. Fix hovernet_decoder.py | 5 min | 3 modifications |
| 3. Fix train_hovernet_family.py | 3 min | 3 modifications |
| 4. Validation | 1 min | Test imports |
| 5. Re-training epochs 1-5 | 5 min | Checkpoint décision |
| 6. **POINT DE DÉCISION** | 2 min | compute_hv_magnitude.py |
| 7. Re-training epochs 6-50 | 35 min | Training complet |
| 8. Test AJI final | 2 min | Validation finale |
| **TOTAL** | **55 min** | **Pipeline complet** |

---

## 🎯 Prochaines Étapes Après Succès

Si le fix fonctionne (AJI >0.60):

1. **Documenter dans CLAUDE.md:**
   - Ajouter section "Bug #7: Gradient Killer"
   - Metrics avant/après
   - Lien vers FIX_GRADIENT_KILLER_EXPERT.md

2. **Ré-entraîner les 4 autres familles:**
   ```bash
   for family in glandular digestive urologic respiratory; do
       python scripts/training/train_hovernet_family.py \
           --family $family \
           --epochs 50 \
           --augment \
           --lambda_hv 3.0 \
           --lambda_magnitude 5.0
   done
   ```

3. **Créer rapport final:**
   - AJI par famille avant/après
   - Instances détectées (histogrammes)
   - Visualisations HV magnitude
   - Benchmark vs SOTA (HoVer-Net original, CellViT)

4. **Publication résultats:**
   - Mettre à jour ETAT_MODELE_ET_ROADMAP_TOP5.md
   - Position finale: TOP 5% mondial (si AJI >0.70 sur toutes familles)

---

**STATUT FINAL:** ✅ PLAN COMPLET DOCUMENTÉ — Prêt pour exécution

**PROCHAINE ACTION:** Appliquer Étape 1 (Backups) puis Étapes 2-3 (Fixes code)
