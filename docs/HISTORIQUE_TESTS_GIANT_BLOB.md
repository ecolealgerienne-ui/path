# Historique Complet des Tests — Giant Blob (AJI 0.09)

**Date début investigation:** 2025-12-24
**Problème:** AJI 0.09 vs objectif 0.60+ (1 instance au lieu de 8)
**Statut:** Investigation en cours

---

## 📊 Tests Déjà Effectués (Chronologique)

### ✅ Test 1: HV Scaling (×1 à ×50) — NÉGATIF

**Date:** 2025-12-24 (matin)
**Script:** `scripts/evaluation/test_hv_scaling.py`
**Objectif:** Déterminer si amplifier HV améliore séparation instances

**Méthode:**
```python
# Multiplier HV predictions par facteur (1x, 5x, 10x, 20x, 50x)
hv_scaled = hv_pred * scale_factor
energy = np.sqrt(hv_scaled[0]**2 + hv_scaled[1]**2)
instance_map = watershed(-energy, markers, mask=binary_mask)
```

**Résultats:**
| Scaling | Energy Range | Energy Mean | Peaks Found | AJI Mean |
|---------|--------------|-------------|-------------|----------|
| 1.0x | [0.0019, 0.0209] | 0.0095 | 137 | 0.0905 |
| 5.0x | [0.0093, 0.1043] | 0.0476 | 137 | 0.0905 |
| 10.0x | [0.0186, 0.2086] | 0.0953 | 137 | 0.0905 |
| 20.0x | [0.0371, 0.4171] | 0.1905 | 137 | 0.0905 |
| 50.0x | [0.0928, 1.0428] | 0.4763 | 137 | 0.0905 |

**Conclusion:** ❌ Scaling n'améliore PAS l'AJI (reste à 0.0905)

**Ce qu'on a appris:**
1. Le modèle détecte CORRECTEMENT les 137 centres de cellules (peaks constants)
2. Le problème n'est PAS juste une amplitude faible
3. Le problème vient APRÈS la détection des peaks (watershed ou GT comparison)

**Hypothèses éliminées:**
- ❌ "Il suffit d'amplifier HV pour améliorer AJI"
- ❌ "Les peaks ne sont pas détectés"

**Fichiers créés:**
- `docs/ANALYSE_TEST_SCALING_NEGATIF.md`

---

### ✅ Test 2: Visualisation Instance Maps — GIANT BLOB CONFIRMÉ

**Date:** 2025-12-24 (matin)
**Script:** `scripts/evaluation/visualize_instance_maps.py`
**Objectif:** Diagnostic visuel pour confirmer Giant Blob vs ID Mismatch

**Méthode:**
```python
# Resize 224 → 256 avec INTER_NEAREST (recommandation expert)
np_pred_256 = cv2.resize(np_pred, (256, 256), interpolation=cv2.INTER_NEAREST)
hv_pred_256 = np.stack([
    cv2.resize(hv_pred[0], (256, 256), interpolation=cv2.INTER_NEAREST),
    cv2.resize(hv_pred[1], (256, 256), interpolation=cv2.INTER_NEAREST)
])

# Post-processing watershed
inst_pred = post_process_hv(np_pred_256, hv_pred_256)

# Visualisation côte à côte
fig, axes = plt.subplots(2, 3)
axes[0, 2].imshow(inst_pred, cmap=colormap)  # Instances PRED
axes[1, 2].imshow(inst_target, cmap=colormap)  # Instances GT
```

**Résultats (Échantillon 9):**
```
🔍 Analyse échantillon 9 (index 8)

Instances prédites: 1
Instances GT:       8

HV magnitude PRED: [0.0022, 0.0221]
HV magnitude GT:   [0.0000, 0.9992]

❌ GIANT BLOB DÉTECTÉ!
   Ratio magnitude: 0.022 / 0.5 = 4.4% (50× trop faible)
```

**Visualisation générée:**
- `results/diagnostic_instance_maps_sample9.png`
- Colonne 1: H&E brut, NP masks
- Colonne 2: HV magnitude maps
- Colonne 3: **Instance maps (1 couleur PRED vs 8 couleurs GT)**

**Conclusion:** ✅ Giant Blob confirmé (1 instance violette géante)

**Ce qu'on a appris:**
1. Le watershed crée effectivement 1 instance au lieu de 8
2. HV magnitude 50× trop faible (0.022 vs >0.5 attendu)
3. Les 137 peaks sont détectés mais ne séparent pas les instances

**Hypothèses éliminées:**
- ❌ "C'est un problème de resize (INTER_LINEAR détruisant IDs)"
- ❌ "C'est un ID Mismatch (décalage spatial)"

**Hypothèses confirmées:**
- ✅ Giant Blob (fusion complète en 1 instance)
- ✅ HV magnitude trop faible

**Fichiers créés:**
- `scripts/evaluation/visualize_instance_maps.py`
- `results/diagnostic_instance_maps_sample9.png`

---

### ✅ Test 3: Vérification Architecture (Code Review) — CORRECTE

**Date:** 2025-12-24 (après-midi)
**Méthode:** Lecture manuelle des fichiers source
**Objectif:** Vérifier que Tanh et Sobel sont présents dans le code

**Fichiers vérifiés:**

#### 3.1. Tanh HV Branch
**Fichier:** `src/models/hovernet_decoder.py` (lignes 118-121)
```python
self.hv_head = nn.Sequential(
    DecoderHead(64, 2),
    nn.Tanh()  # ✅ PRÉSENT - Force HV dans [-1, 1]
)
```

**Statut:** ✅ Tanh présent et actif

---

#### 3.2. Sobel Gradient Loss
**Fichier:** `src/models/hovernet_decoder.py` (lignes 244-280)
```python
def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
    """
    MSGE avec opérateur Sobel pour signal amplifié.
    """
    # Opérateur Sobel (3×3 kernel)
    sobel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], ...)
    sobel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], ...)

    # Convolution avec padding
    pred_grad_h = F.conv2d(pred_reshaped, sobel_h, padding=1)
    pred_grad_v = F.conv2d(pred_reshaped, sobel_v, padding=1)
    # ... masking and MSE ...
```

**Ligne 347 - Usage:**
```python
hv_gradient = self.gradient_loss(hv_pred, hv_target, mask=mask)
hv_loss = hv_l1 + 2.0 * hv_gradient  # Poids 2.0× pour gradients
```

**Statut:** ✅ Sobel gradient loss implémenté et actif (poids 2.0)

---

#### 3.3. Données v8 (Vraies Instances)
**Fichier:** `scripts/preprocessing/prepare_family_data_FIXED_v8.py` (lignes 190-213)
```python
def extract_instance_map(mask: np.ndarray) -> np.ndarray:
    """
    VRAIES INSTANCES (v8): Utilise IDs des canaux 1-4 directement
    (PAS connectedComponents qui fusionne cellules touchantes)
    """
    inst_map = np.zeros((256, 256), dtype=np.int32)
    instance_counter = 1

    # Canaux 1-4: instances déjà annotées (VRAIES instances PanNuke)
    for c in range(1, 5):
        class_instances = mask[:, :, c]
        inst_ids = np.unique(class_instances)
        inst_ids = inst_ids[inst_ids > 0]

        for inst_id in inst_ids:
            inst_mask = class_instances == inst_id
            inst_map[inst_mask] = instance_counter
            instance_counter += 1
```

**Statut:** ✅ Données v8 utilisent vraies instances PanNuke (pas Bug #3)

---

**Conclusion Test 3:** ✅ Architecture complète et correcte

**Ce qu'on a appris:**
1. Tout le code nécessaire est présent
2. Tanh force bien HV dans [-1, 1]
3. Sobel gradient loss actif avec poids 2.0
4. Données v8 utilisent vraies instances

**Hypothèses éliminées:**
- ❌ "Tanh manquant dans le code"
- ❌ "Sobel pas implémenté"
- ❌ "Bug #3 (connectedComponents fusionne cellules)"

**Nouvelle hypothèse émergente:**
- ⚠️ "Checkpoint entraîné AVANT ajout Sobel/Tanh dans le code"

---

### ✅ Test 4: Revue Documentation — SOBEL FIX DATÉ 2025-12-23

**Date:** 2025-12-24 (après-midi)
**Méthode:** Lecture docs existantes
**Objectif:** Vérifier si problème déjà documenté

**Documents consultés:**

#### 4.1. FIX_SOBEL_GRADIENT_LOSS.md
**Contenu clé:**
```markdown
Date: 2025-12-23
Problème: AJI 0.07 vs cible 0.80
Cause racine: Signal gradient_loss trop faible → HV maps "douces"

Solution: Opérateur Sobel (3×3 kernel)
- Différences finies: signal ~0.01 (faible)
- Sobel: signal ~0.04 (4× plus fort)
```

**Statut:** ✅ Problème EXACT déjà documenté et résolu le 2025-12-23

---

#### 4.2. ARCHITECTURE_HV_ACTIVATION.md
**Date:** 2025-12-21
**Décision initiale:** Garder architecture SANS Tanh (tests empiriques OK)

**Note importante:**
> Cette décision a été CHANGÉE plus tard (ligne 118-121 hovernet_decoder.py AJOUTE Tanh)

**Statut:** ⚠️ Décision révisée, Tanh ajouté ultérieurement

---

#### 4.3. GIANT_BLOB_RESOLUTION_PLAN.md (créé aujourd'hui)
**Hypothèses formulées:**
1. Modèle entraîné AVANT Sobel fix (70% probabilité)
2. Watershed params trop conservateurs (15%)
3. Gaussian smoothing trop agressif (15%)

**Statut:** ✅ Hypothèse #1 la plus probable

---

**Conclusion Test 4:** ✅ Documentation confirme hypothèse temporelle

**Ce qu'on a appris:**
1. Sobel fix implémenté le 2025-12-23
2. Tanh ajouté après décision initiale (post 2025-12-21)
3. Problème actuel identique à celui documenté dans FIX_SOBEL_GRADIENT_LOSS.md

**Hypothèse renforcée:**
- ✅ **"Checkpoint entraîné AVANT 2025-12-23"** (70% → 90% probabilité)

---

## 🔍 Tests EN ATTENTE (Non encore effectués)

### ⏳ Test 5: Vérification HV Targets .npz

**Script créé:** `scripts/validation/verify_hv_targets_npz.py`
**Commande:**
```bash
conda activate cellvit
python scripts/validation/verify_hv_targets_npz.py --family epidermal
```

**Objectif:** Vérifier que targets stockés sont bien float32 [-1, 1]

**Checks automatiques:**
1. Dtype (doit être float32, pas int8)
2. Range (doit être [-1.0, 1.0])
3. Symétrie (mean ≈ 0.0)
4. Variance (std dans [0.3, 0.7])

**Scénarios possibles:**

**A. ✅ Targets corrects:**
```
✅ Dtype: float32
✅ Range: [-1.000, 1.000]
✅ Mean: 0.0006 (centré)
✅ Std: 0.4567 (bonne dynamique)
```
→ Confirme problème vient du checkpoint, pas des données
→ Passer à Test 6

**B. ❌ Targets incorrects (int8):**
```
❌ Dtype: int8
❌ Range: [-127, 127]
```
→ Bug normalization (données v8 corrompues)
→ Régénérer v9 AVANT ré-entraînement

**C. ⚠️ Variance trop faible:**
```
✅ Dtype: float32
✅ Range: [-1.0, 1.0]
⚠️ Std: 0.15 (attendu: >0.3)
```
→ Gaussian smoothing trop agressif (sigma=0.5)
→ Régénérer v9 sans smoothing

**Statut:** ⏳ NON EXÉCUTÉ (environnement Claude incompatible)

---

### ⏳ Test 6: Vérification Date Checkpoint

**Commande:**
```bash
find models/checkpoints -name "hovernet_epidermal_best.pth" -exec ls -l {} \;
```

**Objectif:** Comparer date création checkpoint vs date Sobel fix (2025-12-23)

**Scénarios:**

**A. Date < 2025-12-23:**
→ ✅ Confirme "mismatch version logique"
→ Ré-entraînement résoudra le problème
→ Passer à Test 7 (ré-entraînement)

**B. Date ≥ 2025-12-23:**
→ ⚠️ Checkpoint entraîné AVEC Sobel, mais performances mauvaises
→ Investiguer logs training (Test 6b)

**Statut:** ⏳ NON EXÉCUTÉ (tentative échouée: fichier introuvable)

---

### ⏳ Test 6b: Vérification Logs Training

**Fichier:** `results/training_hovernet_epidermal.log` (ou équivalent)

**Commande:**
```bash
grep -i "hv_gradient" results/training_hovernet_epidermal.log
grep -i "sobel" results/training_hovernet_epidermal.log
```

**Objectif:** Vérifier si Sobel gradient loss était actif durant training

**Attendu si Sobel actif:**
```
Epoch 1: hv_l1=0.45, hv_gradient=0.12, hv_loss=0.69
Epoch 10: hv_l1=0.23, hv_gradient=0.08, hv_loss=0.39
```

**Si Sobel absent:**
→ ✅ Confirme checkpoint pré-Sobel
→ Ré-entraînement requis

**Statut:** ⏳ NON EXÉCUTÉ (conditionnel à Test 6 résultat B)

---

### ⏳ Test 7: Ré-entraînement avec Sobel (lambda_hv=3.0)

**Commande recommandée (Expert):**
```bash
python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_np 1.0 \
    --lambda_hv 3.0 \
    --lambda_nt 1.0 \
    --batch_size 16
```

**Changement clé:** `lambda_hv 2.0 → 3.0` (augmenté)

**Durée:** ~40 minutes (571 samples epidermal)

**Métriques à surveiller:**
| Epoch | HV MSE Attendu | Interprétation |
|-------|----------------|----------------|
| 1-5 | 0.30-0.40 | Normal |
| 10-20 | 0.15-0.25 | Convergence |
| 30-50 | **0.05-0.10** | ✅ Sobel actif (descente lente = bon signe) |

**Citation expert:**
> "Si [HV MSE] descend plus lentement ou reste plus haute qu'avant tout en étant stable, c'est bon signe : le modèle travaille plus dur sur les détails complexes du gradient."

**Statut:** ⏳ NON EXÉCUTÉ (en attente validation Tests 5 et 6)

---

### ⏳ Test 8: Validation Post-Training

**8a. Test sur Training Data:**
```bash
python scripts/evaluation/test_on_training_data.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 10
```

**Attendu:**
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| HV Magnitude | 0.022 | >0.50 | +2200% |

---

**8b. Visualisation Instance Maps:**
```bash
python scripts/evaluation/visualize_instance_maps.py
```

**Attendu:**
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| Instances PRED | 1 | 5-8 | +500-700% |

---

**8c. AJI Ground Truth:**
```bash
python scripts/evaluation/test_aji_v8.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 50
```

**Attendu:**
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| AJI | 0.09 | >0.60 | +567% |
| PQ | 0.10 | >0.65 | +550% |

**Statut:** ⏳ NON EXÉCUTÉ (après Test 7)

---

## 📈 Graphe de Dépendances des Tests

```
Test 1 (Scaling) ──────┐
Test 2 (Visualisation) ┤───→ Giant Blob confirmé
Test 3 (Architecture)  ┤     HV magnitude 0.022
Test 4 (Documentation) ┘
         │
         ▼
Test 5 (HV Targets .npz) ← CRITIQUE - Point de décision
         │
         ├─ ✅ Targets OK ─────→ Test 6 (Date Checkpoint)
         │                              │
         │                              ├─ < 2025-12-23 ─→ Test 7 (Ré-entraînement)
         │                              │                          │
         │                              └─ ≥ 2025-12-23 ─→ Test 6b (Logs) ─→ Test 7
         │                                                          │
         └─ ❌ Targets KO ────→ STOP ──→ Régénération v9 ──────────┘
                                                 │
                                                 ▼
                                         Test 7 (Ré-entraînement)
                                                 │
                                                 ▼
                                         Test 8a/b/c (Validation)
                                                 │
                                                 ▼
                                         ✅ AJI 0.60+ RÉSOLU
```

---

## 🎯 Hypothèses Actuelles (Mise à Jour)

### Hypothèse #1: Checkpoint Pré-Sobel (90% probabilité) ✅ PROBABLE

**Preuves:**
1. ✅ Sobel fix daté 2025-12-23 (FIX_SOBEL_GRADIENT_LOSS.md)
2. ✅ Code actuel a Sobel implémenté
3. ✅ HV magnitude 0.022 = signature modèle pré-Sobel
4. ⏳ Date checkpoint non vérifiée (Test 6 en attente)

**Si confirmé:**
→ Ré-entraînement avec lambda_hv=3.0 (Test 7)
→ Prédiction expert: AJI 0.60+ fortement probable

---

### Hypothèse #2: Watershed Params Conservateurs (5% probabilité) ❌ IMPROBABLE

**Contre-preuves:**
1. ❌ Scaling ×50 n'améliore PAS l'AJI (Test 1)
2. ❌ 137 peaks détectés (modèle "voit" les cellules)
3. ❌ HV magnitude 0.022 trop faible même pour watershed optimal

**Statut:** Hypothèse écartée (Test 1 négatif)

---

### Hypothèse #3: Gaussian Smoothing Agressif (5% probabilité) ❌ IMPROBABLE

**Avis expert:**
> "Sigma 0.5 très léger, sert à éviter aliasing. Ne PAS le supprimer. Vrai problème: Sobel au training, pas smoothing au preprocessing."

**Contre-preuves:**
1. ❌ Sigma 0.5 considéré optimal par expert
2. ❌ Smoothing évite crénelage pixels (nécessaire pour watershed)

**Statut:** Hypothèse écartée (recommandation expert)

**Test conditionnel:** Si Test 5 montre std < 0.3, régénérer sans smoothing

---

## 🔑 Conclusion Actuelle

**État de l'investigation:**
- Tests effectués: 4/8 (50%)
- Tests critiques restants: 2 (Tests 5 et 6)
- Hypothèse principale: 90% confiance (checkpoint pré-Sobel)

**Prochaine action CRITIQUE:**
→ **Test 5: Vérifier HV targets .npz**

**Si Test 5 ✅:**
→ Test 6 → Test 7 → Résolution probable

**Si Test 5 ❌:**
→ Régénération v9 → Test 7 → Résolution

**Confiance résolution:** Élevée (expert + documentation alignés)

---

## 📁 Fichiers Créés Durant Investigation

| Fichier | Type | Description |
|---------|------|-------------|
| `scripts/evaluation/test_hv_scaling.py` | Test | Scaling HV ×1 à ×50 |
| `scripts/evaluation/visualize_instance_maps.py` | Diagnostic | Visualisation Giant Blob |
| `scripts/validation/verify_hv_targets_npz.py` | Vérification | Check dtype/range targets |
| `docs/ANALYSE_TEST_SCALING_NEGATIF.md` | Doc | Analyse test scaling |
| `docs/GIANT_BLOB_RESOLUTION_PLAN.md` | Plan | 3 hypothèses + actions |
| `docs/PLAN_VERIFICATION_HOVERNET.md` | Plan | 5 étapes vérification |
| `docs/DECISION_RE_ENTRAINEMENT.md` | Synthèse | Consensus Claude+Expert |
| `docs/HISTORIQUE_TESTS_GIANT_BLOB.md` | **Ce fichier** | Historique complet |

---

**Dernière mise à jour:** 2025-12-24
**Prochaine action:** Exécuter Test 5 (verify_hv_targets_npz.py)
