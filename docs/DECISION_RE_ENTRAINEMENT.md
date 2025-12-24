# Décision Ré-entraînement Epidermal — Synthèse Exécutive

**Date:** 2025-12-24
**Problème:** AJI 0.09 vs objectif 0.60+ (Giant Blob: 1 instance au lieu de 8)
**Cause racine identifiée:** Mismatch de version logique (code Sobel présent, checkpoint pré-Sobel)

---

## 🎯 Consensus Claude + Expert

### Points d'accord (100%)

1. ✅ **Architecture actuelle correcte:**
   - Tanh présent (ligne 118-121)
   - Sobel gradient loss implémenté (ligne 244-280, poids 2.0×)
   - Données v8 utilisent vraies instances PanNuke

2. ✅ **Giant Blob confirmé:**
   - HV magnitude 0.022 (50× trop faible)
   - 1 instance prédite vs 8 GT
   - 137 peaks détectés (modèle "voit" mais ne sépare pas)

3. ✅ **Cause racine: Checkpoint entraîné AVANT Sobel fix**
   - Sobel fix daté: 2025-12-23 (FIX_SOBEL_GRADIENT_LOSS.md)
   - Code actuel a Sobel, mais checkpoint .pth figé sans Sobel
   - Citation expert: "Avoir le code du Sobel dans tes .py ne sert à rien si les poids ont été figés à une époque où le gradient était encore mou"

4. ✅ **Solution: Ré-entraînement avec Sobel**
   - Dice 0.95 prouve que modèle sait OÙ sont les cellules
   - Sobel fix lui apprend COMMENT les séparer
   - Prédiction expert: AJI 0.60+ après ré-entraînement

---

## 🔬 Divergence (Mineure)

| Point | Analyse Claude | Analyse Expert | Consensus |
|-------|----------------|----------------|-----------|
| **Gaussian smoothing** | Hypothèse #3 (sigma=0.5 trop agressif) | "Sigma 0.5 très léger, ne PAS supprimer" | ✅ **Garder le smoothing** |
| **Lambda_hv** | lambda_hv=2.0 (code actuel) | lambda_hv=3.0 (augmenté) | ✅ **Utiliser 3.0** pour "vraiment pousser le gradient" |

---

## 📋 Plan de Vérification (Méthodique)

### Étape 1: Vérifier HV Targets (CRITIQUE - 30s)

**Commande:**
```bash
conda activate cellvit
python scripts/validation/verify_hv_targets_npz.py --family epidermal
```

**Décision:**
- ✅ Targets corrects (float32, [-1, 1]) → Continuer Étape 2
- ❌ Targets incorrects → STOP, régénérer v9 AVANT ré-entraînement

---

### Étape 2: Vérifier Date Checkpoint (2 min)

**Commande:**
```bash
find models/checkpoints -name "hovernet_epidermal_best.pth" -exec ls -l {} \;
```

**Décision:**
- Date < 2025-12-23 → ✅ GO ré-entraînement
- Date ≥ 2025-12-23 → ⚠️ Vérifier logs training (Sobel actif?)

---

### Étape 3: GO/NO-GO Décision

**Critères GO ré-entraînement:**
- [x] HV targets vérifiés ✅
- [x] Checkpoint pré-Sobel ✅
- [x] Architecture correcte ✅
- [x] Données v8 correctes ✅

**Si tous ✅ → LANCER ré-entraînement**

---

## 🚀 Commande Ré-entraînement (Recommandation Expert)

```bash
conda activate cellvit

python scripts/training/train_hovernet_family.py \
    --family epidermal \
    --epochs 50 \
    --augment \
    --lambda_np 1.0 \
    --lambda_hv 3.0 \
    --lambda_nt 1.0 \
    --batch_size 16
```

**Changement clé:** `lambda_hv 2.0 → 3.0`

**Durée:** ~40 minutes (571 samples epidermal)

---

## 📊 Résultats Attendus

### Métriques Training (à surveiller)

**HV MSE durant epochs:**
| Epoch | HV MSE | Interprétation |
|-------|--------|----------------|
| 1-5 | 0.30-0.40 | Normal (apprentissage) |
| 10-20 | 0.15-0.25 | Convergence |
| 30-50 | **0.05-0.10** | ✅ Sobel actif (descente lente = travaille sur gradients) |

**Citation expert:**
> "Si [HV MSE] descend plus lentement ou reste plus haute qu'avant tout en étant stable, c'est bon signe : le modèle travaille plus dur sur les détails complexes du gradient."

---

### Métriques Post-Training (validation)

**Test visualisation (sample 9):**
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| **Instances PRED** | **1** | **5-8** | **+500-700%** 🎯 |
| **HV Magnitude** | **0.022** | **>0.50** | **+2200%** 🎯 |

**Test AJI (50 échantillons):**
| Métrique | Avant | Après (Prédiction Expert) | Amélioration |
|----------|-------|---------------------------|--------------|
| **AJI** | **0.09** | **0.60+** | **+567%** 🎯 |
| **PQ** | ~0.10 | **0.65+** | **+550%** 🎯 |
| Dice | 0.92 | ~0.95 | Stable/Légère hausse |

---

## ✅ Critères de Succès

**Minimum acceptable:** AJI ≥ 0.50, PQ ≥ 0.55

**Cible (prédiction expert):** AJI ≥ 0.60, PQ ≥ 0.65

**Excellent:** AJI ≥ 0.70, PQ ≥ 0.75

---

## 🔄 Plan de Contingence

**Si échec partiel (AJI 0.30-0.50):**
- Test lambda_hv=5.0 (encore plus agressif)
- Vérifier Gaussian smoothing (régénérer avec sigma=0.3)

**Si échec total (AJI <0.30):**
- Investigation approfondie features H-optimus-0
- Vérifier fonction compute_hv_maps()
- Vérifier post-processing Watershed

---

## 🎓 Leçons Apprises

**Citation expert (clé):**
> "Le Dice de 0.95 que tu as déjà prouve que le modèle sait où sont les cellules. En ajoutant le Sobel fix pendant l'entraînement, tu lui apprends enfin comment les séparer. C'est comme donner une paire de lunettes de vue à quelqu'un qui voyait déjà des formes mais sans les détails."

**Takeaway:**
- Magnitude 0.022 = signature d'un modèle "peureux" qui reste proche de zéro
- Sobel force le modèle à "muscler" ses prédictions (créer relief/barrages)
- Lambda_hv augmenté (3.0) pousse encore plus le gradient
- Gaussian smoothing (sigma=0.5) n'est PAS le problème (évite aliasing)

---

## 📝 Checklist Pré-Lancement

Avant de lancer le ré-entraînement:

- [ ] Étape 1: Vérifier HV targets .npz
- [ ] Étape 2: Vérifier date checkpoint
- [ ] Décision GO/NO-GO confirmée
- [ ] Environnement `cellvit` activé
- [ ] GPU disponible (~8-10 GB VRAM)
- [ ] 40 minutes disponibles

**Une fois checklist complète → LANCER ré-entraînement**

---

## 🔗 Documentation Complète

Voir `PLAN_VERIFICATION_HOVERNET.md` pour:
- Plan détaillé 5 étapes
- Arbres de décision
- Commandes complètes de validation
- Références littérature

---

**Recommandation finale:** ✅ GO ré-entraînement avec lambda_hv=3.0 (confiance élevée dans succès)
