# Prochaines Étapes - Amélioration AJI

**Situation actuelle:**
- ✅ NP Dice: 0.92 (EXCELLENT - modèle fonctionne!)
- ❌ AJI: 0.27 (objectif: 0.60+)
- ❌ HV magnitude: 0.022 (attendu: >0.5)

**Diagnostic expert:** Le modèle fonctionne parfaitement, c'est le post-processing qui échoue à cause des gradients HV trop faibles.

---

## Action Immédiate (5 min)

### Test #1: Scaling HV

Lancer le test de scaling pour confirmer le diagnostic:

```bash
python scripts/evaluation/test_hv_scaling.py
```

**Ce que ça fait:**
- Teste différents facteurs de multiplication HV (1x, 5x, 10x, 20x, 50x)
- Mesure l'impact sur l'AJI
- Identifie le facteur optimal

**Résultat attendu:**

| Scénario | AJI avec 10x | Conclusion |
|----------|--------------|------------|
| A | 0.50-0.65 | ✅ Problème confirmé: magnitude HV trop faible |
| B | <0.40 | ❌ Autre problème (Bug #3: instance mismatch) |

---

## Selon Résultats Test

### Si Scénario A (AJI améliore à 0.50+)

**Étape 1:** Corriger resize dans `test_aji_v8.py`

Le script actuel resize **mal** (GT 256→224 au lieu de prédictions 224→256).

**Fix à appliquer:**

```python
# ❌ ACTUEL (lignes 146-160):
np_gt = resize(np_targets[i], (224, 224), interpolation=INTER_NEAREST)  # Mauvais sens!

# ✅ CORRECT:
# Garder GT à 256x256 (natif PanNuke)
np_gt = np_targets[i]  # (256, 256)
hv_gt = hv_targets[i]  # (2, 256, 256)
inst_gt = inst_maps[i]  # (256, 256)

# Resize PRÉDICTIONS 224 → 256 avec INTER_NEAREST
from cv2 import resize, INTER_NEAREST

np_pred_256 = resize(np_pred, (256, 256), interpolation=INTER_NEAREST)
hv_pred_256 = np.stack([
    resize(hv_pred[0], (256, 256), interpolation=INTER_NEAREST),
    resize(hv_pred[1], (256, 256), interpolation=INTER_NEAREST)
])

# Appliquer scaling optimal trouvé dans test
hv_pred_scaled = hv_pred_256 * 10.0  # (ou facteur optimal du test)

# Post-processing
inst_pred = post_process_hv(np_pred_256, hv_pred_scaled)

# Calcul AJI (tout à 256x256)
aji = compute_aji(inst_pred, inst_gt)
```

**Étape 2:** Re-tester

```bash
python scripts/evaluation/test_aji_v8.py \
    --family epidermal \
    --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
    --n_samples 50
```

**Résultat attendu:** AJI 0.27 → 0.55-0.65 ✅

---

### Si Scénario B (AJI reste <0.40)

**Problème plus profond:** Bug #3 (Instance Mismatch)

Le modèle a été entraîné sur des instances **fusionnées** (connectedComponents) au lieu des vraies instances séparées de PanNuke.

**Lire:** `CLAUDE.md` lignes 745-819

**Solution:** Ré-entraîner avec vraies instances PanNuke

**Coût:** ~10 heures (5 familles)

**Décision requise:** Investir le temps ou accepter AJI <0.50?

---

## Résumé Timeline

| Action | Temps | Résultat |
|--------|-------|----------|
| **Test scaling** | 5 min | Identifier facteur optimal |
| **Fix resize** (si A) | 5 min | Corriger test_aji_v8.py |
| **Re-test AJI** (si A) | 5 min | Validation finale |
| **Total (Scénario A)** | **15 min** | **AJI 0.55-0.65** ✅ |
| **Re-training (Scénario B)** | 10h | AJI 0.65+ (avec vraies instances) |

---

## Documentation Créée

- ✅ `scripts/evaluation/test_hv_scaling.py` - Test scaling automatisé
- ✅ `docs/DIAGNOSTIC_AJI_LOW_EXPERT_ANALYSIS.md` - Analyse expert complète
- ✅ `PROCHAINES_ETAPES_AJI.md` - Ce document

---

## Commande Unique (Scénario A Rapide)

Si tu veux tester et fixer d'un coup:

```bash
# 1. Test scaling
python scripts/evaluation/test_hv_scaling.py > results/scaling_test.txt

# 2. Lire facteur optimal
cat results/scaling_test.txt | grep "Meilleur scaling"

# 3. Appliquer fix manuel dans test_aji_v8.py (5 min)

# 4. Re-test
python scripts/evaluation/test_aji_v8.py --family epidermal --n_samples 50
```

---

**Question Clé:** Le modèle a-t-il bien appris à créer des gradients HV forts, ou faut-il juste amplifier sa sortie?

**Réponse dans 5 min avec le test scaling!**
