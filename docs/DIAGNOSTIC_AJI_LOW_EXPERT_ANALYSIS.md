# Diagnostic AJI Faible - Analyse Expert

**Date:** 2025-12-24
**Statut:** ⚠️ EN COURS - Test scaling HV requis
**Problème:** AJI 0.27 (objectif: 0.60+) malgré Dice 0.92

---

## Résumé Exécutif

✅ **Le modèle fonctionne** - Dice 0.92 prouve que la segmentation binaire est excellente
❌ **Le post-processing échoue** - HV magnitude 0.022 trop faible pour séparer instances

---

## Analyse Expert

### Citation Clé

> "Ton Dice est très élevé (0.92), ce qui signifie que ton masque binaire est quasiment parfait.
> Si l'AJI est bas, c'est que l'algorithme de Watershed ne parvient pas à **couper** les noyaux
> qui se touchent. Le coupable probable : **La magnitude de tes gradients HV**."

### Le Paradoxe: Dice (0.92) vs AJI (0.27)

- **Dice mesure:** Chevauchement global des pixels (binaire)
- **AJI mesure:** Séparation correcte de chaque instance individuelle

**Résultats actuels:**

```
NP Dice:  0.8720 ± 0.2926  (0.92 sur échantillon 9) ✅ EXCELLENT
HV MSE:   0.1426 ± 0.0478                            ✅ RAISONNABLE
NT Acc:   0.7976 ± 0.0996                            ✅ RAISONNABLE
HV magnitude: 0.022                                   ❌ TROP FAIBLE (attendu: >0.5)

AJI:      0.2687 ± 0.4306                            ❌ ÉCHEC (objectif: 0.60+)
```

---

## Cause Racine: HV Magnitude Trop Faible

### HV Magnitude Observée: 0.022

**Normalement**, les gradients HoVer-Net doivent osciller entre **-1.0 et +1.0**.

**Avec magnitude 0.02**, la "pente" qui guide le Watershed est **plate**.
L'algorithme voit une seule grande masse au lieu de plusieurs noyaux distincts.

### Pourquoi la Magnitude est-elle si Basse?

Expert a identifié 3 causes possibles:

#### 1. Mismatch de Normalisation des Features ⚠️ À vérifier

> "Si les features utilisées pour le test n'ont pas exactement la même distribution (moyenne/variance)
> que celles du training, le décodeur 's'éteint'."

**Vérification effectuée:**
- ✅ Le décodeur a bien un `Tanh()` sur la branche HV (ligne 120 de hovernet_decoder.py)
- ✅ Les HV targets sont normalisés dans [-1, 1] (float32)
- ❓ Les features training vs test ont-elles la même distribution?

**Action:** Vérifier CLS std des features epidermal utilisées pour le training

#### 2. Activation Manquante ou Écrasée ✅ Non (Tanh présent)

```python
# hovernet_decoder.py ligne 118-121
self.hv_head = nn.Sequential(
    DecoderHead(64, 2),
    nn.Tanh()  # ✅ OBLIGATOIRE: forcer HV dans [-1, 1]
)
```

**Statut:** ✅ Tanh présent, ce n'est pas le problème

#### 3. Problème du Décalage Spatial (Resize) ⚠️ À corriger

Expert:
> "Tes sorties sont en 224x224, mais PanNuke attend du 256x256. L'AJI est extrêmement sensible
> au moindre décalage d'un pixel. Si tu fais un resize bilinaire sur tes cartes HV, tu lisses
> les crêtes des gradients, ce qui fusionne les noyaux."

**Problème identifié dans test_aji_v8.py:**

```python
# ❌ ACTUEL (lignes 146-160):
# Resize GT 256 → 224 (mauvais sens!)
np_gt = resize(np_targets[i], (224, 224), interpolation=INTER_NEAREST)

# ❌ HV targets resizés SANS INTER_NEAREST (lisse gradients!)
hv_gt = np.stack([
    resize(hv_targets[i, 0], (224, 224)),  # Défaut: INTER_LINEAR
    resize(hv_targets[i, 1], (224, 224))
])

# ✅ CORRECT (à implémenter):
# Resize PRÉDICTIONS 224 → 256
np_pred_256 = resize(np_pred, (256, 256), interpolation=INTER_NEAREST)
hv_pred_256 = np.stack([
    resize(hv_pred[0], (256, 256), interpolation=INTER_NEAREST),
    resize(hv_pred[1], (256, 256), interpolation=INTER_NEAREST)
])
```

---

## Plan d'Action (Expert Validé)

### ⚠️ NE PAS RE-ENTRAÎNER

> "Ne relance pas d'entraînement. Le **cerveau** est là (le Dice le prouve).
> Il faut régler les **muscles** (le post-processing)."

### Étape 1: Test Scaling HV (PRIORITÉ 1) 🔜 EN COURS

**Script créé:** `scripts/evaluation/test_hv_scaling.py`

**Test:**
```bash
python scripts/evaluation/test_hv_scaling.py
```

**Facteurs à tester:** 1.0x, 5.0x, 10.0x, 20.0x, 50.0x

**Objectif:** Si multiplication par 10 ou 50 améliore l'AJI → confirme problème magnitude

**Attendu:**
- Si AJI passe de 0.27 → 0.50+ avec scaling 10x: ✅ Problème identifié
- Si AJI reste <0.40 même avec scaling 50x: ❌ Autre problème (Bug #3)

### Étape 2: Vérifier dist_threshold (SI scaling améliore)

Expert:
> "Dans le post-processing HoVer-Net, il y a souvent un paramètre h_tick ou un seuil de détection
> des marqueurs. Avec une magnitude de 0.02, ton seuil actuel est probablement trop haut et ne voit
> aucun 'pic'."

**Paramètres actuels (test_aji_v8.py ligne 44):**
```python
dist_threshold = 2  # CONSERVATIVE
```

**Test:** Si scaling HV donne AJI 0.40-0.50, essayer dist_threshold = 1

### Étape 3: Corriger Resize (PRIORITÉ 2)

**Problèmes identifiés:**

1. **Sens du resize:**
   - ❌ Actuel: GT 256 → 224
   - ✅ Correct: Prédictions 224 → 256

2. **Interpolation HV:**
   - ❌ Actuel: INTER_LINEAR (lisse gradients)
   - ✅ Correct: INTER_NEAREST (préserve crêtes)

**Fix à appliquer dans test_aji_v8.py:**

```python
# Garder GT à 256x256 (natif PanNuke)
np_gt = np_targets[i]  # (256, 256)
hv_gt = hv_targets[i]  # (2, 256, 256)
inst_gt = inst_maps[i]  # (256, 256)

# Resize PRÉDICTIONS 224 → 256
from cv2 import resize, INTER_NEAREST

np_pred_256 = resize(np_pred, (256, 256), interpolation=INTER_NEAREST)
hv_pred_256 = np.stack([
    resize(hv_pred[0], (256, 256), interpolation=INTER_NEAREST),
    resize(hv_pred[1], (256, 256), interpolation=INTER_NEAREST)
])

# Post-processing
inst_pred = post_process_hv(np_pred_256, hv_pred_256)

# Calcul AJI (tout à 256x256)
aji = compute_aji(inst_pred, inst_gt)
```

### Étape 4: Si Échec Persistant → Bug #3

Si après scaling + resize correct, AJI reste <0.50:

**Lire:** `CLAUDE.md` lignes 745-819 (Bug #3: Instance Mismatch)

**Problème possible:** Données training utilisent `connectedComponents` qui fusionne cellules touchantes,
donc le modèle n'a jamais appris à créer des gradients forts aux frontières réelles.

**Solution long terme:** Ré-entraîner avec vraies instances PanNuke (coût: 10h)

---

## Prédiction Expert

> "Ton Dice à 0.97 [0.92 en moyenne] sur le crop 224 montre que ton décodeur est hyper-puissant.
> Il a juste besoin d'apprendre sur un terrain où les cibles ne bougent pas. Une fois le re-training
> terminé avec des features synchronisées, ton AJI va passer de 0.06 à 0.65 en une seule session."

**Note:** Cette prédiction concernait un problème de features corrompues (résolu).
Le problème actuel (HV magnitude faible) est différent mais le principe reste: le modèle fonctionne,
c'est le post-processing qui doit être ajusté.

---

## Timeline Estimée

| Étape | Temps | Commande |
|-------|-------|----------|
| Test scaling HV | 5 min | `python scripts/evaluation/test_hv_scaling.py` |
| Analyse résultats | 2 min | Lire sortie console |
| Fix resize | 5 min | Modifier test_aji_v8.py |
| Validation finale | 5 min | Re-test AJI avec scaling optimal |
| **TOTAL** | **17 min** | |

**Résultat attendu:** AJI 0.27 → 0.50-0.65

---

## Checklist Validation

- [ ] Lancer `test_hv_scaling.py` (vérifier GPU disponible)
- [ ] Noter facteur optimal (probablement 10x ou 20x)
- [ ] Modifier `test_aji_v8.py` avec resize INTER_NEAREST
- [ ] Re-tester AJI avec facteur optimal
- [ ] Si AJI >0.60: ✅ Succès, documenter fix
- [ ] Si AJI <0.50: Lire Bug #3 et décider re-training

---

## Références

- **Expert Analysis:** Messages utilisateur 2025-12-24
- **Bug #3 Documentation:** CLAUDE.md lignes 745-819
- **HoVer-Net Paper:** Graham et al. 2019 (magnitude HV attendue: ±1.0)
- **Script créé:** `scripts/evaluation/test_hv_scaling.py`

---

**Statut:** ⚠️ EN ATTENTE - Test scaling HV requis pour diagnostic complet
