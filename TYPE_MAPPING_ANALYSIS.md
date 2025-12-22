# 🔬 Analyse Type Mapping — Root Cause Investigation

**Date**: 2025-12-21
**Statut**: ✅ +1 Mapping Confirmé REQUIS
**Conclusion**: Le problème n'est PAS le +1, c'est peut-être une vraie erreur de classification OU un problème d'extraction type_probs

---

## 🎯 Résumé Exécutif

**Confusion initiale**: Votre analyse visuelle montrait des couleurs différentes (cyan GT vs vert-jaune Pred), ce qui m'a fait penser que le +1 était faux.

**Réalité**: Le +1 est **OBLIGATOIRE** pour convertir les indices du modèle [0-4] vers les labels PanNuke [1-5].

**Vrai problème possible**: Le modèle pourrait prédire la mauvaise classe OU il y a un bug dans l'extraction des `type_probs` depuis `multifamily_result`.

---

## 📊 Expérience Comparative

### Test SANS +1 (Commit 4bb5b77)

```
Pred Types: [4]
Distribution:
  Dead: 40811 pixels  ← Affiché comme "Dead"

GT:
  Epithelial: 45702 pixels
```

**Interprétation**:
- `pred_type` contient la valeur 4
- Code d'affichage cherche `(pred_type == 4)`
- Liste des noms: `['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']`
- Index 4 dans cette liste (1-indexed) → "Dead"

**Mais en réalité**:
- Le modèle a prédit `argmax() = 4`
- Dans l'indexation du modèle [0-4], 4 = **Epithelial** ✅
- Donc c'est la **bonne prédiction** mais avec le **mauvais label affiché**!

### Test AVEC +1 (Commit 118d7aa) — À RETESTER

```
Pred Types: [5]  ← Attendu
Distribution:
  Epithelial: 40811 pixels  ← Attendu
```

**Interprétation**:
- `pred_type` contiendra la valeur 5
- Code d'affichage cherche `(pred_type == 5)`
- Index 5 dans la liste (1-indexed) → "Epithelial" ✅
- **Bon label affiché** pour la bonne prédiction!

---

## 🎨 Explication Colormap

La visualisation utilise `cmap='tab10'` avec `vmin=0, vmax=5`:

| Valeur | Couleur tab10 | Label PanNuke [1-5] |
|--------|---------------|---------------------|
| 0 | Bleu | Background |
| 1 | Orange | Neoplastic |
| 2 | Vert | Inflammatory |
| 3 | Rouge | Connective |
| 4 | **Violet/Purple** | Dead |
| 5 | **Marron/Brown** | Epithelial |

### Avec +1 (CORRECT)

- Modèle prédit Epithelial → argmax = 4
- Ajout +1 → `pred_type = 5`
- Colormap: valeur 5 → **Marron/Brown**
- GT Epithelial: valeur 5 → **Marron/Brown**
- **Couleurs MATCHENT** ✅

### Sans +1 (INCORRECT)

- Modèle prédit Epithelial → argmax = 4
- Pas de +1 → `pred_type = 4`
- Colormap: valeur 4 → **Violet/Purple**
- GT Epithelial: valeur 5 → **Marron/Brown**
- **Couleurs NE MATCHENT PAS** ❌

---

## 🧩 Pourquoi la Confusion?

Votre observation visuelle (cyan GT vs vert-jaune Pred) était **CORRECTE** quand on avait retiré le +1!

Mais l'interprétation était inversée:
- **Sans +1**: Couleurs différentes car affichage FAUX (violet vs marron)
- **Avec +1**: Couleurs identiques car affichage CORRECT (marron vs marron)

---

## ⚠️ Problème Résiduel Possible

Même AVEC le +1 restauré, il reste 2 hypothèses à tester:

### Hypothèse A: Extraction type_probs Incorrecte

**Code actuel** (`diagnose_gt_failure.py` ligne 171):
```python
pred_np = mf_result.type_probs[1:].sum(axis=0)  # Somme channels 1-5
```

**Problème potentiel**: Si `type_probs` a shape `(6, H, W)` avec channel 0 = background, alors:
- On somme channels [1, 2, 3, 4, 5] pour obtenir NP prob
- Mais pour `pred_type`, on utilise `mf_result.type_map` directement

**Vérification à faire**:
```python
print(f"type_probs shape: {mf_result.type_probs.shape}")
print(f"type_map unique: {np.unique(mf_result.type_map)}")
```

Si `type_map` vient d'un argmax sur `type_probs[0:5]` au lieu de `type_probs[1:6]`, alors il y a un décalage!

### Hypothèse B: Vraie Erreur de Classification

Le modèle pourrait **vraiment** prédire Dead au lieu d'Epithelial pour cette image.

**Test**: Vérifier les logits bruts avant argmax:
```python
# Dans OptimusGate.forward()
nt_logits = self.hovernet_decoders[family].nt_head(...)
print(f"NT logits shape: {nt_logits.shape}")  # Devrait être (1, 5, H, W)
print(f"NT logits channels mean: {nt_logits.mean(dim=[0,2,3])}")  # (5,) - moyenne par classe

# Si channel 3 (Dead) est plus fort que channel 4 (Epi), c'est une vraie erreur
```

---

## ✅ Action Immédiate

**VOUS DEVEZ**:
1. Pull le commit 118d7aa (qui restaure +1)
2. Re-lancer diagnostic:
   ```bash
   python scripts/evaluation/diagnose_gt_failure.py \
       --npz_file data/evaluation/pannuke_fold2_converted/image_00000.npz \
       --checkpoint_dir models/checkpoints_FIXED \
       --output_dir results/diagnostic_gt
   ```

**Résultat attendu AVEC +1**:
```
Pred Types: [5]  ou [1, 2, 5]  ← PAS [4]!
Distribution:
  Epithelial: ~40000 pixels  ← Aligné avec GT (45702)
```

Si vous obtenez encore "Dead" au lieu d'"Epithelial", alors il y a un bug dans l'extraction `type_map` depuis `multifamily_result`.

---

## 📝 Notes Techniques

### Pipeline Complet

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│ PanNuke mask[:, :, 1:6] → [Neo, Inf, Con, Dead, Epi]       │
│         ↓                                                   │
│ np.argmax(axis=-1) → [0, 1, 2, 3, 4]                       │
│         ↓                                                   │
│ nt_targets saved as [0, 1, 2, 3, 4]                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│ HoVerNet decoder.nt_head → logits (B, 5, H, W)             │
│         ↓                                                   │
│ torch.softmax → probs (B, 5, H, W)                          │
│         ↓                                                   │
│ argmax(dim=0) → [0, 1, 2, 3, 4]                            │
│         ↓                                                   │
│ +1 → [1, 2, 3, 4, 5]  ← CONVERSION PanNuke                 │
│         ↓                                                   │
│ pred_type sauvegardé                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    DISPLAY PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│ CELL_TYPES = ['Neoplastic', ..., 'Epithelial']            │
│ for i, name in enumerate(CELL_TYPES, 1):  ← 1-indexed     │
│     if pred_type == i:                                      │
│         print(name)                                         │
└─────────────────────────────────────────────────────────────┘
```

### Checklist Debug

- [ ] `type_probs.shape` = (5, H, W) ou (6, H, W)?
- [ ] `type_map` calculé depuis quel range?
- [ ] Logits bruts: quel channel a le max?
- [ ] Après +1: `np.unique(pred_type)` contient [1-5]?

---

**Créé le**: 2025-12-21
**Par**: Claude (Type Mapping Root Cause)
**Statut**: ✅ +1 Restauré — Test utilisateur requis
**Prochain**: Vérifier extraction type_map si problème persiste
