# GT Extraction Verification Scripts

Scripts pour vérifier empiriquement si `connectedComponents` fusionne les instances vs extraction native PanNuke.

## Scripts Disponibles

### 1. `verify_gt_extraction.py` — Vérification Simple

Teste **1 échantillon** avec visualisation détaillée.

**Usage:**
```bash
python scripts/evaluation/verify_gt_extraction.py \
    --family epidermal \
    --sample_idx 0 \
    --data_dir /home/amar/data/PanNuke
```

**Sortie:**
- Comparaison chiffrée (N instances CC vs M instances Native)
- Détails par canal PanNuke
- Visualisation: `results/verify_gt_{family}_sample{idx}.png`

**Exemple de résultat:**
```
connectedComponents:      1 instance
PanNuke Native:           3 instances
Différence:               2 instances perdues
Perte:                  66.7%
```

---

### 2. `batch_verify_gt_extraction.py` — Analyse Statistique

Teste **N échantillons** et génère rapport statistique complet.

**Usage:**
```bash
python scripts/evaluation/batch_verify_gt_extraction.py \
    --family epidermal \
    --n_samples 20 \
    --data_dir /home/amar/data/PanNuke
```

**Sortie:**
```
RÉSULTATS STATISTIQUES
======================================================================

Images testées:           20
Images avec cellules:     15
Images background:        5

Instances connectedComponents:    78
Instances PanNuke Native:        125
Instances perdues:                47 (37.6%)

Distribution perte par image:
  Min:     0.0%
  Q25:    25.0%
  Médiane: 40.0%
  Q75:    60.0%
  Max:    80.0%

Cas extrêmes:

  🔴 Pire cas (idx 5):
     connectedComponents: 2 instances
     PanNuke Native:      10 instances
     Perte:               8 instances (80.0%)
     Canaux: {'Neo': 5, 'Infl': 3, 'Conn': 2}

  🟢 Images sans perte: 3/15
```

**Fichier de sortie:** `results/batch_verify_{family}.txt`

---

## Prérequis

### Données FIXED Requises

Les scripts nécessitent les données FIXED (avec fold_ids/image_ids):

```bash
# Générer FIXED data pour une famille
python scripts/preprocessing/prepare_family_data_FIXED.py \
    --data_dir /home/amar/data/PanNuke \
    --family epidermal \
    --chunk_size 300
```

**Fichier créé:** `data/family_FIXED/epidermal_data_FIXED.npz`

**Contient:**
- `images`: Images RGB 256×256
- `np_targets`: Masques binaires
- `hv_targets`: Cartes HV float32 [-1, 1]
- `nt_targets`: Types cellulaires
- **`fold_ids`**: Mapping vers PanNuke fold
- **`image_ids`**: Mapping vers index dans le fold

---

## Workflow Complet

### Étape 1: Générer FIXED Data (si absent)

```bash
# Test rapide: epidermal (571 samples, ~3 min)
python scripts/preprocessing/prepare_family_data_FIXED.py \
    --data_dir /home/amar/data/PanNuke \
    --family epidermal \
    --chunk_size 300

# Famille complète: glandular (3535 samples, ~15 min)
python scripts/preprocessing/prepare_family_data_FIXED.py \
    --data_dir /home/amar/data/PanNuke \
    --family glandular \
    --chunk_size 300
```

### Étape 2: Vérification Simple (1 échantillon)

```bash
# Tester un échantillon avec visualisation
python scripts/evaluation/verify_gt_extraction.py \
    --family epidermal \
    --sample_idx 0 \
    --data_dir /home/amar/data/PanNuke

# Consulter la visualisation
open results/verify_gt_epidermal_sample0.png
```

### Étape 3: Analyse Statistique (20 échantillons)

```bash
# Batch testing pour statistiques robustes
python scripts/evaluation/batch_verify_gt_extraction.py \
    --family epidermal \
    --n_samples 20 \
    --data_dir /home/amar/data/PanNuke

# Consulter le rapport détaillé
cat results/batch_verify_epidermal.txt
```

### Étape 4: Tester Toutes les Familles

```bash
# Script pour tester les 5 familles
for family in glandular digestive urologic epidermal respiratory; do
    echo "Testing $family..."
    python scripts/evaluation/batch_verify_gt_extraction.py \
        --family $family \
        --n_samples 50 \
        --data_dir /home/amar/data/PanNuke
done
```

---

## Interprétation des Résultats

### Perte < 10%
✅ Impact limité — connectedComponents préserve bien les instances
→ Pas besoin de ré-entraînement

### Perte 10-40%
⚠️ Impact modéré — Amélioration watershed recommandée
→ Court terme: Améliorer post-processing
→ Gain attendu: AJI +20-40%

### Perte > 40%
❌ Impact critique — Ré-entraînement nécessaire
→ Long terme: Ré-entraîner avec données FIXED
→ Gain attendu: AJI +60-100%

---

## Exemples de Résultats Observés

### Epidermal (sample 0)
```
connectedComponents:    1 instance
PanNuke Native:         3 instances
Perte:                66.7%
```

**Diagnostic:** 2 cellules inflammatoires fusionnées

### Epidermal (sample 19)
```
connectedComponents:    0 instances
PanNuke Native:         0 instances
Perte:                  0.0%
```

**Diagnostic:** Image background (pas de cellules)

---

## Dépannage

### Erreur: "FIXED data not found"

**Cause:** Données FIXED pas encore générées

**Solution:**
```bash
python scripts/preprocessing/prepare_family_data_FIXED.py \
    --data_dir /home/amar/data/PanNuke \
    --family {family} \
    --chunk_size 300
```

### Erreur: "fold_ids/image_ids not found"

**Cause:** Anciennes données OLD format (sans mapping)

**Solution:** Utiliser format FIXED (voir ci-dessus)

---

## Méthodes d'Extraction Comparées

### connectedComponents (BUGGY)
```python
np_binary = (np_target > 0.5).astype(np.uint8)
_, inst_map = cv2.connectedComponents(np_binary)
```

**Problème:** Fusionne toutes les cellules touchantes en une seule instance

### PanNuke Native (CORRECT)
```python
# Canaux 1-4: IDs natifs PanNuke (instances séparées)
for c in range(1, 5):
    channel_mask = mask[:, :, c]
    inst_ids = np.unique(channel_mask)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = channel_mask == inst_id
        inst_map[inst_mask] = instance_counter
        instance_counter += 1

# Canal 5 (Epithelial): binaire, utiliser connectedComponents
```

**Avantage:** Préserve les instances séparées annotées par les experts

---

## Références

- **Documentation complète:** `docs/VERIFICATION_GT_EXTRACTION_STATUS.md`
- **Pipeline données:** `docs/PIPELINE_COMPLET_DONNEES.md`
- **Problème Bug #3:** `CLAUDE.md` section "BUG #3: Training/Eval Instance Mismatch"
