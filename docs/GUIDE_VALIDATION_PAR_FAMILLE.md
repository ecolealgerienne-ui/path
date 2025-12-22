# Guide d'Exécution: Validation par Famille

**Objectif:** Isoler la source du problème de ground truth en testant chaque modèle de famille indépendamment.

**Date de création:** 2025-12-22
**Statut:** Scripts prêts ✅ — En attente de données PanNuke

---

## Contexte

Suite au problème de recall faible (7.69%) identifié dans le diagnostic ground truth, nous devons déterminer si le problème vient de:

1. **Modèles de famille** — Entraînement insuffisant ou mauvaise qualité
2. **Routage OrganHead → Famille** — Mauvaise prédiction d'organe ou mapping incorrect
3. **Instance mismatch** — connectedComponents fusionne les cellules touchantes (problème fondamental)

---

## Scripts Créés

| Script | Rôle | Localisation |
|--------|------|--------------|
| `prepare_test_samples_by_family.py` | Extrait 500 échantillons de fold2, sélectionne 10 par organe, groupe par famille | `scripts/evaluation/` |
| `test_family_models_isolated.py` | Teste chaque modèle de famille sur ses propres données | `scripts/evaluation/` |
| `test_organ_routing.py` | Vérifie la précision OrganHead et le mapping organe → famille | `scripts/evaluation/` |
| `run_family_validation_pipeline.sh` | Orchestre les 3 étapes en séquence | `scripts/evaluation/` |

---

## Prérequis

### 1. Données PanNuke

Les données PanNuke doivent être disponibles au chemin:
```
/home/amar/data/PanNuke/
├── fold0/
│   ├── images.npy
│   ├── masks.npy
│   └── types.npy
├── fold1/
│   └── ...
└── fold2/  ← Utilisé pour les tests (non utilisé en entraînement)
    ├── images.npy
    ├── masks.npy
    └── types.npy
```

**Si PanNuke est ailleurs**, modifier le chemin dans le script bash:
```bash
PANNUKE_DIR=${1:-"/VOTRE/CHEMIN/PanNuke"}
```

### 2. Checkpoints HoVer-Net

Les checkpoints des 5 familles doivent être dans:
```
models/checkpoints/
├── organ_head_best.pth
├── hovernet_glandular_best.pth
├── hovernet_digestive_best.pth
├── hovernet_urologic_best.pth
├── hovernet_epidermal_best.pth
└── hovernet_respiratory_best.pth
```

**Vérification:**
```bash
ls -la models/checkpoints/*.pth
```

### 3. Environnement Python

Activer l'environnement conda:
```bash
conda activate cellvit
```

Vérifier que les packages sont installés:
```bash
python -c "import torch, timm, numpy, cv2; print('OK')"
```

---

## Exécution du Pipeline

### Option A: Exécution Automatique (Recommandée)

Exécuter les 3 étapes en une seule commande:

```bash
bash scripts/evaluation/run_family_validation_pipeline.sh \
    /home/amar/data/PanNuke \
    models/checkpoints
```

**Temps estimé:** ~5-10 minutes (dépend du GPU)

### Option B: Exécution Manuelle (Étape par Étape)

Si vous voulez contrôler chaque étape:

#### Étape 1: Préparation des Échantillons

```bash
python scripts/evaluation/prepare_test_samples_by_family.py \
    --pannuke_dir /home/amar/data/PanNuke \
    --fold 2 \
    --max_samples 500 \
    --output_dir data/test_samples_by_family
```

**Sortie attendue:**
```
data/test_samples_by_family/
├── glandular/
│   ├── test_samples.npz  (images, masks, organs, indices)
│   └── metadata.json
├── digestive/
│   └── ...
├── urologic/
│   └── ...
├── epidermal/
│   └── ...
├── respiratory/
│   └── ...
└── global_report.json
```

**Vérification:**
```bash
cat data/test_samples_by_family/global_report.json | grep -E "(total_samples|families)"
```

#### Étape 2: Test Isolé des Modèles de Famille

```bash
python scripts/evaluation/test_family_models_isolated.py \
    --test_samples_dir data/test_samples_by_family \
    --checkpoint_dir models/checkpoints \
    --output_dir results/family_validation/isolated_tests
```

**Sortie attendue:**
```
results/family_validation/isolated_tests/
├── glandular_results.json
├── digestive_results.json
├── urologic_results.json
├── epidermal_results.json
├── respiratory_results.json
└── global_report.json
```

**Métriques clés:**
- NP Dice > 0.93 → Détection OK
- HV MSE < 0.05 → Séparation instances OK
- NT Acc > 0.85 → Classification OK

#### Étape 3: Test du Routage

```bash
python scripts/evaluation/test_organ_routing.py \
    --test_samples_dir data/test_samples_by_family \
    --checkpoint_dir models/checkpoints \
    --output_dir results/family_validation/routing_tests
```

**Sortie attendue:**
```
results/family_validation/routing_tests/
└── routing_results.json
```

**Métriques clés:**
- Organ Accuracy > 95% → OrganHead prédit correctement
- Family Accuracy > 99% → Mapping organe → famille correct

---

## Interprétation des Résultats

### Scénario 1: Tests Isolés OK, Ground Truth KO

**Observation:**
- NP Dice > 0.93 ✅
- HV MSE < 0.05 ✅
- NT Acc > 0.85 ✅
- Mais Recall Ground Truth = 7.69% ❌

**Diagnostic:** Instance mismatch (Bug #2)

**Explication:**
Le modèle est entraîné sur des instances fusionnées (`connectedComponents`), mais l'évaluation GT utilise les vraies instances PanNuke séparées.

**Solution:** Ré-entraîner avec les vraies instances PanNuke (voir `DIAGNOSTIC_REPORT_LOW_RECALL.md` section "Solution Cible")

---

### Scénario 2: Tests Isolés KO pour certaines familles

**Observation:**
- Glandular: NP Dice 0.96 ✅
- Digestive: NP Dice 0.96 ✅
- Urologic: NP Dice 0.75 ❌ (< 0.93)
- Epidermal: HV MSE 0.35 ❌ (> 0.05)
- Respiratory: NT Acc 0.70 ❌ (< 0.85)

**Diagnostic:** Problèmes d'entraînement sur familles avec peu de données

**Explication:**
Familles avec < 2000 échantillons peuvent avoir des performances dégradées (voir CLAUDE.md "Analyse de Stabilité")

**Solution:**
- Data augmentation plus agressive
- Ré-entraîner avec plus d'epochs
- Vérifier la qualité des features H-optimus-0

---

### Scénario 3: Routage KO

**Observation:**
- Organ Accuracy < 95% ❌
- Family Accuracy < 99% ❌

**Diagnostic:** Problème OrganHead ou mapping ORGAN_TO_FAMILY

**Solutions:**
1. **OrganHead mal calibré:**
   - Vérifier CLS std (doit être 0.70-0.90)
   - Ré-extraire features si nécessaire
   - Ré-entraîner OrganHead

2. **Mapping incorrect:**
   - Vérifier `src/models/organ_families.py`
   - Corriger les erreurs de mapping
   - Revalider

---

## Fichiers de Sortie

### `global_report.json` (Préparation)

```json
{
  "fold": 2,
  "max_samples_loaded": 500,
  "samples_per_organ": 10,
  "total_samples": 87,
  "families": {
    "glandular": {
      "total": 35,
      "organs": {
        "Breast": 10,
        "Prostate": 10,
        "Thyroid": 8,
        "Pancreatic": 5,
        "Adrenal_gland": 2
      }
    },
    ...
  }
}
```

### `{family}_results.json` (Tests Isolés)

```json
{
  "family": "glandular",
  "n_samples": 35,
  "organs": ["Breast", "Prostate", "Thyroid", ...],
  "metrics": {
    "dice": {"mean": 0.9648, "std": 0.0184},
    "hv_mse": {"mean": 0.0106, "std": 0.0021},
    "nt_acc": {"mean": 0.9111, "std": 0.0154}
  },
  "per_sample_metrics": [...]
}
```

### `routing_results.json` (Routage)

```json
{
  "total_samples": 87,
  "correct_organ": 86,
  "correct_family": 87,
  "organ_accuracy": 0.9885,
  "family_accuracy": 1.0,
  "errors": [
    {
      "family_gt": "glandular",
      "organ_gt": "Breast",
      "organ_pred": "Thyroid",
      "family_pred": "glandular",
      "confidence": 0.72,
      "organ_correct": false,
      "family_correct": true,
      "sample_idx": 12
    }
  ]
}
```

---

## Dépannage

### Erreur: "FileNotFoundError: PanNuke fold2"

**Solution:** Vérifier le chemin PanNuke:
```bash
ls -la /home/amar/data/PanNuke/fold2/
```

Si absent, télécharger PanNuke:
```bash
python scripts/setup/download_and_prepare_pannuke.py
```

### Erreur: "Checkpoint manquant: hovernet_X_best.pth"

**Solution:** Vérifier les checkpoints:
```bash
ls -la models/checkpoints/*.pth
```

Si absents, ré-entraîner:
```bash
python scripts/training/train_hovernet_family.py --family X --epochs 50 --augment
```

### Erreur: "CLS std out of range"

**Solution:** Features H-optimus-0 corrompues, ré-extraire:
```bash
python scripts/preprocessing/extract_features.py \
    --data_dir /home/amar/data/PanNuke \
    --fold 2 \
    --batch_size 8 \
    --chunk_size 500
```

### Erreur: "CUDA out of memory"

**Solution:** Réduire batch_size ou utiliser CPU:
```bash
# Dans test_family_models_isolated.py
python ... --device cpu
```

---

## Prochaines Étapes (Après Validation)

1. **Analyser les rapports JSON** générés
2. **Identifier le scénario** correspondant (1, 2 ou 3)
3. **Appliquer la solution** recommandée
4. **Documenter les résultats** dans CLAUDE.md
5. **Décider de la stratégie** à long terme:
   - Ré-entraîner avec vraies instances PanNuke (si scénario 1)
   - Améliorer familles faibles (si scénario 2)
   - Recalibrer OrganHead (si scénario 3)

---

## Ressources Complémentaires

- **Diagnostic initial:** `results/DIAGNOSTIC_REPORT_LOW_RECALL.md`
- **Spécifications complètes:** `docs/PLAN_EVALUATION_GROUND_TRUTH.md`
- **Architecture HoVer-Net:** CLAUDE.md section "Explication du Modèle HoVer-Net"
- **Bugs connus:** CLAUDE.md section "BUG #1, BUG #2, BUG #3"

---

## Contact & Support

Si vous rencontrez des problèmes non documentés ici, consultez:
1. Les logs de sortie des scripts (mode verbose)
2. Les tests unitaires: `pytest tests/unit/test_*.py -v`
3. Le rapport de vérification Phase 1: `results/verification/phase1_verification_report_20251222.md`

**Dernière mise à jour:** 2025-12-22
**Version des scripts:** v1.0 (optimisé 500 échantillons, 10 par organe)
