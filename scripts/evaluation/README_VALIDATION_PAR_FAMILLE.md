# Scripts de Validation par Famille

**Date:** 2025-12-22
**Version:** 1.0
**Statut:** ✅ Prêt pour exécution (nécessite données PanNuke + checkpoints)

---

## Vue d'Ensemble

Ces scripts permettent d'isoler la source du problème de ground truth (Recall 7.69%) en testant:
1. Chaque modèle de famille indépendamment
2. Le routage OrganHead → Famille
3. L'intégration complète du pipeline

---

## Scripts Disponibles

### 1. `prepare_test_samples_by_family.py`

**Objectif:** Extraire et organiser les échantillons de test par famille

**Stratégie optimisée:**
- Charge les 500 premiers échantillons de fold2 (mmap pour économie RAM)
- Sélectionne aléatoirement max 10 échantillons par organe (seed=42)
- Groupe par famille d'organes

**Usage:**
```bash
python scripts/evaluation/prepare_test_samples_by_family.py \
    --pannuke_dir /home/amar/data/PanNuke \
    --fold 2 \
    --max_samples 500 \
    --output_dir data/test_samples_by_family
```

**Arguments:**
- `--pannuke_dir`: Répertoire PanNuke (doit contenir fold0, fold1, fold2)
- `--fold`: Fold à utiliser (défaut: 2, non utilisé en entraînement)
- `--max_samples`: Nombre max d'échantillons à charger (défaut: 500)
- `--output_dir`: Répertoire de sortie (défaut: `data/test_samples_by_family`)

**Sortie:**
```
data/test_samples_by_family/
├── glandular/
│   ├── test_samples.npz      # (images, masks, organs, indices)
│   └── metadata.json
├── digestive/
├── urologic/
├── epidermal/
├── respiratory/
└── global_report.json        # Statistiques globales
```

**Temps d'exécution:** ~30 secondes

---

### 2. `test_family_models_isolated.py`

**Objectif:** Tester chaque modèle HoVer-Net sur ses propres données

**Métriques calculées:**
- **NP Dice:** Qualité de la segmentation binaire (attendu: > 0.93)
- **HV MSE:** Qualité des gradients pour séparation instances (attendu: < 0.05)
- **NT Acc:** Précision classification 5 types (attendu: > 0.85)

**Usage:**
```bash
python scripts/evaluation/test_family_models_isolated.py \
    --test_samples_dir data/test_samples_by_family \
    --checkpoint_dir models/checkpoints \
    --output_dir results/family_validation/isolated_tests \
    --device cuda
```

**Arguments:**
- `--test_samples_dir`: Répertoire avec échantillons par famille
- `--checkpoint_dir`: Répertoire avec checkpoints HoVer-Net
- `--output_dir`: Répertoire de sortie
- `--device`: Device PyTorch (cuda/cpu, défaut: cuda si disponible)
- `--families`: Familles à tester (défaut: toutes)

**Sortie:**
```
results/family_validation/isolated_tests/
├── glandular_results.json
├── digestive_results.json
├── urologic_results.json
├── epidermal_results.json
├── respiratory_results.json
└── global_report.json
```

**Temps d'exécution:** ~3-5 minutes (GPU), ~15-20 minutes (CPU)

---

### 3. `test_organ_routing.py`

**Objectif:** Vérifier la précision du routage OrganHead → Famille

**Tests effectués:**
- Prédiction d'organe (OrganHead sur CLS token)
- Mapping organe → famille (ORGAN_TO_FAMILY)
- Confiance des prédictions

**Usage:**
```bash
python scripts/evaluation/test_organ_routing.py \
    --test_samples_dir data/test_samples_by_family \
    --checkpoint_dir models/checkpoints \
    --output_dir results/family_validation/routing_tests \
    --device cuda
```

**Arguments:**
- `--test_samples_dir`: Répertoire avec échantillons par famille
- `--checkpoint_dir`: Répertoire avec checkpoints (doit contenir `organ_head_best.pth`)
- `--output_dir`: Répertoire de sortie
- `--device`: Device PyTorch

**Sortie:**
```
results/family_validation/routing_tests/
└── routing_results.json
```

**Métriques:**
- `organ_accuracy`: % prédictions d'organe correctes (cible: > 95%)
- `family_accuracy`: % familles correctes après mapping (cible: > 99%)
- `errors`: Liste des erreurs avec confiances

**Temps d'exécution:** ~1-2 minutes

---

### 4. `run_family_validation_pipeline.sh`

**Objectif:** Orchestre les 3 étapes en séquence

**Usage:**
```bash
bash scripts/evaluation/run_family_validation_pipeline.sh \
    /home/amar/data/PanNuke \
    models/checkpoints
```

**Arguments positionnels:**
1. Répertoire PanNuke (défaut: `/home/amar/data/PanNuke`)
2. Répertoire checkpoints (défaut: `models/checkpoints`)
3. Répertoire de sortie (défaut: `results/family_validation_YYYYMMDD_HHMMSS`)

**Workflow:**
```
╔══════════════════════════════════════════════════════╗
║ ÉTAPE 1/3: Préparation des échantillons             ║
╠══════════════════════════════════════════════════════╣
║ • Charge 500 premiers échantillons de fold2         ║
║ • Sélectionne max 10 par organe                     ║
║ • Groupe par famille                                 ║
╠══════════════════════════════════════════════════════╣
║ ÉTAPE 2/3: Test isolé de chaque modèle de famille   ║
╠══════════════════════════════════════════════════════╣
║ • Teste glandular sur ses données                   ║
║ • Teste digestive sur ses données                   ║
║ • Teste urologic, epidermal, respiratory            ║
║ • Calcule NP Dice, HV MSE, NT Acc                   ║
╠══════════════════════════════════════════════════════╣
║ ÉTAPE 3/3: Test du routage OrganHead → Famille      ║
╠══════════════════════════════════════════════════════╣
║ • Vérifie prédiction organe                         ║
║ • Vérifie mapping organe → famille                  ║
║ • Identifie les erreurs                             ║
╚══════════════════════════════════════════════════════╝
```

**Temps total:** ~5-10 minutes (GPU)

**Sortie finale:**
```
results/family_validation_YYYYMMDD_HHMMSS/
├── test_samples/           # Échantillons par famille
├── isolated_tests/         # Résultats tests isolés
└── routing_tests/          # Résultats routage
```

---

## Prérequis

### Données PanNuke

Structure requise:
```
/home/amar/data/PanNuke/
└── fold2/
    ├── images.npy    # (N, 256, 256, 3) uint8
    ├── masks.npy     # (N, 256, 256, 6) uint8
    └── types.npy     # (N,) str (organ names)
```

**Téléchargement:**
```bash
python scripts/setup/download_and_prepare_pannuke.py
```

### Checkpoints HoVer-Net

Fichiers requis:
```
models/checkpoints/
├── organ_head_best.pth           # OrganHead (19 organes)
├── hovernet_glandular_best.pth   # HoVer-Net glandulaire
├── hovernet_digestive_best.pth   # HoVer-Net digestive
├── hovernet_urologic_best.pth    # HoVer-Net urologique
├── hovernet_epidermal_best.pth   # HoVer-Net épidermoïde
└── hovernet_respiratory_best.pth # HoVer-Net respiratoire
```

**Ré-entraînement si nécessaire:**
```bash
# OrganHead
python scripts/training/train_organ_head.py --folds 0 1 2 --epochs 50

# HoVer-Net par famille
for family in glandular digestive urologic epidermal respiratory; do
    python scripts/training/train_hovernet_family.py \
        --family $family --epochs 50 --augment
done
```

### Environnement Python

```bash
conda activate cellvit

# Vérifier
python -c "import torch, timm, numpy, cv2; print('✅ Environnement OK')"
```

---

## Interprétation des Résultats

### Cas 1: Tests Isolés ✅ mais Ground Truth ❌

**Métriques:**
- NP Dice > 0.93 ✅
- HV MSE < 0.05 ✅
- NT Acc > 0.85 ✅
- Recall GT = 7.69% ❌

**Diagnostic:** Instance mismatch (Bug #2)

**Cause:** Entraînement sur instances fusionnées (`connectedComponents`), évaluation sur vraies instances PanNuke

**Solution:** Ré-entraîner avec vraies instances (voir `DIAGNOSTIC_REPORT_LOW_RECALL.md`)

---

### Cas 2: Tests Isolés ❌ pour certaines familles

**Métriques:**
- Glandular: Dice 0.96 ✅
- Urologic: Dice 0.75 ❌

**Diagnostic:** Données insuffisantes ou problème entraînement

**Cause:** Familles avec < 2000 échantillons peuvent sous-performer

**Solutions:**
- Data augmentation plus agressive
- Ré-entraîner avec plus d'epochs
- Vérifier features H-optimus-0 (CLS std 0.70-0.90)

---

### Cas 3: Routage ❌

**Métriques:**
- Organ Accuracy < 95% ❌
- Family Accuracy < 99% ❌

**Diagnostic:** Problème OrganHead ou mapping

**Solutions:**
1. Vérifier features H-optimus-0 (CLS std 0.70-0.90)
2. Ré-entraîner OrganHead si features corrompues
3. Vérifier `ORGAN_TO_FAMILY` dans `src/models/organ_families.py`

---

## Dépannage Rapide

| Erreur | Solution |
|--------|----------|
| `FileNotFoundError: fold2` | Vérifier chemin PanNuke ou télécharger |
| `Checkpoint manquant` | Ré-entraîner le modèle manquant |
| `CLS std out of range` | Ré-extraire features H-optimus-0 |
| `CUDA out of memory` | Réduire batch_size ou utiliser CPU |
| `ModuleNotFoundError` | Activer environnement `conda activate cellvit` |

---

## Références

- **Guide complet:** `docs/GUIDE_VALIDATION_PAR_FAMILLE.md`
- **Diagnostic initial:** `results/DIAGNOSTIC_REPORT_LOW_RECALL.md`
- **Plan évaluation GT:** `docs/PLAN_EVALUATION_GROUND_TRUTH.md`
- **Architecture projet:** `CLAUDE.md`

---

**Dernière mise à jour:** 2025-12-22
**Auteur:** Claude (CellViT-Optimus Project)
