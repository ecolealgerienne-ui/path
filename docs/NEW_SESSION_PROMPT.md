# Prompt Nouvelle Session — CellViT-Optimus

> **Instructions:** Copier-coller ce prompt au démarrage d'une nouvelle session Claude.

---

## Contexte Projet

Je continue le développement de **CellViT-Optimus**, un système de segmentation et classification de noyaux cellulaires pour l'histopathologie.

### État Actuel (2026-01-18)

**Branche Git:** `claude/review-project-context-JQPxq`
- Status: À jour avec `origin/main` (commit 849bbda)
- Dernière PR mergée: #37 (spec v14.0b + amélioration IHM)

**V13 Production (Stable):**
- Architecture: FPN Chimique + H-Channel Ruifrok (images RAW, **SANS Macenko**)
- Résultats:
  - ✅ **Respiratory:** AJI 0.6872 (101% objectif)
  - 🟡 **Urologic:** AJI 0.6743 (99.2%)
  - 🟡 **Glandular:** AJI 0.6566 (96.6%)
  - 🟠 **Epidermal:** AJI 0.6203 (91.2%)
  - 🟠 **Digestive:** AJI 0.6160 (90.6%)
- **7 organes "Grade Clinique"** (AJI ≥ 0.68): Adrenal_gland (0.7236), Liver (0.7207), Bladder, Bile-duct, Kidney, Cervix, Stomach

**IHM Gradio (Complète):**
- R&D Cockpit (port 7860): Interface développeurs avec debug IA complet
- Interface Pathologiste (port 7861): Interface clinique simplifiée
- Features: Loupe ×3, métriques cliniques, export PDF/CSV/JSON, analyse spatiale Phase 3

**Spec v14.0b WSI Triage (Prête pour développement):**
- Pipeline pyramidal: 1.25× (masque tissu) → 5× (CleaningNet) → 40× (moteur v13)
- Motifs de sélection en 2 temps: `motifs_triage` (Phase 5×) + `motifs_detail` (Phase 40×)
- KPIs: < 2 min/lame, < 5s triage, Sensibilité > 95%, max 30-40 ROIs
- Mini-map GPS 256×256 avec marqueur rouge

---

## ⚠️ RÈGLES CRITIQUES (À RESPECTER ABSOLUMENT)

### 1. Ne Pas Tester Localement

> **🚫 INTERDICTION ABSOLUE D'EXÉCUTER DES COMMANDES DE TEST/ENTRAÎNEMENT**

**Actions AUTORISÉES:**
- ✅ Lire des fichiers (code, configs, documentation)
- ✅ Créer/modifier du code Python
- ✅ Créer des scripts que L'UTILISATEUR lancera
- ✅ Faire de la review de code
- ✅ Créer de la documentation

**Actions INTERDITES:**
- ❌ `python scripts/training/...` (pas d'env GPU)
- ❌ `python scripts/evaluation/...` (pas de données PanNuke)
- ❌ Toute commande nécessitant GPU/données

### 2. Utiliser Toujours l'Existant

> **"On ne touche pas l'existant"** — Les scripts existants fonctionnent. Toute modification requiert validation explicite.

**Avant d'écrire du code:**
1. ✅ Vérifier si un script similaire existe déjà
2. ✅ S'inspirer des patterns des scripts existants
3. ✅ Réutiliser les modules partagés (`src/`)
4. ✅ Ne pas réinventer la roue

**Exemples de scripts de référence:**
- Preprocessing: `scripts/preprocessing/prepare_v13_smart_crops.py`
- Training: `scripts/training/train_hovernet_family_v13_smart_crops.py`
- Evaluation: `scripts/evaluation/test_v13_smart_crops_aji.py`
- Optimisation: `scripts/evaluation/optimize_watershed_aji.py`

### 3. Modules Partagés OBLIGATOIRES (Single Source of Truth)

> **🚫 JAMAIS de duplication de code critique**

**Modules partagés existants:**

| Module | Fonction/Constante | Usage |
|--------|-------------------|-------|
| `src/postprocessing/watershed.py` | `hv_guided_watershed()` | Segmentation instances |
| `src/metrics/ground_truth_metrics.py` | `compute_aji()` | Calcul AJI+ |
| `src/evaluation/instance_evaluation.py` | `run_inference()`, `evaluate_sample()` | Évaluation complète |
| `src/models/organ_head.py` | `PANNUKE_ORGANS`, `OrganPrediction`, `predict_with_ood()` | Prédiction organe |
| `src/preprocessing/__init__.py` | `preprocess_image()`, `HOPTIMUS_MEAN`, `HOPTIMUS_STD` | Normalisation images |
| `src/constants.py` | Toutes les constantes globales | Configuration |

**🔍 Comment vérifier AVANT de coder:**

```bash
# Avant d'écrire une fonction
grep -r "def ma_fonction" src/

# Avant de définir une constante
grep -r "MA_CONSTANTE" src/

# Avant de définir une liste d'organes
grep -r "ORGAN\|FAMILY\|PANNUKE" src/
```

### 4. FPN Chimique = use_hybrid + use_fpn_chimique

**Pour training ET évaluation:**
```bash
# ✅ CORRECT
--use_hybrid --use_fpn_chimique

# ❌ INCORRECT
--use_fpn_chimique  # Sans --use_hybrid → Erreur
```

**Nommage checkpoints:**
```
hovernet_{family}_v13_smart_crops_hybrid_fpn_best.pth
```

### 5. Ruifrok > Macenko (Découverte Stratégique)

> **CRITIQUE:** Macenko normalization cause **-4.3% AJI** (conflit avec extraction Ruifrok du FPN Chimique).

**Production V13:**
```bash
# ✅ CORRECT (Images brutes)
python scripts/preprocessing/prepare_v13_smart_crops.py \
    --family respiratory \
    --pannuke_dir /chemin/vers/PanNuke

# ❌ DÉCONSEILLÉ (Macenko)
# --use_normalized  # Régression -4.3% AJI
```

### 6. Mettre à Jour CLAUDE.md

> **OBLIGATOIRE:** Toute information importante doit être documentée dans `CLAUDE.md`.

**Quand mettre à jour:**
- ✅ Nouvelle découverte technique (ex: Ruifrok vs Macenko)
- ✅ Changement d'architecture
- ✅ Nouveau résultat AJI validé
- ✅ Bug critique résolu
- ✅ Nouvelle règle de développement

**Format:**
- Concis et structuré
- Inclure les chiffres (métriques, temps, tailles)
- Citer les fichiers et lignes concernés

---

## 📚 Documentation Clé (À Lire en Priorité)

### Fichiers Essentiels

```bash
# 1. Contexte projet et règles (SOURCE DE VÉRITÉ)
/home/user/path/CLAUDE.md

# 2. Historique complet (bugs, décisions, journal)
/home/user/path/claude_history.md

# 3. Stratégie V13 Smart Crops (architecture validée)
/home/user/path/docs/V13_SMART_CROPS_STRATEGY.md

# 4. IHM Gradio (architecture, API, phases)
/home/user/path/docs/UI_COCKPIT.md

# 5. Spec v14.0b WSI Triage (prête pour implémentation)
/home/user/path/docs/specs/V14_WSI_TRIAGE_SPEC.md
```

### Commandes de Vérification Git

```bash
# Vérifier branche actuelle
git branch

# Vérifier status
git status

# Voir derniers commits
git log --oneline -10

# Comparer avec main
git diff origin/main..HEAD --stat
```

---

## 🎯 Prochaines Étapes Possibles

### Option 1: Amélioration AJI Epidermal/Digestive (91-90%)

**Pistes documentées (CLAUDE.md):**
- Watershed organ-level (Skin vs HeadNeck, Colon problématique)
- Transfer learning depuis Respiratory (AJI 0.6872)
- Investigation outliers (AJI < 0.50)
- NC-based Beta-Switch (Auto-Tuner)

**Scripts existants:**
```bash
# Optimisation watershed par organe
scripts/evaluation/optimize_watershed_aji.py \
    --checkpoint models/.../hovernet_epidermal_...best.pth \
    --family epidermal \
    --organ Skin \
    --n_samples 20

# Transfer learning inter-famille
scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family epidermal \
    --pretrained_checkpoint models/.../hovernet_respiratory_...best.pth \
    --finetune_lr 1e-5 \
    --epochs 30
```

### Option 2: Implémentation v14.0 WSI Triage

**Phases (docs/specs/V14_WSI_TRIAGE_SPEC.md):**
1. **Infrastructure** (Semaine 1-2): OpenSlide, cuCIM, TiledWSIReader
2. **CleaningNet** (Semaine 3-4): Pseudo-labeling, training MobileNetV3
3. **Intégration** (Semaine 5-6): Pipeline complet, QC
4. **Production** (Semaine 7-8): Tests 100 WSI, optimisation

**Architecture CleaningNet:**
- Entrée: Patch RGB 224×224 + H-Channel (Ruifrok)
- Backbone: MobileNetV3-Small ou EfficientNet-B0
- Tâche: Classification binaire (Informative vs Non-informative)
- Seuils dynamiques par organe (Liver 0.40, Lung 0.35, Os 0.20)

### Option 3: Tests IHM Gradio

**Validation workflow pathologiste:**
- Interface clinique (`app_pathologist.py`)
- Overlays simplifiés (4 checkboxes)
- Badge Confiance IA (Élevée/Modérée/Faible)
- Export PDF rapport clinique

**Lancement:**
```bash
# R&D Cockpit
./scripts/run_cockpit.sh --preload --organ Lung

# Interface Pathologiste
./scripts/run_pathologist.sh --preload --organ Breast
```

### Option 4: Review/Documentation

- Audit code (duplication, SSOT)
- Mise à jour diagrammes architecture
- Documentation API export Phase 4
- Tests unitaires critiques

---

## 🔍 Vérifications Avant de Commencer

```bash
# 1. Vérifier branche
git branch
# Attendu: * claude/review-project-context-JQPxq

# 2. Vérifier status
git status
# Attendu: On branch claude/review-project-context-JQPxq, nothing to commit

# 3. Vérifier à jour avec main
git log origin/main..HEAD --oneline
# Attendu: vide (déjà à jour)

# 4. Lister fichiers clés
ls -la CLAUDE.md claude_history.md docs/specs/V14_WSI_TRIAGE_SPEC.md
```

---

## 📊 Constantes Importantes

### Normalisation H-optimus-0

```python
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224
```

### Structure Features

```python
features (B, 261, 1536):
├── features[:, 0, :]       # CLS token → OrganHead
├── features[:, 1:5, :]     # 4 Register tokens (IGNORER)
└── features[:, 5:261, :]   # 256 Patch tokens → HoVer-Net
```

### 5 Familles HoVer-Net

| Famille | Organes | Samples | AJI Actuel | Gap vs 0.68 |
|---------|---------|---------|------------|-------------|
| **Glandular** | Breast, Prostate, Thyroid, Pancreatic, Adrenal_gland | 3391 | **0.6566** | -3.4% |
| **Digestive** | Colon, Stomach, Esophagus, Bile-duct | 2430 | 0.6160 | -9.4% |
| **Urologic** | Kidney, Bladder, Testis, Ovarian, Uterus, Cervix | 1101 | **0.6743** | -0.8% |
| **Respiratory** | Lung, Liver | 408 | **0.6872** ✅ | +1.1% |
| **Epidermal** | Skin, HeadNeck | 574 | 0.6203 | -8.8% |

### Paramètres Watershed Optimisés (V13 Production)

| Famille | np_threshold | min_size | beta | min_distance | AJI |
|---------|--------------|----------|------|--------------|-----|
| **Respiratory** | 0.40 | 30 | 0.50 | 5 | **0.6872** ✅ |
| **Urologic** | 0.45 | 30 | 0.50 | 2 | **0.6743** |
| **Glandular** | 0.40 | 50 | 0.50 | 3 | **0.6566** |
| Epidermal | 0.45 | 20 | 1.00 | 3 | 0.6203 |
| Digestive | 0.45 | 60 | 2.00 | 5 | 0.6160 |

**Override Organ-Specific (exemples):**
- **Breast:** `{"np_threshold": 0.50, "min_size": 30, "beta": 0.50, "min_distance": 2}`
- **Kidney:** `{"min_distance": 1}` (le plus agressif, grâce à H-channel)
- **Adrenal_gland:** `{"min_size": 50, "min_distance": 4}` (record AJI 0.7236)

---

## 🚀 Commandes Rapides

### Pipeline Complet V13 (Exemple Respiratory)

```bash
# 1. Générer Smart Crops (Raw Images - RECOMMANDÉ)
python scripts/preprocessing/prepare_v13_smart_crops.py \
    --family respiratory \
    --pannuke_dir /chemin/vers/PanNuke \
    --max_samples 5000

# 2. Vérifier données
python scripts/validation/verify_v13_smart_crops_data.py --family respiratory --split train

# 3. Extraire features H-optimus-0
python scripts/preprocessing/extract_features_v13_smart_crops.py --family respiratory --split train
python scripts/preprocessing/extract_features_v13_smart_crops.py --family respiratory --split val

# 4. Entraînement FPN Chimique
python scripts/training/train_hovernet_family_v13_smart_crops.py \
    --family respiratory \
    --epochs 60 \
    --use_hybrid \
    --use_fpn_chimique \
    --use_h_alpha

# 5. Évaluation AJI
python scripts/evaluation/test_v13_smart_crops_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth \
    --family respiratory \
    --n_samples 50 \
    --use_hybrid \
    --use_fpn_chimique \
    --np_threshold 0.40 \
    --min_size 30 \
    --min_distance 5
```

### Optimisation Watershed (Organ-Level)

```bash
# Phase 1: Exploration rapide (20 samples, 400 configs)
python scripts/evaluation/optimize_watershed_aji.py \
    --checkpoint models/checkpoints_v13_smart_crops/hovernet_respiratory_v13_smart_crops_hybrid_fpn_best.pth \
    --family respiratory \
    --organ Liver \
    --n_samples 20

# Phase 2: Copier-coller la commande générée automatiquement (100 samples, ~81 configs)
# La commande optimale est affichée à la fin de Phase 1
```

### Lancement IHM

```bash
# R&D Cockpit (développeurs)
./scripts/run_cockpit.sh --preload --organ Lung

# Interface Pathologiste (cliniciens)
./scripts/run_pathologist.sh --preload --organ Breast
```

---

## 💡 Insights Techniques Critiques

### 1. Le Paradoxe du Beta (Liver vs Lung)

- **Liver (β=2.0):** Noyaux vésiculeux (clairs) → Beta élevé (ignore micro-variations NP, se focalise sur HV)
- **Lung (β=0.5):** Noyaux denses, débris inflammatoires → Beta bas (pondère plus NP)

**Principe:** Plus un noyau est "vésiculeux", plus β doit être élevé.

### 2. Efficacité Injection H-Channel

- **Ruifrok:** Vecteurs fixes (Beer-Lambert physique) → Préserve texture chromatinienne
- **Permet:** `min_distance=2` sans sur-fusion (impossible sans H-channel)
- **Vs Macenko:** Macenko adaptatif déplace Éosine vers H → "fantômes" cytoplasme → **-4.3% AJI**

### 3. Stratégie Smart Crops (Split-First-Then-Rotate)

- **5 crops par image 256×256:** Centre + 4 coins avec rotations déterministes
- **CRITIQUE:** Split train/val par `source_image_ids` AVANT rotations → ZERO data leakage
- **HV Maps:** Rotation spatiale ≠ Rotation vectorielle → Correction component swapping obligatoire

### 4. Pistes R&D Prioritaires

| Piste | Faisabilité | Impact | Statut |
|-------|-------------|--------|--------|
| **Watershed adaptatif par incertitude** | Haute | Haut | ⭐ Prioritaire |
| **NC-based Beta-Switch (Auto-Tuner)** | Haute | Haut | ⭐⭐ Prioritaire |
| **Watershed itératif par densité** | Haute | Haut | ⭐ Prioritaire |
| **Attention spatiale via Patch Tokens** | Moyenne-Haute | Haut | ⭐ Exploratoire |

---

## ✅ Checklist Avant Commits

```bash
# 1. Vérifier que le code utilise les modules partagés
grep -r "from src\." mon_script.py

# 2. Pas de duplication de constantes
grep -r "0.707223\|HOPTIMUS" mon_script.py

# 3. Pas de duplication de listes d'organes
grep -r "Adrenal_gland.*Bile-duct" mon_script.py

# 4. Flags FPN Chimique corrects
grep -r "use_hybrid.*use_fpn_chimique" mon_script.py

# 5. Documentation à jour
git diff CLAUDE.md

# 6. Commit messages descriptifs
git log -1 --pretty=%B
```

---

## 🎯 Objectif Immédiat

**Question à l'utilisateur:** Que souhaitez-vous faire maintenant?

1. **Amélioration AJI** des familles Epidermal/Digestive (pistes R&D documentées)
2. **Implémentation v14.0** (CleaningNet, triage WSI)
3. **Tests IHM** (validation workflow pathologiste)
4. **Review/Documentation** (audit code, mise à jour diagrammes)
5. **Autre tâche spécifique**

---

## 📞 Environnement

| Composant | Version |
|-----------|---------|
| OS | WSL2 Ubuntu 24.04.2 LTS |
| GPU | RTX 4070 SUPER (12.9 GB VRAM) |
| Python | 3.10 (Miniconda) |
| PyTorch | 2.6.0+cu124 |
| Conda env | `cellvit` |

**Working directory:** `/home/user/path`

**Git repo:** `ecolealgerienne-ui/path`

---

## 🔗 Références

- **H-optimus-0:** https://huggingface.co/bioptimus/H-optimus-0
- **HoVer-Net:** Graham et al., Medical Image Analysis 2019
- **PanNuke:** https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
- **Ruifrok Deconvolution:** Ruifrok & Johnston, Analytical and Quantitative Cytology and Histology 2001
- **Nottingham Grade:** Elston & Ellis, Histopathology 1991

---

**Version:** 2026-01-18
**Auteur:** Session précédente (claude/review-project-context-JQPxq)
**Statut:** ✅ Prêt pour nouvelle session
