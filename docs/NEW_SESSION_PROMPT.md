# Prompt Nouvelle Session — V14 Cytology Pipeline

> **Date:** 2026-01-21
> **Branche:** `claude/retrieve-project-context-lbkVY`
> **Statut:** Phase 2 APCData — CellPose validé (90.8%), Script E2E créé

---

## 🚨 RÈGLES CRITIQUES (À RESPECTER ABSOLUMENT)

### 1. Utilise TOUJOURS l'existant
```
- NE JAMAIS créer un nouveau fichier si un existant peut être modifié
- NE JAMAIS dupliquer du code — importer depuis src/ ou scripts existants
- VÉRIFIER avec grep/glob si une fonction existe déjà avant de la coder
```

### 2. On ne réinvente pas la roue
```
- Les scripts dans scripts/cytology/ sont la référence
- Les constantes sont dans src/constants.py et src/preprocessing/
- Les algorithmes critiques sont dans src/postprocessing/, src/metrics/, etc.
```

### 3. Pas d'initiatives sans raison
```
- NE PAS ajouter de features non demandées
- NE PAS refactorer du code qui fonctionne
- NE PAS changer les paramètres validés sans demande explicite
```

### 4. S'inspirer des scripts existants
```
scripts/cytology/
├── 00_preprocess_sipakmed.py      # Preprocessing référence
├── 01_extract_embeddings_gt.py    # H-Optimus extraction référence
├── 02_compute_morphometry.py      # Morphometry 20 features référence
├── 03_train_mlp_classifier.py     # MLP architecture référence
├── 04_evaluate_cytology.py        # Évaluation Safety First référence
├── 05_validate_cellpose_apcdata.py # CellPose validation ✅ VALIDÉ
└── 06_end_to_end_apcdata.py       # Pipeline E2E ✅ CRÉÉ (à tester)
```

### 5. Mettre à jour CLAUDE.md
```
Toute décision importante, paramètre validé, ou résultat doit être documenté
dans CLAUDE.md pour les futures sessions.
```

---

## 📊 CONTEXTE ACTUEL — V14 Cytology

### État du Projet

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: SIPaKMeD (POC)** | ✅ DONE | Sensibilité 99.26%, Kappa 0.7205 |
| **Phase 2: APCData** | 🔄 EN COURS | CellPose validé, E2E script créé, **À TESTER** |

### Résultats Validation CellPose (APCData) ✅

**Dataset:** 425 images, 3619 cellules annotées (Bethesda: NILM, ASCUS, ASCH, LSIL, HSIL, SCC)

**Configuration Optimale Validée:**
```python
CELLPOSE_CONFIG = {
    'diameter': 60,
    'flow_threshold': 0.4,
    'cellprob_threshold': 0.0,
    'min_area': 400,      # Filtre débris/lymphocytes
    'max_area': 100000,
    'max_distance': 120   # Tolérance matching GT
}
```

**Résultats (Full Dataset n=425):**

| Métrique | Valeur | Cible | Status |
|----------|--------|-------|--------|
| **Abnormal Detection Rate** | **90.8%** | ≥98% | ⚠️ ACCEPTABLE |
| Detection Rate (All) | 85.5% | ≥90% | - |
| ASCUS | 94.0% | - | ✅ |
| ASCH | 94.5% | - | ✅ |
| LSIL | 91.0% | - | ✅ |
| HSIL | 87.6% | - | ⚠️ |
| SCC | 87.2% | - | ⚠️ |

---

## 🎯 PROCHAINE ÉTAPE IMMÉDIATE

### Exécuter le Pipeline End-to-End

```bash
python scripts/cytology/06_end_to_end_apcdata.py \
    --data_dir data/raw/apcdata/APCData_YOLO \
    --mlp_checkpoint models/cytology/mlp_classifier_best.pth \
    --n_samples 50 \
    --output_dir reports/end_to_end_apcdata
```

### Prérequis
- ✅ APCData_YOLO téléchargé (`data/raw/apcdata/APCData_YOLO/`)
- ⚠️ MLP checkpoint entraîné sur SIPaKMeD (`models/cytology/mlp_classifier_best.pth`)
- ⚠️ H-Optimus-0 accessible (HuggingFace login)

### Si le MLP checkpoint n'existe pas

Le MLP doit être entraîné sur SIPaKMeD (Phase 1) avant de lancer le E2E:

```bash
# 1. Préprocessing SIPaKMeD
python scripts/cytology/00_preprocess_sipakmed.py \
    --raw_dir data/raw/sipakmed/pictures \
    --output_dir data/processed/sipakmed

# 2. Extraction embeddings H-Optimus
python scripts/cytology/01_extract_embeddings_gt.py \
    --data_dir data/processed/sipakmed \
    --output_dir data/embeddings/sipakmed

# 3. Morphometry
python scripts/cytology/02_compute_morphometry.py \
    --data_dir data/processed/sipakmed \
    --embeddings_dir data/embeddings/sipakmed \
    --output_dir data/features/sipakmed

# 4. Train MLP
python scripts/cytology/03_train_mlp_classifier.py \
    --features_dir data/features/sipakmed \
    --output_dir models/cytology \
    --epochs 100 \
    --use_focal_loss
```

### Métriques Attendues (E2E)

| Métrique | Cible | Priorité |
|----------|-------|----------|
| **Sensibilité (Abnormal)** | ≥98% | 🔴 CRITIQUE |
| **Cohen's Kappa** | ≥0.80 | 🔴 CRITIQUE |
| Spécificité | ≥60% | 🟢 Secondaire |

---

## 📁 STRUCTURE DONNÉES

```
data/raw/apcdata/
├── APCData_YOLO/          # ✅ UTILISER CELUI-CI
│   ├── images/            # 425 images JPG (2048×1532)
│   ├── labels/            # Annotations YOLO (.txt)
│   └── classes.txt        # NILM, ASCUS, ASCH, LSIL, HSIL, SCC
│
└── APCData_points/        # ❌ NE PAS UTILISER (noms hashés)

models/cytology/
└── mlp_classifier_best.pth  # MLP entraîné sur SIPaKMeD (Phase 1)

reports/
├── cellpose_apcdata_validation/  # ✅ Résultats CellPose (complet)
└── end_to_end_apcdata/           # 📝 À générer (E2E)
```

---

## 🔧 CONSTANTES IMPORTANTES

### H-Optimus-0
```python
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
HOPTIMUS_INPUT_SIZE = 224
# Output: CLS token (1536) + 256 patches (ignorés pour cytologie)
```

### MLP Architecture
```python
# Input: 1556 dims (1536 CLS + 20 morphometry)
# Hidden: [512, 256, 128]
# Output: 7 classes SIPaKMeD
# Dropout: 0.3, BatchNorm: True
```

### Mapping Classes

**SIPaKMeD (MLP output):**
```python
SIPAKMED_CLASSES = [
    'normal_columnar',      # 0 → Normal
    'normal_intermediate',  # 1 → Normal
    'normal_superficiel',   # 2 → Normal
    'light_dysplastic',     # 3 → Abnormal
    'moderate_dysplastic',  # 4 → Abnormal
    'severe_dysplastic',    # 5 → Abnormal
    'carcinoma_in_situ'     # 6 → Abnormal
]
```

**Bethesda (APCData GT):**
```python
BETHESDA_CLASSES = ['NILM', 'ASCUS', 'ASCH', 'LSIL', 'HSIL', 'SCC']
# NILM → Normal
# Tous les autres → Abnormal
```

---

## 📚 DOCUMENTATION CLÉ

| Document | Description |
|----------|-------------|
| `CLAUDE.md` | **LIRE EN PREMIER** — Contexte projet complet |
| `docs/cytology/V14_PRODUCTION_PIPELINE.md` | Pipeline production avec params validés |
| `docs/cytology/V14_CYTOLOGY_BRANCH.md` | Specs complètes V14 |
| `scripts/cytology/README.md` | Guide scripts avec commandes |

---

## ⚠️ POINTS D'ATTENTION CRITIQUES

### 1. APCData_YOLO vs APCData_points
```
APCData_YOLO: Images avec noms descriptifs → UTILISER
APCData_points: Images avec noms hashés → NE PAS UTILISER
```

### 2. Précision basse = NORMAL
```
La précision CellPose (~7%) est ATTENDUE car:
- APCData annote seulement un sous-ensemble de cellules
- CellPose détecte TOUTES les cellules (correctement)
- Le classifieur MLP filtrera les cellules normales

Métrique importante: ABNORMAL DETECTION RATE (90.8%)
```

### 3. Safety First
```
JAMAIS rater un cancer > Éviter faux positifs
Sensibilité > Précision
Target: Sensibilité ≥98%
```

### 4. CellPose sur cellules isolées
```
CellPose = optimisé pour GROUPES cellulaires (tissus)
Sur cellules isolées (SIPaKMeD) → sur-segmentation
Solution Phase 1: Masques GT
Solution Phase 2: CellPose sur lames réelles ✅
```

---

## 🔄 COMMITS RÉCENTS (Session 2026-01-21)

```
e207425 docs(v14-cyto): Update README with validated CellPose params and e2e script
77440c6 feat(v14-cyto): Add end-to-end pipeline validation script for APCData
344ccbd docs(v14-cyto): Update CellPose config with validated parameters
d316cc5 feat(v14-cyto): Add abnormal detection rate metric (Safety First)
c25c046 feat(v14-cyto): Add area-based filtering to CellPose validation
```

---

## 🎯 RÉSUMÉ POUR NOUVELLE SESSION

**Situation actuelle:**
1. ✅ CellPose validé sur APCData (90.8% abnormal detection)
2. ✅ Script `06_end_to_end_apcdata.py` créé
3. ⏳ E2E pipeline **PAS ENCORE TESTÉ** (besoin checkpoint MLP)

**Action immédiate:**
- Vérifier si `models/cytology/mlp_classifier_best.pth` existe
- Si oui → Lancer `06_end_to_end_apcdata.py`
- Si non → Entraîner MLP sur SIPaKMeD d'abord (scripts 00-03)

**Objectif final V14 Cytology:**
- Sensibilité ≥98% sur cellules anormales
- Cohen's Kappa ≥0.80
- Pipeline production: Image → CellPose → H-Optimus → MLP → Rapport

---

**Dernière mise à jour:** 2026-01-21
**Session précédente:** Validation CellPose APCData complète, script E2E créé
