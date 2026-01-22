# Prompt Nouvelle Session — V15.2 Cytology Pipeline

> **Date:** 2026-01-22
> **Version:** V15.2-Lite (POC)
> **Statut:** ✅ Consensus Final — Architecture documentée, prêt pour Phase 0

---

## 🆕 V15.2 — CHANGEMENT DE PARADIGME

**V15.2 remplace V14** avec une architecture industrielle:

| Composant | V14 | V15.2 |
|-----------|-----|-------|
| Détection | CellPose | **YOLO** |
| Segmentation | CellPose | **HoVerNet-lite** |
| Encoder | H-Optimus (fixe) | **Benchmark 5 encoders** |
| Fusion | Concat simple | **Gated Feature Fusion** |
| Sécurité | — | **Conformal + OOD** |
| Dataset POC | SIPaKMeD | **APCData uniquement** |

**Document de référence:** `docs/cytology/V15_ARCHITECTURE_SPEC.md`

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

## 📊 CONTEXTE ACTUEL — V15.2 Cytology

### État du Projet

| Phase | Status | Description |
|-------|--------|-------------|
| **V14 (Legacy)** | ✅ DONE | POC SIPaKMeD, CellPose validé sur APCData |
| **V15.2 Phase 0** | ⏳ À FAIRE | Benchmark 5 encoders (7-10 jours) |
| **V15.2 Phase 1-3** | ⏳ PENDING | Architecture complète (12 semaines) |

### Dataset POC

**APCData uniquement:** 425 images, 3,619 cellules (Bethesda 6 classes)

| Aspect | Valeur |
|--------|--------|
| Format | LBC (Liquid-Based Cytology) |
| Annotations | Bounding boxes + Points nucleus |
| Classes | NILM, ASCUS, ASCH, LSIL, HSIL, SCC |

---

## 🎯 PROCHAINE ÉTAPE IMMÉDIATE

### Phase 0: Benchmark Encoder (7-10 jours)

**Objectif:** Sélection data-driven de l'encoder (pas de dogme)

```bash
python scripts/cytology/benchmark_encoders.py \
    --dataset apcdata \
    --encoders h-optimus,uni,phikon-v2,convnext-base,resnet50 \
    --method linear_probe \
    --cv_folds 5 \
    --output_dir reports/encoder_benchmark
```

### Encoders à tester

| Encoder | Dims | Attendu (littérature) |
|---------|------|----------------------|
| ResNet50 | 2048 | 70-80% (baseline) |
| H-Optimus | 1536 | 75-85% |
| UNI | 1024 | 78-88% |
| Phikon-v2 | 768 | 80-90% |
| ConvNeXt-Base | 1024 | 80-92% |

### Règle de Décision

```
1. Sélectionner encoder avec meilleure Balanced Accuracy
2. Si écart frozen vs fine-tuned > 5% → Full fine-tuning
3. Sinon → LoRA
```

### Métriques à Collecter

| Métrique | Priorité |
|----------|----------|
| **Balanced Accuracy** | 🔴 CRITIQUE |
| F1-score (macro) | 🔴 CRITIQUE |
| ASC-H Recall | 🔴 CRITIQUE |
| HSIL Recall | 🔴 CRITIQUE |
| ECE (calibration) | 🟡 Important |

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
1. ✅ V15.2 Architecture documentée (consensus final)
2. ✅ Dataset POC défini: APCData uniquement
3. ⏳ Phase 0 (Benchmark Encoder) **À DÉMARRER**

**Action immédiate:**
- Lancer benchmark encoders sur APCData
- Collecter Balanced Accuracy pour 5 encoders
- Décision data-driven sur encoder final

**Objectif V15.2 POC:**
- Démontrer architecture fonctionne
- Sensibilité ≥98% sur cellules anormales
- Pipeline: Image → YOLO → HoVerNet-lite → Encoder → GFF → MLP → Sécurité

**Documents clés:**
- `docs/cytology/V15_ARCHITECTURE_SPEC.md` — Specs complètes
- `docs/cytology/datasets/APCDATA.md` — Dataset POC
- `scripts/cytology/benchmark_encoders.py` — Script benchmark

---

**Dernière mise à jour:** 2026-01-22
**Session actuelle:** Documentation V15.2 finalisée, consensus établi
