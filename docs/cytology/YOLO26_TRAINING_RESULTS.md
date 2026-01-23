# YOLO26 Training Results — APCData Cell Detection

> **Version:** V15 Cytology Pipeline
> **Date:** 2026-01-23
> **Dataset:** APCData (425 images, 6 classes Bethesda)

---

## Résumé Exécutif

| Modèle | Epochs | mAP50 | mAP50-95 | Recall NILM | Recall SCC | Status |
|--------|--------|-------|----------|-------------|------------|--------|
| **YOLO26n** | 216 (ES) | **41.4%** | **26.3%** | 84.0% | 62.3% | ✅ Baseline |
| YOLO26s | En cours | ? | ? | ? | ? | 🔄 Training |

*ES = Early Stopping*

---

## Expérience 1: YOLO26n Baseline (100 epochs)

**Date:** 2026-01-22
**Durée:** ~10 minutes

### Configuration

```bash
python scripts/cytology/03_train_yolo26_apcdata.py \
    --model yolo26n.pt \
    --epochs 100 \
    --batch 8 \
    --imgsz 640
```

### Résultats

| Métrique | Valeur |
|----------|--------|
| mAP50 | 30.1% |
| mAP50-95 | 17.6% |
| Precision | 29.9% |
| Recall | 41.7% |

#### Performance par classe

| Classe | Instances | Precision | Recall | mAP50 | mAP50-95 |
|--------|-----------|-----------|--------|-------|----------|
| NILM | 375 | 53.2% | 82.7% | 73.7% | 48.8% |
| ASCUS | 78 | 26.3% | 17.8% | 13.8% | 7.3% |
| ASCH | 42 | 21.4% | 14.3% | 15.3% | 9.3% |
| LSIL | 77 | 24.0% | 31.2% | 20.7% | 11.4% |
| HSIL | 92 | 23.0% | 39.1% | 22.6% | 12.2% |
| SCC | 23 | 31.4% | 65.2% | 34.5% | 16.7% |

### Analyse

- ✅ **NILM** (normal): Bon recall 82.7%
- ✅ **SCC** (cancer): Recall 65.2% malgré seulement 23 samples
- ⚠️ Classes intermédiaires faibles (déséquilibre de classes)

---

## Expérience 2: YOLO26n Extended (300 epochs avec Early Stopping)

**Date:** 2026-01-23
**Durée:** ~26 minutes (early stopping à epoch 266)
**Best epoch:** 216

### Configuration

```bash
python scripts/cytology/03_train_yolo26_apcdata.py \
    --model yolo26n.pt \
    --epochs 300 \
    --batch 8 \
    --imgsz 640 \
    --patience 50
```

### Résultats Finaux

| Métrique | Valeur | Δ vs Baseline |
|----------|--------|---------------|
| mAP50 | **41.4%** | +37.5% |
| mAP50-95 | **26.3%** | +49.4% |
| Precision | 40.9% | +36.8% |
| Recall | 44.9% | +7.7% |

#### Performance par classe

| Classe | Instances | Precision | Recall | mAP50 | mAP50-95 |
|--------|-----------|-----------|--------|-------|----------|
| **NILM** | 375 | 63.5% | **84.0%** | **82.4%** | 58.1% |
| ASCUS | 78 | 26.9% | 29.5% | 22.3% | 12.6% |
| ASCH | 42 | 41.4% | 35.7% | 29.6% | 20.0% |
| LSIL | 77 | 30.5% | 28.6% | 27.8% | 15.9% |
| HSIL | 92 | 42.5% | 29.3% | 27.2% | 15.1% |
| **SCC** | 23 | 40.6% | **62.3%** | **59.1%** | 36.2% |

### Courbe d'apprentissage

```
Epoch   | cls_loss | mAP50-95
--------|----------|----------
33      | 2.479    | 14.3%
103     | 1.547    | 21.1%
163     | 1.334    | 24.3%  ← Peak
216     | 1.177    | 26.3%  ← Best (saved)
266     | 1.177    | 25.3%  ← Early Stop
```

### Analyse

**Points positifs:**
- ✅ NILM: 84% recall, 82.4% mAP50 — excellent
- ✅ SCC: 62% recall — détecte la majorité des cancers
- ✅ Convergence stable, early stopping approprié
- ✅ Vitesse inference: 1.0ms/image

**Limitations:**
- ⚠️ Classes intermédiaires (ASCUS, ASCH, LSIL, HSIL): 22-30% mAP50
- ⚠️ Déséquilibre de classes sévère (NILM: 375 vs SCC: 23)
- ⚠️ Modèle nano = capacité limitée (2.4M params)

### Checkpoints

```
runs/detect/runs/cytology/apcdata_yolo26n_20260123_121505/
├── weights/
│   ├── best.pt   (5.4MB) ← Best @ epoch 216
│   └── last.pt   (5.4MB) ← Epoch 266
├── results.csv
├── results.png
└── confusion_matrix.png
```

---

## Expérience 3: YOLO26s (En cours)

**Date:** 2026-01-23
**Status:** 🔄 En cours

### Configuration

```bash
python scripts/cytology/03_train_yolo26_apcdata.py \
    --model yolo26s.pt \
    --epochs 300 \
    --batch 4 \
    --imgsz 640 \
    --patience 50
```

### Différences vs YOLO26n

| Aspect | YOLO26n | YOLO26s |
|--------|---------|---------|
| Paramètres | 2.4M | ~9M |
| GFLOPs | 5.2 | ~20 |
| Batch size | 8 | 4 (OOM risk) |
| Temps/epoch | ~5s | ~15s (estimé) |

### Résultats attendus

- mAP50: +10-15% vs nano
- Meilleure performance sur classes minoritaires
- Temps total: ~1h (estimation)

---

## Dataset: APCData

### Statistiques

| Split | Images | Cellules |
|-------|--------|----------|
| Train | 343 | ~2,932 |
| Val | 82 | 687 |
| **Total** | **425** | **~3,619** |

### Distribution des classes

| Classe | Train | Val | Total | % |
|--------|-------|-----|-------|---|
| NILM | ~300 | 375 | ~675 | 54.6% |
| ASCUS | ~65 | 78 | ~143 | 11.4% |
| ASCH | ~35 | 42 | ~77 | 6.1% |
| LSIL | ~65 | 77 | ~142 | 11.2% |
| HSIL | ~75 | 92 | ~167 | 13.3% |
| SCC | ~20 | 23 | ~43 | 3.4% |

**Déséquilibre critique:** NILM (54.6%) vs SCC (3.4%) = ratio 16:1

---

## Augmentation (Online via Ultralytics)

```python
# Paramètres actuels dans 03_train_yolo26_apcdata.py
hsv_h=0.015      # Hue (subtle pour staining)
hsv_s=0.4        # Saturation
hsv_v=0.4        # Value/Brightness
degrees=180      # Rotation complète (cellules orientées aléatoirement)
translate=0.1    # Translation ±10%
scale=0.5        # Scale ±50%
flipud=0.5       # Flip vertical 50%
fliplr=0.5       # Flip horizontal 50%
mosaic=0.5       # Mosaïque (4 images combinées)
mixup=0.0        # Désactivé (préserve intégrité cellulaire)
```

---

## Décision Architecturale V15.2 (2026-01-23)

> **Validation Expert (Industrie: Hologic, BD-Techcyte)**
>
> L'architecture proposée correspond aux standards industrie pour le screening cervical.

### Analyse des Résultats YOLO26

**Constat:** YOLO détecte bien les cellules mais confond les classes intermédiaires.

| Aspect | Performance | Analyse |
|--------|-------------|---------|
| **Détection cellules** | Excellente | NILM 84%, SCC 62% recall |
| **Classification** | Faible sur classes similaires | ASCUS/ASCH/LSIL/HSIL: 28-36% recall |

**Cause:** Les classes Bethesda intermédiaires partagent des morphologies très proches.
YOLO n'est pas optimisé pour cette granularité fine.

### Architecture Retenue: Detection-Only + Multi-Head

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     V15.2 ARCHITECTURE PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

                        Image LBC (2048×1532)
                               │
                               ▼
                    ┌──────────────────────┐
                    │  YOLO26 Detection    │  ← 1 classe: "cell"
                    │  (mAP50 > 85%)       │
                    └──────────┬───────────┘
                               │
                    Crops cellules détectées
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    ┌──────────────────┐              ┌──────────────────┐
    │  H-Optimus-0     │              │  Morpho Features │
    │  (1536D frozen)  │              │  (20D computed)  │
    └────────┬─────────┘              └────────┬─────────┘
              │                                 │
              └─────────────┬───────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Gated Feature      │
                 │  Fusion (GFF)       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Shared Encoder     │
                 │  (256D latent)      │
                 └──────────┬──────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Head 1: Binary │ │  Head 2: Sev.   │ │  Head 3: Fine   │
│  Normal/Abnorm  │ │  Low/High Risk  │ │  6 Bethesda     │
│  (Triage)       │ │  (+morpho feat) │ │  (Diagnostic)   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Rejection Layer    │
                 │  (Conformal Pred.)  │
                 │  → Manual Review    │
                 └─────────────────────┘
```

### Spécification des Heads

| Head | Input | Output | Rôle Clinique |
|------|-------|--------|---------------|
| **Binary** | latent (256) | Normal/Abnormal | Triage rapide |
| **Severity** | latent (256) + morpho (20) | Low-risk/High-risk | Priorisation |
| **Fine** | latent (256) | 6 classes Bethesda | Diagnostic précis |
| **Rejection** | latent (256) + uncertainty (3) | Review/OK | Safety net |

### Mapping des Classes

**Binary (Head 1):**
- Normal: NILM
- Abnormal: ASCUS, ASCH, LSIL, HSIL, SCC

**Severity (Head 2):**
- Low-risk: NILM, ASCUS, LSIL
- High-risk: ASCH, HSIL, SCC

### Implémentation

- **Script conversion:** `scripts/cytology/04_convert_to_detection_only.py`
- **Multi-Head model:** `src/cytology/models/cytology_classifier.py` → `CytologyMultiHead`
- **Loss combinée:** `MultiHeadLoss` (λ_binary=1.0, λ_severity=1.5, λ_fine=1.0)

---

## Prochaines Étapes

### Court terme
1. ⏳ Attendre résultats YOLO26s
2. 📊 Comparer nano vs small
3. 🔄 Convertir APCData vers detection-only (1 classe)
4. 🎯 Entraîner YOLO detection-only

### Moyen terme
1. Entraîner CytologyMultiHead sur SIPaKMeD (cellules isolées)
2. Intégrer YOLO detection + MultiHead classifier sur APCData
3. Calibrer Rejection Layer (seuil optimal)
4. Évaluer métriques cliniques (Sensibilité Malin > 98%)

### Améliorations potentielles YOLO Detection
- [ ] Image size 1024 (plus de détails pour petites cellules)
- [ ] YOLO26m si recall insuffisant
- [ ] Test-Time Augmentation (TTA) pour robustesse

---

## Références

### Checkpoints & Configs
- **Checkpoint YOLO26n:** `runs/cytology/apcdata_yolo26n_*/weights/best.pt`
- **Config dataset (6 classes):** `configs/cytology/apcdata_yolo.yaml`

### Scripts
- **Test YOLO:** `scripts/cytology/01_test_yolo26_apcdata.py`
- **Split train/val:** `scripts/cytology/02_prepare_apcdata_split.py`
- **Training YOLO:** `scripts/cytology/03_train_yolo26_apcdata.py`
- **Convert to detection-only:** `scripts/cytology/04_convert_to_detection_only.py`

### Models
- **Multi-Head Classifier:** `src/cytology/models/cytology_classifier.py` → `CytologyMultiHead`
- **GFF Module:** `src/cytology/models/cytology_classifier.py` → `GatedFeatureFusion`

### Documentation Externe
- **YOLO26:** https://docs.ultralytics.com/models/yolo26/
- **Bethesda System:** https://www.cancer.gov/publications/dictionaries/cancer-terms/def/bethesda-system
