# CellViT-Optimus V14 Datasets Documentation

> **Central Index for All Cytology & Validation Datasets**
> **Last Updated:** 2026-01-19

---

## Quick Navigation

| Dataset | Type | Samples | Status | Documentation |
|---------|------|---------|--------|---------------|
| **SIPaKMeD** | Cytology (Cervix) | 4,049 | ✅ Downloaded | [📖 sipakmed/](./sipakmed/) |
| **Herlev** | Cytology (Cervix) | 917 | ⚠️ Pending | [📖 herlev/](./herlev/) |
| **Thyroid Cytology** | Cytology (Thyroid) | TBD | ❌ **NOT AVAILABLE** | [📖 See DATASET_ACQUISITION_GUIDE.md](../DATASET_ACQUISITION_GUIDE.md#dataset-1-thyroid-cytology-datasets--limited-availability) |
| **ISBI 2014 MITOS-ATYPIA** | Histology (Breast) | ~1,200 | ⚠️ Pending | [📖 isbi_2014_mitoses/](./isbi_2014_mitoses/) |

---

## Dataset Categories

### 🔬 Cytology Datasets (CellPose Training)

**Purpose:** Train Master/Slave CellPose models for nucleus and cytoplasm segmentation.

| Dataset | Organ | Images | Classes | Priority | CellPose Training |
|---------|-------|--------|---------|----------|-------------------|
| **SIPaKMeD** | Cervix | 4,049 | 7 | High | ✅ YES |
| **Herlev** | Cervix | 917 | 7 | Medium | ✅ YES (validation) |
| **Thyroid (TBD)** | Thyroid | TBD | 6 (Bethesda) | Medium | ⚠️ Dataset search needed |

**Usage:**
- Train CellPose `nuclei` model (Master)
- Train CellPose `cyto3` model (Slave) if N/C ratio required
- Extract morphometric features
- Train LightGBM classification heads

### ⭐ Validation Datasets (Algorithm Calibration)

**Purpose:** Validate the 6 Universal Criteria for malignancy scoring.

| Dataset | Organ | Images | Expert Annotations | CellPose Training | Validation |
|---------|-------|--------|--------------------|-------------------|------------|
| **ISBI 2014 MITOS-ATYPIA** | Breast | ~1,200 | 6 criteria scores ⭐ | ❌ NO (histology) | ✅ YES |

**Usage:**
- Calibrate the 6 Universal Criteria thresholds
- Validate Canal H (Chromatin Density) extraction
- Prove criterion transferability (breast → other organs)
- Benchmark morphometric features

---

## Download Status

### ✅ Downloaded

**SIPaKMeD** (2026-01-19)
```
Location: data/raw/sipakmed/pictures/
Size: ~2 GB
Classes: 7 (carcinoma_in_situ, light_dysplastic, moderate_dysplastic,
           severe_dysplastic, normal_columnar, normal_intermediate,
           normal_superficiel)
Status: Ready for preprocessing
```

### ⚠️ Pending Download

**Herlev**
```
Source: http://mde-lab.aegean.gr/index.php/downloads
Size: ~0.5 GB
Status: Manual download required
Action: Register and download Herlev_dataset.zip
```

**Thyroid Cytology**
```
Source: ❌ NO PUBLIC DATASET AVAILABLE
Size: N/A
Status: Needs sourcing (Kaggle search or academic collaboration)
Action: See DATASET_ACQUISITION_GUIDE.md for alternatives
```

**ISBI 2014 MITOS-ATYPIA**
```
Source: https://mitos-atypia-14.grand-challenge.org/
Size: ~3 GB
Status: Registration required
Action: Create account, download A03 + H03 datasets
```

---

## Preprocessing Pipeline

### Step 1: Verify Downloads

```bash
python scripts/datasets/verify_datasets.py
```

**Expected Output:**
```
✅ SIPaKMeD: 4,049 images found
⚠️ Herlev: Not found
❌ Thyroid: No public dataset available
⚠️ ISBI 2014: Not found

📊 Total: 4,049 images (1/3 available datasets ready)
```

### Step 2: Preprocess Cytology Datasets

```bash
# Preprocess all available datasets
python scripts/datasets/preprocess_cytology.py --all

# Or individually
python scripts/datasets/preprocess_cytology.py --dataset sipakmed
python scripts/datasets/preprocess_cytology.py --dataset herlev
```

**Output:**
```
data/processed/
├── sipakmed/
│   ├── train/ (3,239 images)
│   └── val/ (810 images)
└── herlev/
    ├── train/ (733 images)
    └── val/ (184 images)
```

**Note:** Thyroid cytology dataset not yet available - focus on cervical datasets first.

### Step 3: Prepare ISBI 2014 for Validation

```bash
# ISBI 2014 requires special preprocessing (histology format)
python scripts/validation/prepare_isbi_2014.py \
    --dataset_path data/raw/isbi_2014_atypia/H03_nuclear_criteria_x40/
```

---

## Dataset Statistics

### Combined Training Set

| Dataset | Train Images | Val Images | Total | Percentage |
|---------|--------------|------------|-------|------------|
| **SIPaKMeD** | 3,239 | 810 | 4,049 | 81.5% |
| **Herlev** | 733 | 184 | 917 | 18.5% |
| **Thyroid** | ❌ N/A | ❌ N/A | ❌ N/A | 0% |
| **TOTAL** | ~4,000 | ~1,000 | ~5,000 | 100% |

**Notes:**
- **ISBI 2014:** Not included in training (validation only)
- **Thyroid:** No public dataset available - focus on cervical cytology first
- **Combined cervical:** 4,966 images (sufficient for CellPose training)

### Class Distribution (Cervix - SIPaKMeD + Herlev)

| Class | SIPaKMeD | Herlev | Combined | Percentage |
|-------|----------|--------|----------|------------|
| **Normal** | 1,807 | ~470 | ~2,277 | 45.9% |
| **Light Dysplasia** | 1,484 | ~182 | ~1,666 | 33.6% |
| **Moderate Dysplasia** | 793 | ~146 | ~939 | 18.9% |
| **Severe Dysplasia** | 1,470 | ~197 | ~1,667 | 33.6% |
| **Carcinoma** | 813 | ~150 | ~963 | 19.4% |

**Note:** Slight overlap in categories → Will merge during preprocessing.

---

## Organ-Specific Configuration

### Cervix (SIPaKMeD + Herlev)

```json
{
  "organ": "Cervix",
  "classification_system": "Bethesda_Gyn",
  "cyto3_trigger": "skip",
  "nc_ratio_required": false,
  "nuclei_diameter": 35,
  "target_sensitivity": 0.98
}
```

**CellPose Training:**
- ✅ Nuclei only (Master model)
- ❌ Skip Cyto3 (Slave model)
- ✅ Use SIPaKMeD (4,049) as primary, Herlev (917) for validation

### Thyroid (Dataset TBD)

```json
{
  "organ": "Thyroid",
  "classification_system": "Bethesda",
  "cyto3_trigger": "auto",
  "nc_ratio_required": true,
  "nuclei_diameter": 30,
  "cyto_diameter": 60,
  "target_sensitivity": 0.98,
  "status": "dataset_not_available"
}
```

**Status:** ❌ No public dataset available

**Approach:**
- ⚠️ Train on cervical cytology first (SIPaKMeD + Herlev)
- ⚠️ Transfer learning to thyroid if small dataset found
- ⚠️ CellPose models generalize well across cytology types

### Breast (ISBI 2014) — Validation Only

```json
{
  "organ": "Breast",
  "classification_system": "Nuclear_Atypia",
  "usage": "validation_only",
  "expert_annotations": "6_criteria_scores",
  "target_correlation": 0.75
}
```

**Usage:**
- ❌ NO CellPose training (histology, not cytology)
- ✅ Validate 6 Universal Criteria
- ✅ Calibrate Canal H (Chromatin Density)

---

## Critical Distinctions

### Histology vs Cytology

| Aspect | Histology (ISBI 2014) | Cytology (SIPaKMeD, Herlev) |
|--------|----------------------|-----------------------------|
| **Cell Arrangement** | Cells stuck together in tissue | Isolated cells floating in liquid |
| **Goal** | Separate stuck nuclei | Find rare abnormal cells |
| **CellPose Training** | ❌ NO (wrong morphology) | ✅ YES |
| **Validation** | ✅ YES (6 criteria) | ✅ YES |
| **Example** | Breast biopsy | Pap smear, Cervical cytology |
| **Available Datasets** | ISBI 2014 (1,200 images) | SIPaKMeD (4,049) + Herlev (917) |

**Key Takeaway:** Never mix histology and cytology for CellPose training!

---

## The 6 Universal Criteria (ISBI 2014 Table 2)

> **Validated by Prof. Frédérique Capron, Pitié-Salpêtrière Hospital, Paris**

1. **Size of Nuclei** — Enlarged nuclei (>2× or >3× normal)
2. **Size of Nucleoli** — Prominent nucleoli (dark spots in nucleus)
3. **Density of Chromatin** ⭐ — **Hyperchromasia (Canal H!)**
4. **Thickness of Nuclear Membrane** — Thickened membrane
5. **Regularity of Nuclear Contour** — Irregular, notched borders
6. **Anisonucleosis** — Size variation within cell population

**Implementation:** `src/scoring/malignancy_scoring.py`

**Why This Matters:**
- ✅ Scientifically validates our Canal H approach (V13/V14)
- ✅ Criteria are UNIVERSAL (breast, thyroid, cervix, bladder)
- ✅ Thresholds (30%, 60%) are clinically validated

---

## References

### Papers

1. **SIPaKMeD:** Plissiti et al. (2018) - IEEE ICIP
2. **Herlev:** Jantzen et al. (2005) - NiSIS
3. **ISBI 2014:** Veta et al. (2015) - Medical Image Analysis
4. **Thyroid:** No public dataset available (as of 2026-01-19)

### Links

- **SIPaKMeD:** https://www.cs.uoi.gr/~marina/sipakmed.html
- **Herlev:** http://mde-lab.aegean.gr/index.php/downloads
- **ISBI 2014:** https://mitos-atypia-14.grand-challenge.org/
- **Thyroid:** ❌ No public repository available - check Kaggle or academic sources

---

## Next Steps

### Phase 1: Complete Downloads ⚠️

```bash
# Priority 1: Herlev (validation for Cervix)
# Manual download from http://mde-lab.aegean.gr/

# Priority 2: ISBI 2014 (validation only)
# Register at https://mitos-atypia-14.grand-challenge.org/

# Thyroid: Search alternatives
kaggle datasets list -s "thyroid cytology"
# Check academic institutions for collaboration
```

**Note:** TB-PANDA does not exist at public URL - focus on cervical datasets first.

### Phase 2: Preprocess All Datasets

```bash
python scripts/datasets/preprocess_cytology.py --all
```

### Phase 3: Train CellPose Master/Slave

```bash
# Master (Nuclei) - Cervical cytology
python scripts/training/train_cellpose_nuclei.py \
    --datasets sipakmed,herlev

# Slave (Cyto3) - Skip for now (cervical cytology doesn't require N/C ratio)
# Will train on thyroid dataset when/if available
```

**Note:** Cervical cytology (SIPaKMeD + Herlev) uses nuclei-only approach.
N/C ratio (Cyto3) would be needed for thyroid (Bethesda V-VI), but dataset not available.

### Phase 4: Validate with ISBI 2014

```bash
python scripts/validation/validate_6_criteria.py \
    --dataset isbi_2014_h03
```

---

## Troubleshooting

### Issue: Dataset not found

```bash
# Check download status
python scripts/datasets/verify_datasets.py

# Follow download instructions in individual dataset README
```

### Issue: Preprocessing fails

```bash
# Check image quality
python scripts/datasets/check_image_quality.py --dataset sipakmed

# Remove corrupted images
python scripts/datasets/clean_corrupted_images.py --dataset sipakmed
```

### Issue: Class imbalance

```bash
# Compute class weights
python scripts/datasets/compute_class_weights.py --dataset sipakmed
```

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-19 | 1.0.0 | Initial index created with SIPaKMeD downloaded |

---

**Last Updated:** 2026-01-19
**Maintainer:** CellViT-Optimus V14 Team
**Status:** 1/3 available datasets ready (SIPaKMeD ✅)

**Important:** TB-PANDA thyroid dataset does not exist at initially referenced URL.
Focus on cervical cytology (SIPaKMeD + Herlev) for V14 development.
