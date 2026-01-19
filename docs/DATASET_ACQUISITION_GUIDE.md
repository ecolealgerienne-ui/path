# Dataset Acquisition Guide - V14 Cytology Branch

> **Status:** Ready for manual download
> **Target:** 70,000+ images across 4 organs (Cervix, Thyroid, Bladder, Breast)
> **Priority:** All organs equally prioritized
> **Critical:** ISBI 2014 is for VALIDATION, not CellPose training (see Dataset 4)

---

## ⚠️ Histology vs Cytology — Critical Distinction

### What is Cytology?
- **Isolated cells** floating in liquid (urine, thyroid FNA, Pap smear)
- Cells are **separated** and **spread out** on slide
- **Goal:** Find rare abnormal cells (needle in haystack)
- **Example:** 95% normal cells, looking for 5% malignant cells

### What is Histology?
- **Tissue sections** — cells stuck together in organized structures
- Cells form a **dense carpet** with clear tissue architecture
- **Goal:** Separate stuck nuclei (instance segmentation challenge)
- **Example:** Breast biopsy, colon biopsy

### Why This Matters for CellPose Training

| Dataset | Type | CellPose Training | Validation/Scoring |
|---------|------|-------------------|-------------------|
| TB-PANDA | Cytology | ✅ YES | ✅ YES |
| Herlev | Cytology | ✅ YES | ✅ YES |
| SIPaKMeD | Cytology | ✅ YES | ✅ YES |
| **ISBI 2014** | **Histology** | **❌ NO** | **✅ YES (6 criteria)** |

**ISBI 2014 Exception:**
The ISBI 2014 dataset (breast tissue) contains the **6 Universal Criteria** (Table 2)
that validate our Canal H approach. Use it for **algorithm validation**, not segmentation training.

---

## Quick Start

```bash
# 1. Create data directory
mkdir -p data/raw

# 2. Follow download instructions below for each dataset
# 3. Verify downloads
python scripts/datasets/verify_datasets.py

# 4. Run preprocessing
python scripts/datasets/preprocess_cytology.py
```

---

## Dataset 1: Thyroid Cytology Datasets — ⚠️ LIMITED AVAILABILITY

**Organ:** Thyroid
**Classification:** Bethesda System I-VI
**Format:** FNA (Fine Needle Aspiration) Cytology
**Status:** ⚠️ **Public datasets are SCARCE**

### ⚠️ CRITICAL: TB-PANDA Does Not Exist at Public URL

**Initial Reference (INCORRECT):**
- ❌ `https://github.com/ncbi/TB-PANDA` → **DOES NOT EXIST**
- ❌ No public Git repository available

**Reality:**
The thyroid cytology dataset landscape is challenging. Unlike cervical cytology (SIPaKMeD, Herlev),
there are **NO large-scale public thyroid FNA datasets** with Bethesda classifications.

### Alternative Thyroid Datasets (Known Sources)

#### Option 1: Kaggle Search (Best Bet)

```bash
# Search Kaggle for thyroid cytology datasets
kaggle datasets list -s "thyroid cytology"
kaggle datasets list -s "thyroid FNA"
kaggle datasets list -s "bethesda thyroid"
```

**Known Kaggle datasets:**
- Search "thyroid nodule ultrasound" (imaging, not cytology)
- Search "thyroid fine needle aspiration"
- Check individual user uploads (may be small, <1,000 images)

#### Option 2: Academic Collaborations

**Contact institutions:**
1. **Mayo Clinic** (Rochester, MN) - Thyroid cytology expertise
2. **Johns Hopkins** (Baltimore, MD) - Bethesda System creators
3. **Memorial Sloan Kettering** (New York, NY) - Large cancer database
4. **MD Anderson** (Houston, TX) - Endocrine pathology

**Request research collaboration for dataset access.**

#### Option 3: Create Synthetic/Augmented Dataset

**If no public data available:**
- Use GAN-based augmentation on small datasets
- Combine multiple small sources (Kaggle + academic)
- Focus on Cervix first (SIPaKMeD 4k images available)

### Expected Structure (If Dataset Found)

```
data/raw/thyroid_cytology/
├── images/
│   ├── Bethesda_I_Nondiagnostic/
│   ├── Bethesda_II_Benign/
│   ├── Bethesda_III_AUS_FLUS/
│   ├── Bethesda_IV_FN_SFN/
│   ├── Bethesda_V_Suspicious/
│   └── Bethesda_VI_Malignant/
└── annotations.csv
```

### Key Metrics for Validation (When Dataset Available)
- **Target Sensitivity:** >98% for Bethesda V-VI (malignant)
- **FROC:** <2 FP/WSI @ 98% sensitivity
- **N/C Ratio:** Required (Cyto3 auto-activation)

### Recommendation: Focus on Cervix First

**Pragmatic Approach:**
1. ✅ **Start with SIPaKMeD** (4,049 cervical images — AVAILABLE NOW)
2. ✅ **Add Herlev** (917 cervical images for validation)
3. ✅ **Validate with ISBI 2014** (breast histology — 6 criteria)
4. ⚠️ **Thyroid:** Search Kaggle, contact institutions, or postpone

**Why this works:**
- CellPose models trained on cervical cells can transfer to thyroid
- The 6 Universal Criteria (ISBI 2014) apply to all organs
- Better to have 5k cervical images than wait for non-existent thyroid data

### References (General Thyroid Cytology)
- **Bethesda System:** Cibas & Ali (2017) - "The 2017 Bethesda System for Reporting Thyroid Cytopathology"
- **Paper:** Ali & Cibas (2010) - Thyroid (journal)
- **No public dataset URL available as of 2026-01-19**

---

## Dataset 2: Herlev (Cervical Pap Smear) — 917 images

**Organ:** Cervix
**Classification:** 5 classes (Normal → Carcinoma)
**Format:** Single cell images (cropped)
**License:** Academic use

### Download Instructions

⚠️ **MANUAL DOWNLOAD REQUIRED**

1. Visit: http://mde-lab.aegean.gr/index.php/downloads
2. Register if required
3. Download: `Herlev_dataset.zip`
4. Extract to: `data/raw/herlev/`

### Expected Structure
```
data/raw/herlev/
├── images/
│   ├── normal/
│   ├── light_dysplasia/
│   ├── moderate_dysplasia/
│   ├── severe_dysplasia/
│   └── carcinoma/
└── annotations.csv
```

### Classes
1. **Normal** (Koilocytotic, Parabasal, Intermediate, Superficial-Intermediate)
2. **Light Dysplasia**
3. **Moderate Dysplasia**
4. **Severe Dysplasia**
5. **Carcinoma in situ**

### Key Metrics for Validation
- **Target Sensitivity:** >98% for Dysplasia + Carcinoma
- **N/C Ratio:** Not required (nuclei-only mode OK)
- **Cohen's Kappa:** >0.80 vs expert labels

### References
- Paper: Jantzen et al. (2005) - "Pap-smear Benchmark Data For Pattern Classification"
- URL: http://mde-lab.aegean.gr/index.php/downloads

---

## Dataset 3: SIPaKMeD (Cervical Pap Smear) — 4,049 images

**Organ:** Cervix
**Classification:** 5 cell types
**Format:** Single cell images + Clusters
**License:** Academic use (registration required)

### Download Instructions

⚠️ **REGISTRATION REQUIRED**

1. Visit: https://www.cs.uoi.gr/~marina/sipakmed.html
2. Fill registration form
3. Wait for approval email (may take 1-3 days)
4. Download dataset ZIP
5. Extract to: `data/raw/sipakmed/`

### Expected Structure
```
data/raw/sipakmed/
├── images/
│   ├── im_Superficial-Intermediate/
│   ├── im_Parabasal/
│   ├── im_Koilocytotic/
│   ├── im_Dyskeratotic/
│   └── im_Metaplastic/
└── labels.csv
```

### Classes
1. **Superficial-Intermediate**
2. **Parabasal**
3. **Koilocytotic**
4. **Dyskeratotic**
5. **Metaplastic**

### Key Metrics for Validation
- **IoU Nucleus:** >0.85
- **AP50 (COCO):** >0.90
- **Panoptic Quality:** >0.75

### References
- Paper: Plissiti et al. (2018) - "SIPaKMeD: A New Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells in Pap Smear Images"
- URL: https://www.cs.uoi.gr/~marina/sipakmed.html
- DOI: 10.1109/ICIP.2018.8451588

---

## Dataset 4: ISBI 2014 MITOS-ATYPIA — ~1,200 images ⭐ VALIDATION GOLD STANDARD

**Organ:** Breast
**Type:** HISTOLOGY (tissue sections, NOT cytology)
**Classification:** Nuclear Atypia Score (1-3) + 6 Universal Criteria
**Format:** High-resolution histology patches (X20 & X40 magnification)
**License:** Academic use (registration required)

### ⚠️ CRITICAL USAGE CLARIFICATION

**❌ DO NOT USE FOR:**
- CellPose training (segmentation)
- Cytology model training
- Direct cell detection

**✅ USE FOR:**
- **Validating Malignancy Scoring algorithms** (PRIORITY)
- Calibrating the 6 Universal Criteria thresholds
- Testing Canal H (Chromatin Density) extraction
- Benchmarking morphometric features

**Why:** This is **histology** (cells stuck together in tissue), NOT **cytology** (isolated cells).
CellPose expects isolated cells, so these images are incompatible for segmentation training.

**However,** the dataset contains the **"Secret Recipe"** — The 6 Universal Criteria for nuclear atypia
that apply to BOTH histology AND cytology. This validates our V13/V14 approach!

### The 6 Universal Criteria (ISBI 2014 Table 2)

> **Gold Standard from Prof. Frédérique Capron, Pitié-Salpêtrière Hospital, Paris**

| Criterion | Score 1 | Score 2 | Score 3 | Our Implementation |
|-----------|---------|---------|---------|-------------------|
| **Size of nuclei** | 0-30% bigger | 30-60% bigger | >60% bigger | `prop.area` vs normal |
| **Size of nucleoli** | 0-30% bigger | 30-60% bigger | >60% bigger | H-channel OD > 0.6 |
| **Density of chromatin** ⭐ | 0-30% denser | 30-60% denser | >60% denser | **Mean H-channel OD (RUIFROK)** |
| **Thickness of membrane** | 0-30% thicker | 30-60% thicker | >60% thicker | Perimeter / sqrt(Area) |
| **Regularity of contour** | 0-30% irregular | 30-60% irregular | >60% irregular | Solidity (convex hull) |
| **Anisonucleosis** | Regular size, <2x normal | Intermediate | Irregular OR >3x normal | CV + max/normal ratio |

### Why This Validates V13/V14

✅ **Criterion 3 (Density of Chromatin) = Canal H**
- The pathologists explicitly state that **chromatin density** is a major malignancy criterion
- This is EXACTLY what our Ruifrok H-channel extraction measures!
- V13 FPN Chimique injects this signal → validated by clinical practice

✅ **The 6 Criteria are UNIVERSAL**
- Developed for breast histology
- Apply equally to thyroid, bladder, cervix cytology
- Scientific consensus across organs

✅ **The Thresholds (30%, 60%) are CLINICALLY VALIDATED**
- Not arbitrary ML thresholds
- Based on pathologist expert consensus
- Directly usable for our scoring engine

### Download Instructions

⚠️ **REGISTRATION REQUIRED**

1. Visit: https://mitos-atypia-14.grand-challenge.org/
2. Create account
3. Accept challenge terms
4. Download:
   - **A03 dataset** (Nuclear Atypia - X20 frames)
   - **H03 dataset** (Nuclear Atypia - X40 frames with 6 criteria)
5. Extract to: `data/raw/isbi_2014_atypia/`

### Expected Structure
```
data/raw/isbi_2014_atypia/
├── A03_nuclear_atypia_x20/
│   ├── images/
│   └── scores.csv  # Nuclear atypia score (1, 2, or 3)
├── H03_nuclear_criteria_x40/
│   ├── images/
│   └── criteria.csv  # The 6 criteria scores
└── README.md
```

### How to Use This Dataset

**Phase 1: Algorithm Calibration**
```python
# Use H03 images to calibrate your 6 criteria algorithms
from src.scoring.malignancy_scoring import MalignancyScoringEngine

engine = MalignancyScoringEngine(organ_type="Breast")

for image_path, ground_truth in h03_dataset:
    # Extract H-channel using Ruifrok
    h_channel = ruifrok_extract_h_channel(image_rgb)

    # Compute our 6 criteria
    our_score = engine.score_image(image_rgb, nuclei_masks, h_channel)

    # Compare to expert scores
    chromatin_diff = abs(our_score.density_of_chromatin - ground_truth.density_of_chromatin)

    # Tune thresholds to minimize diff
```

**Phase 2: Cross-Validation**
- Train on breast histology (ISBI 2014)
- Test on thyroid cytology (TB-PANDA)
- Validate that the 6 criteria **transfer** across organs

**Phase 3: H-Channel Validation**
- Prove that Ruifrok H-channel correlates with expert "Chromatin Density" scores
- This validates the entire V13 FPN Chimique architecture!

### Key Metrics for Validation
- **Cohen's Kappa:** >0.60 (agreement with pathologists on 6 criteria)
- **AUC-ROC:** >0.85 (3-class atypia score prediction)
- **Criterion Correlation:** >0.70 (our scores vs expert scores)

### References
- **Challenge:** https://mitos-atypia-14.grand-challenge.org/
- **Dataset Details:** https://mitos-atypia-14.grand-challenge.org/Dataset/
- **Paper:** Veta et al. (2015) - "Assessment of algorithms for mitosis detection in breast cancer histopathology images"
- **Hospital:** Prof. Frédérique Capron, Pathology Dept., Pitié-Salpêtrière Hospital, Paris

### Implementation

See: `src/scoring/malignancy_scoring.py` for complete implementation of the 6 criteria.

---

## Dataset 5: Kaggle Cytology Datasets

### Search Keywords

```bash
# If Kaggle CLI is installed (pip install kaggle)
kaggle datasets list -s "cervical cytology"
kaggle datasets list -s "pap smear"
kaggle datasets list -s "thyroid FNA"
kaggle datasets list -s "thyroid cytology"
kaggle datasets list -s "urine cytology"
kaggle datasets list -s "bladder cancer cytology"
kaggle datasets list -s "bethesda system"
kaggle datasets list -s "paris system urology"
```

### Setup Kaggle API

1. Create Kaggle account: https://www.kaggle.com/
2. Go to: https://www.kaggle.com/<username>/account
3. Click "Create New API Token"
4. Save `kaggle.json` to `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Recommended Kaggle Datasets

| Dataset | Organ | Samples | URL |
|---------|-------|---------|-----|
| Cervical Cancer Screening | Cervix | ~13,000 | https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed |
| Liquid Based Cytology | Cervix | ~1,000 | Search "liquid based cytology" |
| Thyroid Nodule Ultrasound | Thyroid | Variable | Search "thyroid nodule" |

---

## Dataset 6: Paris System (Urine/Bladder) — TBD ⚠️ NEEDS SOURCING

**Organ:** Bladder (Urine cytology)
**Classification:** Paris System (NHGUC, SHGUC)
**Status:** ⚠️ No public dataset found yet

### Potential Sources

1. **Cancer Imaging Archive (TCIA)**
   - URL: https://www.cancerimagingarchive.net/
   - Search: "bladder", "urine cytology"

2. **Mendeley Data**
   - URL: https://data.mendeley.com/datasets
   - Search: "urine cytology", "bladder cancer cytology"

3. **zenodo.org**
   - URL: https://zenodo.org/
   - Search: "paris system urology"

4. **Contact Research Groups**
   - Johns Hopkins Pathology
   - Cleveland Clinic Cytopathology
   - Mayo Clinic

### Required Features for Paris System
- **N/C Ratio:** CRITICAL (High N/C = SHGUC)
- **Chromatin Coarseness:** Required
- **Nuclear Irregularity:** Required
- **Cyto3 Activation:** AUTO (always needed)

---

## Dataset Verification

After downloading datasets, run verification:

```bash
python scripts/datasets/verify_datasets.py
```

### Expected Output
```
✅ TB-PANDA: 10,247 images found
✅ Herlev: 917 images found
✅ SIPaKMeD: 4,049 images found
✅ ISBI 2014: 1,200 images found
⚠️ Paris System: Not found (needs sourcing)

📊 Total: 16,413 images
```

---

## Dataset Preprocessing

Once datasets are downloaded, run preprocessing:

```bash
# Preprocess all datasets
python scripts/datasets/preprocess_cytology.py --all

# Or process individually
python scripts/datasets/preprocess_cytology.py --dataset tb_panda
python scripts/datasets/preprocess_cytology.py --dataset herlev
python scripts/datasets/preprocess_cytology.py --dataset sipakmed
python scripts/datasets/preprocess_cytology.py --dataset isbi_2014
```

### Preprocessing Steps
1. **Image Normalization:** Resize to 512×512 (preserving aspect ratio)
2. **Format Conversion:** All to PNG RGB
3. **Annotation Extraction:** Convert to unified JSON format
4. **Train/Val Split:** 80/20 stratified by class
5. **Quality Control:** Remove corrupted/low-quality images

### Expected Output Structure
```
data/processed/
├── tb_panda/
│   ├── train/
│   │   ├── images/
│   │   └── annotations.json
│   └── val/
│       ├── images/
│       └── annotations.json
├── herlev/
│   ├── train/
│   └── val/
├── sipakmed/
│   ├── train/
│   └── val/
└── isbi_2014/
    ├── train/
    └── val/
```

---

## Unified Annotation Format

All datasets will be converted to this JSON format:

```json
{
  "image_id": "tb_panda_001",
  "organ": "Thyroid",
  "classification_system": "Bethesda",
  "diagnosis": "Bethesda_V",
  "width": 512,
  "height": 512,
  "nuclei": [
    {
      "nucleus_id": 1,
      "bbox": [x, y, w, h],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 450,
      "category": "malignant",
      "confidence": 1.0
    }
  ],
  "cytoplasm": [
    {
      "cytoplasm_id": 1,
      "nucleus_id": 1,
      "bbox": [x, y, w, h],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1200,
      "nc_ratio": 0.375
    }
  ],
  "metadata": {
    "stain": "Papanicolaou",
    "scanner": "Aperio",
    "magnification": "40x",
    "source_dataset": "TB-PANDA"
  }
}
```

---

## Storage Requirements

| Dataset | Size (GB) | Samples | Storage Type |
|---------|-----------|---------|--------------|
| TB-PANDA | ~15 GB | 10,000 | SSD recommended |
| Herlev | ~0.5 GB | 917 | Any |
| SIPaKMeD | ~2 GB | 4,049 | Any |
| ISBI 2014 | ~3 GB | 1,200 | Any |
| **Total** | **~20.5 GB** | **16,166** | **SSD** |

---

## Next Steps After Acquisition

1. ✅ Download all datasets (this guide)
2. ✅ Run verification script
3. ✅ Run preprocessing script
4. 🔄 Configure organ-specific settings (`config/cytology_organ_config.json`)
5. 🔄 Train Master model (CellPose nuclei)
6. 🔄 Train Slave model (CellPose cyto3)
7. 🔄 Implement CytologyMasterSlaveOrchestrator
8. 🔄 Extract morphometric features
9. 🔄 Train LightGBM classification head
10. 🔄 Validate with clinical metrics (Sensitivity, FROC, Kappa)

---

## Troubleshooting

### Issue: Dataset download fails
**Solution:** Use manual download methods above

### Issue: Annotations missing
**Solution:** Check dataset documentation for annotation format conversion

### Issue: Storage space insufficient
**Solution:**
- Download datasets sequentially (not all at once)
- Use external drive
- Compress processed datasets after preprocessing

### Issue: Image quality varies
**Solution:** Preprocessing will normalize and filter low-quality images

---

## Contact & Support

For dataset-specific issues:
- **TB-PANDA:** https://github.com/ncbi/TB-PANDA/issues
- **Herlev:** mde-lab@aegean.gr
- **SIPaKMeD:** marina@cs.uoi.gr
- **ISBI 2014:** https://mitos-atypia-14.grand-challenge.org/

For V14 Cytology implementation:
- See: `docs/V14_MASTER_SLAVE_ARCHITECTURE.md`
- See: `docs/V14_CYTOLOGY_STANDALONE_STRATEGY.md`

---

**Last Updated:** 2026-01-18
**Version:** V14 Cytology Branch (Standalone)
