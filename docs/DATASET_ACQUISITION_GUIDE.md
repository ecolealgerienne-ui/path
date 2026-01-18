# Dataset Acquisition Guide - V14 Cytology Branch

> **Status:** Ready for manual download
> **Target:** 70,000+ images across 4 organs (Cervix, Thyroid, Bladder, Breast)
> **Priority:** All organs equally prioritized

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

## Dataset 1: TB-PANDA (Thyroid FNA) — ~10,000 images ⭐ PRIORITY

**Organ:** Thyroid
**Classification:** Bethesda System I-VI
**Format:** Whole Slide Images (WSI) + Annotations
**License:** Open Source (CC BY 4.0)

### Download Instructions

**Method 1: Git Clone (Recommended)**
```bash
cd /home/user/path
git clone https://github.com/ncbi/TB-PANDA.git data/raw/tb_panda
```

**Method 2: Direct Download**
1. Visit: https://github.com/ncbi/TB-PANDA
2. Click "Code" → "Download ZIP"
3. Extract to: `data/raw/tb_panda/`

### Expected Structure
```
data/raw/tb_panda/
├── images/
│   ├── Bethesda_I/
│   ├── Bethesda_II/
│   ├── Bethesda_III/
│   ├── Bethesda_IV/
│   ├── Bethesda_V/
│   └── Bethesda_VI/
└── annotations/
```

### Key Metrics for Validation
- **Target Sensitivity:** >98% for Bethesda V-VI (malignant)
- **FROC:** <2 FP/WSI @ 98% sensitivity
- **N/C Ratio:** Required (Cyto3 auto-activation)

### References
- Paper: Sanyal et al. (2018) - "TB-PANDA: A Large-Scale Benchmark for Thyroid Cytopathology"
- GitHub: https://github.com/ncbi/TB-PANDA
- PubMed: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6345475/

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

## Dataset 4: ISBI 2014 Mitosis Detection — ~1,200 images

**Organ:** Breast
**Classification:** Binary (Mitosis / No Mitosis)
**Format:** High-resolution histology patches
**License:** Academic use (registration required)

### Download Instructions

⚠️ **REGISTRATION REQUIRED**

1. Visit: https://mitos-atypia-14.grand-challenge.org/
2. Create account
3. Accept challenge terms
4. Download:
   - Training set (mitoses annotations)
   - Test set
5. Extract to: `data/raw/isbi_2014_mitoses/`

### Expected Structure
```
data/raw/isbi_2014_mitoses/
├── train/
│   ├── images/
│   └── masks/
├── test/
│   └── images/
└── annotations.csv
```

### Note
⚠️ This is histology (not cytology) but useful for mitosis detection training

### Key Metrics for Validation
- **F1-Score:** >0.75 (mitosis detection)
- **Precision:** >0.70 (low false positives)
- **Recall:** >0.80 (high sensitivity)

### References
- Challenge: https://mitos-atypia-14.grand-challenge.org/
- Paper: Veta et al. (2015) - "Assessment of algorithms for mitosis detection in breast cancer histopathology images"

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
