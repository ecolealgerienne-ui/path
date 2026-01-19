# Herlev Dataset Documentation

> **Pap Smear Benchmark Data For Pattern Classification**
> **Source:** Herlev University Hospital, Denmark
> **Year:** 2005

---

## Overview

| Attribute | Value |
|-----------|-------|
| **Organ** | Cervix |
| **Modality** | Pap Smear Cytology |
| **Type** | Single cell images (cropped) |
| **Total Samples** | 917 images |
| **Classes** | 7 (2 normal + 5 abnormal) |
| **Image Format** | BMP |
| **Resolution** | Variable (cropped cells) |
| **Staining** | Papanicolaou |
| **License** | Academic use |

---

## Dataset Structure

### Directory Layout (Raw)

```
data/raw/herlev/
├── images/
│   ├── normal/
│   │   ├── superficialIntermediate/
│   │   └── parabasal/
│   └── abnormal/
│       ├── lightDysplasia/
│       ├── moderateDysplasia/
│       ├── severeDysplasia/
│       └── carcinoma/
└── smeardb.bmp  # Original database file (optional)
```

### Classes Description

| Class | Type | Count | CIN Grade | Bethesda System |
|-------|------|-------|-----------|-----------------|
| **Superficial-Intermediate** | Normal | ~396 | Normal | NILM |
| **Parabasal** | Normal | ~74 | Normal | NILM |
| **Light Dysplasia** | Abnormal | ~182 | CIN 1 | LSIL |
| **Moderate Dysplasia** | Abnormal | ~146 | CIN 2 | ASC-H / HSIL |
| **Severe Dysplasia** | Abnormal | ~197 | CIN 3 | HSIL |
| **Carcinoma in situ** | Malignant | ~150 | CIN 3+ | SCC |

**Total:** 917 images
- **Normal:** ~470 images (51.3%)
- **Abnormal/Malignant:** ~447 images (48.7%)

---

## Data Acquisition

### Download Instructions

⚠️ **MANUAL DOWNLOAD REQUIRED**

1. Visit: http://mde-lab.aegean.gr/index.php/downloads
2. Scroll to "Pap Smear Database"
3. Register if required (email, affiliation)
4. Download: `Herlev_dataset.zip` or `smeardb.zip`
5. Extract to: `data/raw/herlev/`

### Alternative Sources

- **Kaggle:** Search "Herlev Pap Smear" (may have community uploads)
- **UCI Machine Learning Repository:** (historical archive)

---

## Preprocessing

### Step 1: Verify Download

```bash
python scripts/datasets/verify_datasets.py --dataset herlev
```

**Expected Output:**
```
✅ Herlev: 917 images found
   - normal (superficial/intermediate): ~396
   - normal (parabasal): ~74
   - light_dysplasia: ~182
   - moderate_dysplasia: ~146
   - severe_dysplasia: ~197
   - carcinoma: ~150
```

### Step 2: Preprocess Dataset

```bash
python scripts/datasets/preprocess_cytology.py --dataset herlev
```

**Actions Performed:**
1. Convert BMP → PNG
2. Resize to 512×512 (preserving aspect ratio)
3. Merge normal classes (superficial + parabasal → normal)
4. Create unified annotations.json
5. Stratified 80/20 train/val split

**Output Structure:**
```
data/processed/herlev/
├── train/
│   ├── images/          # 733 images
│   └── annotations.json
└── val/
    ├── images/          # 184 images
    └── annotations.json
```

---

## Image Characteristics

### Format Details

| Attribute | Value |
|-----------|-------|
| **File Format** | BMP (uncompressed) |
| **Color Space** | RGB |
| **Typical Size** | ~100×100 to ~200×200 pixels |
| **File Size** | ~30-100 KB per image |
| **Background** | White (255, 255, 255) |

### Cell Morphology

| Cell Type | Nucleus Size | N/C Ratio | Chromatin Pattern |
|-----------|--------------|-----------|-------------------|
| **Superficial** | Small | Low (~0.1) | Fine, homogeneous |
| **Intermediate** | Medium | Medium (~0.2) | Uniform |
| **Parabasal** | Large | High (~0.4) | Coarse |
| **Dysplastic** | Enlarged | High (>0.5) | Hyperchromatic, irregular |
| **Carcinoma** | Very large | Very high (>0.7) | Extremely coarse, clumped |

---

## Clinical Classification Systems

### CIN (Cervical Intraepithelial Neoplasia)

| Herlev Class | CIN Grade | Description |
|--------------|-----------|-------------|
| Normal | Normal | No dysplasia |
| Light Dysplasia | CIN 1 | Mild dysplasia |
| Moderate Dysplasia | CIN 2 | Moderate dysplasia |
| Severe Dysplasia | CIN 3 | Severe dysplasia |
| Carcinoma | CIN 3+ / Invasive | Carcinoma in situ or invasive |

### Bethesda System Mapping

| Herlev Class | Bethesda Category | Action |
|--------------|-------------------|--------|
| Normal | NILM | Routine screening |
| Light Dysplasia | LSIL | Follow-up 6-12 months |
| Moderate Dysplasia | ASC-H or HSIL | Colposcopy |
| Severe Dysplasia | HSIL | Colposcopy + treatment |
| Carcinoma | SCC | Immediate treatment |

---

## Usage for V14 Cytology

### CellPose Training (Master Model - Nuclei)

✅ **RECOMMENDED USE:**
- Small dataset → Use for **validation** or **fine-tuning** only
- Combine with SIPaKMeD for robust training

```python
from cellpose import models

# Fine-tune on Herlev after SIPaKMeD pre-training
model = models.CellposeModel(model_type='nuclei')
model.train(
    train_data=herlev_train,
    channels=[0, 0],
    diameter=35,
    n_epochs=200,
    learning_rate=0.0001  # Low LR for fine-tuning
)
```

### Classification Head Training

✅ **RECOMMENDED USE:**
- Binary classification: Normal vs Abnormal
- 5-class classification: Normal, Light, Moderate, Severe, Carcinoma

```python
from src.scoring import MalignancyScoringEngine

# Extract features
features = extract_morphometric_features(herlev_images)

# Train classifier (with class weights)
classifier = train_lightgbm(
    features=features,
    labels=herlev_labels,
    n_classes=5,
    class_weights='balanced'
)
```

### Malignancy Scoring (6 Criteria)

✅ **RESEARCH USE:**
- Small expert-annotated dataset → Good for validation
- Compare 6 criteria scores vs dysplasia grade

```python
from src.scoring import MalignancyScoringEngine

engine = MalignancyScoringEngine(organ_type="Cervix")

# Score all Herlev images
for image, label in herlev_val:
    score = engine.score_image(image, nuclei_masks, h_channel)

    # Expected correlation:
    # Normal → Total Score 6-9
    # Light → Total Score 9-12
    # Moderate → Total Score 12-15
    # Severe/Carcinoma → Total Score 15-18
```

---

## Key Metrics

### Validation Metrics (Target)

| Metric | Target | Notes |
|--------|--------|-------|
| **Sensitivity (Dysplasia+)** | >95% | Small dataset → lower target |
| **Specificity (Normal)** | >85% | Balance false positives |
| **Cohen's Kappa** | >0.75 | Expert agreement |
| **AUC-ROC (Binary)** | >0.90 | Normal vs Abnormal |
| **AUC-ROC (5-class)** | >0.85 | Multi-class |

### Dataset Limitations

⚠️ **Small Sample Size:**
- Only 917 images (vs 4,049 in SIPaKMeD)
- Some classes have <100 samples
- Risk of overfitting

**Mitigation:**
- Use for validation, not primary training
- Combine with SIPaKMeD + TB-PANDA
- Heavy data augmentation

---

## Known Issues & Solutions

### Issue 1: Small Dataset Size

**Problem:** Only 917 images → Risk of overfitting.

**Solution:**
- Combine with SIPaKMeD (4,049 images)
- Use Herlev for validation/testing only
- Apply heavy augmentation (rotation, flip, color jitter)

### Issue 2: Class Imbalance

**Problem:** Parabasal class very small (~74 images).

**Solution:**
- Merge with superficial-intermediate → "Normal"
- Use class weights in loss function

### Issue 3: Image Quality Variation

**Problem:** Some images have artifacts, poor staining.

**Solution:**
- Quality filtering during preprocessing
- Remove images with >50% background
- Discard images with staining artifacts

### Issue 4: Background Handling

**Problem:** Large white background in cropped cells.

**Solution:**
```python
# Crop to bounding box during preprocessing
mask = (image < 250).any(axis=2)  # Non-white pixels
coords = np.argwhere(mask)
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)
cropped = image[y_min:y_max, x_min:x_max]
```

---

## Historical Context

### Original Study (2005)

**Purpose:** Benchmark dataset for computer-aided cervical cancer screening.

**Methodology:**
- Expert cytotechnicians manually selected cells
- Each cell photographed individually
- Classes assigned by consensus of 2+ experts

**Impact:**
- One of the first public cervical cytology datasets
- Widely used in medical imaging research
- Cited in 500+ papers

---

## References

### Paper

**Title:** "Pap-smear Benchmark Data For Pattern Classification"

**Authors:** Jantzen, J., Norup, J., Dounias, G., Bjerregaard, B.

**Conference:** NiSIS (Nature inspired Smart Information Systems), 2005

**PDF:** http://mde-lab.aegean.gr/downloads/Pap-smear_Benchmark.pdf

### Links

- **Dataset Page:** http://mde-lab.aegean.gr/index.php/downloads
- **Original Paper:** http://mde-lab.aegean.gr/downloads/Pap-smear_Benchmark.pdf

### Contact

- **Institution:** MDE-Lab, University of the Aegean, Greece
- **Email:** mde-lab@aegean.gr

---

## Citation

```bibtex
@inproceedings{jantzen2005pap,
  title={Pap-smear Benchmark Data For Pattern Classification},
  author={Jantzen, Jan and Norup, Jonas and Dounias, Georgios and Bjerregaard, Beth},
  booktitle={NiSIS 2005: Nature inspired Smart Information Systems},
  pages={1--9},
  year={2005}
}
```

---

## Comparison: Herlev vs SIPaKMeD

| Aspect | Herlev | SIPaKMeD | Winner |
|--------|--------|----------|--------|
| **Size** | 917 | 4,049 | SIPaKMeD |
| **Year** | 2005 | 2018 | SIPaKMeD |
| **Quality** | Variable | Consistent | SIPaKMeD |
| **Classes** | 7 (2+5) | 7 (3+4) | Tie |
| **Availability** | Manual download | Registration | Tie |
| **Citation Impact** | High (500+) | Growing (100+) | Herlev |
| **Best Use** | Validation | Training | - |

**Recommendation:** Use SIPaKMeD for primary training, Herlev for validation/testing.

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-19 | 1.0.0 | Initial documentation for V14 Cytology |

---

**Last Updated:** 2026-01-19
**Status:** ⚠️ Awaiting download
**Maintainer:** CellViT-Optimus V14 Team
