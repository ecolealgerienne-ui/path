# SIPaKMeD Dataset Documentation

> **Single Image Per-class Acute Myeloid Leukemia Dataset**
> **Source:** University of Ioannina, Greece
> **Paper:** Plissiti et al. (2018)

---

## Overview

| Attribute | Value |
|-----------|-------|
| **Organ** | Cervix |
| **Modality** | Pap Smear Cytology |
| **Type** | Single cell images (cropped) |
| **Total Samples** | 4,049 images |
| **Classes** | 7 (3 normal + 4 abnormal) |
| **Image Format** | BMP |
| **Resolution** | Variable (cropped cells) |
| **Staining** | Papanicolaou |
| **License** | Academic use (registration required) |

---

## Dataset Structure

### Directory Layout (Raw)

```
data/raw/sipakmed/pictures/
├── carcinoma_in_situ/          # 813 images
├── light_dysplastic/           # 1,484 images
├── moderate_dysplastic/        # 793 images
├── severe_dysplastic/          # 1,470 images
├── normal_columnar/            # 787 images
├── normal_intermediate/        # 518 images
└── normal_superficiel/         # 502 images
```

**Note:** Each subdirectory contains:
- `.bmp` image files
- `.bmp:Zone.Identifier` files (Windows metadata - can be ignored)

### Classes Description

| Class | Type | Count | Description | Bethesda Mapping |
|-------|------|-------|-------------|------------------|
| **normal_columnar** | Normal | 787 | Columnar epithelial cells (endocervical) | NILM |
| **normal_intermediate** | Normal | 518 | Intermediate squamous cells | NILM |
| **normal_superficiel** | Normal | 502 | Superficial squamous cells | NILM |
| **light_dysplastic** | Abnormal | 1,484 | Mild dysplasia (CIN 1) | LSIL |
| **moderate_dysplastic** | Abnormal | 793 | Moderate dysplasia (CIN 2) | ASC-H / HSIL |
| **severe_dysplastic** | Abnormal | 1,470 | Severe dysplasia (CIN 3) | HSIL |
| **carcinoma_in_situ** | Malignant | 813 | Carcinoma in situ (CIS) | SCC |

**Total:** 4,049 images
- **Normal:** 1,807 images (44.6%)
- **Abnormal/Malignant:** 2,242 images (55.4%)

---

## Clinical Classification Systems

### CIN (Cervical Intraepithelial Neoplasia)

| SIPaKMeD Class | CIN Grade | Description |
|----------------|-----------|-------------|
| Normal | Normal | No dysplasia |
| Light Dysplastic | CIN 1 | Mild dysplasia (lower 1/3 of epithelium) |
| Moderate Dysplastic | CIN 2 | Moderate dysplasia (lower 2/3) |
| Severe Dysplastic | CIN 3 | Severe dysplasia (full thickness) |
| Carcinoma in situ | CIN 3+ | Invasive carcinoma |

### Bethesda System Mapping

| SIPaKMeD Class | Bethesda Category | Clinical Action |
|----------------|-------------------|-----------------|
| Normal | NILM (Negative for Intraepithelial Lesion/Malignancy) | Routine screening |
| Light Dysplastic | LSIL (Low-grade Squamous Intraepithelial Lesion) | Follow-up in 6-12 months |
| Moderate Dysplastic | ASC-H or HSIL | Colposcopy + biopsy |
| Severe Dysplastic | HSIL (High-grade Squamous Intraepithelial Lesion) | Colposcopy + treatment |
| Carcinoma in situ | SCC (Squamous Cell Carcinoma) | Immediate treatment |

---

## Data Acquisition

### Download Instructions

⚠️ **REGISTRATION REQUIRED**

1. Visit: https://www.cs.uoi.gr/~marina/sipakmed.html
2. Fill registration form with:
   - Name, affiliation, email
   - Research purpose
3. Wait for approval email (1-3 days)
4. Download dataset ZIP
5. Extract to: `data/raw/sipakmed/pictures/`

### Download URL (after registration)

```bash
# URL provided in approval email
wget <download_link> -O sipakmed.zip
unzip sipakmed.zip -d data/raw/sipakmed/
```

---

## Preprocessing

### Step 1: Verify Download

```bash
python scripts/datasets/verify_datasets.py --dataset sipakmed
```

**Expected Output:**
```
✅ SIPaKMeD: 4,049 images found
   - carcinoma_in_situ: 813
   - light_dysplastic: 1,484
   - moderate_dysplastic: 793
   - severe_dysplastic: 1,470
   - normal_columnar: 787
   - normal_intermediate: 518
   - normal_superficiel: 502
```

### Step 2: Preprocess Dataset

```bash
python scripts/datasets/preprocess_cytology.py --dataset sipakmed
```

**Actions Performed:**
1. Convert BMP → PNG
2. Resize to 512×512 (preserving aspect ratio)
3. Remove Zone.Identifier files
4. Create unified annotations.json
5. Stratified 80/20 train/val split

**Output Structure:**
```
data/processed/sipakmed/
├── train/
│   ├── images/          # 3,239 images
│   └── annotations.json
└── val/
    ├── images/          # 810 images
    └── annotations.json
```

### Step 3: Clean Zone.Identifier Files (Optional)

```bash
# Remove Windows metadata files
find data/raw/sipakmed/ -name "*.bmp:Zone.Identifier" -delete
```

---

## Image Characteristics

### Format Details

| Attribute | Value |
|-----------|-------|
| **File Format** | BMP (uncompressed) |
| **Color Space** | RGB |
| **Typical Size** | Variable (cropped single cells) |
| **File Naming** | `<slide_id>-<cell_id>-<variant>.bmp` |
| **Example** | `149143370-149143378-001-d.bmp` |

### Image Variants

Each cell may have multiple variants:
- **Base image:** `<id>.BMP` (uppercase)
- **Duplicate/crop:** `<id>-d.bmp` (lowercase + `-d` suffix)

**Note:** Both variants should be included in preprocessing (they are different crops/views).

---

## Usage for V14 Cytology

### CellPose Training (Master Model - Nuclei)

✅ **RECOMMENDED USE:**
- Train CellPose `nuclei` model on all classes
- Objective: Detect all nuclei regardless of class
- Focus: Segmentation accuracy (IoU > 0.85)

```python
from cellpose import models

# Load SIPaKMeD for training
sipakmed_train = load_processed_dataset("sipakmed", split="train")

# Train nuclei model
model = models.CellposeModel(model_type='nuclei')
model.train(
    train_data=sipakmed_train,
    channels=[0, 0],  # Grayscale
    diameter=35,       # Cervical cells
    n_epochs=500
)
```

### Classification Head Training

✅ **RECOMMENDED USE:**
- Use extracted features for 7-class classification
- Binary: Normal vs Abnormal
- Multi-class: 7 classes (as above)

```python
from src.scoring import MalignancyScoringEngine

engine = MalignancyScoringEngine(organ_type="Cervix")

# Extract morphometric features
features = engine.extract_features(image_rgb, nuclei_masks)

# Train LightGBM classifier
classifier = train_lightgbm(
    features=features,
    labels=sipakmed_labels,
    n_classes=7
)
```

### Malignancy Scoring (6 Criteria)

✅ **RESEARCH USE:**
- Apply ISBI 2014 criteria to cervical cells
- Validate criterion transferability (breast → cervix)

```python
from src.scoring import MalignancyScoringEngine

engine = MalignancyScoringEngine(organ_type="Cervix")
score = engine.score_image(image_rgb, nuclei_masks, h_channel)

# Correlation with dysplasia grade
correlation = compute_correlation(
    score.total_score,
    dysplasia_severity  # 0=normal, 1=light, 2=moderate, 3=severe, 4=CIS
)
```

---

## Key Metrics

### Validation Metrics (Target)

| Metric | Target | Purpose |
|--------|--------|---------|
| **Sensitivity (HSIL+)** | >98% | Never miss high-grade lesions |
| **Specificity (NILM)** | >85% | Reduce false positives |
| **Cohen's Kappa** | >0.80 | Agreement with expert pathologists |
| **IoU (Nucleus)** | >0.85 | Segmentation accuracy |
| **AP50 (COCO)** | >0.90 | Detection accuracy |

### Class Imbalance Handling

**Problem:** Light dysplastic (1,484) and Severe dysplastic (1,470) dominate.

**Solution:**
- Stratified sampling
- Class weights: `w_i = N / (n_classes × count_i)`
- SMOTE for minority classes (carcinoma_in_situ, normal_superficiel)

```python
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
```

---

## Known Issues & Solutions

### Issue 1: Zone.Identifier Files

**Problem:** Windows download creates `.bmp:Zone.Identifier` files.

**Solution:**
```bash
find data/raw/sipakmed/ -name "*:Zone.Identifier" -delete
```

### Issue 2: Uppercase vs Lowercase Extensions

**Problem:** Some files are `.BMP` (uppercase), others `.bmp` (lowercase).

**Solution:** Preprocessing script handles both cases automatically.

### Issue 3: Multiple Variants per Cell

**Problem:** Some cells have `-d` variants (different crops).

**Solution:** Include all variants as separate samples (they provide different views).

### Issue 4: Class Imbalance

**Problem:** Light dysplastic (36.7%) dominates.

**Solution:** Use stratified split + class weights in loss function.

---

## References

### Paper

**Title:** "SIPaKMeD: A New Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells in Pap Smear Images"

**Authors:** Plissiti, M.E., Dimitrakopoulos, P., Sfikas, G., Nikou, C., Krikoni, O., Charchanti, A.

**Conference:** IEEE International Conference on Image Processing (ICIP), 2018

**DOI:** [10.1109/ICIP.2018.8451588](https://doi.org/10.1109/ICIP.2018.8451588)

### Links

- **Dataset Page:** https://www.cs.uoi.gr/~marina/sipakmed.html
- **Paper (IEEE):** https://ieeexplore.ieee.org/document/8451588
- **University:** https://www.cs.uoi.gr/

### Contact

- **Maintainer:** Dr. Marina Plissiti
- **Email:** marina@cs.uoi.gr
- **Institution:** University of Ioannina, Department of Computer Science & Engineering

---

## Citation

```bibtex
@inproceedings{plissiti2018sipakmed,
  title={SIPaKMeD: A New Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells in Pap Smear Images},
  author={Plissiti, Marina E and Dimitrakopoulos, Panagiotis and Sfikas, Giorgos and Nikou, Christophoros and Krikoni, Olga and Charchanti, Antonia},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  pages={3144--3148},
  year={2018},
  organization={IEEE}
}
```

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-19 | 1.0.0 | Initial documentation based on downloaded dataset structure |

---

**Last Updated:** 2026-01-19
**Status:** ✅ Dataset downloaded and verified
**Maintainer:** CellViT-Optimus V14 Team
