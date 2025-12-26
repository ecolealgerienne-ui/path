# Clean Split (Grouped Split) Implementation - Validation Guide

**Date:** 2025-12-26
**Objective:** Validate Clean Split implementation to prevent data leakage between train and validation sets

---

## 📋 What Was Implemented

### Problem Statement
**Before:** Simple 80/20 split on individual crops could put crops from the same source image in both train and val sets.
- **Impact:** Data leakage → inflated validation metrics
- **Risk:** Model memorizes specific image patterns rather than generalizing

**After:** Grouped split based on unique `source_image_ids` ensures:
- All crops from the same source image stay in the same set (train OR val, never both)
- Split is locked to disk with `split_types` field
- 80/20 ratio maintained at source image level

### Implementation Details

**File Modified:** `scripts/preprocessing/prepare_v13_hybrid_dataset.py`

**Changes:**
1. ✅ Added sklearn import (line 28):
   ```python
   from sklearn.model_selection import train_test_split
   ```

2. ✅ Created `create_clean_split_and_save()` function (lines 258-393):
   - Splits unique source IDs (80% train, 20% val)
   - Uses `random_state=42` for reproducibility
   - Creates `split_types` array (0=Train, 1=Val)
   - Includes 2 critical safety assertions:
     - No overlap: Ensures no crop appears in both sets
     - Complete assignment: Ensures all crops are assigned
   - Comprehensive logging for validation

3. ✅ Replaced old save logic (lines 585-604) with clean split call

---

## ✅ Validation Checklist

### Phase 1: Code Review (COMPLETED)

- [x] sklearn import added
- [x] `create_clean_split_and_save()` function created
- [x] Safety assertions implemented
- [x] Old save logic replaced
- [x] Metadata dictionary prepared correctly
- [x] Function signature matches expected parameters

### Phase 2: Execution Test (USER ACTION REQUIRED)

**Command:**
```bash
python scripts/preprocessing/prepare_v13_hybrid_dataset.py --family epidermal
```

**Expected Console Output:**
```
[... data loading and preprocessing ...]

================================================================================
🔒 CREATING CLEAN SPLIT (GROUPED BY SOURCE ID)
================================================================================

📊 Split Statistics:
   Total crops:          2570
   Unique source images: 514

📂 Source Image Split:
   Train images: 411 (80.0%)
   Val images:   103 (20.0%)

🔍 Safety Checks:
   ✅ No overlap: 0 crops in both train and val
   ✅ All crops assigned: 2570/2570

📦 Crop Split:
   Train crops: 2055 (80.0%)
   Val crops:   515 (20.0%)

💾 Saving dataset with locked split to: data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz
   ✅ Saved: 1234.56 MB

================================================================================
✅ CLEAN SPLIT CREATED AND LOCKED TO DISK
================================================================================

📝 Usage in training/evaluation scripts:
   Train set:      data['split_types'] == 0  (2055 crops)
   Validation set: data['split_types'] == 1  (515 crops)

⚠️  IMPORTANT: Always use split_types to load train/val data.
   This ensures NO data leakage between sets.

================================================================================
✅ V13-HYBRID DATASET PREPARATION COMPLETE: EPIDERMAL
================================================================================
```

**Validation Points:**
- [ ] Script runs without errors
- [ ] Source image split is exactly 80/20
- [ ] Both safety checks pass (✅)
- [ ] Crop split is approximately 80/20 (may vary slightly due to unequal crops per source)
- [ ] Output file exists: `data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz`
- [ ] File size is reasonable (~1-1.5 GB for epidermal)

### Phase 3: Data Integrity Verification (USER ACTION REQUIRED)

**Verification Script:**
```python
import numpy as np

# Load the dataset
data = np.load('data/family_data_v13_hybrid/epidermal_data_v13_hybrid.npz')

print("🔍 DATASET INTEGRITY VERIFICATION\n")

# 1. Check split_types field exists
assert 'split_types' in data, "❌ CRITICAL: split_types field missing!"
print("✅ split_types field present")

split_types = data['split_types']
source_ids = data['source_image_ids']

# 2. Verify split encoding
unique_splits = np.unique(split_types)
assert set(unique_splits) == {0, 1}, f"❌ Invalid split values: {unique_splits}"
print("✅ Split types encoded correctly (0=Train, 1=Val)")

# 3. Verify no overlap at source level
train_sources = np.unique(source_ids[split_types == 0])
val_sources = np.unique(source_ids[split_types == 1])
overlap = np.intersect1d(train_sources, val_sources)
assert len(overlap) == 0, f"❌ CRITICAL: {len(overlap)} source IDs in both train and val!"
print("✅ No source ID overlap")

# 4. Verify all sources assigned
all_sources = np.unique(source_ids)
assigned_sources = np.union1d(train_sources, val_sources)
assert len(all_sources) == len(assigned_sources), "❌ Some sources not assigned!"
print("✅ All source IDs assigned")

# 5. Print statistics
n_train = np.sum(split_types == 0)
n_val = np.sum(split_types == 1)
print(f"\n📊 SPLIT STATISTICS:")
print(f"   Total crops:      {len(split_types)}")
print(f"   Train crops:      {n_train} ({n_train/len(split_types)*100:.1f}%)")
print(f"   Val crops:        {n_val} ({n_val/len(split_types)*100:.1f}%)")
print(f"   Unique sources:   {len(all_sources)}")
print(f"   Train sources:    {len(train_sources)} ({len(train_sources)/len(all_sources)*100:.1f}%)")
print(f"   Val sources:      {len(val_sources)} ({len(val_sources)/len(all_sources)*100:.1f}%)")

print("\n🎉 ALL INTEGRITY CHECKS PASSED!\n")
```

**Expected Output:**
```
🔍 DATASET INTEGRITY VERIFICATION

✅ split_types field present
✅ Split types encoded correctly (0=Train, 1=Val)
✅ No source ID overlap
✅ All source IDs assigned

📊 SPLIT STATISTICS:
   Total crops:      2570
   Train crops:      2055 (80.0%)
   Val crops:        515 (20.0%)
   Unique sources:   514
   Train sources:    411 (80.0%)
   Val sources:      103 (20.0%)

🎉 ALL INTEGRITY CHECKS PASSED!
```

---

## 🔄 Next Steps After Validation

### 1. Update Downstream Scripts

**Training Script:** `scripts/training/train_hovernet_family_v13_hybrid.py`

Modify the dataset loading to use `split_types`:

```python
# BEFORE (old logic - DELETE):
n_total = len(images)
n_train = int(0.8 * n_total)
train_indices = np.arange(0, n_train)
val_indices = np.arange(n_train, n_total)

# AFTER (new logic - USE):
split_types = data['split_types']
train_indices = np.where(split_types == 0)[0]
val_indices = np.where(split_types == 1)[0]

print(f"📂 Loaded Clean Split:")
print(f"   Train: {len(train_indices)} samples")
print(f"   Val:   {len(val_indices)} samples")
```

**Evaluation Scripts:** Any script that loads validation data should use:
```python
split_types = data['split_types']
val_mask = split_types == 1
val_data = images[val_mask]  # or any other field
```

### 2. Re-evaluate Metrics with Clean Split

**Expected Impact:**
- AJI may decrease slightly (2-5%) due to eliminating data leakage
- This is EXPECTED and CORRECT behavior
- **Target:** AJI should remain ≥0.60 for validation success

**Comparison:**

| Metric | Before (leaky split) | After (clean split) | Status |
|--------|---------------------|---------------------|--------|
| AJI | 0.6447 | ≥0.60 expected | ✅ if ≥0.60 |
| Dice | 0.9316 | ~0.93 expected | Maintained |
| Over-seg | 0.95× | ~0.95× expected | Maintained |

**Quote from specs:**
> "Si on reste au-dessus de 0.60 AJI avec ce split propre, alors c'est gagné pour de bon."

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Solution:**
```bash
conda activate cellvit
pip install scikit-learn
```

### Issue: Safety check fails (overlap or missing crops)

**Cause:** Logic error in split creation

**Debug:**
```python
# Add debug prints in create_clean_split_and_save()
print(f"DEBUG: unique_ids = {unique_ids[:10]}...")
print(f"DEBUG: train_ids = {train_ids[:10]}...")
print(f"DEBUG: is_train sum = {np.sum(is_train)}")
print(f"DEBUG: is_val sum = {np.sum(is_val)}")
```

### Issue: split_types field missing in .npz

**Cause:** Old version of script was used

**Solution:**
1. Verify you're running the latest version
2. Check git commit: `git log -1 --oneline scripts/preprocessing/prepare_v13_hybrid_dataset.py`
3. Re-run data preparation

---

## 📝 Implementation Summary

**What Changed:**
- Data preparation now implements grouped split based on source_image_ids
- Split is locked to disk with `split_types` field
- Safety checks prevent data leakage

**What Stayed the Same:**
- Macenko normalization pipeline
- H-channel extraction (HED deconvolution)
- Target validation (HV float32 check)
- All metadata fields

**Benefits:**
- ✅ Eliminates data leakage
- ✅ More realistic validation metrics
- ✅ Reproducible splits (random_state=42)
- ✅ Easy to use in downstream scripts (single field lookup)

---

## ✅ Sign-Off

**Implementation Status:** ✅ COMPLETE
**Code Review:** ✅ PASSED
**Ready for Testing:** ✅ YES

**Next Action:** User should run Phase 2 (Execution Test) to validate the implementation.

**Expected Outcome:** Clean dataset with locked 80/20 split that prevents data leakage and provides realistic performance metrics.

---

**Document Version:** 1.0
**Date:** 2025-12-26
**Author:** Claude AI (CellViT-Optimus Development)
