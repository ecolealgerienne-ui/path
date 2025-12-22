# 🚨 Watershed Generalization Issue

**Date:** 2025-12-22
**Status:** 🔴 CRITICAL - Parameters optimized for 1 image don't generalize
**Impact:** Recall 7-8.5% across 10 images (missing 91-92% of instances)

---

## 📊 Problem Discovery

### Single Image Success (image_00000.npz)
```
GT:   4 instances
Pred: 4 instances
Error: 0 (PERFECT!)

Parameters: edge=0.2, dist=1, min=10
```

### Multi-Image Failure (10 random images)

**Glandular Family:**
```
Total GT:    117 instances
Total Pred:  146 instances
TP:  8 (6.8%)
FN:  109 (93.2% MISSED!)
Dice: 0.8165
AJI:  0.1424
```

**Digestive Family:**
```
Total GT:    117 instances
Total Pred:  16 instances (86% UNDER-DETECTION!)
TP:  10 (8.5%)
FN:  107 (91.5% MISSED!)
Dice: 0.9158 (good quality but detects nothing)
AJI:  0.0919
```

---

## 🔍 Root Cause Analysis

### Why Parameters Don't Generalize

**Grid search on `optimize_watershed_params.py` found:**
- Tested 245 combinations on **1 image** (image_00000.npz)
- Found perfect match: `edge=0.2, dist=1, min=10` → Error=0
- **But these are overfitted to that specific image!**

**Evidence of overfitting:**

| Image | HV Gradient Max | Nuclei Density | Optimal edge_threshold |
|-------|-----------------|----------------|------------------------|
| image_00000 | 2.516 | Low (4 instances) | 0.2 (works!) |
| Other images | Variable | Variable | Likely different |

**Why Digestive is worse:**
- Digestive tissue has **different HV gradient characteristics**
- Parameters optimized for Glandular-like tissue
- Results in severe under-detection (only 16 instances detected)

---

## 📉 Type Classification Failure

**Ground Truth Distribution (10 images):**
```
Neoplastic:   74 instances (63%)
Inflammatory: 20 instances (17%)
Dead:          4 instances (3%)
Epithelial:   19 instances (16%)
```

**Glandular Predictions:**
```
Neoplastic:    4 instances (94.6% miss!) ❌
Inflammatory: 24 instances (over-detection)
Dead:          0 instances (100% miss!) ❌
Epithelial:  118 instances (6.2x over!) ❌
```

**Digestive Predictions:**
```
Neoplastic:    0 instances (100% miss!) ❌
Inflammatory:  0 instances (100% miss!) ❌
Dead:          0 instances (100% miss!) ❌
Epithelial:   16 instances (84% under-detection)
```

**Conclusion:** Type classification is completely wrong because we're:
1. Missing 91% of instances (watershed failure)
2. Using wrong family routing (no OrganHead)

---

## 🔧 Solutions

### Option 1: Find Global Parameters (Quick Fix)

Run the batch diagnostic to find parameters that work across ALL images:

```bash
python scripts/evaluation/diagnose_watershed_batch.py \
    --dataset_dir data/evaluation/pannuke_fold2_converted \
    --checkpoint_dir models/checkpoints_FIXED \
    --num_samples 10 \
    --force_family glandular
```

**This tests 4 parameter sets:**
1. **CURRENT** (edge=0.2, dist=1, min=10) - optimized for 1 image
2. **CONSERVATIVE** (edge=0.3, dist=2, min=10) - less sensitive
3. **AGGRESSIVE** (edge=0.1, dist=1, min=5) - more sensitive
4. **BALANCED** (edge=0.15, dist=2, min=8) - middle ground

### Option 2: Adaptive Parameters (Long-term Fix)

Implement **per-image adaptive thresholding** based on HV gradient statistics:

```python
def adaptive_watershed(np_mask, hv_map):
    """Adapt parameters based on HV gradient characteristics."""
    gradient = compute_gradient_magnitude(hv_map)

    # Adaptive edge threshold based on gradient percentiles
    edge_threshold = np.percentile(gradient[np_mask > 0], 50)

    # Adaptive dist_threshold based on nuclei density
    density = np_mask.sum() / np_mask.size
    dist_threshold = 3 if density > 0.3 else 1

    # Adaptive min_size based on median instance size
    min_size = estimate_median_nuclei_size(np_mask) * 0.5

    return watershed_post_process(np_mask, hv_map,
                                  edge_threshold, dist_threshold, min_size)
```

### Option 3: Re-optimize on Multiple Images

Re-run grid search on **10-20 representative images** and average the optimal parameters:

```bash
# For each of 10 images, find optimal params
for i in {0..9}; do
    python scripts/evaluation/optimize_watershed_params.py \
        --npz_file data/evaluation/pannuke_fold2_converted/image_0000${i}.npz \
        --checkpoint_dir models/checkpoints_FIXED
done

# Then compute average optimal parameters across all 10 results
```

---

## 🎯 Expected Improvements

If we find good global parameters, we should see:

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Recall | 7-8.5% | >90% | **+82-83%** |
| AJI | 0.09-0.14 | >0.80 | **+0.66-0.71** |
| PQ | 0.04-0.12 | >0.70 | **+0.58-0.66** |
| Detection Rate | 14-136% | ~100% | Better balance |

---

## 📝 Lessons Learned

1. **Never optimize on a single example** - Always test on a held-out validation set
2. **Grid search needs diversity** - Parameters must work across tissue types
3. **Adaptive methods are better** - Image-specific characteristics should guide parameters
4. **Validation is critical** - Single image "perfection" doesn't mean generalization

---

## 🚀 Next Steps

1. **Run batch diagnostic** - Find which parameter set works best globally
2. **Update production code** - Apply the globally-optimal parameters
3. **Re-run full evaluation** - Verify metrics improve to targets
4. **Consider adaptive method** - Long-term solution for robustness

---

**Related Documents:**
- `WATERSHED_FIX_SUMMARY.md` - Algorithm fix (peak_local_max)
- `WATERSHED_OPTIMIZATION_GUIDE.md` - Original single-image optimization
- `scripts/evaluation/optimize_watershed_params.py` - Grid search script
- `scripts/evaluation/diagnose_watershed_batch.py` - Multi-image diagnostic

---

**Created:** 2025-12-22
**By:** Claude (Generalization Analysis)
**Commit:** e02e59e (batch diagnostic)
**Status:** ⏳ Awaiting batch diagnostic results
