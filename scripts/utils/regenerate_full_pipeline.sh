#!/bin/bash
# regenerate_full_pipeline.sh
#
# Pipeline complet de régénération des données CellViT-Optimus V13-Hybrid
# Inclut Clean Split (Grouped Split) pour prévenir data leakage
#
# Usage:
#   bash scripts/utils/regenerate_full_pipeline.sh
#
# Prérequis:
#   - PanNuke raw data dans /home/amar/data/PanNuke
#   - Environnement conda 'cellvit' configuré
#   - Au moins 11 GB d'espace disque
#   - Au moins 16 GB RAM
#
# Date: 2025-12-26
# Auteur: Claude AI (CellViT-Optimus Development)

set -e  # Exit on any error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Banner
echo ""
echo "========================================"
echo "CellViT-Optimus V13-Hybrid"
echo "Full Pipeline Regeneration"
echo "========================================"
echo ""

# Check prerequisites
log "Checking prerequisites..."

# Check PanNuke data
if [ ! -d "/home/amar/data/PanNuke" ]; then
    log_error "PanNuke data not found at /home/amar/data/PanNuke"
    log_error "Please download and extract PanNuke data first."
    log_error "See docs/REGENERATION_COMPLETE_PIPELINE.md Phase 1 for instructions."
    exit 1
fi

log_success "PanNuke data directory found"

# Check conda environment
if ! conda info --envs | grep -q "cellvit"; then
    log_error "Conda environment 'cellvit' not found"
    log_error "Please create the environment first: conda create -n cellvit python=3.10"
    exit 1
fi

log_success "Conda environment 'cellvit' found"

# Activate environment
log "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate cellvit
log_success "Environment activated"

# Check disk space (need at least 11 GB)
AVAILABLE_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
if [ "$AVAILABLE_GB" -lt 11 ]; then
    log_warning "Only ${AVAILABLE_GB} GB disk space available (need 11 GB)"
    log_warning "Continuing anyway, but monitor disk space..."
else
    log_success "${AVAILABLE_GB} GB disk space available"
fi

# Families to process
FAMILIES=("glandular" "digestive" "urologic" "epidermal" "respiratory")

# ============================================================================
# PHASE 2: Generate Family FIXED Data
# ============================================================================
echo ""
echo "========================================"
echo "PHASE 2: Generating Family FIXED Data"
echo "========================================"
echo ""

log "Processing ${#FAMILIES[@]} families: ${FAMILIES[*]}"

for family in "${FAMILIES[@]}"; do
    log "Processing family: $family"

    python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py \
        --family "$family" \
        --pannuke_dir /home/amar/data/PanNuke \
        || { log_error "Failed to generate FIXED data for $family"; exit 1; }

    log_success "FIXED data generated for $family"
done

log_success "Phase 2 complete: All Family FIXED data generated"

# ============================================================================
# PHASE 3: Prepare V13-Hybrid Datasets (with Clean Split)
# ============================================================================
echo ""
echo "========================================"
echo "PHASE 3: Preparing V13-Hybrid Datasets"
echo "        (Clean Split Enabled)"
echo "========================================"
echo ""

for family in "${FAMILIES[@]}"; do
    log "Preparing V13-Hybrid dataset for: $family"

    python scripts/preprocessing/prepare_v13_hybrid_dataset.py \
        --family "$family" \
        --source_data_dir data/family_FIXED \
        || { log_error "Failed to prepare V13-Hybrid for $family"; exit 1; }

    log_success "V13-Hybrid dataset prepared for $family"
done

log_success "Phase 3 complete: All V13-Hybrid datasets prepared"

# ============================================================================
# PHASE 3b: Verify Clean Split Integrity
# ============================================================================
echo ""
echo "========================================"
echo "PHASE 3b: Verifying Clean Split"
echo "========================================"
echo ""

ALL_VALID=true

for family in "${FAMILIES[@]}"; do
    log "Verifying Clean Split for: $family"

    if python scripts/validation/verify_clean_split.py \
        --data_file "data/family_data_v13_hybrid/${family}_data_v13_hybrid.npz"; then
        log_success "Clean Split valid for $family"
    else
        log_error "Clean Split validation FAILED for $family"
        ALL_VALID=false
    fi
done

if [ "$ALL_VALID" = true ]; then
    log_success "Phase 3b complete: All Clean Splits are VALID"
else
    log_error "Phase 3b FAILED: Some Clean Splits are invalid"
    log_error "Please check the error messages above and regenerate affected families."
    exit 1
fi

# ============================================================================
# PHASE 4: Extract H-Features
# ============================================================================
echo ""
echo "========================================"
echo "PHASE 4: Extracting H-Features"
echo "========================================"
echo ""

for family in "${FAMILIES[@]}"; do
    log "Extracting H-features for: $family"

    python scripts/preprocessing/extract_h_features_v13.py \
        --family "$family" \
        --hybrid_data_dir data/family_data_v13_hybrid \
        || { log_error "Failed to extract H-features for $family"; exit 1; }

    log_success "H-features extracted for $family"
done

log_success "Phase 4 complete: All H-features extracted"

# ============================================================================
# PHASE 4b: Extract RGB Features
# ============================================================================
echo ""
echo "========================================"
echo "PHASE 4b: Extracting RGB Features"
echo "========================================"
echo ""

for family in "${FAMILIES[@]}"; do
    log "Extracting RGB features for: $family"

    python scripts/preprocessing/extract_rgb_features_from_hybrid.py \
        --family "$family" \
        --batch_size 8 \
        || { log_error "Failed to extract RGB features for $family"; exit 1; }

    log_success "RGB features extracted for $family"
done

log_success "Phase 4b complete: All RGB features extracted"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo "✅ PIPELINE REGENERATION COMPLETE"
echo "========================================"
echo ""

log_success "All phases completed successfully!"

echo ""
echo "📊 GENERATED FILES:"
echo ""
echo "Family FIXED data (data/family_FIXED/):"
for family in "${FAMILIES[@]}"; do
    FILE="data/family_FIXED/${family}_data_FIXED.npz"
    if [ -f "$FILE" ]; then
        SIZE=$(du -h "$FILE" | cut -f1)
        echo "  ✅ $family: $SIZE"
    else
        echo "  ❌ $family: MISSING"
    fi
done

echo ""
echo "V13-Hybrid datasets (data/family_data_v13_hybrid/):"
for family in "${FAMILIES[@]}"; do
    FILE="data/family_data_v13_hybrid/${family}_data_v13_hybrid.npz"
    if [ -f "$FILE" ]; then
        SIZE=$(du -h "$FILE" | cut -f1)
        echo "  ✅ $family: $SIZE"
    else
        echo "  ❌ $family: MISSING"
    fi
done

echo ""
echo "H-Features (data/cache/family_data/):"
for family in "${FAMILIES[@]}"; do
    FILE="data/cache/family_data/${family}_h_features_v13.npz"
    if [ -f "$FILE" ]; then
        SIZE=$(du -h "$FILE" | cut -f1)
        echo "  ✅ $family: $SIZE"
    else
        echo "  ❌ $family: MISSING"
    fi
done

echo ""
echo "RGB Features (data/cache/family_data/):"
for family in "${FAMILIES[@]}"; do
    FILE="data/cache/family_data/${family}_rgb_features_v13.npz"
    if [ -f "$FILE" ]; then
        SIZE=$(du -h "$FILE" | cut -f1)
        echo "  ✅ $family: $SIZE"
    else
        echo "  ❌ $family: MISSING"
    fi
done

echo ""
echo "📝 NEXT STEPS:"
echo ""
echo "1. Start training (example for epidermal):"
echo "   python scripts/training/train_hovernet_family_v13_hybrid.py \\"
echo "       --family epidermal --epochs 30 --batch_size 16"
echo ""
echo "2. Or train all families:"
echo "   for family in glandular digestive urologic epidermal respiratory; do"
echo "       python scripts/training/train_hovernet_family_v13_hybrid.py \\"
echo "           --family \$family --epochs 30 --batch_size 16"
echo "   done"
echo ""
echo "3. After training, evaluate with:"
echo "   python scripts/evaluation/test_v13_hybrid_aji.py \\"
echo "       --checkpoint models/checkpoints_v13_hybrid/hovernet_epidermal_v13_hybrid_best.pth \\"
echo "       --family epidermal --n_samples 50"
echo ""

log_success "Pipeline regeneration script completed successfully!"
echo ""
