#!/bin/bash
# extract_all_rgb_features.sh
#
# Extract RGB features from V13-Hybrid datasets for all families
#
# Usage: bash scripts/utils/extract_all_rgb_features.sh
#
# Prérequis:
#   - V13-Hybrid datasets déjà générés (data/family_data_v13_hybrid/)
#   - Environnement conda 'cellvit' activé
#   - GPU disponible (recommandé)
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
echo "RGB Features Extraction (All Families)"
echo "========================================"
echo ""

# Check conda environment
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    log_error "Conda environment not detected"
    log_error "Please activate cellvit environment: conda activate cellvit"
    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" != "cellvit" ]; then
    log_warning "Current environment: $CONDA_DEFAULT_ENV (expected: cellvit)"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

log_success "Conda environment: $CONDA_DEFAULT_ENV"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log_success "GPU available: $GPU_COUNT device(s)"
    DEVICE="cuda"
else
    log_warning "No GPU detected, using CPU (will be slower)"
    DEVICE="cpu"
fi

# Families to process
FAMILIES=("epidermal" "respiratory" "urologic" "digestive" "glandular")

# Batch size (adjust based on GPU VRAM)
BATCH_SIZE=8

echo ""
log "Processing ${#FAMILIES[@]} families: ${FAMILIES[*]}"
echo ""

# Extract RGB features for each family
for family in "${FAMILIES[@]}"; do
    echo "========================================"
    log "Extracting RGB features for: $family"
    echo "========================================"

    # Check if hybrid data exists
    HYBRID_FILE="data/family_data_v13_hybrid/${family}_data_v13_hybrid.npz"
    if [ ! -f "$HYBRID_FILE" ]; then
        log_error "Hybrid data not found: $HYBRID_FILE"
        log_error "Run prepare_v13_hybrid_dataset.py first"
        exit 1
    fi

    # Extract features
    python scripts/preprocessing/extract_rgb_features_from_hybrid.py \
        --family "$family" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        || { log_error "Failed to extract RGB features for $family"; exit 1; }

    log_success "RGB features extracted for $family"
    echo ""
done

log_success "All RGB features extracted successfully!"

# Display summary
echo ""
echo "========================================"
echo "📊 RGB FEATURES SUMMARY"
echo "========================================"
echo ""

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
echo "========================================"
echo "📝 NEXT STEP"
echo "========================================"
echo ""
echo "Start training with:"
echo "  python scripts/training/train_hovernet_family_v13_hybrid.py \\"
echo "      --family epidermal --epochs 30 --batch_size 16"
echo ""
echo "Or train all families:"
echo "  for family in glandular digestive urologic epidermal respiratory; do"
echo "      python scripts/training/train_hovernet_family_v13_hybrid.py \\"
echo "          --family \$family --epochs 30 --batch_size 16"
echo "  done"
echo ""

log_success "RGB features extraction script completed successfully!"
echo ""
