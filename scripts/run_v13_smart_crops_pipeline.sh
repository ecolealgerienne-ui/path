#!/bin/bash

################################################################################
# Pipeline V13 Smart Crops - Automation Complète
################################################################################
#
# Ce script orchestre le workflow complet pour V13 Smart Crops:
#   1. Préparation données (5 crops + rotations)
#   2. Validation HV rotation
#   3. Extraction features H-optimus-0 (train + val)
#   4. Entraînement HoVer-Net
#   5. Évaluation AJI
#
# Usage:
#   bash scripts/run_v13_smart_crops_pipeline.sh epidermal
#   bash scripts/run_v13_smart_crops_pipeline.sh glandular --epochs 60
#
# Prérequis:
#   - Données source: data/family_FIXED/{family}_data_FIXED.npz
#   - GPU disponible pour entraînement
#   - Conda env 'cellvit' activé
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

################################################################################
# Configuration
################################################################################

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paramètres par défaut
FAMILY=""
EPOCHS=30
BATCH_SIZE=16
LAMBDA_NP=1.0
LAMBDA_HV=2.0
LAMBDA_NT=1.0
BETA=1.50
MIN_SIZE=40
N_SAMPLES=50
DEVICE="cuda"

# Répertoires
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/family_FIXED"
OUTPUT_DIR="${PROJECT_ROOT}/data/family_data_v13_smart_crops"
CACHE_DIR="${PROJECT_ROOT}/data/cache/family_data"
CHECKPOINT_DIR="${PROJECT_ROOT}/models/checkpoints_v13_smart_crops"
RESULTS_DIR="${PROJECT_ROOT}/results/v13_smart_crops"

################################################################################
# Fonctions Helper
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_separator() {
    echo ""
    echo "================================================================================"
}

print_header() {
    print_separator
    echo -e "${GREEN}$1${NC}"
    print_separator
}

check_conda_env() {
    if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "cellvit" ]]; then
        log_error "Conda environment 'cellvit' not activated"
        log_info "Please run: conda activate cellvit"
        exit 1
    fi
    log_success "Conda environment 'cellvit' is active"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found - GPU required for training"
        exit 1
    fi

    if ! nvidia-smi &> /dev/null; then
        log_error "No GPU detected"
        exit 1
    fi

    log_success "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
}

check_source_data() {
    local source_file="${DATA_DIR}/${FAMILY}_data_FIXED.npz"

    if [[ ! -f "${source_file}" ]]; then
        log_error "Source data not found: ${source_file}"
        log_info ""
        log_info "You need to generate source data first:"
        log_info "  python scripts/preprocessing/prepare_family_data_FIXED.py --family ${FAMILY}"
        exit 1
    fi

    log_success "Source data found: ${source_file}"
}

estimate_time() {
    local step=$1

    case "${step}" in
        prepare)
            echo "~5 minutes"
            ;;
        validate)
            echo "~2 minutes"
            ;;
        extract_train)
            echo "~1 minute"
            ;;
        extract_val)
            echo "~1 minute"
            ;;
        train)
            echo "~40 minutes (${EPOCHS} epochs)"
            ;;
        evaluate)
            echo "~5 minutes (${N_SAMPLES} samples)"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

################################################################################
# Étapes du Pipeline
################################################################################

step_1_prepare_data() {
    print_header "ÉTAPE 1/6: Préparation Données V13 Smart Crops"

    log_info "Generating 5 crops with deterministic rotations..."
    log_info "Time estimate: $(estimate_time prepare)"

    python "${PROJECT_ROOT}/scripts/preprocessing/prepare_v13_smart_crops.py" \
        --family "${FAMILY}" \
        --source_data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}"

    log_success "Data preparation completed"

    # Vérifier fichiers générés
    local train_file="${OUTPUT_DIR}/${FAMILY}_train_v13_smart_crops.npz"
    local val_file="${OUTPUT_DIR}/${FAMILY}_val_v13_smart_crops.npz"

    if [[ ! -f "${train_file}" ]] || [[ ! -f "${val_file}" ]]; then
        log_error "Expected output files not found"
        exit 1
    fi

    log_info "Created files:"
    log_info "  - ${train_file} ($(du -h "${train_file}" | cut -f1))"
    log_info "  - ${val_file} ($(du -h "${val_file}" | cut -f1))"
}

step_2_validate_hv() {
    print_header "ÉTAPE 2/6: Validation HV Rotation"

    log_info "Validating HV divergence after rotation transformations..."
    log_info "Time estimate: $(estimate_time validate)"

    python "${PROJECT_ROOT}/scripts/validation/validate_hv_rotation.py" \
        --data_dir "${OUTPUT_DIR}" \
        --family "${FAMILY}"

    log_success "HV rotation validation completed"
}

step_3_extract_features_train() {
    print_header "ÉTAPE 3/6: Extraction Features H-optimus-0 (TRAIN)"

    log_info "Extracting H-optimus-0 features for training split..."
    log_info "Time estimate: $(estimate_time extract_train)"

    python "${PROJECT_ROOT}/scripts/preprocessing/extract_features_v13_smart_crops.py" \
        --family "${FAMILY}" \
        --split train \
        --data_dir "${OUTPUT_DIR}" \
        --output_dir "${CACHE_DIR}" \
        --batch_size 8 \
        --device "${DEVICE}"

    log_success "Train features extraction completed"

    local features_file="${CACHE_DIR}/${FAMILY}_rgb_features_v13_smart_crops_train.npz"
    log_info "Created: ${features_file} ($(du -h "${features_file}" | cut -f1))"
}

step_4_extract_features_val() {
    print_header "ÉTAPE 4/6: Extraction Features H-optimus-0 (VAL)"

    log_info "Extracting H-optimus-0 features for validation split..."
    log_info "Time estimate: $(estimate_time extract_val)"

    python "${PROJECT_ROOT}/scripts/preprocessing/extract_features_v13_smart_crops.py" \
        --family "${FAMILY}" \
        --split val \
        --data_dir "${OUTPUT_DIR}" \
        --output_dir "${CACHE_DIR}" \
        --batch_size 8 \
        --device "${DEVICE}"

    log_success "Val features extraction completed"

    local features_file="${CACHE_DIR}/${FAMILY}_rgb_features_v13_smart_crops_val.npz"
    log_info "Created: ${features_file} ($(du -h "${features_file}" | cut -f1))"
}

step_5_train_hovernet() {
    print_header "ÉTAPE 5/6: Entraînement HoVer-Net V13 Smart Crops"

    log_info "Training HoVer-Net on V13 Smart Crops data..."
    log_info "Time estimate: $(estimate_time train)"
    log_info "Parameters:"
    log_info "  - Epochs: ${EPOCHS}"
    log_info "  - Batch size: ${BATCH_SIZE}"
    log_info "  - λ_np: ${LAMBDA_NP}, λ_hv: ${LAMBDA_HV}, λ_nt: ${LAMBDA_NT}"

    mkdir -p "${CHECKPOINT_DIR}"

    python "${PROJECT_ROOT}/scripts/training/train_hovernet_family_v13_smart_crops.py" \
        --family "${FAMILY}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lambda_np "${LAMBDA_NP}" \
        --lambda_hv "${LAMBDA_HV}" \
        --lambda_nt "${LAMBDA_NT}" \
        --cache_dir "${CACHE_DIR}" \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --device "${DEVICE}"

    log_success "Training completed"

    local checkpoint_file="${CHECKPOINT_DIR}/hovernet_${FAMILY}_v13_smart_crops_best.pth"
    log_info "Best checkpoint: ${checkpoint_file} ($(du -h "${checkpoint_file}" | cut -f1))"
}

step_6_evaluate_aji() {
    print_header "ÉTAPE 6/6: Évaluation AJI"

    log_info "Evaluating model on validation set..."
    log_info "Time estimate: $(estimate_time evaluate)"
    log_info "Parameters:"
    log_info "  - Samples: ${N_SAMPLES}"
    log_info "  - Beta: ${BETA}, Min size: ${MIN_SIZE}"

    mkdir -p "${RESULTS_DIR}"

    local checkpoint_file="${CHECKPOINT_DIR}/hovernet_${FAMILY}_v13_smart_crops_best.pth"

    python "${PROJECT_ROOT}/scripts/evaluation/test_v13_smart_crops_aji.py" \
        --checkpoint "${checkpoint_file}" \
        --family "${FAMILY}" \
        --n_samples "${N_SAMPLES}" \
        --beta "${BETA}" \
        --min_size "${MIN_SIZE}" \
        --data_dir "${OUTPUT_DIR}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${RESULTS_DIR}" \
        --device "${DEVICE}"

    log_success "Evaluation completed"

    # Afficher résultats
    local latest_result=$(ls -t "${RESULTS_DIR}"/aji_results_${FAMILY}_*.json 2>/dev/null | head -n1)

    if [[ -n "${latest_result}" ]]; then
        log_info ""
        log_info "Results saved to: ${latest_result}"
        log_info ""

        # Extract key metrics with jq if available
        if command -v jq &> /dev/null; then
            local mean_aji=$(jq -r '.mean_aji' "${latest_result}")
            local mean_dice=$(jq -r '.mean_dice' "${latest_result}")
            local mean_pq=$(jq -r '.mean_pq' "${latest_result}")
            local verdict=$(jq -r '.verdict' "${latest_result}")

            log_info "📊 MÉTRIQUES FINALES:"
            log_info "  - Dice:  ${mean_dice}"
            log_info "  - AJI:   ${mean_aji}"
            log_info "  - PQ:    ${mean_pq}"
            log_info ""
            log_info "🎯 VERDICT: ${verdict}"
        fi
    fi
}

################################################################################
# Pipeline Principal
################################################################################

run_pipeline() {
    local start_time=$(date +%s)

    print_header "V13 SMART CROPS PIPELINE - ${FAMILY^^}"

    log_info "Configuration:"
    log_info "  Family:      ${FAMILY}"
    log_info "  Epochs:      ${EPOCHS}"
    log_info "  Batch size:  ${BATCH_SIZE}"
    log_info "  Device:      ${DEVICE}"
    log_info ""

    # Vérifications préalables
    log_info "Running pre-flight checks..."
    check_conda_env
    check_gpu
    check_source_data
    log_info ""

    # Exécuter les étapes
    step_1_prepare_data
    step_2_validate_hv
    step_3_extract_features_train
    step_4_extract_features_val
    step_5_train_hovernet
    step_6_evaluate_aji

    # Résumé final
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))

    print_separator
    log_success "PIPELINE COMPLETED SUCCESSFULLY"
    print_separator
    log_info "Total time: ${hours}h ${minutes}m ${seconds}s"
    log_info ""
    log_info "Generated files:"
    log_info "  - Data:       ${OUTPUT_DIR}/${FAMILY}_*_v13_smart_crops.npz"
    log_info "  - Features:   ${CACHE_DIR}/${FAMILY}_rgb_features_v13_smart_crops_*.npz"
    log_info "  - Checkpoint: ${CHECKPOINT_DIR}/hovernet_${FAMILY}_v13_smart_crops_best.pth"
    log_info "  - Results:    ${RESULTS_DIR}/aji_results_${FAMILY}_*.json"
    print_separator
}

################################################################################
# Parsing Arguments
################################################################################

usage() {
    echo "Usage: $0 FAMILY [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  FAMILY                   Family to process (glandular, digestive, urologic, epidermal, respiratory)"
    echo ""
    echo "Options:"
    echo "  --epochs N               Number of training epochs (default: 30)"
    echo "  --batch-size N           Batch size for training (default: 16)"
    echo "  --lambda-np N            Weight for NP loss (default: 1.0)"
    echo "  --lambda-hv N            Weight for HV loss (default: 2.0)"
    echo "  --lambda-nt N            Weight for NT loss (default: 1.0)"
    echo "  --beta N                 Beta parameter for watershed (default: 1.50)"
    echo "  --min-size N             Minimum instance size (default: 40)"
    echo "  --n-samples N            Number of samples to evaluate (default: 50)"
    echo "  --device DEVICE          Device (cuda or cpu, default: cuda)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 epidermal --epochs 60 --batch-size 32"
    exit 0
}

# Parse arguments
if [[ $# -eq 0 ]]; then
    usage
fi

FAMILY="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lambda-np)
            LAMBDA_NP="$2"
            shift 2
            ;;
        --lambda-hv)
            LAMBDA_HV="$2"
            shift 2
            ;;
        --lambda-nt)
            LAMBDA_NT="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --min-size)
            MIN_SIZE="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate family
case "${FAMILY}" in
    glandular|digestive|urologic|epidermal|respiratory)
        ;;
    *)
        log_error "Invalid family: ${FAMILY}"
        log_info "Valid families: glandular, digestive, urologic, epidermal, respiratory"
        exit 1
        ;;
esac

# Run pipeline
run_pipeline
