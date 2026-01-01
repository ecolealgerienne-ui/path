#!/bin/bash
# =============================================================================
# CellViT-Optimus — Generate Benchmark Reports for All Families
# =============================================================================
#
# Usage:
#   ./scripts/benchmark/run_all_benchmarks.sh          # 30 samples per family
#   ./scripts/benchmark/run_all_benchmarks.sh 10       # 10 samples per family (test)
#
# Output:
#   benchmark/
#   ├── respiratory/
#   ├── urologic/
#   ├── glandular/
#   ├── epidermal/
#   └── digestive/
#
# =============================================================================

set -e

# Nombre de samples (défaut: 30)
N_SAMPLES=${1:-30}

echo "=============================================================="
echo "  CellViT-Optimus Benchmark Report Generator"
echo "=============================================================="
echo ""
echo "  Samples per family: $N_SAMPLES"
echo "  Output directory: benchmark/"
echo ""

FAMILIES=("respiratory" "urologic" "glandular" "epidermal" "digestive")

for family in "${FAMILIES[@]}"; do
    echo ""
    echo "=============================================================="
    echo "  Processing: $family"
    echo "=============================================================="

    python scripts/benchmark/generate_benchmark_report.py \
        --family "$family" \
        --n_samples "$N_SAMPLES" \
        --min_types 2

    echo ""
    echo "  ✅ $family complete"
done

echo ""
echo "=============================================================="
echo "  ALL BENCHMARKS COMPLETE"
echo "=============================================================="
echo ""
echo "  Output structure:"
echo "    benchmark/"
for family in "${FAMILIES[@]}"; do
    echo "    ├── $family/"
    echo "    │   ├── images/ (${N_SAMPLES} comparisons)"
    echo "    │   ├── images/raw/ (${N_SAMPLES} originals)"
    echo "    │   ├── rapport_${family}.html"
    echo "    │   └── metrics_${family}.csv"
done
echo ""
echo "  Open any rapport_*.html in a browser to view results."
echo ""
