#!/bin/bash
#
# Test rapide Ground Truth (5 échantillons)
#
# Vérifie que les modèles FIXED fonctionnent correctement
# avant de lancer l'évaluation complète.
#
# Usage: bash scripts/evaluation/quick_test_fixed.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================================================"
echo "TEST RAPIDE - MODÈLES FIXED"
echo "========================================================================"
echo ""

# Configuration
CHECKPOINT_DIR="$PROJECT_ROOT/models/checkpoints_FIXED"
CONVERTED_DIR="$PROJECT_ROOT/data/evaluation/pannuke_fold2_converted"
OUTPUT_DIR="$PROJECT_ROOT/results/quick_test_FIXED"

# Vérifier checkpoints
echo "🔍 Vérification checkpoints FIXED..."
if [ ! -f "$CHECKPOINT_DIR/hovernet_glandular_best.pth" ]; then
    echo "❌ Glandular checkpoint manquant"
    echo "   Les autres familles peuvent ne pas encore être entraînées."
    echo "   Ce test utilisera uniquement Glandular si disponible."
fi

# Convertir annotations si nécessaire
if [ ! -d "$CONVERTED_DIR" ]; then
    echo "📦 Conversion annotations PanNuke Fold 2..."
    python "$PROJECT_ROOT/scripts/evaluation/convert_annotations.py" \
        --dataset pannuke \
        --input_dir /home/amar/data/PanNuke/fold2 \
        --output_dir "$CONVERTED_DIR"
fi

# Test rapide (5 échantillons)
echo ""
echo "🧪 Test sur 5 échantillons..."
python "$PROJECT_ROOT/scripts/evaluation/evaluate_ground_truth.py" \
    --dataset_dir "$CONVERTED_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --num_samples 5 \
    --dataset pannuke

echo ""
echo "✅ Test rapide terminé"
echo ""
echo "📊 Consulter: cat $OUTPUT_DIR/clinical_report_*.txt"
echo ""
echo "🎯 Si OK → Lancer l'évaluation complète:"
echo "   bash scripts/evaluation/test_fixed_models_ground_truth.sh"
echo ""
