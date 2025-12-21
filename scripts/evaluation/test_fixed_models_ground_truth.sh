#!/bin/bash
#
# Test Ground Truth pour modèles FIXED
#
# Évalue les 5 familles HoVer-Net contre annotations PanNuke expertes.
# Génère un rapport de fidélité clinique complet.
#
# Usage: bash scripts/evaluation/test_fixed_models_ground_truth.sh
#
# Prérequis:
#   - Modèles FIXED entraînés dans models/checkpoints_FIXED/
#   - PanNuke téléchargé dans /home/amar/data/PanNuke
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================================================"
echo "ÉVALUATION GROUND TRUTH - MODÈLES FIXED"
echo "========================================================================"
echo ""

# Configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/models/checkpoints_FIXED}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/ground_truth_FIXED}"
NUM_SAMPLES=50  # Nombre d'échantillons par famille (ajustable)

echo "Configuration:"
echo "  Checkpoints:   $CHECKPOINT_DIR"
echo "  Output:        $OUTPUT_DIR"
echo "  Samples:       $NUM_SAMPLES par famille"
echo ""

# Vérifier que les checkpoints FIXED existent
echo "🔍 Vérification des checkpoints FIXED..."
required_checkpoints=(
    "hovernet_glandular_best.pth"
    "hovernet_digestive_best.pth"
    "hovernet_urologic_best.pth"
    "hovernet_respiratory_best.pth"
    "hovernet_epidermal_best.pth"
)

missing=0
for ckpt in "${required_checkpoints[@]}"; do
    if [ ! -f "$CHECKPOINT_DIR/$ckpt" ]; then
        echo "  ❌ Manquant: $ckpt"
        missing=$((missing + 1))
    else
        echo "  ✅ $ckpt"
    fi
done

if [ $missing -gt 0 ]; then
    echo ""
    echo "❌ ERREUR: $missing checkpoint(s) manquant(s)"
    echo "   Les modèles doivent être entraînés avant l'évaluation GT."
    echo "   Exécuter: bash scripts/training/train_all_families_FIXED.sh"
    exit 1
fi

echo ""
echo "✅ Tous les checkpoints présents"
echo ""

# Créer répertoires de sortie
mkdir -p "$OUTPUT_DIR/reports"
mkdir -p "$OUTPUT_DIR/visualizations"
mkdir -p "$PROJECT_ROOT/logs"

# Timestamp de début
START_TIME=$(date +%s)
echo "⏱️  Début: $(date)"
echo ""

# Note: Le script evaluate_ground_truth.py va évaluer TOUTES les familles
# ensemble en utilisant OptimusGateInferenceMultiFamily.
# Ce système charge automatiquement les 5 checkpoints et route vers
# le bon modèle selon l'organe détecté.

echo "========================================================================"
echo "ÉVALUATION SUR PANNUKE"
echo "========================================================================"
echo "Échantillons: $NUM_SAMPLES images (mélangées de toutes les familles)"
echo ""

# Préparer dataset PanNuke pour évaluation
# On va utiliser Fold 2 qui n'a PAS été utilisé pour l'entraînement
PANNUKE_FOLD2="/home/amar/data/PanNuke/fold2"

if [ ! -d "$PANNUKE_FOLD2" ]; then
    echo "❌ ERREUR: PanNuke Fold 2 introuvable: $PANNUKE_FOLD2"
    echo "   Télécharger PanNuke d'abord."
    exit 1
fi

# Convertir annotations PanNuke Fold 2 si pas déjà fait
CONVERTED_DIR="$PROJECT_ROOT/data/evaluation/pannuke_fold2_converted"

if [ ! -d "$CONVERTED_DIR" ]; then
    echo "📦 Conversion annotations PanNuke Fold 2..."
    python "$PROJECT_ROOT/scripts/evaluation/convert_annotations.py" \
        --dataset pannuke \
        --input_dir "$PANNUKE_FOLD2" \
        --output_dir "$CONVERTED_DIR" \
        2>&1 | tee "$PROJECT_ROOT/logs/convert_pannuke_fold2.log"
    echo ""
fi

# Évaluation Ground Truth
echo "🧪 Évaluation des prédictions vs annotations expertes..."
python "$PROJECT_ROOT/scripts/evaluation/evaluate_ground_truth.py" \
    --dataset_dir "$CONVERTED_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --num_samples $NUM_SAMPLES \
    --dataset pannuke \
    2>&1 | tee "$PROJECT_ROOT/logs/evaluate_ground_truth_FIXED.log"

echo ""

# Résumé final
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "========================================================================"
echo "ÉVALUATION COMPLÈTE ✅"
echo "========================================================================"
echo ""
echo "⏱️  Durée totale: ${MINUTES}min ${SECONDS}s"
echo ""
echo "📊 Rapports générés:"
find "$OUTPUT_DIR" -name "*.txt" -o -name "*.json" 2>/dev/null | while read f; do
    echo "  ✅ $f"
done
echo ""
echo "📈 Visualisations (si générées):"
find "$OUTPUT_DIR/visualizations" -name "*.png" 2>/dev/null | head -5 | while read f; do
    echo "  ✅ $f"
done
echo ""
echo "🎯 PROCHAINE ÉTAPE:"
echo "   1. Consulter le rapport: cat $OUTPUT_DIR/clinical_report_*.txt"
echo "   2. Analyser les métriques JSON"
echo "   3. Si fidélité OK → Déployer les checkpoints FIXED"
echo ""
