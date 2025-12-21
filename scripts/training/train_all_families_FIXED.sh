#!/bin/bash
#
# Entraînement HoVer-Net pour les 4 familles restantes
# (Glandular déjà entraîné)
#
# Usage: bash scripts/training/train_all_families_FIXED.sh
#
# Temps estimé: ~7 heures total
#   - Digestive:    ~2.5h (2430 samples)
#   - Urologic:     ~2.0h (1101 samples)
#   - Respiratory:  ~1.5h (408 samples)
#   - Epidermal:    ~1.5h (571 samples)
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================================================"
echo "ENTRAÎNEMENT HOVERNET - 4 FAMILLES RESTANTES"
echo "========================================================================"
echo "Digestive, Urologic, Respiratory, Epidermal"
echo ""

# Configuration
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/family_FIXED}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/models/checkpoints_FIXED}"
EPOCHS=50
BATCH_SIZE=32

echo "Configuration:"
echo "  Data dir:      $DATA_DIR"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Augmentation:  ✅ Enabled"
echo ""

# Vérifier que les données existent
for family in digestive urologic respiratory epidermal; do
    data_file="$DATA_DIR/${family}_data_FIXED.npz"
    if [ ! -f "$data_file" ]; then
        echo "❌ ERREUR: Données manquantes pour $family"
        echo "   Fichier attendu: $data_file"
        echo "   Exécuter d'abord: bash scripts/preprocessing/generate_all_families_FIXED.sh"
        exit 1
    fi
done

echo "✅ Toutes les données FIXED sont disponibles"
echo ""

# Créer répertoire logs si nécessaire
mkdir -p "$PROJECT_ROOT/logs"

# Timestamp de début
START_TIME=$(date +%s)
echo "⏱️  Début: $(date)"
echo ""

# Famille 1/4: Digestive (~2.5h, 2430 samples)
echo "========================================================================"
echo "[1/4] DIGESTIVE - ~2.5 heures"
echo "========================================================================"
echo "Samples: 2430 (Colon, Stomach, Esophagus, Bile-duct)"
echo ""

python "$PROJECT_ROOT/scripts/training/train_hovernet_family.py" \
    --family digestive \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --augment \
    2>&1 | tee "$PROJECT_ROOT/logs/train_digestive_fixed.log"

echo ""
echo "✅ Digestive terminé - Checkpoint: $OUTPUT_DIR/hovernet_digestive_best.pth"
echo ""

# Famille 2/4: Urologic (~2h, 1101 samples)
echo "========================================================================"
echo "[2/4] UROLOGIC - ~2 heures"
echo "========================================================================"
echo "Samples: 1101 (Kidney, Bladder, Testis, Ovarian, Uterus, Cervix)"
echo ""

python "$PROJECT_ROOT/scripts/training/train_hovernet_family.py" \
    --family urologic \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --augment \
    2>&1 | tee "$PROJECT_ROOT/logs/train_urologic_fixed.log"

echo ""
echo "✅ Urologic terminé - Checkpoint: $OUTPUT_DIR/hovernet_urologic_best.pth"
echo ""

# Famille 3/4: Respiratory (~1.5h, 408 samples)
echo "========================================================================"
echo "[3/4] RESPIRATORY - ~1.5 heures"
echo "========================================================================"
echo "Samples: 408 (Lung, Liver)"
echo ""

python "$PROJECT_ROOT/scripts/training/train_hovernet_family.py" \
    --family respiratory \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --augment \
    2>&1 | tee "$PROJECT_ROOT/logs/train_respiratory_fixed.log"

echo ""
echo "✅ Respiratory terminé - Checkpoint: $OUTPUT_DIR/hovernet_respiratory_best.pth"
echo ""

# Famille 4/4: Epidermal (~1.5h, 571 samples)
echo "========================================================================"
echo "[4/4] EPIDERMAL - ~1.5 heures"
echo "========================================================================"
echo "Samples: 571 (Skin, HeadNeck)"
echo ""

python "$PROJECT_ROOT/scripts/training/train_hovernet_family.py" \
    --family epidermal \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --augment \
    2>&1 | tee "$PROJECT_ROOT/logs/train_epidermal_fixed.log"

echo ""
echo "✅ Epidermal terminé - Checkpoint: $OUTPUT_DIR/hovernet_epidermal_best.pth"
echo ""

# Résumé final
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo "========================================================================"
echo "ENTRAÎNEMENT COMPLET ✅"
echo "========================================================================"
echo ""
echo "⏱️  Durée totale: ${HOURS}h ${MINUTES}min"
echo ""
echo "Checkpoints générés:"
echo "  ✅ $OUTPUT_DIR/hovernet_glandular_best.pth (déjà fait)"
echo "  ✅ $OUTPUT_DIR/hovernet_digestive_best.pth"
echo "  ✅ $OUTPUT_DIR/hovernet_urologic_best.pth"
echo "  ✅ $OUTPUT_DIR/hovernet_respiratory_best.pth"
echo "  ✅ $OUTPUT_DIR/hovernet_epidermal_best.pth"
echo ""
echo "Logs sauvegardés dans: $PROJECT_ROOT/logs/"
echo ""
echo "🎯 PROCHAINES ÉTAPES:"
echo "   1. Tester les 4 modèles avec scripts/validation/test_glandular_model.py"
echo "   2. Mettre à jour l'IHM selon INTEGRATION_PLAN_HV_NORMALIZATION.md"
echo "   3. Déployer les nouveaux checkpoints"
echo ""
