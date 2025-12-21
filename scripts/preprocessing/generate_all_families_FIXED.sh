#!/bin/bash
#
# Génération des données FIXED pour les 4 familles restantes
# (Glandular déjà fait)
#
# Usage: bash scripts/preprocessing/generate_all_families_FIXED.sh
#
# Temps estimé: ~20 minutes total
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================================================"
echo "GÉNÉRATION DONNÉES FIXED - 4 FAMILLES RESTANTES"
echo "========================================================================"
echo "Digestive, Urologic, Respiratory, Epidermal"
echo ""

# Configuration
DATA_DIR="${DATA_DIR:-/home/amar/data/PanNuke}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/family_FIXED}"
CHUNK_SIZE=500

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ ERREUR: Répertoire PanNuke introuvable: $DATA_DIR"
    echo "   Définir avec: export DATA_DIR=/path/to/PanNuke"
    exit 1
fi

echo "Configuration:"
echo "  Data dir:   $DATA_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Chunk size: $CHUNK_SIZE"
echo ""

# Famille 1/4: Digestive (~5 min, 2430 samples)
echo "========================================================================"
echo "[1/4] DIGESTIVE - Colon, Stomach, Esophagus, Bile-duct"
echo "========================================================================"
python "$PROJECT_ROOT/scripts/preprocessing/prepare_family_data_FIXED.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --family digestive \
    --chunk_size $CHUNK_SIZE \
    2>&1 | tee "$PROJECT_ROOT/logs/digestive_fixed_generation.log"

echo ""
echo "✅ Digestive terminé"
echo ""

# Famille 2/4: Urologic (~3 min, 1101 samples)
echo "========================================================================"
echo "[2/4] UROLOGIC - Kidney, Bladder, Testis, Ovarian, Uterus, Cervix"
echo "========================================================================"
python "$PROJECT_ROOT/scripts/preprocessing/prepare_family_data_FIXED.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --family urologic \
    --chunk_size $CHUNK_SIZE \
    2>&1 | tee "$PROJECT_ROOT/logs/urologic_fixed_generation.log"

echo ""
echo "✅ Urologic terminé"
echo ""

# Famille 3/4: Respiratory (~2 min, 408 samples)
echo "========================================================================"
echo "[3/4] RESPIRATORY - Lung, Liver"
echo "========================================================================"
python "$PROJECT_ROOT/scripts/preprocessing/prepare_family_data_FIXED.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --family respiratory \
    --chunk_size $CHUNK_SIZE \
    2>&1 | tee "$PROJECT_ROOT/logs/respiratory_fixed_generation.log"

echo ""
echo "✅ Respiratory terminé"
echo ""

# Famille 4/4: Epidermal (~2 min, 571 samples)
echo "========================================================================"
echo "[4/4] EPIDERMAL - Skin, HeadNeck"
echo "========================================================================"
python "$PROJECT_ROOT/scripts/preprocessing/prepare_family_data_FIXED.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --family epidermal \
    --chunk_size $CHUNK_SIZE \
    2>&1 | tee "$PROJECT_ROOT/logs/epidermal_fixed_generation.log"

echo ""
echo "✅ Epidermal terminé"
echo ""

# Résumé final
echo "========================================================================"
echo "GÉNÉRATION COMPLÈTE ✅"
echo "========================================================================"
echo ""
echo "Fichiers générés dans: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.npz 2>/dev/null || echo "  (aucun fichier .npz trouvé)"
echo ""
echo "Logs sauvegardés dans: $PROJECT_ROOT/logs/"
echo ""
echo "🎯 PROCHAINE ÉTAPE:"
echo "   bash scripts/training/train_all_families_FIXED.sh"
echo ""
