#!/bin/bash

################################################################################
# Script: regenerate_all_families_v8.sh
# Description: Régénère les 5 familles avec prepare_family_data_FIXED_v8.py
#
# Résultats attendus:
# - Distance moyenne <2px pour toutes les familles ✅
# - Precision/Recall 100% ✅
# - inst_maps préservés dans NPZ ✅
#
# Temps estimé: ~5 minutes (5 familles × 1min chacune)
################################################################################

set -e  # Arrêter si erreur

PANNUKE_DIR="/home/amar/data/PanNuke"
OUTPUT_DIR="data/family_FIXED"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "RÉGÉNÉRATION 5 FAMILLES AVEC v8 (inst_maps preservés)"
echo "================================================================================"
echo "PanNuke dir: $PANNUKE_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# Liste des familles (ordre: du plus au moins de samples pour feedback rapide)
FAMILIES=("glandular" "digestive" "urologic" "epidermal" "respiratory")

for family in "${FAMILIES[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] Régénération famille: $family"
    echo "--------------------------------------------------------------------------------"

    python "$SCRIPT_DIR/prepare_family_data_FIXED_v8.py" \
        --family "$family" \
        --pannuke_dir "$PANNUKE_DIR" \
        --output_dir "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✅ $family terminé"
    else
        echo "❌ ERREUR avec $family - Arrêt du script"
        exit 1
    fi
    echo ""
done

echo "================================================================================"
echo "✅ RÉGÉNÉRATION COMPLÈTE - 5/5 familles"
echo "================================================================================"
echo ""
echo "Fichiers générés:"
for family in "${FAMILIES[@]}"; do
    npz_file="$OUTPUT_DIR/${family}_data_FIXED.npz"
    if [ -f "$npz_file" ]; then
        size=$(du -h "$npz_file" | cut -f1)
        echo "  ✅ $npz_file ($size)"
    else
        echo "  ❌ $npz_file (MANQUANT)"
    fi
done

echo ""
echo "================================================================================
PROCHAINE ÉTAPE: Test AJI final
================================================================================"
echo ""
echo "Commandes:"
echo "  # 1. Tester alignement des 5 familles (optionnel)"
echo "  for family in glandular digestive urologic epidermal respiratory; do"
echo "      python scripts/validation/verify_alignment_from_npz.py --family \$family --n_samples 5"
echo "  done"
echo ""
echo "  # 2. Re-entraîner modèle (si nécessaire)"
echo "  for family in glandular digestive urologic epidermal respiratory; do"
echo "      python scripts/training/train_hovernet_family.py --family \$family --epochs 50 --augment"
echo "  done"
echo ""
echo "  # 3. Test AJI final (objectif: 0.06 → 0.60+)"
echo "  python scripts/evaluation/evaluate_ground_truth.py --family epidermal --n_samples 50"
echo ""
