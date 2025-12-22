#!/bin/bash
#
# Ré-génère TOUTES les données des 5 familles avec HV targets corrects (float32 [-1, 1])
#
# BUG CORRIGÉ:
# - AVANT: HV targets en int8 [-127, 127] → MSE = 4681
# - APRÈS: HV targets en float32 [-1, 1] → MSE ~0.01
#

set -e

PANNUKE_DIR="${1:-/home/amar/data/PanNuke}"
OUTPUT_DIR="${2:-data/cache/family_data_FIXED}"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║ RÉ-GÉNÉRATION DES DONNÉES DE FAMILLE (VERSION FIXED)                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "PanNuke:    $PANNUKE_DIR"
echo "Output:     $OUTPUT_DIR"
echo ""

if [ ! -d "$PANNUKE_DIR" ]; then
    echo "❌ ERREUR: PanNuke introuvable dans $PANNUKE_DIR"
    exit 1
fi

# Créer répertoire de sortie
mkdir -p "$OUTPUT_DIR"

# Sauvegarder les anciennes données (au cas où)
if [ -d "data/cache/family_data" ]; then
    echo "💾 Sauvegarde des anciennes données..."
    mv data/cache/family_data data/cache/family_data_OLD_int8_$(date +%Y%m%d_%H%M%S)
    echo "   → Sauvegardées dans data/cache/family_data_OLD_int8_*"
    echo ""
fi

# Générer pour chaque famille
families=("glandular" "digestive" "urologic" "respiratory" "epidermal")

for family in "${families[@]}"; do
    echo "========================================================================"
    echo "FAMILLE: $family"
    echo "========================================================================"
    echo ""

    python scripts/preprocessing/prepare_family_data_FIXED.py \
        --data_dir "$PANNUKE_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --family "$family" \
        --folds 0 1 2

    echo ""
    echo "✅ $family complété"
    echo ""
done

# Créer symlink vers le nouveau répertoire
echo "🔗 Création du symlink data/cache/family_data → family_data_FIXED"
ln -sf family_data_FIXED data/cache/family_data

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║ GÉNÉRATION COMPLÉTÉE                                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Anciennes données (int8): data/cache/family_data_OLD_int8_*"
echo "Nouvelles données (float32): $OUTPUT_DIR"
echo "Symlink: data/cache/family_data → family_data_FIXED"
echo ""
echo "PROCHAINES ÉTAPES:"
echo ""
echo "1. Vérifier les nouvelles données:"
echo "   python scripts/evaluation/diagnose_targets.py --family glandular"
echo "   → Doit afficher: HV dtype=float32, range=[-1, 1]"
echo ""
echo "2. Re-tester sur données d'entraînement:"
echo "   python scripts/evaluation/test_on_training_data.py \\"
echo "     --family glandular \\"
echo "     --checkpoint models/checkpoints/hovernet_glandular_best.pth \\"
echo "     --n_samples 100"
echo "   → Doit afficher: NP Dice ~0.96, HV MSE ~0.01"
echo ""
echo "3. Si tests OK, ré-entraîner les 5 familles (~10h):"
echo "   for family in glandular digestive urologic respiratory epidermal; do"
echo "       python scripts/training/train_hovernet_family.py \\"
echo "         --family \$family \\"
echo "         --epochs 50 \\"
echo "         --augment"
echo "   done"
echo ""
