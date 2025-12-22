#!/bin/bash
#
# Pipeline complet de validation par famille
#
# Ce script orchestre les 3 étapes de validation:
# 1. Préparation des échantillons de test par famille
# 2. Test isolé de chaque modèle de famille
# 3. Test du routage OrganHead → Famille
#
# Usage:
#   bash scripts/evaluation/run_family_validation_pipeline.sh \
#       /path/to/PanNuke \
#       models/checkpoints
#

set -e  # Exit on error

# Arguments
PANNUKE_DIR=${1:-"/home/amar/data/PanNuke"}
CHECKPOINT_DIR=${2:-"models/checkpoints"}
OUTPUT_BASE=${3:-"results/family_validation_$(date +%Y%m%d_%H%M%S)"}

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║           PIPELINE DE VALIDATION PAR FAMILLE                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "PanNuke:     $PANNUKE_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Output:      $OUTPUT_BASE"
echo ""

# Vérifier que PanNuke existe
if [ ! -d "$PANNUKE_DIR/fold2" ]; then
    echo "❌ ERREUR: PanNuke Fold 2 introuvable dans $PANNUKE_DIR"
    exit 1
fi

# Vérifier que les checkpoints existent
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ ERREUR: Répertoire checkpoints introuvable: $CHECKPOINT_DIR"
    exit 1
fi

# Créer répertoire de sortie
mkdir -p "$OUTPUT_BASE"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║ ÉTAPE 1/3: Préparation des échantillons de test                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

python scripts/evaluation/prepare_test_samples_by_family.py \
    --pannuke_dir "$PANNUKE_DIR" \
    --fold 2 \
    --samples_per_organ 10 \
    --output_dir "$OUTPUT_BASE/test_samples"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERREUR: Échec préparation des échantillons"
    exit 1
fi

echo ""
echo "✅ Échantillons préparés: $OUTPUT_BASE/test_samples"
echo ""

# Vérifier qu'on a des échantillons
n_families=$(find "$OUTPUT_BASE/test_samples" -name "test_samples.npz" | wc -l)
if [ $n_families -eq 0 ]; then
    echo "❌ ERREUR: Aucun échantillon extrait"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║ ÉTAPE 2/3: Test isolé de chaque modèle de famille                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

python scripts/evaluation/test_family_models_isolated.py \
    --test_samples_dir "$OUTPUT_BASE/test_samples" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_BASE/isolated_tests"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERREUR: Échec test isolé des modèles"
    exit 1
fi

echo ""
echo "✅ Tests isolés complétés: $OUTPUT_BASE/isolated_tests"
echo ""

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║ ÉTAPE 3/3: Test du routage OrganHead → Famille                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

python scripts/evaluation/test_organ_routing.py \
    --test_samples_dir "$OUTPUT_BASE/test_samples" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_BASE/routing_tests"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERREUR: Échec test de routage"
    exit 1
fi

echo ""
echo "✅ Tests de routage complétés: $OUTPUT_BASE/routing_tests"
echo ""

# Résumé final
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║ PIPELINE COMPLÉTÉ AVEC SUCCÈS                                            ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Résultats disponibles dans: $OUTPUT_BASE"
echo ""
echo "Fichiers générés:"
echo "  📁 $OUTPUT_BASE/test_samples/        - Échantillons de test par famille"
echo "  📁 $OUTPUT_BASE/isolated_tests/      - Résultats tests isolés"
echo "  📁 $OUTPUT_BASE/routing_tests/       - Résultats tests de routage"
echo ""
echo "Fichiers clés:"
echo "  📄 $OUTPUT_BASE/test_samples/global_report.json"
echo "  📄 $OUTPUT_BASE/isolated_tests/global_report.json"
echo "  📄 $OUTPUT_BASE/routing_tests/routing_results.json"
echo ""
echo "═════════════════════════════════════════════════════════════════════════════"
echo "PROCHAINES ÉTAPES"
echo "═════════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Consulter les rapports JSON pour identifier les problèmes"
echo ""
echo "2. Si tests isolés OK mais ground truth KO:"
echo "   → Problème d'instance mismatch (connectedComponents vs vraies instances)"
echo "   → Solution: Ré-entraîner avec vraies instances PanNuke"
echo ""
echo "3. Si tests isolés KO:"
echo "   → Problème d'entraînement du modèle de famille"
echo "   → Solution: Ré-entraîner avec plus de données ou augmentation"
echo ""
echo "4. Si routage KO:"
echo "   → Problème OrganHead ou ORGAN_TO_FAMILY mapping"
echo "   → Solution: Ré-calibrer OrganHead ou corriger mapping"
echo ""
