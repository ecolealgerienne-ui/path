#!/bin/bash
# Préparation complète pour training sur bases propres
#
# Ce script:
# 1. Audit complet du projet (scripts + données)
# 2. Nettoyage données redondantes
# 3. Validation que tout est cohérent
# 4. Prêt pour lancer training
#
# Usage:
#   bash scripts/utils/prepare_clean_training.sh --dry-run  # Voir ce qui sera fait
#   bash scripts/utils/prepare_clean_training.sh            # Exécuter

set -e  # Exit on error

DRY_RUN=false

# Parse arguments
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "=============================================================================="
echo "PRÉPARATION COMPLÈTE POUR TRAINING SUR BASES PROPRES"
echo "=============================================================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "🔍 MODE DRY-RUN: Aucune modification ne sera faite"
    echo ""
fi

# ============================================================================
# ÉTAPE 1: AUDIT COMPLET
# ============================================================================

echo "ÉTAPE 1/5: AUDIT COMPLET DU PROJET"
echo "------------------------------------------------------------------------------"
echo ""

python scripts/utils/audit_project_paths.py

if [ $? -ne 0 ]; then
    echo "❌ Erreur durant l'audit"
    exit 1
fi

echo ""
read -p "Continuer avec le nettoyage? (oui/non): " response
if [[ ! "$response" =~ ^(oui|yes|y)$ ]]; then
    echo "Annulé."
    exit 0
fi
echo ""

# ============================================================================
# ÉTAPE 2: MIGRATION SCRIPTS (SI NÉCESSAIRE)
# ============================================================================

echo "ÉTAPE 2/5: MIGRATION SCRIPTS VERS CONSTANTES"
echo "------------------------------------------------------------------------------"
echo ""

if [ "$DRY_RUN" = true ]; then
    python scripts/utils/migrate_to_centralized_paths.py --dry-run
else
    python scripts/utils/migrate_to_centralized_paths.py
fi

if [ $? -ne 0 ]; then
    echo "⚠️  Erreur durant la migration (non-fatal)"
fi

echo ""

# ============================================================================
# ÉTAPE 3: NETTOYAGE DONNÉES
# ============================================================================

echo "ÉTAPE 3/5: NETTOYAGE DONNÉES REDONDANTES"
echo "------------------------------------------------------------------------------"
echo ""

if [ "$DRY_RUN" = true ]; then
    python scripts/utils/cleanup_project_data.py --dry-run
else
    python scripts/utils/cleanup_project_data.py --force
fi

if [ $? -ne 0 ]; then
    echo "❌ Erreur durant le nettoyage"
    exit 1
fi

echo ""

# ============================================================================
# ÉTAPE 4: VALIDATION DONNÉES ESSENTIELLES
# ============================================================================

echo "ÉTAPE 4/5: VALIDATION DONNÉES ESSENTIELLES"
echo "------------------------------------------------------------------------------"
echo ""

# Vérifier que data/family_data existe et contient des fichiers
if [ ! -d "data/family_data" ]; then
    echo "❌ ERREUR CRITIQUE: data/family_data n'existe pas!"
    echo "   Lancez: python scripts/preprocessing/extract_features_from_fixed.py --family <FAMILY>"
    exit 1
fi

# Compter fichiers features
FEATURES_COUNT=$(find data/family_data -name "*_features.npz" 2>/dev/null | wc -l)
TARGETS_COUNT=$(find data/family_data -name "*_targets.npz" 2>/dev/null | wc -l)

echo "Fichiers trouvés dans data/family_data:"
echo "  Features: $FEATURES_COUNT"
echo "  Targets:  $TARGETS_COUNT"
echo ""

if [ $FEATURES_COUNT -eq 0 ] || [ $TARGETS_COUNT -eq 0 ]; then
    echo "❌ ERREUR: Données manquantes dans data/family_data"
    echo "   Lancez: python scripts/preprocessing/extract_features_from_fixed.py --family <FAMILY>"
    exit 1
fi

echo "✅ Données essentielles présentes"
echo ""

# ============================================================================
# ÉTAPE 5: DIAGNOSTIC PATH MISMATCH
# ============================================================================

echo "ÉTAPE 5/5: DIAGNOSTIC PATH MISMATCH (FAMILLE EPIDERMAL)"
echo "------------------------------------------------------------------------------"
echo ""

python scripts/validation/diagnose_training_data_mismatch.py --family epidermal

if [ $? -ne 0 ]; then
    echo "⚠️  Diagnostic path mismatch échoué (non-fatal)"
fi

echo ""

# ============================================================================
# RÉSUMÉ ET PROCHAINES ÉTAPES
# ============================================================================

echo "=============================================================================="
echo "RÉSUMÉ"
echo "=============================================================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "🔍 MODE DRY-RUN COMPLÉTÉ"
    echo ""
    echo "Pour exécuter réellement:"
    echo "  bash scripts/utils/prepare_clean_training.sh"
    echo ""
else
    echo "✅ PRÉPARATION COMPLÈTE - PROJET PROPRE"
    echo ""
    echo "Prochaines étapes recommandées:"
    echo ""
    echo "1. Re-train epidermal (40 min):"
    echo "   python scripts/training/train_hovernet_family.py \\"
    echo "       --family epidermal \\"
    echo "       --epochs 50 \\"
    echo "       --augment"
    echo ""
    echo "2. Test sur training data (1 min):"
    echo "   python scripts/validation/test_on_training_data.py \\"
    echo "       --family epidermal \\"
    echo "       --checkpoint models/checkpoints/hovernet_epidermal_best.pth \\"
    echo "       --n_samples 10"
    echo ""
    echo "   Attendu: NP Dice ~0.95 (au lieu de 0.0000)"
    echo ""
    echo "3. Test AJI (5 min):"
    echo "   python scripts/evaluation/test_aji_v8.py \\"
    echo "       --family epidermal \\"
    echo "       --checkpoint models/checkpoints/hovernet_epidermal_best.pth \\"
    echo "       --n_samples 50"
    echo ""
    echo "   Attendu: AJI >0.60"
    echo ""
fi

exit 0
