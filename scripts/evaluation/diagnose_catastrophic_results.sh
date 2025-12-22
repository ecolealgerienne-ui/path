#!/bin/bash

# Script de diagnostic pour les résultats catastrophiques du pipeline de validation
# À exécuter sur la machine où se trouvent les checkpoints et les données

set -e

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║ DIAGNOSTIC: Performances Catastrophiques des Modèles HoVer-Net          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# PRIORITÉ 1: Vérifier Date des Checkpoints vs Commits de Fix
# =============================================================================

echo "========================================================================"
echo "PRIORITÉ 1: Date des Checkpoints"
echo "========================================================================"
echo ""

if [ ! -d "models/checkpoints" ]; then
    echo "❌ ERREUR: models/checkpoints/ introuvable"
    echo "Assurez-vous d'exécuter ce script depuis la racine du projet"
    exit 1
fi

echo "Dates de création/modification des checkpoints:"
echo ""

for family in glandular digestive urologic respiratory epidermal; do
    ckpt="models/checkpoints/hovernet_${family}_best.pth"
    if [ -f "$ckpt" ]; then
        echo "=== $family ==="
        stat "$ckpt" | grep -E "Birth|Modify|Change" || ls -lh "$ckpt"
        echo ""
    else
        echo "⚠️  MANQUANT: $ckpt"
        echo ""
    fi
done

echo ""
echo "Dates des commits de fix critiques:"
echo ""
git log --oneline --all --date=short --format="%ad %h %s" | grep -E "2025-12-20|2025-12-21" | head -20

echo ""
echo "========================================================================"
echo "ANALYSE:"
echo "========================================================================"
echo ""
echo "Si les checkpoints datent d'AVANT 2025-12-21:"
echo "  → Hypothèse #1 CONFIRMÉE (features corrompues)"
echo "  → Solution: Ré-extraire features + ré-entraîner (~12-15h)"
echo ""
echo "Si les checkpoints datent d'APRÈS 2025-12-21:"
echo "  → Chercher une autre cause (hypothèse #2 ou #3)"
echo ""

# =============================================================================
# PRIORITÉ 2: Vérifier CLS std dans Features d'Entraînement
# =============================================================================

echo ""
echo "========================================================================"
echo "PRIORITÉ 2: Vérification CLS std des Features"
echo "========================================================================"
echo ""

if [ ! -d "data/cache/pannuke_features" ]; then
    echo "❌ ERREUR: data/cache/pannuke_features/ introuvable"
    echo "Les features n'ont pas été extraites ou sont dans un autre répertoire"
    exit 1
fi

echo "Vérification des features d'entraînement..."
echo ""

python scripts/validation/verify_features.py --features_dir data/cache/pannuke_features

echo ""
echo "========================================================================"
echo "ANALYSE:"
echo "========================================================================"
echo ""
echo "Si CLS std ~0.77 pour tous les folds:"
echo "  → Features correctes (après fix)"
echo "  → Mais checkpoints peut-être entraînés avec anciennes features"
echo ""
echo "Si CLS std ~0.28 pour tous les folds:"
echo "  → Features corrompues (Bug #2 LayerNorm mismatch)"
echo "  → Doit ré-extraire avec fix"
echo ""

# =============================================================================
# PRIORITÉ 3: Comparer Préparation GT Train vs Eval
# =============================================================================

echo ""
echo "========================================================================"
echo "PRIORITÉ 3: Comparaison Préparation Ground Truth"
echo "========================================================================"
echo ""

echo "=== TRAIN (prepare_family_data.py) ==="
grep -A 15 "# Préparer ground truth" scripts/preprocessing/prepare_family_data.py 2>/dev/null || \
    grep -A 15 "np_mask = " scripts/preprocessing/prepare_family_data.py 2>/dev/null || \
    echo "Pattern non trouvé - inspect manuellement"

echo ""
echo "=== EVAL (test_family_models_isolated.py) ==="
grep -A 15 "# Préparer ground truth" scripts/evaluation/test_family_models_isolated.py 2>/dev/null || \
    grep -A 15 "np_gt = " scripts/evaluation/test_family_models_isolated.py 2>/dev/null || \
    echo "Pattern non trouvé - inspect manuellement"

echo ""
echo "========================================================================"
echo "ANALYSE:"
echo "========================================================================"
echo ""
echo "Vérifier visuellement si les deux méthodes de préparation du GT sont identiques."
echo "Différences possibles:"
echo "  - connectedComponents vs vraies instances PanNuke"
echo "  - Resize différent (224 vs 256)"
echo "  - Canaux utilisés (1-4 vs 1-5)"
echo ""

# =============================================================================
# RÉSUMÉ ET RECOMMANDATIONS
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║ RÉSUMÉ ET PROCHAINES ÉTAPES                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Basé sur les vérifications ci-dessus, déterminer l'hypothèse la plus probable:"
echo ""
echo "1. Checkpoints datent d'avant 2025-12-21 ET CLS std ~0.77"
echo "   → Ré-entraîner avec checkpoints actuels sur features FIXED (~10h)"
echo ""
echo "2. Checkpoints datent d'avant 2025-12-21 ET CLS std ~0.28"
echo "   → Ré-extraire features (~3h) + ré-entraîner (~10h) = 13h total"
echo ""
echo "3. Checkpoints datent d'après 2025-12-21 ET différence dans GT"
echo "   → Harmoniser préparation GT + ré-entraîner (~2-3h)"
echo ""
echo "4. Autre cas"
echo "   → Investigation plus approfondie requise"
echo ""

echo "Rapport complet disponible dans:"
echo "  results/family_validation_20251222_153551/DIAGNOSTIC_CRITICAL_ISSUE.md"
echo ""
