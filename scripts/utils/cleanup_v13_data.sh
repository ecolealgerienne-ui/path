#!/bin/bash
# Script de nettoyage des anciennes données V13
#
# Usage: bash scripts/utils/cleanup_v13_data.sh [--dry-run]
#
# Date: 2025-12-26

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 MODE DRY-RUN (aucune suppression réelle)"
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         NETTOYAGE DONNÉES V13 - CellViT-Optimus               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="/home/amar/projects/cellvit-optimus"
cd "$PROJECT_ROOT" || exit 1

# ============================================================================
# 1. IDENTIFIER LES DONNÉES OBSOLÈTES
# ============================================================================

echo "📂 Recherche des données V13 obsolètes..."
echo ""

# Arrays pour stocker les chemins
declare -a TO_DELETE
declare -a TO_KEEP

# Fonction pour ajouter à la liste de suppression
add_to_delete() {
    local path="$1"
    local reason="$2"
    if [[ -e "$path" ]]; then
        TO_DELETE+=("$path|$reason")
    fi
}

# Fonction pour afficher la taille
get_size() {
    if [[ -e "$1" ]]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

# ============================================================================
# CATÉGORIES DE DONNÉES À SUPPRIMER
# ============================================================================

# 1. Données int8 corrompues (Bug #3)
echo "🔍 Recherche données int8 corrompues (Bug #3)..."
add_to_delete "data/family_data_OLD_int8_20251222_163212" "Bug #3 - HV int8 au lieu de float32"
add_to_delete "data/cache/family_data_OLD_int8_*" "Bug #3 - HV int8 au lieu de float32"

# 2. Features corrompues (Bugs #1 et #2)
echo "🔍 Recherche features corrompues (Bugs #1 #2)..."
add_to_delete "data/cache/pannuke_features_OLD_CORRUPTED_20251223" "Bugs #1 #2 - ToPILImage float64 + LayerNorm mismatch"
add_to_delete "data/cache/pannuke_features/fold*_features.npz" "Features avec preprocessing corrompu (avant 2025-12-22)"

# 3. Checkpoints V13 POC (remplacés par V13-Hybrid)
echo "🔍 Recherche checkpoints V13 POC obsolètes..."
add_to_delete "models/checkpoints/hovernet_epidermal_v13_poc_*.pth" "V13 POC - remplacé par V13-Hybrid"
add_to_delete "models/checkpoints/hovernet_*_v13_multi_crop_*.pth" "V13 Multi-Crop POC - remplacé par V13-Hybrid"

# 4. Données temporaires V13 Multi-Crop (si existent)
echo "🔍 Recherche données temporaires V13 Multi-Crop..."
add_to_delete "data/family_data_v13_multi_crop" "V13 Multi-Crop - architecture changée vers Hybrid"
add_to_delete "data/cache/family_data/*_v13_multi_crop_*" "V13 Multi-Crop temporaire"

# 5. Logs et snapshots anciens
echo "🔍 Recherche logs et snapshots anciens (>30 jours)..."
if [[ -d "data/snapshots" ]]; then
    find data/snapshots -type f -mtime +30 2>/dev/null | while read -r file; do
        add_to_delete "$file" "Snapshot ancien (>30 jours)"
    done
fi

# 6. Fichiers de diagnostic temporaires
add_to_delete "results/DIAGNOSTIC_*" "Rapports diagnostic temporaires"
add_to_delete "results/image_*_diagnosis.png" "Images diagnostic temporaires"

# ============================================================================
# 2. AFFICHER LE RÉCAPITULATIF
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    RÉCAPITULATIF NETTOYAGE                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL_SIZE=0
TOTAL_FILES=0

if [[ ${#TO_DELETE[@]} -eq 0 ]]; then
    echo "✅ Aucune donnée obsolète trouvée ! Projet déjà propre."
    exit 0
fi

echo "📋 Fichiers/dossiers à supprimer:"
echo ""

for entry in "${TO_DELETE[@]}"; do
    IFS='|' read -r path reason <<< "$entry"

    if [[ -e "$path" ]]; then
        size=$(get_size "$path")
        echo "  ❌ $path"
        echo "     Raison: $reason"
        echo "     Taille: $size"
        echo ""

        TOTAL_FILES=$((TOTAL_FILES + 1))
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total: $TOTAL_FILES éléments à supprimer"
echo ""

# ============================================================================
# 3. DONNÉES À CONSERVER (VÉRIFICATION)
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  DONNÉES À CONSERVER                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "✅ Ces données DOIVENT être conservées:"
echo ""

# Données FIXED (validées)
if [[ -d "data/family_FIXED" ]]; then
    echo "  🟢 data/family_FIXED/ ($(get_size data/family_FIXED))"
    echo "     → Données validées HV float32"
else
    echo "  ⚠️  data/family_FIXED/ MANQUANT ! (requis pour V13-Hybrid)"
fi

# Checkpoints production
if [[ -d "models/checkpoints" ]]; then
    ls models/checkpoints/hovernet_*_best.pth 2>/dev/null | while read -r ckpt; do
        if [[ ! "$ckpt" =~ v13_poc ]] && [[ ! "$ckpt" =~ v13_multi_crop ]]; then
            echo "  🟢 $ckpt ($(get_size "$ckpt"))"
        fi
    done
fi

# Features H-optimus-0 propres (si ré-extraites après 2025-12-23)
if [[ -d "data/cache/pannuke_features" ]]; then
    # Vérifier date modification
    mod_date=$(stat -c %Y data/cache/pannuke_features 2>/dev/null || echo 0)
    cutoff_date=$(date -d "2025-12-23" +%s)

    if [[ $mod_date -gt $cutoff_date ]]; then
        echo "  🟢 data/cache/pannuke_features/ ($(get_size data/cache/pannuke_features))"
        echo "     → Features propres (post-fix preprocessing)"
    fi
fi

# Données V13-Hybrid (si générées)
if [[ -d "data/family_data_v13_hybrid" ]]; then
    echo "  🟢 data/family_data_v13_hybrid/ ($(get_size data/family_data_v13_hybrid))"
    echo "     → Données V13-Hybrid (Macenko + H-channel)"
fi

echo ""

# ============================================================================
# 4. EXÉCUTION DU NETTOYAGE
# ============================================================================

if [[ "$DRY_RUN" == true ]]; then
    echo "🔍 DRY-RUN terminé. Aucune suppression effectuée."
    echo ""
    echo "Pour exécuter le nettoyage réel:"
    echo "  bash scripts/utils/cleanup_v13_data.sh"
    exit 0
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    CONFIRMATION SUPPRESSION                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  ATTENTION: Cette opération est IRRÉVERSIBLE !"
echo ""
echo "Voulez-vous supprimer les $TOTAL_FILES éléments listés ci-dessus ? (oui/non)"
read -r confirmation

if [[ "$confirmation" != "oui" ]]; then
    echo "❌ Nettoyage annulé."
    exit 0
fi

echo ""
echo "🗑️  Suppression en cours..."
echo ""

DELETED_COUNT=0

for entry in "${TO_DELETE[@]}"; do
    IFS='|' read -r path reason <<< "$entry"

    if [[ -e "$path" ]]; then
        echo "  Suppression: $path"
        rm -rf "$path"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    NETTOYAGE TERMINÉ                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ $DELETED_COUNT éléments supprimés"
echo ""
echo "🔄 Prochaines étapes:"
echo "  1. Vérifier data/family_FIXED/ existe (requis pour V13-Hybrid)"
echo "  2. Lancer Phase 1.1: python scripts/preprocessing/prepare_v13_hybrid_dataset.py --family epidermal"
echo ""
