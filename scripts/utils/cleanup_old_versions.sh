#!/bin/bash

################################################################################
# Script: cleanup_old_versions.sh
# Description: Purge TOUTES les versions v1-v7 pour éviter contamination
#
# CRITIQUE: Ne JAMAIS mélanger v8 (correct) avec v1-v7 (corrompus)
#
# Ce script supprime:
# - Anciens NPZ (v1-v7) dans data/family_FIXED/
# - Anciens checkpoints entraînés sur données corrompues
# - Anciens résultats d'évaluation obsolètes
#
# GARDE:
# - v8 (version gold)
# - Diagnostics/logs (pour analyse post-mortem)
################################################################################

set -e

echo "================================================================================"
echo "NETTOYAGE VERSIONS OBSOLÈTES (v1-v7)"
echo "================================================================================"
echo ""
echo "⚠️  ATTENTION: Ce script va SUPPRIMER définitivement:"
echo "   - Tous les NPZ v1-v7 (données corrompues)"
echo "   - Checkpoints entraînés avant 2025-12-24 14:40"
echo "   - Résultats d'évaluation obsolètes"
echo ""
read -p "Confirmer la suppression? (oui/non): " confirmation

if [ "$confirmation" != "oui" ]; then
    echo "❌ Annulé par l'utilisateur"
    exit 0
fi

echo ""
echo "================================================================================
ÉTAPE 1: Identification fichiers obsolètes
================================================================================"

# Timestamp de référence v8: 2025-12-24 14:40:38
V8_TIMESTAMP="2025-12-24 14:40:00"

echo ""
echo "1.1 - NPZ data files (data/family_FIXED/)"
echo "--------------------------------------------------------------------------------"

if [ -d "data/family_FIXED" ]; then
    # Lister tous les NPZ
    npz_files=$(find data/family_FIXED -name "*_data_FIXED.npz" -type f)

    if [ -n "$npz_files" ]; then
        echo "Fichiers NPZ trouvés:"
        for npz in $npz_files; do
            # Vérifier si le NPZ contient inst_maps (signature v8)
            has_inst_maps=$(python3 -c "
import numpy as np
import sys
try:
    data = np.load('$npz')
    print('v8' if 'inst_maps' in data else 'OLD')
except:
    print('ERROR')
" 2>/dev/null)

            timestamp=$(stat -c %y "$npz" | cut -d' ' -f1,2 | cut -d'.' -f1)
            size=$(du -h "$npz" | cut -f1)

            if [ "$has_inst_maps" == "v8" ]; then
                echo "  ✅ KEEP: $npz ($size, $timestamp) - Version v8"
            else
                echo "  🗑️  DELETE: $npz ($size, $timestamp) - Version OLD (pas inst_maps)"
            fi
        done
    else
        echo "  Aucun fichier NPZ trouvé"
    fi
else
    echo "  ⚠️  Répertoire data/family_FIXED n'existe pas"
fi

echo ""
echo "1.2 - Checkpoints models (models/checkpoints/)"
echo "--------------------------------------------------------------------------------"

if [ -d "models/checkpoints" ]; then
    old_checkpoints=$(find models/checkpoints -name "hovernet_*_best.pth" -type f ! -newermt "$V8_TIMESTAMP")

    if [ -n "$old_checkpoints" ]; then
        echo "Checkpoints obsolètes (avant v8):"
        for ckpt in $old_checkpoints; do
            timestamp=$(stat -c %y "$ckpt" | cut -d' ' -f1,2 | cut -d'.' -f1)
            size=$(du -h "$ckpt" | cut -f1)
            echo "  🗑️  DELETE: $ckpt ($size, $timestamp)"
        done
    else
        echo "  Aucun checkpoint obsolète trouvé"
    fi
else
    echo "  ⚠️  Répertoire models/checkpoints n'existe pas"
fi

echo ""
echo "1.3 - Résultats d'évaluation (results/)"
echo "--------------------------------------------------------------------------------"

if [ -d "results" ]; then
    old_results=$(find results -type d -name "alignment_*" ! -newermt "$V8_TIMESTAMP")

    if [ -n "$old_results" ]; then
        echo "Résultats obsolètes (avant v8):"
        for res_dir in $old_results; do
            timestamp=$(stat -c %y "$res_dir" | cut -d' ' -f1,2 | cut -d'.' -f1)
            size=$(du -sh "$res_dir" | cut -f1)
            echo "  🗑️  DELETE: $res_dir ($size, $timestamp)"
        done
    else
        echo "  Aucun résultat obsolète trouvé"
    fi
else
    echo "  ⚠️  Répertoire results n'existe pas"
fi

echo ""
echo "================================================================================"
echo "ÉTAPE 2: Suppression confirmée"
echo "================================================================================"
echo ""
read -p "Procéder à la suppression? (oui/non): " final_confirm

if [ "$final_confirm" != "oui" ]; then
    echo "❌ Annulé par l'utilisateur"
    exit 0
fi

# Compteurs
deleted_npz=0
deleted_ckpt=0
deleted_results=0
space_freed=0

echo ""
echo "2.1 - Suppression NPZ obsolètes"
echo "--------------------------------------------------------------------------------"

if [ -d "data/family_FIXED" ]; then
    for npz in $(find data/family_FIXED -name "*_data_FIXED.npz" -type f); do
        has_inst_maps=$(python3 -c "
import numpy as np
try:
    data = np.load('$npz')
    print('v8' if 'inst_maps' in data else 'OLD')
except:
    print('ERROR')
" 2>/dev/null)

        if [ "$has_inst_maps" != "v8" ]; then
            size_bytes=$(stat -c %s "$npz")
            space_freed=$((space_freed + size_bytes))
            rm -f "$npz"
            deleted_npz=$((deleted_npz + 1))
            echo "  ✅ Supprimé: $npz"
        fi
    done
fi

echo "  Total NPZ supprimés: $deleted_npz"

echo ""
echo "2.2 - Suppression checkpoints obsolètes"
echo "--------------------------------------------------------------------------------"

if [ -d "models/checkpoints" ]; then
    for ckpt in $(find models/checkpoints -name "hovernet_*_best.pth" -type f ! -newermt "$V8_TIMESTAMP"); do
        size_bytes=$(stat -c %s "$ckpt")
        space_freed=$((space_freed + size_bytes))
        rm -f "$ckpt"
        deleted_ckpt=$((deleted_ckpt + 1))
        echo "  ✅ Supprimé: $ckpt"
    done
fi

echo "  Total checkpoints supprimés: $deleted_ckpt"

echo ""
echo "2.3 - Suppression résultats obsolètes"
echo "--------------------------------------------------------------------------------"

if [ -d "results" ]; then
    for res_dir in $(find results -type d -name "alignment_*" ! -newermt "$V8_TIMESTAMP"); do
        size_bytes=$(du -sb "$res_dir" | cut -f1)
        space_freed=$((space_freed + size_bytes))
        rm -rf "$res_dir"
        deleted_results=$((deleted_results + 1))
        echo "  ✅ Supprimé: $res_dir"
    done
fi

echo "  Total résultats supprimés: $deleted_results"

echo ""
echo "================================================================================"
echo "BILAN NETTOYAGE"
echo "================================================================================"

space_freed_mb=$((space_freed / 1024 / 1024))
space_freed_gb=$(echo "scale=2; $space_freed / 1024 / 1024 / 1024" | bc)

echo ""
echo "Fichiers supprimés:"
echo "  - NPZ obsolètes:         $deleted_npz"
echo "  - Checkpoints obsolètes: $deleted_ckpt"
echo "  - Résultats obsolètes:   $deleted_results"
echo ""
echo "Espace disque libéré: ${space_freed_mb} MB (${space_freed_gb} GB)"
echo ""
echo "✅ NETTOYAGE TERMINÉ - Seuls les fichiers v8 sont conservés"
echo ""
