#!/bin/bash
# Script de diagnostic: Localiser les données sources pour V13-Hybrid
#
# Usage: bash scripts/utils/diagnose_data_location.sh
#
# Date: 2025-12-26

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         DIAGNOSTIC DONNÉES SOURCES V13-HYBRID                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="/home/amar/projects/cellvit-optimus"
cd "$PROJECT_ROOT" || exit 1

# ============================================================================
# FONCTION HELPER
# ============================================================================

check_file() {
    local path="$1"
    local label="$2"

    if [[ -e "$path" ]]; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        mod_date=$(stat -c %y "$path" 2>/dev/null | cut -d'.' -f1)
        echo "  ✅ $label"
        echo "     Path: $path"
        echo "     Taille: $size"
        echo "     Modifié: $mod_date"
        echo ""
        return 0
    else
        echo "  ❌ $label"
        echo "     Path: $path"
        echo "     Statut: MANQUANT"
        echo ""
        return 1
    fi
}

check_npz_content() {
    local path="$1"

    if [[ ! -e "$path" ]]; then
        return 1
    fi

    echo "  📦 Contenu du fichier .npz:"

    python3 - "$path" <<'EOF'
import sys
import numpy as np

try:
    data = np.load(sys.argv[1])
    print(f"     Clés: {list(data.keys())}")
    print("")
    for key in data.keys():
        arr = data[key]
        if hasattr(arr, 'shape'):
            print(f"     {key}:")
            print(f"       Shape: {arr.shape}")
            print(f"       Dtype: {arr.dtype}")
            if 'hv' in key.lower():
                print(f"       Range: [{arr.min():.4f}, {arr.max():.4f}]")
        else:
            print(f"     {key}: {arr}")
        print("")
except Exception as e:
    print(f"     ❌ Erreur lecture: {e}")
EOF

    echo ""
}

# ============================================================================
# 1. VÉRIFIER DONNÉES FAMILLE FIXED
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  DONNÉES FAMILLE FIXED (source attendue)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

FAMILIES=("glandular" "digestive" "urologic" "epidermal" "respiratory")
FIXED_DIR="data/family_FIXED"

if [[ -d "$FIXED_DIR" ]]; then
    echo "📂 Répertoire: $FIXED_DIR ($(du -sh "$FIXED_DIR" 2>/dev/null | cut -f1))"
    echo ""

    for family in "${FAMILIES[@]}"; do
        file_path="$FIXED_DIR/${family}_data_FIXED.npz"

        if check_file "$file_path" "$family (FIXED)"; then
            check_npz_content "$file_path"
        fi
    done
else
    echo "  ❌ Répertoire $FIXED_DIR MANQUANT !"
    echo ""
fi

# ============================================================================
# 2. VÉRIFIER SYMLINK family_data
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  SYMLINK family_data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ -L "data/family_data" ]]; then
    target=$(readlink -f "data/family_data")
    echo "  ✅ Symlink existe"
    echo "     Source: data/family_data"
    echo "     Cible: $target"
    echo ""

    if [[ -d "$target" ]]; then
        echo "  ✅ Cible existe ($(du -sh "$target" 2>/dev/null | cut -f1))"
    else
        echo "  ❌ Cible MANQUANTE !"
    fi
elif [[ -d "data/family_data" ]]; then
    echo "  ℹ️  Répertoire (pas symlink): data/family_data"
    echo "     Taille: $(du -sh data/family_data 2>/dev/null | cut -f1)"
else
    echo "  ❌ Symlink/répertoire MANQUANT"
fi

echo ""

# ============================================================================
# 3. VÉRIFIER DONNÉES PANNUKE BRUTES
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  DONNÉES PANNUKE BRUTES (fallback)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PANNUKE_DIR="/home/amar/data/PanNuke"

if [[ -d "$PANNUKE_DIR" ]]; then
    echo "  ✅ Répertoire PanNuke: $PANNUKE_DIR"
    echo "     Taille: $(du -sh "$PANNUKE_DIR" 2>/dev/null | cut -f1)"
    echo ""

    for fold in fold0 fold1 fold2; do
        images_path="$PANNUKE_DIR/$fold/images.npy"
        masks_path="$PANNUKE_DIR/$fold/masks.npy"
        types_path="$PANNUKE_DIR/$fold/types.npy"

        if [[ -e "$images_path" ]] && [[ -e "$masks_path" ]] && [[ -e "$types_path" ]]; then
            echo "  ✅ $fold complet"
        else
            echo "  ❌ $fold incomplet"
        fi
    done
else
    echo "  ❌ Répertoire PanNuke MANQUANT: $PANNUKE_DIR"
fi

echo ""

# ============================================================================
# 4. VÉRIFIER FEATURES H-OPTIMUS-0
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  FEATURES H-OPTIMUS-0 (pour training)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

FEATURES_DIR="data/cache/pannuke_features"

if [[ -d "$FEATURES_DIR" ]]; then
    echo "  📂 Répertoire: $FEATURES_DIR ($(du -sh "$FEATURES_DIR" 2>/dev/null | cut -f1))"
    echo ""

    for fold in 0 1 2; do
        file_path="$FEATURES_DIR/fold${fold}_features.npz"

        if check_file "$file_path" "Fold $fold features"; then
            # Vérifier date (post-fix = après 2025-12-23)
            mod_date=$(stat -c %Y "$file_path" 2>/dev/null)
            cutoff_date=$(date -d "2025-12-23" +%s 2>/dev/null || echo 0)

            if [[ $mod_date -gt $cutoff_date ]]; then
                echo "  ✅ Features PROPRES (post-fix preprocessing 2025-12-23)"
            else
                echo "  ⚠️  Features POTENTIELLEMENT CORROMPUES (avant fix 2025-12-23)"
                echo "     → Recommandé: Ré-extraire avec extract_features.py"
            fi
            echo ""
        fi
    done
else
    echo "  ❌ Répertoire features MANQUANT: $FEATURES_DIR"
    echo ""
fi

# ============================================================================
# 5. RECOMMANDATIONS
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      RECOMMANDATIONS                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Déterminer quelle action prendre
if [[ ! -d "$FIXED_DIR" ]] && [[ ! -d "data/family_data" ]]; then
    echo "🔴 PROBLÈME: Aucune donnée famille trouvée !"
    echo ""
    echo "Solution 1 (RECOMMANDÉ): Générer données FIXED depuis PanNuke"
    echo ""
    echo "  # Générer données famille avec HV float32"
    echo "  for family in glandular digestive urologic epidermal respiratory; do"
    echo "      python scripts/preprocessing/prepare_family_data_FIXED.py --family \$family"
    echo "  done"
    echo ""
    echo "Solution 2: Vérifier backup/archives existantes"
    echo ""

elif [[ -d "$FIXED_DIR" ]]; then
    # Compter fichiers présents
    count=$(ls -1 "$FIXED_DIR"/*_data_FIXED.npz 2>/dev/null | wc -l)

    if [[ $count -eq 5 ]]; then
        echo "🟢 DONNÉES COMPLÈTES: Les 5 familles sont présentes dans $FIXED_DIR"
        echo ""
        echo "Prochaine étape:"
        echo "  python scripts/preprocessing/prepare_v13_hybrid_dataset.py --family epidermal"
        echo ""
    else
        echo "🟡 DONNÉES PARTIELLES: $count/5 familles dans $FIXED_DIR"
        echo ""
        echo "Familles manquantes à générer:"
        for family in "${FAMILIES[@]}"; do
            if [[ ! -e "$FIXED_DIR/${family}_data_FIXED.npz" ]]; then
                echo "  - $family"
            fi
        done
        echo ""
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Diagnostic terminé."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
