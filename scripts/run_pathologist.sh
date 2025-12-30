#!/bin/bash
# =============================================================================
# CellViT-Optimus — Lancement Interface Pathologiste
# =============================================================================
#
# Usage:
#   ./scripts/run_pathologist.sh
#   ./scripts/run_pathologist.sh --preload
#   ./scripts/run_pathologist.sh --share
#
# Port par défaut: 7861 (différent du R&D Cockpit: 7860)
# =============================================================================

set -e

# Activer l'environnement conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellvit

# Aller à la racine du projet
cd "$(dirname "$0")/.."

echo "=============================================="
echo "  CellViT-Optimus — Interface Pathologiste"
echo "=============================================="
echo ""
echo "  Port: 7861"
echo "  URL:  http://localhost:7861"
echo ""
echo "  Note: Interface clinique simplifiée"
echo "        Les indicateurs techniques sont masqués"
echo ""
echo "=============================================="

# Lancer l'application
python -m src.ui.app_pathologist "$@"
