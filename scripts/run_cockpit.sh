#!/bin/bash
# ==============================================================================
# CellViT-Optimus R&D Cockpit — Script de lancement
# ==============================================================================
#
# Usage:
#   ./scripts/run_cockpit.sh              # Lancement standard
#   ./scripts/run_cockpit.sh --preload    # Précharge le moteur
#   ./scripts/run_cockpit.sh --share      # Lien public Gradio
#
# ==============================================================================

set -e

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        CellViT-Optimus — R&D Cockpit                          ║${NC}"
echo -e "${BLUE}║        Interface Gradio pour exploration IA                    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo

# Vérifier conda
if ! command -v conda &> /dev/null; then
    echo "❌ conda non trouvé. Activer l'environnement cellvit manuellement."
    exit 1
fi

# Activer l'environnement
echo -e "${GREEN}[1/3] Activation de l'environnement cellvit...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cellvit

# Vérifier gradio
echo -e "${GREEN}[2/3] Vérification des dépendances...${NC}"
python -c "import gradio" 2>/dev/null || {
    echo "Installation de gradio..."
    pip install gradio>=4.0.0
}

# Lancer l'interface
echo -e "${GREEN}[3/3] Lancement de l'interface Gradio...${NC}"
echo
echo "═══════════════════════════════════════════════════════════════"
echo "  L'interface sera disponible sur: http://localhost:7860"
echo "  Ctrl+C pour arrêter"
echo "═══════════════════════════════════════════════════════════════"
echo

cd "$(dirname "$0")/.."
python -m src.ui.app "$@"
