#!/bin/bash
# Script de lancement du démo CellViT-Optimus

echo "╔════════════════════════════════════════════╗"
echo "║     CellViT-Optimus — Démonstration        ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Aller dans le répertoire du projet
cd "$(dirname "$0")"

# Générer les données synthétiques si nécessaires
if [ ! -f "data/demo/images.npy" ]; then
    echo "📦 Génération des données de démonstration..."
    python scripts/demo/synthetic_cells.py
    echo ""
fi

# Lancer le démo Gradio
echo "🚀 Lancement de l'interface web..."
echo "   Ouvrir: http://localhost:7860"
echo ""
python scripts/demo/gradio_demo.py
