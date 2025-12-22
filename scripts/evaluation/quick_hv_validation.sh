#!/bin/bash
# Test rapide: 1 epoch avec checkpoint existant pour vérifier HV MSE
# Si le script montre HV MSE ~0.32 → Le problème était toujours là
# Si le script montre HV MSE ~0.0000 → Checkpoint différent ou erreur de calcul

set -e

echo "=========================================="
echo "TEST RAPIDE: 1 EPOCH VALIDATION HV MSE"
echo "=========================================="
echo ""

FAMILY="glandular"
CHECKPOINT="models/checkpoints/hovernet_${FAMILY}_best.pth"
DATA_DIR="data/cache/family_data_FIXED"

# Vérifier que le checkpoint existe
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERREUR: Checkpoint introuvable: $CHECKPOINT"
    exit 1
fi

# Vérifier que les données FIXED existent
if [ ! -f "$DATA_DIR/${FAMILY}_features.npz" ]; then
    echo "❌ ERREUR: Features FIXED introuvables: $DATA_DIR/${FAMILY}_features.npz"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo "Données: $DATA_DIR"
echo ""
echo "⏱️  Lancement 1 epoch (train + validation)..."
echo ""

# Lancer 1 epoch avec checkpoint existant
python scripts/training/train_hovernet_family.py \
    --family "$FAMILY" \
    --epochs 1 \
    --cache_dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --augment \
    --dropout 0.1

echo ""
echo "=========================================="
echo "INTERPRÉTATION DES RÉSULTATS"
echo "=========================================="
echo ""
echo "Si HV MSE affiché ~0.0000-0.0002:"
echo "  → Le checkpoint original était différent"
echo "  → Ou erreur dans la sauvegarde du checkpoint"
echo ""
echo "Si HV MSE affiché ~0.30-0.35:"
echo "  → Le problème était toujours là pendant l'entraînement"
echo "  → L'affichage tronquait ou erreur de calcul"
echo ""
