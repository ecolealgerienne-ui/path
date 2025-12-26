#!/usr/bin/env python3
"""
Script d'entraînement simplifié utilisant la configuration de TEST centralisée.

⚠️ PHASE DE TEST: fold0 uniquement, 20 epochs
   Après validation, passer à PROD_CONFIG (tous folds, 50 epochs)

Usage:
    python scripts/training/train_test_config.py

Configuration (définie dans src/constants.py):
    TEST_CONFIG = {
        "folds": [0],
        "epochs": 20,
        "batch_size": 8,
        "family": "epidermal",
    }
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import TEST_CONFIG, CURRENT_DATA_VERSION, get_family_data_path


def main():
    print("=" * 80)
    print("🧪 ENTRAÎNEMENT EN MODE TEST")
    print("=" * 80)
    print(f"""
Configuration de test (src/constants.py):
  - Famille:  {TEST_CONFIG['family']}
  - Folds:    {TEST_CONFIG['folds']}
  - Epochs:   {TEST_CONFIG['epochs']}
  - Batch:    {TEST_CONFIG['batch_size']}
  - Version:  {CURRENT_DATA_VERSION}

Données source: {get_family_data_path(TEST_CONFIG['family'])}
""")

    # Étape 1: Vérifier que les données existent
    data_path = Path(get_family_data_path(TEST_CONFIG['family']))
    if not data_path.exists():
        print(f"❌ ERREUR: Fichier de données non trouvé: {data_path}")
        print(f"\n   Générer d'abord les données v12:")
        print(f"   python scripts/preprocessing/prepare_family_data_FIXED_v12_COHERENT.py --family {TEST_CONFIG['family']}")
        sys.exit(1)

    print(f"✅ Données trouvées: {data_path}")

    # Étape 2: Vérifier que les features existent
    from src.constants import get_family_features_path, get_family_targets_path
    features_path = Path(get_family_features_path(TEST_CONFIG['family']))
    targets_path = Path(get_family_targets_path(TEST_CONFIG['family']))

    if not features_path.exists() or not targets_path.exists():
        print(f"\n⚠️ Features/Targets non trouvés. Extraction nécessaire...")
        print(f"   Features: {features_path}")
        print(f"   Targets:  {targets_path}")
        print(f"\n   Commande à exécuter:")
        print(f"   python scripts/preprocessing/extract_features_from_v9.py --family {TEST_CONFIG['family']}")
        sys.exit(1)

    print(f"✅ Features trouvées: {features_path}")
    print(f"✅ Targets trouvés:  {targets_path}")

    # Étape 3: Lancer l'entraînement
    print("\n" + "=" * 80)
    print("🚀 LANCEMENT ENTRAÎNEMENT")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/training/train_hovernet_family.py"),
        "--family", TEST_CONFIG['family'],
        "--epochs", str(TEST_CONFIG['epochs']),
        "--batch_size", str(TEST_CONFIG['batch_size']),
        "--augment",
        "--lambda_hv", "2.0",
    ]

    print(f"Commande: {' '.join(cmd)}")
    print()

    # Exécuter
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ ENTRAÎNEMENT TERMINÉ")
        print("=" * 80)
        print(f"""
Prochaines étapes:

1. Vérifier le modèle sur ses données d'entraînement:
   python scripts/evaluation/verify_model_on_training_data.py \\
       --checkpoint models/checkpoints/hovernet_{TEST_CONFIG['family']}_best.pth

2. Tester l'AJI:
   python scripts/evaluation/test_epidermal_aji_FINAL.py \\
       --checkpoint models/checkpoints/hovernet_{TEST_CONFIG['family']}_best.pth \\
       --n_samples 50

3. Si validation OK, passer à PROD_CONFIG (éditer src/constants.py)
""")
    else:
        print(f"\n❌ ERREUR: L'entraînement a échoué (code {result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    main()
