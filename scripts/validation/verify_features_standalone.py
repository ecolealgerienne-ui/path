#!/usr/bin/env python3
"""
Script de vérification CLS std des features - version standalone
Fonctionne sans dépendances sur src.*
"""

import numpy as np
import sys
from pathlib import Path

def verify_features_standalone(features_dir: str):
    """Vérifie CLS std dans les features extraites."""

    features_path = Path(features_dir)

    if not features_path.exists():
        print(f"❌ ERREUR: {features_dir} introuvable")
        return False

    print(f"Vérification des features dans: {features_path}")
    print("")

    # Chercher les fichiers .npz
    npz_files = sorted(features_path.glob("*.npz"))

    if not npz_files:
        print(f"❌ ERREUR: Aucun fichier .npz trouvé dans {features_dir}")
        return False

    print(f"Fichiers trouvés: {len(npz_files)}")
    print("")

    results = []

    for npz_file in npz_files:
        try:
            data = np.load(npz_file)

            # Chercher la clé 'features' ou 'layer_24'
            if 'features' in data:
                features = data['features']
            elif 'layer_24' in data:
                features = data['layer_24']
            else:
                print(f"⚠️  {npz_file.name}: Clé 'features' ou 'layer_24' introuvable")
                print(f"   Clés disponibles: {list(data.keys())}")
                continue

            # Extraire CLS token (premier token)
            cls_tokens = features[:, 0, :]  # (N, 1536)

            # Calculer std
            cls_std = cls_tokens.std()

            # Déterminer statut
            if 0.70 <= cls_std <= 0.90:
                status = "✅"
                verdict = "OK (features correctes)"
            elif cls_std < 0.40:
                status = "❌"
                verdict = "CORROMPU (Bug #2 LayerNorm mismatch)"
            else:
                status = "⚠️"
                verdict = "SUSPECT (vérifier preprocessing)"

            print(f"{status} {npz_file.name:30s} CLS std = {cls_std:.3f}  ({verdict})")

            results.append({
                'file': npz_file.name,
                'cls_std': cls_std,
                'status': status,
                'verdict': verdict
            })

        except Exception as e:
            print(f"❌ {npz_file.name}: Erreur lors du chargement - {e}")

    print("")
    print("=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print("")

    if not results:
        print("❌ Aucune feature valide trouvée")
        return False

    # Statistiques globales
    all_stds = [r['cls_std'] for r in results]
    mean_std = np.mean(all_stds)
    min_std = np.min(all_stds)
    max_std = np.max(all_stds)

    print(f"CLS std moyen  : {mean_std:.3f}")
    print(f"CLS std min    : {min_std:.3f}")
    print(f"CLS std max    : {max_std:.3f}")
    print("")

    # Verdict global
    if mean_std < 0.40:
        print("❌ VERDICT: Features CORROMPUES (Bug #2 LayerNorm mismatch)")
        print("   → Checkpoints entraînés avec forward_features() SANS LayerNorm final")
        print("   → CLS std ~0.28 au lieu de ~0.77")
        print("")
        print("   SOLUTION: Ré-extraire features avec fix LayerNorm + ré-entraîner")
        return False
    elif 0.70 <= mean_std <= 0.90:
        print("✅ VERDICT: Features CORRECTES")
        print("   → forward_features() avec LayerNorm final OK")
        print("   → CLS std dans la plage attendue [0.70-0.90]")
        print("")
        print("   Si checkpoints échouent avec ces features:")
        print("   → Chercher autre cause (GT mismatch, bug métrique, etc.)")
        return True
    else:
        print("⚠️  VERDICT: Features SUSPECTES")
        print(f"   → CLS std = {mean_std:.3f} hors plage [0.70-0.90]")
        print("   → Vérifier preprocessing (conversion uint8, normalisation)")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_features_standalone.py <features_dir>")
        print("")
        print("Exemple:")
        print("  python verify_features_standalone.py data/cache/pannuke_features")
        sys.exit(1)

    features_dir = sys.argv[1]
    success = verify_features_standalone(features_dir)

    sys.exit(0 if success else 1)
