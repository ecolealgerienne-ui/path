#!/usr/bin/env python3
"""
Test du script train_hovernet_family.py avec données synthétiques.

Ce script:
1. Génère des données synthétiques au format attendu par train_hovernet_family.py
2. Lance un entraînement court (3 epochs)
3. Vérifie que le pipeline fonctionne correctement

Usage:
    python scripts/validation/test_train_hovernet_family.py
    python scripts/validation/test_train_hovernet_family.py --n_samples 50 --epochs 5
"""

import argparse
import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np

# Ajouter le répertoire racine au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_synthetic_nuclei(size=256):
    """
    Génère une image synthétique avec des noyaux.

    Returns:
        np_mask: (256, 256) masque binaire
        hv_map: (2, 256, 256) cartes H/V en float [-1, 1]
        nt_mask: (256, 256) types de noyaux (0-4)
    """
    np_mask = np.zeros((size, size), dtype=np.float32)
    hv_map = np.zeros((2, size, size), dtype=np.float32)
    nt_mask = np.zeros((size, size), dtype=np.int64)

    # Générer 10-30 noyaux aléatoires
    n_nuclei = np.random.randint(10, 30)

    for _ in range(n_nuclei):
        # Position et taille du noyau
        cx = np.random.randint(20, size - 20)
        cy = np.random.randint(20, size - 20)
        radius = np.random.randint(5, 15)

        # Type cellulaire (0-4)
        cell_type = np.random.randint(0, 5)

        # Créer le masque circulaire
        y, x = np.ogrid[:size, :size]
        mask = ((x - cx)**2 + (y - cy)**2 <= radius**2)

        # NP mask
        np_mask[mask] = 1.0

        # HV maps (distance normalisée au centre)
        for i in range(size):
            for j in range(size):
                if mask[i, j]:
                    # H = horizontal (x direction)
                    h = (j - cx) / max(radius, 1)
                    # V = vertical (y direction)
                    v = (i - cy) / max(radius, 1)
                    hv_map[0, i, j] = np.clip(h, -1, 1)
                    hv_map[1, i, j] = np.clip(v, -1, 1)

        # NT mask
        nt_mask[mask] = cell_type

    return np_mask, hv_map, nt_mask


def generate_synthetic_features(n_samples: int, embed_dim: int = 1536):
    """
    Génère des features synthétiques au format H-optimus-0.

    Returns:
        features: (N, 261, 1536) avec std ~0.77 (comme après LayerNorm)
    """
    # CLS token + 256 patch tokens + 4 registres = 261
    features = np.random.randn(n_samples, 261, embed_dim).astype(np.float32)

    # Ajuster std pour simuler LayerNorm (~0.77)
    features = features * 0.77

    return features


def create_synthetic_family_data(
    family: str,
    n_samples: int,
    cache_dir: Path,
):
    """
    Crée les fichiers de données synthétiques pour une famille.

    Fichiers créés:
        - {family}_features.npz: features H-optimus-0
        - {family}_targets.npz: np_targets, hv_targets, nt_targets
    """
    print(f"\n📦 Génération données synthétiques pour famille '{family}'...")
    print(f"   Samples: {n_samples}")
    print(f"   Cache dir: {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Générer features
    print("   ⏳ Génération features (261, 1536)...")
    features = generate_synthetic_features(n_samples)
    print(f"      Shape: {features.shape}")
    print(f"      CLS std: {features[:, 0, :].std():.4f}")

    # Générer targets
    print("   ⏳ Génération targets (256x256)...")
    np_targets = np.zeros((n_samples, 256, 256), dtype=np.float32)
    hv_targets = np.zeros((n_samples, 2, 256, 256), dtype=np.int8)
    nt_targets = np.zeros((n_samples, 256, 256), dtype=np.int64)

    for i in range(n_samples):
        np_mask, hv_map, nt_mask = generate_synthetic_nuclei()
        np_targets[i] = np_mask
        # Convertir HV float [-1, 1] → int8 [-127, 127]
        hv_targets[i] = (hv_map * 127).astype(np.int8)
        nt_targets[i] = nt_mask

        if (i + 1) % 10 == 0:
            print(f"      {i + 1}/{n_samples} samples générés")

    # Sauvegarder features
    features_path = cache_dir / f"{family}_features.npz"
    np.savez_compressed(features_path, features=features)
    print(f"   ✅ Features sauvées: {features_path}")

    # Sauvegarder targets
    targets_path = cache_dir / f"{family}_targets.npz"
    np.savez_compressed(
        targets_path,
        np_targets=np_targets,
        hv_targets=hv_targets,
        nt_targets=nt_targets
    )
    print(f"   ✅ Targets sauvés: {targets_path}")

    # Stats
    total_size = (features.nbytes + np_targets.nbytes +
                  hv_targets.nbytes + nt_targets.nbytes) / 1e6
    print(f"   📊 Taille totale: {total_size:.1f} MB")

    return features_path, targets_path


def run_training_test(
    family: str,
    cache_dir: Path,
    epochs: int,
    output_dir: Path,
):
    """
    Lance le script d'entraînement et vérifie le résultat.
    """
    print(f"\n🚀 Lancement entraînement test...")
    print(f"   Famille: {family}")
    print(f"   Epochs: {epochs}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "training" / "train_hovernet_family.py"),
        "--family", family,
        "--cache_dir", str(cache_dir),
        "--epochs", str(epochs),
        "--batch_size", "4",
        "--output_dir", str(output_dir),
        "--val_split", "0.2",
    ]

    print(f"   Commande: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max
        )

        print("\n" + "=" * 60)
        print("STDOUT:")
        print("=" * 60)
        print(result.stdout)

        if result.stderr:
            print("\n" + "=" * 60)
            print("STDERR:")
            print("=" * 60)
            print(result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("   ❌ Timeout (> 5 minutes)")
        return False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def verify_checkpoint(output_dir: Path, family: str):
    """
    Vérifie que le checkpoint a été créé et contient les bonnes clés.
    """
    checkpoint_path = output_dir / f"hovernet_{family}_best.pth"

    if not checkpoint_path.exists():
        print(f"   ❌ Checkpoint non trouvé: {checkpoint_path}")
        return False

    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    required_keys = ['epoch', 'model_state_dict', 'best_dice', 'family']
    missing = [k for k in required_keys if k not in checkpoint]

    if missing:
        print(f"   ❌ Clés manquantes: {missing}")
        return False

    print(f"\n✅ Checkpoint vérifié: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best Dice: {checkpoint['best_dice']:.4f}")
    print(f"   Best HV MSE: {checkpoint.get('best_hv_mse', 'N/A')}")
    print(f"   Best NT Acc: {checkpoint.get('best_nt_acc', 'N/A')}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test du script train_hovernet_family.py"
    )
    parser.add_argument("--family", type=str, default="respiratory",
                        help="Famille à tester (default: respiratory)")
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Nombre de samples synthétiques")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Nombre d'epochs")
    parser.add_argument("--keep_data", action="store_true",
                        help="Garder les données après le test")
    args = parser.parse_args()

    print("=" * 60)
    print("TEST TRAIN_HOVERNET_FAMILY.PY")
    print("=" * 60)

    # Répertoires temporaires
    cache_dir = PROJECT_ROOT / "data" / "cache" / "test_family_data"
    output_dir = PROJECT_ROOT / "models" / "checkpoints" / "test"

    # Nettoyer si existant
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    try:
        # 1. Générer données synthétiques
        print("\n" + "=" * 60)
        print("ÉTAPE 1: Génération des données synthétiques")
        print("=" * 60)

        create_synthetic_family_data(
            family=args.family,
            n_samples=args.n_samples,
            cache_dir=cache_dir,
        )
        results["data_generation"] = True

        # 2. Lancer l'entraînement
        print("\n" + "=" * 60)
        print("ÉTAPE 2: Entraînement test")
        print("=" * 60)

        training_ok = run_training_test(
            family=args.family,
            cache_dir=cache_dir,
            epochs=args.epochs,
            output_dir=output_dir,
        )
        results["training"] = training_ok

        # 3. Vérifier le checkpoint
        print("\n" + "=" * 60)
        print("ÉTAPE 3: Vérification du checkpoint")
        print("=" * 60)

        if training_ok:
            checkpoint_ok = verify_checkpoint(output_dir, args.family)
            results["checkpoint"] = checkpoint_ok
        else:
            results["checkpoint"] = False

    finally:
        # Nettoyage
        if not args.keep_data:
            print("\n🧹 Nettoyage...")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            if output_dir.exists():
                shutil.rmtree(output_dir)

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)

    all_ok = all(results.values())
    for step, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {step}")

    print()
    if all_ok:
        print("🎉 TOUS LES TESTS PASSENT!")
        print("\nLe pipeline train_hovernet_family.py est validé.")
        print("Vous pouvez lancer l'entraînement sur les vraies données:")
        print(f"  python scripts/training/train_hovernet_family.py --family {args.family} --epochs 50 --augment")
        return 0
    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        return 1


if __name__ == "__main__":
    sys.exit(main())
