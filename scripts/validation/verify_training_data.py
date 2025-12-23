#!/usr/bin/env python3
"""
Vérifie quelles données ont été utilisées pour l'entraînement actuel.

Objectif: Déterminer si FIXED (instances séparées) ou OLD (connectedComponents)

Usage:
    python scripts/validation/verify_training_data.py \
        --checkpoint models/checkpoints/hovernet_glandular_best.pth
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_data_files():
    """Cherche et analyse les fichiers de données disponibles."""
    print(f"\n{'='*70}")
    print(f"VÉRIFICATION DES DONNÉES D'ENTRAÎNEMENT")
    print(f"{'='*70}\n")

    # Chercher fichiers de données
    data_patterns = [
        "data/cache/family_data/*.npz",
        "data/cache/family_data_FIXED/*.npz",
        "data/cache/family_data_OLD*/*.npz",
        "data/family_data/*.npz",
        "data/*/glandular*.npz",
        "data/*/digestive*.npz",
    ]

    found_files = []
    for pattern in data_patterns:
        files = list(Path(".").glob(pattern))
        found_files.extend(files)

    if not found_files:
        print("❌ Aucun fichier de données trouvé!")
        print("\n💡 Suggestions:")
        print("   1. Les données sont peut-être ailleurs sur le système")
        print("   2. Vérifier dans ~/data/ ou /mnt/data/")
        print("   3. Vérifier les logs d'entraînement pour le chemin exact")
        return None

    print(f"✅ {len(found_files)} fichier(s) de données trouvé(s):\n")
    for f in found_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"   📁 {f}")
        print(f"      Taille: {size_mb:.1f} MB")
        print()

    return found_files


def analyze_data_file(data_file: Path, n_samples: int = 10):
    """Analyse un fichier de données pour détecter FIXED vs OLD."""
    print(f"\n{'='*70}")
    print(f"ANALYSE: {data_file.name}")
    print(f"{'='*70}\n")

    try:
        data = np.load(data_file)
        print(f"📋 Clés disponibles: {list(data.keys())}\n")

        # Vérifier présence de hv_targets
        if 'hv_targets' not in data:
            print("❌ Pas de 'hv_targets' trouvé!")
            return

        hv_targets = data['hv_targets']
        print(f"📊 HV Targets:")
        print(f"   Shape: {hv_targets.shape}")
        print(f"   Dtype: {hv_targets.dtype}")
        print(f"   Min: {hv_targets.min():.4f}")
        print(f"   Max: {hv_targets.max():.4f}")
        print(f"   Mean: {hv_targets.mean():.4f}")
        print(f"   Std: {hv_targets.std():.4f}")

        # Vérifier la normalisation HV
        print(f"\n🔍 Vérification Normalisation HV:")
        if hv_targets.dtype == np.int8:
            print(f"   ❌ PROBLÈME: dtype int8 (devrait être float32)")
            print(f"   ❌ Range [-127, 127] au lieu de [-1, 1]")
            print(f"   → BUG: Anciennes données avec int8!")
            verdict = "OLD (int8 bug)"
        elif hv_targets.min() >= -1.1 and hv_targets.max() <= 1.1:
            print(f"   ✅ Range [-1, 1] correct (float32)")
            verdict = "FIXED (float32)"
        else:
            print(f"   ⚠️  Range anormal: [{hv_targets.min():.2f}, {hv_targets.max():.2f}]")
            verdict = "INCERTAIN"

        # Analyser les instances
        if 'np_masks' in data:
            np_masks = data['np_masks']
            print(f"\n📊 NP Masks:")
            print(f"   Shape: {np_masks.shape}")

            # Compter instances par image (échantillon)
            n_to_check = min(n_samples, len(np_masks))
            inst_counts = []

            for i in range(n_to_check):
                np_mask = np_masks[i]
                # Si np_mask est binaire, on ne peut pas compter les instances
                # Si np_mask a des IDs, on peut
                unique_vals = np.unique(np_mask)
                n_instances = len(unique_vals) - 1  # -1 pour background

                if np_mask.max() <= 1:
                    # Binaire - on ne peut pas conclure
                    inst_counts.append(-1)
                else:
                    inst_counts.append(n_instances)

            valid_counts = [c for c in inst_counts if c > 0]
            if valid_counts:
                mean_inst = np.mean(valid_counts)
                print(f"\n📊 Instances par image (échantillon {n_to_check}):")
                print(f"   Moyenne: {mean_inst:.1f}")
                print(f"   Min: {min(valid_counts)}")
                print(f"   Max: {max(valid_counts)}")

                if mean_inst > 40:
                    print(f"   ✅ {mean_inst:.0f} instances/image → FIXED (instances séparées)")
                    verdict += " + instances séparées"
                elif mean_inst < 20:
                    print(f"   ❌ {mean_inst:.0f} instances/image → OLD (fusionnées)")
                    verdict += " + instances fusionnées"
            else:
                print(f"   ⚠️  NP masks sont binaires, impossible de compter instances")

        print(f"\n🎯 VERDICT: {verdict}")

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")


def check_training_logs():
    """Cherche les logs d'entraînement pour trouver le chemin des données."""
    print(f"\n{'='*70}")
    print(f"RECHERCHE LOGS D'ENTRAÎNEMENT")
    print(f"{'='*70}\n")

    log_patterns = [
        "logs/*.log",
        "logs/training/*.log",
        "results/**/training.log",
        "*.log",
    ]

    found_logs = []
    for pattern in log_patterns:
        logs = list(Path(".").glob(pattern))
        found_logs.extend(logs)

    if not found_logs:
        print("❌ Aucun log d'entraînement trouvé")
        return

    print(f"✅ {len(found_logs)} log(s) trouvé(s)\n")

    # Chercher mentions de data path dans les logs
    for log_file in found_logs[:5]:  # Limiter à 5 pour ne pas surcharger
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if 'data' in content.lower():
                    print(f"📄 {log_file}")
                    # Extraire lignes avec 'data'
                    lines = content.split('\n')
                    data_lines = [l for l in lines if 'data' in l.lower() or 'family' in l.lower()][:3]
                    for line in data_lines:
                        print(f"   {line[:100]}")
                    print()
        except:
            pass


def check_checkpoint_metadata(checkpoint_path: Path):
    """Vérifie les métadonnées du checkpoint."""
    print(f"\n{'='*70}")
    print(f"ANALYSE CHECKPOINT: {checkpoint_path.name}")
    print(f"{'='*70}\n")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint introuvable: {checkpoint_path}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print(f"📋 Clés du checkpoint: {list(checkpoint.keys())}\n")

        # Vérifier métadonnées
        if 'metadata' in checkpoint:
            meta = checkpoint['metadata']
            print(f"📊 Métadonnées:")
            for key, val in meta.items():
                print(f"   {key}: {val}")

        # Vérifier epoch, metrics
        if 'epoch' in checkpoint:
            print(f"\n📊 Training Info:")
            print(f"   Epoch: {checkpoint['epoch']}")

        if 'metrics' in checkpoint:
            print(f"\n📊 Metrics:")
            for key, val in checkpoint['metrics'].items():
                print(f"   {key}: {val:.4f}")

        # Vérifier si le checkpoint contient des infos sur les données
        keys_to_check = ['data_path', 'data_version', 'preprocessing', 'family']
        for key in keys_to_check:
            if key in checkpoint:
                print(f"\n🔍 {key}: {checkpoint[key]}")

    except Exception as e:
        print(f"❌ Erreur lors de la lecture du checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Vérifie quelles données ont été utilisées pour l'entraînement"
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Chemin vers un checkpoint à analyser'
    )
    parser.add_argument(
        '--data_file',
        type=Path,
        help='Chemin vers un fichier de données à analyser'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=10,
        help='Nombre d\'échantillons à analyser (default: 10)'
    )

    args = parser.parse_args()

    # 1. Chercher fichiers de données
    data_files = check_data_files()

    # 2. Si un fichier spécifique fourni, l'analyser
    if args.data_file:
        analyze_data_file(args.data_file, args.n_samples)
    elif data_files:
        # Analyser le premier fichier trouvé
        analyze_data_file(data_files[0], args.n_samples)

    # 3. Chercher logs d'entraînement
    check_training_logs()

    # 4. Si checkpoint fourni, l'analyser
    if args.checkpoint:
        check_checkpoint_metadata(args.checkpoint)

    print(f"\n{'='*70}")
    print(f"RÉSUMÉ")
    print(f"{'='*70}\n")
    print("Pour confirmer, vérifier:")
    print("  1. Les scripts d'entraînement utilisés (train_hovernet_family.py)")
    print("  2. Les logs d'entraînement pour le data_path exact")
    print("  3. Comparer avec prepare_family_data.py vs prepare_family_data_FIXED.py")
    print()


if __name__ == '__main__':
    main()
