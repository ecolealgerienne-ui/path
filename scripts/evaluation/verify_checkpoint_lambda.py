#!/usr/bin/env python3
"""
Vérifie si le checkpoint a été mis à jour avec le bon lambda_hv.

Ce script charge le checkpoint et affiche:
1. Date de modification du fichier
2. Hyperparamètres sauvegardés (si disponibles)
3. Comparaison avec lambda_hv attendu

Usage:
    python scripts/evaluation/verify_checkpoint_lambda.py \
        --checkpoint models/checkpoints/hovernet_epidermal_best.pth \
        --expected_lambda_hv 5.0
"""

import torch
import argparse
from pathlib import Path
import os
from datetime import datetime


def verify_checkpoint_lambda(checkpoint_path: str, expected_lambda_hv: float):
    """
    Vérifie le checkpoint et affiche ses métadonnées.

    Args:
        checkpoint_path: Chemin vers le checkpoint
        expected_lambda_hv: Lambda_hv attendu (ex: 5.0)
    """
    checkpoint_path = Path(checkpoint_path)

    print("\n" + "="*80)
    print("VÉRIFICATION CHECKPOINT — Lambda_hv")
    print("="*80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Lambda_hv attendu: {expected_lambda_hv}")
    print("")

    # 1. Vérifier existence
    if not checkpoint_path.exists():
        print(f"❌ ERREUR: Checkpoint introuvable: {checkpoint_path}")
        return None

    # 2. Informations fichier
    stat = os.stat(checkpoint_path)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    size_mb = stat.st_size / (1024 * 1024)

    print("="*80)
    print("INFORMATIONS FICHIER")
    print("="*80)
    print(f"Taille: {size_mb:.2f} MB")
    print(f"Date modification: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # 3. Charger checkpoint
    try:
        print("Chargement checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint chargé\n")
    except Exception as e:
        print(f"❌ ERREUR lors du chargement: {e}")
        return None

    # 4. Afficher structure
    print("="*80)
    print("STRUCTURE CHECKPOINT")
    print("="*80)
    print(f"Clés disponibles: {list(checkpoint.keys())}\n")

    # 5. Extraire métadonnées
    has_metadata = False

    if 'hyperparameters' in checkpoint:
        print("="*80)
        print("HYPERPARAMÈTRES SAUVEGARDÉS")
        print("="*80)
        hyperparams = checkpoint['hyperparameters']

        for key, value in hyperparams.items():
            print(f"{key}: {value}")

        # Vérifier lambda_hv
        if 'lambda_hv' in hyperparams:
            saved_lambda_hv = hyperparams['lambda_hv']
            print("\n" + "="*80)
            print("DIAGNOSTIC LAMBDA_HV")
            print("="*80)
            print(f"Lambda_hv sauvegardé: {saved_lambda_hv}")
            print(f"Lambda_hv attendu:    {expected_lambda_hv}")

            if abs(saved_lambda_hv - expected_lambda_hv) < 0.01:
                print(f"✅ MATCH: Checkpoint correspond à lambda_hv={expected_lambda_hv}")
                has_metadata = True
            else:
                print(f"❌ MISMATCH: Checkpoint a lambda_hv={saved_lambda_hv}, "
                      f"attendu {expected_lambda_hv}")
                print(f"\n💡 DIAGNOSTIC:")
                print(f"   Le checkpoint n'a PAS été mis à jour avec lambda_hv={expected_lambda_hv}")
                print(f"   Vous testez avec un ancien checkpoint (lambda_hv={saved_lambda_hv})")
                has_metadata = True
        else:
            print("\n⚠️ Pas de 'lambda_hv' dans hyperparameters")

    # 6. Informations epoch/metrics si disponibles
    if 'epoch' in checkpoint:
        print("\n" + "="*80)
        print("INFORMATIONS TRAINING")
        print("="*80)
        print(f"Epoch: {checkpoint['epoch']}")

    if 'metrics' in checkpoint:
        print("\nMétriques:")
        for key, value in checkpoint['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # 7. Recommandation si pas de metadata
    if not has_metadata:
        print("\n" + "="*80)
        print("⚠️ IMPOSSIBLE DE VÉRIFIER LAMBDA_HV")
        print("="*80)
        print("Le checkpoint ne contient pas de métadonnées hyperparameters.")
        print("\n💡 SOLUTIONS:")
        print("1. Comparer date modification checkpoint avec date training:")
        print(f"   Date checkpoint: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("   Si date AVANT le training lambda_hv=5.0 → checkpoint pas mis à jour")
        print("\n2. Vérifier logs training pour confirmer sauvegarde:")
        print("   Chercher message 'Saved best checkpoint'")
        print("\n3. Ré-entraîner avec lambda_hv=5.0 en vérifiant que:")
        print("   - Le répertoire models/checkpoints/ existe")
        print("   - Les permissions permettent l'écriture")
        print("   - Le script affiche 'Saved best checkpoint'")

    print("\n" + "="*80)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérification checkpoint lambda_hv")
    parser.add_argument('--checkpoint', type=str,
                       default='models/checkpoints/hovernet_epidermal_best.pth',
                       help="Chemin vers le checkpoint")
    parser.add_argument('--expected_lambda_hv', type=float, default=5.0,
                       help="Lambda_hv attendu (ex: 5.0)")

    args = parser.parse_args()

    result = verify_checkpoint_lambda(args.checkpoint, args.expected_lambda_hv)

    if result:
        print("\n✅ Vérification terminée")
