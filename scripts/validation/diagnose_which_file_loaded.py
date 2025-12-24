#!/usr/bin/env python3
"""
Diagnostic: Quel fichier est chargé par verify_spatial_alignment.py ?

Ce script aide à identifier le bug "Ghost Path" où un ancien fichier
corrompu est chargé au lieu du nouveau généré avec v4.

Usage:
    python scripts/validation/diagnose_which_file_loaded.py --family epidermal
"""

import argparse
import numpy as np
from pathlib import Path
import hashlib


def compute_file_hash(filepath: Path) -> str:
    """Calcule le hash MD5 d'un fichier pour détecter les doublons."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


def analyze_hv_orientation(hv_target: np.ndarray) -> dict:
    """
    Analyse l'orientation des canaux HV pour détecter si inversés.

    Returns:
        {
            'channel_0_is_vertical': bool,
            'channel_1_is_horizontal': bool,
            'likely_correct': bool,
            'evidence': str
        }
    """
    # Extraire un petit patch pour analyse
    patch_0 = hv_target[0, 100:156, 100:156]  # Canal 0
    patch_1 = hv_target[1, 100:156, 100:156]  # Canal 1

    # Calculer variance par direction
    var_y_ch0 = np.var(np.diff(patch_0, axis=0))  # Variance verticale canal 0
    var_x_ch0 = np.var(np.diff(patch_0, axis=1))  # Variance horizontale canal 0
    var_y_ch1 = np.var(np.diff(patch_1, axis=0))
    var_x_ch1 = np.var(np.diff(patch_1, axis=1))

    # Si canal 0 est Vertical, variance verticale devrait dominer
    ch0_vertical_dominant = var_y_ch0 > var_x_ch0
    # Si canal 1 est Horizontal, variance horizontale devrait dominer
    ch1_horizontal_dominant = var_x_ch1 > var_y_ch1

    likely_correct = ch0_vertical_dominant and ch1_horizontal_dominant

    evidence = f"Ch0 var_y={var_y_ch0:.4f} vs var_x={var_x_ch0:.4f} | "
    evidence += f"Ch1 var_y={var_y_ch1:.4f} vs var_x={var_x_ch1:.4f}"

    return {
        'channel_0_is_vertical': ch0_vertical_dominant,
        'channel_1_is_horizontal': ch1_horizontal_dominant,
        'likely_correct': likely_correct,
        'evidence': evidence
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnostiquer quel fichier est chargé")
    parser.add_argument('--family', type=str, required=True,
                        help='Famille à diagnostiquer (ex: epidermal)')
    args = parser.parse_args()

    family = args.family

    print("=" * 80)
    print("DIAGNOSTIC: Quel fichier est chargé ?")
    print("=" * 80)

    # Liste des chemins possibles (comme dans verify_spatial_alignment.py)
    possible_paths = [
        Path(f"data/cache/family_data/{family}_data_FIXED.npz"),
        Path(f"data/cache/family_data_FIXED/{family}_data_FIXED.npz"),
        Path(f"data/cache/family_data/{family}_data.npz"),
        Path(f"data/family_FIXED/{family}_data_FIXED.npz"),  # Ajout du chemin source
    ]

    existing_files = []

    print("\n1. SCAN DES FICHIERS EXISTANTS:")
    print("-" * 80)

    for path in possible_paths:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            file_hash = compute_file_hash(path)

            # Charger pour analyser HV orientation
            data = np.load(path)
            hv_analysis = analyze_hv_orientation(data['hv_targets'][0])

            existing_files.append({
                'path': path,
                'size_mb': size_mb,
                'hash': file_hash,
                'hv_correct': hv_analysis['likely_correct'],
                'evidence': hv_analysis['evidence']
            })

            status = "✅ CORRECT" if hv_analysis['likely_correct'] else "❌ INVERSÉ"
            print(f"\n📂 {path}")
            print(f"   Taille: {size_mb:.2f} MB")
            print(f"   Hash: {file_hash[:16]}...")
            print(f"   HV Orientation: {status}")
            print(f"   Evidence: {hv_analysis['evidence']}")
        else:
            print(f"\n❌ {path} (n'existe pas)")

    if not existing_files:
        print("\n🚨 ERREUR: Aucun fichier trouvé !")
        return 1

    print("\n" + "=" * 80)
    print("2. DÉTECTION DE DOUBLONS:")
    print("-" * 80)

    # Grouper par hash
    hash_groups = {}
    for file_info in existing_files:
        h = file_info['hash']
        if h not in hash_groups:
            hash_groups[h] = []
        hash_groups[h].append(file_info)

    for hash_val, files in hash_groups.items():
        if len(files) > 1:
            print(f"\n⚠️  DOUBLONS DÉTECTÉS (hash {hash_val[:16]}...):")
            for f in files:
                print(f"   - {f['path']}")
            print(f"   → Ces fichiers sont IDENTIQUES (gaspillage disque)")

    print("\n" + "=" * 80)
    print("3. QUEL FICHIER EST CHARGÉ PAR verify_spatial_alignment.py ?")
    print("-" * 80)

    # Simuler la logique de verify_spatial_alignment.py
    for path in possible_paths[:3]:  # Les 3 premiers chemins du script original
        if path.exists():
            matching_file = next(f for f in existing_files if f['path'] == path)
            status = "✅ CORRECT" if matching_file['hv_correct'] else "❌ INVERSÉ"

            print(f"\n🎯 FICHIER CHARGÉ: {path}")
            print(f"   Taille: {matching_file['size_mb']:.2f} MB")
            print(f"   Hash: {matching_file['hash'][:16]}...")
            print(f"   HV Orientation: {status}")

            if not matching_file['hv_correct']:
                print(f"\n🚨 PROBLÈME IDENTIFIÉ:")
                print(f"   Le script charge un fichier avec HV INVERSÉ !")
                print(f"   C'est pourquoi vous obtenez 96.29 px de distance.")
                print(f"\n💡 SOLUTION:")
                print(f"   1. Supprimer ce fichier: rm {path}")
                print(f"   2. Copier le fichier CORRECT depuis data/family_FIXED/")
                print(f"      OU modifier verify_spatial_alignment.py pour chercher")
                print(f"      directement dans data/family_FIXED/")
            else:
                print(f"\n✅ Le fichier chargé semble CORRECT")
                print(f"   Si vous obtenez toujours 96px, le problème est ailleurs.")

            break

    print("\n" + "=" * 80)
    print("4. RECOMMANDATIONS:")
    print("-" * 80)

    # Compter fichiers corrects vs inversés
    correct_files = [f for f in existing_files if f['hv_correct']]
    inverted_files = [f for f in existing_files if not f['hv_correct']]

    if inverted_files:
        print(f"\n⚠️  {len(inverted_files)} fichier(s) avec HV INVERSÉ détecté(s):")
        for f in inverted_files:
            print(f"   rm {f['path']}")
        print(f"\n   Total espace libéré: {sum(f['size_mb'] for f in inverted_files):.2f} MB")

    if len(hash_groups) < len(existing_files):
        duplicates = sum(len(files) - 1 for files in hash_groups.values())
        total_wasted = sum(
            sum(f['size_mb'] for f in files[1:])
            for files in hash_groups.values() if len(files) > 1
        )
        print(f"\n💾 {duplicates} doublon(s) détecté(s)")
        print(f"   Espace disque gaspillé: {total_wasted:.2f} MB")

    print(f"\n📍 SOURCE DE VÉRITÉ RECOMMANDÉE:")
    print(f"   data/family_FIXED/{family}_data_FIXED.npz")
    print(f"   → Modifier verify_spatial_alignment.py pour ne chercher QUE là")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
