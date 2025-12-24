#!/usr/bin/env python3
"""
Compare models/checkpoints et models/checkpoints_FIXED pour détecter les doublons.

Usage:
    python scripts/utils/compare_checkpoints.py
"""

from pathlib import Path
import hashlib


def compute_file_hash(filepath: Path) -> str:
    """Calcule le hash MD5 d'un fichier."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


def main():
    print("=" * 80)
    print("COMPARAISON CHECKPOINTS vs CHECKPOINTS_FIXED")
    print("=" * 80)

    dir1 = Path("models/checkpoints")
    dir2 = Path("models/checkpoints_FIXED")

    if not dir1.exists():
        print(f"\n❌ {dir1} n'existe pas")
        return 0

    if not dir2.exists():
        print(f"\n❌ {dir2} n'existe pas")
        return 0

    # Lister fichiers
    files1 = {f.name: f for f in dir1.glob('*.pth')}
    files2 = {f.name: f for f in dir2.glob('*.pth')}

    print(f"\n📂 {dir1}: {len(files1)} fichier(s)")
    print(f"📂 {dir2}: {len(files2)} fichier(s)")

    # Comparer
    common_files = set(files1.keys()) & set(files2.keys())

    if not common_files:
        print("\n✅ Aucun fichier en commun")
        return 0

    print(f"\n🔍 {len(common_files)} fichier(s) en commun:")
    print("-" * 80)

    identical = []
    different = []

    for filename in sorted(common_files):
        f1 = files1[filename]
        f2 = files2[filename]

        size1 = f1.stat().st_size
        size2 = f2.stat().st_size

        # Si même taille, calculer hash
        if size1 == size2:
            hash1 = compute_file_hash(f1)
            hash2 = compute_file_hash(f2)

            if hash1 == hash2:
                identical.append((filename, size1, hash1))
                print(f"   ✅ {filename:40s} IDENTIQUE ({size1/(1024*1024):.2f} MB)")
            else:
                different.append((filename, size1, size2))
                print(f"   ⚠️  {filename:40s} DIFFÉRENT (même taille)")
        else:
            different.append((filename, size1, size2))
            print(f"   ⚠️  {filename:40s} DIFFÉRENT ({size1/(1024*1024):.2f} MB vs {size2/(1024*1024):.2f} MB)")

    # Résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ:")
    print("-" * 80)

    if identical:
        total_waste = sum(size for _, size, _ in identical)
        print(f"\n🗑️  {len(identical)} DOUBLON(S) DÉTECTÉ(S):")
        for filename, size, hash in identical:
            print(f"   - {filename} ({size/(1024*1024):.2f} MB, hash: {hash[:16]}...)")
        print(f"\n💾 Espace gaspillé: {total_waste/(1024*1024):.2f} MB")
        print(f"\n💡 RECOMMANDATION:")
        print(f"   Supprimer {dir1}/ (entraînés avec features corrompues)")
        print(f"   Garder {dir2}/ uniquement")
        print(f"\n   rm -rf {dir1}")
    else:
        print(f"\n✅ Aucun doublon détecté")

    if different:
        print(f"\n⚠️  {len(different)} fichier(s) différent(s)")
        print("   Vérifier manuellement ces fichiers")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
