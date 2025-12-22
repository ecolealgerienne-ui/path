#!/usr/bin/env python3
"""
Organise les images de test PanNuke par famille.

Créée des sous-dossiers pour chaque famille avec les images appropriées:
- glandular/
- digestive/
- urologic/
- respiratory/
- epidermal/
"""
import argparse
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# Mapping PanNuke organe → famille
ORGAN_TO_FAMILY = {
    # Glandulaire & Hormonale
    "Breast": "glandular",
    "Prostate": "glandular",
    "Thyroid": "glandular",
    "Pancreatic": "glandular",
    "Adrenal_gland": "glandular",

    # Digestive
    "Colon": "digestive",
    "Stomach": "digestive",
    "Esophagus": "digestive",
    "Bile-duct": "digestive",

    # Urologique & Reproductif
    "Kidney": "urologic",
    "Bladder": "urologic",
    "Testis": "urologic",
    "Ovarian": "urologic",
    "Uterus": "urologic",
    "Cervix": "urologic",

    # Respiratoire & Hépatique
    "Lung": "respiratory",
    "Liver": "respiratory",

    # Épidermoïde
    "Skin": "epidermal",
    "HeadNeck": "epidermal",
}

FAMILIES = ["glandular", "digestive", "urologic", "respiratory", "epidermal"]


def extract_organ_from_filename(npz_file: Path) -> str:
    """
    Extrait l'organe du nom de fichier NPZ.

    Format attendu: image_XXXXX.npz ou <organ>_XXXXX.npz
    """
    # Charger pour voir si l'info organe est dans le fichier
    data = np.load(npz_file, allow_pickle=True)

    if 'organ' in data:
        organ = str(data['organ'])
        if isinstance(organ, np.ndarray):
            organ = organ.item()
        return organ

    # Sinon essayer de deviner depuis le nom de fichier
    # (Cette partie dépend de votre structure de noms de fichiers)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Organize PanNuke test images by family"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory with NPZ files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory to create family subdirectories"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without copying files"
    )

    args = parser.parse_args()

    # Get all NPZ files
    npz_files = sorted(args.input_dir.glob("*.npz"))

    print("="*70)
    print("ORGANIZING TEST IMAGES BY FAMILY")
    print("="*70)
    print(f"\nInput:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Files:  {len(npz_files)}")

    if args.dry_run:
        print("\n⚠️ DRY RUN MODE - No files will be copied\n")

    # Count by family
    family_counts = defaultdict(list)
    unknown = []

    print("\n📊 Scanning files...")
    for npz_file in npz_files:
        organ = extract_organ_from_filename(npz_file)

        if organ and organ in ORGAN_TO_FAMILY:
            family = ORGAN_TO_FAMILY[organ]
            family_counts[family].append((npz_file, organ))
        else:
            unknown.append(npz_file)

    # Display statistics
    print("\n" + "="*70)
    print("DISTRIBUTION BY FAMILY")
    print("="*70)

    total_organized = 0
    for family in FAMILIES:
        count = len(family_counts[family])
        total_organized += count
        print(f"{family.capitalize():12}: {count:4d} images")

        if count > 0:
            # Show organ breakdown
            organ_breakdown = defaultdict(int)
            for _, organ in family_counts[family]:
                organ_breakdown[organ] += 1

            for organ, cnt in sorted(organ_breakdown.items()):
                print(f"  └─ {organ:15}: {cnt:4d}")

    print(f"\nTotal organized: {total_organized}")
    print(f"Unknown:         {len(unknown)}")

    if unknown:
        print(f"\n⚠️ {len(unknown)} files without organ info:")
        for f in unknown[:5]:
            print(f"  - {f.name}")
        if len(unknown) > 5:
            print(f"  ... and {len(unknown)-5} more")

    # Create directories and copy files
    if not args.dry_run:
        print("\n📁 Creating family directories...")

        for family in FAMILIES:
            family_dir = args.output_dir / family
            family_dir.mkdir(parents=True, exist_ok=True)

            for npz_file, organ in family_counts[family]:
                dest = family_dir / npz_file.name
                shutil.copy2(npz_file, dest)

            print(f"  ✓ {family.capitalize():12}: {len(family_counts[family])} files → {family_dir}")

        print(f"\n✅ Done! Files organized in {args.output_dir}")
    else:
        print("\n💡 Run without --dry_run to actually copy files")


if __name__ == "__main__":
    main()
