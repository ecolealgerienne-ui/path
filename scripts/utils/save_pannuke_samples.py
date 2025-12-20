#!/usr/bin/env python3
"""
Sauvegarde quelques échantillons PanNuke en PNG pour tester l'IHM Gradio.

Usage:
    python scripts/utils/save_pannuke_samples.py --data_dir /chemin/vers/PanNuke
"""

import numpy as np
from pathlib import Path
import cv2
import random
import argparse

# Configuration par défaut
DEFAULT_DATA_DIR = Path("/home/amar/data/PanNuke")
OUTPUT_DIR = Path("data/samples")
N_IMAGES_PER_ORGAN = 5
ORGANS = ["Prostate", "Breast", "Colon"]  # 3 organes différents


def main():
    parser = argparse.ArgumentParser(description="Sauvegarder des échantillons PanNuke")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Chemin vers le dossier PanNuke")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Dossier de sortie pour les images PNG")
    parser.add_argument("--n_per_organ", type=int, default=N_IMAGES_PER_ORGAN,
                        help="Nombre d'images par organe")
    parser.add_argument("--organs", nargs="+", default=ORGANS,
                        help="Liste des organes à extraire")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Créer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger les données PanNuke
    print(f"Chargement des données depuis {data_dir}...")

    # Chercher les fichiers dans fold_0 ou directement
    if (data_dir / "fold_0" / "images.npy").exists():
        images_path = data_dir / "fold_0" / "images.npy"
        types_path = data_dir / "fold_0" / "types.npy"
    elif (data_dir / "images.npy").exists():
        images_path = data_dir / "images.npy"
        types_path = data_dir / "types.npy"
    else:
        print(f"❌ Données PanNuke non trouvées dans {data_dir}")
        print(f"   Attendu: {data_dir}/fold_0/images.npy")
        print(f"   ou:      {data_dir}/images.npy")
        return

    images = np.load(images_path)
    types = np.load(types_path)

    print(f"Images chargées: {images.shape}")
    print(f"Organes disponibles: {np.unique(types)}")

    saved_count = 0

    for organ in args.organs:
        # Trouver les indices pour cet organe
        organ_indices = np.where(types == organ)[0]

        if len(organ_indices) == 0:
            print(f"⚠️ Aucune image trouvée pour {organ}")
            continue

        print(f"\n{organ}: {len(organ_indices)} images disponibles")

        # Sélectionner N images aléatoirement
        n_select = min(args.n_per_organ, len(organ_indices))
        selected = random.sample(list(organ_indices), n_select)

        for i, idx in enumerate(selected):
            img = images[idx]

            # Convertir de [0,1] à [0,255] si nécessaire
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Sauvegarder (OpenCV attend BGR)
            filename = f"{organ.lower()}_{i+1:02d}.png"
            filepath = output_dir / filename

            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  ✓ {filename}")
            saved_count += 1

    print(f"\n✅ {saved_count} images sauvegardées dans {output_dir}/")
    print(f"\nPour tester dans Gradio:")
    print(f"  1. Lancer: python scripts/demo/gradio_demo.py")
    print(f"  2. Aller sur http://localhost:7860")
    print(f"  3. Onglet 'Analyser votre Image'")
    print(f"  4. Uploader les images depuis {output_dir}/")


if __name__ == "__main__":
    main()
